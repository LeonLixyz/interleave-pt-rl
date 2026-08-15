from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from chess_rl_miles.scripts import upload_interleaved_checkpoints as uploader


def _write_checkpoint(root: Path, step: int) -> None:
    checkpoint = root / f"iter_{step:07d}"
    (checkpoint / "model").mkdir(parents=True)
    (checkpoint / "model" / ".metadata").write_bytes(b"metadata")
    (checkpoint / "model" / "__0_0.distcp").write_bytes(b"model shard")
    (checkpoint / "rng.pt").write_bytes(b"rng")
    (checkpoint / "meta.json").write_text(
        json.dumps(
            {
                "iteration": step,
                "next_rollout_id": step,
            }
        ),
        encoding="utf-8",
    )


def _identity(run_name: str) -> dict[str, object]:
    return {
        "kind": "chess_rl_miles_interleave_run",
        "run": {
            "app_name": "chess-interleave-rl",
            "run_name": run_name,
            "model_id": "interleave_47m_qwen3",
            "num_rollout": uploader.TARGET_STEP,
            "dynamic_filter": uploader.RUNS[run_name]["dynamic_filter"],
            "rollout_seed": 42,
            "save_interval": uploader.SAVE_INTERVAL,
            "eval_interval": 0,
            "canary": False,
        },
        "checkpoint_publication": uploader.EXPECTED_CHECKPOINT_PUBLICATION,
        "policy_update_profile": uploader.EXPECTED_POLICY,
        "fixed_rl_semantics": uploader.EXPECTED_SEMANTICS,
        "balanced_data": {
            "logical_path": uploader.modal_interleave.BALANCED_TRAIN_FILE,
            "sha256": uploader.modal_interleave.BALANCED_TRAIN_SHA256,
        },
        "origin_hf": {
            "logical_path": str(uploader.ORIGIN_HF),
            "manifest_sha256": uploader.EXPECTED_ORIGIN_MANIFEST_SHA256,
        },
        "sources": {
            "chess_rl_miles": {
                "manifest_sha256": uploader.EXPECTED_CHESS_SOURCE_SHA256
            },
            "miles": {
                "manifest_sha256": uploader.EXPECTED_MILES_SOURCE_SHA256
            },
        },
        "runtime": {
            "image": uploader.EXPECTED_RUNTIME_IMAGE,
            "installed_packages_sha256": (
                uploader.EXPECTED_INSTALLED_PACKAGES_SHA256
            ),
        },
    }


def _provenance(run_name: str) -> dict[str, object]:
    identity = _identity(run_name)
    command = ["python", "train.py"]
    return {
        "identity": identity,
        "identity_sha256": uploader._canonical_sha256(identity),
        "initial_command": command,
        "initial_command_sha256": uploader.hashlib.sha256(
            json.dumps(command, separators=(",", ":")).encode()
        ).hexdigest(),
    }


@pytest.mark.parametrize("run_name", sorted(uploader.RUNS))
def test_registered_run_provenance_is_fail_closed(run_name):
    provenance = _provenance(run_name)
    observed = uploader._validate_run_provenance(
        provenance,
        run_name=run_name,
    )
    assert observed["run"]["run_name"] == run_name

    drifted = json.loads(json.dumps(provenance))
    drifted["identity"]["run"]["save_interval"] = 20
    drifted["identity_sha256"] = uploader._canonical_sha256(
        drifted["identity"]
    )
    with pytest.raises(RuntimeError, match="run provenance drift"):
        uploader._validate_run_provenance(
            drifted,
            run_name=run_name,
        )


def test_tracker_never_exposes_partial_checkpoint(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    (root / "latest_checkpointed_iteration.txt").write_text("40")

    with pytest.raises(RuntimeError, match="readiness contract"):
        uploader._complete_checkpoint_records(root)

    _write_checkpoint(root, 40)
    tracker, records = uploader._complete_checkpoint_records(root)
    assert tracker == 40
    assert [record["iteration"] for record in records] == [40]
    assert [item["path"] for item in records[0]["model_files"]] == [
        "model/.metadata",
        "model/__0_0.distcp",
    ]

    (root / "iter_0000040" / "rng.pt").unlink()
    with pytest.raises(RuntimeError, match="readiness contract"):
        uploader._complete_checkpoint_records(root)


def test_tracker_requires_registered_save_interval(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    (root / "latest_checkpointed_iteration.txt").write_text("41")
    with pytest.raises(RuntimeError, match="Invalid E2 checkpoint tracker"):
        uploader._complete_checkpoint_records(root)


def test_incremental_scan_hashes_only_new_steps(tmp_path, monkeypatch):
    root = tmp_path / "run"
    root.mkdir()
    _write_checkpoint(root, 40)
    (root / "latest_checkpointed_iteration.txt").write_text("40")
    calls = []
    original = uploader._sha256_file

    def recording_sha(path):
        calls.append(path)
        return original(path)

    monkeypatch.setattr(uploader, "_sha256_file", recording_sha)
    _, first = uploader._complete_checkpoint_records(root)
    assert len(first) == 1
    first_calls = len(calls)
    assert first_calls > 0

    _, unchanged = uploader._complete_checkpoint_records(
        root,
        after_step=40,
    )
    assert unchanged == []
    assert len(calls) == first_calls

    _write_checkpoint(root, 80)
    (root / "latest_checkpointed_iteration.txt").write_text("80")
    _, second = uploader._complete_checkpoint_records(
        root,
        after_step=40,
    )
    assert [record["iteration"] for record in second] == [80]
    assert len(calls) > first_calls


def test_checkpoint_manifest_binds_raw_and_run_identity():
    run_name = "core-e2-u-rl3000-seed42"
    identity = _identity(run_name)
    raw = {
        "logical_path": "/raw/iter_0000040",
        "iteration": 40,
        "next_rollout_id": 40,
        "markers": {"meta": {"sha256": "a" * 64}},
    }
    manifest = uploader._checkpoint_manifest(
        run_name=run_name,
        repo_id=str(uploader.RUNS[run_name]["repo_id"]),
        step=40,
        raw_checkpoint=raw,
        run_provenance_sha256="b" * 64,
        identity=identity,
        output_files=[
            {
                "path": "config.json",
                "bytes": 2,
                "sha256": "c" * 64,
            },
            {
                "path": "model.safetensors",
                "bytes": 4,
                "sha256": "d" * 64,
            },
        ],
        uploader_source_sha256="e" * 64,
    )
    uploader._validate_checkpoint_manifest(
        manifest,
        run_name=run_name,
        repo_id=str(uploader.RUNS[run_name]["repo_id"]),
        step=40,
        raw_checkpoint=raw,
        run_provenance_sha256="b" * 64,
        identity=identity,
        uploader_source_sha256="e" * 64,
    )
    manifest["raw_checkpoint"]["iteration"] = 80
    with pytest.raises(RuntimeError, match="self-hash mismatch"):
        uploader._validate_checkpoint_manifest(
            manifest,
            run_name=run_name,
            repo_id=str(uploader.RUNS[run_name]["repo_id"]),
            step=40,
            raw_checkpoint=raw,
            run_provenance_sha256="b" * 64,
            identity=identity,
            uploader_source_sha256="e" * 64,
        )


def test_publication_contract_is_self_hashed_and_append_only():
    run_name = "core-e2-d-rl3000-seed42"
    identity = _identity(run_name)
    contract = uploader._publication_contract(
        run_name=run_name,
        repo_id=str(uploader.RUNS[run_name]["repo_id"]),
        run_provenance_sha256="f" * 64,
        identity=identity,
        uploader_source_sha256="1" * 64,
    )
    uploader._validate_publication_contract(contract, contract)
    assert contract["append_only"] is True
    assert len(contract["expected_steps"]) == 75
    assert contract["expected_steps"][0] == 40
    assert contract["expected_steps"][-1] == 3000


def test_repo_items_preserves_hub_prefixed_tree_paths():
    class FakeApi:
        def list_repo_tree(self, **kwargs):
            assert kwargs["path_in_repo"] == "global_step_40"
            assert kwargs["recursive"] is True
            assert kwargs["expand"] is True
            return [
                SimpleNamespace(
                    path="global_step_40/config.json",
                    size=2,
                    lfs=None,
                ),
                SimpleNamespace(
                    path="global_step_40/model.safetensors",
                    size=4,
                    lfs={"sha256": "d" * 64, "size": 4},
                ),
            ]

    items = uploader._repo_items(
        FakeApi(),
        "owner/repo",
        path_in_repo="global_step_40",
    )
    assert set(items) == {
        "global_step_40/config.json",
        "global_step_40/model.safetensors",
    }


def test_remote_checkpoint_verifies_lfs_and_non_lfs_content(
    tmp_path,
    monkeypatch,
):
    run_name = "core-e2-u-rl3000-seed42"
    repo_id = str(uploader.RUNS[run_name]["repo_id"])
    identity = _identity(run_name)
    raw = {
        "logical_path": "/raw/iter_0000040",
        "iteration": 40,
        "next_rollout_id": 40,
        "model_files": [],
        "markers": {},
    }
    config = tmp_path / "config.json"
    config.write_bytes(b"{}")
    config_sha = uploader._sha256_file(config)
    model_sha = "d" * 64
    output_files = [
        {
            "path": "config.json",
            "bytes": config.stat().st_size,
            "sha256": config_sha,
        },
        {
            "path": "model.safetensors",
            "bytes": 4,
            "sha256": model_sha,
        },
        {
            "path": "tokenizer_config.json",
            "bytes": config.stat().st_size,
            "sha256": config_sha,
        },
    ]
    manifest = uploader._checkpoint_manifest(
        run_name=run_name,
        repo_id=repo_id,
        step=40,
        raw_checkpoint=raw,
        run_provenance_sha256="b" * 64,
        identity=identity,
        output_files=output_files,
        uploader_source_sha256="e" * 64,
    )
    prefix = "global_step_40"
    items = {
        f"{prefix}/checkpoint_manifest.json": SimpleNamespace(
            path=f"{prefix}/checkpoint_manifest.json",
            size=1,
            lfs=None,
        ),
        f"{prefix}/config.json": SimpleNamespace(
            path=f"{prefix}/config.json",
            size=2,
            lfs=None,
        ),
        f"{prefix}/model.safetensors": SimpleNamespace(
            path=f"{prefix}/model.safetensors",
            size=4,
            lfs={"sha256": model_sha, "size": 4},
        ),
        f"{prefix}/tokenizer_config.json": SimpleNamespace(
            path=f"{prefix}/tokenizer_config.json",
            size=2,
            lfs=None,
        ),
    }
    monkeypatch.setattr(
        uploader,
        "_repo_items",
        lambda *args, **kwargs: items,
    )
    monkeypatch.setattr(
        uploader,
        "_remote_json",
        lambda *args, **kwargs: manifest,
    )
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        lambda **kwargs: str(config),
    )

    assert uploader._verify_existing_remote_checkpoint(
        object(),
        repo_id=repo_id,
        run_name=run_name,
        step=40,
        raw_checkpoint=raw,
        run_provenance_sha256="b" * 64,
        identity=identity,
        uploader_source_sha256="e" * 64,
    )

    items[f"{prefix}/model.safetensors"].lfs["sha256"] = "f" * 64
    with pytest.raises(RuntimeError, match="remote LFS identity mismatch"):
        uploader._verify_existing_remote_checkpoint(
            object(),
            repo_id=repo_id,
            run_name=run_name,
            step=40,
            raw_checkpoint=raw,
            run_provenance_sha256="b" * 64,
            identity=identity,
            uploader_source_sha256="e" * 64,
        )


def test_remote_checkpoint_rejects_unprefixed_tree_inventory(
    monkeypatch,
):
    monkeypatch.setattr(
        uploader,
        "_repo_items",
        lambda *args, **kwargs: {
            "checkpoint_manifest.json": SimpleNamespace(
                path="checkpoint_manifest.json",
                size=1,
                lfs=None,
            )
        },
    )
    run_name = "core-e2-d-rl3000-seed42"
    with pytest.raises(RuntimeError, match="prefix exists without manifest"):
        uploader._verify_existing_remote_checkpoint(
            object(),
            repo_id=str(uploader.RUNS[run_name]["repo_id"]),
            run_name=run_name,
            step=40,
            raw_checkpoint={
                "logical_path": "/raw/iter_0000040",
                "iteration": 40,
                "next_rollout_id": 40,
            },
            run_provenance_sha256="b" * 64,
            identity=_identity(run_name),
            uploader_source_sha256="e" * 64,
        )
