from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from chess_rl_miles.full_train_pass16 import canonical_sha256, sha256_file
from chess_rl_miles.scripts import modal_p1_full_train_pass16 as launcher


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _with_self_hash(value: dict, field: str) -> dict:
    value[field] = canonical_sha256(value)
    return value


def _record(path: Path, relative: str, **extra: object) -> dict:
    return {
        "path": relative,
        **extra,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _complete_final_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict, Path]:
    final_root = tmp_path / "final"
    artifacts_root = tmp_path / "artifacts"
    final_root.mkdir()
    monkeypatch.setattr(launcher, "FINAL_ROOT", final_root)
    monkeypatch.setattr(launcher, "ARTIFACTS_ROOT", artifacts_root)

    summary = _with_self_hash({"metric": 0.25}, "summary_sha256")
    _write_json(final_root / "summary.json", summary)
    prepared = _with_self_hash({"source": "test"}, "manifest_sha256")
    _write_json(final_root / "prepared_manifest.json", prepared)
    contract = _with_self_hash(
        {"prepared_manifest_sha256": prepared["manifest_sha256"]},
        "contract_sha256",
    )
    _write_json(final_root / "contract.json", contract)
    (final_root / "README.md").write_text("authenticated card\n", encoding="utf-8")
    (final_root / "preview.jsonl").write_text("{}\n", encoding="utf-8")
    (final_root / "per_prompt.parquet").write_bytes(b"parquet-test")

    shard_root = final_root / "shard_success"
    shard_root.mkdir()
    for shard_id in range(launcher.SHARD_COUNT):
        shard = _with_self_hash(
            {
                "shard_id": shard_id,
                "prepared_manifest_sha256": prepared["manifest_sha256"],
            },
            "success_sha256",
        )
        _write_json(shard_root / f"shard_{shard_id:02d}.json", shard)

    data_root = artifacts_root / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    for shard_id in range(launcher.SHARD_COUNT):
        (data_root / f"rollouts-shard-{shard_id:02d}.jsonl.gz").write_bytes(
            f"raw-{shard_id}".encode()
        )

    records = []
    for relative in sorted(launcher._expected_artifact_paths()):
        if relative.startswith("data/"):
            source = artifacts_root / relative
            records.append(
                _record(
                    source,
                    relative,
                    volume_path=str(source),
                )
            )
        else:
            records.append(_record(final_root / relative, relative))
    manifest = _with_self_hash(
        {
            "schema": "p1-full-rl-train-pass16-artifact-manifest-v1",
            "version": launcher.VERSION,
            "files": records,
        },
        "manifest_sha256",
    )
    manifest_path = final_root / "artifact_manifest.json"
    _write_json(manifest_path, manifest)
    per_prompt_record = next(
        record for record in records if record["path"] == "per_prompt.parquet"
    )
    final_core = {
        "schema": "p1-full-rl-train-pass16-final-success-v1",
        "version": launcher.VERSION,
        "summary": summary,
        "artifact_manifest": str(manifest_path),
        "artifact_manifest_file": {
            "bytes": manifest_path.stat().st_size,
            "sha256": sha256_file(manifest_path),
        },
        "artifact_manifest_sha256": manifest["manifest_sha256"],
        "per_prompt": {
            "path": str(final_root / "per_prompt.parquet"),
            "rows": launcher.SOURCE_ROWS,
            "bytes": per_prompt_record["bytes"],
            "sha256": per_prompt_record["sha256"],
        },
    }
    success = {
        **final_core,
        "success_sha256": canonical_sha256(final_core),
    }
    _write_json(final_root / "_SUCCESS.json", success)
    return success, final_root


def _complete_legacy_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict, dict, Path, Path]:
    final_root = tmp_path / "final"
    backup_root = tmp_path / "legacy-final"
    artifacts_root = tmp_path / "artifacts"
    final_root.mkdir()
    monkeypatch.setattr(launcher, "FINAL_ROOT", final_root)
    monkeypatch.setattr(launcher, "LEGACY_FINAL_BACKUP_ROOT", backup_root)
    monkeypatch.setattr(launcher, "ARTIFACTS_ROOT", artifacts_root)

    summary = _with_self_hash({"metric": 0.125}, "summary_sha256")
    _write_json(final_root / "summary.json", summary)
    prepared = _with_self_hash({"source": "legacy-test"}, "manifest_sha256")
    _write_json(final_root / "prepared_manifest.json", prepared)
    contract = _with_self_hash(
        {"prepared_manifest_sha256": prepared["manifest_sha256"]},
        "contract_sha256",
    )
    _write_json(final_root / "contract.json", contract)
    (final_root / "README.md").write_text("legacy card\n", encoding="utf-8")
    (final_root / "preview.jsonl").write_text("{}\n", encoding="utf-8")
    (final_root / "per_prompt.parquet").write_bytes(b"legacy-parquet")

    shard_root = final_root / "shard_success"
    shard_root.mkdir()
    for shard_id in range(launcher.SHARD_COUNT):
        shard = _with_self_hash(
            {
                "shard_id": shard_id,
                "prepared_manifest_sha256": prepared["manifest_sha256"],
            },
            "success_sha256",
        )
        _write_json(shard_root / f"shard_{shard_id:02d}.json", shard)

    data_root = artifacts_root / "data"
    data_root.mkdir(parents=True)
    for shard_id in range(launcher.SHARD_COUNT):
        (data_root / f"rollouts-shard-{shard_id:02d}.jsonl.gz").write_bytes(
            f"raw-{shard_id}".encode()
        )

    legacy_manifest_paths = launcher._expected_artifact_paths() - {
        "README.md"
    }
    manifest_records = []
    for relative in sorted(legacy_manifest_paths):
        if relative.startswith("data/"):
            source = artifacts_root / relative
            manifest_records.append(
                _record(source, relative, volume_path=str(source))
            )
        else:
            manifest_records.append(_record(final_root / relative, relative))
    manifest = _with_self_hash(
        {
            "schema": "p1-full-rl-train-pass16-artifact-manifest-v1",
            "version": launcher.VERSION,
            "files": manifest_records,
        },
        "manifest_sha256",
    )
    manifest_path = final_root / "artifact_manifest.json"
    _write_json(manifest_path, manifest)
    per_prompt_record = next(
        record
        for record in manifest_records
        if record["path"] == "per_prompt.parquet"
    )
    final_core = {
        "schema": "p1-full-rl-train-pass16-final-success-v1",
        "version": launcher.VERSION,
        "summary": summary,
        "artifact_manifest": str(manifest_path),
        "artifact_manifest_sha256": manifest["manifest_sha256"],
        "per_prompt": {
            "path": str(final_root / "per_prompt.parquet"),
            "rows": launcher.SOURCE_ROWS,
            "bytes": per_prompt_record["bytes"],
            "sha256": per_prompt_record["sha256"],
        },
    }
    success = {
        **final_core,
        "success_sha256": canonical_sha256(final_core),
    }
    _write_json(final_root / "_SUCCESS.json", success)

    receipt_records = []
    for relative in sorted(
        launcher._expected_artifact_paths() | {"artifact_manifest.json"}
    ):
        source = (
            artifacts_root / relative
            if relative.startswith("data/")
            else final_root / relative
        )
        receipt_records.append(_record(source, relative))
    receipt_core = {
        "schema": "p1-full-rl-train-pass16-hf-receipt-v1",
        "version": launcher.VERSION,
        "repo_id": launcher.HF_DATASET_REPO,
        "repo_type": "dataset",
        "url": (
            "https://huggingface.co/datasets/"
            f"{launcher.HF_DATASET_REPO}"
        ),
        "commit_sha": "legacy-commit",
        "files": receipt_records,
        "all_remote_bytes_and_sha256_verified": True,
    }
    receipt = {
        **receipt_core,
        "receipt_sha256": canonical_sha256(receipt_core),
    }
    _write_json(final_root / "hf_upload_receipt.json", receipt)
    return success, receipt, final_root, backup_root


def test_final_artifact_closure_accepts_exact_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    success, _ = _complete_final_bundle(tmp_path, monkeypatch)

    records, manifest_file = launcher._validated_final_artifacts(success)

    assert {record["path"] for record in records} == (
        launcher._expected_artifact_paths()
    )
    assert manifest_file["sha256"] == sha256_file(
        launcher.FINAL_ROOT / "artifact_manifest.json"
    )


def test_final_artifact_closure_rejects_post_success_byte_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    success, final_root = _complete_final_bundle(tmp_path, monkeypatch)
    (final_root / "README.md").write_text("drifted\n", encoding="utf-8")

    with pytest.raises(ValueError, match="byte identity drifted"):
        launcher._validated_final_artifacts(success)


def test_finalizer_forces_multiturn_reward_mode_for_zero_env_call_rows():
    args = launcher._finalizer_reward_args()

    assert args.chess_reward_model_type == "RULE_BASED"
    assert args.chess_multiturn is True


def test_hf_verification_uses_lfs_metadata_without_downloading_blob(
    tmp_path: Path,
):
    small = tmp_path / "small.json"
    small.write_bytes(b"small")
    small_record = _record(small, "small.json")
    lfs_record = {
        "path": "data/raw.jsonl.gz",
        "bytes": 10_000_000,
        "sha256": "a" * 64,
    }
    info = SimpleNamespace(
        siblings=[
            SimpleNamespace(rfilename=".gitattributes", size=1, lfs=None),
            SimpleNamespace(
                rfilename="small.json",
                size=small_record["bytes"],
                lfs=None,
            ),
            SimpleNamespace(
                rfilename="data/raw.jsonl.gz",
                size=lfs_record["bytes"],
                lfs=SimpleNamespace(
                    size=lfs_record["bytes"],
                    sha256=lfs_record["sha256"],
                ),
            ),
        ]
    )
    calls = []

    class FakeApi:
        def repo_info(self, *args, **kwargs):
            calls.append((args, kwargs))
            return info

    downloads = []

    def download_file(**kwargs):
        downloads.append(kwargs)
        return str(small)

    launcher._verify_hf_commit(
        FakeApi(),
        commit_sha="deadbeef",
        records=[small_record, lfs_record],
        download_file=download_file,
    )

    assert calls[0][1]["revision"] == "deadbeef"
    assert calls[0][1]["files_metadata"] is True
    assert [call["filename"] for call in downloads] == ["small.json"]


def test_hf_verification_rejects_extra_remote_file():
    record = {"path": "summary.json", "bytes": 1, "sha256": "a" * 64}
    info = SimpleNamespace(
        siblings=[
            SimpleNamespace(rfilename=".gitattributes"),
            SimpleNamespace(rfilename="summary.json"),
            SimpleNamespace(rfilename="stale.bin"),
        ]
    )

    class FakeApi:
        def repo_info(self, *args, **kwargs):
            return info

    with pytest.raises(ValueError, match="extra=.*stale.bin"):
        launcher._verify_hf_commit(
            FakeApi(),
            commit_sha="deadbeef",
            records=[record],
            download_file=lambda **kwargs: None,
        )


def test_rebuild_archive_moves_completed_legacy_tree_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    success, receipt, final_root, backup_root = _complete_legacy_bundle(
        tmp_path,
        monkeypatch,
    )

    first = launcher._archive_legacy_final_for_rebuild()
    second = launcher._archive_legacy_final_for_rebuild()

    assert first["state"] == "legacy_archived"
    assert second["state"] == "legacy_already_archived"
    assert not final_root.exists()
    assert backup_root.is_dir()
    assert first["success_sha256"] == success["success_sha256"]
    assert first["receipt_sha256"] == receipt["receipt_sha256"]


def test_rebuild_archive_accepts_new_canonical_on_idempotent_rerun(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _complete_legacy_bundle(tmp_path, monkeypatch)
    launcher._archive_legacy_final_for_rebuild()
    _complete_final_bundle(tmp_path, monkeypatch)

    result = launcher._archive_legacy_final_for_rebuild()

    assert result["state"] == "canonical_already_rebuilt"
    assert launcher.FINAL_ROOT.is_dir()
    assert launcher.LEGACY_FINAL_BACKUP_ROOT.is_dir()


def test_rebuild_archive_refuses_conflicting_legacy_trees(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _, _, final_root, backup_root = _complete_legacy_bundle(
        tmp_path,
        monkeypatch,
    )
    launcher._archive_legacy_final_for_rebuild()
    shutil.copytree(backup_root, final_root)

    with pytest.raises(RuntimeError, match="rebuild conflict"):
        launcher._archive_legacy_final_for_rebuild()

    assert final_root.is_dir()
    assert backup_root.is_dir()


def test_rebuild_archive_requires_completed_legacy_upload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _, _, final_root, backup_root = _complete_legacy_bundle(
        tmp_path,
        monkeypatch,
    )
    (final_root / "hf_upload_receipt.json").unlink()

    with pytest.raises(RuntimeError, match="exact completed legacy tree"):
        launcher._archive_legacy_final_for_rebuild()

    assert final_root.is_dir()
    assert not backup_root.exists()


def test_pinned_model_weights_sha_matches_hf_lfs_identity():
    assert launcher.P1_WEIGHTS_SHA256 == (
        "e7879e769771af4284e5c78acbdd77ce91669470d882cc999cf7b88e4917fd19"
    )
