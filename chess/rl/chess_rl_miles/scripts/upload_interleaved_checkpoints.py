"""Incrementally publish complete interleaved E2 RL checkpoints to HF.

This uploader is deliberately narrower than the historical sweep uploader:

* only the two registered E2 run identities are accepted;
* a checkpoint is visible to the uploader only after the same tracker/model/
  RNG/meta readiness contract used by the trainer's Modal Volume publisher;
* the exact run provenance and origin-HF identity are verified before any
  repository write;
* every converted checkpoint is added in one atomic, append-only Hub commit;
* an existing checkpoint prefix is verified and skipped, never overwritten.

Usage:
  modal run --detach -m \
    chess_rl_miles.scripts.upload_interleaved_checkpoints
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

import modal

from chess_rl_miles.provenance import directory_identity
from chess_rl_miles.scripts import modal_interleave
from chess_rl_miles.scripts import modal_train as base


APP_NAME = "chess-interleave-rl-hf-uploader"
SCHEMA_VERSION = 1
POLL_SECONDS = 30
SAVE_INTERVAL = 40
TARGET_STEP = 3_000
HF_ORG = "Pre-to-Post-2"
RAW_ROOT = Path(modal_interleave.RAW_RL_ROOT)
ORIGIN_HF = Path(
    "/pretrain-checkpoints/interleave_50m/pretrain/"
    "mix10b_sft90k_3072_v1_20260730/exp2_monolithic/final"
)
EXPECTED_ORIGIN_MANIFEST_SHA256 = (
    "2dbcc3637470fea83ca34124224afd77862f5493f4e5455a222712a3b5314019"
)
EXPECTED_CHESS_SOURCE_SHA256 = (
    "38f829c60815cb8f7a07776561af51cc22a8b6740c431c59bd3d9847ff4c019f"
)
EXPECTED_MILES_SOURCE_SHA256 = (
    "9aeb9274e3c4d38ed2d2bdf80cd37df6b5ad3cb0202775de443bf1c6d60b9f0d"
)
EXPECTED_RUNTIME_IMAGE = (
    "radixark/miles@sha256:"
    "5b41bff2ecd42f1e71b5d8658e777541a821ef96556ae06b48333d521e0ca25e"
)
EXPECTED_INSTALLED_PACKAGES_SHA256 = (
    "1b10de190cd3f6da8afb3dad2537ef4f09e0fa05189e7007542a2de17ce29e69"
)

RUNS: dict[str, dict[str, object]] = {
    "core-e2-u-rl3000-seed42": {
        "dynamic_filter": False,
        "repo_id": (
            "Pre-to-Post-2/"
            "rl_core-e2-u-rl3000-seed42-verified-v2"
        ),
    },
    "core-e2-d-rl3000-seed42": {
        "dynamic_filter": True,
        "repo_id": (
            "Pre-to-Post-2/"
            "rl_core-e2-d-rl3000-seed42-verified-v2"
        ),
    },
}

EXPECTED_CHECKPOINT_PUBLICATION = {
    "mode": "incremental_modal_volume_commit",
    "poll_seconds": 5.0,
    "readiness_markers": [
        "latest_checkpointed_iteration.txt",
        "iter_<step>/model/.metadata",
        "iter_<step>/rng.pt",
        "iter_<step>/meta.json",
    ],
}
EXPECTED_POLICY = {
    "name": "small-model-h200",
    "max_tokens_per_gpu": 131_072,
    "gradient_checkpointing": False,
    "train_backend": "fsdp",
    "actor_num_nodes": 1,
    "actor_num_gpus_per_node": 8,
    "gpu_type": "H200",
    "host_memory_gb": 192,
    "sglang_server_concurrency": 128,
}
EXPECTED_SEMANTICS = {
    "rollout_batch_size": 256,
    "samples_per_prompt": 8,
    "global_batch_size": 2_048,
    "policy_loss_agg_mode": "token-mean",
    "advantage_estimator": "grpo",
    "cispo": False,
    "optimizer": "adamw",
    "lr": 1e-5,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_eps": 1e-8,
    "weight_decay": 0.01,
    "kl_loss_coef": 0.001,
    "rollout_max_prompt_len": 512,
    "rollout_max_response_len": 2_560,
    "rollout_max_context_len": 3_072,
}

pretrain_ckpt_vol = modal.Volume.from_name(
    "rl-reasoning-checkpoints", create_if_missing=False
)
app = modal.App(
    APP_NAME,
    image=base.image,
    secrets=base.runtime_secrets,
)


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON document is not an object: {path}")
    return value


def _require_exact_mapping(
    observed: object,
    expected: Mapping[str, object],
    *,
    label: str,
) -> None:
    if not isinstance(observed, Mapping):
        raise RuntimeError(f"{label} is not an object")
    mismatches = {
        key: {"observed": observed.get(key), "expected": value}
        for key, value in expected.items()
        if observed.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"{label} provenance drift: {mismatches}")


def _validate_run_provenance(
    payload: Mapping[str, object],
    *,
    run_name: str,
) -> Mapping[str, object]:
    if run_name not in RUNS:
        raise ValueError(f"Unregistered E2 run: {run_name}")
    identity = payload.get("identity")
    if not isinstance(identity, Mapping):
        raise RuntimeError("run provenance lacks identity")
    recorded_identity_sha = payload.get("identity_sha256")
    if recorded_identity_sha != _canonical_sha256(identity):
        raise RuntimeError("run provenance identity SHA-256 mismatch")

    expected_run = {
        "app_name": "chess-interleave-rl",
        "run_name": run_name,
        "model_id": "interleave_47m_qwen3",
        "num_rollout": TARGET_STEP,
        "dynamic_filter": RUNS[run_name]["dynamic_filter"],
        "rollout_seed": 42,
        "save_interval": SAVE_INTERVAL,
        "eval_interval": 0,
        "canary": False,
    }
    _require_exact_mapping(identity.get("run"), expected_run, label="run")
    _require_exact_mapping(
        identity.get("checkpoint_publication"),
        EXPECTED_CHECKPOINT_PUBLICATION,
        label="checkpoint_publication",
    )
    _require_exact_mapping(
        identity.get("policy_update_profile"),
        EXPECTED_POLICY,
        label="policy_update_profile",
    )
    _require_exact_mapping(
        identity.get("fixed_rl_semantics"),
        EXPECTED_SEMANTICS,
        label="fixed_rl_semantics",
    )
    _require_exact_mapping(
        identity.get("balanced_data"),
        {
            "logical_path": modal_interleave.BALANCED_TRAIN_FILE,
            "sha256": modal_interleave.BALANCED_TRAIN_SHA256,
        },
        label="balanced_data",
    )
    if identity.get("kind") != "chess_rl_miles_interleave_run":
        raise RuntimeError("unexpected run provenance kind")

    origin = identity.get("origin_hf")
    _require_exact_mapping(
        origin,
        {
            "logical_path": str(ORIGIN_HF),
            "manifest_sha256": EXPECTED_ORIGIN_MANIFEST_SHA256,
        },
        label="origin_hf",
    )
    sources = identity.get("sources")
    if not isinstance(sources, Mapping):
        raise RuntimeError("run provenance lacks sources")
    _require_exact_mapping(
        sources.get("chess_rl_miles"),
        {"manifest_sha256": EXPECTED_CHESS_SOURCE_SHA256},
        label="sources.chess_rl_miles",
    )
    _require_exact_mapping(
        sources.get("miles"),
        {"manifest_sha256": EXPECTED_MILES_SOURCE_SHA256},
        label="sources.miles",
    )
    runtime = identity.get("runtime")
    _require_exact_mapping(
        runtime,
        {
            "image": EXPECTED_RUNTIME_IMAGE,
            "installed_packages_sha256": EXPECTED_INSTALLED_PACKAGES_SHA256,
        },
        label="runtime",
    )

    command = payload.get("initial_command")
    if not isinstance(command, list) or not all(
        isinstance(value, str) for value in command
    ):
        raise RuntimeError("run provenance lacks the exact initial command")
    command_sha = hashlib.sha256(
        json.dumps(command, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if payload.get("initial_command_sha256") != command_sha:
        raise RuntimeError("run provenance initial-command SHA-256 mismatch")
    return identity


def _validate_launch_manifest(
    payload: Mapping[str, object],
    *,
    identity_sha256: str,
) -> None:
    if payload.get("identity_sha256") != identity_sha256:
        raise RuntimeError("launch provenance identity mismatch")
    command = payload.get("command")
    if not isinstance(command, list) or not all(
        isinstance(value, str) for value in command
    ):
        raise RuntimeError("launch provenance lacks command")
    command_sha = hashlib.sha256(
        json.dumps(command, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if payload.get("command_sha256") != command_sha:
        raise RuntimeError("launch provenance command SHA-256 mismatch")


def _tracker_step(run_root: Path) -> int | None:
    path = run_root / "latest_checkpointed_iteration.txt"
    try:
        step = int(path.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, OSError, ValueError):
        return None
    if step <= 0 or step > TARGET_STEP or step % SAVE_INTERVAL:
        raise RuntimeError(f"Invalid E2 checkpoint tracker step: {step}")
    return step


def _checkpoint_record(
    run_root: Path,
    *,
    step: int,
    tracker_step: int,
) -> dict[str, object] | None:
    """Return authenticated raw-checkpoint markers, or None while incomplete."""

    if (
        step <= 0
        or step > tracker_step
        or step > TARGET_STEP
        or step % SAVE_INTERVAL
    ):
        return None
    checkpoint = run_root / f"iter_{step:07d}"
    markers = {
        "model_metadata": checkpoint / "model" / ".metadata",
        "rng": checkpoint / "rng.pt",
        "meta": checkpoint / "meta.json",
    }
    if not all(path.is_file() for path in markers.values()):
        return None
    try:
        metadata = _load_json(markers["meta"])
        iteration = int(metadata["iteration"])
        next_rollout_id = int(metadata["next_rollout_id"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if iteration != step or next_rollout_id != step:
        return None
    model_files = sorted(
        path
        for path in (checkpoint / "model").rglob("*")
        if path.is_file()
    )
    if (
        not model_files
        or markers["model_metadata"] not in model_files
        or not any(path.suffix == ".distcp" for path in model_files)
    ):
        return None
    return {
        "logical_path": str(checkpoint),
        "iteration": iteration,
        "next_rollout_id": next_rollout_id,
        "model_files": [
            {
                "path": path.relative_to(checkpoint).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
            for path in model_files
        ],
        "markers": {
            name: {
                "path": path.relative_to(checkpoint).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
            for name, path in sorted(markers.items())
        },
    }


def _complete_checkpoint_records(
    run_root: Path,
    *,
    after_step: int = 0,
) -> tuple[int | None, list[dict[str, object]]]:
    tracker = _tracker_step(run_root)
    if tracker is None:
        return None, []
    if (
        after_step < 0
        or after_step > tracker
        or after_step % SAVE_INTERVAL
    ):
        raise ValueError(f"Invalid already-audited step: {after_step}")
    records: list[dict[str, object]] = []
    for step in range(
        after_step + SAVE_INTERVAL,
        tracker + 1,
        SAVE_INTERVAL,
    ):
        record = _checkpoint_record(
            run_root,
            step=step,
            tracker_step=tracker,
        )
        if record is None:
            raise RuntimeError(
                f"Tracker exposes step {tracker}, but checkpoint {step} "
                "does not satisfy the immutable readiness contract"
            )
        records.append(record)
    return tracker, records


def _output_file_records(output: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.name == "checkpoint_manifest.json":
            continue
        records.append(
            {
                "path": path.relative_to(output).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    if not records:
        raise RuntimeError("converted HF checkpoint is empty")
    names = {str(record["path"]) for record in records}
    if "config.json" not in names or "model.safetensors" not in names:
        raise RuntimeError(
            "converted checkpoint lacks config.json or model.safetensors"
        )
    if not ({"tokenizer.json", "tokenizer_config.json"} & names):
        raise RuntimeError("converted checkpoint lacks tokenizer assets")
    return records


def _checkpoint_manifest(
    *,
    run_name: str,
    repo_id: str,
    step: int,
    raw_checkpoint: Mapping[str, object],
    run_provenance_sha256: str,
    identity: Mapping[str, object],
    output_files: list[dict[str, object]],
    uploader_source_sha256: str,
) -> dict[str, object]:
    core: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "interleaved_e2_rl_hf_checkpoint",
        "repo_id": repo_id,
        "path_in_repo": f"global_step_{step}",
        "run_name": run_name,
        "step": step,
        "raw_checkpoint": dict(raw_checkpoint),
        "run_provenance": {
            "sha256": run_provenance_sha256,
            "identity_sha256": _canonical_sha256(identity),
        },
        "origin_hf": identity["origin_hf"],
        "sources": identity["sources"],
        "runtime": identity["runtime"],
        "converter": {
            "script": "miles/tools/convert_fsdp_to_hf.py",
            "uploader_source_sha256": uploader_source_sha256,
        },
        "files": output_files,
    }
    return {**core, "manifest_sha256": _canonical_sha256(core)}


def _validate_checkpoint_manifest(
    manifest: Mapping[str, object],
    *,
    run_name: str,
    repo_id: str,
    step: int,
    raw_checkpoint: Mapping[str, object],
    run_provenance_sha256: str,
    identity: Mapping[str, object],
    uploader_source_sha256: str,
) -> None:
    recorded_hash = manifest.get("manifest_sha256")
    unhashed = {
        str(key): value
        for key, value in manifest.items()
        if key != "manifest_sha256"
    }
    if recorded_hash != _canonical_sha256(unhashed):
        raise RuntimeError("remote checkpoint manifest self-hash mismatch")
    expected = {
        "schema_version": SCHEMA_VERSION,
        "kind": "interleaved_e2_rl_hf_checkpoint",
        "repo_id": repo_id,
        "path_in_repo": f"global_step_{step}",
        "run_name": run_name,
        "step": step,
        "raw_checkpoint": dict(raw_checkpoint),
        "run_provenance": {
            "sha256": run_provenance_sha256,
            "identity_sha256": _canonical_sha256(identity),
        },
        "origin_hf": identity["origin_hf"],
        "sources": identity["sources"],
        "runtime": identity["runtime"],
        "converter": {
            "script": "miles/tools/convert_fsdp_to_hf.py",
            "uploader_source_sha256": uploader_source_sha256,
        },
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(
                f"remote checkpoint manifest drift at {key}"
            )
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise RuntimeError("remote checkpoint manifest lacks files")
    paths = {
        item.get("path")
        for item in files
        if isinstance(item, Mapping)
    }
    if "config.json" not in paths or "model.safetensors" not in paths:
        raise RuntimeError("remote checkpoint manifest is incomplete")


def _publication_contract(
    *,
    run_name: str,
    repo_id: str,
    run_provenance_sha256: str,
    identity: Mapping[str, object],
    uploader_source_sha256: str,
) -> dict[str, object]:
    core: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "interleaved_e2_incremental_hf_publication",
        "append_only": True,
        "repo_id": repo_id,
        "run_name": run_name,
        "target_step": TARGET_STEP,
        "save_interval": SAVE_INTERVAL,
        "expected_steps": list(
            range(SAVE_INTERVAL, TARGET_STEP + 1, SAVE_INTERVAL)
        ),
        "readiness": EXPECTED_CHECKPOINT_PUBLICATION,
        "run_provenance": {
            "sha256": run_provenance_sha256,
            "identity_sha256": _canonical_sha256(identity),
        },
        "origin_hf": identity["origin_hf"],
        "sources": identity["sources"],
        "runtime": identity["runtime"],
        "uploader_source_sha256": uploader_source_sha256,
    }
    return {**core, "contract_sha256": _canonical_sha256(core)}


def _validate_publication_contract(
    observed: Mapping[str, object],
    expected: Mapping[str, object],
) -> None:
    recorded_hash = observed.get("contract_sha256")
    unhashed = {
        str(key): value
        for key, value in observed.items()
        if key != "contract_sha256"
    }
    if recorded_hash != _canonical_sha256(unhashed):
        raise RuntimeError("publication contract self-hash mismatch")
    if dict(observed) != dict(expected):
        raise RuntimeError("existing HF publication contract differs")


def _repo_items(
    api,
    repo_id: str,
    *,
    path_in_repo: str = "",
) -> dict[str, object]:
    try:
        items = api.list_repo_tree(
            repo_id=repo_id,
            repo_type="model",
            path_in_repo=path_in_repo,
            recursive=True,
            expand=True,
        )
        return {str(item.path): item for item in items}
    except Exception as exc:
        text = str(exc)
        if "404" in text or "not found" in text.lower():
            return {}
        raise


def _repo_paths(api, repo_id: str, *, path_in_repo: str = "") -> set[str]:
    return set(_repo_items(api, repo_id, path_in_repo=path_in_repo))


def _remote_json(api, repo_id: str, path: str) -> dict[str, Any]:
    from huggingface_hub import hf_hub_download

    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=path,
        repo_type="model",
        token=os.environ.get("HF_TOKEN"),
    )
    return _load_json(Path(downloaded))


def _initialize_or_verify_repo(
    api,
    *,
    repo_id: str,
    run_root: Path,
    run_provenance: Mapping[str, object],
    contract: Mapping[str, object],
) -> set[str]:
    from huggingface_hub import CommitOperationAdd

    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    paths = _repo_paths(api, repo_id)
    root_files = {"run_provenance.json", "publication_contract.json"}
    present_root = root_files & paths
    if present_root:
        if present_root != root_files:
            raise RuntimeError(
                f"HF repo has a partial publication root: {repo_id}"
            )
        remote_provenance = _remote_json(
            api, repo_id, "run_provenance.json"
        )
        if remote_provenance != dict(run_provenance):
            raise RuntimeError("existing HF run provenance differs")
        _validate_publication_contract(
            _remote_json(api, repo_id, "publication_contract.json"),
            contract,
        )
    else:
        unexpected = {
            path for path in paths if path != ".gitattributes"
        }
        if unexpected:
            raise RuntimeError(
                f"Refusing to initialize nonempty HF repo {repo_id}: "
                f"{sorted(unexpected)[:5]}"
            )
        with tempfile.TemporaryDirectory(
            prefix="interleave-hf-root-"
        ) as temporary:
            temporary_root = Path(temporary)
            provenance_path = temporary_root / "run_provenance.json"
            contract_path = temporary_root / "publication_contract.json"
            provenance_path.write_text(
                json.dumps(
                    run_provenance,
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=True,
                )
                + "\n",
                encoding="utf-8",
            )
            contract_path.write_text(
                json.dumps(
                    contract,
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=True,
                )
                + "\n",
                encoding="utf-8",
            )
            head = api.model_info(repo_id=repo_id).sha
            api.create_commit(
                repo_id=repo_id,
                repo_type="model",
                operations=[
                    CommitOperationAdd(
                        path_in_repo="run_provenance.json",
                        path_or_fileobj=str(provenance_path),
                    ),
                    CommitOperationAdd(
                        path_in_repo="publication_contract.json",
                        path_or_fileobj=str(contract_path),
                    ),
                ],
                commit_message="Initialize immutable E2 RL publication",
                parent_commit=head,
            )
        remote_provenance = _remote_json(
            api, repo_id, "run_provenance.json"
        )
        if remote_provenance != dict(run_provenance):
            raise RuntimeError("post-commit HF run provenance mismatch")
        _validate_publication_contract(
            _remote_json(api, repo_id, "publication_contract.json"),
            contract,
        )

    # Launch manifests are append-only and may grow if Modal resumes training.
    identity_sha = str(run_provenance["identity_sha256"])
    for local_path in sorted((run_root / "provenance").glob("launch_*.json")):
        launch = _load_json(local_path)
        _validate_launch_manifest(launch, identity_sha256=identity_sha)
        remote_path = f"provenance/{local_path.name}"
        if remote_path in paths:
            if _remote_json(api, repo_id, remote_path) != launch:
                raise RuntimeError(
                    f"existing HF launch provenance differs: {remote_path}"
                )
            continue
        head = api.model_info(repo_id=repo_id).sha
        api.create_commit(
            repo_id=repo_id,
            repo_type="model",
            operations=[
                CommitOperationAdd(
                    path_in_repo=remote_path,
                    path_or_fileobj=str(local_path),
                )
            ],
            commit_message=f"Add immutable launch provenance {local_path.name}",
            parent_commit=head,
        )
    return {
        path.name
        for path in (run_root / "provenance").glob("launch_*.json")
    }


def _verify_existing_remote_checkpoint(
    api,
    *,
    repo_id: str,
    run_name: str,
    step: int,
    raw_checkpoint: Mapping[str, object],
    run_provenance_sha256: str,
    identity: Mapping[str, object],
    uploader_source_sha256: str,
) -> bool:
    from huggingface_hub import hf_hub_download

    prefix = f"global_step_{step}"
    items = _repo_items(api, repo_id, path_in_repo=prefix)
    if not items:
        return False
    paths = set(items)
    manifest_path = f"{prefix}/checkpoint_manifest.json"
    if manifest_path not in paths:
        raise RuntimeError(
            f"HF checkpoint prefix exists without manifest: {repo_id}/{prefix}"
        )
    manifest = _remote_json(api, repo_id, manifest_path)
    _validate_checkpoint_manifest(
        manifest,
        run_name=run_name,
        repo_id=repo_id,
        step=step,
        raw_checkpoint=raw_checkpoint,
        run_provenance_sha256=run_provenance_sha256,
        identity=identity,
        uploader_source_sha256=uploader_source_sha256,
    )
    expected_paths = {
        f"{prefix}/{item['path']}"
        for item in manifest["files"]
        if isinstance(item, Mapping)
    }
    expected_paths.add(manifest_path)
    if paths != expected_paths:
        raise RuntimeError(
            f"HF checkpoint file inventory differs at {repo_id}/{prefix}: "
            f"missing={sorted(expected_paths - paths)} "
            f"extra={sorted(paths - expected_paths)}"
        )
    for record in manifest["files"]:
        if not isinstance(record, Mapping):
            raise RuntimeError("remote checkpoint file record is invalid")
        relative = str(record.get("path") or "")
        expected_size = int(record.get("bytes", -1))
        expected_sha = str(record.get("sha256") or "")
        if not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
            raise RuntimeError(
                f"remote checkpoint manifest has invalid SHA: {relative}"
            )
        remote_path = f"{prefix}/{relative}"
        item = items[remote_path]
        observed_size = int(getattr(item, "size", -1))
        if observed_size != expected_size:
            raise RuntimeError(
                f"remote file size mismatch at {repo_id}/{remote_path}: "
                f"{observed_size} != {expected_size}"
            )
        lfs = getattr(item, "lfs", None)
        if lfs is not None:
            observed_sha = str(
                getattr(lfs, "sha256", None)
                or (
                    lfs.get("sha256")
                    if isinstance(lfs, Mapping)
                    else ""
                )
            )
            observed_lfs_size = int(
                getattr(lfs, "size", -1)
                if not isinstance(lfs, Mapping)
                else lfs.get("size", -1)
            )
            if (
                observed_sha != expected_sha
                or observed_lfs_size != expected_size
            ):
                raise RuntimeError(
                    f"remote LFS identity mismatch at "
                    f"{repo_id}/{remote_path}"
                )
            continue
        downloaded = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=remote_path,
                repo_type="model",
                token=os.environ.get("HF_TOKEN"),
            )
        )
        if (
            downloaded.stat().st_size != expected_size
            or _sha256_file(downloaded) != expected_sha
        ):
            raise RuntimeError(
                f"remote non-LFS content mismatch at "
                f"{repo_id}/{remote_path}"
            )
    return True


def _convert_and_publish_step(
    api,
    *,
    run_name: str,
    repo_id: str,
    step: int,
    raw_checkpoint: Mapping[str, object],
    run_provenance_sha256: str,
    identity: Mapping[str, object],
    uploader_source_sha256: str,
) -> str:
    from huggingface_hub import CommitOperationAdd

    source = RAW_ROOT / run_name / f"iter_{step:07d}"
    with tempfile.TemporaryDirectory(
        prefix=f"e2-{run_name}-step-{step}-"
    ) as temporary:
        output = Path(temporary) / f"global_step_{step}"
        command = [
            sys.executable,
            str(Path(base.MILES_DIR) / "tools" / "convert_fsdp_to_hf.py"),
            "--input-dir",
            str(source),
            "--origin-hf-dir",
            str(ORIGIN_HF),
            "--output-dir",
            str(output),
            "--force",
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = (
            f"{base.PROJECT_DIR}:{base.MILES_DIR}:"
            f"{env.get('PYTHONPATH', '')}"
        )
        env["HF_HOME"] = base.HF_CACHE_DIR
        env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        env["TQDM_DISABLE"] = "1"
        result = subprocess.run(
            command,
            cwd=base.MILES_DIR,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if result.returncode:
            print(result.stdout[-12_000:], flush=True)
            raise RuntimeError(
                f"convert_fsdp_to_hf failed for {run_name} step {step}: "
                f"exit {result.returncode}"
            )
        output_files = _output_file_records(output)
        manifest = _checkpoint_manifest(
            run_name=run_name,
            repo_id=repo_id,
            step=step,
            raw_checkpoint=raw_checkpoint,
            run_provenance_sha256=run_provenance_sha256,
            identity=identity,
            output_files=output_files,
            uploader_source_sha256=uploader_source_sha256,
        )
        manifest_path = output / "checkpoint_manifest.json"
        manifest_path.write_text(
            json.dumps(
                manifest,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
            )
            + "\n",
            encoding="utf-8",
        )

        prefix = f"global_step_{step}"
        operations = [
            CommitOperationAdd(
                path_in_repo=f"{prefix}/{path.relative_to(output).as_posix()}",
                path_or_fileobj=str(path),
            )
            for path in sorted(output.rglob("*"))
            if path.is_file()
        ]
        head = api.model_info(repo_id=repo_id).sha
        commit = api.create_commit(
            repo_id=repo_id,
            repo_type="model",
            operations=operations,
            commit_message=(
                f"Publish complete {run_name} checkpoint step {step}"
            ),
            parent_commit=head,
        )
        commit_id = str(getattr(commit, "oid", "") or "")
        if not re.fullmatch(r"[0-9a-f]{40}", commit_id):
            raise RuntimeError(
                f"HF returned an invalid commit ID for step {step}: "
                f"{commit_id!r}"
            )

    if not _verify_existing_remote_checkpoint(
        api,
        repo_id=repo_id,
        run_name=run_name,
        step=step,
        raw_checkpoint=raw_checkpoint,
        run_provenance_sha256=run_provenance_sha256,
        identity=identity,
        uploader_source_sha256=uploader_source_sha256,
    ):
        raise RuntimeError("post-commit HF checkpoint verification failed")
    return commit_id


@app.function(
    cpu=8.0,
    memory=64 * 1024,
    timeout=47 * 60 * 60,
    retries=modal.Retries(initial_delay=10.0, max_retries=2),
    max_containers=2,
    volumes={
        "/rl-checkpoints": base.ckpt_vol,
        "/pretrain-checkpoints": pretrain_ckpt_vol,
        base.HF_CACHE_DIR: base.hf_cache,
    },
)
def monitor_and_upload(run_name: str) -> dict[str, object]:
    from huggingface_hub import HfApi

    if run_name not in RUNS:
        raise ValueError(f"Unregistered E2 run: {run_name}")
    repo_id = str(RUNS[run_name]["repo_id"])
    uploader_source_sha256 = _sha256_file(Path(__file__))
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN is required before any repository operation"
        )
    api = HfApi(token=token)
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TQDM_DISABLE"] = "1"

    base.ckpt_vol.reload()
    pretrain_ckpt_vol.reload()
    run_root = RAW_ROOT / run_name
    provenance_path = run_root / "run_provenance.json"
    if not provenance_path.is_file():
        raise FileNotFoundError(provenance_path)
    run_provenance = _load_json(provenance_path)
    identity = _validate_run_provenance(
        run_provenance,
        run_name=run_name,
    )
    observed_origin = directory_identity(
        ORIGIN_HF,
        logical_path=str(ORIGIN_HF),
    )
    if observed_origin != identity["origin_hf"]:
        raise RuntimeError(
            "origin HF bytes differ from immutable RL run provenance"
        )
    run_provenance_sha256 = _sha256_file(provenance_path)
    contract = _publication_contract(
        run_name=run_name,
        repo_id=repo_id,
        run_provenance_sha256=run_provenance_sha256,
        identity=identity,
        uploader_source_sha256=uploader_source_sha256,
    )
    known_launch_names = _initialize_or_verify_repo(
        api,
        repo_id=repo_id,
        run_root=run_root,
        run_provenance=run_provenance,
        contract=contract,
    )
    print(
        "[e2-hf] initialized "
        f"run={run_name} repo={repo_id} "
        f"identity={run_provenance['identity_sha256']} "
        f"contract={contract['contract_sha256']}",
        flush=True,
    )

    uploaded: dict[int, str] = {}
    raw_records: dict[int, dict[str, object]] = {}
    while True:
        base.ckpt_vol.reload()
        after_step = max(raw_records, default=0)
        tracker, new_records = _complete_checkpoint_records(
            run_root,
            after_step=after_step,
        )
        for record in new_records:
            raw_records[int(record["iteration"])] = record
        local_launch_names = {
            path.name
            for path in (run_root / "provenance").glob("launch_*.json")
        }
        if local_launch_names != known_launch_names:
            if not known_launch_names.issubset(local_launch_names):
                raise RuntimeError(
                    f"local launch provenance was removed for {run_name}"
                )
            known_launch_names = _initialize_or_verify_repo(
                api,
                repo_id=repo_id,
                run_root=run_root,
                run_provenance=run_provenance,
                contract=contract,
            )
        for step in sorted(set(raw_records) - set(uploaded)):
            raw_checkpoint = raw_records[step]
            if _verify_existing_remote_checkpoint(
                api,
                repo_id=repo_id,
                run_name=run_name,
                step=step,
                raw_checkpoint=raw_checkpoint,
                run_provenance_sha256=run_provenance_sha256,
                identity=identity,
                uploader_source_sha256=uploader_source_sha256,
            ):
                uploaded.setdefault(step, "verified_existing")
                continue
            print(
                f"[e2-hf] converting complete checkpoint "
                f"run={run_name} step={step}",
                flush=True,
            )
            commit_id = _convert_and_publish_step(
                api,
                run_name=run_name,
                repo_id=repo_id,
                step=step,
                raw_checkpoint=raw_checkpoint,
                run_provenance_sha256=run_provenance_sha256,
                identity=identity,
                uploader_source_sha256=uploader_source_sha256,
            )
            uploaded[step] = commit_id
            print(
                f"[e2-hf] published run={run_name} step={step} "
                f"commit={commit_id}",
                flush=True,
            )

        expected_count = tracker // SAVE_INTERVAL if tracker else 0
        if (
            len(raw_records) != expected_count
            or len(uploaded) != expected_count
        ):
            raise RuntimeError(
                f"HF publication accounting mismatch for {run_name}: "
                f"tracker={tracker} audited={sorted(raw_records)} "
                f"uploaded={sorted(uploaded)}"
            )
        print(
            f"[e2-hf] status run={run_name} tracker={tracker or 0}/"
            f"{TARGET_STEP} uploaded={len(uploaded)}/"
            f"{TARGET_STEP // SAVE_INTERVAL}",
            flush=True,
        )
        if tracker == TARGET_STEP:
            return {
                "run_name": run_name,
                "repo_id": repo_id,
                "tracker_step": tracker,
                "uploaded_count": len(uploaded),
                "uploaded_steps": sorted(uploaded),
                "run_identity_sha256": run_provenance[
                    "identity_sha256"
                ],
                "publication_contract_sha256": contract[
                    "contract_sha256"
                ],
                "uploader_source_sha256": uploader_source_sha256,
            }
        time.sleep(POLL_SECONDS)


@app.local_entrypoint()
def main(runs: str = ",".join(RUNS)) -> None:
    chosen = [value.strip() for value in runs.split(",") if value.strip()]
    unknown = sorted(set(chosen) - set(RUNS))
    if unknown:
        raise ValueError(f"Unregistered E2 runs: {unknown}")
    if len(chosen) != len(set(chosen)):
        raise ValueError("Duplicate E2 run requested")
    handles = {
        run_name: monitor_and_upload.spawn(run_name)
        for run_name in chosen
    }
    print(
        json.dumps(
            {
                run_name: {
                    "call_id": handle.object_id,
                    "repo_id": RUNS[run_name]["repo_id"],
                }
                for run_name, handle in handles.items()
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
