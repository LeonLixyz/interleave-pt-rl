"""Exhaustive P1 rollout-only pass@16 on all 53,225 RL source rows.

Examples:
    modal run .../modal_p1_full_train_pass16.py --action prepare
    modal run .../modal_p1_full_train_pass16.py --action upload-input
    modal run --detach .../modal_p1_full_train_pass16.py --action launch
    modal run .../modal_p1_full_train_pass16.py --action status
    modal run --detach .../modal_p1_full_train_pass16.py --action rebuild-final
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any

import modal

_launcher_path = Path(__file__).resolve()
_PROJECT_DIR_CANDIDATES = (
    *(
        (_launcher_path.parents[2],)
        if len(_launcher_path.parents) > 2
        else ()
    ),
    Path("/root/chess-rl-miles"),
)
for _project_dir in _PROJECT_DIR_CANDIDATES:
    if (
        (_project_dir / "chess_rl_miles").is_dir()
        and str(_project_dir) not in sys.path
    ):
        sys.path.insert(0, str(_project_dir))
        break

from chess_rl_miles.full_train_pass16 import (
    BASE_SEED,
    EXPECTED_TRAJECTORIES,
    EXHAUSTIVE_PROMPT_CAP,
    RL_PROMPT_CAP,
    ROLLOUTS_PER_SHARD,
    ROLLOUT_BATCH_SIZES,
    SAMPLES_PER_PROMPT,
    SHARD_COUNT,
    SOURCE_ROWS,
    SOURCE_SHA256,
    atomic_json,
    canonical_sha256,
    checkpoint_fingerprint,
    iter_jsonl,
    prepare_shards,
    sha256_file,
    shard_ranges,
    validate_rollout_rows,
    wilson_interval,
)
from chess_rl_miles.scripts import modal_interleave as shared


base = shared.base
APP_NAME = "chess-p1-full-train-pass16"
VERSION = "p1-full-rl-train-pass16-20260730-r1"
RESULTS_VOLUME_NAME = "chess-p1-full-train-pass16-20260730-v1"
RESULTS_MOUNT = Path("/full-eval")
INPUT_ROOT = RESULTS_MOUNT / "input"
RUNS_ROOT = RESULTS_MOUNT / "runs"
ARTIFACTS_ROOT = RESULTS_MOUNT / "artifacts"
FINAL_ROOT = RESULTS_MOUNT / "final"
LEGACY_FINAL_BACKUP_ROOT = (
    RESULTS_MOUNT
    / "final-before-hardening-p1-full-rl-train-pass16-20260730-r1"
)
P1_CHECKPOINT = Path(
    "/pretrain-checkpoints/interleave_50m/pretrain/"
    "mix10b_sft90k_3072_v1_20260730/p1_shared/final"
)
P1_CHECKPOINT_FINGERPRINT = (
    "0f402347123cf8e7524d3e31ff3a60d5bd8c86b4d81e1fc1c5f7d28d276be503"
)
P1_HF_REPO = "Pre-to-Post-2/pretrain_interleave_47m_v1_p1_5b"
P1_HF_REVISION = "2e9aa09ae3357cc1007e6724754e0c0255ee4c79"
P1_WEIGHTS_SHA256 = (
    "e7879e769771af4284e5c78acbdd77ce91669470d882cc999cf7b88e4917fd19"
)
HF_DATASET_REPO = (
    "Pre-to-Post-2/interleave_47m_v1_p1_5b_full_train_pass16"
)
STRICT_SOURCE = (
    "chess_rl_miles.exhaustive_data_source."
    "StrictExhaustiveRolloutDataSource"
)
_LOCAL_WORKSPACE = (
    _launcher_path.parents[3]
    if len(_launcher_path.parents) > 3
    else Path("/root")
)
LOCAL_SOURCE_DEFAULT = str(
    _LOCAL_WORKSPACE / "train_v4_dataset_balanced_multi_turn.parquet"
)
LOCAL_TOKENIZER_DEFAULT = str(
    _LOCAL_WORKSPACE / ".cache" / "p1_contract"
)
LOCAL_PREPARED_DEFAULT = str(
    _LOCAL_WORKSPACE / ".artifacts" / VERSION / "input"
)
LOCAL_LEDGER_DEFAULT = str(
    _LOCAL_WORKSPACE / "P1_FULL_TRAIN_PASS16_LAUNCH_LEDGER.json"
)
NODE_TIMEOUT_SECONDS = 3 * 60 * 60
ROUTER_HEALTH_INTERVAL_SECONDS = 1e18

results_vol = modal.Volume.from_name(
    RESULTS_VOLUME_NAME,
    create_if_missing=True,
    version=2,
)
app = modal.App(
    APP_NAME,
    image=base.image,
    secrets=base.runtime_secrets,
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _self_hashed(value: dict[str, Any], field: str) -> bool:
    expected = value.get(field)
    core = {key: item for key, item in value.items() if key != field}
    return isinstance(expected, str) and expected == canonical_sha256(core)


def _expected_artifact_paths() -> set[str]:
    return {
        "README.md",
        "summary.json",
        "contract.json",
        "prepared_manifest.json",
        "per_prompt.parquet",
        "preview.jsonl",
        *{
            f"shard_success/shard_{shard_id:02d}.json"
            for shard_id in range(SHARD_COUNT)
        },
        *{
            f"data/rollouts-shard-{shard_id:02d}.jsonl.gz"
            for shard_id in range(SHARD_COUNT)
        },
    }


def _validate_record_identity(
    path: Path,
    record: dict[str, Any],
    *,
    label: str,
) -> None:
    expected_bytes = record.get("bytes")
    expected_sha256 = record.get("sha256")
    if (
        isinstance(expected_bytes, bool)
        or not isinstance(expected_bytes, int)
        or expected_bytes < 0
        or not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise ValueError(f"{label} has an invalid byte/SHA256 identity")
    if (
        not path.is_file()
        or path.stat().st_size != expected_bytes
        or sha256_file(path) != expected_sha256
    ):
        raise ValueError(f"{label} byte identity drifted: {path}")


def _validated_final_artifacts(
    success: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Authenticate the complete local upload closure from final _SUCCESS."""

    if (
        success.get("schema")
        != "p1-full-rl-train-pass16-final-success-v1"
        or success.get("version") != VERSION
        or not _self_hashed(success, "success_sha256")
    ):
        raise ValueError("final success marker is not authenticated")
    manifest_path = FINAL_ROOT / "artifact_manifest.json"
    if success.get("artifact_manifest") != str(manifest_path):
        raise ValueError("final success points at an unexpected artifact manifest")
    manifest_file = success.get("artifact_manifest_file")
    if not isinstance(manifest_file, dict):
        raise ValueError("final success lacks artifact-manifest byte identity")
    _validate_record_identity(
        manifest_path,
        manifest_file,
        label="artifact manifest",
    )
    manifest = _read_json(manifest_path)
    if (
        manifest.get("schema")
        != "p1-full-rl-train-pass16-artifact-manifest-v1"
        or manifest.get("version") != VERSION
        or not _self_hashed(manifest, "manifest_sha256")
        or success.get("artifact_manifest_sha256")
        != manifest.get("manifest_sha256")
    ):
        raise ValueError("artifact manifest authentication failed")

    raw_records = manifest.get("files")
    if not isinstance(raw_records, list):
        raise ValueError("artifact manifest files is not a list")
    records: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for raw_record in raw_records:
        if not isinstance(raw_record, dict):
            raise ValueError("artifact manifest file record is not an object")
        record = dict(raw_record)
        relative = record.get("path")
        if not isinstance(relative, str):
            raise ValueError("artifact path is not a string")
        normalized = PurePosixPath(relative)
        if (
            normalized.is_absolute()
            or relative != normalized.as_posix()
            or not normalized.parts
            or any(part in {"", ".", ".."} for part in normalized.parts)
            or relative in seen_paths
        ):
            raise ValueError(f"invalid or duplicate artifact path: {relative!r}")
        seen_paths.add(relative)

        is_raw = relative.startswith("data/")
        expected_keys = (
            {"path", "volume_path", "bytes", "sha256"}
            if is_raw
            else {"path", "bytes", "sha256"}
        )
        if set(record) != expected_keys:
            raise ValueError(f"unexpected artifact record fields: {relative}")
        if is_raw:
            expected_volume_path = str(ARTIFACTS_ROOT / relative)
            if record.get("volume_path") != expected_volume_path:
                raise ValueError(f"raw artifact volume path drifted: {relative}")
            source = Path(expected_volume_path)
        else:
            source = FINAL_ROOT / relative
        _validate_record_identity(source, record, label=f"artifact {relative}")
        records.append(record)

    expected_paths = _expected_artifact_paths()
    if seen_paths != expected_paths:
        raise ValueError(
            "artifact inventory drifted: "
            f"missing={sorted(expected_paths - seen_paths)} "
            f"extra={sorted(seen_paths - expected_paths)}"
        )

    summary = _read_json(FINAL_ROOT / "summary.json")
    if (
        not _self_hashed(summary, "summary_sha256")
        or summary != success.get("summary")
    ):
        raise ValueError("summary is not closed by final success")
    contract = _read_json(FINAL_ROOT / "contract.json")
    if not _self_hashed(contract, "contract_sha256"):
        raise ValueError("contract authentication failed")
    prepared = _read_json(FINAL_ROOT / "prepared_manifest.json")
    if (
        not _self_hashed(prepared, "manifest_sha256")
        or contract.get("prepared_manifest_sha256")
        != prepared.get("manifest_sha256")
    ):
        raise ValueError("prepared manifest authentication failed")
    for shard_id in range(SHARD_COUNT):
        shard_success = _read_json(
            FINAL_ROOT / "shard_success" / f"shard_{shard_id:02d}.json"
        )
        if (
            not _self_hashed(shard_success, "success_sha256")
            or shard_success.get("prepared_manifest_sha256")
            != prepared.get("manifest_sha256")
        ):
            raise ValueError(f"shard {shard_id} success authentication failed")

    per_prompt = success.get("per_prompt")
    if (
        not isinstance(per_prompt, dict)
        or per_prompt.get("path") != str(FINAL_ROOT / "per_prompt.parquet")
        or per_prompt.get("rows") != SOURCE_ROWS
    ):
        raise ValueError("final per-prompt identity drifted")
    per_prompt_record = next(
        record for record in records if record["path"] == "per_prompt.parquet"
    )
    if (
        per_prompt.get("bytes") != per_prompt_record["bytes"]
        or per_prompt.get("sha256") != per_prompt_record["sha256"]
    ):
        raise ValueError("per-prompt identities disagree")
    return records, manifest_file


def _verify_hf_commit(
    api: Any,
    *,
    commit_sha: str,
    records: list[dict[str, Any]],
    download_file: Any,
) -> None:
    """Verify exact Hub inventory and content without re-downloading LFS blobs."""

    info = api.repo_info(
        HF_DATASET_REPO,
        repo_type="dataset",
        revision=commit_sha,
        files_metadata=True,
    )
    items = {sibling.rfilename: sibling for sibling in info.siblings}
    expected_paths = {record["path"] for record in records}
    remote_paths = set(items)
    allowed_remote_paths = expected_paths | {".gitattributes"}
    if remote_paths != allowed_remote_paths:
        raise ValueError(
            "HF remote file inventory drifted: "
            f"missing={sorted(allowed_remote_paths - remote_paths)} "
            f"extra={sorted(remote_paths - allowed_remote_paths)}"
        )

    for record in records:
        relative = record["path"]
        item = items[relative]
        expected_bytes = int(record["bytes"])
        expected_sha256 = str(record["sha256"])
        if int(getattr(item, "size", -1)) != expected_bytes:
            raise ValueError(f"HF remote size verification failed: {relative}")
        lfs = getattr(item, "lfs", None)
        if lfs is not None:
            observed_sha256 = str(
                lfs.get("sha256", "")
                if isinstance(lfs, Mapping)
                else getattr(lfs, "sha256", "")
            )
            observed_lfs_bytes = int(
                lfs.get("size", -1)
                if isinstance(lfs, Mapping)
                else getattr(lfs, "size", -1)
            )
            if (
                observed_sha256 != expected_sha256
                or observed_lfs_bytes != expected_bytes
            ):
                raise ValueError(
                    f"HF remote LFS verification failed: {relative}"
                )
            continue
        downloaded = Path(
            download_file(
                repo_id=HF_DATASET_REPO,
                filename=relative,
                repo_type="dataset",
                revision=commit_sha,
                token=os.environ.get("HF_TOKEN"),
            )
        )
        if (
            not downloaded.is_file()
            or downloaded.stat().st_size != expected_bytes
            or sha256_file(downloaded) != expected_sha256
        ):
            raise ValueError(
                f"HF remote non-LFS verification failed: {relative}"
            )


def _validate_legacy_final_tree(root: Path) -> dict[str, Any]:
    """Validate the complete output emitted by the pre-hardening controller."""

    success = _read_json(root / "_SUCCESS.json")
    if (
        success.get("schema")
        != "p1-full-rl-train-pass16-final-success-v1"
        or success.get("version") != VERSION
        or not _self_hashed(success, "success_sha256")
        or success.get("artifact_manifest")
        != str(FINAL_ROOT / "artifact_manifest.json")
    ):
        raise ValueError(f"legacy final success is not authenticated: {root}")
    manifest = _read_json(root / "artifact_manifest.json")
    if (
        manifest.get("schema")
        != "p1-full-rl-train-pass16-artifact-manifest-v1"
        or manifest.get("version") != VERSION
        or not _self_hashed(manifest, "manifest_sha256")
        or success.get("artifact_manifest_sha256")
        != manifest.get("manifest_sha256")
    ):
        raise ValueError(f"legacy artifact manifest is not authenticated: {root}")

    raw_manifest_records = manifest.get("files")
    if not isinstance(raw_manifest_records, list):
        raise ValueError("legacy artifact manifest files is not a list")
    manifest_records: dict[str, dict[str, Any]] = {}
    for raw_record in raw_manifest_records:
        if not isinstance(raw_record, dict):
            raise ValueError("legacy artifact record is not an object")
        record = dict(raw_record)
        relative = record.get("path")
        if not isinstance(relative, str):
            raise ValueError("legacy artifact path is not a string")
        normalized = PurePosixPath(relative)
        if (
            normalized.is_absolute()
            or relative != normalized.as_posix()
            or not normalized.parts
            or any(part in {"", ".", ".."} for part in normalized.parts)
            or relative in manifest_records
        ):
            raise ValueError(
                f"invalid or duplicate legacy artifact path: {relative!r}"
            )
        is_raw = relative.startswith("data/")
        expected_keys = (
            {"path", "volume_path", "bytes", "sha256"}
            if is_raw
            else {"path", "bytes", "sha256"}
        )
        if set(record) != expected_keys:
            raise ValueError(
                f"unexpected legacy artifact record fields: {relative}"
            )
        if is_raw:
            expected_volume_path = str(ARTIFACTS_ROOT / relative)
            if record.get("volume_path") != expected_volume_path:
                raise ValueError(
                    f"legacy raw artifact volume path drifted: {relative}"
                )
            source = Path(expected_volume_path)
        else:
            source = root / relative
        _validate_record_identity(
            source,
            record,
            label=f"legacy artifact {relative}",
        )
        manifest_records[relative] = record

    expected_manifest_paths = _expected_artifact_paths() - {"README.md"}
    if set(manifest_records) != expected_manifest_paths:
        raise ValueError(
            "legacy artifact inventory drifted: "
            f"missing={sorted(expected_manifest_paths - set(manifest_records))} "
            f"extra={sorted(set(manifest_records) - expected_manifest_paths)}"
        )
    summary = _read_json(root / "summary.json")
    if (
        not _self_hashed(summary, "summary_sha256")
        or summary != success.get("summary")
    ):
        raise ValueError("legacy summary is not closed by final success")
    prepared = _read_json(root / "prepared_manifest.json")
    contract = _read_json(root / "contract.json")
    if (
        not _self_hashed(prepared, "manifest_sha256")
        or not _self_hashed(contract, "contract_sha256")
        or contract.get("prepared_manifest_sha256")
        != prepared.get("manifest_sha256")
    ):
        raise ValueError("legacy contract/prepared authentication failed")
    for shard_id in range(SHARD_COUNT):
        shard_success = _read_json(
            root / "shard_success" / f"shard_{shard_id:02d}.json"
        )
        if (
            not _self_hashed(shard_success, "success_sha256")
            or shard_success.get("prepared_manifest_sha256")
            != prepared.get("manifest_sha256")
        ):
            raise ValueError(
                f"legacy shard {shard_id} success authentication failed"
            )
    per_prompt = success.get("per_prompt")
    per_prompt_record = manifest_records["per_prompt.parquet"]
    if (
        not isinstance(per_prompt, dict)
        or per_prompt.get("path") != str(FINAL_ROOT / "per_prompt.parquet")
        or per_prompt.get("rows") != SOURCE_ROWS
        or per_prompt.get("bytes") != per_prompt_record["bytes"]
        or per_prompt.get("sha256") != per_prompt_record["sha256"]
    ):
        raise ValueError("legacy per-prompt identity drifted")

    receipt = _read_json(root / "hf_upload_receipt.json")
    if (
        receipt.get("schema")
        != "p1-full-rl-train-pass16-hf-receipt-v1"
        or receipt.get("version") != VERSION
        or receipt.get("repo_id") != HF_DATASET_REPO
        or receipt.get("repo_type") != "dataset"
        or not isinstance(receipt.get("commit_sha"), str)
        or not receipt["commit_sha"]
        or receipt.get("all_remote_bytes_and_sha256_verified") is not True
        or not _self_hashed(receipt, "receipt_sha256")
    ):
        raise ValueError("legacy HF upload receipt is not authenticated")
    raw_receipt_records = receipt.get("files")
    if not isinstance(raw_receipt_records, list):
        raise ValueError("legacy HF receipt files is not a list")
    receipt_records: dict[str, dict[str, Any]] = {}
    for raw_record in raw_receipt_records:
        if not isinstance(raw_record, dict) or set(raw_record) != {
            "path",
            "bytes",
            "sha256",
        }:
            raise ValueError("legacy HF receipt file record is invalid")
        record = dict(raw_record)
        relative = record.get("path")
        if not isinstance(relative, str):
            raise ValueError("legacy HF receipt path is not a string")
        normalized = PurePosixPath(relative)
        if (
            normalized.is_absolute()
            or relative != normalized.as_posix()
            or not normalized.parts
            or any(part in {"", ".", ".."} for part in normalized.parts)
            or relative in receipt_records
        ):
            raise ValueError(
                f"invalid or duplicate legacy receipt path: {relative!r}"
            )
        source = (
            ARTIFACTS_ROOT / relative
            if relative.startswith("data/")
            else root / relative
        )
        _validate_record_identity(
            source,
            record,
            label=f"legacy uploaded artifact {relative}",
        )
        receipt_records[relative] = record
    expected_receipt_paths = _expected_artifact_paths() | {
        "artifact_manifest.json"
    }
    if set(receipt_records) != expected_receipt_paths:
        raise ValueError(
            "legacy HF receipt inventory drifted: "
            f"missing={sorted(expected_receipt_paths - set(receipt_records))} "
            f"extra={sorted(set(receipt_records) - expected_receipt_paths)}"
        )

    expected_tree_paths = {
        relative
        for relative in expected_manifest_paths
        if not relative.startswith("data/")
    } | {
        "README.md",
        "artifact_manifest.json",
        "_SUCCESS.json",
        "hf_upload_receipt.json",
    }
    observed_tree_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    if observed_tree_paths != expected_tree_paths:
        raise ValueError(
            "legacy final tree inventory drifted: "
            f"missing={sorted(expected_tree_paths - observed_tree_paths)} "
            f"extra={sorted(observed_tree_paths - expected_tree_paths)}"
        )
    return {
        "success_sha256": success["success_sha256"],
        "receipt_sha256": receipt["receipt_sha256"],
        "legacy_hf_commit_sha": receipt["commit_sha"],
    }


def _atomic_rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically rename a directory while refusing an existing destination."""

    if destination.exists():
        raise FileExistsError(destination)
    if sys.platform.startswith("linux"):
        import ctypes
        import errno

        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise RuntimeError(
                "renameat2 is unavailable; refusing a non-atomic archive move"
            )
        renameat2.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renameat2.restype = ctypes.c_int
        at_fdcwd = -100
        rename_noreplace = 1
        result = renameat2(
            at_fdcwd,
            os.fsencode(source),
            at_fdcwd,
            os.fsencode(destination),
            rename_noreplace,
        )
        if result == 0:
            return
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise FileExistsError(destination)
        raise OSError(
            error_number,
            os.strerror(error_number),
            str(destination),
        )
    # This branch is used by local macOS tests only. The live Modal runtime is
    # Linux and therefore always uses renameat2(RENAME_NOREPLACE).
    os.rename(source, destination)


def _archive_legacy_final_for_rebuild() -> dict[str, Any]:
    """Move the authenticated legacy tree once; never replace either tree."""

    backup_exists = LEGACY_FINAL_BACKUP_ROOT.exists()
    canonical_exists = FINAL_ROOT.exists()
    if backup_exists:
        legacy = _validate_legacy_final_tree(LEGACY_FINAL_BACKUP_ROOT)
        if canonical_exists:
            try:
                canonical_success = _read_json(FINAL_ROOT / "_SUCCESS.json")
                _validated_final_artifacts(canonical_success)
            except Exception as exc:
                raise RuntimeError(
                    "rebuild conflict: authenticated legacy backup and a "
                    "non-canonical final tree both exist; refusing mutation"
                ) from exc
            return {
                "state": "canonical_already_rebuilt",
                "canonical": str(FINAL_ROOT),
                "legacy_backup": str(LEGACY_FINAL_BACKUP_ROOT),
                **legacy,
            }
        return {
            "state": "legacy_already_archived",
            "canonical": str(FINAL_ROOT),
            "legacy_backup": str(LEGACY_FINAL_BACKUP_ROOT),
            **legacy,
        }
    if not canonical_exists:
        raise FileNotFoundError(
            "rebuild requires the completed legacy final tree, but neither "
            f"{FINAL_ROOT} nor {LEGACY_FINAL_BACKUP_ROOT} exists"
        )

    try:
        legacy = _validate_legacy_final_tree(FINAL_ROOT)
    except Exception as exc:
        raise RuntimeError(
            "canonical final is not the exact completed legacy tree; "
            "refusing to archive or overwrite it"
        ) from exc
    _atomic_rename_noreplace(FINAL_ROOT, LEGACY_FINAL_BACKUP_ROOT)
    os.sync()
    moved = _validate_legacy_final_tree(LEGACY_FINAL_BACKUP_ROOT)
    if moved != legacy:
        raise RuntimeError("legacy final identity changed during atomic archive")
    return {
        "state": "legacy_archived",
        "canonical": str(FINAL_ROOT),
        "legacy_backup": str(LEGACY_FINAL_BACKUP_ROOT),
        **legacy,
    }


def _load_prepared_manifest() -> dict[str, Any]:
    path = INPUT_ROOT / "prepared_manifest.json"
    manifest = _read_json(path)
    if (
        manifest.get("schema") != "p1-full-rl-train-pass16-prepared-v1"
        or not _self_hashed(manifest, "manifest_sha256")
        or manifest.get("source", {}).get("sha256") != SOURCE_SHA256
        or manifest.get("source", {}).get("rows") != SOURCE_ROWS
        or manifest.get("contract", {}).get("samples_per_prompt")
        != SAMPLES_PER_PROMPT
        or manifest.get("contract", {}).get("expected_trajectories")
        != EXPECTED_TRAJECTORIES
        or manifest.get("prompt_lengths", {}).get("rl_eligible_rows")
        != 53_157
        or manifest.get("prompt_lengths", {}).get(
            "supplemental_long_rows"
        )
        != 68
        or manifest.get("prompt_lengths", {}).get("max") != 885
        or len(manifest.get("shards", [])) != SHARD_COUNT
    ):
        raise ValueError("prepared input manifest drifted")
    return manifest


def _checkpoint_identity() -> str:
    shared.pretrain_ckpt_vol.reload()
    observed = checkpoint_fingerprint(P1_CHECKPOINT)
    if observed != P1_CHECKPOINT_FINGERPRINT:
        raise ValueError(
            "P1 checkpoint fingerprint drifted: "
            f"{observed} != {P1_CHECKPOINT_FINGERPRINT}"
        )
    return observed


def _run_name(shard_id: int | None) -> str:
    return (
        f"{VERSION}-canary"
        if shard_id is None
        else f"{VERSION}-shard-{shard_id:02d}"
    )


def _node_spec(
    *,
    shard_id: int | None,
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = manifest or _load_prepared_manifest()
    if shard_id is None:
        canary = manifest["canary"]
        return {
            "kind": "canary",
            "shard_id": None,
            "run_name": _run_name(None),
            "artifact_tag": "canary",
            "relative_path": canary["relative_path"],
            "sha256": canary["sha256"],
            "rows": canary["rows"],
            "rollout_batch_size": canary["rows"],
            "num_rollout": 1,
            "source_row_start": None,
            "source_row_stop": None,
        }
    if not 0 <= shard_id < SHARD_COUNT:
        raise ValueError(f"invalid shard id: {shard_id}")
    shard = dict(manifest["shards"][shard_id])
    if shard.get("shard_id") != shard_id:
        raise ValueError("prepared shard order drifted")
    return {
        "kind": "production",
        "run_name": _run_name(shard_id),
        "artifact_tag": f"rollouts-shard-{shard_id:02d}",
        **shard,
    }


def build_rollout_command(spec: dict[str, Any]) -> list[str]:
    batch_size = int(spec["rollout_batch_size"])
    num_rollout = int(spec["num_rollout"])
    rows = int(spec["rows"])
    if batch_size * num_rollout != rows:
        raise ValueError("node spec is not exact-batch divisible")
    run_name = str(spec["run_name"])
    shard_path = INPUT_ROOT / str(spec["relative_path"])
    return [
        sys.executable,
        "-m",
        "chess_rl_miles.scripts.run_chess_miles",
        "--miles-dir",
        base.MILES_DIR,
        "--project-dir",
        base.PROJECT_DIR,
        "--hf-checkpoint",
        str(P1_CHECKPOINT),
        "--model-id",
        "interleave_47m_v1_p1_5b",
        "--small-model-profile",
        shared.POLICY_UPDATE_PROFILE,
        "--no-gradient-checkpointing",
        "--run-name",
        run_name,
        "--io-layout",
        "flat",
        "--data-dir",
        str(INPUT_ROOT),
        "--train-file",
        str(shard_path),
        "--train-file-sha256",
        str(spec["sha256"]),
        "--data-source-path",
        STRICT_SOURCE,
        "--save-dir",
        str(RUNS_ROOT),
        "--rollout-seed",
        str(BASE_SEED),
        "--num-rollout",
        str(num_rollout),
        "--save-interval",
        "0",
        "--rollout-batch-size",
        str(batch_size),
        "--n-samples-per-prompt",
        str(SAMPLES_PER_PROMPT),
        "--over-sampling-batch-size",
        str(batch_size),
        "--global-batch-size",
        str(batch_size * SAMPLES_PER_PROMPT),
        "--policy-loss-agg-mode",
        "token-mean",
        "--no-cispo",
        "--optim-tag",
        "adamw",
        "--lr",
        "1e-5",
        "--adam-beta1",
        "0.9",
        "--adam-beta2",
        "0.999",
        "--adam-eps",
        "1e-8",
        "--weight-decay",
        "0.01",
        "--kl-loss-coef",
        "0.001",
        "--rollout-max-prompt-len",
        str(EXHAUSTIVE_PROMPT_CAP),
        "--rollout-max-response-len",
        "2560",
        "--rollout-max-context-len",
        "3072",
        "--rollout-temperature",
        "1.0",
        "--rollout-top-p",
        "1.0",
        "--actor-num-nodes",
        "1",
        "--actor-num-gpus-per-node",
        str(base.GPUS_PER_NODE),
        "--rollout-num-gpus-per-engine",
        "1",
        "--sglang-server-concurrency",
        "128",
        "--eval-sglang-server-concurrency",
        "16",
        "--max-tokens-per-gpu",
        "131072",
        "--attn-implementation",
        "flash_attention_3",
        "--batched-rollout",
        "--sglang-token-id-only",
        "--use-miles-router",
        "--rollout-health-check-interval",
        str(ROUTER_HEALTH_INTERVAL_SECONDS),
        "--no-use-fault-tolerance",
        "--no-log-passrate",
        "--save-rollouts",
        "--wandb-project",
        "",
        "--reward-model-type",
        "RULE_BASED",
        "--debug-rollout-only",
        "--sglang-enable-deterministic-inference",
        "--chess-deterministic-seed-by-sample-index",
    ]


def _source_indices(shard_path: Path) -> set[int]:
    import pyarrow.parquet as pq

    table = pq.read_table(shard_path, columns=["extra_info"])
    indices = {
        int(value["source_row_index"])
        for value in table["extra_info"].to_pylist()
    }
    if len(indices) != table.num_rows:
        raise ValueError("input shard source_row_index is not unique")
    return indices


def _gzip_raw_rollouts(
    paths: list[Path],
    output_path: Path,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(
        f".{output_path.name}.{os.getpid()}.tmp"
    )
    with temporary.open("wb") as raw_output:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw_output,
            compresslevel=6,
            mtime=0,
        ) as compressed:
            for path in paths:
                with path.open("rb") as source:
                    shutil.copyfileobj(source, compressed, length=8 << 20)
        raw_output.flush()
        os.fsync(raw_output.fileno())
    os.replace(temporary, output_path)
    return {
        "path": str(output_path),
        "bytes": output_path.stat().st_size,
        "sha256": sha256_file(output_path),
    }


@app.function(
    gpu=f"{base.GPU_TYPE}:{base.GPUS_PER_NODE}",
    cpu=base.CPU_COUNT,
    memory=shared.SMALL_MODEL_HOST_MEMORY_MB,
    timeout=NODE_TIMEOUT_SECONDS,
    max_containers=SHARD_COUNT,
    volumes={
        str(RESULTS_MOUNT): results_vol,
        shared.PRETRAIN_CKPT_ROOT: shared.pretrain_ckpt_vol,
        base.HF_CACHE_DIR: base.hf_cache,
    },
)
def run_pass16_node(spec: dict[str, Any]) -> dict[str, Any]:
    started_at = time.time()
    manifest = _load_prepared_manifest()
    expected = _node_spec(
        shard_id=spec.get("shard_id"),
        manifest=manifest,
    )
    if spec != expected:
        raise ValueError("submitted node spec drifted from prepared manifest")
    checkpoint_sha256 = _checkpoint_identity()
    shard_path = INPUT_ROOT / str(spec["relative_path"])
    if (
        not shard_path.is_file()
        or sha256_file(shard_path) != spec["sha256"]
    ):
        raise ValueError("input shard SHA256 drifted")
    source_indices = _source_indices(shard_path)
    if len(source_indices) != int(spec["rows"]):
        raise ValueError("input shard row count drifted")

    run_root = RUNS_ROOT / str(spec["run_name"])
    success_path = run_root / "_SUCCESS.json"
    if success_path.is_file():
        success = _read_json(success_path)
        if _self_hashed(success, "success_sha256"):
            return success
        raise ValueError("existing node success marker is invalid")
    if run_root.exists():
        raise FileExistsError(
            f"incomplete node root already exists; inspect before retry: {run_root}"
        )
    run_root.mkdir(parents=True, exist_ok=False)
    command = build_rollout_command(spec)
    intent = {
        "schema": "p1-full-rl-train-pass16-node-intent-v1",
        "version": VERSION,
        "spec": spec,
        "checkpoint_path": str(P1_CHECKPOINT),
        "checkpoint_fingerprint": checkpoint_sha256,
        "prepared_manifest_sha256": manifest["manifest_sha256"],
        "command": command,
        "command_sha256": canonical_sha256(command),
    }
    intent["intent_sha256"] = canonical_sha256(intent)
    atomic_json(run_root / "_INTENT.json", intent)
    os.sync()

    env = shared._runtime_env(
        run_name=str(spec["run_name"]),
        deterministic_seed_mode="sample-index",
    )
    env["CHESS_RL_MILES_ARTIFACT_ROOT"] = str(run_root)
    env["WANDB_MODE"] = "disabled"
    env["WANDB_DISABLED"] = "true"
    env.pop("WANDB_API_KEY", None)
    base._cleanup_runtime()
    try:
        base._start_ray_head(env, cpu_threads=int(base.CPU_COUNT))
        env["RAY_ADDRESS"] = base.RAY_ADDRESS
        result = subprocess.run(
            command,
            env=env,
            cwd=base.PROJECT_DIR,
            timeout=NODE_TIMEOUT_SECONDS - 15 * 60,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"pass@16 node timed out: {spec['run_name']}"
        ) from exc
    finally:
        base._cleanup_runtime()
    if result.returncode:
        raise RuntimeError(
            f"pass@16 node failed: {spec['run_name']} exit={result.returncode}"
        )

    raw_root = run_root / "rollouts" / "training"
    raw_paths = [
        raw_root / f"rollout_{rollout_id}.jsonl"
        for rollout_id in range(int(spec["num_rollout"]))
    ]
    missing = [str(path) for path in raw_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "missing raw rollout files: " + ", ".join(missing)
        )
    metrics = validate_rollout_rows(
        iter_jsonl(raw_paths),
        expected_source_indices=source_indices,
    )
    raw_records = [
        {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in raw_paths
    ]
    compressed = _gzip_raw_rollouts(
        raw_paths,
        ARTIFACTS_ROOT
        / "data"
        / f"{spec['artifact_tag']}.jsonl.gz",
    )
    success_core = {
        "schema": "p1-full-rl-train-pass16-node-success-v1",
        "version": VERSION,
        "spec": spec,
        "checkpoint_fingerprint": checkpoint_sha256,
        "prepared_manifest_sha256": manifest["manifest_sha256"],
        "command_sha256": intent["command_sha256"],
        "expected_rows": int(spec["rows"]) * SAMPLES_PER_PROMPT,
        "metrics": metrics,
        "raw_files": raw_records,
        "compressed_raw": compressed,
        "started_unix": started_at,
        "finished_unix": time.time(),
        "duration_seconds": round(time.time() - started_at, 3),
    }
    success = {
        **success_core,
        "success_sha256": canonical_sha256(success_core),
    }
    atomic_json(success_path, success)
    os.sync()
    return success


def _cohort_metrics(
    records: list[dict[str, Any]],
    *,
    cohort: str,
) -> dict[str, Any]:
    prompts = len(records)
    trajectories = prompts * SAMPLES_PER_PROMPT
    positives = sum(int(record["success_count"]) for record in records)
    solved = sum(int(record["success_count"] > 0) for record in records)
    histogram = Counter(int(record["success_count"]) for record in records)
    pass1 = positives / trajectories if trajectories else None
    pass16 = solved / prompts if prompts else None
    pass1_ci = wilson_interval(positives, trajectories)
    pass16_ci = wilson_interval(solved, prompts)
    return {
        "cohort": cohort,
        "prompts": prompts,
        "trajectories": trajectories,
        "positive_trajectories": positives,
        "solved_prompts": solved,
        "pass_at_1": pass1,
        "avg_reward": pass1,
        "pass_at_16": pass16,
        "solve_at_16": pass16,
        "pass_at_1_wilson_95": list(pass1_ci),
        "pass_at_16_wilson_95": list(pass16_ci),
        "all_zero_prompts": histogram.get(0, 0),
        "all_one_prompts": histogram.get(SAMPLES_PER_PROMPT, 0),
        "nonzero_variance_prompts": sum(
            histogram.get(count, 0)
            for count in range(1, SAMPLES_PER_PROMPT)
        ),
        "success_count_histogram": {
            str(count): histogram.get(count, 0)
            for count in range(SAMPLES_PER_PROMPT + 1)
        },
    }


def _finalizer_reward_args() -> SimpleNamespace:
    """Return the exact reward-mode contract used by production rollouts."""

    return SimpleNamespace(
        chess_reward_model_type="RULE_BASED",
        # The production rollout command always uses the chess multi-turn
        # contract, including trajectories that make zero environment calls.
        # Inferring from an empty ``env_replies`` list would incorrectly
        # rescore those rows with the single-turn parser.
        chess_multiturn=True,
    )


@app.function(
    cpu=1.0,
    memory=4 * 1024,
    timeout=10 * 60,
    max_containers=1,
    volumes={str(RESULTS_MOUNT): results_vol},
)
def archive_legacy_final_for_rebuild() -> dict[str, Any]:
    return _archive_legacy_final_for_rebuild()


@app.function(
    cpu=16.0,
    memory=64 * 1024,
    timeout=2 * 60 * 60,
    volumes={str(RESULTS_MOUNT): results_vol},
)
def finalize_results() -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from miles.utils.types import Sample

    from chess_rl_miles.reward import _score_sample

    final_success_path = FINAL_ROOT / "_SUCCESS.json"
    if final_success_path.is_file():
        existing = _read_json(final_success_path)
        _validated_final_artifacts(existing)
        return existing
    if FINAL_ROOT.exists():
        raise FileExistsError(
            f"incomplete final root already exists; inspect before retry: {FINAL_ROOT}"
        )
    manifest = _load_prepared_manifest()
    successes: list[dict[str, Any]] = []
    for shard_id in range(SHARD_COUNT):
        success_path = RUNS_ROOT / _run_name(shard_id) / "_SUCCESS.json"
        success = _read_json(success_path)
        if (
            not _self_hashed(success, "success_sha256")
            or success.get("spec") != _node_spec(
                shard_id=shard_id,
                manifest=manifest,
            )
        ):
            raise ValueError(f"shard {shard_id} success authentication failed")
        successes.append(success)

    prompt_records: dict[int, dict[str, Any]] = {}
    seen_samples: set[int] = set()
    status_counts: Counter[str] = Counter()
    preview_rows: list[dict[str, Any]] = []
    reward_args = _finalizer_reward_args()
    for success in successes:
        compressed = success["compressed_raw"]
        compressed_path = Path(compressed["path"])
        if (
            not compressed_path.is_file()
            or compressed_path.stat().st_size != compressed["bytes"]
            or sha256_file(compressed_path) != compressed["sha256"]
        ):
            raise ValueError("compressed raw rollout artifact drifted")
        with gzip.open(compressed_path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                metadata = row.get("metadata")
                if not isinstance(metadata, dict):
                    raise ValueError("aggregate row lacks metadata")
                source_index = metadata.get("source_row_index")
                slot = metadata.get("pass_at_16_sample_slot")
                sample_index = metadata.get("pass_at_16_sample_index")
                if (
                    not isinstance(source_index, int)
                    or isinstance(source_index, bool)
                    or not 0 <= source_index < SOURCE_ROWS
                    or not isinstance(slot, int)
                    or isinstance(slot, bool)
                    or not 0 <= slot < SAMPLES_PER_PROMPT
                    or sample_index
                    != source_index * SAMPLES_PER_PROMPT + slot
                    or row.get("group_index") != source_index
                    or row.get("sample_index") != sample_index
                    or metadata.get("sampling_seed")
                    != BASE_SEED + sample_index
                    or metadata.get("sampling_seed_mode") != "sample-index"
                ):
                    raise ValueError("aggregate rollout identity/seed drifted")
                if sample_index in seen_samples:
                    raise ValueError("duplicate global trajectory identity")
                seen_samples.add(sample_index)
                score = row.get("score")
                if isinstance(score, bool) or score not in (0, 0.0, 1, 1.0):
                    raise ValueError("aggregate score is not binary")

                sample = Sample(
                    group_index=source_index,
                    index=sample_index,
                    prompt=row.get("input") or "",
                    response=row.get("output") or "",
                    label=row.get("label"),
                    metadata=metadata,
                )
                independently_scored = _score_sample(
                    reward_args,
                    sample,
                )
                if independently_scored.get("score") != score:
                    raise ValueError(
                        f"independent reward mismatch at sample {sample_index}"
                    )

                record = prompt_records.setdefault(
                    source_index,
                    {
                        "source_row_index": source_index,
                        "source_row_fingerprint": metadata.get(
                            "source_row_fingerprint"
                        ),
                        "PuzzleId": metadata.get("PuzzleId"),
                        "FEN": metadata.get("FEN"),
                        "difficulty": metadata.get("source_difficulty"),
                        "prompt_token_length": metadata.get(
                            "prompt_token_length"
                        ),
                        "rl_prompt_cap_eligible": metadata.get(
                            "rl_prompt_cap_eligible"
                        ),
                        "success_count": 0,
                        "slot_mask": 0,
                        "completed_trajectories": 0,
                        "truncated_trajectories": 0,
                        "response_token_sum": 0,
                    },
                )
                bit = 1 << slot
                if int(record["slot_mask"]) & bit:
                    raise ValueError("duplicate sibling slot in prompt group")
                record["slot_mask"] = int(record["slot_mask"]) | bit
                record["success_count"] = (
                    int(record["success_count"]) + int(float(score) == 1.0)
                )
                status = str(row.get("status"))
                if status not in {"completed", "truncated"}:
                    raise ValueError(f"invalid terminal status: {status}")
                status_counts[status] += 1
                record[f"{status}_trajectories"] = (
                    int(record[f"{status}_trajectories"]) + 1
                )
                record["response_token_sum"] = int(
                    record["response_token_sum"]
                ) + int(row.get("response_length") or 0)
                if len(preview_rows) < 64:
                    preview_rows.append(row)

    if len(seen_samples) != EXPECTED_TRAJECTORIES:
        raise ValueError(
            "global trajectory count mismatch: "
            f"{len(seen_samples)}/{EXPECTED_TRAJECTORIES}"
        )
    if set(prompt_records) != set(range(SOURCE_ROWS)):
        raise ValueError("global prompt coverage mismatch")
    full_mask = (1 << SAMPLES_PER_PROMPT) - 1
    records = [prompt_records[index] for index in range(SOURCE_ROWS)]
    for record in records:
        if record.pop("slot_mask") != full_mask:
            raise ValueError("prompt lacks exact sibling coverage")
        record["pass_at_16"] = float(record["success_count"] > 0)
        record["avg_reward"] = (
            record["success_count"] / SAMPLES_PER_PROMPT
        )
        record["mean_response_tokens"] = (
            record.pop("response_token_sum") / SAMPLES_PER_PROMPT
        )

    eligible = [
        record for record in records if record["rl_prompt_cap_eligible"] is True
    ]
    supplemental = [
        record for record in records if record["rl_prompt_cap_eligible"] is False
    ]
    if len(eligible) != 53_157 or len(supplemental) != 68:
        raise ValueError("RL prompt-cap cohort split drifted")
    summary = {
        "schema": "p1-full-rl-train-pass16-summary-v1",
        "version": VERSION,
        "model": {
            "modal_checkpoint": str(P1_CHECKPOINT),
            "checkpoint_fingerprint": P1_CHECKPOINT_FINGERPRINT,
            "hf_repo": P1_HF_REPO,
            "hf_revision": P1_HF_REVISION,
            "model_weights_sha256": P1_WEIGHTS_SHA256,
        },
        "dataset": {
            "rows": SOURCE_ROWS,
            "sha256": SOURCE_SHA256,
            "rl_eligible_rows": len(eligible),
            "supplemental_long_rows": len(supplemental),
        },
        "generation": {
            "backend": "Miles/SGLang batched multi-turn rollout",
            "policy_updates": False,
            "dynamic_filter": False,
            "samples_per_prompt": SAMPLES_PER_PROMPT,
            "temperature": 1.0,
            "top_p": 1.0,
            "prompt_cap": EXHAUSTIVE_PROMPT_CAP,
            "response_cap": 2_560,
            "context_cap": 3_072,
            "base_seed": BASE_SEED,
            "gpu_topology": "16 nodes x 8 H200 = 128 H200",
        },
        "metrics": {
            "full_parquet": _cohort_metrics(
                records,
                cohort="full_parquet",
            ),
            "rl_eligible_le_512": _cohort_metrics(
                eligible,
                cohort="rl_eligible_le_512",
            ),
            "supplemental_long_gt_512": _cohort_metrics(
                supplemental,
                cohort="supplemental_long_gt_512",
            ),
        },
        "status_counts": dict(sorted(status_counts.items())),
        "completeness": {
            "expected_prompts": SOURCE_ROWS,
            "actual_prompts": len(records),
            "expected_trajectories": EXPECTED_TRAJECTORIES,
            "actual_trajectories": len(seen_samples),
            "exact_slots_0_through_15": True,
            "independent_rewards_recomputed": True,
        },
    }
    summary["summary_sha256"] = canonical_sha256(summary)

    FINAL_ROOT.parent.mkdir(parents=True, exist_ok=True)
    work_root = FINAL_ROOT.with_name(
        f".{FINAL_ROOT.name}.tmp-{os.getpid()}-{time.time_ns()}"
    )
    work_root.mkdir(parents=False, exist_ok=False)
    per_prompt_path = work_root / "per_prompt.parquet"
    pq.write_table(
        pa.Table.from_pylist(records),
        per_prompt_path,
        compression="zstd",
        use_dictionary=True,
    )
    summary_path = work_root / "summary.json"
    atomic_json(summary_path, summary)
    preview_path = work_root / "preview.jsonl"
    with preview_path.open("w", encoding="utf-8") as handle:
        for row in preview_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    contract = {
        "schema": "p1-full-rl-train-pass16-contract-v1",
        "version": VERSION,
        "prepared_manifest_sha256": manifest["manifest_sha256"],
        "source_sha256": SOURCE_SHA256,
        "source_rows": SOURCE_ROWS,
        "checkpoint_fingerprint": P1_CHECKPOINT_FINGERPRINT,
        "hf_model": f"{P1_HF_REPO}@{P1_HF_REVISION}",
        "policy_updates": False,
        "dynamic_filter": False,
        "raw_rollouts_saved": True,
        "settings": summary["generation"],
    }
    contract["contract_sha256"] = canonical_sha256(contract)
    contract_path = work_root / "contract.json"
    atomic_json(contract_path, contract)
    prepared_copy = work_root / "prepared_manifest.json"
    shutil.copy2(INPUT_ROOT / "prepared_manifest.json", prepared_copy)
    success_copies: list[Path] = []
    shard_success_root = work_root / "shard_success"
    shard_success_root.mkdir()
    for shard_id in range(SHARD_COUNT):
        target = shard_success_root / f"shard_{shard_id:02d}.json"
        shutil.copy2(
            RUNS_ROOT / _run_name(shard_id) / "_SUCCESS.json",
            target,
        )
        success_copies.append(target)

    readme = f"""---
pretty_name: P1 5B Full RL Train Pass@16 Rollouts
task_categories:
- text-generation
---

# P1 5B full RL-training-set pass@16

This dataset contains all **{EXPECTED_TRAJECTORIES:,}** raw rollout-only
trajectories generated from all **{SOURCE_ROWS:,}** rows of the balanced chess
RL training parquet (16 samples per prompt). No policy update and no dynamic
filter were used.

Model: [`{P1_HF_REPO}`](https://huggingface.co/{P1_HF_REPO}) pinned at
`{P1_HF_REVISION}`.

The primary full-file pass@16 is
**{summary['metrics']['full_parquet']['pass_at_16']:.8f}**
({summary['metrics']['full_parquet']['solved_prompts']:,}/
{SOURCE_ROWS:,} prompts). The production RL loader's 512-token cohort has
53,157 prompts; the remaining 68 longer prompts are included and reported as
a separate context-limited supplement.

Files:

- `data/*.jsonl.gz`: lossless raw rollouts, including prompt, output, binary
  reward, parsed moves, status, source-row identity, sibling slot, and seed.
- `per_prompt.parquet`: one row per source prompt with success count and
  pass@16.
- `summary.json`: full, RL-eligible, and long-prompt cohort metrics.
- `contract.json`, `prepared_manifest.json`, and `shard_success/`: exact
  identities and completeness proof.
- `preview.jsonl`: small human-readable raw sample.
"""
    readme_path = work_root / "README.md"
    with readme_path.open("w", encoding="utf-8") as handle:
        handle.write(readme)
        handle.flush()
        os.fsync(handle.fileno())

    files = [
        readme_path,
        per_prompt_path,
        summary_path,
        preview_path,
        contract_path,
        prepared_copy,
        *success_copies,
    ]
    artifact_records = [
        {
            "path": path.relative_to(work_root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]
    for shard_id, success in enumerate(successes):
        compressed = success["compressed_raw"]
        artifact_records.append(
            {
                "path": f"data/rollouts-shard-{shard_id:02d}.jsonl.gz",
                "volume_path": compressed["path"],
                "bytes": compressed["bytes"],
                "sha256": compressed["sha256"],
            }
        )
    artifact_manifest = {
        "schema": "p1-full-rl-train-pass16-artifact-manifest-v1",
        "version": VERSION,
        "files": artifact_records,
    }
    artifact_manifest["manifest_sha256"] = canonical_sha256(
        artifact_manifest
    )
    artifact_manifest_path = work_root / "artifact_manifest.json"
    atomic_json(artifact_manifest_path, artifact_manifest)
    artifact_manifest_file = {
        "bytes": artifact_manifest_path.stat().st_size,
        "sha256": sha256_file(artifact_manifest_path),
    }
    final_core = {
        "schema": "p1-full-rl-train-pass16-final-success-v1",
        "version": VERSION,
        "summary": summary,
        "artifact_manifest": str(FINAL_ROOT / "artifact_manifest.json"),
        "artifact_manifest_file": artifact_manifest_file,
        "artifact_manifest_sha256": artifact_manifest["manifest_sha256"],
        "per_prompt": {
            "path": str(FINAL_ROOT / "per_prompt.parquet"),
            "rows": SOURCE_ROWS,
            "bytes": per_prompt_path.stat().st_size,
            "sha256": sha256_file(per_prompt_path),
        },
    }
    final_success = {
        **final_core,
        "success_sha256": canonical_sha256(final_core),
    }
    atomic_json(work_root / "_SUCCESS.json", final_success)
    os.sync()
    os.replace(work_root, FINAL_ROOT)
    os.sync()
    _validated_final_artifacts(final_success)
    return final_success


@app.function(
    cpu=8.0,
    memory=32 * 1024,
    timeout=2 * 60 * 60,
    volumes={str(RESULTS_MOUNT): results_vol},
)
def upload_results_to_hf() -> dict[str, Any]:
    from huggingface_hub import HfApi, hf_hub_download

    success = _read_json(FINAL_ROOT / "_SUCCESS.json")
    artifact_records, manifest_file = _validated_final_artifacts(success)
    local_records = sorted(
        [
            {
                "path": record["path"],
                "bytes": record["bytes"],
                "sha256": record["sha256"],
            }
            for record in artifact_records
        ]
        + [
            {
                "path": "artifact_manifest.json",
                "bytes": manifest_file["bytes"],
                "sha256": manifest_file["sha256"],
            }
        ],
        key=lambda record: record["path"],
    )
    api = HfApi()
    receipt_path = FINAL_ROOT / "hf_upload_receipt.json"
    if receipt_path.is_file():
        receipt = _read_json(receipt_path)
        if (
            not _self_hashed(receipt, "receipt_sha256")
            or receipt.get("repo_id") != HF_DATASET_REPO
            or receipt.get("files") != local_records
            or not isinstance(receipt.get("commit_sha"), str)
        ):
            raise ValueError("existing HF receipt is invalid")
        _verify_hf_commit(
            api,
            commit_sha=receipt["commit_sha"],
            records=local_records,
            download_file=hf_hub_download,
        )
        return receipt

    with tempfile.TemporaryDirectory(prefix="p1-pass16-hf-") as temporary:
        staging = Path(temporary)
        for record in artifact_records:
            relative = record["path"]
            source = (
                Path(record["volume_path"])
                if "volume_path" in record
                else FINAL_ROOT / relative
            )
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            _validate_record_identity(
                target,
                record,
                label=f"HF staging artifact {relative}",
            )
        manifest_target = staging / "artifact_manifest.json"
        shutil.copy2(FINAL_ROOT / "artifact_manifest.json", manifest_target)
        _validate_record_identity(
            manifest_target,
            manifest_file,
            label="HF staging artifact manifest",
        )
        staged_paths = {
            path.relative_to(staging).as_posix()
            for path in staging.rglob("*")
            if path.is_file()
        }
        if staged_paths != {record["path"] for record in local_records}:
            raise ValueError("HF staging inventory drifted")

        api.create_repo(
            HF_DATASET_REPO,
            repo_type="dataset",
            private=False,
            exist_ok=True,
        )
        existing_info = api.repo_info(
            HF_DATASET_REPO,
            repo_type="dataset",
        )
        existing_paths = {
            sibling.rfilename for sibling in existing_info.siblings
        }
        unexpected_existing = existing_paths - staged_paths - {
            ".gitattributes"
        }
        if unexpected_existing:
            raise ValueError(
                "HF target contains unexpected pre-existing files: "
                f"{sorted(unexpected_existing)}"
            )
        commit_sha = ""
        existing_commit_sha = str(getattr(existing_info, "sha", "") or "")
        if (
            existing_commit_sha
            and existing_paths
            == staged_paths | {".gitattributes"}
        ):
            try:
                _verify_hf_commit(
                    api,
                    commit_sha=existing_commit_sha,
                    records=local_records,
                    download_file=hf_hub_download,
                )
            except ValueError:
                pass
            else:
                commit_sha = existing_commit_sha
        if not commit_sha:
            commit = api.upload_folder(
                folder_path=str(staging),
                repo_id=HF_DATASET_REPO,
                repo_type="dataset",
                commit_message=(
                    f"Upload exhaustive P1 full-train pass@16 ({VERSION})"
                ),
            )
            commit_sha = str(commit.oid)

    _verify_hf_commit(
        api,
        commit_sha=commit_sha,
        records=local_records,
        download_file=hf_hub_download,
    )
    receipt_core = {
        "schema": "p1-full-rl-train-pass16-hf-receipt-v1",
        "version": VERSION,
        "repo_id": HF_DATASET_REPO,
        "repo_type": "dataset",
        "url": f"https://huggingface.co/datasets/{HF_DATASET_REPO}",
        "commit_sha": commit_sha,
        "files": local_records,
        "all_remote_bytes_and_sha256_verified": True,
    }
    receipt = {
        **receipt_core,
        "receipt_sha256": canonical_sha256(receipt_core),
    }
    atomic_json(receipt_path, receipt)
    os.sync()
    return receipt


@app.function(
    cpu=1.0,
    memory=1024,
    timeout=5 * 60,
    volumes={str(RESULTS_MOUNT): results_vol},
)
def remote_status() -> dict[str, Any]:
    def marker(path: Path) -> dict[str, Any] | None:
        if not path.is_file():
            return None
        value = _read_json(path)
        return {
            "path": str(path),
            "valid": (
                _self_hashed(value, "success_sha256")
                or _self_hashed(value, "receipt_sha256")
            ),
            "duration_seconds": value.get("duration_seconds"),
            "metrics": value.get("metrics"),
        }

    shards = [
        marker(RUNS_ROOT / _run_name(shard_id) / "_SUCCESS.json")
        for shard_id in range(SHARD_COUNT)
    ]
    return {
        "version": VERSION,
        "volume": RESULTS_VOLUME_NAME,
        "prepared": (INPUT_ROOT / "prepared_manifest.json").is_file(),
        "canary": marker(
            RUNS_ROOT / _run_name(None) / "_SUCCESS.json"
        ),
        "shards_complete": sum(item is not None for item in shards),
        "shards": shards,
        "final": marker(FINAL_ROOT / "_SUCCESS.json"),
        "legacy_final_backup": marker(
            LEGACY_FINAL_BACKUP_ROOT / "_SUCCESS.json"
        ),
        "hf_upload": marker(FINAL_ROOT / "hf_upload_receipt.json"),
    }


def _write_ledger(path: Path, value: dict[str, Any]) -> None:
    core = {key: item for key, item in value.items() if key != "ledger_sha256"}
    value["ledger_sha256"] = canonical_sha256(core)
    atomic_json(path, value)


@app.local_entrypoint()
def main(
    action: str,
    local_source: str = LOCAL_SOURCE_DEFAULT,
    local_tokenizer: str = LOCAL_TOKENIZER_DEFAULT,
    local_prepared: str = LOCAL_PREPARED_DEFAULT,
    ledger_path: str = LOCAL_LEDGER_DEFAULT,
) -> None:
    if action == "prepare":
        result = prepare_shards(
            source_path=local_source,
            tokenizer_path=local_tokenizer,
            output_root=local_prepared,
            tokenizer_revision=P1_HF_REVISION,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if action == "upload-input":
        local_root = Path(local_prepared).resolve(strict=True)
        with results_vol.batch_upload(force=False) as batch:
            batch.put_directory(str(local_root), "/input")
        print(
            json.dumps(
                {
                    "uploaded": str(local_root),
                    "volume": RESULTS_VOLUME_NAME,
                    "remote": str(INPUT_ROOT),
                },
                sort_keys=True,
            )
        )
        return
    if action == "status":
        print(json.dumps(remote_status.remote(), indent=2, sort_keys=True))
        return
    if action == "rebuild-final":
        archive = archive_legacy_final_for_rebuild.remote()
        final = finalize_results.remote()
        receipt = upload_results_to_hf.remote()
        print(
            json.dumps(
                {
                    "archive": archive,
                    "final_success_sha256": final["success_sha256"],
                    "hf_upload": receipt,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if action == "finalize":
        print(json.dumps(finalize_results.remote(), indent=2, sort_keys=True))
        return
    if action == "upload-hf":
        print(
            json.dumps(
                upload_results_to_hf.remote(),
                indent=2,
                sort_keys=True,
            )
        )
        return
    if action != "launch":
        raise ValueError(
            "action must be prepare, upload-input, launch, status, "
            "rebuild-final, finalize, or upload-hf"
        )

    status = remote_status.remote()
    if not status["prepared"]:
        raise FileNotFoundError("prepared inputs are not on the results volume")
    if status["shards_complete"]:
        raise FileExistsError("one or more production shard roots already exist")
    ledger = Path(ledger_path).resolve()
    if ledger.exists():
        raise FileExistsError(ledger)
    launch: dict[str, Any] = {
        "schema": "p1-full-rl-train-pass16-launch-ledger-v1",
        "version": VERSION,
        "state": "canary_launching",
        "canary_call_id": None,
        "shard_calls": [],
    }
    _write_ledger(ledger, launch)
    local_manifest = _read_json(
        Path(local_prepared).resolve(strict=True)
        / "prepared_manifest.json"
    )
    if not _self_hashed(local_manifest, "manifest_sha256"):
        raise ValueError("local prepared manifest is not authenticated")

    if status["canary"] is None:
        canary_spec = _node_spec(
            shard_id=None,
            manifest=local_manifest,
        )
        canary = run_pass16_node.spawn(canary_spec)
        launch["canary_call_id"] = canary.object_id
        launch["state"] = "canary_running"
        _write_ledger(ledger, launch)
        print(f"SPAWNED canary {canary.object_id}", flush=True)
        canary_success = canary.get()
    else:
        canary_success = status["canary"]
    launch["state"] = "canary_succeeded"
    launch["canary_success"] = canary_success
    _write_ledger(ledger, launch)

    handles = []
    for shard_id in range(SHARD_COUNT):
        spec = _node_spec(
            shard_id=shard_id,
            manifest=local_manifest,
        )
        call = run_pass16_node.spawn(spec)
        handles.append((shard_id, call))
        launch["shard_calls"].append(
            {
                "shard_id": shard_id,
                "function_call_id": call.object_id,
            }
        )
        print(
            f"SPAWNED shard={shard_id:02d} call={call.object_id}",
            flush=True,
        )
    launch["state"] = "shards_running"
    _write_ledger(ledger, launch)
    failures: list[str] = []
    successes: list[dict[str, Any]] = []
    for shard_id, call in handles:
        try:
            success = call.get()
        except Exception as exc:
            failures.append(f"shard {shard_id:02d}: {type(exc).__name__}: {exc}")
            continue
        successes.append(success)
        print(f"COMPLETE shard={shard_id:02d}", flush=True)
    if failures:
        launch["state"] = "shards_failed"
        launch["failures"] = failures
        launch["completed_shards"] = len(successes)
        _write_ledger(ledger, launch)
        raise RuntimeError("; ".join(failures))

    launch["state"] = "finalizing"
    _write_ledger(ledger, launch)
    final = finalize_results.remote()
    launch["final_success_sha256"] = final["success_sha256"]
    launch["state"] = "uploading_hf"
    _write_ledger(ledger, launch)
    receipt = upload_results_to_hf.remote()
    launch["hf_receipt"] = receipt
    launch["state"] = "complete"
    _write_ledger(ledger, launch)
    print(json.dumps(receipt, indent=2, sort_keys=True))
