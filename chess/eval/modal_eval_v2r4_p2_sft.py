"""Exact P2-heldout-at-P1 masked-SFT evaluation for the v2r4 gate.

This is deliberately a separate Modal app from the endpoint PT/chess
evaluator.  It has one GPU function with a frozen three-candidate contract:

    modal run modal_eval_v2r4_p2_sft.py --mode dry-run
    modal run --detach modal_eval_v2r4_p2_sft.py --mode launch
    modal run modal_eval_v2r4_p2_sft.py --mode status

The worker authenticates every source byte, rebuilds the frozen 4,096-row P2
selection and its response-mask shape, authenticates the complete recursive
HF snapshot and P1-only training cursor, and only then loads the model.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import modal

try:
    import v2r4_p2_sft_eval as p2_contract
except ImportError:
    from chess_reasoning.training import v2r4_p2_sft_eval as p2_contract


HERE = Path(__file__).resolve().parent
RUNNER_LOCAL_PATH = Path(__file__).resolve()
RUNNER_REMOTE_PATH = Path("/root/modal_eval_v2r4_p2_sft.py")
CONTRACT_REMOTE_PATH = Path("/root/v2r4_p2_sft_eval.py")
_WORKSPACE_CONTRACT_PATH = (
    HERE.parent / "chess_reasoning/training/v2r4_p2_sft_eval.py"
)
CONTRACT_LOCAL_PATH = (
    _WORKSPACE_CONTRACT_PATH
    if _WORKSPACE_CONTRACT_PATH.is_file()
    else CONTRACT_REMOTE_PATH
)

APP_NAME = "chess-interleave-v2r4-p2-sft-eval-v2"
FUNCTION_NAME = "evaluate_p2_sft_candidate"
CONTRACT_VERSION = "v2r4_p2_sft_at_p1_20260730_v2"
OUTPUT_SCHEMA = "interleaved-v2r4-p2-sft-modal-result-v2"
RESULT_NAMESPACE = "v2r4_p2_sft_at_p1_20260730_v2"
DEFAULT_LAUNCH_LEDGER_PATH = (
    "INTERLEAVED_V2R4_P2_SFT_V2_LAUNCH_LEDGER.json"
)

DATA_VOLUME_NAME = "rl-reasoning-training-data"
CHECKPOINT_VOLUME_NAME = "rl-reasoning-checkpoints"
RESULTS_VOLUME_NAME = "chess-rl-eval-results-r6"
DATA_MOUNT = Path("/data")
CHECKPOINT_MOUNT = Path("/pretrain-checkpoints")
RESULTS_MOUNT = Path("/results")
DATA_ROOT = (
    DATA_MOUNT
    / "50m_interleaved_mix10b_sft90k_v2r1_clean_verify_gate"
)
RESULTS_ROOT = RESULTS_MOUNT / RESULT_NAMESPACE

MANIFEST_SET_PATH = DATA_ROOT / "manifest_set.json"
SFT_CACHE_METADATA_PATH = DATA_ROOT / "sft_cache/metadata.json"
SFT_INPUT_IDS_PATH = DATA_ROOT / "sft_cache/input_ids.i32"
SFT_LABELS_PATH = DATA_ROOT / "sft_cache/labels.i32"
SFT_OFFSETS_PATH = DATA_ROOT / "sft_cache/offsets.npy"
P1_METADATA_PATH = DATA_ROOT / "legs/p1/metadata.json"
P2_METADATA_PATH = DATA_ROOT / "legs/p2/metadata.json"
P1_ORDER_PATH = DATA_ROOT / "legs/p1/order.npy"
P2_ORDER_PATH = DATA_ROOT / "legs/p2/order.npy"

SELECTION_HASH = (
    "99d20a1ee7dad9ab88ab5de2dfe0df50cc9d9e076636cf41252fbb1db2ea371e"
)
CACHE_SHAPE_HASH = (
    "6b8b068a1d02480d9c0a9933c19a534bb64eb15fe16e9ae7a313f4ea66c4d5c5"
)
SELECTED_CODES_SHA256 = (
    "6c7d3714543f0fbc9c97061cc8c129a63af8fcccfc74d1a9543db1427a8c24aa"
)
SELECTED_LABEL_PAYLOAD_SHA256 = (
    "5ed3524fd47012774b7e2f858c5fdddf6be1e1c588c666a78779142a8d3be581"
)
EXPECTED_SELECTED_SHAPE = {
    "num_records": 4_096,
    "total_aligned_positions": 3_560_000,
    "supervised_targets": 2_759_776,
    "ignored_positions": 800_224,
    "aligned_positions_per_row_min": 160,
    "aligned_positions_per_row_max": 1_645,
    "supervised_targets_per_row_min": 37,
    "supervised_targets_per_row_max": 1_448,
    "selected_label_payload_sha256": SELECTED_LABEL_PAYLOAD_SHA256,
}

SFT_ARTIFACT_SHA256 = {
    "manifest_set.json": p2_contract.MANIFEST_SET_FILE_SHA256,
    "sft_cache/metadata.json": (
        p2_contract.SFT_CACHE_METADATA_FILE_SHA256
    ),
    "sft_cache/input_ids.i32": p2_contract.SFT_INPUT_IDS_SHA256,
    "sft_cache/labels.i32": p2_contract.SFT_LABELS_SHA256,
    "sft_cache/offsets.npy": p2_contract.SFT_OFFSETS_SHA256,
    "legs/p1/metadata.json": p2_contract.P1_METADATA_FILE_SHA256,
    "legs/p1/order.npy": p2_contract.P1_ORDER_SHA256,
    "legs/p2/metadata.json": p2_contract.P2_METADATA_FILE_SHA256,
    "legs/p2/order.npy": p2_contract.P2_ORDER_SHA256,
}
SFT_ARTIFACT_PATHS = {
    "manifest_set.json": MANIFEST_SET_PATH,
    "sft_cache/metadata.json": SFT_CACHE_METADATA_PATH,
    "sft_cache/input_ids.i32": SFT_INPUT_IDS_PATH,
    "sft_cache/labels.i32": SFT_LABELS_PATH,
    "sft_cache/offsets.npy": SFT_OFFSETS_PATH,
    "legs/p1/metadata.json": P1_METADATA_PATH,
    "legs/p1/order.npy": P1_ORDER_PATH,
    "legs/p2/metadata.json": P2_METADATA_PATH,
    "legs/p2/order.npy": P2_ORDER_PATH,
}

SNAPSHOT_ROOT = (
    CHECKPOINT_MOUNT
    / "interleave_50m/pretrain"
    / "mix10b_sft90k_3072_v2r3_diagnostic_20260730"
    / "p1_w4067c60eaba84b1e/snapshots"
)
SFT_LOSS_WEIGHT = 190.189290837
P1_EXPERIMENT_VERSION = (
    "mix10b_sft90k_3072_v2r3_diagnostic_20260730"
)
P1_SOURCE_TREE_SHA256 = (
    "490b7cd758fce7e0187204449071d82da3e1ff42687f41323740c756287a7065"
)
P1_SOURCE_FLAT_MANIFEST_SHA256 = (
    "07ae91cded540a00e9b6554d1d54ed46310715b7fd68e3520a64b7f5967f99aa"
)

CANDIDATES: dict[int, dict[str, Any]] = {
    6_000: {
        "candidate_id": "v2r4-w190-step6000",
        "path": SNAPSHOT_ROOT / "step_6000/hf",
        "recursive_hf_identity": (
            "5df40e4794193a490297e19837ea5d8ec49326329ab405e58234b67519862425"
        ),
        "directory_manifest_sha256": (
            "3285baeb7c6ca4de2a320522906b031f6538c75106cb13fddc68194c96d23d70"
        ),
        "endpoint_checkpoint_sha256": (
            "17acd19dd1e89390c609a3f0f6c72ab543b8869f2d2ffd10528c8fe84cb20690"
        ),
        "training_state_sha256": (
            "5f6707dc5d641b64b9bd90b3cfd857c61943c6892ea6a33997eca609ceb91e0f"
        ),
    },
    8_000: {
        "candidate_id": "v2r4-w190-step8000",
        "path": SNAPSHOT_ROOT / "step_8000/hf",
        "recursive_hf_identity": (
            "d17a709df6debd483932e3e38214a91a1ec1f62814dd73dd6cad1f51a9b6070e"
        ),
        "directory_manifest_sha256": (
            "13fde44ba75511e8cd7d23a9e73db507bade841fcc682dc2851261690e918758"
        ),
        "endpoint_checkpoint_sha256": (
            "e1006a970b5b7c9c9e5aefdbae3c716740e69970c0bcb4bb32b4cbab7af43634"
        ),
        "training_state_sha256": (
            "0b1312d9a758c890a90b3ac704a4561ab9b5aaf666bc8fddac4240777c9b8e4b"
        ),
    },
    9_920: {
        "candidate_id": "v2r4-w190-step9920",
        "path": SNAPSHOT_ROOT / "step_9920/hf",
        "recursive_hf_identity": (
            "d0c013bf51c17691ef9bdf5e5d65561912471ef949a161f80b4aa818da96c4fd"
        ),
        "directory_manifest_sha256": (
            "49fe6fe87d78ba58ebd96cf154567bd1526b6c12a4193809652b875a7af5d186"
        ),
        "endpoint_checkpoint_sha256": (
            "9a89d52a60b87b0f27108e5b08e33395757e374a4b59a592babb9435edb4b1c8"
        ),
        "training_state_sha256": (
            "b042157d06bb7b89b1a69cb190cbde1e5d17455e76eda9f8c15639d90d4c05b7"
        ),
    },
}

TOKENIZER_FILE_SHA256 = {
    "special_tokens_map.json": (
        "803b05e0e15611d8fc9c3c159543939cdacd4bd6f225c6ea64aa5efa888c5e7b"
    ),
    "tokenizer.py": (
        "f10ebd5e21acc8a6422b27aae76a0805ab80143457a185ea22f52701569970c0"
    ),
    "tokenizer_config.json": (
        "a89b75502a9fabc68a68f52e9221cd9e5f2e1a4d46f5bc8475de1d0e96c25930"
    ),
    "vocab.json": (
        "f2857dc7f704632254b28366c65e1f78f75bbc67b7e84f26d4b88162cf00859d"
    ),
}
EXPECTED_MODEL_CONFIG = {
    "model_type": "qwen3",
    "vocab_size": p2_contract.VOCAB_SIZE,
    "max_position_embeddings": p2_contract.SEQUENCE_LENGTH,
    "hidden_size": 512,
    "head_dim": 128,
    "num_hidden_layers": 12,
    "num_attention_heads": 8,
    "num_key_value_heads": 4,
    "intermediate_size": 1536,
    "tie_word_embeddings": True,
    "bos_token_id": 0,
    "eos_token_id": 1,
    "pad_token_id": 0,
    "env_token_id": 84,
}

CONTAINER_IMAGE = "nvidia/cuda:12.8.0-runtime-ubuntu22.04"
PYTHON_VERSION = "3.11"
TORCH_VERSION = "2.9.0"
TRANSFORMERS_VERSION = "4.57.0"
NUMPY_VERSION = "2.2.6"
SAFETENSORS_VERSION = "0.6.2"
CHESS_VERSION = "1.11.2"
ATTENTION_BACKEND = "sdpa"
MODEL_DTYPE = "bfloat16"
BATCH_SIZE = 64

_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


RUNNER_SOURCE_SHA256 = _sha256_file(RUNNER_LOCAL_PATH)
CONTRACT_SOURCE_SHA256 = _sha256_file(CONTRACT_LOCAL_PATH)
EVALUATOR_SOURCE_SHA256 = hashlib.sha256(
    p2_contract.canonical_json(
        {
            "runner": RUNNER_SOURCE_SHA256,
            "pure_contract": CONTRACT_SOURCE_SHA256,
        }
    )
).hexdigest()
RUNTIME_CONTRACT = {
    "schema": "interleaved-v2r4-p2-sft-runtime-v2",
    "app_name": APP_NAME,
    "function_name": FUNCTION_NAME,
    "container_image": CONTAINER_IMAGE,
    "python": PYTHON_VERSION,
    "torch": TORCH_VERSION,
    "transformers": TRANSFORMERS_VERSION,
    "numpy": NUMPY_VERSION,
    "safetensors": SAFETENSORS_VERSION,
    "chess": CHESS_VERSION,
    "gpu": "H200",
    "attention_backend": ATTENTION_BACKEND,
    "model_dtype": MODEL_DTYPE,
    "batch_size": BATCH_SIZE,
    "source_sha256": EVALUATOR_SOURCE_SHA256,
    "pure_contract_version": p2_contract.CONTRACT_VERSION,
    "selection_hash": SELECTION_HASH,
    "cache_shape_hash": CACHE_SHAPE_HASH,
}
RUNTIME_CONTRACT_SHA256 = hashlib.sha256(
    p2_contract.canonical_json(RUNTIME_CONTRACT)
).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"expected one JSON object: {path}")
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    encoded = (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    with temporary.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    expected = dict(payload)
    if path.is_file():
        if _read_json(path) != expected:
            raise ValueError(f"immutable result differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f".{path.name}.immutable-lock")
    try:
        lock_fd = os.open(
            lock_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError as exc:
        raise RuntimeError(
            f"concurrent or interrupted immutable result writer: {path}"
        ) from exc
    try:
        os.close(lock_fd)
        if path.is_file():
            if _read_json(path) != expected:
                raise ValueError(f"immutable result differs: {path}")
            return
        _atomic_json(path, expected)
        if _read_json(path) != expected:
            raise RuntimeError(f"atomic result verification failed: {path}")
    finally:
        lock_path.unlink(missing_ok=True)


def _exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Create a durable launch intent exactly once without replacement."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o644,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _directory_identities(root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    files: list[dict[str, Any]] = []
    tab_rows: list[str] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"checkpoint must not contain symlinks: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        digest = _sha256_file(path)
        size = path.stat().st_size
        files.append({"path": relative, "bytes": size, "sha256": digest})
        tab_rows.append(f"{relative}\t{size}\t{digest}\n")
    if not files:
        raise ValueError(f"empty checkpoint directory: {root}")
    recursive = hashlib.sha256(
        p2_contract.canonical_json(files)
    ).hexdigest()
    directory_manifest = hashlib.sha256(
        "".join(tab_rows).encode("utf-8")
    ).hexdigest()
    return {
        "file_count": len(files),
        "total_bytes": sum(int(item["bytes"]) for item in files),
        "files": files,
        "recursive_hf_identity": recursive,
        "directory_manifest_sha256": directory_manifest,
    }


def _endpoint_checkpoint_fingerprint(root: Path) -> str:
    included: set[Path] = set()
    weights = set(root.glob("model*.safetensors"))
    if not weights:
        raise FileNotFoundError(f"missing model safetensors: {root}")
    included.update(weights)
    for name in (
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "interleaved_training_state.json",
    ):
        path = root / name
        if path.is_file():
            included.add(path)
    for pattern in (
        "tokenizer*",
        "vocab*",
        "merges*",
        "special_tokens_map.json",
        "added_tokens.json",
        "sentencepiece*",
        "spiece*",
    ):
        included.update(path for path in root.glob(pattern) if path.is_file())
    digest = hashlib.sha256()
    for path in sorted(included, key=lambda item: item.relative_to(root).as_posix()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            while chunk := handle.read(8 << 20):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _validate_p1_training_state(
    state: Mapping[str, Any], *, candidate_step: int
) -> None:
    expected_top = {
        "global_step": candidate_step,
        "manifest_cursor": candidate_step,
        "manifest_hash": p2_contract.P1_METADATA_FILE_SHA256,
        "sft_loss_weight": SFT_LOSS_WEIGHT,
        "world_size": 8,
        "local_batch_size": 21,
        "gradient_accumulation_steps": 1,
        "snapshot_steps": [1_000, 2_000, 4_000, 6_000, 8_000, 9_920],
        "arc_steps": [9_920],
    }
    mismatches = {
        key: (state.get(key), expected)
        for key, expected in expected_top.items()
        if state.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"candidate P1 training-state drifted: {mismatches}")
    data_state = state.get("data_state")
    if not isinstance(data_state, Mapping) or dict(data_state) != {
        "schema": "interleaved-stream-state-v1",
        "cursor": candidate_step,
        "local_batch_size": 21,
        "world_size": 8,
        "manifest_hash": p2_contract.P1_METADATA_FILE_SHA256,
        "selection_hash": p2_contract.SELECTION_HASH,
        "sft_cache_hash": p2_contract.SFT_CACHE_HASH,
        "source_manifest_hash": p2_contract.SOURCE_MANIFEST_HASH,
    }:
        raise ValueError("candidate is not at the exact P1-only data cursor")
    configured = state.get("configured_provenance")
    expected_configured = {
        "experiment_version": P1_EXPERIMENT_VERSION,
        "data_artifact_version": p2_contract.DATA_ARTIFACT_VERSION,
        "sft_loss_weight": SFT_LOSS_WEIGHT,
        "source_repo": p2_contract.SOURCE_REPO,
        "source_revision": p2_contract.SOURCE_REVISION,
        "sft_repo": p2_contract.SFT_REPO,
        "sft_revision": p2_contract.SFT_REVISION,
        "source_tree_sha256": P1_SOURCE_TREE_SHA256,
        "source_flat_manifest_sha256": P1_SOURCE_FLAT_MANIFEST_SHA256,
        "sft_response_normalization": (
            "strip-numeric-verify-score-pairs-normalize-whitespace-v1"
        ),
        "sft_supervised_unk_policy": "reject-supervised-unk-v1",
    }
    if not isinstance(configured, Mapping):
        raise ValueError("candidate lacks configured P1 provenance")
    configured_mismatches = {
        key: (configured.get(key), expected)
        for key, expected in expected_configured.items()
        if configured.get(key) != expected
    }
    if configured_mismatches:
        raise ValueError(
            f"candidate configured P1 provenance drifted: "
            f"{configured_mismatches}"
        )


def _authenticate_candidate_checkpoint(
    candidate_step: int,
    *,
    expected_recursive_hf_identity: str,
    expected_endpoint_checkpoint_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if candidate_step not in CANDIDATES:
        raise ValueError(f"unsupported v2r4 candidate step: {candidate_step}")
    candidate = CANDIDATES[candidate_step]
    if expected_recursive_hf_identity != candidate["recursive_hf_identity"]:
        raise ValueError("caller recursive HF identity differs from frozen value")
    if (
        expected_endpoint_checkpoint_sha256
        != candidate["endpoint_checkpoint_sha256"]
    ):
        raise ValueError(
            "caller endpoint checkpoint identity differs from frozen value"
        )
    root = Path(candidate["path"]).resolve(strict=True)
    identities = _directory_identities(root)
    if (
        identities["recursive_hf_identity"]
        != candidate["recursive_hf_identity"]
        or identities["directory_manifest_sha256"]
        != candidate["directory_manifest_sha256"]
    ):
        raise ValueError("recursive candidate HF directory identity drifted")
    endpoint_identity = _endpoint_checkpoint_fingerprint(root)
    if endpoint_identity != candidate["endpoint_checkpoint_sha256"]:
        raise ValueError("candidate endpoint checkpoint fingerprint drifted")
    for relative, expected in TOKENIZER_FILE_SHA256.items():
        if _sha256_file(root / relative) != expected:
            raise ValueError(f"exact tokenizer artifact drifted: {relative}")
    config = _read_json(root / "config.json")
    model_mismatches = {
        key: (config.get(key), expected)
        for key, expected in EXPECTED_MODEL_CONFIG.items()
        if config.get(key) != expected
    }
    if model_mismatches:
        raise ValueError(f"candidate model architecture drifted: {model_mismatches}")
    state_path = root / "interleaved_training_state.json"
    if _sha256_file(state_path) != candidate["training_state_sha256"]:
        raise ValueError("candidate training-state byte identity drifted")
    state = _read_json(state_path)
    _validate_p1_training_state(state, candidate_step=candidate_step)
    return (
        {
            "candidate_id": candidate["candidate_id"],
            "checkpoint_sha256": candidate["recursive_hf_identity"],
            "checkpoint_step": candidate_step,
            "sft_loss_weight": SFT_LOSS_WEIGHT,
            "training_data_manifest_set_hash": p2_contract.MANIFEST_SET_HASH,
            "training_leg": "p1",
            "has_consumed_p2": False,
        },
        {
            "path": str(root),
            **identities,
            "endpoint_checkpoint_sha256": endpoint_identity,
            "training_state_sha256": candidate["training_state_sha256"],
            "tokenizer_file_sha256": dict(TOKENIZER_FILE_SHA256),
            "p1_only_provenance": {
                "global_step": candidate_step,
                "manifest_cursor": candidate_step,
                "p1_metadata_file_sha256": (
                    p2_contract.P1_METADATA_FILE_SHA256
                ),
                "p1_metadata_hash": p2_contract.P1_METADATA_HASH,
                "manifest_set_hash": p2_contract.MANIFEST_SET_HASH,
                "has_consumed_p2": False,
            },
        },
    )


def _authenticate_data_contract() -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any]
]:
    import numpy as np

    observed_hashes = {
        relative: _sha256_file(path)
        for relative, path in SFT_ARTIFACT_PATHS.items()
    }
    mismatches = {
        relative: (observed_hashes[relative], expected)
        for relative, expected in SFT_ARTIFACT_SHA256.items()
        if observed_hashes[relative] != expected
    }
    if mismatches:
        raise ValueError(f"P2 SFT source artifact drifted: {mismatches}")
    if SFT_INPUT_IDS_PATH.stat().st_size != (
        p2_contract.SFT_TOTAL_POSITIONS * 4
    ):
        raise ValueError("input_ids.i32 byte length drifted")
    if SFT_LABELS_PATH.stat().st_size != (
        p2_contract.SFT_TOTAL_POSITIONS * 4
    ):
        raise ValueError("labels.i32 byte length drifted")

    raw = {
        "manifest_set_json": MANIFEST_SET_PATH.read_bytes(),
        "sft_cache_metadata_json": SFT_CACHE_METADATA_PATH.read_bytes(),
        "p1_metadata_json": P1_METADATA_PATH.read_bytes(),
        "p2_metadata_json": P2_METADATA_PATH.read_bytes(),
        "p1_order_npy": P1_ORDER_PATH.read_bytes(),
        "p2_order_npy": P2_ORDER_PATH.read_bytes(),
    }
    selection = p2_contract.build_p2_at_p1_sft_selection(**raw)
    p2_contract.validate_p2_at_p1_sft_selection(selection, **raw)
    if (
        selection.get("selection_hash") != SELECTION_HASH
        or selection["selection"].get("signed_sft_codes_sha256")
        != SELECTED_CODES_SHA256
    ):
        raise ValueError("frozen P2 selection identity drifted")

    labels = np.memmap(SFT_LABELS_PATH, dtype="<i4", mode="r")
    cache_shape = p2_contract.build_selected_cache_shape(
        selection,
        offsets_npy=SFT_OFFSETS_PATH.read_bytes(),
        labels_i32=labels,
    )
    p2_contract.validate_selected_cache_shape(cache_shape, selection)
    if (
        cache_shape.get("cache_shape_hash") != CACHE_SHAPE_HASH
        or cache_shape.get("shape") != EXPECTED_SELECTED_SHAPE
    ):
        raise ValueError("frozen selected cache shape drifted")
    return selection, cache_shape, {
        "artifact_sha256": observed_hashes,
        "input_ids_bytes": SFT_INPUT_IDS_PATH.stat().st_size,
        "labels_bytes": SFT_LABELS_PATH.stat().st_size,
        "selection_hash": SELECTION_HASH,
        "cache_shape_hash": CACHE_SHAPE_HASH,
    }


def _validate_exact_tokenizer(checkpoint: Path) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint),
        trust_remote_code=True,
        use_fast=False,
        local_files_only=True,
    )
    observed = {
        "class": type(tokenizer).__name__,
        "vocab_size": len(tokenizer),
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "unk_token_id": tokenizer.unk_token_id,
        "env_token_id": tokenizer.convert_tokens_to_ids("<call_env>"),
    }
    expected = {
        "class": "HFTokenizerWrapper",
        "vocab_size": p2_contract.VOCAB_SIZE,
        "bos_token_id": 0,
        "eos_token_id": 1,
        "pad_token_id": 0,
        "unk_token_id": 2,
        "env_token_id": 84,
    }
    if observed != expected:
        raise ValueError(f"loaded exact tokenizer semantics drifted: {observed}")
    return observed


def _runtime_source_audit() -> dict[str, str]:
    observed = {
        "runner": _sha256_file(RUNNER_REMOTE_PATH),
        "pure_contract": _sha256_file(CONTRACT_REMOTE_PATH),
    }
    expected = {
        "runner": RUNNER_SOURCE_SHA256,
        "pure_contract": CONTRACT_SOURCE_SHA256,
    }
    if observed != expected:
        raise RuntimeError(
            f"remote P2 evaluator source identity drifted: {observed}"
        )
    if hashlib.sha256(
        p2_contract.canonical_json(observed)
    ).hexdigest() != EVALUATOR_SOURCE_SHA256:
        raise RuntimeError("remote P2 evaluator source bundle hash drifted")
    return observed


def _result_path(candidate_step: int) -> Path:
    if candidate_step not in CANDIDATES:
        raise ValueError(f"unsupported v2r4 candidate step: {candidate_step}")
    return _result_root(candidate_step) / "_SUCCESS.json"


def _result_root(candidate_step: int) -> Path:
    if candidate_step not in CANDIDATES:
        raise ValueError(f"unsupported v2r4 candidate step: {candidate_step}")
    identity = CANDIDATES[candidate_step]["recursive_hf_identity"]
    return (
        RESULTS_ROOT
        / CANDIDATES[candidate_step]["candidate_id"]
        / identity
    )


def direct_call_contract(candidate_step: int) -> dict[str, Any]:
    if candidate_step not in CANDIDATES:
        raise ValueError(f"unsupported v2r4 candidate step: {candidate_step}")
    candidate = CANDIDATES[candidate_step]
    return {
        "app_name": APP_NAME,
        "function_name": FUNCTION_NAME,
        "kwargs": {
            "candidate_step": candidate_step,
            "expected_recursive_hf_identity": (
                candidate["recursive_hf_identity"]
            ),
            "expected_endpoint_checkpoint_sha256": (
                candidate["endpoint_checkpoint_sha256"]
            ),
            "runtime_contract_sha256": RUNTIME_CONTRACT_SHA256,
        },
        "expected_result_path": str(_result_path(candidate_step)),
    }


loss_image = (
    modal.Image.from_registry(CONTAINER_IMAGE, add_python=PYTHON_VERSION)
    .pip_install(
        f"torch=={TORCH_VERSION}",
        f"transformers=={TRANSFORMERS_VERSION}",
        f"numpy=={NUMPY_VERSION}",
        f"safetensors=={SAFETENSORS_VERSION}",
        f"chess=={CHESS_VERSION}",
    )
    .env({"PYTHONUNBUFFERED": "1", "PYTHONPATH": "/root"})
    .add_local_file(
        str(RUNNER_LOCAL_PATH),
        remote_path=str(RUNNER_REMOTE_PATH),
        copy=True,
    )
    .add_local_file(
        str(CONTRACT_LOCAL_PATH),
        remote_path=str(CONTRACT_REMOTE_PATH),
        copy=True,
    )
)
control_image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .pip_install(f"numpy=={NUMPY_VERSION}")
    .env({"PYTHONUNBUFFERED": "1", "PYTHONPATH": "/root"})
    .add_local_file(
        str(RUNNER_LOCAL_PATH),
        remote_path=str(RUNNER_REMOTE_PATH),
        copy=True,
    )
    .add_local_file(
        str(CONTRACT_LOCAL_PATH),
        remote_path=str(CONTRACT_REMOTE_PATH),
        copy=True,
    )
)

data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=False)
checkpoint_volume = modal.Volume.from_name(
    CHECKPOINT_VOLUME_NAME, create_if_missing=False
)
results_volume = modal.Volume.from_name(
    RESULTS_VOLUME_NAME, create_if_missing=True
)
app = modal.App(APP_NAME)


@app.function(
    image=loss_image,
    cpu=4.0,
    memory=16 * 1024,
    timeout=30 * 60,
    retries=0,
    volumes={str(CHECKPOINT_MOUNT): checkpoint_volume},
)
def preflight_p2_sft_v2_runtime(
    runtime_contract_sha256: str,
) -> dict[str, Any]:
    """Prove the exact GPU-image dependencies without allocating a GPU."""

    import chess
    import numpy as np
    import torch
    import transformers

    if runtime_contract_sha256 != RUNTIME_CONTRACT_SHA256:
        raise ValueError(
            "dependency preflight runtime contract differs from frozen value"
        )
    observed_versions = {
        "torch": str(torch.__version__).split("+")[0],
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "chess": chess.__version__,
    }
    expected_versions = {
        "torch": TORCH_VERSION,
        "transformers": TRANSFORMERS_VERSION,
        "numpy": NUMPY_VERSION,
        "chess": CHESS_VERSION,
    }
    if observed_versions != expected_versions:
        raise RuntimeError(
            "v2 GPU-image dependency versions drifted: "
            f"{observed_versions} != {expected_versions}"
        )
    checkpoint_volume.reload()
    source_files = _runtime_source_audit()
    tokenizer_semantics: dict[str, Any] = {}
    for step in sorted(CANDIDATES):
        tokenizer_semantics[str(step)] = _validate_exact_tokenizer(
            Path(CANDIDATES[step]["path"])
        )
    value: dict[str, Any] = {
        "schema": "interleaved-v2r4-p2-sft-runtime-preflight-v2",
        "state": "complete",
        "runtime_contract_sha256": RUNTIME_CONTRACT_SHA256,
        "source_files": source_files,
        "versions": observed_versions,
        "tokenizer_semantics": tokenizer_semantics,
        "gpu_allocated": False,
    }
    value["dependency_preflight_hash"] = p2_contract.content_hash(
        value, "dependency_preflight_hash"
    )
    return value


@app.function(
    image=control_image,
    cpu=8.0,
    memory=64 * 1024,
    timeout=60 * 60,
    retries=0,
    volumes={
        str(DATA_MOUNT): data_volume,
        str(CHECKPOINT_MOUNT): checkpoint_volume,
        str(RESULTS_MOUNT): results_volume,
    },
)
def preflight_p2_sft_grid(
    runtime_contract_sha256: str,
) -> dict[str, Any]:
    """Authenticate all inputs and prove all three result roots are absent."""

    if runtime_contract_sha256 != RUNTIME_CONTRACT_SHA256:
        raise ValueError("preflight runtime contract differs from frozen value")
    data_volume.reload()
    checkpoint_volume.reload()
    results_volume.reload()
    source_files = _runtime_source_audit()
    selection, cache_shape, data_identity = _authenticate_data_contract()
    roots = {
        str(step): str(_result_root(step)) for step in sorted(CANDIDATES)
    }
    existing = [path for path in roots.values() if Path(path).exists()]
    if existing:
        raise FileExistsError(
            "P2 SFT preflight found preexisting result roots: "
            + ", ".join(existing)
        )
    checkpoints: dict[str, Any] = {}
    for step in sorted(CANDIDATES):
        candidate = CANDIDATES[step]
        _, identity = _authenticate_candidate_checkpoint(
            step,
            expected_recursive_hf_identity=(
                candidate["recursive_hf_identity"]
            ),
            expected_endpoint_checkpoint_sha256=(
                candidate["endpoint_checkpoint_sha256"]
            ),
        )
        checkpoints[str(step)] = {
            "recursive_hf_identity": identity["recursive_hf_identity"],
            "directory_manifest_sha256": (
                identity["directory_manifest_sha256"]
            ),
            "endpoint_checkpoint_sha256": (
                identity["endpoint_checkpoint_sha256"]
            ),
            "training_state_sha256": identity["training_state_sha256"],
        }
    value: dict[str, Any] = {
        "schema": "interleaved-v2r4-p2-sft-preflight-v1",
        "state": "complete",
        "runtime_contract_sha256": RUNTIME_CONTRACT_SHA256,
        "source_files": source_files,
        "selection_hash": selection["selection_hash"],
        "cache_shape_hash": cache_shape["cache_shape_hash"],
        "data": data_identity,
        "checkpoints": checkpoints,
        "result_roots": roots,
        "all_three_result_roots_absent": True,
    }
    value["preflight_hash"] = p2_contract.content_hash(
        value, "preflight_hash"
    )
    return value


def _unshifted_masked_sums(
    logits: Any,
    labels: Any,
    attention_mask: Any,
) -> tuple[float, int, int]:
    """Score cache-aligned labels without applying a second causal shift."""

    import torch
    import torch.nn.functional as F

    if (
        logits.ndim != 3
        or labels.ndim != 2
        or attention_mask.ndim != 2
        or tuple(logits.shape[:2]) != tuple(labels.shape)
        or tuple(labels.shape) != tuple(attention_mask.shape)
    ):
        raise ValueError("logits/labels/attention cache alignment drifted")
    supervised = labels.ne(
        p2_contract.OBJECTIVE["ignore_index"]
    ) & attention_mask.bool()
    if bool(
        (
            labels.ne(p2_contract.OBJECTIVE["ignore_index"])
            & ~attention_mask.bool()
        )
        .any()
        .item()
    ):
        raise ValueError("supervised label exists outside the attention mask")
    nll = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=p2_contract.OBJECTIVE["ignore_index"],
        reduction="sum",
    )
    correct = (
        logits.argmax(dim=-1).eq(labels) & supervised
    ).sum()
    return (
        float(nll.item()),
        int(correct.item()),
        int(supervised.sum().item()),
    )


@app.function(
    name=FUNCTION_NAME,
    image=loss_image,
    gpu="H200",
    cpu=16.0,
    memory=128 * 1024,
    timeout=4 * 60 * 60,
    retries=0,
    max_containers=3,
    volumes={
        str(DATA_MOUNT): data_volume,
        str(CHECKPOINT_MOUNT): checkpoint_volume,
        str(RESULTS_MOUNT): results_volume,
    },
)
def evaluate_p2_sft_candidate(
    candidate_step: int,
    expected_recursive_hf_identity: str,
    expected_endpoint_checkpoint_sha256: str,
    runtime_contract_sha256: str,
) -> dict[str, Any]:
    """Evaluate one exact P1 snapshot on the exact P2-heldout SFT rows."""

    import numpy as np
    import torch
    import transformers
    import chess
    from transformers import AutoModelForCausalLM

    if runtime_contract_sha256 != RUNTIME_CONTRACT_SHA256:
        raise ValueError("caller runtime contract differs from frozen evaluator")
    if not _SHA256_RE.fullmatch(expected_recursive_hf_identity):
        raise ValueError("expected recursive HF identity must be a SHA-256")
    if not _SHA256_RE.fullmatch(expected_endpoint_checkpoint_sha256):
        raise ValueError("expected endpoint checkpoint identity must be a SHA-256")

    started_at = _utc_now()
    started_clock = time.monotonic()
    data_volume.reload()
    checkpoint_volume.reload()
    results_volume.reload()
    runtime = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "chess": chess.__version__,
        "attention_backend": ATTENTION_BACKEND,
        "model_dtype": MODEL_DTYPE,
        "batch_size": BATCH_SIZE,
    }
    expected_versions = {
        "torch": TORCH_VERSION,
        "transformers": TRANSFORMERS_VERSION,
        "numpy": NUMPY_VERSION,
        "chess": CHESS_VERSION,
    }
    version_mismatches = {
        key: (str(runtime[key]).split("+")[0], expected)
        for key, expected in expected_versions.items()
        if str(runtime[key]).split("+")[0] != expected
    }
    if version_mismatches or "H200" not in str(runtime["gpu"]):
        raise RuntimeError(
            f"runtime differs from frozen P2 evaluator: {version_mismatches}, "
            f"gpu={runtime['gpu']!r}"
        )
    if candidate_step not in CANDIDATES:
        raise ValueError(f"unsupported v2r4 candidate step: {candidate_step}")
    result_root = _result_root(candidate_step)
    if result_root.exists():
        raise FileExistsError(
            f"P2 SFT worker refuses preexisting initial result root: "
            f"{result_root}"
        )
    source_files = _runtime_source_audit()
    selection, cache_shape, data_identity = _authenticate_data_contract()
    pure_candidate, checkpoint_identity = _authenticate_candidate_checkpoint(
        candidate_step,
        expected_recursive_hf_identity=expected_recursive_hf_identity,
        expected_endpoint_checkpoint_sha256=(
            expected_endpoint_checkpoint_sha256
        ),
    )
    checkpoint = Path(checkpoint_identity["path"])
    tokenizer_identity = _validate_exact_tokenizer(checkpoint)
    success_path = _result_path(candidate_step)

    model = AutoModelForCausalLM.from_pretrained(
        str(checkpoint),
        torch_dtype=torch.bfloat16,
        attn_implementation=ATTENTION_BACKEND,
        local_files_only=True,
        trust_remote_code=False,
    ).to("cuda")
    model.eval()
    input_ids = np.memmap(SFT_INPUT_IDS_PATH, dtype="<i4", mode="r")
    labels = np.memmap(SFT_LABELS_PATH, dtype="<i4", mode="r")
    offsets = np.load(SFT_OFFSETS_PATH, mmap_mode="r", allow_pickle=False)
    selected_rows = selection["selection"]["global_sft_row_ids"]

    total_nll = 0.0
    total_correct = 0
    total_supervised = 0
    total_positions = 0
    batches = 0
    with torch.inference_mode():
        for batch_start in range(0, len(selected_rows), BATCH_SIZE):
            row_ids = selected_rows[batch_start : batch_start + BATCH_SIZE]
            lengths = [
                int(offsets[row_id + 1]) - int(offsets[row_id])
                for row_id in row_ids
            ]
            max_length = max(lengths)
            batch_inputs = np.zeros(
                (len(row_ids), max_length), dtype=np.int64
            )
            batch_labels = np.full(
                (len(row_ids), max_length),
                p2_contract.OBJECTIVE["ignore_index"],
                dtype=np.int64,
            )
            attention = np.zeros(
                (len(row_ids), max_length), dtype=np.int64
            )
            for row_index, (row_id, length) in enumerate(
                zip(row_ids, lengths, strict=True)
            ):
                start = int(offsets[row_id])
                stop = int(offsets[row_id + 1])
                row_inputs = np.asarray(input_ids[start:stop], dtype=np.int64)
                row_labels = np.asarray(labels[start:stop], dtype=np.int64)
                if (
                    length <= 1
                    or bool(np.any(row_inputs < 0))
                    or bool(np.any(row_inputs >= p2_contract.VOCAB_SIZE))
                ):
                    raise ValueError(f"selected cache row {row_id} drifted")
                batch_inputs[row_index, :length] = row_inputs
                batch_labels[row_index, :length] = row_labels
                attention[row_index, :length] = 1
                total_positions += length

            gpu_inputs = torch.from_numpy(batch_inputs).to(
                "cuda", non_blocking=True
            )
            gpu_labels = torch.from_numpy(batch_labels).to(
                "cuda", non_blocking=True
            )
            gpu_attention = torch.from_numpy(attention).to(
                "cuda", non_blocking=True
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(
                    input_ids=gpu_inputs,
                    attention_mask=gpu_attention,
                    use_cache=False,
                ).logits
            batch_nll, batch_correct, batch_supervised = (
                _unshifted_masked_sums(
                    logits,
                    gpu_labels,
                    gpu_attention,
                )
            )
            total_nll += batch_nll
            total_correct += batch_correct
            total_supervised += batch_supervised
            batches += 1

    shape = cache_shape["shape"]
    if (
        total_positions != int(shape["total_aligned_positions"])
        or total_supervised != int(shape["supervised_targets"])
        or batches != math.ceil(p2_contract.SELECTION_RECORDS / BATCH_SIZE)
    ):
        raise ValueError(
            "evaluated P2 denominator differs from frozen cache shape"
        )
    pure_result = p2_contract.build_candidate_sft_result(
        selection_manifest=selection,
        cache_shape=cache_shape,
        candidate=pure_candidate,
        negative_log_likelihood_sum=total_nll,
        correct_supervised_tokens=total_correct,
        batches=batches,
    )
    p2_contract.validate_candidate_sft_result(
        pure_result,
        selection_manifest=selection,
        cache_shape=cache_shape,
        expected_candidate=pure_candidate,
    )
    payload: dict[str, Any] = {
        "schema": OUTPUT_SCHEMA,
        "schema_version": 1,
        "state": "complete",
        "contract_version": CONTRACT_VERSION,
        "runtime_contract": dict(RUNTIME_CONTRACT),
        "runtime_contract_sha256": RUNTIME_CONTRACT_SHA256,
        "evaluator_source": {
            **source_files,
            "bundle_sha256": EVALUATOR_SOURCE_SHA256,
        },
        "runtime": runtime,
        "candidate": pure_candidate,
        "checkpoint": checkpoint_identity,
        "tokenizer": tokenizer_identity,
        "data": data_identity,
        "selection": selection,
        "cache_shape": cache_shape,
        "p2_sft_result": pure_result,
        "hash_bindings": {
            "selection_hash": selection["selection_hash"],
            "cache_shape_hash": cache_shape["cache_shape_hash"],
            "candidate_result_hash": pure_result["result_hash"],
            "checkpoint_recursive_hf_identity": (
                checkpoint_identity["recursive_hf_identity"]
            ),
            "evaluator_source_sha256": EVALUATOR_SOURCE_SHA256,
            "runtime_contract_sha256": RUNTIME_CONTRACT_SHA256,
        },
        "started_at": started_at,
        "finished_at": _utc_now(),
        "duration_seconds": round(time.monotonic() - started_clock, 3),
    }
    payload["output_hash"] = p2_contract.content_hash(
        payload, "output_hash"
    )
    _immutable_json(success_path, payload)
    results_volume.commit()
    return payload


@app.local_entrypoint()
def main(
    mode: str = "dry-run",
    candidate_step: int = 0,
    launch_ledger_path: str = DEFAULT_LAUNCH_LEDGER_PATH,
) -> None:
    """Print contracts, launch all three cells, or inspect immutable results."""

    mode = mode.strip().lower()
    steps = (
        [candidate_step]
        if candidate_step
        else sorted(CANDIDATES)
    )
    if any(step not in CANDIDATES for step in steps):
        raise ValueError(f"candidate_step must be one of {sorted(CANDIDATES)}")
    contracts = [direct_call_contract(step) for step in steps]
    if mode == "dry-run":
        print(
            json.dumps(
                {
                    "state": "dry-run",
                    "app_name": APP_NAME,
                    "runtime_contract": RUNTIME_CONTRACT,
                    "runtime_contract_sha256": RUNTIME_CONTRACT_SHA256,
                    "evaluator_source_sha256": EVALUATOR_SOURCE_SHA256,
                    "preflight": {
                        "runtime_dependency_function_name": (
                            "preflight_p2_sft_v2_runtime"
                        ),
                        "data_and_root_function_name": (
                            "preflight_p2_sft_grid"
                        ),
                        "runtime_contract_sha256": (
                            RUNTIME_CONTRACT_SHA256
                        ),
                        "requires_all_three_result_roots_absent": True,
                    },
                    "launch_ledger_path": str(
                        Path(launch_ledger_path).expanduser().resolve()
                    ),
                    "calls": contracts,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if mode == "launch":
        if candidate_step:
            raise ValueError(
                "launch is an exact three-call operation; "
                "--candidate-step is forbidden"
            )
        ledger_path = Path(launch_ledger_path).expanduser().resolve()
        ledger: dict[str, Any] = {
            "schema": "interleaved-v2r4-p2-sft-launch-ledger-v2",
            "state": "dependency_preflight_pending",
            "app_name": APP_NAME,
            "function_name": FUNCTION_NAME,
            "runtime_contract_sha256": RUNTIME_CONTRACT_SHA256,
            "evaluator_source_sha256": EVALUATOR_SOURCE_SHA256,
            "expected_call_count": 3,
            "calls": [],
        }

        def persist_ledger() -> None:
            core = {
                key: value
                for key, value in ledger.items()
                if key != "ledger_hash"
            }
            ledger["ledger_hash"] = p2_contract.content_hash(
                core, "ledger_hash"
            )
            _atomic_json(ledger_path, ledger)

        initial_core = dict(ledger)
        ledger["ledger_hash"] = p2_contract.content_hash(
            initial_core, "ledger_hash"
        )
        _exclusive_json(ledger_path, ledger)
        dependency_call = preflight_p2_sft_v2_runtime.spawn(
            RUNTIME_CONTRACT_SHA256
        )
        ledger["dependency_preflight_function_call_id"] = (
            dependency_call.object_id
        )
        ledger["state"] = "dependency_preflight_running"
        persist_ledger()
        try:
            dependency_preflight = dependency_call.get()
        except Exception as exc:
            ledger["state"] = "dependency_preflight_failed"
            ledger["error"] = f"{type(exc).__name__}: {exc}"
            persist_ledger()
            raise
        if (
            dependency_preflight.get("state") != "complete"
            or dependency_preflight.get("runtime_contract_sha256")
            != RUNTIME_CONTRACT_SHA256
            or dependency_preflight.get("gpu_allocated") is not False
            or set(dependency_preflight.get("tokenizer_semantics", {}))
            != {"6000", "8000", "9920"}
            or dependency_preflight.get("dependency_preflight_hash")
            != p2_contract.content_hash(
                dependency_preflight, "dependency_preflight_hash"
            )
        ):
            ledger["state"] = "dependency_preflight_failed"
            ledger["error"] = (
                "dependency preflight result shape or self hash drifted"
            )
            persist_ledger()
            raise RuntimeError("P2 SFT v2 dependency preflight drifted")
        ledger["dependency_preflight"] = dependency_preflight
        ledger["state"] = "preflight_pending"
        persist_ledger()
        preflight_call = preflight_p2_sft_grid.spawn(
            RUNTIME_CONTRACT_SHA256
        )
        ledger["preflight_function_call_id"] = preflight_call.object_id
        ledger["state"] = "preflight_running"
        persist_ledger()
        try:
            preflight = preflight_call.get()
        except Exception as exc:
            ledger["state"] = "preflight_failed"
            ledger["error"] = f"{type(exc).__name__}: {exc}"
            persist_ledger()
            raise
        if (
            preflight.get("state") != "complete"
            or preflight.get("runtime_contract_sha256")
            != RUNTIME_CONTRACT_SHA256
            or preflight.get("all_three_result_roots_absent") is not True
            or set(preflight.get("checkpoints", {}))
            != {"6000", "8000", "9920"}
            or preflight.get("preflight_hash")
            != p2_contract.content_hash(preflight, "preflight_hash")
        ):
            ledger["state"] = "preflight_failed"
            ledger["error"] = "preflight result shape or self hash drifted"
            persist_ledger()
            raise RuntimeError("P2 SFT preflight result drifted")
        ledger["preflight"] = preflight
        ledger["state"] = "launching"
        persist_ledger()
        for contract in contracts:
            try:
                call = evaluate_p2_sft_candidate.spawn(
                    **contract["kwargs"]
                )
            except Exception as exc:
                ledger["state"] = "launch_failed"
                ledger["failed_candidate_step"] = contract["kwargs"][
                    "candidate_step"
                ]
                ledger["error"] = f"{type(exc).__name__}: {exc}"
                persist_ledger()
                raise
            ledger["calls"].append(
                {
                    **contract,
                    "function_call_id": call.object_id,
                }
            )
            persist_ledger()
            print(
                json.dumps(ledger["calls"][-1], sort_keys=True),
                flush=True,
            )
        ledger["state"] = "launched_all"
        persist_ledger()
        print(json.dumps(ledger, sort_keys=True), flush=True)
        return
    if mode == "preflight":
        if candidate_step:
            raise ValueError(
                "preflight always authenticates the complete three-cell grid"
            )
        dependency = preflight_p2_sft_v2_runtime.remote(
            RUNTIME_CONTRACT_SHA256
        )
        result = preflight_p2_sft_grid.remote(
            RUNTIME_CONTRACT_SHA256
        )
        print(
            json.dumps(
                {
                    "dependency_preflight": dependency,
                    "data_and_root_preflight": result,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if mode == "status":
        results_volume.reload()
        status = []
        for contract in contracts:
            path = Path(contract["expected_result_path"])
            if not path.is_file():
                status.append(
                    {
                        "candidate_step": (
                            contract["kwargs"]["candidate_step"]
                        ),
                        "state": "missing",
                        "path": str(path),
                    }
                )
                continue
            value = _read_json(path)
            valid = (
                value.get("schema") == OUTPUT_SCHEMA
                and value.get("state") == "complete"
                and value.get("output_hash")
                == p2_contract.content_hash(value, "output_hash")
            )
            status.append(
                {
                    "candidate_step": contract["kwargs"]["candidate_step"],
                    "state": "complete" if valid else "invalid",
                    "path": str(path),
                    "output_hash": value.get("output_hash"),
                    "metrics": value.get("p2_sft_result", {}).get("metrics"),
                }
            )
        print(json.dumps(status, indent=2, sort_keys=True))
        return
    raise ValueError(
        "mode must be 'dry-run', 'preflight', 'launch', or 'status'"
    )


__all__ = [
    "APP_NAME",
    "CACHE_SHAPE_HASH",
    "CANDIDATES",
    "EVALUATOR_SOURCE_SHA256",
    "FUNCTION_NAME",
    "RUNTIME_CONTRACT_SHA256",
    "SELECTION_HASH",
    "SFT_ARTIFACT_SHA256",
    "TOKENIZER_FILE_SHA256",
    "direct_call_contract",
    "evaluate_p2_sft_candidate",
]
