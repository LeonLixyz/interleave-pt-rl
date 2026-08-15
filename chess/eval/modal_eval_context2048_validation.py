"""Deterministic held-out pretraining loss for the four context-2048 models.

The evaluation set contains 4,096 records with 2,048 next-token targets per
record.  Records are selected only from source shards whose paths are absent
from every pretraining-selection span.  The persisted holdout authenticates
both the selected source files and the exact token records consumed by the
model.  Checkpoint results are immutable, self-hashed success markers.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import struct
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import modal


APP_NAME = "chess-context2048-fp32-master-v13-heldout-pt-validation"
VERSION = "context2048-fp32-master-v13-heldout-pt-v1-20260814"
RESULT_SCHEMA = "context2048-heldout-pt-result-v1"
HOLDOUT_SCHEMA = "context2048-heldout-pt-manifest-v1"
RUN_SCHEMA = "context2048-heldout-pt-run-v1"

DATA_MOUNT = Path("/data")
CHECKPOINT_MOUNT = Path("/pretrain-checkpoints")
RESULTS_MOUNT = Path("/results")
ARTIFACT_ROOT = (
    DATA_MOUNT / "context2048_vocab_mixing_fp32_master_v13_20260813"
)
SOURCE_MANIFEST_PATH = ARTIFACT_ROOT / "source_manifest.json"
SELECTION_PATH = ARTIFACT_ROOT / "pretrain_selection.json"
SOURCE_ROOT = DATA_MOUNT / "pretrain_v1_20b"
RESULTS_ROOT = RESULTS_MOUNT / VERSION
HOLDOUT_PATH = RESULTS_ROOT / "holdout.json"
RUN_PATH = RESULTS_ROOT / "run.json"

SOURCE_MANIFEST_HASH = (
    "5e2bd529811066c0c9c264eaf39a820f139ad4a4b1e9c9395fca42118e95a275"
)
SELECTION_HASH = (
    "c5440b93bcf6f35db143ff5b3c22ba91b021b3a01e02a4ec17ba2337c8d29823"
)
SOURCE_TOTAL_TOKENS = 53_970_293_905
TRAIN_TARGET_TOKENS = 9_181_735_000
TRAIN_SOURCE_TOKENS = 9_181_735_001

SEQUENCE_LENGTH = 2_048
HOLDOUT_RECORDS = 4_096
HOLDOUT_TARGET_TOKENS = HOLDOUT_RECORDS * SEQUENCE_LENGTH
RECORDS_PER_SHARD = 128
HOLDOUT_SEED = "context2048-heldout-pt-v1-20260813"
EVAL_BATCH_SIZE = 64
ATTENTION_BACKEND = "sdpa"
SHA256_RE = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class CheckpointSpec:
    key: str
    label: str
    subpath: str
    expected_fingerprint: str


CHECKPOINTS: dict[str, CheckpointSpec] = {
    "vocab81_then_sft3": CheckpointSpec(
        key="vocab81_then_sft3",
        label=(
            "81-token pretraining, deterministic expansion to 85 tokens, "
            "then SFT for 3 epochs"
        ),
        subpath=(
            "context2048_vocab_mixing_fp32_master_v13_20260813/"
            "vocab81_then_sft3/sft/final"
        ),
        expected_fingerprint=(
            "350e1eb7dd87e5fb0107437a3ccdb1dc42efdc034edd4cc0b502738c04de7270"
        ),
    ),
    "vocab85_then_sft3": CheckpointSpec(
        key="vocab85_then_sft3",
        label="85-token pretraining, then SFT for 3 epochs",
        subpath=(
            "context2048_vocab_mixing_fp32_master_v13_20260813/"
            "vocab85_then_sft3/sft/final"
        ),
        expected_fingerprint=(
            "0b286a1ad928c1efefb135cdd8d8bf28d867276e28a7dc682ade3684e6ee6c19"
        ),
    ),
    "mixed_sft1": CheckpointSpec(
        key="mixed_sft1",
        label=(
            "85-token uniformly shuffled mixed pretraining plus one "
            "independently placed SFT copy"
        ),
        subpath=(
            "context2048_vocab_mixing_fp32_master_v13_20260813/"
            "mixed_sft1/mixed/final"
        ),
        expected_fingerprint=(
            "e42a2ed9a5e2b0550c5e5e06ef48e4089ff046d4415d2b4c9c28af0745c0c139"
        ),
    ),
    "mixed_sft3": CheckpointSpec(
        key="mixed_sft3",
        label=(
            "85-token uniformly shuffled mixed pretraining plus three "
            "independently shuffled SFT copies"
        ),
        subpath=(
            "context2048_vocab_mixing_fp32_master_v13_20260813/"
            "mixed_sft3/mixed/final"
        ),
        expected_fingerprint=(
            "61193269be0afc01e310705fef7ed071ea8b224da83242db52594279edf32075"
        ),
    ),
}


control_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "numpy==2.2.6"
)
gpu_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-runtime-ubuntu22.04", add_python="3.11"
    )
    .pip_install(
        "numpy==2.2.6",
        "torch==2.9.0",
        "transformers==4.57.0",
        "safetensors==0.6.2",
    )
    .env(
        {
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
)

data_volume = modal.Volume.from_name(
    "rl-reasoning-training-data", create_if_missing=False
)
checkpoint_volume = modal.Volume.from_name(
    "rl-reasoning-checkpoints", create_if_missing=False
)
results_volume = modal.Volume.from_name(
    "chess-rl-eval-results-r6", create_if_missing=False
)
app = modal.App(APP_NAME)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def content_hash(value: Mapping[str, Any], hash_field: str) -> str:
    body = {key: item for key, item in value.items() if key != hash_field}
    return hashlib.sha256(canonical_json(body)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.is_file():
        existing = read_json(path)
        if existing != dict(value):
            raise FileExistsError(f"immutable JSON differs: {path}")
        return
    atomic_json(path, value)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def checkpoint_path(spec: CheckpointSpec) -> Path:
    return CHECKPOINT_MOUNT / spec.subpath


def result_path(key: str) -> Path:
    return RESULTS_ROOT / key / "success.json"


def checkpoint_fingerprint(checkpoint: str | Path) -> str:
    root = Path(checkpoint).resolve(strict=True)
    required = (root / "config.json", root / "interleaved_training_state.json")
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    weights = sorted(root.glob("model*.safetensors"))
    if not weights:
        raise FileNotFoundError(f"missing safetensors weights under {root}")
    files = list(weights)
    for name in (
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "interleaved_training_state.json",
    ):
        candidate = root / name
        if candidate.is_file():
            files.append(candidate)
    for pattern in (
        "tokenizer*",
        "vocab*",
        "merges*",
        "special_tokens_map.json",
        "added_tokens.json",
        "sentencepiece*",
        "spiece*",
    ):
        files.extend(path for path in root.glob(pattern) if path.is_file())
    files = sorted(set(files), key=lambda path: path.relative_to(root).as_posix())
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 << 20), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _checked_self_hash(
    value: Mapping[str, Any], *, schema: str, hash_field: str
) -> None:
    if value.get("schema") != schema:
        raise ValueError(
            f"unexpected schema {value.get('schema')!r}; expected {schema!r}"
        )
    observed = value.get(hash_field)
    expected = content_hash(value, hash_field)
    if observed != expected:
        raise ValueError(f"{hash_field} mismatch: {observed!r} != {expected!r}")


def validate_training_contract(
    source_manifest: Mapping[str, Any], selection: Mapping[str, Any]
) -> None:
    _checked_self_hash(
        source_manifest,
        schema="interleaved-source-shards-v1",
        hash_field="manifest_hash",
    )
    _checked_self_hash(
        selection,
        schema="interleaved-pretrain-selection-v1",
        hash_field="selection_hash",
    )
    if source_manifest.get("manifest_hash") != SOURCE_MANIFEST_HASH:
        raise ValueError("source manifest is not the pinned training corpus")
    if int(source_manifest.get("total_tokens", -1)) != SOURCE_TOTAL_TOKENS:
        raise ValueError("source manifest token total drifted")
    if selection.get("selection_hash") != SELECTION_HASH:
        raise ValueError("pretraining selection hash drifted")
    if selection.get("source_manifest_hash") != SOURCE_MANIFEST_HASH:
        raise ValueError("pretraining selection references another source corpus")
    if int(selection.get("target_tokens", -1)) != TRAIN_TARGET_TOKENS:
        raise ValueError("pretraining target-token count drifted")
    if int(selection.get("source_tokens", -1)) != TRAIN_SOURCE_TOKENS:
        raise ValueError("pretraining source-token count drifted")

    shards = source_manifest.get("shards")
    spans = selection.get("spans")
    if not isinstance(shards, list) or not shards:
        raise ValueError("source manifest lacks shards")
    if not isinstance(spans, list) or not spans:
        raise ValueError("pretraining selection lacks spans")
    by_path = {str(shard["relative_path"]): shard for shard in shards}
    if len(by_path) != len(shards):
        raise ValueError("source shard paths are not unique")
    shard_numbers = [int(shard["shard_number"]) for shard in shards]
    if len(set(shard_numbers)) != len(shard_numbers):
        raise ValueError("source shard numbers are not unique")
    if sum(int(shard["num_tokens"]) for shard in shards) != SOURCE_TOTAL_TOKENS:
        raise ValueError("source shard token accounting drifted")
    selected_ranges: dict[str, list[tuple[int, int]]] = {}
    selected_tokens = 0
    for span in spans:
        relative = str(span["relative_path"])
        shard = by_path.get(relative)
        if shard is None:
            raise ValueError(f"selection references an unknown shard: {relative}")
        start, stop = int(span["start"]), int(span["stop"])
        if not 0 <= start < stop <= int(shard["num_tokens"]):
            raise ValueError(f"invalid selection span: {span}")
        selected_ranges.setdefault(relative, []).append((start, stop))
        selected_tokens += stop - start
    if selected_tokens != TRAIN_SOURCE_TOKENS:
        raise ValueError("selection span accounting drifted")
    for relative, ranges in selected_ranges.items():
        ordered = sorted(ranges)
        if any(left[1] > right[0] for left, right in zip(ordered, ordered[1:])):
            raise ValueError(f"training spans overlap within {relative}")


def plan_holdout(
    source_manifest: Mapping[str, Any],
    selection: Mapping[str, Any],
    *,
    num_records: int = HOLDOUT_RECORDS,
    sequence_length: int = SEQUENCE_LENGTH,
    records_per_shard: int = RECORDS_PER_SHARD,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    validate_training_contract(source_manifest, selection)
    if num_records <= 0 or sequence_length <= 0 or records_per_shard <= 0:
        raise ValueError("holdout sizes must be positive")
    used_paths = {str(span["relative_path"]) for span in selection["spans"]}
    candidates = [
        dict(shard)
        for shard in source_manifest["shards"]
        if str(shard["relative_path"]) not in used_paths
        and int(shard["num_tokens"]) >= sequence_length + 1
    ]
    candidates.sort(
        key=lambda shard: hashlib.sha256(
            (
                f"{HOLDOUT_SEED}\0{int(shard['shard_number'])}\0"
                f"{shard['relative_path']}"
            ).encode("utf-8")
        ).digest()
    )
    selected_shards: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for shard in candidates:
        block_count = (int(shard["num_tokens"]) - 1) // sequence_length
        blocks = list(range(block_count))
        blocks.sort(
            key=lambda block: hashlib.sha256(
                (
                    f"{HOLDOUT_SEED}\0{shard['relative_path']}\0{block}"
                ).encode("utf-8")
            ).digest()
        )
        take = min(records_per_shard, num_records - len(records), len(blocks))
        if take <= 0:
            break
        selected_shards.append(
            {
                "shard_number": int(shard["shard_number"]),
                "relative_path": str(shard["relative_path"]),
                "num_tokens": int(shard["num_tokens"]),
                "dtype": str(shard["dtype"]),
                "byte_size": int(shard["byte_size"]),
            }
        )
        for block in blocks[:take]:
            start = int(block) * sequence_length
            records.append(
                {
                    "ordinal": len(records),
                    "relative_path": str(shard["relative_path"]),
                    "start": start,
                    "stop": start + sequence_length + 1,
                    "target_start": start + 1,
                    "target_stop": start + sequence_length + 1,
                    "target_tokens": sequence_length,
                }
            )
        if len(records) == num_records:
            break
    if len(records) != num_records:
        raise ValueError(f"found only {len(records):,}/{num_records:,} records")
    return selected_shards, records


def token_content_sha256(
    records: list[Mapping[str, Any]], source_root: str | Path
) -> str:
    import numpy as np

    root = Path(source_root)
    arrays: dict[str, Any] = {}
    digest = hashlib.sha256(b"context2048-heldout-token-content-v1\0")
    for expected_ordinal, record in enumerate(records):
        ordinal = int(record["ordinal"])
        if ordinal != expected_ordinal:
            raise ValueError("holdout ordinals are not contiguous")
        relative = str(record["relative_path"])
        array = arrays.get(relative)
        if array is None:
            array = np.load(root / relative, mmap_mode="r", allow_pickle=False)
            if array.ndim != 1:
                raise ValueError(f"source shard is not one-dimensional: {relative}")
            arrays[relative] = array
        start, stop = int(record["start"]), int(record["stop"])
        row = np.asarray(array[start:stop], dtype="<i8")
        if row.shape != (stop - start,):
            raise ValueError(f"short held-out record: {record}")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(struct.pack("<QQQ", ordinal, start, stop))
        digest.update(row.tobytes(order="C"))
    return digest.hexdigest()


def _validate_target_ranges(records: list[Mapping[str, Any]]) -> None:
    by_path: dict[str, list[tuple[int, int]]] = {}
    for expected_ordinal, record in enumerate(records):
        if int(record["ordinal"]) != expected_ordinal:
            raise ValueError("holdout ordinals are not contiguous")
        start, stop = int(record["start"]), int(record["stop"])
        target_start = int(record["target_start"])
        target_stop = int(record["target_stop"])
        if target_start != start + 1 or target_stop != stop:
            raise ValueError(f"incorrect target range: {record}")
        if target_stop - target_start != int(record["target_tokens"]):
            raise ValueError(f"incorrect target-token count: {record}")
        by_path.setdefault(str(record["relative_path"]), []).append(
            (target_start, target_stop)
        )
    for relative, ranges in by_path.items():
        ordered = sorted(ranges)
        if any(left[1] > right[0] for left, right in zip(ordered, ordered[1:])):
            raise ValueError(f"held-out target ranges overlap within {relative}")


def build_holdout_manifest(
    source_manifest: Mapping[str, Any],
    selection: Mapping[str, Any],
    *,
    source_root: str | Path = SOURCE_ROOT,
    num_records: int = HOLDOUT_RECORDS,
    sequence_length: int = SEQUENCE_LENGTH,
    records_per_shard: int = RECORDS_PER_SHARD,
) -> dict[str, Any]:
    selected_shards, records = plan_holdout(
        source_manifest,
        selection,
        num_records=num_records,
        sequence_length=sequence_length,
        records_per_shard=records_per_shard,
    )
    root = Path(source_root)
    for shard in selected_shards:
        path = root / str(shard["relative_path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != int(shard["byte_size"]):
            raise ValueError(f"source shard byte size drifted: {path}")
        shard["content_sha256"] = sha256_file(path)
    _validate_target_ranges(records)
    value: dict[str, Any] = {
        "schema": HOLDOUT_SCHEMA,
        "schema_version": 1,
        "version": VERSION,
        "algorithm": "hash-ranked-wholly-unselected-shards-v1",
        "seed": HOLDOUT_SEED,
        "sequence_length": int(sequence_length),
        "num_records": int(num_records),
        "target_tokens": int(num_records) * int(sequence_length),
        "source_manifest_hash": SOURCE_MANIFEST_HASH,
        "selection_hash": SELECTION_HASH,
        "non_overlap_proof": (
            "every held-out relative_path is absent from every training span; "
            "held-out target ranges are pairwise non-overlapping within each shard"
        ),
        "shards": selected_shards,
        "records": records,
        "token_content_sha256": token_content_sha256(records, root),
    }
    value["holdout_hash"] = content_hash(value, "holdout_hash")
    return value


def validate_holdout_identity(holdout: Mapping[str, Any]) -> None:
    """Validate the persisted contract without reading the source volume."""

    _checked_self_hash(holdout, schema=HOLDOUT_SCHEMA, hash_field="holdout_hash")
    if holdout.get("version") != VERSION:
        raise ValueError("holdout version drifted")
    if holdout.get("source_manifest_hash") != SOURCE_MANIFEST_HASH:
        raise ValueError("holdout source-manifest identity drifted")
    if holdout.get("selection_hash") != SELECTION_HASH:
        raise ValueError("holdout selection identity drifted")
    if int(holdout.get("sequence_length", -1)) != SEQUENCE_LENGTH:
        raise ValueError("holdout sequence length drifted")
    if int(holdout.get("num_records", -1)) != HOLDOUT_RECORDS:
        raise ValueError("holdout record count drifted")
    if int(holdout.get("target_tokens", -1)) != HOLDOUT_TARGET_TOKENS:
        raise ValueError("holdout target-token accounting drifted")
    shards = holdout.get("shards")
    records = holdout.get("records")
    if not isinstance(shards, list) or not shards:
        raise ValueError("holdout lacks selected shards")
    if not isinstance(records, list) or len(records) != HOLDOUT_RECORDS:
        raise ValueError("holdout record inventory drifted")
    _validate_target_ranges(records)
    token_hash = str(holdout.get("token_content_sha256", ""))
    if not SHA256_RE.fullmatch(token_hash):
        raise ValueError("holdout lacks an exact token-content hash")


def validate_holdout_manifest(
    holdout: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    selection: Mapping[str, Any],
    *,
    source_root: str | Path = SOURCE_ROOT,
    verify_shard_hashes: bool = True,
    verify_token_content: bool = True,
) -> None:
    validate_holdout_identity(holdout)
    sequence_length = int(holdout.get("sequence_length", -1))
    num_records = int(holdout.get("num_records", -1))
    expected_shards, expected_records = plan_holdout(
        source_manifest,
        selection,
        num_records=num_records,
        sequence_length=sequence_length,
        records_per_shard=RECORDS_PER_SHARD,
    )
    actual_shards = list(holdout.get("shards", ()))
    comparable = [
        {key: item for key, item in shard.items() if key != "content_sha256"}
        for shard in actual_shards
    ]
    if comparable != expected_shards:
        raise ValueError("held-out shard selection drifted")
    records = list(holdout.get("records", ()))
    if records != expected_records:
        raise ValueError("held-out record selection drifted")
    _validate_target_ranges(records)
    used_paths = {str(span["relative_path"]) for span in selection["spans"]}
    root = Path(source_root)
    for shard in actual_shards:
        relative = str(shard["relative_path"])
        if relative in used_paths:
            raise ValueError(f"held-out shard overlaps training: {relative}")
        digest = str(shard.get("content_sha256", ""))
        if not SHA256_RE.fullmatch(digest):
            raise ValueError(f"held-out shard lacks a content hash: {relative}")
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != int(shard["byte_size"]):
            raise ValueError(f"held-out shard byte size drifted: {relative}")
        if verify_shard_hashes and sha256_file(path) != digest:
            raise ValueError(f"held-out shard content drifted: {relative}")
    token_hash = str(holdout["token_content_sha256"])
    if verify_token_content and token_content_sha256(records, root) != token_hash:
        raise ValueError("held-out token content drifted")


def safe_perplexity(loss: float) -> float:
    if not math.isfinite(loss) or loss < 0:
        raise ValueError(f"invalid cross-entropy loss: {loss}")
    try:
        value = math.exp(loss)
    except OverflowError:
        return math.inf
    return value


def validate_result(
    value: Mapping[str, Any],
    *,
    key: str,
    holdout_hash: str,
) -> None:
    spec = CHECKPOINTS[key]
    _checked_self_hash(value, schema=RESULT_SCHEMA, hash_field="result_sha256")
    if value.get("version") != VERSION:
        raise ValueError("result version drifted")
    if value.get("checkpoint") != key:
        raise ValueError("result checkpoint key drifted")
    if value.get("checkpoint_path") != str(checkpoint_path(spec)):
        raise ValueError("result checkpoint path drifted")
    if value.get("checkpoint_fingerprint") != spec.expected_fingerprint:
        raise ValueError("result checkpoint fingerprint drifted")
    if value.get("holdout_hash") != holdout_hash:
        raise ValueError("result holdout hash drifted")
    if not isinstance(value.get("finished_at"), str):
        raise ValueError("result lacks finished_at")
    metrics = value.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("result lacks metrics")
    required = {
        "heldout_pretrain_loss",
        "heldout_pretrain_perplexity",
        "heldout_pretrain_token_accuracy",
        "heldout_pretrain_correct_tokens",
        "heldout_pretrain_target_tokens",
    }
    if set(metrics) != required:
        raise ValueError("result metric fields drifted")
    loss = float(metrics["heldout_pretrain_loss"])
    perplexity = float(metrics["heldout_pretrain_perplexity"])
    accuracy = float(metrics["heldout_pretrain_token_accuracy"])
    correct = int(metrics["heldout_pretrain_correct_tokens"])
    targets = int(metrics["heldout_pretrain_target_tokens"])
    if not all(math.isfinite(item) for item in (loss, perplexity, accuracy)):
        raise ValueError("result contains non-finite metrics")
    if targets != HOLDOUT_TARGET_TOKENS or not 0 <= correct <= targets:
        raise ValueError("result token accounting drifted")
    if not math.isclose(accuracy, correct / targets, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("result token accuracy is inconsistent")
    if not math.isclose(perplexity, safe_perplexity(loss), rel_tol=1e-12):
        raise ValueError("result perplexity is inconsistent")


@app.function(
    image=control_image,
    cpu=8.0,
    memory=32 * 1024,
    timeout=4 * 60 * 60,
    retries=modal.Retries(initial_delay=5.0, max_retries=1),
    volumes={
        str(DATA_MOUNT): data_volume,
        str(RESULTS_MOUNT): results_volume,
    },
)
def prepare_holdout() -> dict[str, Any]:
    data_volume.reload()
    results_volume.reload()
    source_manifest = read_json(SOURCE_MANIFEST_PATH)
    selection = read_json(SELECTION_PATH)
    validate_training_contract(source_manifest, selection)
    if HOLDOUT_PATH.is_file():
        holdout = read_json(HOLDOUT_PATH)
        validate_holdout_manifest(
            holdout,
            source_manifest,
            selection,
            source_root=SOURCE_ROOT,
            verify_shard_hashes=True,
            verify_token_content=True,
        )
        return holdout
    holdout = build_holdout_manifest(
        source_manifest, selection, source_root=SOURCE_ROOT
    )
    validate_holdout_manifest(
        holdout,
        source_manifest,
        selection,
        source_root=SOURCE_ROOT,
        verify_shard_hashes=True,
        verify_token_content=True,
    )
    immutable_json(HOLDOUT_PATH, holdout)
    results_volume.commit()
    return holdout


@app.function(
    image=gpu_image,
    gpu="H200",
    cpu=16.0,
    memory=128 * 1024,
    timeout=6 * 60 * 60,
    retries=modal.Retries(initial_delay=10.0, max_retries=1),
    max_containers=4,
    volumes={
        str(DATA_MOUNT): data_volume,
        str(CHECKPOINT_MOUNT): checkpoint_volume,
        str(RESULTS_MOUNT): results_volume,
    },
)
def eval_checkpoint(key: str, expected_holdout_hash: str) -> dict[str, Any]:
    import numpy as np
    import torch
    import torch.nn.functional as F
    import transformers
    from transformers import AutoModelForCausalLM

    if key not in CHECKPOINTS:
        raise ValueError(f"unknown checkpoint key: {key}")
    started_at = utc_now()
    data_volume.reload()
    checkpoint_volume.reload()
    results_volume.reload()
    source_manifest = read_json(SOURCE_MANIFEST_PATH)
    selection = read_json(SELECTION_PATH)
    holdout = read_json(HOLDOUT_PATH)
    validate_holdout_manifest(
        holdout,
        source_manifest,
        selection,
        source_root=SOURCE_ROOT,
        verify_shard_hashes=False,
        verify_token_content=True,
    )
    if holdout["holdout_hash"] != expected_holdout_hash:
        raise ValueError("controller and worker holdout hashes differ")

    spec = CHECKPOINTS[key]
    checkpoint = checkpoint_path(spec)
    observed_fingerprint = checkpoint_fingerprint(checkpoint)
    if observed_fingerprint != spec.expected_fingerprint:
        raise ValueError(
            f"checkpoint fingerprint drifted for {key}: "
            f"{observed_fingerprint} != {spec.expected_fingerprint}"
        )
    config = read_json(checkpoint / "config.json")
    if int(config.get("max_position_embeddings", -1)) != SEQUENCE_LENGTH:
        raise ValueError(f"checkpoint context length drifted for {key}")
    vocab_size = int(config.get("vocab_size", -1))
    if vocab_size != 85:
        raise ValueError(f"checkpoint vocabulary size drifted for {key}")

    success_path = result_path(key)
    if success_path.is_file():
        success = read_json(success_path)
        validate_result(success, key=key, holdout_hash=expected_holdout_hash)
        return success

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    model = AutoModelForCausalLM.from_pretrained(
        str(checkpoint),
        torch_dtype=torch.bfloat16,
        attn_implementation=ATTENTION_BACKEND,
    ).to("cuda")
    model.eval()

    arrays: dict[str, Any] = {}
    total_cross_entropy = 0.0
    total_correct = 0
    total_targets = 0
    records = list(holdout["records"])
    with torch.inference_mode():
        for batch_start in range(0, len(records), EVAL_BATCH_SIZE):
            batch_records = records[batch_start : batch_start + EVAL_BATCH_SIZE]
            rows = []
            for record in batch_records:
                relative = str(record["relative_path"])
                array = arrays.get(relative)
                if array is None:
                    array = np.load(
                        SOURCE_ROOT / relative,
                        mmap_mode="r",
                        allow_pickle=False,
                    )
                    arrays[relative] = array
                start, stop = int(record["start"]), int(record["stop"])
                row = np.asarray(array[start:stop], dtype=np.int64)
                if row.shape != (SEQUENCE_LENGTH + 1,):
                    raise ValueError(f"short held-out record: {record}")
                if int(row.min()) < 0 or int(row.max()) >= vocab_size:
                    raise ValueError(f"token ID outside checkpoint vocabulary: {record}")
                rows.append(row)
            raw = torch.from_numpy(np.stack(rows)).to("cuda", non_blocking=True)
            input_ids = raw[:, :-1]
            labels = raw[:, 1:]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(input_ids=input_ids, use_cache=False).logits
            token_losses = F.cross_entropy(
                logits.float().reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                reduction="none",
            )
            total_cross_entropy += float(token_losses.double().sum().item())
            total_correct += int(logits.argmax(dim=-1).eq(labels).sum().item())
            total_targets += int(labels.numel())
    if total_targets != HOLDOUT_TARGET_TOKENS:
        raise ValueError(
            f"evaluated target count drifted: {total_targets} != "
            f"{HOLDOUT_TARGET_TOKENS}"
        )
    loss = total_cross_entropy / total_targets
    payload: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "schema_version": 1,
        "version": VERSION,
        "state": "complete",
        "checkpoint": key,
        "checkpoint_label": spec.label,
        "checkpoint_path": str(checkpoint),
        "checkpoint_fingerprint": observed_fingerprint,
        "holdout_hash": expected_holdout_hash,
        "metrics": {
            "heldout_pretrain_loss": loss,
            "heldout_pretrain_perplexity": safe_perplexity(loss),
            "heldout_pretrain_token_accuracy": total_correct / total_targets,
            "heldout_pretrain_correct_tokens": total_correct,
            "heldout_pretrain_target_tokens": total_targets,
        },
        "runtime": {
            "gpu": torch.cuda.get_device_name(0),
            "dtype": "bfloat16",
            "attention_backend": ATTENTION_BACKEND,
            "batch_size": EVAL_BATCH_SIZE,
            "deterministic_algorithms": True,
            "torch": str(torch.__version__),
            "transformers": str(transformers.__version__),
        },
        "started_at": started_at,
        "finished_at": utc_now(),
    }
    payload["result_sha256"] = content_hash(payload, "result_sha256")
    validate_result(payload, key=key, holdout_hash=expected_holdout_hash)
    immutable_json(success_path, payload)
    results_volume.commit()
    print(
        f"[success] {key}: loss={loss:.8f}, "
        f"ppl={payload['metrics']['heldout_pretrain_perplexity']:.8f}, "
        f"accuracy={payload['metrics']['heldout_pretrain_token_accuracy']:.8f}",
        flush=True,
    )
    # Keep the Modal return value free of library-specific string subclasses.
    # The controller image intentionally does not install torch.
    return json.loads(json.dumps(payload, allow_nan=False))


def complete_run_ledger(
    ledger: Mapping[str, Any],
    results: Mapping[str, Mapping[str, Any]],
    *,
    finished_at: str | None = None,
) -> dict[str, Any]:
    if set(results) != set(CHECKPOINTS):
        raise ValueError("run completion requires all checkpoint results")
    completed = dict(ledger)
    completed["state"] = "complete"
    completed["finished_at"] = finished_at or utc_now()
    completed["results"] = {
        key: {
            "result_sha256": result["result_sha256"],
            "metrics": result["metrics"],
        }
        for key, result in results.items()
    }
    completed["run_sha256"] = content_hash(completed, "run_sha256")
    return completed


def read_completed_results(holdout_hash: str) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for key in CHECKPOINTS:
        path = result_path(key)
        if not path.is_file():
            raise FileNotFoundError(path)
        result = read_json(path)
        validate_result(result, key=key, holdout_hash=holdout_hash)
        results[key] = result
    return results


@app.function(
    image=control_image,
    cpu=4.0,
    memory=8 * 1024,
    timeout=8 * 60 * 60,
    retries=0,
    volumes={str(RESULTS_MOUNT): results_volume},
)
def run_all() -> dict[str, Any]:
    results_volume.reload()
    if RUN_PATH.is_file():
        existing = read_json(RUN_PATH)
        _checked_self_hash(existing, schema=RUN_SCHEMA, hash_field="run_sha256")
        holdout = read_json(HOLDOUT_PATH)
        validate_holdout_identity(holdout)
        if existing.get("state") == "complete":
            results = read_completed_results(holdout["holdout_hash"])
            for key, result in results.items():
                if (
                    existing.get("results", {})
                    .get(key, {})
                    .get("result_sha256")
                    != result["result_sha256"]
                ):
                    raise ValueError(f"run/result identity mismatch for {key}")
            return existing
        if existing.get("state") != "evaluating":
            raise ValueError(f"unexpected run state: {existing.get('state')!r}")
        # A controller can terminate after all immutable worker markers are
        # committed (for example, while deserializing a worker return value).
        # Finalize from the authenticated markers without spawning duplicates.
        try:
            results = read_completed_results(holdout["holdout_hash"])
        except FileNotFoundError as exc:
            raise FileExistsError(
                "an incomplete held-out-loss controller already exists"
            ) from exc
        completed = complete_run_ledger(existing, results)
        atomic_json(RUN_PATH, completed)
        results_volume.commit()
        return completed

    holdout = prepare_holdout.remote()
    calls: dict[str, Any] = {}
    ledger: dict[str, Any] = {
        "schema": RUN_SCHEMA,
        "schema_version": 1,
        "version": VERSION,
        "state": "evaluating",
        "holdout_hash": holdout["holdout_hash"],
        "started_at": utc_now(),
        "calls": {},
    }
    for key in CHECKPOINTS:
        call = eval_checkpoint.spawn(key, holdout["holdout_hash"])
        calls[key] = call
        ledger["calls"][key] = call.object_id
    ledger["run_sha256"] = content_hash(ledger, "run_sha256")
    immutable_json(RUN_PATH, ledger)
    results_volume.commit()

    for call in calls.values():
        call.get()
    results_volume.reload()
    results = read_completed_results(holdout["holdout_hash"])
    completed = complete_run_ledger(ledger, results)
    atomic_json(RUN_PATH, completed)
    results_volume.commit()
    return completed


@app.function(
    image=control_image,
    cpu=2.0,
    memory=4 * 1024,
    timeout=10 * 60,
    volumes={str(RESULTS_MOUNT): results_volume},
)
def remote_status() -> dict[str, Any]:
    results_volume.reload()
    holdout = read_json(HOLDOUT_PATH) if HOLDOUT_PATH.is_file() else None
    if holdout is not None:
        validate_holdout_identity(holdout)
    holdout_hash = str(holdout.get("holdout_hash")) if holdout else None
    run = read_json(RUN_PATH) if RUN_PATH.is_file() else None
    if run is not None:
        _checked_self_hash(run, schema=RUN_SCHEMA, hash_field="run_sha256")
    checkpoints: dict[str, Any] = {}
    for key in CHECKPOINTS:
        path = result_path(key)
        result = read_json(path) if path.is_file() else None
        if result is not None:
            if holdout_hash is None:
                raise ValueError("result exists without a holdout manifest")
            validate_result(result, key=key, holdout_hash=holdout_hash)
        checkpoints[key] = result
    return {
        "version": VERSION,
        "holdout": (
            {
                "path": str(HOLDOUT_PATH),
                "holdout_hash": holdout_hash,
                "records": holdout["num_records"],
                "target_tokens": holdout["target_tokens"],
                "token_content_sha256": holdout["token_content_sha256"],
            }
            if holdout
            else None
        ),
        "run": run,
        "checkpoints": checkpoints,
    }


@app.local_entrypoint()
def main(action: str = "status") -> None:
    normalized = action.strip().lower()
    if normalized == "launch":
        call = run_all.spawn()
        print(
            json.dumps(
                {
                    "version": VERSION,
                    "run_call_id": call.object_id,
                    "checkpoints": list(CHECKPOINTS),
                    "results_root": str(RESULTS_ROOT),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if normalized == "status":
        print(json.dumps(remote_status.remote(), indent=2, sort_keys=True))
        return
    raise ValueError("action must be launch or status")
