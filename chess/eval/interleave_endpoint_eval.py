"""Fail-closed identities and discovery for interleaved pretrain endpoints.

The endpoint evaluator deliberately keeps three different claims separate:

* pretraining loss is measured on a fixed set of records cut only from source
  shards that are wholly absent from the 10B-token training selection;
* masked-SFT loss is unavailable because P1 and P2 jointly consumed every
  pinned SFT row exactly once; and
* chess behavior is measured by the existing immutable B1--B5 suite.

This module has no Modal dependency.  It is imported by the Modal workers and
locally unit tested so dataset and checkpoint identities cannot silently drift.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


EXPERIMENT_VERSION = "mix10b_sft90k_3072_v1_20260730"
EXP4_VERSION = "positive-rollout-transfer-v1-20260730"
ENDPOINT_NAMESPACE = "endpoint_v1"
V2_EXPERIMENT_VERSION = (
    "mix10b_sft90k_3072_v2r1_weighted_clean_20260730"
)
V2_DATA_ARTIFACT_VERSION = "mix10b_sft90k_v2r1_clean_verify_gate"
V2_SFT_RESPONSE_NORMALIZATION = (
    "strip-numeric-verify-score-pairs-normalize-whitespace-v1"
)
V2_SFT_SUPERVISED_UNK_POLICY = "reject-supervised-unk-v1"
V2_SFT_STRICT_AUDIT_SCHEMA = "interleaved-sft-strict-audit-v1"
V2_SFT_SUPERVISED_TARGETS = 52_482_753
V2_P1_SFT_SUPERVISED_TARGETS = 26_289_598
V2_P2_SFT_SUPERVISED_TARGETS = 26_193_155
V2_END_THINKING_TARGETS = 77_717
V2_CALL_ENV_TARGETS = 187_354

SOURCE_REPO = "chess-pre-to-post/pretrain_v1_20b"
SOURCE_REVISION = "07dd1b7090ca5f0fb05ef624c26b20bff19483c8"
SOURCE_MANIFEST_HASH = (
    "5e2bd529811066c0c9c264eaf39a820f139ad4a4b1e9c9395fca42118e95a275"
)
SOURCE_FLAT_MANIFEST_SHA256 = (
    "07ae91cded540a00e9b6554d1d54ed46310715b7fd68e3520a64b7f5967f99aa"
)
SOURCE_TOTAL_TOKENS = 53_970_293_905
TRAIN_SELECTION_HASH = (
    "e80f95f89ee2b6a872157cede635f4f130ebb91685f08283774526ad4562ac00"
)
TRAIN_SELECTION_TOKENS = 10_000_000_001

SFT_REPO = "chess-pre-to-post/sft_v1_200m_90k"
SFT_REVISION = "97f60746dd253b4e130beeb5e66f39e9d42ef25c"
SFT_CACHE_HASH = (
    "b66cfcd8001af1f900c660594f46c4e03d622afd1ecf25b9d29c715fe0f343f6"
)
SFT_ROWS_SHA256 = (
    "10226961f9fc59c6b7c611fc27fab8651d7a7dae9ffdc351b2ef5016617939cc"
)
SFT_INPUT_IDS_SHA256 = (
    "807d52c3634444e47b765dbd22b560e5f1a0120c92be4b22d2f93c11decaa57b"
)
SFT_LABELS_SHA256 = (
    "1d9dfe6b247f7abfc7b836a9435fd1fb7439f694202fb75b0273c6960eb535e1"
)
SFT_OFFSETS_SHA256 = (
    "8ed790717d153e1ca944a96beb598d78ad300e838394afdba46f4b71beecaf62"
)
SFT_ROWS = 77_717
P1_SFT_ROWS = 38_858
P2_SFT_ROWS = 38_859
P1_ORDER_SHA256 = (
    "68fc5a3934ea677f31365d998f67380e7b5d2fa12f7b5bbbed9756a3f8bd9ac4"
)
P2_ORDER_SHA256 = (
    "95d70dfc4474be1b8b301196875db511f9c9d63036332ca6913dd0504d7c17b8"
)

PT_HOLDOUT_SCHEMA = "interleaved-pt-heldout-v1"
SFT_AUDIT_SCHEMA = "interleaved-sft-heldout-audit-v1"
ENDPOINT_RESULT_SCHEMA = "interleaved-endpoint-result-v1"
SEQUENCE_LENGTH = 3_072
PT_HOLDOUT_RECORDS = 4_096
PT_HOLDOUT_TARGET_TOKENS = PT_HOLDOUT_RECORDS * SEQUENCE_LENGTH
PT_RECORDS_PER_SHARD = 128
PT_HOLDOUT_SEED = "interleaved-endpoint-pt-heldout-v1"
PAD_RECORD = np.iinfo(np.int64).min

CHESS_DATA_SHA256 = {
    "test_B1_multi_turn": (
        "3ac5df0af21b395c23f864dd75b6a64335e3fe681c2b774f1485b276c6893c78"
    ),
    "test_B2_multi_turn": (
        "9b315fe82a676b9b817ae77f96f7987be04ab34ec18513e3d42544896a133c3f"
    ),
    "test_B3_multi_turn": (
        "8e41e0cf7c17babf6ae9a17a5b51607eef5674788dd09042e7dbbf90a945a5b9"
    ),
    "test_B4_multi_turn": (
        "9583e4f6621ffee456eefc3e9d9de15800ec24226d20b882ff4805e82c4a985b"
    ),
    "test_B5_multi_turn": (
        "927d62a4994d39e61ffb6719f85961ba14dbd55f365c539477fe6db72288c5cc"
    ),
}

EXPECTED_MODEL_CONFIG = {
    "model_type": "qwen3",
    "vocab_size": 85,
    "max_position_embeddings": SEQUENCE_LENGTH,
    "hidden_size": 512,
    "head_dim": 128,
    "num_hidden_layers": 12,
    "num_attention_heads": 8,
    "num_key_value_heads": 4,
    "intermediate_size": 1536,
    "tie_word_embeddings": True,
}

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_VERSION_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,191}")


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def content_hash(value: Mapping[str, Any], field: str) -> str:
    unhashed = {key: item for key, item in value.items() if key != field}
    return hashlib.sha256(canonical_json(unhashed)).hexdigest()


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _checked_object(
    value: Mapping[str, Any],
    *,
    schema: str,
    hash_field: str,
) -> None:
    if value.get("schema") != schema:
        raise ValueError(
            f"unexpected schema {value.get('schema')!r}; expected {schema!r}"
        )
    expected = value.get(hash_field)
    actual = content_hash(value, hash_field)
    if not isinstance(expected, str) or expected != actual:
        raise ValueError(f"{hash_field} mismatch: {expected!r} != {actual!r}")


def _validate_training_manifests(
    source_manifest: Mapping[str, Any],
    training_selection: Mapping[str, Any],
) -> None:
    _checked_object(
        source_manifest,
        schema="interleaved-source-shards-v1",
        hash_field="manifest_hash",
    )
    _checked_object(
        training_selection,
        schema="interleaved-pretrain-selection-v1",
        hash_field="selection_hash",
    )
    if source_manifest.get("manifest_hash") != SOURCE_MANIFEST_HASH:
        raise ValueError("source manifest is not the pinned training corpus")
    if int(source_manifest.get("total_tokens", -1)) != SOURCE_TOTAL_TOKENS:
        raise ValueError("source corpus token total drifted")
    if training_selection.get("selection_hash") != TRAIN_SELECTION_HASH:
        raise ValueError("training selection is not the pinned 10B selection")
    if training_selection.get("source_manifest_hash") != SOURCE_MANIFEST_HASH:
        raise ValueError("training selection references a different source corpus")
    if int(training_selection.get("source_tokens", -1)) != TRAIN_SELECTION_TOKENS:
        raise ValueError("training selection token total drifted")


def plan_pt_holdout(
    source_manifest: Mapping[str, Any],
    training_selection: Mapping[str, Any],
    *,
    num_records: int = PT_HOLDOUT_RECORDS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select exact records from whole shards absent from every train span."""

    _validate_training_manifests(source_manifest, training_selection)
    if int(num_records) <= 0:
        raise ValueError("num_records must be positive")
    used_paths = {
        str(span["relative_path"]) for span in training_selection["spans"]
    }
    shards = source_manifest.get("shards")
    if not isinstance(shards, list):
        raise ValueError("source manifest lacks a shard list")
    candidates = [
        dict(shard)
        for shard in shards
        if str(shard["relative_path"]) not in used_paths
        and int(shard["num_tokens"]) >= SEQUENCE_LENGTH + 1
    ]
    candidates.sort(
        key=lambda shard: hashlib.sha256(
            (
                f"{PT_HOLDOUT_SEED}\0{int(shard['shard_number'])}\0"
                f"{shard['relative_path']}"
            ).encode()
        ).digest()
    )
    selected_shards: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for shard in candidates:
        block_count = (int(shard["num_tokens"]) - 1) // SEQUENCE_LENGTH
        blocks = list(range(block_count))
        blocks.sort(
            key=lambda block: hashlib.sha256(
                (
                    f"{PT_HOLDOUT_SEED}\0{shard['relative_path']}\0{block}"
                ).encode()
            ).digest()
        )
        take = min(
            PT_RECORDS_PER_SHARD,
            int(num_records) - len(records),
            len(blocks),
        )
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
            start = int(block) * SEQUENCE_LENGTH
            records.append(
                {
                    "ordinal": len(records),
                    "relative_path": str(shard["relative_path"]),
                    "start": start,
                    "stop": start + SEQUENCE_LENGTH + 1,
                    "target_tokens": SEQUENCE_LENGTH,
                }
            )
        if len(records) == int(num_records):
            break
    if len(records) != int(num_records):
        raise ValueError(
            f"only found {len(records):,}/{int(num_records):,} held-out records"
        )
    return selected_shards, records


def build_pt_holdout_manifest(
    source_manifest: Mapping[str, Any],
    training_selection: Mapping[str, Any],
    *,
    shard_sha256: Callable[[str], str],
    num_records: int = PT_HOLDOUT_RECORDS,
) -> dict[str, Any]:
    selected_shards, records = plan_pt_holdout(
        source_manifest, training_selection, num_records=num_records
    )
    for shard in selected_shards:
        digest = shard_sha256(str(shard["relative_path"]))
        if not _SHA256_RE.fullmatch(str(digest)):
            raise ValueError(
                f"invalid content hash for {shard['relative_path']}: {digest!r}"
            )
        shard["content_sha256"] = str(digest)
    value: dict[str, Any] = {
        "schema": PT_HOLDOUT_SCHEMA,
        "schema_version": 1,
        "algorithm": "hash-ranked-wholly-unselected-shards-v1",
        "seed": PT_HOLDOUT_SEED,
        "sequence_length": SEQUENCE_LENGTH,
        "num_records": int(num_records),
        "target_tokens": int(num_records) * SEQUENCE_LENGTH,
        "source": {
            "repo": SOURCE_REPO,
            "revision": SOURCE_REVISION,
            "manifest_hash": SOURCE_MANIFEST_HASH,
            "flat_name_size_sha256": SOURCE_FLAT_MANIFEST_SHA256,
            "total_tokens": SOURCE_TOTAL_TOKENS,
        },
        "non_overlap_proof": {
            "training_selection_hash": TRAIN_SELECTION_HASH,
            "training_source_tokens": TRAIN_SELECTION_TOKENS,
            "rule": "every evaluation relative_path is absent from every training span",
        },
        "shards": selected_shards,
        "records": records,
    }
    value["holdout_hash"] = content_hash(value, "holdout_hash")
    return value


def validate_pt_holdout_manifest(
    holdout: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    training_selection: Mapping[str, Any],
    *,
    source_root: Path | None = None,
) -> None:
    _checked_object(
        holdout, schema=PT_HOLDOUT_SCHEMA, hash_field="holdout_hash"
    )
    expected_shards, expected_records = plan_pt_holdout(
        source_manifest,
        training_selection,
        num_records=int(holdout.get("num_records", -1)),
    )
    if int(holdout.get("sequence_length", -1)) != SEQUENCE_LENGTH:
        raise ValueError("held-out sequence length drifted")
    if int(holdout.get("target_tokens", -1)) != (
        int(holdout["num_records"]) * SEQUENCE_LENGTH
    ):
        raise ValueError("held-out target-token accounting drifted")
    if list(holdout.get("records", ())) != expected_records:
        raise ValueError("held-out record selection drifted")

    actual_shards = list(holdout.get("shards", ()))
    comparable_shards = [
        {key: value for key, value in shard.items() if key != "content_sha256"}
        for shard in actual_shards
    ]
    if comparable_shards != expected_shards:
        raise ValueError("held-out shard selection drifted")
    if len({record["ordinal"] for record in expected_records}) != len(
        expected_records
    ):
        raise ValueError("held-out record ordinals are not unique")
    if [record["ordinal"] for record in expected_records] != list(
        range(len(expected_records))
    ):
        raise ValueError("held-out record ordinals are not contiguous")

    used_paths = {
        str(span["relative_path"]) for span in training_selection["spans"]
    }
    for shard in actual_shards:
        relative = str(shard["relative_path"])
        if relative in used_paths:
            raise ValueError(f"held-out shard overlaps training: {relative}")
        digest = shard.get("content_sha256")
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
            raise ValueError(f"held-out shard lacks a content hash: {relative}")
        if source_root is not None:
            path = source_root / relative
            if not path.is_file():
                raise FileNotFoundError(path)
            if path.stat().st_size != int(shard["byte_size"]):
                raise ValueError(f"held-out shard byte size drifted: {relative}")
            if sha256_file(path) != digest:
                raise ValueError(f"held-out shard content drifted: {relative}")

    by_path: dict[str, list[tuple[int, int]]] = {}
    for record in expected_records:
        by_path.setdefault(str(record["relative_path"]), []).append(
            (int(record["start"]) + 1, int(record["stop"]))
        )
    for relative, target_ranges in by_path.items():
        ordered = sorted(target_ranges)
        if any(left[1] > right[0] for left, right in zip(ordered, ordered[1:])):
            raise ValueError(f"held-out target ranges overlap within {relative}")


def build_sft_holdout_audit(
    manifest_set: Mapping[str, Any],
    sft_cache: Mapping[str, Any],
    p1_metadata: Mapping[str, Any],
    p2_metadata: Mapping[str, Any],
    p1_order: np.ndarray,
    p2_order: np.ndarray,
) -> dict[str, Any]:
    """Prove that no same-source SFT row remains genuinely held out."""

    expected_manifest_set_hash = (
        "fc1721b9562b6b66048e3b2b085c9247e21006362ce46efe5b4714f4a8991ba8"
    )
    if manifest_set.get("manifest_set_hash") != expected_manifest_set_hash:
        raise ValueError("unexpected mixed-data manifest set")
    if (
        manifest_set.get("sft_cache_hash") != SFT_CACHE_HASH
        or manifest_set.get("sft_repo") != SFT_REPO
        or manifest_set.get("sft_revision") != SFT_REVISION
        or int(manifest_set.get("sft_rows", -1)) != SFT_ROWS
    ):
        raise ValueError("SFT source identity drifted")
    if (
        sft_cache.get("cache_hash") != SFT_CACHE_HASH
        or sft_cache.get("rows_sha256") != SFT_ROWS_SHA256
        or sft_cache.get("input_ids_sha256") != SFT_INPUT_IDS_SHA256
        or sft_cache.get("labels_sha256") != SFT_LABELS_SHA256
        or sft_cache.get("offsets_sha256") != SFT_OFFSETS_SHA256
        or int(sft_cache.get("num_rows", -1)) != SFT_ROWS
    ):
        raise ValueError("SFT cache identity drifted")
    expected_legs = (
        ("p1", P1_ORDER_SHA256, P1_SFT_ROWS),
        ("p2", P2_ORDER_SHA256, P2_SFT_ROWS),
    )
    indices: list[np.ndarray] = []
    for metadata, order, (leg, order_hash, expected_rows) in zip(
        (p1_metadata, p2_metadata),
        (p1_order, p2_order),
        expected_legs,
    ):
        if (
            metadata.get("leg") != leg
            or metadata.get("order_sha256") != order_hash
            or int(metadata.get("sft_records", -1)) != expected_rows
        ):
            raise ValueError(f"{leg} SFT manifest identity drifted")
        values = np.asarray(order, dtype=np.int64)
        codes = values[(values < 0) & (values != PAD_RECORD)]
        row_ids = -codes - 1
        if len(row_ids) != expected_rows or len(np.unique(row_ids)) != expected_rows:
            raise ValueError(f"{leg} does not contain its expected unique SFT rows")
        indices.append(row_ids)
    if len(np.intersect1d(indices[0], indices[1])) != 0:
        raise ValueError("P1 and P2 SFT rows overlap")
    union = np.sort(np.concatenate(indices))
    if not np.array_equal(union, np.arange(SFT_ROWS, dtype=np.int64)):
        raise ValueError("P1/P2 do not cover exactly every pinned SFT row")

    value: dict[str, Any] = {
        "schema": SFT_AUDIT_SCHEMA,
        "schema_version": 1,
        "status": "unavailable_no_heldout",
        "reason": (
            "P1 and P2 consume disjoint row sets whose union is exactly every "
            "pinned SFT row; reporting train-row loss as held-out is forbidden."
        ),
        "metrics": {
            "masked_sft_loss": None,
            "masked_sft_perplexity": None,
            "masked_sft_token_accuracy": None,
        },
        "source": {
            "repo": SFT_REPO,
            "revision": SFT_REVISION,
            "rows": SFT_ROWS,
            "cache_hash": SFT_CACHE_HASH,
            "rows_sha256": SFT_ROWS_SHA256,
            "input_ids_sha256": SFT_INPUT_IDS_SHA256,
            "labels_sha256": SFT_LABELS_SHA256,
            "offsets_sha256": SFT_OFFSETS_SHA256,
        },
        "coverage_proof": {
            "p1_order_sha256": P1_ORDER_SHA256,
            "p1_unique_rows": P1_SFT_ROWS,
            "p2_order_sha256": P2_ORDER_SHA256,
            "p2_unique_rows": P2_SFT_ROWS,
            "intersection_rows": 0,
            "union_rows": SFT_ROWS,
            "union_is_exact_range_0_through_77716": True,
        },
    }
    value["audit_hash"] = content_hash(value, "audit_hash")
    return value


def build_v2_sft_holdout_audit(
    manifest_set: Mapping[str, Any],
    sft_cache: Mapping[str, Any],
    p1_metadata: Mapping[str, Any],
    p2_metadata: Mapping[str, Any],
    p1_order: np.ndarray,
    p2_order: np.ndarray,
    *,
    p1_metadata_file_sha256: str,
    p2_metadata_file_sha256: str,
    p1_order_file_sha256: str,
    p2_order_file_sha256: str,
) -> dict[str, Any]:
    """Authenticate the cleaned v2 SFT corpus and prove zero held-out rows.

    The v2 cache hash was intentionally not preregistered before data
    materialization.  Instead, this proof binds every self-hashed manifest,
    the exact cleaned-data invariants, the metadata file hashes recorded by
    the manifest set, and the exact order-file hashes.  It therefore remains
    fail-closed without copying mutable hashes from a live run into source.
    """

    _checked_object(
        manifest_set,
        schema="interleaved-manifest-set-v1",
        hash_field="manifest_set_hash",
    )
    _checked_object(
        sft_cache,
        schema="interleaved-sft-cache-v1",
        hash_field="cache_hash",
    )
    for leg, metadata in (("p1", p1_metadata), ("p2", p2_metadata)):
        _checked_object(
            metadata,
            schema="interleaved-leg-manifest-v1",
            hash_field="metadata_hash",
        )
        if metadata.get("leg") != leg:
            raise ValueError(f"{leg} leg metadata identity drifted")

    if (
        manifest_set.get("experiment_version") != V2_DATA_ARTIFACT_VERSION
        or manifest_set.get("source_repo") != SOURCE_REPO
        or manifest_set.get("source_revision") != SOURCE_REVISION
        or manifest_set.get("sft_repo") != SFT_REPO
        or manifest_set.get("sft_revision") != SFT_REVISION
        or int(manifest_set.get("pretrain_tokens", -1)) != 10_000_000_000
        or int(manifest_set.get("sft_rows", -1)) != SFT_ROWS
        or manifest_set.get("source_manifest_hash") != SOURCE_MANIFEST_HASH
        or manifest_set.get("selection_hash") != TRAIN_SELECTION_HASH
    ):
        raise ValueError("v2 manifest-set identity drifted")

    cache_hash = str(sft_cache.get("cache_hash", ""))
    if (
        not _SHA256_RE.fullmatch(cache_hash)
        or manifest_set.get("sft_cache_hash") != cache_hash
        or int(sft_cache.get("num_rows", -1)) != SFT_ROWS
        or int(sft_cache.get("sequence_length", -1)) != SEQUENCE_LENGTH
        or sft_cache.get("response_normalization")
        != V2_SFT_RESPONSE_NORMALIZATION
        or sft_cache.get("supervised_unk_policy")
        != V2_SFT_SUPERVISED_UNK_POLICY
        or sft_cache.get("strict_sft_audit_required") is not True
        or int(sft_cache.get("supervised_targets", -1))
        != V2_SFT_SUPERVISED_TARGETS
        or sft_cache.get("supervised_unk_targets") != 0
    ):
        raise ValueError("cleaned v2 SFT cache identity drifted")
    for hash_key in (
        "rows_sha256",
        "input_ids_sha256",
        "labels_sha256",
        "offsets_sha256",
    ):
        if not _SHA256_RE.fullmatch(str(sft_cache.get(hash_key, ""))):
            raise ValueError(f"cleaned v2 SFT cache lacks {hash_key}")

    strict_audit = sft_cache.get("strict_sft_audit")
    delimiters = sft_cache.get("supervised_delimiter_counts")
    if (
        not isinstance(strict_audit, Mapping)
        or strict_audit.get("schema") != V2_SFT_STRICT_AUDIT_SCHEMA
        or strict_audit.get("expected_supervised_targets")
        != V2_SFT_SUPERVISED_TARGETS
        or int(strict_audit.get("t_end_rows_exactly_one", -1)) != SFT_ROWS
        or int(strict_audit.get("call_env_rows_at_least_one", -1))
        != SFT_ROWS
        or not isinstance(delimiters, Mapping)
        or int(delimiters.get("</T>", -1)) != V2_END_THINKING_TARGETS
        or int(delimiters.get("<call_env>", -1)) != V2_CALL_ENV_TARGETS
    ):
        raise ValueError("cleaned v2 strict SFT audit drifted")

    manifest_entries = manifest_set.get("manifests")
    if not isinstance(manifest_entries, Mapping):
        raise ValueError("v2 manifest set lacks manifest entries")
    expected_files = {
        "p1": (
            p1_metadata,
            p1_metadata_file_sha256,
            p1_order_file_sha256,
            P1_SFT_ROWS,
            V2_P1_SFT_SUPERVISED_TARGETS,
        ),
        "p2": (
            p2_metadata,
            p2_metadata_file_sha256,
            p2_order_file_sha256,
            P2_SFT_ROWS,
            V2_P2_SFT_SUPERVISED_TARGETS,
        ),
    }
    row_sets: list[np.ndarray] = []
    for leg, order in (("p1", p1_order), ("p2", p2_order)):
        (
            metadata,
            metadata_file_sha256,
            order_file_sha256,
            expected_rows,
            expected_targets,
        ) = expected_files[leg]
        entry = manifest_entries.get(leg)
        if (
            not isinstance(entry, Mapping)
            or entry.get("path") != f"legs/{leg}/metadata.json"
            or entry.get("sha256") != metadata_file_sha256
            or not _SHA256_RE.fullmatch(metadata_file_sha256)
            or metadata.get("order_sha256") != order_file_sha256
            or not _SHA256_RE.fullmatch(order_file_sha256)
            or metadata.get("source_manifest_hash") != SOURCE_MANIFEST_HASH
            or metadata.get("selection_hash") != TRAIN_SELECTION_HASH
            or metadata.get("sft_cache_hash") != cache_hash
            or int(metadata.get("sft_records", -1)) != expected_rows
            or int(metadata.get("sft_supervised_targets", -1))
            != expected_targets
        ):
            raise ValueError(f"{leg} cleaned v2 leg identity drifted")
        values = np.asarray(order)
        if values.ndim != 1 or values.dtype != np.dtype("int64"):
            raise ValueError(f"{leg} order must be a one-dimensional int64 array")
        if int(metadata.get("num_order_records", -1)) != len(values):
            raise ValueError(f"{leg} order length drifted")
        codes = values[(values < 0) & (values != PAD_RECORD)]
        row_ids = -codes - 1
        if (
            len(row_ids) != expected_rows
            or len(np.unique(row_ids)) != expected_rows
            or bool(np.any(row_ids < 0))
            or bool(np.any(row_ids >= SFT_ROWS))
        ):
            raise ValueError(
                f"{leg} does not contain its expected unique cleaned SFT rows"
            )
        row_sets.append(row_ids)

    if len(np.intersect1d(row_sets[0], row_sets[1])) != 0:
        raise ValueError("P1 and P2 cleaned SFT rows overlap")
    union = np.sort(np.concatenate(row_sets))
    if not np.array_equal(union, np.arange(SFT_ROWS, dtype=np.int64)):
        raise ValueError(
            "P1/P2 do not cover exactly every cleaned pinned SFT row"
        )

    value: dict[str, Any] = {
        "schema": SFT_AUDIT_SCHEMA,
        "schema_version": 2,
        "experiment_version": V2_EXPERIMENT_VERSION,
        "data_artifact_version": V2_DATA_ARTIFACT_VERSION,
        "status": "unavailable_no_heldout",
        "reason": (
            "P1 and P2 consume disjoint row sets whose union is exactly every "
            "cleaned pinned SFT row; reporting train-row loss as held-out is "
            "forbidden."
        ),
        "metrics": {
            "masked_sft_loss": None,
            "masked_sft_perplexity": None,
            "masked_sft_token_accuracy": None,
        },
        "source": {
            "repo": SFT_REPO,
            "revision": SFT_REVISION,
            "rows": SFT_ROWS,
            "manifest_set_hash": manifest_set["manifest_set_hash"],
            "cache_hash": cache_hash,
            "rows_sha256": sft_cache["rows_sha256"],
            "input_ids_sha256": sft_cache["input_ids_sha256"],
            "labels_sha256": sft_cache["labels_sha256"],
            "offsets_sha256": sft_cache["offsets_sha256"],
            "response_normalization": V2_SFT_RESPONSE_NORMALIZATION,
            "supervised_unk_policy": V2_SFT_SUPERVISED_UNK_POLICY,
            "supervised_targets": V2_SFT_SUPERVISED_TARGETS,
            "supervised_unk_targets": 0,
            "end_thinking_targets": V2_END_THINKING_TARGETS,
            "call_env_targets": V2_CALL_ENV_TARGETS,
        },
        "coverage_proof": {
            "p1_metadata_sha256": p1_metadata_file_sha256,
            "p1_order_sha256": p1_order_file_sha256,
            "p1_unique_rows": P1_SFT_ROWS,
            "p1_supervised_targets": V2_P1_SFT_SUPERVISED_TARGETS,
            "p2_metadata_sha256": p2_metadata_file_sha256,
            "p2_order_sha256": p2_order_file_sha256,
            "p2_unique_rows": P2_SFT_ROWS,
            "p2_supervised_targets": V2_P2_SFT_SUPERVISED_TARGETS,
            "intersection_rows": 0,
            "union_rows": SFT_ROWS,
            "union_is_exact_range_0_through_77716": True,
        },
    }
    value["audit_hash"] = content_hash(value, "audit_hash")
    return value


def validate_sft_holdout_audit(value: Mapping[str, Any]) -> None:
    _checked_object(value, schema=SFT_AUDIT_SCHEMA, hash_field="audit_hash")
    if value.get("status") != "unavailable_no_heldout":
        raise ValueError("SFT held-out status must remain unavailable")
    metrics = value.get("metrics")
    if not isinstance(metrics, Mapping) or any(
        metric is not None for metric in metrics.values()
    ):
        raise ValueError("unavailable held-out SFT metrics must be null")


def checkpoint_files(checkpoint: Path) -> list[Path]:
    checkpoint = checkpoint.resolve(strict=True)
    config_path = checkpoint / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = json.loads(config_path.read_text())
    mismatches = {
        key: (config.get(key), expected)
        for key, expected in EXPECTED_MODEL_CONFIG.items()
        if config.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"endpoint architecture mismatch: {mismatches}")
    weights = sorted(checkpoint.glob("model*.safetensors"))
    if not weights:
        raise FileNotFoundError(f"missing safetensors weights under {checkpoint}")
    if not (checkpoint / "interleaved_training_state.json").is_file():
        raise FileNotFoundError(
            f"missing interleaved_training_state.json under {checkpoint}"
        )
    files = list(weights)
    for name in (
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "interleaved_training_state.json",
    ):
        candidate = checkpoint / name
        if candidate.is_file():
            files.append(candidate)
    # Hash every top-level tokenizer serialization asset, including the
    # custom tokenizer implementation and tiny chess vocabulary.  Explicit
    # patterns avoid accidentally incorporating unrelated logs or markers.
    for pattern in (
        "tokenizer*",
        "vocab*",
        "merges*",
        "special_tokens_map.json",
        "added_tokens.json",
        "sentencepiece*",
        "spiece*",
    ):
        files.extend(
            path for path in checkpoint.glob(pattern) if path.is_file()
        )
    if not any(path.name.startswith("tokenizer") for path in files):
        raise FileNotFoundError(f"missing tokenizer under {checkpoint}")
    if not any(path.name.startswith("vocab") for path in files):
        raise FileNotFoundError(f"missing vocabulary under {checkpoint}")
    return sorted(
        set(files), key=lambda path: str(path.relative_to(checkpoint))
    )


def checkpoint_fingerprint(checkpoint: Path) -> str:
    checkpoint = checkpoint.resolve(strict=True)
    digest = hashlib.sha256()
    for path in checkpoint_files(checkpoint):
        digest.update(str(path.relative_to(checkpoint)).encode())
        digest.update(b"\0")
        with path.open("rb") as handle:
            while chunk := handle.read(8 << 20):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _complete_endpoint(checkpoint: Path) -> bool:
    try:
        checkpoint_files(checkpoint)
    except (FileNotFoundError, NotADirectoryError, ValueError):
        return False
    return True


def _single_glob(root: Path, pattern: str, endpoint_id: str) -> Path | None:
    matches = [path for path in sorted(root.glob(pattern)) if _complete_endpoint(path)]
    if len(matches) > 1:
        raise ValueError(
            f"ambiguous endpoint {endpoint_id}: {[str(path) for path in matches]}"
        )
    return matches[0] if matches else None


def discover_endpoints(
    checkpoint_mount: Path,
    *,
    experiment_version: str = EXPERIMENT_VERSION,
    include_exp4: bool = True,
    exp4_version: str = EXP4_VERSION,
) -> list[dict[str, Any]]:
    """Discover only complete, architecture-authenticated endpoint exports.

    The defaults preserve the deployed v1 discovery contract exactly.  A
    caller may select another immutable pretraining version and explicitly
    disable Exp4 discovery when that version has no collision-safe Exp4 root.
    """

    checkpoint_mount = checkpoint_mount.resolve()
    if not _VERSION_RE.fullmatch(experiment_version):
        raise ValueError(f"unsafe experiment version: {experiment_version!r}")
    if include_exp4 and not _VERSION_RE.fullmatch(exp4_version):
        raise ValueError(f"unsafe Exp4 version: {exp4_version!r}")
    pretrain = (
        checkpoint_mount
        / "interleave_50m"
        / "pretrain"
        / experiment_version
    )
    fixed: Sequence[tuple[str, str, str, str | None, str]] = (
        ("p1", "P1", "P1", None, "p1_shared/final"),
        ("e2-final", "E2", "P2", None, "exp2_monolithic/final"),
        (
            "e3-p2",
            "E3",
            "P2",
            None,
            "p2/exp3-two-cosine-control-from-p1-from-*/final",
        ),
        (
            "e1-u-p2",
            "E1",
            "P2",
            "U",
            "p2/exp1-u-after-rl1500-from-*/final",
        ),
        (
            "e1-d-p2",
            "E1",
            "P2",
            "D",
            "p2/exp1-d-after-rl1500-from-*/final",
        ),
    )
    endpoints: list[dict[str, Any]] = []
    for endpoint_id, experiment, phase, filter_setting, relative_glob in fixed:
        path = _single_glob(pretrain, relative_glob, endpoint_id)
        if path is None:
            continue
        state = json.loads((path / "interleaved_training_state.json").read_text())
        endpoints.append(
            {
                "endpoint_id": endpoint_id,
                "experiment": experiment,
                "phase": phase,
                "filter": filter_setting,
                "method": None,
                "checkpoint_path": str(path),
                "training_state": state,
                "completion_marker": None,
                "declared_checkpoint_sha256": None,
            }
        )

    if include_exp4:
        exp4_root = (
            checkpoint_mount / "interleave_50m" / "exp4" / exp4_version
        )
        for marker_path in sorted(exp4_root.glob("*/*/*/complete.json")):
            marker = json.loads(marker_path.read_text())
            setting = str(marker.get("filter_setting", "")).upper()
            method = str(marker.get("method", ""))
            fingerprint = str(marker.get("fingerprint", ""))
            if (
                marker.get("kind") != "exp4_method_complete"
                or marker.get("state") != "complete"
                or setting not in {"U", "D"}
                or method not in {"hard-sft", "soft-kl", "scratch-replay"}
                or not _SHA256_RE.fullmatch(fingerprint)
                or marker_path.parent.name != fingerprint
                or marker_path.parent.parent.name != method
                or marker_path.parent.parent.parent.name != setting.lower()
            ):
                raise ValueError(f"invalid Exp4 completion marker: {marker_path}")
            final_declared = str(marker.get("final_hf", ""))
            prefix = "/checkpoints/"
            if not final_declared.startswith(prefix):
                raise ValueError(f"invalid Exp4 final_hf path: {final_declared}")
            final = checkpoint_mount / final_declared.removeprefix(prefix)
            expected_parent = (
                marker_path.parent
                / ("scratch_p2_replay" if method == "scratch-replay" else "p2")
                / "final"
            )
            if final.resolve() != expected_parent.resolve():
                raise ValueError(f"Exp4 final path mismatch in {marker_path}")
            if not _complete_endpoint(final):
                raise ValueError(
                    f"Exp4 marker points to incomplete endpoint: {final}"
                )
            endpoint_id = f"exp4-{setting.lower()}-{method}-{fingerprint[:12]}"
            endpoints.append(
                {
                    "endpoint_id": endpoint_id,
                    "experiment": "E4",
                    "phase": "P2",
                    "filter": setting,
                    "method": method,
                    "checkpoint_path": str(final),
                    "training_state": json.loads(
                        (final / "interleaved_training_state.json").read_text()
                    ),
                    "completion_marker": str(marker_path),
                    "declared_checkpoint_sha256": marker.get(
                        "final_hf_sha256"
                    ),
                }
            )
    ids = [endpoint["endpoint_id"] for endpoint in endpoints]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate endpoint IDs")
    return sorted(endpoints, key=lambda endpoint: endpoint["endpoint_id"])


def summarize_chess_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    benchmarks: dict[str, dict[str, Any]] = {}
    for benchmark in ("B1", "B2", "B3", "B4", "B5"):
        mean = metrics.get(f"val-core/test_{benchmark}/reward/mean@16")
        pass_at_1 = metrics.get(f"val-aux/test_{benchmark}/reward/pass@1")
        fallback = not isinstance(pass_at_1, (int, float))
        if fallback:
            pass_at_1 = mean
        benchmarks[benchmark] = {
            "avg_reward": float(mean) if isinstance(mean, (int, float)) else None,
            "pass_at_1": (
                float(pass_at_1)
                if isinstance(pass_at_1, (int, float))
                else None
            ),
            "pass_at_1_from_mean_fallback": fallback,
        }
    missing = [
        name
        for name, values in benchmarks.items()
        if values["avg_reward"] is None or values["pass_at_1"] is None
    ]
    if missing:
        raise ValueError("incomplete B1--B5 metrics: " + ", ".join(missing))
    rewards = [values["avg_reward"] for values in benchmarks.values()]
    passes = [values["pass_at_1"] for values in benchmarks.values()]
    b3_b4 = [benchmarks[name]["avg_reward"] for name in ("B3", "B4")]
    return {
        "pass_at_1": statistics.fmean(passes),
        "avg_reward": statistics.fmean(rewards),
        "b3_avg": benchmarks["B3"]["avg_reward"],
        "b4_avg": benchmarks["B4"]["avg_reward"],
        "b3_b4_avg": statistics.fmean(b3_b4),
        "benchmarks": benchmarks,
        "pass_at_1_semantics": (
            "explicit_reward_pass@1"
            if not any(
                values["pass_at_1_from_mean_fallback"]
                for values in benchmarks.values()
            )
            else "binary_reward_mean@16_fallback"
        ),
    }


def safe_perplexity(loss: float) -> float:
    if not math.isfinite(float(loss)):
        raise ValueError("loss must be finite")
    return math.exp(float(loss)) if float(loss) < 709.0 else math.inf


__all__ = [
    "CHESS_DATA_SHA256",
    "ENDPOINT_NAMESPACE",
    "ENDPOINT_RESULT_SCHEMA",
    "EXPERIMENT_VERSION",
    "EXPECTED_MODEL_CONFIG",
    "PT_HOLDOUT_RECORDS",
    "PT_HOLDOUT_SCHEMA",
    "PT_HOLDOUT_TARGET_TOKENS",
    "SEQUENCE_LENGTH",
    "SFT_AUDIT_SCHEMA",
    "V2_DATA_ARTIFACT_VERSION",
    "V2_EXPERIMENT_VERSION",
    "build_pt_holdout_manifest",
    "build_sft_holdout_audit",
    "build_v2_sft_holdout_audit",
    "checkpoint_fingerprint",
    "checkpoint_files",
    "content_hash",
    "discover_endpoints",
    "plan_pt_holdout",
    "safe_perplexity",
    "sha256_file",
    "summarize_chess_metrics",
    "validate_pt_holdout_manifest",
    "validate_sft_holdout_audit",
]
