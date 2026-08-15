"""Pure P2-at-P1 masked-SFT evaluation contracts for the v2r4 gate.

The cleaned v2 corpus has no SFT split that is held out from the complete
P1+P2 experiment: the two legs are a disjoint exact partition of all 77,717
rows.  A P2 row *is*, however, held out from a checkpoint that has consumed
only P1.  This module encodes that narrower claim and refuses results from a
candidate that has consumed any P2 data.

There is deliberately no filesystem, Modal, model, or launcher dependency.
Callers provide the exact artifact bytes.  The selector authenticates their
byte hashes and self hashes, reconstructs both leg code sets, and applies one
frozen hash-ranking rule.  A separate cache-shape audit authenticates the
offset and label files and freezes the exact response-masked denominator used
by every candidate result.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
import struct
from collections.abc import Mapping
from typing import Any

import numpy as np


CONTRACT_VERSION = "interleaved-v2r4-p2-sft-at-p1-20260730"
SELECTION_SCHEMA = "interleaved-v2r4-p2-at-p1-sft-selection-v1"
CACHE_SHAPE_SCHEMA = "interleaved-v2r4-p2-at-p1-sft-cache-shape-v1"
RESULT_SCHEMA = "interleaved-v2r4-p2-at-p1-sft-result-v1"

SELECTION_RECORDS = 4_096
SELECTION_SEED = "interleaved-v2r4-p2-at-p1-sft-heldout-v1"
SELECTION_ALGORITHM = (
    "sha256(seed_utf8 || 0x00 || int64le(signed_sft_code))-rank-v1"
)
EVALUATION_CLAIM = "heldout_at_p1_only"
OBJECTIVE = {
    "name": "response-masked-unweighted-token-cross-entropy-v1",
    "ignore_index": -100,
    "reduction": "sum_nll_then_divide_by_supervised_targets",
    "token_accuracy": "correct_argmax_over_supervised_targets",
    "sft_loss_weight_applied": False,
}

DATA_ARTIFACT_VERSION = "mix10b_sft90k_v2r1_clean_verify_gate"
MANIFEST_SET_HASH = (
    "6f2cc9093b2515e0a6a3aedc56a0cfd597c6b0f76933c5dbcd69eefd22440a23"
)
MANIFEST_SET_FILE_SHA256 = (
    "d2d741998a258ed1367587f922df07e0d7a2b46d906a965208c781e1380feb6e"
)

SOURCE_REPO = "chess-pre-to-post/pretrain_v1_20b"
SOURCE_REVISION = "07dd1b7090ca5f0fb05ef624c26b20bff19483c8"
SOURCE_MANIFEST_HASH = (
    "5e2bd529811066c0c9c264eaf39a820f139ad4a4b1e9c9395fca42118e95a275"
)
SELECTION_HASH = (
    "e80f95f89ee2b6a872157cede635f4f130ebb91685f08283774526ad4562ac00"
)
SFT_REPO = "chess-pre-to-post/sft_v1_200m_90k"
SFT_REVISION = "97f60746dd253b4e130beeb5e66f39e9d42ef25c"

SFT_ROWS = 77_717
P1_SFT_ROWS = 38_858
P2_SFT_ROWS = 38_859
SFT_TOTAL_POSITIONS = 67_601_182
SFT_SUPERVISED_TARGETS = 52_482_753
P1_SFT_SUPERVISED_TARGETS = 26_289_598
P2_SFT_SUPERVISED_TARGETS = 26_193_155
SFT_CALL_ENV_TARGETS = 187_354
SEQUENCE_LENGTH = 3_072
VOCAB_SIZE = 85

SFT_CACHE_HASH = (
    "d82378522d43d5db3e8333588c24b1f864bb9e8ecd46303e1d2cd2e31d31df98"
)
SFT_CACHE_METADATA_FILE_SHA256 = (
    "48b30362e729603798a14daa2c9f42e484fd68942a689c763d18095e8f3baeac"
)
SFT_ROWS_SHA256 = (
    "dffc8a3520a3d5b0866242f1fa66f5906e8f40eae1dbb9a26ec76cf895eb0e9d"
)
SFT_INPUT_IDS_SHA256 = (
    "c8c75b6eec58c6d9943a799d04f3e054221f4e2207873b521e5b8eae548bb8a8"
)
SFT_LABELS_SHA256 = (
    "7bb6b16fdd6a7fe1b1e0702f21e9535334421a5c12d074848f60f8d76d357373"
)
SFT_OFFSETS_SHA256 = (
    "0c6f777a79ae8f0d397f1e623724e30137fa5c89060efed1ba24e5ce48c83701"
)

P1_METADATA_HASH = (
    "2d69b95a2b4829b9ead23ff3950fc74bb8fc2f7be90de8043984f29b5273e07e"
)
P2_METADATA_HASH = (
    "51cc5231026d8b75171816dc84460dd0763ac9390dadc40f94a10606a7baa023"
)
P1_METADATA_FILE_SHA256 = (
    "b3a67af83912a6f82290b23ff7463b22e9cb9cad6403e9d2a54c783d588a55ba"
)
P2_METADATA_FILE_SHA256 = (
    "2536c129a5bbd04c082533b9a4ffed2d318723ea8ac3dec6b85583f217691eed"
)
P1_ORDER_SHA256 = (
    "68fc5a3934ea677f31365d998f67380e7b5d2fa12f7b5bbbed9756a3f8bd9ac4"
)
P2_ORDER_SHA256 = (
    "95d70dfc4474be1b8b301196875db511f9c9d63036332ca6913dd0504d7c17b8"
)
ORDER_RECORDS = 1_666_560
PRETRAIN_RECORDS_PER_LEG = 1_627_605
P1_PADDING_RECORDS = 97
P2_PADDING_RECORDS = 96
PAD_RECORD = int(np.iinfo(np.int64).min)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_CANDIDATE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,191}")


def canonical_json(value: Any) -> bytes:
    """Encode JSON exactly as used by all self-hashed gate objects."""

    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def content_hash(value: Mapping[str, Any], field: str) -> str:
    unhashed = {key: item for key, item in value.items() if key != field}
    return canonical_json_sha256(unhashed)


def _sha256_bytes(value: bytes | bytearray | memoryview) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_exact_sha256(value: Any, expected: str, label: str) -> None:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    if value != expected:
        raise ValueError(f"{label} drifted: {value} != {expected}")


def _read_json_artifact(
    raw: bytes,
    *,
    label: str,
    expected_file_sha256: str,
    schema: str,
    hash_field: str,
    expected_content_hash: str,
) -> dict[str, Any]:
    if not isinstance(raw, bytes):
        raise TypeError(f"{label} must be supplied as exact bytes")
    actual_file_sha256 = _sha256_bytes(raw)
    if actual_file_sha256 != expected_file_sha256:
        raise ValueError(
            f"{label} byte SHA-256 drifted: "
            f"{actual_file_sha256} != {expected_file_sha256}"
        )
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    if value.get("schema") != schema:
        raise ValueError(
            f"{label} schema drifted: {value.get('schema')!r} != {schema!r}"
        )
    actual_content_hash = content_hash(value, hash_field)
    if value.get(hash_field) != actual_content_hash:
        raise ValueError(f"{label} has an invalid {hash_field} self hash")
    if actual_content_hash != expected_content_hash:
        raise ValueError(
            f"{label} {hash_field} drifted: "
            f"{actual_content_hash} != {expected_content_hash}"
        )
    return value


def _read_order_artifact(
    raw: bytes,
    *,
    label: str,
    expected_file_sha256: str,
) -> np.ndarray:
    if not isinstance(raw, bytes):
        raise TypeError(f"{label} must be supplied as exact bytes")
    actual = _sha256_bytes(raw)
    if actual != expected_file_sha256:
        raise ValueError(
            f"{label} byte SHA-256 drifted: {actual} != {expected_file_sha256}"
        )
    try:
        order = np.load(io.BytesIO(raw), allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"{label} is not a valid non-pickle NPY array") from exc
    if order.ndim != 1 or order.dtype != np.dtype("int64"):
        raise ValueError(f"{label} must be a one-dimensional int64 array")
    if len(order) != ORDER_RECORDS:
        raise ValueError(
            f"{label} has {len(order)} records, expected {ORDER_RECORDS}"
        )
    return order


def _exact_mapping(
    value: Mapping[str, Any],
    expected: Mapping[str, Any],
    label: str,
) -> None:
    mismatches = {
        key: (value.get(key), wanted)
        for key, wanted in expected.items()
        if value.get(key) != wanted
    }
    if mismatches:
        raise ValueError(f"{label} identity drifted: {mismatches}")


def _authenticate_manifest_and_cache(
    *,
    manifest_set_json: bytes,
    sft_cache_metadata_json: bytes,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = _read_json_artifact(
        manifest_set_json,
        label="manifest_set.json",
        expected_file_sha256=MANIFEST_SET_FILE_SHA256,
        schema="interleaved-manifest-set-v1",
        hash_field="manifest_set_hash",
        expected_content_hash=MANIFEST_SET_HASH,
    )
    cache = _read_json_artifact(
        sft_cache_metadata_json,
        label="sft_cache/metadata.json",
        expected_file_sha256=SFT_CACHE_METADATA_FILE_SHA256,
        schema="interleaved-sft-cache-v1",
        hash_field="cache_hash",
        expected_content_hash=SFT_CACHE_HASH,
    )
    _exact_mapping(
        manifest,
        {
            "experiment_version": DATA_ARTIFACT_VERSION,
            "source_repo": SOURCE_REPO,
            "source_revision": SOURCE_REVISION,
            "source_manifest_hash": SOURCE_MANIFEST_HASH,
            "selection_hash": SELECTION_HASH,
            "pretrain_tokens": 10_000_000_000,
            "sft_repo": SFT_REPO,
            "sft_revision": SFT_REVISION,
            "sft_rows": SFT_ROWS,
            "sft_cache_hash": SFT_CACHE_HASH,
        },
        "manifest set",
    )
    entries = manifest.get("manifests")
    if not isinstance(entries, Mapping):
        raise ValueError("manifest set lacks a manifests object")
    expected_entries = {
        "p1": {
            "path": "legs/p1/metadata.json",
            "sha256": P1_METADATA_FILE_SHA256,
        },
        "p2": {
            "path": "legs/p2/metadata.json",
            "sha256": P2_METADATA_FILE_SHA256,
        },
    }
    for leg, expected in expected_entries.items():
        entry = entries.get(leg)
        if not isinstance(entry, Mapping) or dict(entry) != expected:
            raise ValueError(f"manifest-set {leg} file identity drifted")

    _exact_mapping(
        cache,
        {
            "num_rows": SFT_ROWS,
            "total_positions": SFT_TOTAL_POSITIONS,
            "sequence_length": SEQUENCE_LENGTH,
            "cot_field": (
                "cot_by_method.trajectory_sep.cot_format_no_labels"
            ),
            "prompt_field": "pgn",
            "response_normalization": (
                "strip-numeric-verify-score-pairs-normalize-whitespace-v1"
            ),
            "supervised_unk_policy": "reject-supervised-unk-v1",
            "strict_sft_audit_required": True,
            "supervised_targets": SFT_SUPERVISED_TARGETS,
            "supervised_unk_targets": 0,
            "masking": "multiturn-prompt-and-env-v1",
            "dtype": "<i4",
            "rows_sha256": SFT_ROWS_SHA256,
            "input_ids_sha256": SFT_INPUT_IDS_SHA256,
            "labels_sha256": SFT_LABELS_SHA256,
            "offsets_sha256": SFT_OFFSETS_SHA256,
        },
        "cleaned SFT cache",
    )
    strict = cache.get("strict_sft_audit")
    delimiters = cache.get("supervised_delimiter_counts")
    if (
        not isinstance(strict, Mapping)
        or strict.get("schema") != "interleaved-sft-strict-audit-v1"
        or strict.get("expected_supervised_targets")
        != SFT_SUPERVISED_TARGETS
        or strict.get("t_end_rows_exactly_one") != SFT_ROWS
        or strict.get("call_env_rows_at_least_one") != SFT_ROWS
        or not isinstance(delimiters, Mapping)
        or delimiters.get("</T>") != SFT_ROWS
        or delimiters.get("<call_env>") != SFT_CALL_ENV_TARGETS
    ):
        raise ValueError("cleaned SFT strict audit drifted")
    return manifest, cache


def _authenticate_leg(
    *,
    leg: str,
    metadata_json: bytes,
    order_npy: bytes,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    if leg == "p1":
        expected_metadata_hash = P1_METADATA_HASH
        expected_metadata_file_hash = P1_METADATA_FILE_SHA256
        expected_order_hash = P1_ORDER_SHA256
        expected_sft_rows = P1_SFT_ROWS
        expected_sft_targets = P1_SFT_SUPERVISED_TARGETS
        expected_padding = P1_PADDING_RECORDS
        shuffle_seed = 42
        target_start = 0
    elif leg == "p2":
        expected_metadata_hash = P2_METADATA_HASH
        expected_metadata_file_hash = P2_METADATA_FILE_SHA256
        expected_order_hash = P2_ORDER_SHA256
        expected_sft_rows = P2_SFT_ROWS
        expected_sft_targets = P2_SFT_SUPERVISED_TARGETS
        expected_padding = P2_PADDING_RECORDS
        shuffle_seed = 43
        target_start = 5_000_000_000
    else:
        raise ValueError(f"unsupported leg {leg!r}")

    metadata = _read_json_artifact(
        metadata_json,
        label=f"legs/{leg}/metadata.json",
        expected_file_sha256=expected_metadata_file_hash,
        schema="interleaved-leg-manifest-v1",
        hash_field="metadata_hash",
        expected_content_hash=expected_metadata_hash,
    )
    _exact_mapping(
        metadata,
        {
            "leg": leg,
            "num_order_records": ORDER_RECORDS,
            "order_file": "order.npy",
            "order_sha256": expected_order_hash,
            "sequence_length": SEQUENCE_LENGTH,
            "source_manifest_hash": SOURCE_MANIFEST_HASH,
            "selection_hash": SELECTION_HASH,
            "sft_cache_hash": SFT_CACHE_HASH,
            "sft_records": expected_sft_rows,
            "sft_supervised_targets": expected_sft_targets,
            "pretrain_records": PRETRAIN_RECORDS_PER_LEG,
            "padding_records": expected_padding,
            "shuffle_seed": shuffle_seed,
            "target_start": target_start,
            "target_count": 5_000_000_000,
            "total_steps": 9_920,
            "physical_steps": 9_920,
            "world_size": 8,
            "local_batch_size": 21,
            "global_batch_size": 168,
        },
        f"{leg} metadata",
    )
    if metadata.get("order_encoding") != {
        "padding": PAD_RECORD,
        "pretrain": "nonnegative local packed-record index",
        "sft": "-(global_sft_row_index+1)",
    }:
        raise ValueError(f"{leg} order encoding drifted")

    order = _read_order_artifact(
        order_npy,
        label=f"legs/{leg}/order.npy",
        expected_file_sha256=expected_order_hash,
    )
    pad_count = int(np.count_nonzero(order == PAD_RECORD))
    pretrain_count = int(np.count_nonzero(order >= 0))
    codes = order[(order < 0) & (order != PAD_RECORD)]
    if (
        pad_count != expected_padding
        or pretrain_count != PRETRAIN_RECORDS_PER_LEG
        or len(codes) != expected_sft_rows
        or len(np.unique(codes)) != expected_sft_rows
        or (
            pad_count + pretrain_count + len(codes)
            != ORDER_RECORDS
        )
    ):
        raise ValueError(f"{leg} order record accounting drifted")
    row_ids = -codes - 1
    if bool(np.any(row_ids < 0)) or bool(np.any(row_ids >= SFT_ROWS)):
        raise ValueError(f"{leg} contains an out-of-range SFT code")
    return metadata, order, codes


def _rank_key(code: int) -> tuple[bytes, int]:
    if (
        isinstance(code, bool)
        or not isinstance(code, (int, np.integer))
        or int(code) >= 0
        or int(code) == PAD_RECORD
    ):
        raise ValueError(f"invalid signed SFT code {code!r}")
    code = int(code)
    digest = hashlib.sha256(
        SELECTION_SEED.encode("utf-8")
        + b"\0"
        + struct.pack("<q", code)
    ).digest()
    return digest, code


def rank_p2_sft_codes(codes: Any) -> list[int]:
    """Return the exact frozen 4,096-code selection in hash-rank order."""

    materialized = [int(code) for code in codes]
    if len(materialized) != len(set(materialized)):
        raise ValueError("P2 SFT codes must be unique before hash ranking")
    if len(materialized) < SELECTION_RECORDS:
        raise ValueError(
            f"P2 has only {len(materialized)} codes; "
            f"{SELECTION_RECORDS} are required"
        )
    ranked = sorted((_rank_key(code) for code in materialized))
    return [code for _, code in ranked[:SELECTION_RECORDS]]


def _selection_source() -> dict[str, Any]:
    return {
        "data_artifact_version": DATA_ARTIFACT_VERSION,
        "manifest_set_hash": MANIFEST_SET_HASH,
        "manifest_set_file_sha256": MANIFEST_SET_FILE_SHA256,
        "sft_cache_hash": SFT_CACHE_HASH,
        "sft_cache_metadata_file_sha256": (
            SFT_CACHE_METADATA_FILE_SHA256
        ),
        "sft_rows_sha256": SFT_ROWS_SHA256,
        "sft_input_ids_sha256": SFT_INPUT_IDS_SHA256,
        "sft_labels_sha256": SFT_LABELS_SHA256,
        "sft_offsets_sha256": SFT_OFFSETS_SHA256,
        "p1_metadata_hash": P1_METADATA_HASH,
        "p1_metadata_file_sha256": P1_METADATA_FILE_SHA256,
        "p1_order_sha256": P1_ORDER_SHA256,
        "p2_metadata_hash": P2_METADATA_HASH,
        "p2_metadata_file_sha256": P2_METADATA_FILE_SHA256,
        "p2_order_sha256": P2_ORDER_SHA256,
    }


def build_p2_at_p1_sft_selection(
    *,
    manifest_set_json: bytes,
    sft_cache_metadata_json: bytes,
    p1_metadata_json: bytes,
    p2_metadata_json: bytes,
    p1_order_npy: bytes,
    p2_order_npy: bytes,
) -> dict[str, Any]:
    """Authenticate cleaned v2 and select exactly 4,096 P2-only SFT codes."""

    _authenticate_manifest_and_cache(
        manifest_set_json=manifest_set_json,
        sft_cache_metadata_json=sft_cache_metadata_json,
    )
    _, _, p1_codes_array = _authenticate_leg(
        leg="p1",
        metadata_json=p1_metadata_json,
        order_npy=p1_order_npy,
    )
    _, _, p2_codes_array = _authenticate_leg(
        leg="p2",
        metadata_json=p2_metadata_json,
        order_npy=p2_order_npy,
    )
    p1_codes = {int(code) for code in p1_codes_array}
    p2_codes = {int(code) for code in p2_codes_array}
    if p1_codes.intersection(p2_codes):
        raise ValueError("P1 and P2 SFT code sets overlap")
    union_rows = sorted(-code - 1 for code in p1_codes.union(p2_codes))
    if union_rows != list(range(SFT_ROWS)):
        raise ValueError("P1/P2 SFT codes are not the exact full-row partition")

    selected_codes = rank_p2_sft_codes(p2_codes)
    selected_rows = [-code - 1 for code in selected_codes]
    if p1_codes.intersection(selected_codes):
        raise ValueError("selected P2 codes overlap P1")
    if not set(selected_codes).issubset(p2_codes):
        raise ValueError("selected codes are not all members of P2")

    all_ranked = sorted((_rank_key(code) for code in p2_codes))
    value: dict[str, Any] = {
        "schema": SELECTION_SCHEMA,
        "schema_version": 1,
        "contract_version": CONTRACT_VERSION,
        "status": "complete",
        "evaluation_claim": EVALUATION_CLAIM,
        "candidate_eligibility": {
            "required_training_leg": "p1",
            "candidate_must_not_have_consumed_p2": True,
            "forbidden_for_p2_or_p1_plus_p2_candidates": True,
        },
        "selection": {
            "algorithm": SELECTION_ALGORITHM,
            "seed": SELECTION_SEED,
            "num_records": SELECTION_RECORDS,
            "signed_sft_codes": selected_codes,
            "global_sft_row_ids": selected_rows,
            "signed_sft_codes_sha256": canonical_json_sha256(
                selected_codes
            ),
            "last_selected_rank_sha256": all_ranked[
                SELECTION_RECORDS - 1
            ][0].hex(),
            "first_excluded_rank_sha256": all_ranked[
                SELECTION_RECORDS
            ][0].hex(),
        },
        "source": _selection_source(),
        "non_overlap_proof": {
            "p1_unique_sft_codes": len(p1_codes),
            "p2_unique_sft_codes": len(p2_codes),
            "p1_p2_intersection_codes": 0,
            "p1_p2_union_rows": len(union_rows),
            "union_is_exact_range_0_through_77716": (
                union_rows == list(range(SFT_ROWS))
            ),
            "selected_unique_codes": len(set(selected_codes)),
            "selected_p1_overlap_codes": len(
                p1_codes.intersection(selected_codes)
            ),
            "selected_p2_members": len(
                p2_codes.intersection(selected_codes)
            ),
        },
    }
    value["selection_hash"] = content_hash(value, "selection_hash")
    return value


def _validate_selection_envelope(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("selection manifest must be an object")
    if value.get("schema") != SELECTION_SCHEMA:
        raise ValueError("selection schema drifted")
    if value.get("selection_hash") != content_hash(value, "selection_hash"):
        raise ValueError("selection manifest self hash is invalid")
    _exact_mapping(
        value,
        {
            "schema_version": 1,
            "contract_version": CONTRACT_VERSION,
            "status": "complete",
            "evaluation_claim": EVALUATION_CLAIM,
        },
        "selection manifest",
    )
    if value.get("candidate_eligibility") != {
        "required_training_leg": "p1",
        "candidate_must_not_have_consumed_p2": True,
        "forbidden_for_p2_or_p1_plus_p2_candidates": True,
    }:
        raise ValueError("selection candidate-eligibility claim drifted")
    if value.get("source") != _selection_source():
        raise ValueError("selection source identity drifted")

    selection = value.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("selection manifest lacks its selection object")
    _exact_mapping(
        selection,
        {
            "algorithm": SELECTION_ALGORITHM,
            "seed": SELECTION_SEED,
            "num_records": SELECTION_RECORDS,
        },
        "selection rule",
    )
    codes = selection.get("signed_sft_codes")
    rows = selection.get("global_sft_row_ids")
    if (
        not isinstance(codes, list)
        or len(codes) != SELECTION_RECORDS
        or any(
            isinstance(code, bool)
            or not isinstance(code, int)
            or code >= 0
            or code == PAD_RECORD
            for code in codes
        )
        or len(set(codes)) != SELECTION_RECORDS
    ):
        raise ValueError(
            f"selection must expose {SELECTION_RECORDS} unique signed SFT codes"
        )
    if rows != [-code - 1 for code in codes]:
        raise ValueError("selected row IDs do not match signed SFT codes")
    if selection.get("signed_sft_codes_sha256") != canonical_json_sha256(
        codes
    ):
        raise ValueError("selected-code hash drifted")
    ranked = [_rank_key(code)[0].hex() for code in codes]
    if ranked != sorted(ranked):
        raise ValueError("selected codes are not in frozen hash-rank order")
    if selection.get("last_selected_rank_sha256") != ranked[-1]:
        raise ValueError("last-selected hash-rank boundary drifted")
    proof = value.get("non_overlap_proof")
    if not isinstance(proof, Mapping) or proof.get(
        "selected_unique_codes"
    ) != SELECTION_RECORDS:
        raise ValueError("selection non-overlap proof is incomplete")
    return dict(value)


def validate_p2_at_p1_sft_selection(
    value: Mapping[str, Any],
    *,
    manifest_set_json: bytes,
    sft_cache_metadata_json: bytes,
    p1_metadata_json: bytes,
    p2_metadata_json: bytes,
    p1_order_npy: bytes,
    p2_order_npy: bytes,
) -> dict[str, Any]:
    """Rebuild the frozen selection and require byte-for-byte JSON equality."""

    _validate_selection_envelope(value)
    expected = build_p2_at_p1_sft_selection(
        manifest_set_json=manifest_set_json,
        sft_cache_metadata_json=sft_cache_metadata_json,
        p1_metadata_json=p1_metadata_json,
        p2_metadata_json=p2_metadata_json,
        p1_order_npy=p1_order_npy,
        p2_order_npy=p2_order_npy,
    )
    if dict(value) != expected:
        raise ValueError("P2-at-P1 SFT selection differs from frozen rebuild")
    return expected


def _labels_view(labels_i32: Any) -> np.ndarray:
    if isinstance(labels_i32, bytes):
        labels = np.frombuffer(labels_i32, dtype="<i4")
    else:
        labels = np.asarray(labels_i32)
        if labels.ndim != 1 or labels.dtype != np.dtype("int32"):
            raise ValueError("labels.i32 must be a one-dimensional int32 array")
        if not labels.flags.c_contiguous:
            raise ValueError("labels.i32 must be C-contiguous")
    if labels.ndim != 1 or labels.dtype != np.dtype("int32"):
        raise ValueError("labels.i32 must be a one-dimensional int32 array")
    return labels


def build_selected_cache_shape(
    selection_manifest: Mapping[str, Any],
    *,
    offsets_npy: bytes,
    labels_i32: Any,
) -> dict[str, Any]:
    """Freeze exact selected-row positions and response-masked target count."""

    selection_manifest = _validate_selection_envelope(selection_manifest)
    if not isinstance(offsets_npy, bytes):
        raise TypeError("offsets.npy must be supplied as exact bytes")
    actual_offsets_hash = _sha256_bytes(offsets_npy)
    if actual_offsets_hash != SFT_OFFSETS_SHA256:
        raise ValueError(
            "offsets.npy byte SHA-256 drifted: "
            f"{actual_offsets_hash} != {SFT_OFFSETS_SHA256}"
        )
    try:
        offsets = np.load(io.BytesIO(offsets_npy), allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError("offsets.npy is not a valid non-pickle NPY array") from exc
    if (
        offsets.ndim != 1
        or offsets.dtype != np.dtype("int64")
        or len(offsets) != SFT_ROWS + 1
        or int(offsets[0]) != 0
        or int(offsets[-1]) != SFT_TOTAL_POSITIONS
        or bool(np.any(np.diff(offsets) <= 0))
    ):
        raise ValueError("offsets.npy shape or monotonicity drifted")

    labels = _labels_view(labels_i32)
    labels_hash = hashlib.sha256(memoryview(labels).cast("B")).hexdigest()
    if labels_hash != SFT_LABELS_SHA256:
        raise ValueError(
            f"labels.i32 SHA-256 drifted: {labels_hash} != {SFT_LABELS_SHA256}"
        )
    if len(labels) != SFT_TOTAL_POSITIONS:
        raise ValueError("labels.i32 length differs from cache metadata")

    codes = selection_manifest["selection"]["signed_sft_codes"]
    total_positions = 0
    supervised_targets = 0
    position_counts: list[int] = []
    supervised_counts: list[int] = []
    selected_digest = hashlib.sha256()
    for code in codes:
        row_id = -int(code) - 1
        start = int(offsets[row_id])
        stop = int(offsets[row_id + 1])
        row_labels = labels[start:stop]
        if not (0 < len(row_labels) <= SEQUENCE_LENGTH):
            raise ValueError(
                f"selected SFT row {row_id} has invalid aligned length "
                f"{len(row_labels)}"
            )
        valid = (row_labels == OBJECTIVE["ignore_index"]) | (
            (row_labels >= 0) & (row_labels < VOCAB_SIZE)
        )
        if not bool(np.all(valid)):
            raise ValueError(
                f"selected SFT row {row_id} has an invalid response label"
            )
        supervised = int(
            np.count_nonzero(row_labels != OBJECTIVE["ignore_index"])
        )
        if supervised <= 0:
            raise ValueError(
                f"selected SFT row {row_id} has no supervised response target"
            )
        total_positions += len(row_labels)
        supervised_targets += supervised
        position_counts.append(len(row_labels))
        supervised_counts.append(supervised)
        selected_digest.update(struct.pack("<qQ", int(code), len(row_labels)))
        selected_digest.update(memoryview(row_labels).cast("B"))

    value: dict[str, Any] = {
        "schema": CACHE_SHAPE_SCHEMA,
        "schema_version": 1,
        "contract_version": CONTRACT_VERSION,
        "status": "complete",
        "selection_hash": selection_manifest["selection_hash"],
        "cache_identity": {
            "cache_hash": SFT_CACHE_HASH,
            "offsets_sha256": SFT_OFFSETS_SHA256,
            "labels_sha256": SFT_LABELS_SHA256,
            "masking": "multiturn-prompt-and-env-v1",
            "ignore_index": OBJECTIVE["ignore_index"],
            "vocab_size": VOCAB_SIZE,
        },
        "shape": {
            "num_records": SELECTION_RECORDS,
            "total_aligned_positions": total_positions,
            "supervised_targets": supervised_targets,
            "ignored_positions": total_positions - supervised_targets,
            "aligned_positions_per_row_min": min(position_counts),
            "aligned_positions_per_row_max": max(position_counts),
            "supervised_targets_per_row_min": min(supervised_counts),
            "supervised_targets_per_row_max": max(supervised_counts),
            "selected_label_payload_sha256": selected_digest.hexdigest(),
        },
    }
    value["cache_shape_hash"] = content_hash(value, "cache_shape_hash")
    return value


def validate_selected_cache_shape(
    value: Mapping[str, Any],
    selection_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    selection_manifest = _validate_selection_envelope(selection_manifest)
    if not isinstance(value, Mapping):
        raise ValueError("selected cache shape must be an object")
    if value.get("schema") != CACHE_SHAPE_SCHEMA:
        raise ValueError("selected cache-shape schema drifted")
    if value.get("cache_shape_hash") != content_hash(
        value, "cache_shape_hash"
    ):
        raise ValueError("selected cache-shape self hash is invalid")
    _exact_mapping(
        value,
        {
            "schema_version": 1,
            "contract_version": CONTRACT_VERSION,
            "status": "complete",
            "selection_hash": selection_manifest["selection_hash"],
        },
        "selected cache shape",
    )
    if value.get("cache_identity") != {
        "cache_hash": SFT_CACHE_HASH,
        "offsets_sha256": SFT_OFFSETS_SHA256,
        "labels_sha256": SFT_LABELS_SHA256,
        "masking": "multiturn-prompt-and-env-v1",
        "ignore_index": OBJECTIVE["ignore_index"],
        "vocab_size": VOCAB_SIZE,
    }:
        raise ValueError("selected cache-shape identity drifted")
    shape = value.get("shape")
    if not isinstance(shape, Mapping):
        raise ValueError("selected cache shape lacks aggregate shape")
    integer_fields = (
        "num_records",
        "total_aligned_positions",
        "supervised_targets",
        "ignored_positions",
        "aligned_positions_per_row_min",
        "aligned_positions_per_row_max",
        "supervised_targets_per_row_min",
        "supervised_targets_per_row_max",
    )
    for field in integer_fields:
        item = shape.get(field)
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ValueError(f"cache shape {field} must be non-negative integer")
    if (
        shape["num_records"] != SELECTION_RECORDS
        or shape["supervised_targets"] <= 0
        or shape["total_aligned_positions"]
        != shape["supervised_targets"] + shape["ignored_positions"]
        or shape["aligned_positions_per_row_min"] <= 0
        or shape["aligned_positions_per_row_max"] > SEQUENCE_LENGTH
        or shape["aligned_positions_per_row_min"]
        > shape["aligned_positions_per_row_max"]
        or shape["supervised_targets_per_row_min"] <= 0
        or shape["supervised_targets_per_row_min"]
        > shape["supervised_targets_per_row_max"]
        or shape["supervised_targets_per_row_max"]
        > shape["aligned_positions_per_row_max"]
    ):
        raise ValueError("selected cache aggregate shape is inconsistent")
    digest = shape.get("selected_label_payload_sha256")
    if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
        raise ValueError("selected label payload lacks a SHA-256")
    return dict(value)


def _validate_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(candidate, Mapping):
        raise ValueError("candidate must be an object")
    expected_keys = {
        "candidate_id",
        "checkpoint_sha256",
        "checkpoint_step",
        "sft_loss_weight",
        "training_data_manifest_set_hash",
        "training_leg",
        "has_consumed_p2",
    }
    if set(candidate) != expected_keys:
        raise ValueError(
            f"candidate fields must equal {sorted(expected_keys)}, "
            f"got {sorted(candidate)}"
        )
    candidate_id = candidate.get("candidate_id")
    if not isinstance(candidate_id, str) or not _CANDIDATE_ID_RE.fullmatch(
        candidate_id
    ):
        raise ValueError("candidate_id is invalid")
    checkpoint_sha = candidate.get("checkpoint_sha256")
    if not isinstance(checkpoint_sha, str) or not _SHA256_RE.fullmatch(
        checkpoint_sha
    ):
        raise ValueError("candidate checkpoint_sha256 is invalid")
    step = candidate.get("checkpoint_step")
    if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
        raise ValueError("candidate checkpoint_step must be a positive integer")
    weight = candidate.get("sft_loss_weight")
    if (
        isinstance(weight, bool)
        or not isinstance(weight, (int, float))
        or not math.isfinite(float(weight))
        or float(weight) <= 0
    ):
        raise ValueError("candidate sft_loss_weight must be finite and positive")
    if (
        candidate.get("training_data_manifest_set_hash") != MANIFEST_SET_HASH
        or candidate.get("training_leg") != "p1"
        or candidate.get("has_consumed_p2") is not False
    ):
        raise ValueError(
            "candidate is not eligible for the P2-heldout-at-P1 claim"
        )
    return dict(candidate)


def build_candidate_sft_result(
    *,
    selection_manifest: Mapping[str, Any],
    cache_shape: Mapping[str, Any],
    candidate: Mapping[str, Any],
    negative_log_likelihood_sum: float,
    correct_supervised_tokens: int,
    batches: int,
) -> dict[str, Any]:
    """Construct a self-hashed, token-sum-weighted candidate SFT result."""

    selection_manifest = _validate_selection_envelope(selection_manifest)
    cache_shape = validate_selected_cache_shape(
        cache_shape, selection_manifest
    )
    candidate = _validate_candidate(candidate)
    if isinstance(negative_log_likelihood_sum, bool) or not isinstance(
        negative_log_likelihood_sum, (int, float)
    ):
        raise ValueError("negative_log_likelihood_sum must be numeric")
    nll_sum = float(negative_log_likelihood_sum)
    if not math.isfinite(nll_sum) or nll_sum < 0:
        raise ValueError(
            "negative_log_likelihood_sum must be finite and non-negative"
        )
    if (
        isinstance(correct_supervised_tokens, bool)
        or not isinstance(correct_supervised_tokens, int)
    ):
        raise ValueError("correct_supervised_tokens must be an integer")
    supervised = int(cache_shape["shape"]["supervised_targets"])
    if not 0 <= correct_supervised_tokens <= supervised:
        raise ValueError("correct_supervised_tokens is outside the denominator")
    if (
        isinstance(batches, bool)
        or not isinstance(batches, int)
        or not 1 <= batches <= SELECTION_RECORDS
    ):
        raise ValueError("batches must be between 1 and the selected row count")

    mean_ce = nll_sum / supervised
    if mean_ce >= 709.0:
        raise ValueError("mean CE is too large for a finite perplexity")
    accuracy = correct_supervised_tokens / supervised
    shape = cache_shape["shape"]
    value: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "schema_version": 1,
        "contract_version": CONTRACT_VERSION,
        "status": "complete",
        "evaluation_claim": EVALUATION_CLAIM,
        "selection_hash": selection_manifest["selection_hash"],
        "cache_shape_hash": cache_shape["cache_shape_hash"],
        "candidate": candidate,
        "objective": dict(OBJECTIVE),
        "aggregate": {
            "rows_evaluated": SELECTION_RECORDS,
            "batches": batches,
            "total_aligned_positions": shape["total_aligned_positions"],
            "supervised_targets": supervised,
            "ignored_positions": shape["ignored_positions"],
            "negative_log_likelihood_sum": nll_sum,
            "correct_supervised_tokens": correct_supervised_tokens,
        },
        "metrics": {
            "masked_sft_unweighted_token_ce": mean_ce,
            "masked_sft_perplexity": math.exp(mean_ce),
            "masked_sft_token_accuracy": accuracy,
        },
    }
    value["result_hash"] = content_hash(value, "result_hash")
    return value


def validate_candidate_sft_result(
    value: Mapping[str, Any],
    *,
    selection_manifest: Mapping[str, Any],
    cache_shape: Mapping[str, Any],
    expected_candidate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fail closed on weighted, mis-denominated, or P2-trained results."""

    selection_manifest = _validate_selection_envelope(selection_manifest)
    cache_shape = validate_selected_cache_shape(
        cache_shape, selection_manifest
    )
    if not isinstance(value, Mapping):
        raise ValueError("candidate result must be an object")
    if value.get("schema") != RESULT_SCHEMA:
        raise ValueError("candidate result schema drifted")
    if value.get("result_hash") != content_hash(value, "result_hash"):
        raise ValueError("candidate result self hash is invalid")
    _exact_mapping(
        value,
        {
            "schema_version": 1,
            "contract_version": CONTRACT_VERSION,
            "status": "complete",
            "evaluation_claim": EVALUATION_CLAIM,
            "selection_hash": selection_manifest["selection_hash"],
            "cache_shape_hash": cache_shape["cache_shape_hash"],
        },
        "candidate result",
    )
    candidate = _validate_candidate(value.get("candidate"))
    if expected_candidate is not None:
        expected = _validate_candidate(expected_candidate)
        if candidate != expected:
            raise ValueError("candidate result identity differs from expectation")
    if value.get("objective") != OBJECTIVE:
        raise ValueError(
            "candidate result is not response-masked unweighted token CE"
        )

    aggregate = value.get("aggregate")
    if not isinstance(aggregate, Mapping):
        raise ValueError("candidate result lacks an aggregate object")
    expected_aggregate_keys = {
        "rows_evaluated",
        "batches",
        "total_aligned_positions",
        "supervised_targets",
        "ignored_positions",
        "negative_log_likelihood_sum",
        "correct_supervised_tokens",
    }
    if set(aggregate) != expected_aggregate_keys:
        raise ValueError("candidate aggregate fields drifted")
    shape = cache_shape["shape"]
    _exact_mapping(
        aggregate,
        {
            "rows_evaluated": SELECTION_RECORDS,
            "total_aligned_positions": shape["total_aligned_positions"],
            "supervised_targets": shape["supervised_targets"],
            "ignored_positions": shape["ignored_positions"],
        },
        "candidate aggregate denominator",
    )
    batches = aggregate.get("batches")
    if (
        isinstance(batches, bool)
        or not isinstance(batches, int)
        or not 1 <= batches <= SELECTION_RECORDS
    ):
        raise ValueError("candidate aggregate batches is invalid")
    nll = aggregate.get("negative_log_likelihood_sum")
    if (
        isinstance(nll, bool)
        or not isinstance(nll, (int, float))
        or not math.isfinite(float(nll))
        or float(nll) < 0
    ):
        raise ValueError("candidate aggregate NLL sum is invalid")
    correct = aggregate.get("correct_supervised_tokens")
    supervised = int(shape["supervised_targets"])
    if (
        isinstance(correct, bool)
        or not isinstance(correct, int)
        or not 0 <= correct <= supervised
    ):
        raise ValueError("candidate aggregate correct-token count is invalid")

    mean_ce = float(nll) / supervised
    if mean_ce >= 709.0:
        raise ValueError("candidate mean CE cannot produce finite perplexity")
    expected_metrics = {
        "masked_sft_unweighted_token_ce": mean_ce,
        "masked_sft_perplexity": math.exp(mean_ce),
        "masked_sft_token_accuracy": correct / supervised,
    }
    metrics = value.get("metrics")
    if not isinstance(metrics, Mapping) or set(metrics) != set(expected_metrics):
        raise ValueError("candidate metrics shape drifted")
    for key, expected in expected_metrics.items():
        observed = metrics.get(key)
        if (
            isinstance(observed, bool)
            or not isinstance(observed, (int, float))
            or not math.isfinite(float(observed))
            or not math.isclose(
                float(observed),
                expected,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(f"candidate metric {key} is inconsistent")
    return dict(value)


__all__ = [
    "CACHE_SHAPE_SCHEMA",
    "CONTRACT_VERSION",
    "EVALUATION_CLAIM",
    "OBJECTIVE",
    "RESULT_SCHEMA",
    "SELECTION_ALGORITHM",
    "SELECTION_RECORDS",
    "SELECTION_SCHEMA",
    "SELECTION_SEED",
    "build_candidate_sft_result",
    "build_p2_at_p1_sft_selection",
    "build_selected_cache_shape",
    "canonical_json",
    "canonical_json_sha256",
    "content_hash",
    "rank_p2_sft_codes",
    "validate_candidate_sft_result",
    "validate_p2_at_p1_sft_selection",
    "validate_selected_cache_shape",
]
