"""Predeclared, diagnostic-only metrics for the v2r3 structure sweep."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .v2r2_gate import audit_rollout_rows, is_joint_valid_protocol_row


CONTRACT_SCHEMA = "interleaved-v2r3-diagnostic-contract-v1"
CONTRACT_VERSION = "mix10b_sft90k_3072_v2r3_diagnostic_20260730"
PRIMARY_SEED = 42
RESPONSE_CAP = 2_560

# A raw-move row lacks the ordered protocol parse and contains at least two
# whitespace-delimited LAN/UCI/castling tokens.  Requiring two tokens avoids
# classifying a chance square-like word as the raw-move failure mode.
RAW_MOVE_TOKEN_RE = re.compile(
    r"^(?:(?:[PNBRQK])?[a-h][1-8][-x]?[a-h][1-8]"
    r"(?:=[QRBN]|[qrbn])?[+#]?|O-O(?:-O)?[+#]?)$"
)
RAW_MOVE_TOKEN_MIN = 2


def _required_nonnegative_int(
    row: Mapping[str, Any],
    key: str,
    row_number: int,
) -> int:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            f"rollout row {row_number} has invalid {key}: {value!r}"
        )
    return value


def _nearest_rank(values: Sequence[int], quantile: float) -> int:
    """Return the predeclared empirical nearest-rank percentile."""

    if not values:
        raise ValueError("cannot summarize an empty response-length list")
    if not 0.0 < quantile <= 1.0:
        raise ValueError("quantile must be in (0, 1]")
    ordered = sorted(int(value) for value in values)
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def _length_distribution(
    values: Sequence[int],
    *,
    prefix: str,
) -> dict[str, int | float]:
    if not values:
        raise ValueError("cannot summarize an empty response-length list")
    return {
        f"{prefix}_mean": sum(values) / len(values),
        f"{prefix}_min": min(values),
        f"{prefix}_p50_nearest_rank": _nearest_rank(values, 0.50),
        f"{prefix}_p90_nearest_rank": _nearest_rank(values, 0.90),
        f"{prefix}_p99_nearest_rank": _nearest_rank(values, 0.99),
        f"{prefix}_max": max(values),
    }


def raw_move_token_count(output: str) -> int:
    return sum(
        bool(RAW_MOVE_TOKEN_RE.fullmatch(token))
        for token in str(output).split()
    )


def _has_parsed_moves(row: Mapping[str, Any]) -> bool:
    value = row.get("extracted_moves")
    if value is None:
        reward = row.get("reward")
        if isinstance(reward, Mapping):
            value = reward.get("extracted_moves")
    return bool(str(value or "").strip())


def audit_diagnostic_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    seed: int,
) -> dict[str, Any]:
    """Authenticate shape/protocol and add frozen failure-mode diagnostics."""

    materialized = list(rows)
    metrics = audit_rollout_rows(
        materialized,
        seed=seed,
        allowed_statuses=("completed", "truncated"),
    )
    response_lengths: list[int] = []
    effective_lengths: list[int] = []
    env_lengths: list[int] = []
    raw_rows = 0
    raw_tokens = 0
    outputs_with_end_thinking = 0
    outputs_with_call_env = 0
    rows_with_parsed_moves = 0
    group_sampling_siblings: dict[int, set[int]] = {}
    for row_number, row in enumerate(materialized, start=1):
        response_length = _required_nonnegative_int(
            row, "response_length", row_number
        )
        effective_length = _required_nonnegative_int(
            row, "effective_response_length", row_number
        )
        metadata = row.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError(
                f"rollout row {row_number} lacks metadata"
            )
        model_token_count = _required_nonnegative_int(
            metadata, "model_token_count", row_number
        )
        top_level_model_token_count = _required_nonnegative_int(
            row, "model_token_count", row_number
        )
        env_token_count = _required_nonnegative_int(
            metadata, "env_token_count", row_number
        )
        top_level_env_token_count = _required_nonnegative_int(
            row, "env_token_count", row_number
        )
        sampling_seed = _required_nonnegative_int(
            metadata, "sampling_seed", row_number
        )
        top_level_sampling_seed = _required_nonnegative_int(
            row, "sampling_seed", row_number
        )
        sibling_index = _required_nonnegative_int(
            metadata, "sampling_seed_sibling_index", row_number
        )
        top_level_sibling_index = _required_nonnegative_int(
            row, "sampling_seed_sibling_index", row_number
        )
        sample_index = _required_nonnegative_int(
            row, "sample_index", row_number
        )
        if (
            effective_length != model_token_count
            or effective_length != top_level_model_token_count
        ):
            raise ValueError(
                f"rollout row {row_number} effective_response_length "
                "disagrees with top-level/metadata model_token_count"
            )
        if env_token_count != top_level_env_token_count:
            raise ValueError(
                f"rollout row {row_number} top-level/metadata "
                "env_token_count disagree"
            )
        if response_length != model_token_count + env_token_count:
            raise ValueError(
                f"rollout row {row_number} response_length does not equal "
                "model_token_count + env_token_count"
            )
        if effective_length > RESPONSE_CAP:
            raise ValueError(
                f"rollout row {row_number} model response exceeds the "
                f"{RESPONSE_CAP}-token cap"
            )
        if (
            sampling_seed != top_level_sampling_seed
            or sibling_index != top_level_sibling_index
            or sibling_index >= 8
            or sampling_seed != seed + sibling_index
        ):
            raise ValueError(
                f"rollout row {row_number} deterministic sibling seed "
                "identity is invalid"
            )
        group_index = int(row["group_index"])
        if (
            sibling_index != sample_index % 8
            or sample_index != group_index * 8 + sibling_index
        ):
            raise ValueError(
                f"rollout row {row_number} sample/sibling/group index "
                "identity is invalid"
            )
        group_sampling_siblings.setdefault(group_index, set()).add(
            sibling_index
        )
        response_lengths.append(response_length)
        effective_lengths.append(effective_length)
        env_lengths.append(env_token_count)
        output = str(row.get("output") or "")
        outputs_with_end_thinking += int("</T>" in output)
        outputs_with_call_env += int("<call_env>" in output)
        rows_with_parsed_moves += int(_has_parsed_moves(row))
        token_count = raw_move_token_count(output)
        if (
            not is_joint_valid_protocol_row(row)
            and token_count >= RAW_MOVE_TOKEN_MIN
        ):
            raw_rows += 1
            raw_tokens += token_count

    count = len(response_lengths)
    expected_sibling_indices = set(range(8))
    invalid_seed_groups = {
        group_index: sorted(indices)
        for group_index, indices in group_sampling_siblings.items()
        if indices != expected_sibling_indices
    }
    if invalid_seed_groups:
        raise ValueError(
            "deterministic sibling seed indices are not exactly 0..7 in "
            f"every group: {invalid_seed_groups}"
        )
    return {
        **metrics,
        "outputs_with_end_thinking": outputs_with_end_thinking,
        "outputs_with_call_env": outputs_with_call_env,
        "rows_with_parsed_moves": rows_with_parsed_moves,
        "response_length_definition": (
            "total response token segment including generated model tokens "
            "and inserted environment-reply tokens"
        ),
        **_length_distribution(
            response_lengths,
            prefix="response_length",
        ),
        "effective_response_length_definition": (
            "generated model tokens selected by the loss mask; exactly "
            "top-level and metadata model_token_count"
        ),
        **_length_distribution(
            effective_lengths,
            prefix="effective_response_length",
        ),
        "env_token_count_definition": (
            "environment-reply tokens inserted into the response segment; "
            "top-level and metadata counts must agree"
        ),
        **_length_distribution(
            env_lengths,
            prefix="env_token_count",
        ),
        "model_response_cap": RESPONSE_CAP,
        "model_response_at_cap_rows": sum(
            value >= RESPONSE_CAP for value in effective_lengths
        ),
        "model_response_at_cap_rate": (
            sum(value >= RESPONSE_CAP for value in effective_lengths) / count
        ),
        "status_definition": (
            "diagnostic rows accept only completed or truncated; pending, "
            "aborted, failed, and unknown states are rejected"
        ),
        "reward_definition": (
            "binary Miles chess reward is audited for completed and "
            "truncated rows alike; any positive row must have a joint-valid "
            "ordered protocol parse"
        ),
        "sampling_seed_definition": (
            "metadata and top-level sampling_seed equal rollout seed plus "
            "the persisted sibling index; every prompt group contains "
            "indices 0 through 7 exactly once, sample_index equals "
            "group_index * 8 + sibling index, and each seed is reused "
            "across multi-turn continuations"
        ),
        "sampling_seed_groups_verified": len(group_sampling_siblings),
        "raw_move_definition": (
            "no joint-valid ordered </T> then <call_env> parse and at "
            f"least {RAW_MOVE_TOKEN_MIN} whitespace-delimited LAN/UCI/"
            "castling tokens, including dash/capture separators and UCI/LAN "
            "promotion suffixes"
        ),
        "raw_move_without_protocol_rows": raw_rows,
        "raw_move_without_protocol_rate": raw_rows / count,
        "raw_move_tokens_in_flagged_rows": raw_tokens,
    }


__all__ = [
    "CONTRACT_SCHEMA",
    "CONTRACT_VERSION",
    "PRIMARY_SEED",
    "RAW_MOVE_TOKEN_MIN",
    "RAW_MOVE_TOKEN_RE",
    "RESPONSE_CAP",
    "audit_diagnostic_rows",
    "raw_move_token_count",
]
