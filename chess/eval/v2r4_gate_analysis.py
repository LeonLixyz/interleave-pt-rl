"""Pure, fail-closed analysis for the paired v2r4 production rollout gate."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


SCHEMA = "interleaved-v2r4a-production-gate-report-v1"
CONTRACT_VERSION = "v2r4a_production_gate_20260730"
CANDIDATE_STEPS = (6_000, 8_000, 9_920)
BATCH_LABELS = ("A", "B")
PAIRWISE_CONTRASTS = ((6_000, 8_000), (6_000, 9_920), (8_000, 9_920))
GROUPS_PER_CELL = 1_024
SIBLINGS_PER_GROUP = 8
ROWS_PER_CELL = GROUPS_PER_CELL * SIBLINGS_PER_GROUP
COMPLETED_ROWS_MIN = 8_184
SOLVE_GROUPS_MIN = 16
VARIANCE_GROUPS_MIN = 16
PROTOCOL_ROWS_MIN = 443
ONE_SIDED_95_Z = 1.6448536269514722


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def content_hash(value: Mapping[str, Any], field: str) -> str:
    return hashlib.sha256(
        canonical_json({key: item for key, item in value.items() if key != field})
    ).hexdigest()


def _prompt_fingerprint(row: Mapping[str, Any]) -> str:
    value = {
        "input": str(row.get("input") or ""),
        "FEN": str(row.get("FEN") or ""),
        "PuzzleId": str(row.get("PuzzleId") or ""),
        "ground_truth": str(row.get("ground_truth") or ""),
    }
    if not value["input"]:
        raise ValueError("rollout row has an empty input")
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _score(row: Mapping[str, Any]) -> int:
    value = row.get("score", 0.0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("rollout score must be numeric")
    score = float(value)
    if not math.isfinite(score) or score not in (0.0, 1.0):
        raise ValueError("rollout score must be finite and binary")
    return int(score)


def _joint_valid_protocol(row: Mapping[str, Any]) -> bool:
    output = str(row.get("output") or "")
    end = output.find("</T>")
    call = output.find("<call_env>", end + len("</T>"))
    parsed = row.get("extracted_moves")
    if parsed is None and isinstance(row.get("reward"), Mapping):
        parsed = row["reward"].get("extracted_moves")
    return end >= 0 and call > end and bool(str(parsed or "").strip())


def _wilson_lower(successes: int, total: int) -> float:
    if not 0 <= successes <= total or total <= 0:
        raise ValueError("invalid Wilson interval counts")
    p = successes / total
    z = ONE_SIDED_95_Z
    denominator = 1.0 + z * z / total
    center = p + z * z / (2.0 * total)
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4 * total * total))
    return (center - radius) / denominator


def audit_cell(
    rows: Iterable[Mapping[str, Any]],
    *,
    candidate_step: int,
    batch_label: str,
    rollout_seed: int,
) -> dict[str, Any]:
    """Authenticate one 1,024×8 cell and compute its absolute gate metrics."""

    if candidate_step not in CANDIDATE_STEPS:
        raise ValueError("candidate step is outside the frozen grid")
    batch = str(batch_label).upper()
    if batch not in BATCH_LABELS:
        raise ValueError("batch label must be A or B")
    if isinstance(rollout_seed, bool) or not isinstance(rollout_seed, int):
        raise ValueError("rollout seed must be an integer")
    materialized = list(rows)
    if len(materialized) != ROWS_PER_CELL:
        raise ValueError(
            f"cell must contain exactly {ROWS_PER_CELL} rows"
        )

    status_counts = {"completed": 0, "truncated": 0}
    protocol_rows = 0
    prompt_outcomes: list[dict[str, Any]] = []
    for group_index in range(GROUPS_PER_CELL):
        group_rows = materialized[
            group_index * SIBLINGS_PER_GROUP :
            (group_index + 1) * SIBLINGS_PER_GROUP
        ]
        fingerprints = {_prompt_fingerprint(row) for row in group_rows}
        if len(fingerprints) != 1:
            raise ValueError("prompt identity changed inside a sibling group")
        scores: list[int] = []
        for sibling_index, row in enumerate(group_rows):
            sample_index = group_index * SIBLINGS_PER_GROUP + sibling_index
            if row.get("group_index") != group_index:
                raise ValueError("global group index drifted")
            if row.get("sample_index") != sample_index:
                raise ValueError("global sample index drifted")
            if row.get("sampling_seed_sibling_index") != sibling_index:
                raise ValueError("sampling sibling index drifted")
            if row.get("sampling_seed") != rollout_seed + sample_index:
                raise ValueError("top-level sample-index seed drifted")
            metadata = row.get("metadata")
            if (
                not isinstance(metadata, Mapping)
                or metadata.get("sampling_seed_sibling_index") != sibling_index
                or metadata.get("sampling_seed") != rollout_seed + sample_index
                or metadata.get("sampling_seed_mode") != "sample-index"
            ):
                raise ValueError("metadata sample-index seed drifted")
            status = row.get("status")
            if status not in status_counts:
                raise ValueError("cell contains a disallowed rollout status")
            status_counts[str(status)] += 1
            score = _score(row)
            joint_valid = _joint_valid_protocol(row)
            if score and not joint_valid:
                raise ValueError(
                    "positive rollout row lacks a joint-valid protocol parse"
                )
            protocol_rows += int(joint_valid)
            scores.append(score)
        prompt_outcomes.append(
            {
                "prompt_fingerprint": next(iter(fingerprints)),
                "solve_at_8": int(any(scores)),
                "nonzero_sibling_variance": int(min(scores) < max(scores)),
                "positive_rows": sum(scores),
            }
        )
    fingerprints = [
        str(outcome["prompt_fingerprint"]) for outcome in prompt_outcomes
    ]
    if len(set(fingerprints)) != GROUPS_PER_CELL:
        raise ValueError("cell contains duplicate prompt groups")

    solved = sum(int(value["solve_at_8"]) for value in prompt_outcomes)
    variance = sum(
        int(value["nonzero_sibling_variance"])
        for value in prompt_outcomes
    )
    positive_rows = sum(int(value["positive_rows"]) for value in prompt_outcomes)
    absolute_checks = {
        "inventory_exact": True,
        "completed_rows": status_counts["completed"] >= COMPLETED_ROWS_MIN,
        "solve_at_8_groups": solved >= SOLVE_GROUPS_MIN,
        "nonzero_variance_groups": variance >= VARIANCE_GROUPS_MIN,
        "joint_valid_protocol_rows": protocol_rows >= PROTOCOL_ROWS_MIN,
    }
    return {
        "candidate_step": candidate_step,
        "batch_label": batch,
        "rollout_seed": rollout_seed,
        "rows": ROWS_PER_CELL,
        "prompt_groups": GROUPS_PER_CELL,
        "siblings_per_group": SIBLINGS_PER_GROUP,
        "status_counts": status_counts,
        "solve_at_8_groups": solved,
        "solve_at_8_rate": solved / GROUPS_PER_CELL,
        "solve_at_8_wilson_one_sided_95_lower": _wilson_lower(
            solved, GROUPS_PER_CELL
        ),
        "nonzero_sibling_variance_groups": variance,
        "nonzero_sibling_variance_rate": variance / GROUPS_PER_CELL,
        "variance_wilson_one_sided_95_lower": _wilson_lower(
            variance, GROUPS_PER_CELL
        ),
        "joint_valid_protocol_rows": protocol_rows,
        "joint_valid_protocol_rate": protocol_rows / ROWS_PER_CELL,
        "protocol_wilson_one_sided_95_lower": _wilson_lower(
            protocol_rows, ROWS_PER_CELL
        ),
        "positive_rows": positive_rows,
        "row_reward_rate": positive_rows / ROWS_PER_CELL,
        "prompt_order_sha256": hashlib.sha256(
            canonical_json(fingerprints)
        ).hexdigest(),
        "prompt_set_sha256": hashlib.sha256(
            canonical_json(sorted(fingerprints))
        ).hexdigest(),
        "prompt_outcomes": prompt_outcomes,
        "absolute_checks": absolute_checks,
        "absolute_gate_pass": all(absolute_checks.values()),
    }


def _exact_mcnemar(
    first: Sequence[int],
    second: Sequence[int],
) -> dict[str, Any]:
    if len(first) != len(second) or not first:
        raise ValueError("paired McNemar vectors must be nonempty and equal")
    if any(value not in (0, 1) for value in (*first, *second)):
        raise ValueError("paired McNemar outcomes must be binary")
    first_only = sum(a == 1 and b == 0 for a, b in zip(first, second))
    second_only = sum(a == 0 and b == 1 for a, b in zip(first, second))
    discordant = first_only + second_only
    if discordant == 0:
        p_value = 1.0
    else:
        smaller = min(first_only, second_only)
        tail = sum(
            math.comb(discordant, k) for k in range(smaller + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    return {
        "pairs": len(first),
        "first_only": first_only,
        "second_only": second_only,
        "discordant_pairs": discordant,
        "first_rate": sum(first) / len(first),
        "second_rate": sum(second) / len(second),
        "rate_difference_first_minus_second": (
            sum(first) - sum(second)
        ) / len(first),
        "exact_two_sided_p": p_value,
    }


def _finite_endpoint(value: Mapping[str, Any]) -> bool:
    metrics = value.get("metrics")
    return (
        value.get("state") == "complete"
        and isinstance(metrics, Mapping)
        and bool(metrics)
        and all(
            not isinstance(item, bool)
            and isinstance(item, (int, float))
            and math.isfinite(float(item))
            for item in metrics.values()
        )
    )


def analyze_grid(
    cells: Sequence[Mapping[str, Any]],
    endpoints: Mapping[int, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    """Validate all six cells/endpoints and return a self-hashed report."""

    by_key: dict[tuple[int, str], dict[str, Any]] = {}
    for value in cells:
        step = int(value.get("candidate_step", -1))
        batch = str(value.get("batch_label", "")).upper()
        key = (step, batch)
        if key in by_key:
            raise ValueError("duplicate v2r4 gate cell")
        if step not in CANDIDATE_STEPS or batch not in BATCH_LABELS:
            raise ValueError("unexpected v2r4 gate cell")
        by_key[key] = dict(value)
    expected_keys = {
        (step, batch)
        for step in CANDIDATE_STEPS
        for batch in BATCH_LABELS
    }
    if set(by_key) != expected_keys:
        raise ValueError("v2r4 gate grid is incomplete")

    for batch in BATCH_LABELS:
        reference = [
            item["prompt_fingerprint"]
            for item in by_key[(9_920, batch)]["prompt_outcomes"]
        ]
        for step in CANDIDATE_STEPS:
            observed = [
                item["prompt_fingerprint"]
                for item in by_key[(step, batch)]["prompt_outcomes"]
            ]
            if observed != reference:
                raise ValueError("paired prompt order differs across candidates")
    batch_sets = {
        batch: {
            item["prompt_fingerprint"]
            for item in by_key[(9_920, batch)]["prompt_outcomes"]
        }
        for batch in BATCH_LABELS
    }
    if batch_sets["A"] & batch_sets["B"]:
        raise ValueError("v2r4 prompt batches overlap")

    endpoint_summary: dict[str, Any] = {}
    for step in CANDIDATE_STEPS:
        components = endpoints.get(step)
        if not isinstance(components, Mapping) or set(components) != {
            "pt",
            "chess",
            "p2_sft",
        }:
            raise ValueError(f"endpoint grid is incomplete for step {step}")
        finite = {
            component: _finite_endpoint(value)
            for component, value in components.items()
        }
        if not all(finite.values()):
            raise ValueError(
                f"endpoint metrics are incomplete or non-finite for step {step}"
            )
        endpoint_summary[str(step)] = {
            "components": {
                component: dict(value)
                for component, value in components.items()
            },
            "finite_complete": finite,
            "all_finite_complete": True,
        }

    comparisons: list[dict[str, Any]] = []
    for first_step, second_step in PAIRWISE_CONTRASTS:
        batch_comparisons: dict[str, dict[str, Any]] = {}
        pooled_first: list[int] = []
        pooled_second: list[int] = []
        for batch in BATCH_LABELS:
            first = [
                int(value["solve_at_8"])
                for value in by_key[(first_step, batch)]["prompt_outcomes"]
            ]
            second = [
                int(value["solve_at_8"])
                for value in by_key[(second_step, batch)]["prompt_outcomes"]
            ]
            batch_comparisons[batch] = _exact_mcnemar(first, second)
            pooled_first.extend(first)
            pooled_second.extend(second)
        pooled = _exact_mcnemar(pooled_first, pooled_second)
        directions = [
            math.copysign(
                1.0,
                batch_comparisons[batch][
                    "rate_difference_first_minus_second"
                ],
            )
            if batch_comparisons[batch][
                "rate_difference_first_minus_second"
            ]
            != 0
            else 0.0
            for batch in BATCH_LABELS
        ]
        comparisons.append(
            {
                "first_step": first_step,
                "second_step": second_step,
                "primary_outcome": "prompt_solve_at_8",
                "batches": batch_comparisons,
                "pooled": pooled,
                "effect_direction_agrees_in_a_and_b": (
                    directions[0] == directions[1]
                ),
            }
        )

    ordered = sorted(
        range(len(comparisons)),
        key=lambda index: comparisons[index]["pooled"][
            "exact_two_sided_p"
        ],
    )
    running_adjusted = 0.0
    for rank, index in enumerate(ordered):
        raw = comparisons[index]["pooled"]["exact_two_sided_p"]
        adjusted = min(1.0, raw * (len(comparisons) - rank))
        running_adjusted = max(running_adjusted, adjusted)
        comparisons[index]["holm_adjusted_p"] = running_adjusted
        comparisons[index]["holm_reject_at_0.05"] = (
            running_adjusted <= 0.05
        )

    pooled_solve_rates = {
        step: sum(
            int(value["solve_at_8"])
            for batch in BATCH_LABELS
            for value in by_key[(step, batch)]["prompt_outcomes"]
        )
        / (GROUPS_PER_CELL * len(BATCH_LABELS))
        for step in CANDIDATE_STEPS
    }
    largest_rate = max(pooled_solve_rates.values())
    largest_steps = [
        step
        for step, rate in pooled_solve_rates.items()
        if rate == largest_rate
    ]
    unique_winner: int | None = None
    if len(largest_steps) == 1:
        candidate = largest_steps[0]
        wins_all = True
        for comparison in comparisons:
            first_step = int(comparison["first_step"])
            second_step = int(comparison["second_step"])
            if candidate not in (first_step, second_step):
                continue
            candidate_is_first = candidate == first_step
            per_batch_positive = all(
                (
                    comparison["batches"][batch][
                        "rate_difference_first_minus_second"
                    ]
                    > 0
                )
                if candidate_is_first
                else (
                    comparison["batches"][batch][
                        "rate_difference_first_minus_second"
                    ]
                    < 0
                )
                for batch in BATCH_LABELS
            )
            wins_all = (
                wins_all
                and comparison["holm_reject_at_0.05"] is True
                and per_batch_positive
            )
        if wins_all:
            unique_winner = candidate

    endpoint_9920 = endpoint_summary["9920"]
    chess_metrics = endpoint_9920["components"]["chess"]["metrics"]
    chess_positive = (
        isinstance(chess_metrics.get("avg_reward"), (int, float))
        and not isinstance(chess_metrics.get("avg_reward"), bool)
        and float(chess_metrics["avg_reward"]) > 0
    )
    authorization_checks = {
        "batch_a_absolute_gate": by_key[(9_920, "A")][
            "absolute_gate_pass"
        ]
        is True,
        "batch_b_absolute_gate": by_key[(9_920, "B")][
            "absolute_gate_pass"
        ]
        is True,
        "all_step_9920_endpoints_finite": endpoint_9920[
            "all_finite_complete"
        ]
        is True,
        "step_9920_b1_b5_has_positive": chess_positive,
    }
    core: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": 1,
        "contract_version": CONTRACT_VERSION,
        "status": "complete",
        "cells": [
            by_key[(step, batch)]
            for step in CANDIDATE_STEPS
            for batch in BATCH_LABELS
        ],
        "endpoints": endpoint_summary,
        "paired_comparisons": comparisons,
        "behavioral_selection": {
            "status": (
                "unique_winner"
                if unique_winner is not None
                else "inconclusive_no_selection"
            ),
            "winner_step": unique_winner,
            "pooled_solve_at_8_rates": {
                str(step): rate
                for step, rate in pooled_solve_rates.items()
            },
            "criteria": (
                "unique largest pooled solve@8; Holm-significant wins versus "
                "both alternatives across all three pairwise tests; positive "
                "direction separately in batches A and B"
            ),
            "can_select_early_step_for_original_experiment": False,
        },
        "authorization": {
            "eligible_step": 9_920,
            "checks": authorization_checks,
            "pass": all(authorization_checks.values()),
            "early_steps_can_authorize_original_experiment": False,
        },
    }
    return {**core, "report_sha256": content_hash(core, "report_sha256")}


__all__ = [
    "BATCH_LABELS",
    "CANDIDATE_STEPS",
    "COMPLETED_ROWS_MIN",
    "GROUPS_PER_CELL",
    "PROTOCOL_ROWS_MIN",
    "PAIRWISE_CONTRASTS",
    "ROWS_PER_CELL",
    "SCHEMA",
    "SOLVE_GROUPS_MIN",
    "VARIANCE_GROUPS_MIN",
    "analyze_grid",
    "audit_cell",
    "content_hash",
]
