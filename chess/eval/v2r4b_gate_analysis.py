"""Corrected, direction-neutral analysis for the v2r4b production gate.

The frozen v2r4a plan predeclared chess reward and joint protocol validity as
separate outcomes.  Its implementation accidentally asserted that every
positive chess reward must also be protocol-valid.  The reward model can
legitimately score a correct move even when the response omits ``</T>``.
This module removes only that unstated implication; all inventories,
thresholds, paired tests, and authorization rules remain frozen.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from Eval import v2r4_gate_analysis as frozen


SCHEMA = "interleaved-v2r4b-production-gate-report-v1"
CONTRACT_VERSION = "v2r4b_production_gate_20260730"
CANDIDATE_STEPS = frozen.CANDIDATE_STEPS
BATCH_LABELS = frozen.BATCH_LABELS
PAIRWISE_CONTRASTS = frozen.PAIRWISE_CONTRASTS
GROUPS_PER_CELL = frozen.GROUPS_PER_CELL
SIBLINGS_PER_GROUP = frozen.SIBLINGS_PER_GROUP
ROWS_PER_CELL = frozen.ROWS_PER_CELL
COMPLETED_ROWS_MIN = frozen.COMPLETED_ROWS_MIN
SOLVE_GROUPS_MIN = frozen.SOLVE_GROUPS_MIN
VARIANCE_GROUPS_MIN = frozen.VARIANCE_GROUPS_MIN
PROTOCOL_ROWS_MIN = frozen.PROTOCOL_ROWS_MIN
content_hash = frozen.content_hash
canonical_json = frozen.canonical_json


def audit_cell(
    rows: Iterable[Mapping[str, Any]],
    *,
    candidate_step: int,
    batch_label: str,
    rollout_seed: int,
) -> dict[str, Any]:
    """Authenticate a cell and count reward/protocol outcomes independently."""

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
    positive_protocol_rows = 0
    positive_nonprotocol_rows = 0
    prompt_outcomes: list[dict[str, Any]] = []
    for group_index in range(GROUPS_PER_CELL):
        group_rows = materialized[
            group_index * SIBLINGS_PER_GROUP :
            (group_index + 1) * SIBLINGS_PER_GROUP
        ]
        fingerprints = {
            frozen._prompt_fingerprint(row) for row in group_rows
        }
        if len(fingerprints) != 1:
            raise ValueError(
                "prompt identity changed inside a sibling group"
            )
        scores: list[int] = []
        for sibling_index, row in enumerate(group_rows):
            sample_index = (
                group_index * SIBLINGS_PER_GROUP + sibling_index
            )
            if row.get("group_index") != group_index:
                raise ValueError("global group index drifted")
            if row.get("sample_index") != sample_index:
                raise ValueError("global sample index drifted")
            if (
                row.get("sampling_seed_sibling_index")
                != sibling_index
            ):
                raise ValueError("sampling sibling index drifted")
            if row.get("sampling_seed") != rollout_seed + sample_index:
                raise ValueError(
                    "top-level sample-index seed drifted"
                )
            metadata = row.get("metadata")
            if (
                not isinstance(metadata, Mapping)
                or metadata.get("sampling_seed_sibling_index")
                != sibling_index
                or metadata.get("sampling_seed")
                != rollout_seed + sample_index
                or metadata.get("sampling_seed_mode") != "sample-index"
            ):
                raise ValueError("metadata sample-index seed drifted")
            status = row.get("status")
            if status not in status_counts:
                raise ValueError(
                    "cell contains a disallowed rollout status"
                )
            status_counts[str(status)] += 1
            score = frozen._score(row)
            joint_valid = frozen._joint_valid_protocol(row)
            protocol_rows += int(joint_valid)
            positive_protocol_rows += int(score and joint_valid)
            positive_nonprotocol_rows += int(
                score and not joint_valid
            )
            scores.append(score)
        prompt_outcomes.append(
            {
                "prompt_fingerprint": next(iter(fingerprints)),
                "solve_at_8": int(any(scores)),
                "nonzero_sibling_variance": int(
                    min(scores) < max(scores)
                ),
                "positive_rows": sum(scores),
            }
        )

    fingerprints = [
        str(outcome["prompt_fingerprint"])
        for outcome in prompt_outcomes
    ]
    if len(set(fingerprints)) != GROUPS_PER_CELL:
        raise ValueError("cell contains duplicate prompt groups")
    solved = sum(
        int(value["solve_at_8"]) for value in prompt_outcomes
    )
    variance = sum(
        int(value["nonzero_sibling_variance"])
        for value in prompt_outcomes
    )
    positive_rows = sum(
        int(value["positive_rows"]) for value in prompt_outcomes
    )
    if (
        positive_protocol_rows + positive_nonprotocol_rows
        != positive_rows
    ):
        raise AssertionError("positive/protocol cross-tab drifted")
    absolute_checks = {
        "inventory_exact": True,
        "completed_rows": (
            status_counts["completed"] >= COMPLETED_ROWS_MIN
        ),
        "solve_at_8_groups": solved >= SOLVE_GROUPS_MIN,
        "nonzero_variance_groups": (
            variance >= VARIANCE_GROUPS_MIN
        ),
        "joint_valid_protocol_rows": (
            protocol_rows >= PROTOCOL_ROWS_MIN
        ),
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
        "solve_at_8_wilson_one_sided_95_lower": (
            frozen._wilson_lower(solved, GROUPS_PER_CELL)
        ),
        "nonzero_sibling_variance_groups": variance,
        "nonzero_sibling_variance_rate": (
            variance / GROUPS_PER_CELL
        ),
        "variance_wilson_one_sided_95_lower": (
            frozen._wilson_lower(variance, GROUPS_PER_CELL)
        ),
        "joint_valid_protocol_rows": protocol_rows,
        "joint_valid_protocol_rate": protocol_rows / ROWS_PER_CELL,
        "protocol_wilson_one_sided_95_lower": (
            frozen._wilson_lower(protocol_rows, ROWS_PER_CELL)
        ),
        "positive_rows": positive_rows,
        "positive_joint_valid_protocol_rows": (
            positive_protocol_rows
        ),
        "positive_nonprotocol_rows": positive_nonprotocol_rows,
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


def analyze_grid(
    cells: Sequence[Mapping[str, Any]],
    endpoints: Mapping[int, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    """Apply the frozen v2r4 grid analysis with corrected report identity."""

    prior = frozen.analyze_grid(cells, endpoints)
    core = {
        key: value
        for key, value in prior.items()
        if key != "report_sha256"
    }
    core["schema"] = SCHEMA
    core["contract_version"] = CONTRACT_VERSION
    core["analysis_correction"] = {
        "kind": (
            "count_chess_reward_and_joint_protocol_validity_"
            "independently"
        ),
        "positive_reward_implies_protocol_validity": False,
        "protocol_threshold_unchanged": PROTOCOL_ROWS_MIN,
        "reward_thresholds_unchanged": True,
        "paired_tests_unchanged": True,
        "authorization_rules_unchanged": True,
    }
    return {
        **core,
        "report_sha256": content_hash(core, "report_sha256"),
    }


__all__ = [
    "BATCH_LABELS",
    "CANDIDATE_STEPS",
    "COMPLETED_ROWS_MIN",
    "CONTRACT_VERSION",
    "GROUPS_PER_CELL",
    "PAIRWISE_CONTRASTS",
    "PROTOCOL_ROWS_MIN",
    "ROWS_PER_CELL",
    "SCHEMA",
    "SOLVE_GROUPS_MIN",
    "VARIANCE_GROUPS_MIN",
    "analyze_grid",
    "audit_cell",
    "content_hash",
]
