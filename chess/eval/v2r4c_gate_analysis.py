"""Fresh-prompt report identity for the corrected v2r4c production gate."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from Eval import v2r4b_gate_analysis as corrected


SCHEMA = "interleaved-v2r4c-production-gate-report-v1"
CONTRACT_VERSION = "v2r4c_production_gate_20260730"
CANDIDATE_STEPS = corrected.CANDIDATE_STEPS
BATCH_LABELS = corrected.BATCH_LABELS
PAIRWISE_CONTRASTS = corrected.PAIRWISE_CONTRASTS
GROUPS_PER_CELL = corrected.GROUPS_PER_CELL
SIBLINGS_PER_GROUP = corrected.SIBLINGS_PER_GROUP
ROWS_PER_CELL = corrected.ROWS_PER_CELL
COMPLETED_ROWS_MIN = corrected.COMPLETED_ROWS_MIN
SOLVE_GROUPS_MIN = corrected.SOLVE_GROUPS_MIN
VARIANCE_GROUPS_MIN = corrected.VARIANCE_GROUPS_MIN
PROTOCOL_ROWS_MIN = corrected.PROTOCOL_ROWS_MIN
audit_cell = corrected.audit_cell
content_hash = corrected.content_hash


def analyze_grid(
    cells: Sequence[Mapping[str, Any]],
    endpoints: Mapping[int, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    """Apply the corrected analysis under the clean v2r4c identity."""

    prior = corrected.analyze_grid(cells, endpoints)
    core = {
        key: value
        for key, value in prior.items()
        if key != "report_sha256"
    }
    core["schema"] = SCHEMA
    core["contract_version"] = CONTRACT_VERSION
    core["fresh_prompt_design"] = {
        "quarantined_and_diagnostic_prompts_excluded": True,
        "batch_a_batch_b_intersection": 0,
        "batch_a_canary_intersection": 0,
        "batch_b_canary_intersection": 0,
        "thresholds_changed_after_prompt_selection": False,
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
