from __future__ import annotations

from Eval.v2r4b_gate_analysis import (
    CONTRACT_VERSION,
    GROUPS_PER_CELL,
    PROTOCOL_ROWS_MIN,
    ROWS_PER_CELL,
    SCHEMA,
    analyze_grid,
    audit_cell,
    content_hash,
)
from Eval import v2r4_gate_analysis as frozen


SEEDS = {"A": 1_567_877_051, "B": 923_570_888}


def _rows(batch: str, *, solved: int, invalid_positive: bool):
    seed = SEEDS[batch]
    rows = []
    for group in range(GROUPS_PER_CELL):
        for sibling in range(8):
            sample = group * 8 + sibling
            protocol = sample < PROTOCOL_ROWS_MIN
            positive = group < solved and sibling == 0
            if invalid_positive and sample == PROTOCOL_ROWS_MIN:
                positive = True
            rows.append(
                {
                    "input": f"{batch}-prompt-{group}",
                    "FEN": f"{batch}-fen-{group}",
                    "PuzzleId": f"{batch}-puzzle-{group}",
                    "ground_truth": "e2e4",
                    "group_index": group,
                    "sample_index": sample,
                    "sampling_seed_sibling_index": sibling,
                    "sampling_seed": seed + sample,
                    "status": "completed",
                    "score": float(positive),
                    "output": (
                        "thinking </T> then <call_env> move"
                        if protocol
                        else "e2e4 <call_env>"
                    ),
                    "extracted_moves": "e2e4",
                    "metadata": {
                        "sampling_seed_sibling_index": sibling,
                        "sampling_seed": seed + sample,
                        "sampling_seed_mode": "sample-index",
                    },
                }
            )
    assert len(rows) == ROWS_PER_CELL
    return rows


def test_positive_nonprotocol_row_counts_reward_without_weakening_protocol():
    cell = audit_cell(
        _rows("A", solved=20, invalid_positive=True),
        candidate_step=9_920,
        batch_label="A",
        rollout_seed=SEEDS["A"],
    )

    assert cell["joint_valid_protocol_rows"] == PROTOCOL_ROWS_MIN
    assert cell["positive_nonprotocol_rows"] == 1
    assert cell["positive_rows"] == 21
    assert cell["solve_at_8_groups"] == 21
    assert cell["absolute_gate_pass"] is True


def test_grid_wrapper_changes_only_identity_and_disclosed_correction():
    cells = [
        audit_cell(
            _rows(batch, solved=20, invalid_positive=False),
            candidate_step=step,
            batch_label=batch,
            rollout_seed=SEEDS[batch],
        )
        for step in (6_000, 8_000, 9_920)
        for batch in ("A", "B")
    ]
    endpoints = {
        step: {
            "pt": {
                "state": "complete",
                "metrics": {"ce": 0.6},
            },
            "chess": {
                "state": "complete",
                "metrics": {
                    "avg_reward": 0.01,
                    "pass_at_1": 0.01,
                },
            },
            "p2_sft": {
                "state": "complete",
                "metrics": {"ce": 0.5},
            },
        }
        for step in (6_000, 8_000, 9_920)
    }

    corrected = analyze_grid(cells, endpoints)
    prior = frozen.analyze_grid(cells, endpoints)
    assert corrected["schema"] == SCHEMA
    assert corrected["contract_version"] == CONTRACT_VERSION
    assert corrected["report_sha256"] == content_hash(
        corrected, "report_sha256"
    )
    for key in (
        "cells",
        "endpoints",
        "paired_comparisons",
        "behavioral_selection",
        "authorization",
        "status",
        "schema_version",
    ):
        assert corrected[key] == prior[key]
