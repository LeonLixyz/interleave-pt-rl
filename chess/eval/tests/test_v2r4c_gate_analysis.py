from __future__ import annotations

from Eval import v2r4b_gate_analysis as corrected
from Eval.v2r4c_gate_analysis import (
    CONTRACT_VERSION,
    SCHEMA,
    analyze_grid,
    content_hash,
)


def test_v2r4c_wrapper_changes_only_identity_and_fresh_prompt_disclosure():
    cells = [
        {
            "candidate_step": step,
            "batch_label": batch,
            "absolute_gate_pass": True,
            "prompt_outcomes": [
                {
                    "prompt_fingerprint": (
                        f"{batch}-prompt-{group:04d}"
                    ),
                    "solve_at_8": int(group < 20),
                }
                for group in range(1_024)
            ],
        }
        for step in (6_000, 8_000, 9_920)
        for batch in ("A", "B")
    ]
    endpoints = {
        step: {
            "pt": {"state": "complete", "metrics": {"ce": 0.6}},
            "chess": {
                "state": "complete",
                "metrics": {"avg_reward": 0.01, "pass_at_1": 0.01},
            },
            "p2_sft": {
                "state": "complete",
                "metrics": {"ce": 0.5},
            },
        }
        for step in (6_000, 8_000, 9_920)
    }

    fresh = analyze_grid(cells, endpoints)
    prior = corrected.analyze_grid(cells, endpoints)
    assert fresh["schema"] == SCHEMA
    assert fresh["contract_version"] == CONTRACT_VERSION
    assert fresh["report_sha256"] == content_hash(
        fresh, "report_sha256"
    )
    assert fresh["fresh_prompt_design"] == {
        "quarantined_and_diagnostic_prompts_excluded": True,
        "batch_a_batch_b_intersection": 0,
        "batch_a_canary_intersection": 0,
        "batch_b_canary_intersection": 0,
        "thresholds_changed_after_prompt_selection": False,
    }
    for key in (
        "cells",
        "endpoints",
        "paired_comparisons",
        "behavioral_selection",
        "authorization",
        "status",
        "schema_version",
        "analysis_correction",
    ):
        assert fresh[key] == prior[key]
