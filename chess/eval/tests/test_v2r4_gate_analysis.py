from __future__ import annotations

import copy

import pytest

from Eval.v2r4_gate_analysis import (
    BATCH_LABELS,
    CANDIDATE_STEPS,
    GROUPS_PER_CELL,
    ROWS_PER_CELL,
    analyze_grid,
    audit_cell,
    content_hash,
)


SEEDS = {"A": 1_567_877_051, "B": 923_570_888}


def _rows(batch: str, solved: int, *, protocol_rows: int = 443):
    rows = []
    for group in range(GROUPS_PER_CELL):
        for sibling in range(8):
            sample = group * 8 + sibling
            protocol = sample < protocol_rows
            positive = group < solved and sibling == 0
            assert not positive or protocol
            rows.append(
                {
                    "input": f"{batch}-prompt-{group}",
                    "FEN": f"{batch}-fen-{group}",
                    "PuzzleId": f"{batch}-puzzle-{group}",
                    "ground_truth": "e2e4",
                    "group_index": group,
                    "sample_index": sample,
                    "sampling_seed_sibling_index": sibling,
                    "sampling_seed": SEEDS[batch] + sample,
                    "status": "completed",
                    "score": float(positive),
                    "output": (
                        "thinking </T> then <call_env> move"
                        if protocol
                        else "thinking"
                    ),
                    "extracted_moves": "e2e4" if protocol else "",
                    "metadata": {
                        "sampling_seed_sibling_index": sibling,
                        "sampling_seed": SEEDS[batch] + sample,
                        "sampling_seed_mode": "sample-index",
                    },
                }
            )
    assert len(rows) == ROWS_PER_CELL
    return rows


def _grid():
    solved = {6_000: 24, 8_000: 18, 9_920: 20}
    return [
        audit_cell(
            _rows(batch, solved[step]),
            candidate_step=step,
            batch_label=batch,
            rollout_seed=SEEDS[batch],
        )
        for step in CANDIDATE_STEPS
        for batch in BATCH_LABELS
    ]


def _endpoints():
    return {
        step: {
            "pt": {"state": "complete", "metrics": {"ce": 0.6}},
            "chess": {
                "state": "complete",
                "metrics": {"avg_reward": 0.01, "pass_at_1": 0.01},
            },
            "p2_sft": {"state": "complete", "metrics": {"ce": 0.5}},
        }
        for step in CANDIDATE_STEPS
    }


def test_cell_absolute_gate_and_grid_report_are_exact_and_self_hashed():
    cells = _grid()
    assert all(cell["absolute_gate_pass"] for cell in cells)
    assert cells[0]["joint_valid_protocol_rows"] == 443
    assert cells[0]["status_counts"] == {
        "completed": ROWS_PER_CELL,
        "truncated": 0,
    }

    report = analyze_grid(cells, _endpoints())
    assert report["authorization"]["pass"] is True
    assert report["authorization"][
        "early_steps_can_authorize_original_experiment"
    ] is False
    assert len(report["paired_comparisons"]) == 3
    assert report["behavioral_selection"]["status"] == "unique_winner"
    assert report["behavioral_selection"]["winner_step"] == 6_000
    assert report["behavioral_selection"][
        "can_select_early_step_for_original_experiment"
    ] is False
    assert report["report_sha256"] == content_hash(
        report, "report_sha256"
    )
    for comparison in report["paired_comparisons"]:
        assert 0 <= comparison["holm_adjusted_p"] <= 1
        assert comparison["effect_direction_agrees_in_a_and_b"] is True


def test_cell_rejects_seed_status_and_unparsed_positive_tampering():
    rows = _rows("A", 20)
    rows[0]["sampling_seed"] += 1
    with pytest.raises(ValueError, match="top-level sample-index seed"):
        audit_cell(
            rows,
            candidate_step=9_920,
            batch_label="A",
            rollout_seed=SEEDS["A"],
        )

    rows = _rows("A", 20)
    rows[0]["status"] = "failed"
    with pytest.raises(ValueError, match="disallowed rollout status"):
        audit_cell(
            rows,
            candidate_step=9_920,
            batch_label="A",
            rollout_seed=SEEDS["A"],
        )

    rows = _rows("A", 20)
    rows[0]["output"] = "raw e2e4"
    rows[0]["extracted_moves"] = ""
    with pytest.raises(ValueError, match="joint-valid protocol"):
        audit_cell(
            rows,
            candidate_step=9_920,
            batch_label="A",
            rollout_seed=SEEDS["A"],
        )


def test_grid_rejects_unpaired_or_overlapping_prompt_batches():
    cells = _grid()
    bad = copy.deepcopy(cells)
    bad[0]["prompt_outcomes"][0]["prompt_fingerprint"] = "0" * 64
    with pytest.raises(ValueError, match="paired prompt order"):
        analyze_grid(bad, _endpoints())

    overlapping = copy.deepcopy(cells)
    for cell in overlapping:
        if cell["batch_label"] == "B":
            paired = next(
                value
                for value in overlapping
                if value["candidate_step"] == cell["candidate_step"]
                and value["batch_label"] == "A"
            )
            cell["prompt_outcomes"] = copy.deepcopy(
                paired["prompt_outcomes"]
            )
    with pytest.raises(ValueError, match="prompt batches overlap"):
        analyze_grid(overlapping, _endpoints())


def test_step_9920_requires_positive_chess_and_all_finite_endpoints():
    endpoints = _endpoints()
    endpoints[9_920]["chess"]["metrics"]["avg_reward"] = 0.0
    report = analyze_grid(_grid(), endpoints)
    assert report["authorization"]["pass"] is False
    assert report["authorization"]["checks"][
        "step_9920_b1_b5_has_positive"
    ] is False

    endpoints = _endpoints()
    endpoints[9_920]["p2_sft"]["metrics"]["ce"] = float("nan")
    with pytest.raises(ValueError, match="non-finite for step 9920"):
        analyze_grid(_grid(), endpoints)
