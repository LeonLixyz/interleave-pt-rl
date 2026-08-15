from __future__ import annotations

import pytest

from training.v2r2_gate import AUDIT_GROUPS
from training.v2r3_diagnostics import (
    RAW_MOVE_TOKEN_MIN,
    audit_diagnostic_rows,
    raw_move_token_count,
)


def _rows():
    rows = []
    for group in range(AUDIT_GROUPS):
        for sample in range(8):
            length = group * 8 + sample + 1
            raw = group == 0 and sample < 2
            output = (
                "Ke1d2 Pb5b4 Nc3b1"
                if raw
                else "ordinary continuation"
            )
            rows.append(
                {
                    "status": "completed",
                    "group_index": group,
                    "sample_index": group * 8 + sample,
                    "input": f"prompt-{group} <T>",
                    "FEN": f"fen-{group}",
                    "PuzzleId": f"puzzle-{group}",
                    "ground_truth": "e2e4",
                    "output": output,
                    "extracted_moves": "",
                    "score": 0.0,
                    "response_length": length,
                    "effective_response_length": length,
                    "model_token_count": length,
                    "env_token_count": 0,
                    "sampling_seed": 42 + sample,
                    "sampling_seed_sibling_index": sample,
                    "metadata": {
                        "model_token_count": length,
                        "env_token_count": 0,
                        "sampling_seed": 42 + sample,
                        "sampling_seed_sibling_index": sample,
                    },
                }
            )
    return rows


def test_raw_move_metric_and_response_distribution_are_predeclared():
    metrics = audit_diagnostic_rows(_rows(), seed=42)
    assert metrics["response_length_mean"] == 1024.5
    assert metrics["response_length_min"] == 1
    assert metrics["response_length_p50_nearest_rank"] == 1024
    assert metrics["response_length_p90_nearest_rank"] == 1844
    assert metrics["response_length_p99_nearest_rank"] == 2028
    assert metrics["response_length_max"] == 2048
    assert metrics["effective_response_length_mean"] == 1024.5
    assert metrics["effective_response_length_max"] == 2048
    assert metrics["env_token_count_mean"] == 0
    assert metrics["env_token_count_max"] == 0
    assert metrics["status_counts"] == {
        "completed": 2048,
        "truncated": 0,
    }
    assert metrics["sampling_seed_groups_verified"] == 256
    assert metrics["raw_move_without_protocol_rows"] == 2
    assert metrics["raw_move_without_protocol_rate"] == 2 / 2048
    assert metrics["raw_move_tokens_in_flagged_rows"] == 6
    assert metrics["outputs_with_end_thinking"] == 0
    assert metrics["outputs_with_call_env"] == 0
    assert metrics["rows_with_parsed_moves"] == 0


def test_raw_move_requires_two_exact_move_tokens_and_no_valid_protocol():
    assert raw_move_token_count("Ke1d2 Pb5b4") == RAW_MOVE_TOKEN_MIN
    assert raw_move_token_count("not-a-move e2e4") == 1
    for token in (
        "e2e4",
        "e2-e4",
        "e5xd6",
        "Ke1d2",
        "e7e8q",
        "e7e8=Q",
        "O-O",
        "O-O-O",
    ):
        assert raw_move_token_count(token) == 1
    assert raw_move_token_count("e7e8x") == 0
    rows = _rows()
    rows[0]["output"] = "reason </T> <call_env> e2e4 Ke1d2 Pb5b4"
    rows[0]["extracted_moves"] = "e2e4"
    metrics = audit_diagnostic_rows(rows, seed=42)
    assert metrics["raw_move_without_protocol_rows"] == 1
    assert metrics["outputs_with_end_thinking"] == 1
    assert metrics["outputs_with_call_env"] == 1
    assert metrics["rows_with_parsed_moves"] == 1

    rows[0]["extracted_moves"] = None
    rows[0]["reward"] = {"extracted_moves": "e2e4"}
    metrics = audit_diagnostic_rows(rows, seed=42)
    assert metrics["rows_with_parsed_moves"] == 1


def test_multiturn_env_tokens_have_distinct_total_and_model_lengths():
    rows = _rows()
    row = rows[0]
    row["output"] = "reason </T> <call_env> e2e4"
    row["extracted_moves"] = "e2e4"
    row["env_token_count"] = 7
    row["metadata"]["env_token_count"] = 7
    row["response_length"] = row["model_token_count"] + 7
    metrics = audit_diagnostic_rows(rows, seed=42)
    assert metrics["effective_response_length_mean"] == 1024.5
    assert metrics["env_token_count_mean"] == 7 / 2048
    assert metrics["env_token_count_max"] == 7
    assert metrics["response_length_mean"] == 1024.5 + 7 / 2048


def test_truncated_model_cap_row_is_auditable_and_counted():
    rows = _rows()
    row = rows[0]
    row["status"] = "truncated"
    row["effective_response_length"] = 2560
    row["model_token_count"] = 2560
    row["metadata"]["model_token_count"] = 2560
    row["env_token_count"] = 11
    row["metadata"]["env_token_count"] = 11
    row["response_length"] = 2571
    metrics = audit_diagnostic_rows(rows, seed=42)
    assert metrics["status_counts"] == {
        "completed": 2047,
        "truncated": 1,
    }
    assert metrics["status_rates"]["truncated"] == 1 / 2048
    assert metrics["model_response_at_cap_rows"] == 1
    assert metrics["model_response_at_cap_rate"] == 1 / 2048
    assert metrics["response_length_max"] == 2571
    assert metrics["effective_response_length_max"] == 2560


def test_deterministic_seed_indices_must_cover_each_group_once():
    rows = _rows()
    row = rows[0]
    row["sampling_seed_sibling_index"] = 1
    row["metadata"]["sampling_seed_sibling_index"] = 1
    row["sampling_seed"] = 43
    row["metadata"]["sampling_seed"] = 43
    row["sample_index"] = 1
    with pytest.raises(ValueError, match="not exactly 0..7"):
        audit_diagnostic_rows(rows, seed=42)


def test_sample_index_must_bind_group_and_sibling_identity():
    rows = _rows()
    rows[0]["sample_index"] = 8
    with pytest.raises(ValueError, match="sample/sibling/group index"):
        audit_diagnostic_rows(rows, seed=42)


@pytest.mark.parametrize(
    "mutation,error",
    [
        (
            lambda row: row.__setitem__("response_length", 1.5),
            "invalid response_length",
        ),
        (
            lambda row: row["metadata"].__setitem__(
                "model_token_count", 999
            ),
            "effective_response_length disagrees",
        ),
        (
            lambda row: row.__setitem__("model_token_count", 999),
            "effective_response_length disagrees",
        ),
        (
            lambda row: row.__setitem__(
                "effective_response_length",
                row["response_length"] + 1,
            ),
            "effective_response_length disagrees",
        ),
        (
            lambda row: row["metadata"].__setitem__("env_token_count", 9),
            "env_token_count disagree",
        ),
        (
            lambda row: row.__setitem__("env_token_count", 9),
            "env_token_count disagree",
        ),
        (
            lambda row: row.__setitem__(
                "response_length", row["response_length"] + 1
            ),
            "model_token_count \\+ env_token_count",
        ),
        (
            lambda row: (
                row.__setitem__("effective_response_length", 2561),
                row.__setitem__("model_token_count", 2561),
                row["metadata"].__setitem__("model_token_count", 2561),
                row.__setitem__("response_length", 2561),
            ),
            "exceeds the 2560-token cap",
        ),
        (
            lambda row: row.__setitem__("status", "pending"),
            "disallowed status",
        ),
        (
            lambda row: row["metadata"].__setitem__("sampling_seed", 99),
            "sibling seed identity",
        ),
        (
            lambda row: row.__setitem__(
                "sampling_seed_sibling_index", 7
            ),
            "sibling seed identity",
        ),
    ],
)
def test_response_length_integrity_fails_closed(mutation, error):
    rows = _rows()
    mutation(rows[0])
    with pytest.raises(ValueError, match=error):
        audit_diagnostic_rows(rows, seed=42)
