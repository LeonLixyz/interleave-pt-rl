import json
from pathlib import Path

from training.positive_replay import (
    ExtractionConfig,
    _selection_priority,
    canonical_json,
    extract_positive_replay,
    sha256_bytes,
    sha256_file,
    token_ids_sha256,
    validate_positive_row,
)


def _row(
    *,
    group_index=0,
    sample_index=0,
    response="analysis </T> Ke7-e5 <call_env>",
    score=1.0,
):
    prompt_ids = [1, 2]
    response_ids = [3, 4, 5, 6]
    return {
        "input": "1. e4 <T>",
        "output": response,
        "score": score,
        "reward": {
            "score": score,
            "extracted_moves": "e7e5",
        },
        "extracted_moves": "e7e5",
        "status": "completed",
        "sampling_scope": "all_completed_attempts_before_dynamic_filter",
        "step": 17,
        "rollout_id": 16,
        "group_index": group_index,
        "sample_index": sample_index,
        "weight_versions": ["16"],
        "response_length": len(response_ids),
        "token_artifact_schema": 1,
        "prompt_token_ids": prompt_ids,
        "response_token_ids": response_ids,
        "response_loss_mask": [1, 1, 1, 0],
        "token_ids_sha256": token_ids_sha256(prompt_ids, response_ids),
        "metadata": {
            "FEN": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "Moves": "e2e4 e7e5",
            "PuzzleId": "unit",
            "Rating": 1200,
        },
    }


def _config(seed=42):
    return ExtractionConfig(
        run_id="e1_u_rl1",
        policy_checkpoint="iter_0001500",
        filter_setting="U",
        extraction_seed=seed,
    )


def _write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def test_qc_requires_legal_completed_structured_lossless_success():
    valid = _row()
    result = validate_positive_row(valid, _config())
    assert result.accepted
    assert result.record["puzzle_id"] == "unit"
    assert result.record["difficulty"] == 1200
    assert result.record["weight_versions"] == ["16"]

    invalid = _row()
    invalid["metadata"]["Moves"] = "e2e5 e7e5"
    result = validate_positive_row(invalid, _config())
    assert result.rejection_reason == "illegal_trajectory"

    invalid = _row(response="missing close <call_env>")
    result = validate_positive_row(invalid, _config())
    assert result.rejection_reason == "think_end_count_not_one"

    invalid = _row()
    invalid["response_loss_mask"] = [1, 1]
    result = validate_positive_row(invalid, _config())
    assert result.rejection_reason == "response_mask_length_mismatch"


def test_extraction_selects_one_per_group_then_exactly_deduplicates(tmp_path):
    source = tmp_path / "rollout_1.jsonl"
    duplicate_response = "same </T> Ke7-e5 <call_env>"
    rows = [
        _row(group_index=5, sample_index=40, response=duplicate_response),
        _row(group_index=5, sample_index=41, response=duplicate_response),
        _row(group_index=6, sample_index=48, response=duplicate_response),
        _row(group_index=7, sample_index=56, score=0.0),
    ]
    _write_jsonl(source, rows)
    output = tmp_path / "positive.jsonl"

    manifest = extract_positive_replay(
        [source],
        output_path=output,
        manifest_path=None,
        config=_config(),
    )

    output_rows = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert len(output_rows) == 1
    assert output_rows[0]["group_index"] == 5
    assert output_rows[0]["eligible_siblings_in_group"] == 2
    assert manifest["counters"]["input_rows"] == 4
    assert manifest["counters"]["eligible_rows"] == 3
    assert manifest["counters"]["eligible_groups"] == 2
    assert manifest["counters"]["selected_before_exact_dedupe"] == 2
    assert manifest["counters"]["exact_prompt_response_duplicates_dropped"] == 1
    assert manifest["counters"]["output_rows"] == 1
    assert manifest["output"]["sha256"] == sha256_file(output)


def test_seed_keyed_hash_selection_matches_declared_priority(tmp_path):
    source = tmp_path / "rollout_9.jsonl"
    rows = [
        _row(
            group_index=22,
            sample_index=100 + index,
            response=f"candidate {index} </T> Ke7-e5 <call_env>",
        )
        for index in range(8)
    ]
    _write_jsonl(source, rows)
    source_sha = sha256_file(source)
    seed = 31415
    priorities = []
    for line_number, row in enumerate(rows, start=1):
        row_sha = sha256_bytes(canonical_json(row).encode("utf-8"))
        priorities.append(
            (
                _selection_priority(
                    seed=seed,
                    run_id="e1_u_rl1",
                    group_index=22,
                    sample_index=row["sample_index"],
                    source_file_sha256=source_sha,
                    source_line=line_number,
                    source_row_sha256=row_sha,
                ),
                row["sample_index"],
            )
        )
    expected_sample = min(priorities)[1]
    output = tmp_path / "selected.jsonl"

    extract_positive_replay(
        [source],
        output_path=output,
        manifest_path=None,
        config=_config(seed),
    )
    selected = json.loads(output.read_text(encoding="utf-8"))

    assert selected["sample_index"] == expected_sample
