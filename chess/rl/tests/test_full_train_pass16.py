from __future__ import annotations

import json

import pytest

from chess_rl_miles.full_train_pass16 import (
    EXPECTED_TRAJECTORIES,
    SHARD_LENGTHS,
    SOURCE_ROWS,
    canonical_sha256,
    shard_ranges,
    validate_rollout_rows,
)


def test_shards_partition_every_source_row_once():
    ranges = shard_ranges()
    observed = [
        index
        for start, stop in ranges
        for index in range(start, stop)
    ]

    assert len(ranges) == 16
    assert tuple(stop - start for start, stop in ranges) == SHARD_LENGTHS
    assert observed == list(range(SOURCE_ROWS))
    assert EXPECTED_TRAJECTORIES == SOURCE_ROWS * 16


def test_validate_rollout_rows_requires_exact_16_siblings():
    rows = []
    for source_index in (9, 12):
        for slot in range(16):
            sample_index = source_index * 16 + slot
            score = float(source_index == 9 and slot == 3)
            rows.append(
                {
                    "group_index": source_index,
                    "sample_index": sample_index,
                    "score": score,
                    "reward": {"score": score},
                    "status": "completed",
                    "metadata": {
                        "source_row_index": source_index,
                        "pass_at_16_sample_slot": slot,
                        "pass_at_16_sample_index": sample_index,
                    },
                }
            )

    result = validate_rollout_rows(
        rows,
        expected_source_indices={9, 12},
    )

    assert result["rows"] == 32
    assert result["positive_trajectories"] == 1
    assert result["solved_prompts"] == 1
    assert result["success_count_histogram"]["0"] == 1
    assert result["success_count_histogram"]["1"] == 1


def test_validate_rollout_rows_rejects_duplicate_slot():
    rows = []
    for slot in range(16):
        sample_index = 5 * 16 + slot
        rows.append(
            {
                "group_index": 5,
                "sample_index": sample_index,
                "score": 0.0,
                "reward": {"score": 0.0},
                "status": "completed",
                "metadata": {
                    "source_row_index": 5,
                    "pass_at_16_sample_slot": slot,
                    "pass_at_16_sample_index": sample_index,
                },
            }
        )
    rows[-1] = json.loads(json.dumps(rows[-2]))

    with pytest.raises(ValueError, match="duplicate"):
        validate_rollout_rows(rows, expected_source_indices={5})


def test_canonical_sha256_is_key_order_independent():
    assert canonical_sha256({"b": 2, "a": 1}) == canonical_sha256(
        {"a": 1, "b": 2}
    )
