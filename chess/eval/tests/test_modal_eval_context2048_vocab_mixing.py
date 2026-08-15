from __future__ import annotations

import math

import numpy as np

from modal_eval_context2048_vocab_mixing import (
    CHECKPOINTS,
    CONTEXT_MARGIN,
    MAX_MODEL_LEN,
    N_SAMPLES,
    PROMPT_CAP,
    RESPONSE_BUDGET,
    RL_LR,
    RL_CHECKPOINTS,
    SHARD_COUNT,
    SOURCE_ROWS,
    _pass_at_k,
    common_mixed_outcome_indices,
    frame_prompt_ids,
    rl_launch_kwargs,
    shard_bounds,
    to_python_value,
)


def test_shards_cover_the_source_once_without_gaps():
    bounds = [shard_bounds(index) for index in range(SHARD_COUNT)]
    assert bounds[0][0] == 0
    assert bounds[-1][1] == SOURCE_ROWS
    assert all(left[1] == right[0] for left, right in zip(bounds, bounds[1:]))
    assert sum(stop - start for start, stop in bounds) == SOURCE_ROWS


def test_pass_at_k_uses_the_unbiased_estimator():
    histogram = {0: 1, 8: 1, 16: 1}
    assert _pass_at_k(histogram, n=16, k=1) == 0.5
    expected_pass16 = 2 / 3
    assert math.isclose(
        _pass_at_k(histogram, n=16, k=16), expected_pass16
    )


def test_all_four_checkpoints_have_rl_runs():
    assert RL_CHECKPOINTS == tuple(CHECKPOINTS)
    assert CHECKPOINTS["mixed_sft1"].rl_run_name
    assert CHECKPOINTS["mixed_sft3"].rl_run_name
    assert all(
        len(spec.expected_fingerprint) == 64 for spec in CHECKPOINTS.values()
    )


def test_rl_launch_is_native_context_offline_filtered_and_low_var_kl():
    summary = {
        "filtered_parquet": {
            "path": "/data/chess-rl-data/filtered.parquet",
            "sha256": "a" * 64,
        }
    }
    kwargs = rl_launch_kwargs("vocab85_then_sft3", summary)
    assert kwargs["lr"] == RL_LR == "1e-5"
    assert kwargs["dynamic_filter"] is False
    assert kwargs["num_rollout"] == 1500
    assert kwargs["train_file"] == summary["filtered_parquet"]["path"]
    assert kwargs["train_file_sha256"] == "a" * 64
    assert kwargs["kl_loss_type"] == "low_var_kl"
    assert kwargs["rollout_max_prompt_len"] == PROMPT_CAP == 512
    assert kwargs["rollout_max_response_len"] == RESPONSE_BUDGET == 1536
    assert kwargs["rollout_max_context_len"] == MAX_MODEL_LEN == 2048
    assert kwargs["max_tokens_per_gpu"] == 131_072
    assert kwargs["sglang_server_concurrency"] == 128
    assert N_SAMPLES == 16


def test_evaluation_matches_post_bos_prompt_and_exact_context_caps():
    assert CONTEXT_MARGIN == 0
    assert frame_prompt_ids([7] * 511, bos_id=0) == [0, *([7] * 511)]
    assert frame_prompt_ids([7] * 512, bos_id=0) is None


def test_source_bos_fails_closed_instead_of_becoming_a_duplicate():
    try:
        frame_prompt_ids([7, 0, 8], bos_id=0)
    except RuntimeError as exc:
        assert "contains BOS" in str(exc)
    else:
        raise AssertionError("source BOS should fail closed")


def test_pandas_array_metadata_matches_production_pyarrow_lists():
    value = {"env_replies": np.asarray([], dtype=object)}
    converted = to_python_value(value)
    assert converted == {"env_replies": []}
    assert bool(converted["env_replies"]) is False


def test_common_filter_is_the_identical_mixed_outcome_intersection():
    wins = {
        key: {0: 0, 1: 1, 2: 15, 3: 16, 4: 8}
        for key in CHECKPOINTS
    }
    wins["mixed_sft3"][2] = 16
    assert common_mixed_outcome_indices(wins) == [1, 4]


def test_common_filter_rejects_different_prompt_cohorts():
    wins = {key: {0: 1, 1: 2} for key in CHECKPOINTS}
    wins["mixed_sft1"] = {0: 1}
    try:
        common_mixed_outcome_indices(wins)
    except ValueError as exc:
        assert "cohorts differ" in str(exc)
    else:
        raise AssertionError("different prompt cohorts should fail closed")
