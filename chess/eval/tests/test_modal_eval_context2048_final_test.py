from __future__ import annotations

import math

from context2048_eval_core import (
    deterministic_sample_seed,
    frame_prompt_ids,
    pass_at_k_curve,
    summarize_histogram,
    unbiased_pass_at_k,
)
from modal_eval_context2048_final_test import (
    BASE_SEED,
    CHECKPOINTS,
    DATASETS,
    EXPECTED_EVALUATED_PROMPTS,
    EXPECTED_RAW_PROMPTS,
    EXPECTED_SKIPPED_OVERLONG,
    MAX_MODEL_LEN,
    N_SAMPLES,
    PROMPT_CAP,
    RESPONSE_BUDGET,
    evaluation_contract,
)


def test_heldout_registry_is_exact_and_complete():
    assert tuple(DATASETS) == ("B1", "B2", "B3", "B4", "B5")
    assert sum(spec.rows for spec in DATASETS.values()) == EXPECTED_RAW_PROMPTS == 1484
    assert all(len(spec.sha256) == 64 for spec in DATASETS.values())
    assert EXPECTED_EVALUATED_PROMPTS == 1480
    assert EXPECTED_SKIPPED_OVERLONG == 4


def test_final_checkpoint_registry_has_unique_runs_and_commits():
    assert len(CHECKPOINTS) == 5
    assert len({spec.run_name for spec in CHECKPOINTS.values()}) == 5
    assert len(
        {spec.checkpoint_commit_sha256 for spec in CHECKPOINTS.values()}
    ) == 5
    assert all(
        len(spec.checkpoint_commit_sha256) == 64
        for spec in CHECKPOINTS.values()
    )
    assert (
        CHECKPOINTS["mixed_sft3_fresh_adam"].origin_subpath
        == CHECKPOINTS["mixed_sft3_continued_adam"].origin_subpath
    )


def test_native_context_contract_includes_exactly_one_bos():
    assert PROMPT_CAP + RESPONSE_BUDGET == MAX_MODEL_LEN == 2048
    assert frame_prompt_ids(
        [7] * (PROMPT_CAP - 1), bos_id=0, prompt_cap=PROMPT_CAP
    ) == [0, *([7] * (PROMPT_CAP - 1))]
    assert (
        frame_prompt_ids(
            [7] * PROMPT_CAP, bos_id=0, prompt_cap=PROMPT_CAP
        )
        is None
    )


def test_source_bos_fails_closed():
    try:
        frame_prompt_ids([7, 0, 8], bos_id=0, prompt_cap=PROMPT_CAP)
    except RuntimeError as exc:
        assert "contains BOS" in str(exc)
    else:
        raise AssertionError("source BOS should fail closed")


def test_pass_at_k_curve_uses_unbiased_estimator_for_every_k():
    histogram = {0: 1, 8: 1, 16: 1}
    curve = pass_at_k_curve(histogram, n=16)
    assert curve["1"] == 0.5
    assert math.isclose(curve["16"], 2 / 3)
    assert len(curve) == 16
    assert unbiased_pass_at_k(histogram, n=16, k=8) == curve["8"]


def test_histogram_summary_reports_degenerate_prompt_percentages():
    summary = summarize_histogram({0: 2, 8: 1, 16: 1}, n=16)
    assert summary["evaluated_prompts"] == 4
    assert summary["all_zero_prompts"] == 2
    assert summary["all_zero_percentage"] == 0.5
    assert summary["all_one_percentage"] == 0.25
    assert summary["mixed_outcome_prompts"] == 1


def test_sampling_seeds_are_stable_and_model_independent():
    kwargs = {
        "base_seed": BASE_SEED,
        "dataset_key": "B3",
        "row_index": 17,
        "sample_slot": 4,
        "generation_round": 2,
    }
    first = deterministic_sample_seed(**kwargs)
    assert first == deterministic_sample_seed(**kwargs)
    assert first != deterministic_sample_seed(**{**kwargs, "sample_slot": 5})
    assert first != deterministic_sample_seed(**{**kwargs, "generation_round": 3})
    assert 0 <= first <= 0x7FFF_FFFF


def test_production_contract_seals_all_five_buckets_and_samples():
    contract = evaluation_contract("production", "mixed_sft3_fresh_adam")
    assert [dataset["key"] for dataset in contract["datasets"]] == list(DATASETS)
    assert contract["generation"]["n_samples"] == N_SAMPLES == 16
    assert contract["generation"]["prompt_cap_including_bos"] == 512
    assert contract["generation"]["response_tokens"] == 1536
    assert contract["generation"]["inference_dtype"] == "bfloat16"
    assert contract["max_prompts_per_dataset"] == 0
