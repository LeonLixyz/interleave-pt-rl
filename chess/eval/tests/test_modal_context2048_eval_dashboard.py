from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from modal_context2048_eval_dashboard import (
    CHECKPOINTS,
    EVAL_HTML,
    N_SAMPLES,
    RL_RUNS,
    RL_TARGET_UPDATES,
    RL_WANDB_GROUP,
    RL_WANDB_PROJECT,
    SOURCE_SHA256,
    TRAINING_STAGES,
    VALIDATION_TARGET_TOKENS,
    VALIDATION_VERSION,
    VERSION,
    _aggregate_checkpoint,
    _canonical_sha256,
    _collect_rl_launch_records,
    _find_modal_call_status,
    _history_points,
    _pass_at_k,
    _read_heldout_validation,
    _read_training_stage,
    _rl_checkpoint_status,
)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_pass_at_k_uses_grouped_sample_counts():
    histogram = {0: 1, 8: 1, 16: 1}
    assert _pass_at_k(histogram, n=16, k=1) == 0.5
    assert _pass_at_k(histogram, n=16, k=16) == 2 / 3


def test_aggregate_checkpoint_authenticates_and_merges_shards(tmp_path: Path):
    key = "vocab85_then_sft3"
    evaluated_per_shard = (13_268, 13_295, 13_294, 13_299)
    skipped_per_shard = (38, 11, 12, 8)
    for shard_id, evaluated in enumerate(evaluated_per_shard):
        core = {
            "schema": "context2048-pass16-shard-summary-v1",
            "version": VERSION,
            "checkpoint": key,
            "checkpoint_fingerprint": CHECKPOINTS[key]["fingerprint"],
            "shard_id": shard_id,
            "source_sha256": SOURCE_SHA256,
            "evaluated_prompts": evaluated,
            "skipped_overlong": list(range(skipped_per_shard[shard_id])),
            "trajectories": evaluated * N_SAMPLES,
            "n_samples": N_SAMPLES,
            "format_rate": 1.0,
            "wins_histogram": {"8": evaluated},
            "generation": {
                "bos_prepended_exactly_once_by_evaluator": True,
                "prompt_cap_including_bos": 512,
                "dataset_prefilter_cap_excluding_bos": 511,
                "response_budget": 1_536,
                "model_context": 2_048,
                "context_margin": 0,
            },
            "finished_at": f"2026-08-13T00:00:0{shard_id}+00:00",
        }
        marker = {**core, "summary_sha256": _canonical_sha256(core)}
        _write_json(
            tmp_path / key / "n16" / f"shard-{shard_id:02d}" / "success.json",
            marker,
        )

    filter_core = {
        "schema": "context2048-mixed-outcome-filter-v1",
        "version": VERSION,
        "checkpoint": key,
        "checkpoint_fingerprint": CHECKPOINTS[key]["fingerprint"],
        "rule": "1 <= success_count <= 15 from exactly 16 samples",
        "filtered_parquet": {"rows": 53_156, "sha256": "a" * 64},
        "created_at": "2026-08-13T00:01:00+00:00",
    }
    _write_json(
        tmp_path / key / "filter" / "success.json",
        {**filter_core, "filter_sha256": _canonical_sha256(filter_core)},
    )

    result = _aggregate_checkpoint(tmp_path, key)
    assert result["state"] == "complete"
    assert result["metrics"]["evaluated_prompts"] == 53_156
    assert result["metrics"]["skipped_overlong"] == 69
    assert result["metrics"]["pass_at_1"] == 0.5
    assert result["metrics"]["pass_at_16"] == 1.0
    assert result["filter"]["rows"] == 53_156


def test_incomplete_checkpoint_does_not_publish_partial_pass_rates(tmp_path: Path):
    result = _aggregate_checkpoint(tmp_path, "mixed_sft3")
    assert result["state"] == "running"
    assert result["shards_complete"] == 0
    assert result["metrics"] is None


def test_training_stage_parses_only_authenticated_train_metrics(tmp_path: Path):
    key = "mixed_sft1"
    spec = dict(TRAINING_STAGES[key][0])
    spec["target_steps"] = 20
    stage_root = tmp_path / spec["relative_root"]
    state = {
        "global_step": 20,
        "configured_provenance": {
            "experiment_version": (
                "context2048_vocab_mixing_fp32_master_v13_20260813"
            ),
            "experiment": key,
            "stage": "mixed",
            "context_length": 2048,
            "vocab_size": 85,
        },
    }
    _write_json(stage_root / "final" / "interleaved_training_state.json", state)
    stage_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for step, pt, sft in ((10, 0.6, 0.8), (20, 0.5, 0.7)):
        rows.append(
            {
                "schema": "interleaved-local-metrics-v1",
                "manifest_hash": spec["manifest_hash"],
                "step": step,
                "metrics": {
                    "train/loss": (pt + sft) / 2,
                    "train/lr": 1e-5,
                    "train/pretrain_token_loss": pt,
                    "train/sft_token_loss": sft,
                    "train/global_pretrain_valid_tokens": 100,
                    "train/global_sft_valid_tokens": 20,
                    "train/effective_sft_loss_mass_share": 1 / 6,
                    "train/token_positions_per_second": 1_000,
                },
            }
        )
    (stage_root / "metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    result = _read_training_stage(tmp_path, key, spec)
    assert result["state"] == "complete"
    assert result["last"]["pretrain_token_loss"] == 0.5
    assert result["last"]["sft_token_loss"] == 0.7
    assert result["tail_10_mean"]["pretrain_token_loss"] == 0.55


def test_heldout_validation_result_is_self_authenticated(tmp_path: Path):
    key = "mixed_sft3"
    metrics = {
        "heldout_pretrain_loss": 0.5,
        "heldout_pretrain_perplexity": 1.6487212707,
        "heldout_pretrain_token_accuracy": 0.8,
        "heldout_pretrain_correct_tokens": round(VALIDATION_TARGET_TOKENS * 0.8),
        "heldout_pretrain_target_tokens": VALIDATION_TARGET_TOKENS,
    }
    core = {
        "schema": "context2048-heldout-pt-result-v1",
        "version": VALIDATION_VERSION,
        "state": "complete",
        "checkpoint": key,
        "checkpoint_path": CHECKPOINTS[key]["validation_checkpoint_path"],
        "checkpoint_fingerprint": CHECKPOINTS[key]["fingerprint"],
        "holdout_hash": "b" * 64,
        "metrics": metrics,
        "finished_at": "2026-08-13T00:00:00+00:00",
    }
    _write_json(
        tmp_path / key / "success.json",
        {**core, "result_sha256": _canonical_sha256(core)},
    )
    result = _read_heldout_validation(tmp_path, key)
    assert result["state"] == "complete"
    assert result["metrics"]["heldout_pretrain_loss"] == 0.5


def test_rl_launch_records_are_authenticated_and_match_the_filtered_data(
    tmp_path: Path,
):
    filtered = {
        "path": "/data/chess-rl-data/common.parquet",
        "sha256": "a" * 64,
        "rows": 28_419,
        "bytes": 27_334_080,
    }
    common_filter_core = {
        "schema": "context2048-common-mixed-outcome-filter-v1",
        "version": VERSION,
        "n_samples_per_checkpoint": 16,
        "comparison_contract": (
            "all four RL runs must use this exact parquet and SHA-256"
        ),
        "source": {"rows": 53_225, "sha256": "bcf131d88cd9916f6ec12a12f8e85901a05e0dabc5d5a1fdebd8ea0b0ae86c30"},
        "filtered_parquet": filtered,
    }
    _write_json(
        tmp_path / "common_filter" / "success.json",
        {
            **common_filter_core,
            "filter_sha256": _canonical_sha256(common_filter_core),
        },
    )
    calls = []
    for index, (key, spec) in enumerate(RL_RUNS.items()):
        calls.append(
            {
                "checkpoint": key,
                "run_name": spec["run_name"],
                "function_call_id": f"fc-{index + 1:026d}",
                "wandb_project": RL_WANDB_PROJECT,
                "wandb_group": spec.get("wandb_group", RL_WANDB_GROUP),
                "filtered_parquet": filtered,
            }
        )
    ledger_core = {
        "schema": "context2048-pass16-rl-pipeline-v1",
        "version": VERSION,
        "state": "rl_launched",
        "created_at": "2026-08-13T00:00:00+00:00",
        "finished_at": "2026-08-13T00:01:00+00:00",
        "rl_calls": calls,
    }
    _write_json(
        tmp_path / "pipeline.json",
        {**ledger_core, "ledger_sha256": _canonical_sha256(ledger_core)},
    )

    records, errors = _collect_rl_launch_records(tmp_path)
    assert errors == []
    assert tuple(records) == tuple(RL_RUNS)
    assert records["mixed_sft3"]["filtered_parquet"]["rows"] == 28_419
    assert records["mixed_sft3"]["modal_url"].startswith("https://modal.com/id/fc-")


def test_wandb_history_rows_merge_by_rollout_and_train_step():
    points = _history_points(
        [
            [
                {
                    "rollout/step": 7,
                    "rollout/raw_reward": 0.25,
                    "rollout/entropy": 0.08,
                }
            ],
            [
                {
                    "rollout/step": 7,
                    "passrate/pass@1": 0.25,
                    "passrate/pass@2": 0.4,
                    "passrate/pass@4": 0.6,
                    "passrate/pass@8": 0.75,
                }
            ],
            [{"train/step": 7, "train/grad_norm": 0.3, "train/ppo_kl": -0.01}],
            [
                {
                    "rollout/step": 7,
                    "rollout/zero_std/all_zero_percentage": 0.2,
                }
            ],
        ]
    )
    assert points == [
        {
            "step": 7,
            "reward": 0.25,
            "entropy": 0.08,
            "pass_at_1": 0.25,
            "pass_at_2": 0.4,
            "pass_at_4": 0.6,
            "pass_at_8": 0.75,
            "grad_norm": 0.3,
            "ppo_kl": -0.01,
            "all_zero_percentage": 0.2,
        }
    ]


def test_rl_checkpoint_and_nested_modal_call_status_are_validated(tmp_path: Path):
    run_name = RL_RUNS["mixed_sft1"]["run_name"]
    checkpoint = tmp_path / run_name / "iter_0000040"
    files = {
        "model/.metadata": b"model metadata",
        "model/__0_0.distcp": b"model shard",
        "optimizer/.metadata": b"optimizer metadata",
        "optimizer/__0_0.distcp": b"optimizer shard",
        "lr_scheduler/.metadata": b"scheduler metadata",
        "lr_scheduler/__0_0.distcp": b"scheduler shard",
        "rollout_state.pt": b"rollout state",
        "rng_rank_00000.pt": b"rng zero",
        "rng_rank_00001.pt": b"rng one",
    }
    for relative, contents in files.items():
        path = checkpoint / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
    _write_json(checkpoint / "meta.json", {"iteration": 40, "next_rollout_id": 40})
    payload = []
    for path in sorted(item for item in checkpoint.rglob("*") if item.is_file()):
        payload.append(
            {
                "path": path.relative_to(checkpoint).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": "0" * 64,
            }
        )
    marker_core = {
        "schema": "miles-fsdp-checkpoint-commit-v1",
        "iteration": 40,
        "optimizer_included": True,
        "rng_included": True,
        "rollout_state_included": True,
        "world_size": 2,
        "payload": payload,
    }
    marker = {**marker_core, "commit_sha256": _canonical_sha256(marker_core)}
    _write_json(checkpoint / "COMMITTED.json", marker)
    (tmp_path / run_name / "latest_checkpointed_iteration.txt").write_text("40")
    status = _rl_checkpoint_status(tmp_path, run_name)
    assert status == {
        "state": "checkpointed",
        "step": 40,
        "validated": True,
        "commit_sha256": marker["commit_sha256"],
    }

    wanted = "fc-01KZTEST"
    graph = [
        SimpleNamespace(
            function_call_id="fc-PARENT",
            status=SimpleNamespace(name="SUCCESS"),
            children=[
                SimpleNamespace(
                    function_call_id=wanted,
                    status=SimpleNamespace(name="PENDING"),
                    children=[],
                )
            ],
        )
    ]
    assert _find_modal_call_status(graph, wanted) == "pending"


def test_dashboard_distinguishes_pre_rl_pass16_and_exposes_rl_diagnostics():
    assert "Pre-RL full-dataset Pass@k" in EVAL_HTML
    assert "Pass@16 here is not an RL-training curve" in EVAL_HTML
    assert "RL training results" in EVAL_HTML
    for field in (
        "reward",
        "pass_at_1",
        "pass_at_2",
        "pass_at_4",
        "pass_at_8",
        "entropy",
        "grad_norm",
        "ppo_kl",
        "all_zero_percentage",
    ):
        assert f'data-field="{field}"' in EVAL_HTML
    assert "Raw + MA(25)" in EVAL_HTML
    assert RL_TARGET_UPDATES == 1_500
