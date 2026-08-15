from __future__ import annotations

import json
from pathlib import Path

from Eval.interleave_dashboard_schema import (
    build_result_rows,
    flatten_core_registry,
    parse_interleave_marker_path,
    select_terminal_marker,
    summarize_metrics,
)
from Eval.interleave_live_schema import (
    PRETRAIN_TOKEN_POSITIONS_PER_STEP,
    build_live_feed,
    canonical_json_sha256,
    parse_pretrain_metrics_jsonl,
    validate_live_feed,
    validate_pretrain_trainer_state,
)


ROOT = Path(__file__).resolve().parents[2]


def _registry() -> dict:
    return json.loads((ROOT / "INTERLEAVED_CORE_REGISTRY.json").read_text())


def test_registry_flattens_six_arms_into_eight_rl_phases() -> None:
    stages = flatten_core_registry(_registry())

    assert len(stages) == 8
    assert len({stage["run_name"] for stage in stages}) == 8
    e1_dynamic_rl2 = next(
        stage
        for stage in stages
        if stage["arm"] == "E1-D" and stage["phase"] == "RL2"
    )
    assert e1_dynamic_rl2["filter_mode"] == "dynamic"
    assert e1_dynamic_rl2["target_step"] == 1_500
    assert e1_dynamic_rl2["effective_step_offset"] == 1_500


def test_marker_parser_accepts_existing_and_interleave_namespaces() -> None:
    existing = parse_interleave_marker_path(
        "v1/core-e2-u-rl3000-seed42/global_step_40/"
        "production_467f569b87ae/_SUCCESS.json"
    )
    future = parse_interleave_marker_path(
        "/interleave_v1/core-e1-d-rl2-seed43/global_step_80/"
        "production_deadbeef1234/_RUNNING.json"
    )

    assert existing == {
        "namespace": "v1",
        "run_name": "core-e2-u-rl3000-seed42",
        "step": 40,
        "profile": "production_467f569b87ae",
        "marker": "_SUCCESS.json",
    }
    assert future is not None
    assert future["step"] == 80
    assert future["marker"] == "_RUNNING.json"
    assert parse_interleave_marker_path("v1/run/global_step_40/smoke_x/_SUCCESS.json") is None


def test_terminal_marker_precedence_and_metric_fallback() -> None:
    state, marker = select_terminal_marker(
        {
            "_QUEUED.json": {"path": "queued"},
            "_RUNNING.json": {"path": "running"},
            "_SUCCESS.json": {"path": "success"},
        }
    )
    assert state == "success"
    assert marker["path"] == "success"

    metrics = {}
    for index, benchmark in enumerate(("B1", "B2", "B3", "B4", "B5"), start=1):
        metrics[f"val-core/test_{benchmark}/reward/mean@16"] = index / 10
    summary = summarize_metrics(metrics)
    assert summary["pass_at_1"] == 0.3
    assert summary["avg_reward"] == 0.3
    assert summary["b3_b4_avg"] == 0.35
    assert summary["pass_at_1_semantics"] == "binary_reward_mean@16_fallback"


def test_rows_apply_e1_phase_two_effective_step_offset() -> None:
    stages = flatten_core_registry(_registry())
    stage = next(
        item
        for item in stages
        if item["arm"] == "E1-U" and item["phase"] == "RL2"
    )
    rows = build_result_rows(
        [stage],
        {stage["run_name"]: [40]},
        {
            (stage["run_name"], 40): {
                "state": "success",
                "metrics": {
                    "pass_at_1": 0.25,
                    "avg_reward": 0.25,
                    "b3_b4_avg": 0.3,
                    "pass_at_1_semantics": "binary_reward_mean@16_fallback",
                },
            }
        },
    )

    assert rows == [
        {
            "model": "interleave_47m_qwen3",
            "experiment": "E1",
            "arm": "E1-U",
            "filter": "U",
            "filter_mode": "unfiltered",
            "phase": "RL2",
            "run_name": "core-e1-u-rl2-seed43",
            "phase_step": 40,
            "effective_rl_step": 1_540,
            "training_status": "checkpointed",
            "eval_status": "success",
            "pass_at_1": 0.25,
            "avg_reward": 0.25,
            "b3_b4_avg": 0.3,
            "pass_at_1_semantics": "binary_reward_mean@16_fallback",
        }
    ]


def _metric_line(step: int, *, manifest_hash: str = "a" * 64) -> str:
    return json.dumps(
        {
            "schema": "interleaved-local-metrics-v1",
            "step": step,
            "manifest_hash": manifest_hash,
            "runtime_provenance": {
                "attention_backend": "sdpa",
                "torch_compile_mode": "none",
                "torch_version": "2.9.0+cu128",
                "transformers_version": "4.57.0",
                "flash_attention_version": None,
                "data_num_workers": 8,
            },
            "metrics": {
                "train/loss": 0.75,
                "train/lr": 1e-3,
                "train/global_valid_tokens": 500_000,
                "train/token_positions_per_second": 1_000_000,
                "train/manifest_cursor": step,
            },
        }
    )


def test_pretrain_metric_and_resume_state_validation() -> None:
    summary = parse_pretrain_metrics_jsonl(
        _metric_line(10) + "\n" + _metric_line(20) + "\n",
        target_step=100,
        last_update_at="2026-07-30T08:00:00+00:00",
    )
    assert summary["step"] == 20
    assert summary["metric_records"] == 2
    assert summary["eta_seconds"] == (
        80 * PRETRAIN_TOKEN_POSITIONS_PER_STEP / 1_000_000
    )
    state = validate_pretrain_trainer_state(
        {
            "schema_version": 1,
            "global_step": 10,
            "manifest_hash": "a" * 64,
            "manifest_cursor": 10,
            "local_batch_size": 21,
            "world_size": 8,
            "gradient_accumulation_steps": 1,
            "runtime_provenance": summary["runtime_provenance"],
        },
        metrics=summary,
    )
    assert state["step"] == 10


def test_pretrain_metric_parser_fails_closed_on_malformed_tail_and_drift() -> None:
    import pytest

    with pytest.raises(ValueError, match="malformed JSON"):
        parse_pretrain_metrics_jsonl(
            _metric_line(10) + "\n{\"incomplete\":",
            target_step=100,
            last_update_at=None,
        )
    with pytest.raises(ValueError, match="manifest_hash changed"):
        parse_pretrain_metrics_jsonl(
            _metric_line(10) + "\n" + _metric_line(20, manifest_hash="b" * 64),
            target_step=100,
            last_update_at=None,
        )


def test_live_feed_is_content_addressed_and_rejects_tampering() -> None:
    import pytest

    registry = _registry()
    pretraining = {"stages": {"p1": {"step": 10}}}
    feed = build_live_feed(
        registry,
        pretraining,
        generated_at="2026-07-30T08:00:00+00:00",
    )
    assert feed["registry_sha256"] == canonical_json_sha256(registry)
    assert validate_live_feed(feed)["pretraining"] == pretraining

    tampered = json.loads(json.dumps(feed))
    tampered["registry"]["shared_pretraining"]["p1"]["status"] = (
        "tampered-after-feed-build"
    )
    with pytest.raises(ValueError, match="registry_sha256 mismatch"):
        validate_live_feed(tampered)
