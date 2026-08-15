from __future__ import annotations

import json
from pathlib import Path

import pytest

from Eval.interleave_eval_queue import (
    INTERLEAVE_NAMESPACE,
    build_interleave_dry_run_plan,
    cadence_steps,
    final_table_metrics,
    flatten_interleave_eval_registry,
    parse_raw_checkpoint_step,
)


ROOT = Path(__file__).resolve().parents[2]


def _registry() -> dict:
    return json.loads((ROOT / "INTERLEAVED_CORE_REGISTRY.json").read_text())


def test_dry_run_finds_eight_phases_and_448_cadence_checkpoints() -> None:
    plan = build_interleave_dry_run_plan(_registry())

    assert plan["namespace"] == INTERLEAVE_NAMESPACE == "interleave_v1"
    assert plan["eval_interval"] == 40
    assert plan["stage_count"] == 8
    assert plan["checkpoint_count"] == 448
    assert {stage["run_name"] for stage in plan["stages"]} == {
        "core-e1-u-rl1-seed42",
        "core-e1-u-rl2-seed43",
        "core-e1-d-rl1-seed42",
        "core-e1-d-rl2-seed43",
        "core-e2-u-rl3000-seed42",
        "core-e2-d-rl3000-seed42",
        "core-e3-u-rl3000-seed42",
        "core-e3-d-rl3000-seed42",
    }


def test_registry_paths_and_explicit_conversion_origin_fallback() -> None:
    registry = _registry()
    # This endpoint is deliberately mutable as P2 completes. Exercise the
    # pre-completion fallback independently of current operational status.
    e1_u = next(
        arm
        for arm in registry["core_arms"]
        if arm["experiment"] == "E1" and arm["filter"] == "U"
    )
    e1_u["p2"]["endpoint"] = None
    stages = flatten_interleave_eval_registry(registry)
    e1_rl1 = next(
        stage
        for stage in stages
        if stage["arm"] == "E1-U" and stage["phase"] == "RL1"
    )
    e1_rl2 = next(
        stage
        for stage in stages
        if stage["arm"] == "E1-U" and stage["phase"] == "RL2"
    )

    assert e1_rl1["raw_checkpoint_root"].endswith(
        "/core-e1-u-rl1-seed42"
    )
    assert not e1_rl1["conversion_origin_fallback"]
    assert e1_rl1["conversion_origin_hf"].endswith("/p1_shared/final")
    assert e1_rl2["conversion_origin_fallback"]
    assert e1_rl2["conversion_origin_hf"].endswith("/p1_shared/final")


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("iter_0000040", 40),
        ("iter_0003000", 3000),
        ("iter_40", 40),
        ("global_step_40", None),
        ("iter_0000000", None),
        ("iter_bad", None),
    ],
)
def test_raw_checkpoint_parser(name: str, expected: int | None) -> None:
    assert parse_raw_checkpoint_step(name) == expected


def test_cadence_excludes_off_interval_forced_final() -> None:
    steps = cadence_steps(1500)

    assert len(steps) == 37
    assert steps[0] == 40
    assert steps[-1] == 1480
    assert 1500 not in steps


def test_exact_final_table_metrics_include_b3_b4_and_effective_step() -> None:
    stage = next(
        stage
        for stage in flatten_interleave_eval_registry(_registry())
        if stage["arm"] == "E1-D" and stage["phase"] == "RL2"
    )
    metrics = {}
    for index, benchmark in enumerate(("B1", "B2", "B3", "B4", "B5"), 1):
        metrics[f"val-core/test_{benchmark}/reward/mean@16"] = index / 10
        metrics[f"val-aux/test_{benchmark}/reward/pass@1"] = index / 20

    table = final_table_metrics(metrics, stage, 40)

    assert table["model"] == "interleave_47m_qwen3"
    assert table["run_name"] == "core-e1-d-rl2-seed43"
    assert table["phase_step"] == 40
    assert table["effective_rl_step"] == 1540
    assert table["pass_at_1"] == pytest.approx(0.15)
    assert table["avg_reward"] == pytest.approx(0.3)
    assert table["b3_avg"] == pytest.approx(0.3)
    assert table["b4_avg"] == pytest.approx(0.4)
    assert table["b3_b4_avg"] == pytest.approx(0.35)
    assert table["pass_at_1_semantics"] == "explicit_reward_pass@1"


def test_final_table_metrics_reject_partial_benchmark_output() -> None:
    stage = flatten_interleave_eval_registry(_registry())[0]
    metrics = {
        f"val-core/test_{benchmark}/reward/mean@16": 0.5
        for benchmark in ("B1", "B2", "B3", "B4")
    }

    with pytest.raises(ValueError, match="B5"):
        final_table_metrics(metrics, stage, 40)
