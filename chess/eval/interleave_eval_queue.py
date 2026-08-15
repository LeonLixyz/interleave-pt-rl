"""Pure helpers for the interleaved RL checkpoint evaluation queue.

This module has no Modal dependency.  It is shared by local dry runs/tests and
the Modal evaluator so registry parsing, cadence selection, and final-table
metrics cannot silently diverge.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

try:
    from .interleave_dashboard_schema import (
        INTERLEAVE_EVAL_INTERVAL,
        flatten_core_registry,
        summarize_metrics,
    )
except ImportError:
    from interleave_dashboard_schema import (
        INTERLEAVE_EVAL_INTERVAL,
        flatten_core_registry,
        summarize_metrics,
    )


INTERLEAVE_NAMESPACE = "interleave_v1"
_RAW_CHECKPOINT_PATTERN = re.compile(r"^iter_0*(?P<step>[1-9]\d*)$")


def _resolve_registry_reference(
    registry: Mapping[str, Any], value: Any
) -> Any:
    """Resolve a dotted registry reference, leaving literal values unchanged."""

    if not isinstance(value, str) or not value or value.startswith("/"):
        return value
    current: Any = registry
    for component in value.split("."):
        if not isinstance(current, Mapping) or component not in current:
            return None
        current = current[component]
    return current


def _usable_hf_origin(value: Any) -> str | None:
    if isinstance(value, str) and value.startswith("/pretrain-checkpoints/"):
        return value
    return None


def flatten_interleave_eval_registry(
    registry: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return the eight core RL phases with raw and conversion paths.

    FSDP-to-HF conversion only needs an architecture-compatible config and
    tokenizer; it does not consume weights from the origin model.  Until a P2
    endpoint is materialized in the mutable registry, the immutable shared P1
    export is therefore a safe, explicit conversion-asset fallback.
    """

    stages = flatten_core_registry(registry)
    canonical_origin = _usable_hf_origin(
        registry["shared_pretraining"]["p1"]["endpoint_rl_mount"]
    )
    if canonical_origin is None:
        raise ValueError("Registry is missing the canonical P1 HF endpoint")

    raw_root = str(registry["rl_raw_root"]).rstrip("/")
    if not raw_root.startswith("/rl-checkpoints/"):
        raise ValueError(
            f"Registry raw RL root is outside the evaluator mount: {raw_root}"
        )
    arms = {
        (str(arm["experiment"]), str(arm["filter"]).upper()): arm
        for arm in registry["core_arms"]
    }
    enriched: list[dict[str, Any]] = []
    for stage in stages:
        arm = arms[(stage["experiment"], stage["filter"])]
        if stage["experiment"] == "E1" and stage["phase"] == "RL1":
            requested_origin = _resolve_registry_reference(
                registry, arm["stage1"].get("init")
            )
        elif stage["experiment"] == "E1":
            requested_origin = arm["p2"].get("endpoint")
        else:
            requested_origin = _resolve_registry_reference(
                registry, arm["stage"].get("init")
            )

        conversion_origin = _usable_hf_origin(requested_origin)
        fallback_used = conversion_origin is None
        if fallback_used:
            conversion_origin = canonical_origin

        enriched.append(
            {
                **stage,
                "raw_checkpoint_root": f"{raw_root}/{stage['run_name']}",
                "conversion_origin_hf": conversion_origin,
                "conversion_origin_fallback": fallback_used,
            }
        )
    return enriched


def parse_raw_checkpoint_step(name: str) -> int | None:
    """Parse a complete Miles checkpoint directory name."""

    match = _RAW_CHECKPOINT_PATTERN.fullmatch(name)
    return int(match.group("step")) if match else None


def cadence_steps(target_step: int) -> list[int]:
    """Return exact 40-step checkpoints, excluding off-cadence forced finals."""

    if target_step <= 0:
        raise ValueError("target_step must be positive")
    return list(
        range(
            INTERLEAVE_EVAL_INTERVAL,
            target_step + 1,
            INTERLEAVE_EVAL_INTERVAL,
        )
    )


def final_table_metrics(
    metrics: Mapping[str, Any],
    stage: Mapping[str, Any],
    phase_step: int,
) -> dict[str, Any]:
    """Build the exact user-facing row and reject incomplete B1--B5 output."""

    summary = summarize_metrics(metrics)
    missing = [
        benchmark
        for benchmark, values in summary["benchmarks"].items()
        if values["avg_reward"] is None or values["pass_at_1"] is None
    ]
    if missing:
        raise ValueError(
            "Incomplete canonical B1-B5 metrics: " + ", ".join(missing)
        )
    if summary["b3_b4_avg"] is None:
        raise ValueError("Incomplete B3-B4 metrics")
    if phase_step <= 0 or phase_step > int(stage["target_step"]):
        raise ValueError(
            f"phase_step {phase_step} is outside 1..{stage['target_step']}"
        )

    return {
        "model": stage["model"],
        "experiment": stage["experiment"],
        "arm": stage["arm"],
        "filter": stage["filter"],
        "filter_mode": stage["filter_mode"],
        "phase": stage["phase"],
        "run_name": stage["run_name"],
        "phase_step": phase_step,
        "effective_rl_step": int(stage["effective_step_offset"]) + phase_step,
        "pass_at_1": summary["pass_at_1"],
        "avg_reward": summary["avg_reward"],
        "b3_avg": summary["benchmarks"]["B3"]["avg_reward"],
        "b4_avg": summary["benchmarks"]["B4"]["avg_reward"],
        "b3_b4_avg": summary["b3_b4_avg"],
        "pass_at_1_semantics": summary["pass_at_1_semantics"],
        "benchmarks": summary["benchmarks"],
    }


def build_interleave_dry_run_plan(
    registry: Mapping[str, Any],
) -> dict[str, Any]:
    stages = flatten_interleave_eval_registry(registry)
    planned_stages = []
    for stage in stages:
        steps = cadence_steps(int(stage["target_step"]))
        planned_stages.append(
            {
                **stage,
                "checkpoint_count": len(steps),
                "first_step": steps[0],
                "last_step": steps[-1],
            }
        )
    return {
        "namespace": INTERLEAVE_NAMESPACE,
        "eval_interval": INTERLEAVE_EVAL_INTERVAL,
        "stage_count": len(planned_stages),
        "checkpoint_count": sum(
            stage["checkpoint_count"] for stage in planned_stages
        ),
        "stages": planned_stages,
    }
