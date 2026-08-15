"""Pure schema helpers for the interleaved 47.245M dashboard.

This module deliberately has no Modal dependency so registry flattening and
evaluation-result parsing can be tested locally without contacting Modal.
"""

from __future__ import annotations

import re
import statistics
from collections.abc import Iterable, Mapping
from typing import Any


INTERLEAVE_MODEL = "interleave_47m_qwen3"
INTERLEAVE_EVAL_INTERVAL = 40
TERMINAL_MARKER_PRECEDENCE = (
    "_SUCCESS.json",
    "_FAILED.json",
    "_RUNNING.json",
    "_QUEUED.json",
)

_MARKER_PATTERN = re.compile(
    r"^(?P<namespace>v1|interleave_v1)/"
    r"(?P<run_name>[^/]+)/global_step_(?P<step>\d+)/"
    r"(?P<profile>production(?:_[^/]+)?)/"
    r"(?P<marker>_(?:SUCCESS|FAILED|RUNNING|QUEUED)\.json)$"
)


def flatten_core_registry(registry: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return one normalized record for every RL phase in the core registry."""

    experiment_version = str(registry["experiment_version"])
    model = str(registry.get("model_id") or INTERLEAVE_MODEL)
    stages: list[dict[str, Any]] = []
    seen_runs: set[str] = set()

    for arm in registry["core_arms"]:
        experiment = str(arm["experiment"])
        filter_code = str(arm["filter"]).upper()
        if filter_code not in {"U", "D"}:
            raise ValueError(f"Unsupported filter code: {filter_code}")
        filter_mode = "dynamic" if filter_code == "D" else "unfiltered"
        arm_name = f"{experiment}-{filter_code}"

        phase_records: list[tuple[str, Mapping[str, Any]]]
        if experiment == "E1":
            phase_records = [
                ("RL1", arm["stage1"]),
                ("RL2", arm["stage2"]),
            ]
        else:
            phase_records = [("RL", arm["stage"])]

        for phase, stage in phase_records:
            run_name = str(stage["run_name"])
            if run_name in seen_runs:
                raise ValueError(f"Duplicate interleave run_name: {run_name}")
            seen_runs.add(run_name)
            steps = int(stage["steps"])
            offset = int(stage.get("effective_step_offset", 0))
            if steps <= 0 or offset < 0:
                raise ValueError(
                    f"Invalid step range for {run_name}: offset={offset}, steps={steps}"
                )
            stages.append(
                {
                    "experiment_version": experiment_version,
                    "model": model,
                    "experiment": experiment,
                    "arm": arm_name,
                    "filter": filter_code,
                    "filter_mode": filter_mode,
                    "phase": phase,
                    "run_name": run_name,
                    "target_step": steps,
                    "effective_step_offset": offset,
                    "rollout_seed": int(stage["rollout_seed"]),
                    "dynamic_filter": bool(stage["dynamic_filter"]),
                    "registry_status": str(arm.get("status", "planned")),
                }
            )

    return stages


def parse_interleave_marker_path(path: str) -> dict[str, Any] | None:
    """Parse a production eval marker path from either supported namespace."""

    match = _MARKER_PATTERN.fullmatch(path.lstrip("/"))
    if match is None:
        return None
    parsed: dict[str, Any] = match.groupdict()
    parsed["step"] = int(parsed["step"])
    return parsed


def select_terminal_marker(
    markers: Mapping[str, Mapping[str, Any]],
) -> tuple[str, Mapping[str, Any]] | None:
    """Select a job state with terminal markers taking precedence."""

    for marker_name in TERMINAL_MARKER_PRECEDENCE:
        if marker_name in markers:
            return marker_name.removeprefix("_").removesuffix(".json").lower(), markers[
                marker_name
            ]
    return None


def _numeric(values: Iterable[Any]) -> list[float]:
    return [float(value) for value in values if isinstance(value, (int, float))]


def summarize_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the final-table metrics from a canonical B1--B5 eval output.

    ``reward/pass@1`` is used when the evaluator writes it explicitly. Older
    evaluator versions only write ``reward/mean@16``; for the binary chess
    reward this is the same empirical single-sample success rate, so it is the
    documented fallback.
    """

    benchmarks: dict[str, dict[str, float | None]] = {}
    for benchmark in ("B1", "B2", "B3", "B4", "B5"):
        mean = metrics.get(f"val-core/test_{benchmark}/reward/mean@16")
        pass_at_1 = metrics.get(f"val-aux/test_{benchmark}/reward/pass@1")
        fallback_used = not isinstance(pass_at_1, (int, float))
        if fallback_used:
            pass_at_1 = mean
        benchmarks[benchmark] = {
            "avg_reward": float(mean) if isinstance(mean, (int, float)) else None,
            "pass_at_1": (
                float(pass_at_1)
                if isinstance(pass_at_1, (int, float))
                else None
            ),
            "pass_at_1_from_mean_fallback": fallback_used,
        }

    rewards = _numeric(item["avg_reward"] for item in benchmarks.values())
    pass_values = _numeric(item["pass_at_1"] for item in benchmarks.values())
    b3_b4 = _numeric(
        benchmarks[benchmark]["avg_reward"] for benchmark in ("B3", "B4")
    )
    return {
        "pass_at_1": statistics.fmean(pass_values) if pass_values else None,
        "avg_reward": statistics.fmean(rewards) if rewards else None,
        "b3_b4_avg": statistics.fmean(b3_b4) if len(b3_b4) == 2 else None,
        "benchmarks": benchmarks,
        "pass_at_1_semantics": (
            "explicit_reward_pass@1"
            if pass_values
            and not any(
                item["pass_at_1_from_mean_fallback"]
                for item in benchmarks.values()
                if item["pass_at_1"] is not None
            )
            else "binary_reward_mean@16_fallback"
        ),
    }


def build_result_rows(
    stages: Iterable[Mapping[str, Any]],
    checkpoint_steps: Mapping[str, Iterable[int]],
    eval_jobs: Mapping[tuple[str, int], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Join durable checkpoints and eval jobs into experiment-aware rows."""

    rows: list[dict[str, Any]] = []
    for stage in stages:
        run_name = str(stage["run_name"])
        trained = {int(step) for step in checkpoint_steps.get(run_name, ())}
        evaluated = {
            step
            for (job_run, step) in eval_jobs
            if job_run == run_name
        }
        for phase_step in sorted(trained | evaluated):
            job = eval_jobs.get((run_name, phase_step), {})
            metrics = job.get("metrics") or {}
            rows.append(
                {
                    "model": stage["model"],
                    "experiment": stage["experiment"],
                    "arm": stage["arm"],
                    "filter": stage["filter"],
                    "filter_mode": stage["filter_mode"],
                    "phase": stage["phase"],
                    "run_name": run_name,
                    "phase_step": phase_step,
                    "effective_rl_step": (
                        int(stage["effective_step_offset"]) + phase_step
                    ),
                    "training_status": (
                        "checkpointed" if phase_step in trained else "not_observed"
                    ),
                    "eval_status": str(job.get("state", "not_queued")),
                    "pass_at_1": metrics.get("pass_at_1"),
                    "avg_reward": metrics.get("avg_reward"),
                    "b3_b4_avg": metrics.get("b3_b4_avg"),
                    "pass_at_1_semantics": metrics.get("pass_at_1_semantics"),
                }
            )

    return sorted(
        rows,
        key=lambda row: (
            row["experiment"],
            row["filter"],
            row["effective_rl_step"],
            row["phase"],
        ),
    )
