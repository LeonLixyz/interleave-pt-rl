"""Summarize steady-state Miles benchmark metrics from W&B runs.

The script only reads run history through the W&B Public API. A run may be
specified as a bare ID (with ``--entity`` and ``--project``), an
``entity/project/id`` path, or a W&B run URL.
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
from collections.abc import Iterable, Sequence
from urllib.parse import urlparse


DEFAULT_METRICS = (
    "perf/rollout_time",
    "perf/tokens_per_gpu_per_sec",
    "perf/effective_tokens_per_gpu_per_sec",
    "rollout/prefix_cache_hit_rate",
    "rollout/avg_cached_tokens_per_sample",
    "rollout/response_len/mean",
    "rollout/chess_batch_generate/attempts",
    "rollout/chess_batch_generate/successes",
    "rollout/chess_batch_generate/fallbacks",
    "rollout/chess_batch_generate/unsupported",
)


def resolve_run_path(run_ref: str, *, entity: str | None, project: str | None) -> str:
    """Return the ``entity/project/run_id`` path expected by ``wandb.Api``."""

    value = run_ref.strip()
    if not value:
        raise ValueError("W&B run reference cannot be empty.")

    if "://" in value:
        parts = [part for part in urlparse(value).path.split("/") if part]
        try:
            runs_index = parts.index("runs")
            value = "/".join((parts[runs_index - 2], parts[runs_index - 1], parts[runs_index + 1]))
        except (ValueError, IndexError) as exc:
            raise ValueError(f"Not a recognizable W&B run URL: {run_ref}") from exc

    parts = [part for part in value.strip("/").split("/") if part]
    if len(parts) == 1:
        if not entity or not project:
            raise ValueError(f"Bare run ID {run_ref!r} requires both --entity and --project.")
        return f"{entity}/{project}/{parts[0]}"
    if len(parts) == 3:
        return "/".join(parts)
    if len(parts) == 4 and parts[2] == "runs":
        return "/".join((parts[0], parts[1], parts[3]))
    raise ValueError(f"Expected a run ID, entity/project/id, or W&B run URL; got {run_ref!r}.")


def _finite_number(value) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _metric_by_step(run, *, step_key: str, metric: str) -> dict[float, float]:
    """Read one metric exactly and keep the final logged value for each step."""

    values: dict[float, float] = {}
    for row in run.scan_history(keys=[step_key, metric], page_size=1000):
        step = _finite_number(row.get(step_key))
        value = _finite_number(row.get(metric))
        if step is not None and value is not None:
            values[step] = value
    return values


def summarize_run(
    run,
    *,
    metrics: Sequence[str] = DEFAULT_METRICS,
    step_key: str = "rollout/step",
    warmup_steps: int = 1,
) -> dict:
    """Compute unweighted per-step means after dropping earliest warmup steps."""

    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative.")

    points = {metric: _metric_by_step(run, step_key=step_key, metric=metric) for metric in metrics}
    observed_steps = sorted({step for metric_points in points.values() for step in metric_points})
    skipped_steps = observed_steps[:warmup_steps]
    used_steps = observed_steps[warmup_steps:]
    used_step_set = set(used_steps)

    metric_summaries = {}
    for metric, metric_points in points.items():
        values = [value for step, value in sorted(metric_points.items()) if step in used_step_set]
        metric_summaries[metric] = {
            "mean": statistics.fmean(values) if values else None,
            "count": len(values),
        }

    path = getattr(run, "path", None)
    if isinstance(path, Iterable) and not isinstance(path, str):
        path = "/".join(str(part) for part in path)

    return {
        "name": getattr(run, "name", None) or getattr(run, "id", "unknown"),
        "path": path or getattr(run, "id", "unknown"),
        "skipped_steps": skipped_steps,
        "used_steps": used_steps,
        "metrics": metric_summaries,
    }


def _format_steps(steps: Sequence[float]) -> str:
    def format_step(step: float) -> str:
        return str(int(step)) if step.is_integer() else f"{step:g}"

    return ", ".join(format_step(step) for step in steps) if steps else "none"


def print_summary(summary: dict) -> None:
    print(f"{summary['name']} ({summary['path']})")
    print(f"  skipped warmup steps: {_format_steps(summary['skipped_steps'])}")
    print(f"  measured steps: {_format_steps(summary['used_steps'])}")
    for metric, result in summary["metrics"].items():
        if result["mean"] is None:
            print(f"  {metric}: unavailable (n=0)")
        else:
            print(f"  {metric}: mean={result['mean']:.6g} (n={result['count']})")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="W&B run IDs, entity/project/id paths, or run URLs.")
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"), help="Entity for bare run IDs.")
    parser.add_argument("--project", default=None, help="Project for bare run IDs.")
    parser.add_argument("--warmup-steps", type=int, default=1, help="Earliest distinct rollout steps to skip.")
    parser.add_argument("--step-key", default="rollout/step", help="History key used as the rollout step.")
    parser.add_argument(
        "--metric",
        action="append",
        dest="metrics",
        help="Metric to average. Repeat to override the default metric set.",
    )
    parser.add_argument("--timeout", type=int, default=60, help="W&B API timeout in seconds.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.warmup_steps < 0:
        raise SystemExit("--warmup-steps must be non-negative")

    try:
        import wandb
    except ImportError as exc:
        raise SystemExit("The optional 'wandb' package is required: pip install wandb") from exc

    api = wandb.Api(timeout=args.timeout)
    metrics = tuple(args.metrics or DEFAULT_METRICS)
    failures = 0
    for index, run_ref in enumerate(args.runs):
        if index:
            print()
        try:
            run_path = resolve_run_path(run_ref, entity=args.entity, project=args.project)
            run = api.run(run_path)
            print_summary(
                summarize_run(
                    run,
                    metrics=metrics,
                    step_key=args.step_key,
                    warmup_steps=args.warmup_steps,
                )
            )
        except Exception as exc:  # Keep summarizing independent run IDs after one API failure.
            failures += 1
            print(f"ERROR {run_ref}: {exc}", file=sys.stderr)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
