"""Public Modal dashboard for r6 and interleaved 47.245M Chess RL experiments.

Deploy:
    cd Eval && modal deploy modal_rl_dashboard.py

The web endpoint reads the durable training checkpoint Volume, the evaluation
results Volume, and the public Hugging Face repositories. It never reads the
large generation JSONL files.
"""

from __future__ import annotations

import json
import math
import os
import re
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import modal

try:
    from .interleave_dashboard_schema import (
        INTERLEAVE_EVAL_INTERVAL,
        build_result_rows,
        flatten_core_registry,
        parse_interleave_marker_path,
        select_terminal_marker,
        summarize_metrics,
    )
    from .interleave_live_schema import (
        build_live_feed,
        canonical_json_sha256,
        parse_pretrain_metrics_jsonl,
        validate_live_feed,
        validate_pretrain_trainer_state,
    )
    from .interleave_exp4_eval_queue import flatten_exp4_rl_registry
except ImportError:
    from interleave_dashboard_schema import (
        INTERLEAVE_EVAL_INTERVAL,
        build_result_rows,
        flatten_core_registry,
        parse_interleave_marker_path,
        select_terminal_marker,
        summarize_metrics,
    )
    from interleave_live_schema import (
        build_live_feed,
        canonical_json_sha256,
        parse_pretrain_metrics_jsonl,
        validate_live_feed,
        validate_pretrain_trainer_state,
    )
    from interleave_exp4_eval_queue import flatten_exp4_rl_registry


APP_NAME = "chess-rl-live-dashboard"
ENVIRONMENT_NAME = "leon-dev"
WORKSPACE_NAME = "modal-labs"

TRAINING_VOLUME_NAME = "chess-rl-miles-checkpoints"
EVAL_VOLUME_NAME = "chess-rl-eval-results-r6"
PRETRAIN_VOLUME_NAME = "rl-reasoning-checkpoints"
DASHBOARD_STATE_NAME = "chess-rl-live-dashboard-state"
TRAINING_MOUNT = Path("/training")
EVAL_MOUNT = Path("/results")
PRETRAIN_MOUNT = Path("/pretraining")
LIVE_FEED_ROOT = EVAL_MOUNT / "dashboard_live" / "interleave"
LIVE_FEED_POINTER = LIVE_FEED_ROOT / "latest.json"
INTERLEAVE_RAW_ROOT = "chess-rl-miles-interleave"
INTERLEAVE_REGISTRY_LOCAL = (
    Path(__file__).resolve().parent.parent / "INTERLEAVED_CORE_REGISTRY.json"
)
INTERLEAVE_REGISTRY_REMOTE = Path(
    "/opt/chess-dashboard/INTERLEAVED_CORE_REGISTRY.json"
)
INTERLEAVE_SCHEMA_LOCAL = (
    Path(__file__).resolve().parent / "interleave_dashboard_schema.py"
)
INTERLEAVE_SCHEMA_REMOTE = Path("/root/interleave_dashboard_schema.py")
INTERLEAVE_LIVE_SCHEMA_LOCAL = (
    Path(__file__).resolve().parent / "interleave_live_schema.py"
)
INTERLEAVE_LIVE_SCHEMA_REMOTE = Path("/root/interleave_live_schema.py")
INTERLEAVE_EXP4_QUEUE_LOCAL = (
    Path(__file__).resolve().parent / "interleave_exp4_eval_queue.py"
)
INTERLEAVE_EXP4_QUEUE_REMOTE = Path("/root/interleave_exp4_eval_queue.py")

HPARAM = (
    "multi_turn_lr1e-5_bs2048_kl0.001_res2560_adamw_grpo_miles_sglang_grpo_"
    "adamw_fastpath_c128_bs2048_fresh_sft_20260725_r6"
)
HF_PREFIX = "miles_sglang_grpo_r6"
PRODUCTION_FINGERPRINT = (
    "467f569b87ae80ba83d6dabd6e499293b374d0160a3e6d3ea577fd858ba618c1"
)
CHECKPOINT_UPLOAD_INTERVAL = 20
EVAL_INTERVAL = 40
EVAL_WORKER_CEILING = int(os.environ.get("EVAL_WORKER_CEILING", "128"))
EXPECTED_ROWS = 23_680
ENDPOINT_RESULT_SCHEMA = "interleaved-endpoint-result-v1"
ENDPOINT_NAMESPACE = "endpoint_v1"
ENDPOINT_COMPONENTS = ("losses", "chess")
ENDPOINT_BENCHMARKS = ("B1", "B2", "B3", "B4", "B5")
ENDPOINT_COMPLETE_CACHE_SECONDS = 15 * 60
ENDPOINT_PARTIAL_CACHE_SECONDS = 60
ENDPOINT_LISTING_BACKOFF_SECONDS = 5 * 60
ENDPOINT_LISTING_ERROR_PREFIX = "endpoint result listing:"
ENDPOINT_SUMMARY_SPECS: dict[str, dict[str, str]] = {
    "p1": {"label": "Shared P1", "experiment": "P1"},
    "e2-final": {"label": "Exp 2 · monolithic", "experiment": "E2"},
    "e3-p2": {"label": "Exp 3 · two cosine", "experiment": "E3"},
}
ENDPOINT_SUCCESS_RE = re.compile(
    rf"^{re.escape(ENDPOINT_NAMESPACE)}/"
    r"(?P<endpoint>[a-z0-9][a-z0-9.-]{0,127})/"
    r"(?P<checkpoint>[0-9a-f]{64})/"
    r"(?P<component>losses|chess)_(?P<fingerprint>[0-9a-f]{12})/"
    r"_SUCCESS\.json$"
)
V2R3_DIAGNOSTIC_SCHEMA = "interleaved-v2r3-diagnostic-contract-v1"
V2R3_DIAGNOSTIC_REPORT_SCHEMA = "interleaved-v2r3-diagnostic-report-v1"
V2R3_TRAJECTORY_SPECS: tuple[
    tuple[str, float, int, tuple[int, ...]], ...
] = (
    (
        "190.189290837",
        190.189290837,
        9_920,
        (1_000, 2_000, 4_000, 6_000, 8_000, 9_920),
    ),
    ("256", 256.0, 2_000, (1_000, 2_000)),
    ("384", 384.0, 2_000, (1_000, 2_000)),
    ("768", 768.0, 2_000, (1_000, 2_000)),
)
V2R3_EXPECTED_SNAPSHOTS = sum(
    len(snapshot_steps)
    for _, _, _, snapshot_steps in V2R3_TRAJECTORY_SPECS
)
V2R3_ROLLOUT_STATES = {
    "running_or_queued": "running",
    "success_pending_artifact_audit": "pending",
    "inspected_success": "complete",
    "failed": "failed",
}

RUNS: dict[str, dict[str, Any]] = {
    "C6p5e18_32m_alpha0.200_beta0.013": {
        "short": "32M · α 0.200",
        "family": "32M",
        "alpha": 0.2,
        "target": 4_000,
        "app_id": "ap-5p0pKeFEF1LnD19yGOyRRk",
        "color": "#57d6b5",
    },
    "C6p5e18_32m_alpha0.400_beta0.013": {
        "short": "32M · α 0.400",
        "family": "32M",
        "alpha": 0.4,
        "target": 4_000,
        "app_id": "ap-nZeyLcH4sqTt1M1Ey0oORL",
        "color": "#72a7ff",
    },
    "C6p5e18_410m_alpha0.750_beta0.148": {
        "short": "410M · α 0.750",
        "family": "410M",
        "alpha": 0.75,
        "target": 3_000,
        "app_id": "ap-VvrIQZ9s5gjqMYXnDSxj8o",
        "color": "#c18cff",
    },
    "C6p5e18_410m_alpha1.000_beta0.148": {
        "short": "410M · α 1.000",
        "family": "410M",
        "alpha": 1.0,
        "target": 3_000,
        "app_id": "ap-nSltc0HdnzKWatP45FNkBR",
        "color": "#ffb45c",
    },
}


dashboard_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi==0.116.1",
        "huggingface_hub==0.36.2",
    )
    .add_local_file(
        str(INTERLEAVE_REGISTRY_LOCAL),
        remote_path=str(INTERLEAVE_REGISTRY_REMOTE),
        copy=True,
    )
    .add_local_file(
        str(INTERLEAVE_SCHEMA_LOCAL),
        remote_path=str(INTERLEAVE_SCHEMA_REMOTE),
        copy=True,
    )
    .add_local_file(
        str(INTERLEAVE_LIVE_SCHEMA_LOCAL),
        remote_path=str(INTERLEAVE_LIVE_SCHEMA_REMOTE),
        copy=True,
    )
    .add_local_file(
        str(INTERLEAVE_EXP4_QUEUE_LOCAL),
        remote_path=str(INTERLEAVE_EXP4_QUEUE_REMOTE),
        copy=True,
    )
)
training_volume = modal.Volume.from_name(
    TRAINING_VOLUME_NAME, create_if_missing=False
)
eval_volume = modal.Volume.from_name(EVAL_VOLUME_NAME, create_if_missing=False)
pretrain_volume = modal.Volume.from_name(
    PRETRAIN_VOLUME_NAME, create_if_missing=False
)
dashboard_state = modal.Dict.from_name(
    DASHBOARD_STATE_NAME, create_if_missing=True
)
app = modal.App(APP_NAME)

_snapshot_lock = threading.Lock()
_snapshot: dict[str, Any] | None = None
_snapshot_at = 0.0
_SNAPSHOT_TTL_SECONDS = 25

_hf_lock = threading.Lock()
_hf_cache: dict[str, dict[str, Any]] = {}
_hf_cache_at = 0.0
_HF_TTL_SECONDS = 90


def _iso_from_epoch(value: float | int | None) -> str | None:
    if not value:
        return None
    return datetime.fromtimestamp(float(value), timezone.utc).isoformat()


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _endpoint_number(
    value: Any,
    field: str,
    *,
    minimum: float = 0.0,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    if number < minimum:
        raise ValueError(f"{field} must be >= {minimum}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{field} must be <= {maximum}")
    return number


def _verified_endpoint_result(
    marker_path: Path,
    match: re.Match[str],
) -> dict[str, Any]:
    """Validate one immutable endpoint-evaluation success marker."""

    value = _read_json(marker_path)
    endpoint_id = match.group("endpoint")
    checkpoint_sha256 = match.group("checkpoint")
    component = match.group("component")
    fingerprint_prefix = match.group("fingerprint")
    if value.get("schema") != ENDPOINT_RESULT_SCHEMA:
        raise ValueError("unsupported endpoint result schema")
    if value.get("schema_version") != 1:
        raise ValueError("unsupported endpoint result schema_version")
    if value.get("state") != "complete":
        raise ValueError("endpoint success marker is not complete")
    if value.get("namespace") != ENDPOINT_NAMESPACE:
        raise ValueError("endpoint result namespace drifted")
    if value.get("endpoint_id") != endpoint_id:
        raise ValueError("endpoint ID does not match marker path")
    if value.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("checkpoint SHA-256 does not match marker path")
    if value.get("component") != component:
        raise ValueError("component does not match marker path")
    fingerprint = value.get("eval_fingerprint")
    if (
        not isinstance(fingerprint, str)
        or not re.fullmatch(r"[0-9a-f]{64}", fingerprint)
        or not fingerprint.startswith(fingerprint_prefix)
    ):
        raise ValueError("evaluation fingerprint does not match marker path")
    result_hash = value.get("result_hash")
    if not isinstance(result_hash, str) or not re.fullmatch(
        r"[0-9a-f]{64}", result_hash
    ):
        raise ValueError("endpoint result hash is invalid")
    unhashed = {key: item for key, item in value.items() if key != "result_hash"}
    if canonical_json_sha256(unhashed) != result_hash:
        raise ValueError("endpoint result hash mismatch")
    finished_at = value.get("finished_at")
    finished = _parse_iso(finished_at)
    if finished is None or finished.tzinfo is None:
        raise ValueError("endpoint result finished_at is invalid")
    duration_seconds = _endpoint_number(
        value.get("duration_seconds"),
        "duration_seconds",
    )
    metrics = value.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("endpoint result metrics must be an object")

    normalized: dict[str, Any] = {
        "component": component,
        "endpoint_id": endpoint_id,
        "checkpoint_sha256": checkpoint_sha256,
        "eval_fingerprint": fingerprint,
        "result_hash": result_hash,
        "finished_at": str(finished_at),
        "duration_seconds": duration_seconds,
    }
    if component == "losses":
        target_tokens = metrics.get("heldout_pretrain_target_tokens")
        if (
            isinstance(target_tokens, bool)
            or not isinstance(target_tokens, int)
            or target_tokens <= 0
        ):
            raise ValueError(
                "heldout_pretrain_target_tokens must be a positive integer"
            )
        normalized["metrics"] = {
            "heldout_pretrain_loss": _endpoint_number(
                metrics.get("heldout_pretrain_loss"),
                "heldout_pretrain_loss",
            ),
            "heldout_pretrain_perplexity": _endpoint_number(
                metrics.get("heldout_pretrain_perplexity"),
                "heldout_pretrain_perplexity",
                minimum=1.0,
            ),
            "heldout_pretrain_token_accuracy": _endpoint_number(
                metrics.get("heldout_pretrain_token_accuracy"),
                "heldout_pretrain_token_accuracy",
                maximum=1.0,
            ),
            "heldout_pretrain_target_tokens": target_tokens,
            "masked_sft_status": metrics.get("masked_sft_status"),
        }
        return normalized

    expected_rows = value.get("expected_rows")
    actual_rows = value.get("actual_rows")
    if (
        isinstance(expected_rows, bool)
        or not isinstance(expected_rows, int)
        or expected_rows != EXPECTED_ROWS
        or actual_rows != expected_rows
    ):
        raise ValueError(
            f"endpoint chess row accounting is not {EXPECTED_ROWS}/{EXPECTED_ROWS}"
        )
    raw_benchmarks = metrics.get("benchmarks")
    if not isinstance(raw_benchmarks, dict):
        raise ValueError("endpoint chess benchmarks must be an object")
    benchmarks: dict[str, dict[str, float]] = {}
    for benchmark in ENDPOINT_BENCHMARKS:
        raw = raw_benchmarks.get(benchmark)
        if not isinstance(raw, dict):
            raise ValueError(f"endpoint chess benchmark {benchmark} is missing")
        benchmarks[benchmark] = {
            "pass_at_1": _endpoint_number(
                raw.get("pass_at_1"),
                f"{benchmark}.pass_at_1",
                maximum=1.0,
            ),
            "avg_reward": _endpoint_number(
                raw.get("avg_reward"),
                f"{benchmark}.avg_reward",
                maximum=1.0,
            ),
        }
    pass_at_1 = _endpoint_number(
        metrics.get("pass_at_1"), "pass_at_1", maximum=1.0
    )
    avg_reward = _endpoint_number(
        metrics.get("avg_reward"), "avg_reward", maximum=1.0
    )
    b3_avg = _endpoint_number(
        metrics.get("b3_avg"), "b3_avg", maximum=1.0
    )
    b4_avg = _endpoint_number(
        metrics.get("b4_avg"), "b4_avg", maximum=1.0
    )
    b3_b4_avg = _endpoint_number(
        metrics.get("b3_b4_avg"), "b3_b4_avg", maximum=1.0
    )
    benchmark_macro = statistics.mean(
        benchmarks[name]["avg_reward"] for name in ENDPOINT_BENCHMARKS
    )
    if not math.isclose(avg_reward, benchmark_macro, abs_tol=1e-12):
        raise ValueError("endpoint avg_reward is not the B1-B5 macro")
    if not math.isclose(
        b3_avg, benchmarks["B3"]["avg_reward"], abs_tol=1e-12
    ):
        raise ValueError("endpoint b3_avg does not match benchmark B3")
    if not math.isclose(
        b4_avg, benchmarks["B4"]["avg_reward"], abs_tol=1e-12
    ):
        raise ValueError("endpoint b4_avg does not match benchmark B4")
    if not math.isclose(
        b3_b4_avg, statistics.mean((b3_avg, b4_avg)), abs_tol=1e-12
    ):
        raise ValueError("endpoint b3_b4_avg is inconsistent")
    normalized.update(
        {
            "actual_rows": actual_rows,
            "expected_rows": expected_rows,
            "metrics": {
                "pass_at_1": pass_at_1,
                "avg_reward": avg_reward,
                "b3_avg": b3_avg,
                "b4_avg": b4_avg,
                "b3_b4_avg": b3_b4_avg,
                "benchmarks": benchmarks,
                "pass_at_1_semantics": metrics.get(
                    "pass_at_1_semantics"
                ),
            },
        }
    )
    return normalized


def _collect_endpoint_evaluations() -> dict[str, Any]:
    """Collect compact P1/E2/E3 endpoint summaries without generation reads."""

    errors: list[str] = []
    grouped: dict[
        tuple[str, str], dict[str, dict[str, Any]]
    ] = {}
    endpoint_root = EVAL_MOUNT / ENDPOINT_NAMESPACE
    entries: list[Any] = []
    if endpoint_root.exists():
        try:
            entries = list(
                eval_volume.listdir(ENDPOINT_NAMESPACE, recursive=True)
            )
        except Exception as exc:
            errors.append(
                f"endpoint result listing: {type(exc).__name__}: {exc}"
            )
    for entry in entries:
        match = ENDPOINT_SUCCESS_RE.fullmatch(str(entry.path))
        if match is None or match.group("endpoint") not in ENDPOINT_SUMMARY_SPECS:
            continue
        marker_path = EVAL_MOUNT / str(entry.path)
        try:
            result = _verified_endpoint_result(marker_path, match)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(
                f"{entry.path}: {type(exc).__name__}: {exc}"
            )
            continue
        group = grouped.setdefault(
            (match.group("endpoint"), match.group("checkpoint")),
            {},
        )
        component = match.group("component")
        previous = group.get(component)
        previous_at = _parse_iso(previous.get("finished_at")) if previous else None
        current_at = _parse_iso(result["finished_at"])
        if previous is None or (
            current_at is not None
            and (previous_at is None or current_at > previous_at)
        ):
            group[component] = result

    summaries: dict[str, dict[str, Any]] = {}
    for endpoint_id, spec in ENDPOINT_SUMMARY_SPECS.items():
        candidates = [
            (checkpoint, components)
            for (candidate_endpoint, checkpoint), components in grouped.items()
            if candidate_endpoint == endpoint_id
        ]
        if not candidates:
            summaries[endpoint_id] = {
                "endpoint_id": endpoint_id,
                **spec,
                "state": "missing",
                "checkpoint_sha256": None,
                "loss": None,
                "chess": None,
                "result_hashes": {"losses": None, "chess": None},
                "superseded_checkpoint_count": 0,
            }
            continue

        def candidate_key(
            candidate: tuple[str, dict[str, dict[str, Any]]]
        ) -> tuple[int, float, str]:
            checkpoint, components = candidate
            finished = [
                parsed.timestamp()
                for parsed in (
                    _parse_iso(item.get("finished_at"))
                    for item in components.values()
                )
                if parsed is not None
            ]
            return (
                int(all(name in components for name in ENDPOINT_COMPONENTS)),
                max(finished, default=0.0),
                checkpoint,
            )

        checkpoint_sha256, components = max(candidates, key=candidate_key)
        loss_result = components.get("losses")
        chess_result = components.get("chess")
        summaries[endpoint_id] = {
            "endpoint_id": endpoint_id,
            **spec,
            "state": (
                "complete"
                if loss_result is not None and chess_result is not None
                else "partial"
            ),
            "checkpoint_sha256": checkpoint_sha256,
            "loss": loss_result,
            "chess": chess_result,
            "result_hashes": {
                "losses": (
                    loss_result.get("result_hash") if loss_result else None
                ),
                "chess": (
                    chess_result.get("result_hash") if chess_result else None
                ),
            },
            "superseded_checkpoint_count": max(0, len(candidates) - 1),
        }

    result = {
        "schema_version": 1,
        "namespace": ENDPOINT_NAMESPACE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "endpoints": summaries,
        "aggregate": {
            "expected": len(ENDPOINT_SUMMARY_SPECS),
            "complete": sum(
                item["state"] == "complete" for item in summaries.values()
            ),
            "partial": sum(
                item["state"] == "partial" for item in summaries.values()
            ),
            "missing": sum(
                item["state"] == "missing" for item in summaries.values()
            ),
        },
        "errors": errors,
        "warnings": [],
    }
    if not errors:
        complete = result["aggregate"]["complete"]
        expected = result["aggregate"]["expected"]
        cache_seconds = (
            ENDPOINT_COMPLETE_CACHE_SECONDS
            if complete == expected
            else ENDPOINT_PARTIAL_CACHE_SECONDS
        )
        generated_at = _parse_iso(result["generated_at"])
        assert generated_at is not None
        result["cache_status"] = "verified_live"
        result["next_list_after"] = (
            generated_at + timedelta(seconds=cache_seconds)
        ).isoformat()
    else:
        result["cache_status"] = "live_scan_with_errors"
    return result


def _valid_endpoint_summary_shape(value: Any) -> bool:
    """Validate the compact summary before it is eligible for cache reuse."""

    if not isinstance(value, dict):
        return False
    endpoints = value.get("endpoints")
    aggregate = value.get("aggregate")
    if (
        value.get("schema_version") != 1
        or value.get("namespace") != ENDPOINT_NAMESPACE
        or not isinstance(endpoints, dict)
        or set(endpoints) != set(ENDPOINT_SUMMARY_SPECS)
        or not isinstance(aggregate, dict)
    ):
        return False
    states = [
        endpoint.get("state")
        for endpoint in endpoints.values()
        if isinstance(endpoint, dict)
    ]
    if len(states) != len(ENDPOINT_SUMMARY_SPECS):
        return False
    if any(state not in {"complete", "partial", "missing"} for state in states):
        return False
    expected_counts = {
        "expected": len(ENDPOINT_SUMMARY_SPECS),
        "complete": states.count("complete"),
        "partial": states.count("partial"),
        "missing": states.count("missing"),
    }
    try:
        observed_counts = {
            key: int(aggregate.get(key, -1)) for key in expected_counts
        }
    except (TypeError, ValueError):
        return False
    return observed_counts == expected_counts


def _endpoint_listing_errors(errors: Any) -> list[str]:
    if not isinstance(errors, list):
        return []
    return [
        str(item)
        for item in errors
        if str(item).startswith(ENDPOINT_LISTING_ERROR_PREFIX)
    ]


def _utc_datetime(value: Any) -> datetime | None:
    parsed = _parse_iso(value if isinstance(value, str) else None)
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _cached_endpoint_evaluations(
    previous: Any,
    *,
    now: datetime | None = None,
) -> dict[str, Any] | None:
    """Return a verified recent endpoint summary without another Volume list.

    Endpoint results are immutable and content-authenticated.  Reusing a
    complete summary for 15 minutes removes redundant recursive Volume calls.
    Partial summaries are reused for only one publisher interval.  Any
    per-marker or hash error makes the summary ineligible, so validation
    failures continue to fail closed.
    """

    if not _valid_endpoint_summary_shape(previous):
        return None
    errors = previous.get("errors", [])
    listing_errors = _endpoint_listing_errors(errors)
    non_listing_errors = [
        str(item)
        for item in errors
        if not str(item).startswith(ENDPOINT_LISTING_ERROR_PREFIX)
    ] if isinstance(errors, list) else ["malformed endpoint errors"]
    if non_listing_errors:
        return None

    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    now = now.astimezone(timezone.utc)
    deadline = _utc_datetime(previous.get("next_list_after"))
    if deadline is None:
        generated_at = _utc_datetime(previous.get("generated_at"))
        if generated_at is None:
            return None
        aggregate = previous["aggregate"]
        cache_seconds = (
            ENDPOINT_COMPLETE_CACHE_SECONDS
            if aggregate["complete"] == aggregate["expected"]
            else ENDPOINT_PARTIAL_CACHE_SECONDS
        )
        deadline = generated_at + timedelta(seconds=cache_seconds)
    if now >= deadline:
        return None

    cached = json.loads(json.dumps(previous))
    warnings = [
        str(item) for item in cached.get("warnings", [])
    ] if isinstance(cached.get("warnings", []), list) else []
    for warning in listing_errors:
        if warning not in warnings:
            warnings.append(warning)
    cached["errors"] = []
    cached["warnings"] = warnings
    cached["cache_status"] = (
        "listing_backoff"
        if listing_errors
        or previous.get("cache_status") == "listing_backoff"
        else "verified_cache"
    )
    cached["next_list_after"] = deadline.isoformat()
    cached["served_from_cache_at"] = now.isoformat()
    return cached


def _retain_endpoint_summary_on_listing_failure(
    current: dict[str, Any],
    previous: Any,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Keep the last verified endpoint table across transient Volume limits.

    A recursive Volume listing can be rate-limited while all immutable result
    markers remain healthy.  Replacing a previously verified 3/3 table with
    synthetic ``missing`` rows would report a false scientific regression.
    Only a top-level listing failure is eligible for this fallback; individual
    marker/hash validation failures remain visible and fail closed.
    """

    errors = current.get("errors")
    listing_errors = _endpoint_listing_errors(errors)
    listing_failed = bool(listing_errors)
    if not listing_failed or not isinstance(previous, dict):
        return current
    if not _valid_endpoint_summary_shape(previous):
        return current

    current_aggregate = current.get("aggregate")
    previous_aggregate = previous["aggregate"]
    if not isinstance(current_aggregate, dict):
        return current
    current_information = (
        2 * int(current_aggregate.get("complete", 0))
        + int(current_aggregate.get("partial", 0))
    )
    previous_information = (
        2 * int(previous_aggregate.get("complete", 0))
        + int(previous_aggregate.get("partial", 0))
    )
    if previous_information <= current_information:
        return current

    retained = json.loads(json.dumps(previous))
    retained["last_success_generated_at"] = previous.get(
        "last_success_generated_at",
        previous.get("generated_at"),
    )
    retained["generated_at"] = current.get("generated_at")
    retained["stale"] = True
    previous_errors = previous.get("errors", [])
    retained["errors"] = [
        str(item)
        for item in previous_errors
        if not str(item).startswith(ENDPOINT_LISTING_ERROR_PREFIX)
    ] if isinstance(previous_errors, list) else []
    retained["errors"].extend(
        str(item)
        for item in errors
        if not str(item).startswith(ENDPOINT_LISTING_ERROR_PREFIX)
    )
    warnings = [
        str(item) for item in previous.get("warnings", [])
    ] if isinstance(previous.get("warnings", []), list) else []
    for warning in (
        _endpoint_listing_errors(previous_errors) + listing_errors
    ):
        if warning not in warnings:
            warnings.append(warning)
    retained["warnings"] = warnings
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    now = now.astimezone(timezone.utc)
    retained["cache_status"] = "listing_backoff"
    retained["next_list_after"] = (
        now + timedelta(seconds=ENDPOINT_LISTING_BACKOFF_SECONDS)
    ).isoformat()
    return retained


def _training_root(run_key: str) -> Path:
    return (
        TRAINING_MOUNT
        / "chess-rl-miles"
        / "trajectory_sep_no_labels"
        / HPARAM
        / run_key
        / "checkpoints"
    )


def _checkpoint_is_valid(root: Path, step: int) -> tuple[bool, str]:
    step_root = root / f"iter_{step:07d}"
    required = ("model", "optimizer", "lr_scheduler", "rng.pt", "meta.json")
    missing = [name for name in required if not (step_root / name).exists()]
    if missing:
        return False, "missing " + ", ".join(missing)
    try:
        meta = _read_json(step_root / "meta.json")
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"invalid meta.json: {type(exc).__name__}"
    if meta.get("iteration") != step:
        return False, "meta iteration mismatch"
    if int(meta.get("next_rollout_id", -1)) < step:
        return False, "meta rollout mismatch"
    return True, "ok"


def _training_status(run_key: str, spec: dict[str, Any]) -> dict[str, Any]:
    root = _training_root(run_key)
    volume_root = str(root.relative_to(TRAINING_MOUNT))
    target = int(spec["target"])
    errors: list[str] = []
    step = 0
    validated = False
    validation_reason = "checkpoint unavailable"
    marker = root / "latest_checkpointed_iteration.txt"

    checkpoint_entries: list[tuple[int, float]] = []
    try:
        # One Volume RPC is much faster than stat'ing every FUSE entry.
        volume_entries = list(training_volume.listdir(volume_root))
        marker_entry = next(
            (
                entry
                for entry in volume_entries
                if Path(entry.path).name == "latest_checkpointed_iteration.txt"
            ),
            None,
        )
        marker_mtime = marker_entry.mtime if marker_entry else None
        for entry in volume_entries:
            match = re.fullmatch(r"iter_0*(\d+)", Path(entry.path).name)
            if match:
                checkpoint_entries.append((int(match.group(1)), float(entry.mtime)))
    except Exception as exc:
        marker_mtime = None
        errors.append(f"checkpoint listing: {type(exc).__name__}")
    checkpoint_entries.sort()

    try:
        step = int(marker.read_text().strip())
        if marker_mtime is None:
            marker_mtime = marker.stat().st_mtime
    except (OSError, ValueError) as exc:
        errors.append(f"marker: {type(exc).__name__}")

    if step:
        validated, validation_reason = _checkpoint_is_valid(root, step)
        if not validated:
            errors.append(validation_reason)

    if checkpoint_entries and checkpoint_entries[-1][0] > step:
        candidate = checkpoint_entries[-1][0]
        candidate_valid, candidate_reason = _checkpoint_is_valid(root, candidate)
        if candidate_valid:
            step = candidate
            validated = True
            validation_reason = "ok"
            marker_mtime = checkpoint_entries[-1][1]
        else:
            errors.append(f"latest directory: {candidate_reason}")

    now = time.time()
    age_seconds = max(0.0, now - marker_mtime) if marker_mtime else None
    if step >= target and validated:
        state = "complete"
        activity_source = "durable_target"
    elif validated and age_seconds is not None and age_seconds <= 45 * 60:
        state = "running"
        activity_source = "checkpoint_freshness"
    elif step:
        state = "stale"
        activity_source = "checkpoint_freshness"
    else:
        state = "unknown"
        activity_source = "checkpoint_freshness"

    # Estimate speed from recent durable checkpoint mtimes. Checkpoint saves can
    # share timestamps, so use the widest recent window with positive elapsed.
    steps_per_hour = None
    eta_seconds = None
    recent = checkpoint_entries[-8:]
    if len(recent) >= 2:
        start_step, start_time = recent[0]
        end_step, end_time = recent[-1]
        elapsed = end_time - start_time
        if elapsed > 0 and end_step > start_step:
            steps_per_hour = (end_step - start_step) / elapsed * 3600
            if step < target and steps_per_hour > 0:
                eta_seconds = (target - step) / steps_per_hour * 3600

    return {
        "run_key": run_key,
        "short": spec["short"],
        "family": spec["family"],
        "alpha": spec["alpha"],
        "color": spec["color"],
        "step": step,
        "target": target,
        "remaining": max(0, target - step),
        "progress": min(1.0, step / target) if target else 0.0,
        "state": state,
        "activity_source": activity_source,
        "validated": validated,
        "validation_reason": validation_reason,
        "last_checkpoint_at": _iso_from_epoch(marker_mtime),
        "checkpoint_age_seconds": age_seconds,
        "checkpoint_count": len(checkpoint_entries),
        "steps_per_hour": steps_per_hour,
        "eta_seconds": eta_seconds,
        "modal_app_id": spec["app_id"],
        "modal_url": (
            f"https://modal.com/apps/{WORKSPACE_NAME}/{ENVIRONMENT_NAME}/"
            f"{spec['app_id']}"
        ),
        "errors": errors,
    }


def _fetch_hf_run(run_key: str) -> dict[str, Any]:
    from huggingface_hub import HfApi

    repo_id = f"Pre-to-Post-2/rl_{run_key}"
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    api = HfApi(token=token)
    items = api.list_repo_tree(
        repo_id=repo_id,
        path_in_repo=HF_PREFIX,
        recursive=False,
        expand=False,
    )
    steps: list[int] = []
    for item in items:
        match = re.search(r"/global_step_(\d+)$", item.path)
        if match:
            steps.append(int(match.group(1)))
    steps.sort()
    return {
        "repo_id": repo_id,
        "count": len(steps),
        "latest_step": steps[-1] if steps else 0,
        "url": f"https://huggingface.co/{repo_id}/tree/main/{HF_PREFIX}",
        "error": None,
    }


def _hf_statuses(force: bool = False) -> dict[str, dict[str, Any]]:
    global _hf_cache, _hf_cache_at
    now = time.time()
    with _hf_lock:
        if not force and _hf_cache and now - _hf_cache_at < _HF_TTL_SECONDS:
            return _hf_cache

    fetched: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_fetch_hf_run, run_key): run_key for run_key in RUNS}
        for future in as_completed(futures):
            run_key = futures[future]
            try:
                fetched[run_key] = future.result()
            except Exception as exc:  # A partial HF outage should not break the page.
                previous = _hf_cache.get(run_key, {})
                fetched[run_key] = {
                    "repo_id": f"Pre-to-Post-2/rl_{run_key}",
                    "count": previous.get("count", 0),
                    "latest_step": previous.get("latest_step", 0),
                    "url": (
                        f"https://huggingface.co/Pre-to-Post-2/rl_{run_key}/"
                        f"tree/main/{HF_PREFIX}"
                    ),
                    "error": f"{type(exc).__name__}: {exc}",
                }
    with _hf_lock:
        _hf_cache = fetched
        _hf_cache_at = now
    return fetched


def _metric_record(metrics: dict[str, Any], benchmark: str) -> dict[str, Any]:
    core = f"val-core/test_{benchmark}/reward"
    aux = f"val-aux/test_{benchmark}"
    return {
        "mean": metrics.get(f"{core}/mean@16"),
        "best": metrics.get(f"{core}/best@16/mean"),
        "best_std": metrics.get(f"{core}/best@16/std"),
        "first_move": metrics.get(f"{aux}/first_move_score/mean@16"),
        "legality": metrics.get(
            f"{aux}/first_move_legality_score/mean@16"
        ),
        "pass4": metrics.get(f"{aux}/reward/pass@4"),
    }


def _v2r3_object(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _v2r3_list(value: Any, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    return value


def _v2r3_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _v2r3_int(
    value: Any,
    *,
    label: str,
    minimum: int = 0,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _v2r3_number(value: Any, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _v2r3_rate(value: Any, *, label: str) -> float:
    number = _v2r3_number(value, label=label)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{label} must be in [0, 1]")
    return number


def _v2r3_sha256(value: Any, *, label: str) -> str:
    digest = _v2r3_string(value, label=label)
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _v2r3_modal_id(value: Any, *, label: str, prefix: str) -> str:
    identifier = _v2r3_string(value, label=label)
    if not identifier.startswith(prefix):
        raise ValueError(f"{label} must start with {prefix!r}")
    return identifier


def _v2r3_weight_key(value: Any, *, label: str) -> str:
    weight = _v2r3_number(value, label=label)
    matches = [
        key
        for key, expected, _, _ in V2R3_TRAJECTORY_SPECS
        if math.isclose(weight, expected, rel_tol=0.0, abs_tol=1e-12)
    ]
    if len(matches) != 1:
        raise ValueError(f"{label} is not a frozen v2r3 trajectory weight")
    return matches[0]


def _v2r3_path(
    value: Any,
    *,
    label: str,
    prefix: str,
) -> str:
    path = _v2r3_string(value, label=label)
    if not path.startswith(prefix) or ".." in Path(path).parts:
        raise ValueError(f"{label} must be a safe path under {prefix}")
    return path


def _v2r3_distribution(
    value: Any,
    *,
    label: str,
) -> dict[str, float]:
    source = _v2r3_object(value, label=label)
    required = (
        "mean",
        "min",
        "p50_nearest_rank",
        "p90_nearest_rank",
        "p99_nearest_rank",
        "max",
    )
    normalized = {
        key: _v2r3_number(source.get(key), label=f"{label}.{key}")
        for key in required
    }
    if not (
        normalized["min"]
        <= normalized["p50_nearest_rank"]
        <= normalized["p90_nearest_rank"]
        <= normalized["p99_nearest_rank"]
        <= normalized["max"]
    ):
        raise ValueError(f"{label} quantiles are not monotonic")
    if not normalized["min"] <= normalized["mean"] <= normalized["max"]:
        raise ValueError(f"{label}.mean is outside [min, max]")
    return normalized


def _v2r3_inspector_metrics(
    value: Any,
    *,
    label: str,
    rollout: dict[str, Any],
    prompt_set_sha256: str,
) -> dict[str, Any]:
    source = _v2r3_object(value, label=label)
    if source.get("status") != "pass":
        raise ValueError(f"{label}.status must be 'pass'")
    inspected_at = _v2r3_string(
        source.get("inspected_at"), label=f"{label}.inspected_at"
    )
    if _parse_iso(inspected_at) is None:
        raise ValueError(f"{label}.inspected_at must be an ISO-8601 timestamp")

    expected_rows = rollout["rows_per_audit"]
    expected_groups = rollout["prompt_groups"]
    expected_samples = rollout["samples_per_group"]
    rollout_rows = _v2r3_int(
        source.get("rollout_rows"), label=f"{label}.rollout_rows", minimum=1
    )
    prompt_groups = _v2r3_int(
        source.get("prompt_groups"), label=f"{label}.prompt_groups", minimum=1
    )
    samples_per_group = _v2r3_int(
        source.get("samples_per_group"),
        label=f"{label}.samples_per_group",
        minimum=1,
    )
    if (
        rollout_rows != expected_rows
        or prompt_groups != expected_groups
        or samples_per_group != expected_samples
        or rollout_rows != prompt_groups * samples_per_group
    ):
        raise ValueError(f"{label} does not match the frozen rollout shape")
    if (
        _v2r3_sha256(
            source.get("prompt_set_sha256"),
            label=f"{label}.prompt_set_sha256",
        )
        != prompt_set_sha256
    ):
        raise ValueError(f"{label}.prompt_set_sha256 drifted")

    status_counts_source = _v2r3_object(
        source.get("status_counts"), label=f"{label}.status_counts"
    )
    if set(status_counts_source) != {"completed", "truncated"}:
        raise ValueError(
            f"{label}.status_counts must contain only completed/truncated"
        )
    status_counts = {
        state: _v2r3_int(
            status_counts_source.get(state),
            label=f"{label}.status_counts.{state}",
        )
        for state in ("completed", "truncated")
    }
    if sum(status_counts.values()) != rollout_rows:
        raise ValueError(f"{label}.status_counts do not sum to rollout_rows")

    row_counts = {
        key: _v2r3_int(source.get(key), label=f"{label}.{key}")
        for key in (
            "outputs_with_end_thinking",
            "outputs_with_call_env",
            "rows_with_parsed_moves",
            "joint_valid_protocol_rows",
            "positive_samples",
            "model_response_at_cap_rows",
            "raw_move_without_protocol_rows",
            "raw_move_tokens_in_flagged_rows",
        )
    }
    for key in (
        "outputs_with_end_thinking",
        "outputs_with_call_env",
        "rows_with_parsed_moves",
        "joint_valid_protocol_rows",
        "positive_samples",
        "model_response_at_cap_rows",
        "raw_move_without_protocol_rows",
    ):
        if row_counts[key] > rollout_rows:
            raise ValueError(f"{label}.{key} exceeds rollout_rows")
    if row_counts["joint_valid_protocol_rows"] > row_counts[
        "rows_with_parsed_moves"
    ]:
        raise ValueError(
            f"{label}.joint_valid_protocol_rows exceeds parsed rows"
        )

    joint_groups = _v2r3_int(
        source.get("joint_valid_protocol_groups"),
        label=f"{label}.joint_valid_protocol_groups",
    )
    variance_groups = _v2r3_int(
        source.get("nonzero_variance_groups"),
        label=f"{label}.nonzero_variance_groups",
    )
    if joint_groups > prompt_groups or variance_groups > prompt_groups:
        raise ValueError(f"{label} group counts exceed prompt_groups")

    p_protocol = _v2r3_rate(
        source.get("p_protocol"), label=f"{label}.p_protocol"
    )
    p_solve_given_protocol = _v2r3_rate(
        source.get("p_solve_given_protocol"),
        label=f"{label}.p_solve_given_protocol",
    )
    diagnostic_positive_rate = _v2r3_rate(
        source.get("p_total"), label=f"{label}.p_total"
    )
    variance_rate = _v2r3_rate(
        source.get("variance_rate"), label=f"{label}.variance_rate"
    )
    positive_samples = row_counts["positive_samples"]
    protocol_rows = row_counts["joint_valid_protocol_rows"]
    expected_conditional = (
        positive_samples / protocol_rows if protocol_rows else 0.0
    )
    exact_rates = (
        (
            p_protocol,
            protocol_rows / rollout_rows,
            f"{label}.p_protocol",
        ),
        (
            p_solve_given_protocol,
            expected_conditional,
            f"{label}.p_solve_given_protocol",
        ),
        (
            diagnostic_positive_rate,
            positive_samples / rollout_rows,
            f"{label}.p_total",
        ),
        (
            variance_rate,
            variance_groups / prompt_groups,
            f"{label}.variance_rate",
        ),
    )
    for observed, expected, rate_label in exact_rates:
        if not math.isclose(
            observed, expected, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(f"{rate_label} is inconsistent with counts")

    cap = _v2r3_int(
        source.get("model_response_cap"),
        label=f"{label}.model_response_cap",
        minimum=1,
    )
    if cap != rollout["response_model_token_cap"]:
        raise ValueError(f"{label}.model_response_cap drifted")
    cap_rate = _v2r3_rate(
        source.get("model_response_at_cap_rate"),
        label=f"{label}.model_response_at_cap_rate",
    )
    raw_move_rate = _v2r3_rate(
        source.get("raw_move_without_protocol_rate"),
        label=f"{label}.raw_move_without_protocol_rate",
    )
    if not math.isclose(
        cap_rate,
        row_counts["model_response_at_cap_rows"] / rollout_rows,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(f"{label}.model_response_at_cap_rate is inconsistent")
    if not math.isclose(
        raw_move_rate,
        row_counts["raw_move_without_protocol_rows"] / rollout_rows,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            f"{label}.raw_move_without_protocol_rate is inconsistent"
        )
    seed_groups = _v2r3_int(
        source.get("sampling_seed_groups_verified"),
        label=f"{label}.sampling_seed_groups_verified",
        minimum=1,
    )
    if seed_groups != prompt_groups:
        raise ValueError(f"{label}.sampling seed coverage is incomplete")
    if source.get("deterministic_identity_passed") is not True:
        raise ValueError(
            f"{label}.deterministic_identity_passed must be true"
        )

    return {
        "state": "inspected_pass",
        "inspected_at": inspected_at,
        "rollout_rows": rollout_rows,
        "prompt_groups": prompt_groups,
        "samples_per_group": samples_per_group,
        "status_counts": status_counts,
        **row_counts,
        "joint_valid_protocol_groups": joint_groups,
        "nonzero_variance_groups": variance_groups,
        "p_protocol": p_protocol,
        "p_solve_given_protocol": p_solve_given_protocol,
        "diagnostic_positive_rate": diagnostic_positive_rate,
        "variance_rate": variance_rate,
        "response_length": _v2r3_distribution(
            source.get("response_length"),
            label=f"{label}.response_length",
        ),
        "effective_response_length": _v2r3_distribution(
            source.get("effective_response_length"),
            label=f"{label}.effective_response_length",
        ),
        "env_token_count": _v2r3_distribution(
            source.get("env_token_count"),
            label=f"{label}.env_token_count",
        ),
        "model_response_cap": cap,
        "model_response_at_cap_rate": cap_rate,
        "raw_move_without_protocol_rate": raw_move_rate,
        "sampling_seed_groups_verified": seed_groups,
        "deterministic_identity_passed": True,
        "metric_semantics": (
            "diagnostic binary-positive audit; not an official evaluation"
        ),
    }


def _v2r3_final_report(
    contract: dict[str, Any],
    *,
    expected_records: int,
) -> dict[str, Any]:
    report_path = _v2r3_path(
        contract.get("report_path"),
        label="v2r3_diagnostic_contract.report_path",
        prefix="/artifacts/",
    )
    value = contract.get("final_report")
    if value is None:
        return {
            "state": "pending",
            "path": report_path,
            "immutable": False,
            "report_sha256": None,
            "record_count": 0,
        }

    report = _v2r3_object(
        value, label="v2r3_diagnostic_contract.final_report"
    )
    status = _v2r3_string(
        report.get("status"),
        label="v2r3_diagnostic_contract.final_report.status",
    )
    if status not in {
        "immutable_authenticated",
        "authenticated_immutable",
        "complete_authenticated",
    }:
        raise ValueError(
            "v2r3_diagnostic_contract.final_report.status is not authenticated"
        )
    if report.get("immutable") is not True:
        raise ValueError(
            "v2r3_diagnostic_contract.final_report.immutable must be true"
        )
    if (
        _v2r3_path(
            report.get("path"),
            label="v2r3_diagnostic_contract.final_report.path",
            prefix="/artifacts/",
        )
        != report_path
    ):
        raise ValueError("v2r3 final report path does not match the contract")
    schema = _v2r3_string(
        report.get("schema"),
        label="v2r3_diagnostic_contract.final_report.schema",
    )
    if schema != V2R3_DIAGNOSTIC_REPORT_SCHEMA:
        raise ValueError("v2r3 final report schema drifted")
    record_count = _v2r3_int(
        report.get("record_count"),
        label="v2r3_diagnostic_contract.final_report.record_count",
    )
    if record_count != expected_records:
        raise ValueError("v2r3 final report has incomplete record coverage")
    call_id = _v2r3_modal_id(
        report.get("audit_call_id"),
        label="v2r3_diagnostic_contract.final_report.audit_call_id",
        prefix="fc-",
    )
    reported_at = _v2r3_string(
        report.get("reported_at"),
        label="v2r3_diagnostic_contract.final_report.reported_at",
    )
    if _parse_iso(reported_at) is None:
        raise ValueError("v2r3 final report timestamp is invalid")
    return {
        "state": status,
        "path": report_path,
        "immutable": True,
        "schema": schema,
        "report_sha256": _v2r3_sha256(
            report.get("report_sha256"),
            label="v2r3_diagnostic_contract.final_report.report_sha256",
        ),
        "record_count": record_count,
        "audit_call_id": call_id,
        "reported_at": reported_at,
    }


def _validate_v2r3_diagnostic_contract(
    registry: dict[str, Any],
) -> dict[str, Any]:
    """Validate and normalize the optional non-authorizing v2r3 grid."""

    value = registry.get("v2r3_diagnostic_contract")
    if value is None:
        return {
            "present": False,
            "status": "not_registered",
            "authorization": {
                "scope": "diagnostic_only",
                "production": False,
                "p1": False,
                "exp2": False,
                "rl": False,
            },
            "trajectories": [],
            "rollouts": [],
            "aggregate": {
                "trajectory_count": 0,
                "training_call_count": 0,
                "terminal_training_calls": 0,
                "expected_snapshot_count": V2R3_EXPECTED_SNAPSHOTS,
                "authenticated_snapshot_count": 0,
                "rollout_launched": 0,
                "rollout_completed": 0,
            },
            "final_report": {
                "state": "not_registered",
                "path": None,
                "immutable": False,
                "report_sha256": None,
                "record_count": 0,
            },
        }

    contract = _v2r3_object(
        value, label="v2r3_diagnostic_contract"
    )
    if contract.get("schema") != V2R3_DIAGNOSTIC_SCHEMA:
        raise ValueError("v2r3 diagnostic contract schema drifted")
    version = _v2r3_string(
        contract.get("version"), label="v2r3_diagnostic_contract.version"
    )
    status = _v2r3_string(
        contract.get("status"), label="v2r3_diagnostic_contract.status"
    )
    if contract.get("diagnostic_only") is not True:
        raise ValueError("v2r3 contract must remain diagnostic_only")
    authorization_fields = {
        "production": "production_authorized",
        "p1": "p1_authorized",
        "exp2": "exp2_authorized",
        "rl": "rl_authorized",
    }
    for field in authorization_fields.values():
        if contract.get(field) is not False:
            raise ValueError(f"v2r3 contract {field} must be false")
    plan_path = _v2r3_string(
        contract.get("plan_path"), label="v2r3_diagnostic_contract.plan_path"
    )
    if Path(plan_path).name != plan_path or ".." in Path(plan_path).parts:
        raise ValueError("v2r3 diagnostic plan_path must be a local filename")
    plan_sha256 = _v2r3_sha256(
        contract.get("plan_sha256"),
        label="v2r3_diagnostic_contract.plan_sha256",
    )

    data = _v2r3_object(
        contract.get("data"), label="v2r3_diagnostic_contract.data"
    )
    prompt_set_sha256 = _v2r3_sha256(
        data.get("seed42_prompt_set_sha256"),
        label=(
            "v2r3_diagnostic_contract.data.seed42_prompt_set_sha256"
        ),
    )
    pretraining = _v2r3_object(
        contract.get("pretraining"),
        label="v2r3_diagnostic_contract.pretraining",
    )
    gpus_per_trajectory = _v2r3_int(
        pretraining.get("gpus_per_trajectory"),
        label="v2r3_diagnostic_contract.pretraining.gpus_per_trajectory",
        minimum=1,
    )
    gpu_type = _v2r3_string(
        pretraining.get("gpu_type"),
        label="v2r3_diagnostic_contract.pretraining.gpu_type",
    )

    trajectory_source = _v2r3_list(
        contract.get("trajectories"),
        label="v2r3_diagnostic_contract.trajectories",
    )
    if len(trajectory_source) != len(V2R3_TRAJECTORY_SPECS):
        raise ValueError("v2r3 contract must contain exactly four trajectories")
    trajectories: list[dict[str, Any]] = []
    for index, (source, spec) in enumerate(
        zip(trajectory_source, V2R3_TRAJECTORY_SPECS, strict=True)
    ):
        key, expected_weight, max_steps, snapshot_steps = spec
        record = _v2r3_object(
            source, label=f"v2r3_diagnostic_contract.trajectories[{index}]"
        )
        if (
            _v2r3_weight_key(
                record.get("sft_loss_weight"),
                label=(
                    "v2r3_diagnostic_contract."
                    f"trajectories[{index}].sft_loss_weight"
                ),
            )
            != key
        ):
            raise ValueError("v2r3 trajectory order or weight drifted")
        if (
            _v2r3_int(
                record.get("max_steps"),
                label=(
                    "v2r3_diagnostic_contract."
                    f"trajectories[{index}].max_steps"
                ),
                minimum=1,
            )
            != max_steps
        ):
            raise ValueError("v2r3 trajectory max_steps drifted")
        observed_steps = tuple(
            _v2r3_int(
                item,
                label=(
                    "v2r3_diagnostic_contract."
                    f"trajectories[{index}].snapshot_steps"
                ),
                minimum=1,
            )
            for item in _v2r3_list(
                record.get("snapshot_steps"),
                label=(
                    "v2r3_diagnostic_contract."
                    f"trajectories[{index}].snapshot_steps"
                ),
            )
        )
        if observed_steps != snapshot_steps:
            raise ValueError("v2r3 trajectory snapshot inventory drifted")
        trajectories.append(
            {
                "weight_key": key,
                "sft_loss_weight": expected_weight,
                "target_step": max_steps,
                "snapshot_steps": list(snapshot_steps),
            }
        )
    if (
        _v2r3_int(
            contract.get("training_call_count"),
            label="v2r3_diagnostic_contract.training_call_count",
        )
        != len(trajectories)
    ):
        raise ValueError("v2r3 training_call_count is inconsistent")
    if (
        _v2r3_int(
            contract.get("snapshot_count"),
            label="v2r3_diagnostic_contract.snapshot_count",
        )
        != V2R3_EXPECTED_SNAPSHOTS
    ):
        raise ValueError("v2r3 snapshot_count is inconsistent")

    training_calls_source = _v2r3_object(
        contract.get("training_calls"),
        label="v2r3_diagnostic_contract.training_calls",
    )
    expected_weight_keys = {item[0] for item in V2R3_TRAJECTORY_SPECS}
    if set(training_calls_source) != expected_weight_keys:
        raise ValueError(
            "v2r3 training_calls must bind all four frozen trajectories"
        )
    training_call_ids: set[str] = set()
    trajectory_by_key = {item["weight_key"]: item for item in trajectories}
    for key, _, max_steps, snapshot_steps in V2R3_TRAJECTORY_SPECS:
        source = _v2r3_object(
            training_calls_source[key],
            label=f"v2r3_diagnostic_contract.training_calls.{key}",
        )
        if (
            _v2r3_weight_key(
                source.get("sft_loss_weight"),
                label=(
                    "v2r3_diagnostic_contract."
                    f"training_calls.{key}.sft_loss_weight"
                ),
            )
            != key
        ):
            raise ValueError("v2r3 training call weight drifted")
        call_id = _v2r3_modal_id(
            source.get("call_id"),
            label=f"v2r3_diagnostic_contract.training_calls.{key}.call_id",
            prefix="fc-",
        )
        if call_id in training_call_ids:
            raise ValueError("v2r3 training call IDs must be distinct")
        training_call_ids.add(call_id)
        call_status = _v2r3_string(
            source.get("status"),
            label=f"v2r3_diagnostic_contract.training_calls.{key}.status",
        )
        latest_snapshot = source.get("last_authenticated_snapshot_step", 0)
        latest_snapshot = _v2r3_int(
            latest_snapshot,
            label=(
                "v2r3_diagnostic_contract."
                f"training_calls.{key}.last_authenticated_snapshot_step"
            ),
        )
        if latest_snapshot and latest_snapshot not in snapshot_steps:
            raise ValueError(
                "v2r3 last authenticated snapshot is outside the inventory"
            )
        completed_steps = source.get("completed_max_steps")
        if completed_steps is not None:
            completed_steps = _v2r3_int(
                completed_steps,
                label=(
                    "v2r3_diagnostic_contract."
                    f"training_calls.{key}.completed_max_steps"
                ),
                minimum=1,
            )
            if completed_steps != max_steps:
                raise ValueError("v2r3 completed_max_steps drifted")
        terminal = call_status in {
            "success_authenticated",
            "failed",
            "cancelled",
            "terminated",
        }
        if call_status == "success_authenticated" and (
            completed_steps != max_steps
            or latest_snapshot != snapshot_steps[-1]
            or not isinstance(source.get("result_path"), str)
        ):
            raise ValueError(
                "v2r3 authenticated success lacks terminal provenance"
            )
        progress_step = (
            max_steps if call_status == "success_authenticated"
            else latest_snapshot
        )
        trajectory_by_key[key].update(
            {
                "training_call_id": call_id,
                "training_status": call_status,
                "terminal": terminal,
                "latest_authenticated_snapshot_step": latest_snapshot,
                "progress_step": progress_step,
                "progress": progress_step / max_steps,
                "authenticated_snapshot_count": 0,
                "snapshot_file_count": 0,
                "snapshot_total_bytes": 0,
            }
        )

    rollout = _v2r3_object(
        contract.get("rollout"), label="v2r3_diagnostic_contract.rollout"
    )
    frozen_rollout_values = {
        "audits": V2R3_EXPECTED_SNAPSHOTS,
        "seed": 42,
        "prompt_groups": 256,
        "samples_per_group": 8,
        "rows_per_audit": 2_048,
        "gpus_per_audit": 8,
        "max_tokens_per_gpu": 131_072,
        "sglang_server_concurrency": 128,
        "response_model_token_cap": 2_560,
    }
    normalized_rollout: dict[str, Any] = {}
    for key, expected in frozen_rollout_values.items():
        observed = _v2r3_int(
            rollout.get(key),
            label=f"v2r3_diagnostic_contract.rollout.{key}",
            minimum=1,
        )
        if observed != expected:
            raise ValueError(f"v2r3 rollout {key} drifted")
        normalized_rollout[key] = observed
    for key, expected in {
        "dynamic_filter": False,
        "debug_rollout_only": True,
        "deterministic_inference": True,
    }.items():
        if rollout.get(key) is not expected:
            raise ValueError(f"v2r3 rollout {key} drifted")
        normalized_rollout[key] = expected
    normalized_rollout["gpu_type"] = _v2r3_string(
        rollout.get("gpu_type"),
        label="v2r3_diagnostic_contract.rollout.gpu_type",
    )

    expected_pairs = {
        (key, step)
        for key, _, _, snapshot_steps in V2R3_TRAJECTORY_SPECS
        for step in snapshot_steps
    }
    rollout_calls_source = _v2r3_list(
        contract.get("rollout_calls"),
        label="v2r3_diagnostic_contract.rollout_calls",
    )
    if len(rollout_calls_source) > V2R3_EXPECTED_SNAPSHOTS:
        raise ValueError("v2r3 rollout call count exceeds the frozen inventory")
    rollout_by_pair: dict[tuple[str, int], dict[str, Any]] = {}
    rollout_call_ids: set[str] = set()
    snapshot_paths: set[str] = set()
    snapshot_identity_sets = {
        name: set()
        for name in (
            "snapshot_marker_sha256",
            "trainer_state_sha256",
            "hf_identity_sha256",
            "resume_identity_sha256",
        )
    }
    total_files = 0
    total_bytes = 0
    for index, source_value in enumerate(rollout_calls_source):
        label = f"v2r3_diagnostic_contract.rollout_calls[{index}]"
        source = _v2r3_object(source_value, label=label)
        weight_key = _v2r3_weight_key(
            source.get("sft_loss_weight"),
            label=f"{label}.sft_loss_weight",
        )
        step = _v2r3_int(
            source.get("step"), label=f"{label}.step", minimum=1
        )
        pair = (weight_key, step)
        if pair not in expected_pairs or pair in rollout_by_pair:
            raise ValueError(
                "v2r3 rollout calls contain an unexpected or duplicate pair"
            )
        trajectory = trajectory_by_key[weight_key]
        training_call_id = _v2r3_modal_id(
            source.get("training_call_id"),
            label=f"{label}.training_call_id",
            prefix="fc-",
        )
        if training_call_id != trajectory["training_call_id"]:
            raise ValueError("v2r3 rollout training call provenance drifted")
        snapshot_path = _v2r3_path(
            source.get("snapshot_path"),
            label=f"{label}.snapshot_path",
            prefix="/checkpoints/",
        )
        if (
            not snapshot_path.endswith(f"/snapshots/step_{step}")
            or snapshot_path in snapshot_paths
        ):
            raise ValueError(
                "v2r3 snapshot path is inconsistent or duplicated"
            )
        snapshot_paths.add(snapshot_path)
        identities: dict[str, str] = {}
        for identity_name, seen in snapshot_identity_sets.items():
            identity = _v2r3_sha256(
                source.get(identity_name),
                label=f"{label}.{identity_name}",
            )
            if identity in seen:
                raise ValueError(
                    f"v2r3 {identity_name} must be unique per snapshot"
                )
            seen.add(identity)
            identities[identity_name] = identity
        file_count = _v2r3_int(
            source.get("snapshot_file_count"),
            label=f"{label}.snapshot_file_count",
            minimum=1,
        )
        total_bytes_for_snapshot = _v2r3_int(
            source.get("snapshot_total_bytes"),
            label=f"{label}.snapshot_total_bytes",
            minimum=1,
        )
        total_files += file_count
        total_bytes += total_bytes_for_snapshot

        rollout_call_id = _v2r3_modal_id(
            source.get("rollout_call_id"),
            label=f"{label}.rollout_call_id",
            prefix="fc-",
        )
        if rollout_call_id in rollout_call_ids:
            raise ValueError("v2r3 rollout FunctionCall IDs must be distinct")
        rollout_call_ids.add(rollout_call_id)
        rollout_app_id = _v2r3_modal_id(
            source.get("rollout_app_id"),
            label=f"{label}.rollout_app_id",
            prefix="ap-",
        )
        run_name = _v2r3_string(
            source.get("run_name"), label=f"{label}.run_name"
        )
        call_status = _v2r3_string(
            source.get("status"), label=f"{label}.status"
        )
        if call_status not in V2R3_ROLLOUT_STATES:
            raise ValueError(f"{label}.status is not a recognized state")
        for exact_flag, expected in {
            "exactly_once": True,
            "rl_root_absent_before_launch": True,
            "deterministic_inference": True,
            "debug_rollout_only": True,
        }.items():
            if source.get(exact_flag) is not expected:
                raise ValueError(f"{label}.{exact_flag} drifted")

        inspector = None
        artifact_sha256 = None
        if call_status == "inspected_success":
            inspector = _v2r3_inspector_metrics(
                source.get("inspector"),
                label=f"{label}.inspector",
                rollout=normalized_rollout,
                prompt_set_sha256=prompt_set_sha256,
            )
            artifact_source = _v2r3_object(
                source.get("artifact_sha256"),
                label=f"{label}.artifact_sha256",
            )
            required_artifacts = (
                "provenance",
                "rollout",
                "positive_rows",
                "positive_summary",
            )
            artifact_sha256 = {
                key: _v2r3_sha256(
                    artifact_source.get(key),
                    label=f"{label}.artifact_sha256.{key}",
                )
                for key in required_artifacts
            }
            _v2r3_sha256(
                source.get("provenance_identity_sha256"),
                label=f"{label}.provenance_identity_sha256",
            )
            _v2r3_sha256(
                source.get("command_sha256"),
                label=f"{label}.command_sha256",
            )
        elif source.get("inspector") is not None:
            raise ValueError(
                f"{label}.inspector is present before inspected_success"
            )

        normalized = {
            "weight_key": weight_key,
            "sft_loss_weight": trajectory["sft_loss_weight"],
            "step": step,
            "launched": True,
            "training_call_id": training_call_id,
            "snapshot_path": snapshot_path,
            **identities,
            "snapshot_file_count": file_count,
            "snapshot_total_bytes": total_bytes_for_snapshot,
            "rollout_app_id": rollout_app_id,
            "rollout_call_id": rollout_call_id,
            "run_name": run_name,
            "status": call_status,
            "state": V2R3_ROLLOUT_STATES[call_status],
            "inspector": inspector,
            "artifact_sha256": artifact_sha256,
            "metric_semantics": (
                "diagnostic_only; never an official pass@1 evaluation"
            ),
        }
        rollout_by_pair[pair] = normalized
        trajectory["authenticated_snapshot_count"] += 1
        trajectory["snapshot_file_count"] += file_count
        trajectory["snapshot_total_bytes"] += total_bytes_for_snapshot

    rollout_rows: list[dict[str, Any]] = []
    for key, weight, _, snapshot_steps in V2R3_TRAJECTORY_SPECS:
        trajectory = trajectory_by_key[key]
        for step in snapshot_steps:
            row = rollout_by_pair.get((key, step))
            if row is None:
                row = {
                    "weight_key": key,
                    "sft_loss_weight": weight,
                    "step": step,
                    "launched": False,
                    "training_call_id": trajectory["training_call_id"],
                    "snapshot_path": None,
                    "snapshot_marker_sha256": None,
                    "trainer_state_sha256": None,
                    "hf_identity_sha256": None,
                    "resume_identity_sha256": None,
                    "snapshot_file_count": 0,
                    "snapshot_total_bytes": 0,
                    "rollout_app_id": None,
                    "rollout_call_id": None,
                    "run_name": None,
                    "status": "not_launched",
                    "state": "pending",
                    "inspector": None,
                    "artifact_sha256": None,
                    "metric_semantics": (
                        "diagnostic_only; never an official pass@1 evaluation"
                    ),
                }
            rollout_rows.append(row)

    progress = _v2r3_object(
        contract.get("rollout_launch_progress"),
        label="v2r3_diagnostic_contract.rollout_launch_progress",
    )
    observed_counts = {
        "running_or_queued": sum(
            row["status"] == "running_or_queued"
            for row in rollout_by_pair.values()
        ),
        "success_pending_artifact_audit": sum(
            row["status"] == "success_pending_artifact_audit"
            for row in rollout_by_pair.values()
        ),
        "completed": sum(
            row["status"] == "inspected_success"
            for row in rollout_by_pair.values()
        ),
        "failed": sum(
            row["status"] == "failed" for row in rollout_by_pair.values()
        ),
    }
    expected_progress = {
        "launched": len(rollout_by_pair),
        "total": V2R3_EXPECTED_SNAPSHOTS,
        **observed_counts,
        "duplicate_submissions": 0,
    }
    for key, expected in expected_progress.items():
        observed = _v2r3_int(
            progress.get(key),
            label=(
                "v2r3_diagnostic_contract.rollout_launch_progress."
                f"{key}"
            ),
        )
        if observed != expected:
            raise ValueError(
                f"v2r3 rollout_launch_progress.{key} is inconsistent"
            )
    progress_updated_at = _v2r3_string(
        progress.get("updated_at"),
        label=(
            "v2r3_diagnostic_contract."
            "rollout_launch_progress.updated_at"
        ),
    )
    if _parse_iso(progress_updated_at) is None:
        raise ValueError("v2r3 rollout progress timestamp is invalid")

    final_report = _v2r3_final_report(
        contract, expected_records=V2R3_EXPECTED_SNAPSHOTS
    )
    return {
        "present": True,
        "schema": V2R3_DIAGNOSTIC_SCHEMA,
        "version": version,
        "status": status,
        "plan": {"path": plan_path, "sha256": plan_sha256},
        "authorization": {
            "scope": "diagnostic_only",
            **{
                output_name: False
                for output_name in authorization_fields
            },
            "statement": (
                "Diagnostic-only and non-authorizing: no production, P1, "
                "Exp2, or RL launch authorization."
            ),
        },
        "hardware": {
            "pretraining": {
                "gpus_per_trajectory": gpus_per_trajectory,
                "gpu_type": gpu_type,
            },
            "rollout": {
                "gpus_per_audit": normalized_rollout["gpus_per_audit"],
                "gpu_type": normalized_rollout["gpu_type"],
            },
        },
        "trajectories": trajectories,
        "rollouts": rollout_rows,
        "aggregate": {
            "trajectory_count": len(trajectories),
            "training_call_count": len(training_call_ids),
            "terminal_training_calls": sum(
                item["terminal"] for item in trajectories
            ),
            "expected_snapshot_count": V2R3_EXPECTED_SNAPSHOTS,
            "authenticated_snapshot_count": len(rollout_by_pair),
            "snapshot_file_count": total_files,
            "snapshot_total_bytes": total_bytes,
            "rollout_launched": len(rollout_by_pair),
            "rollout_running_or_queued": observed_counts[
                "running_or_queued"
            ],
            "rollout_pending_artifact_audit": observed_counts[
                "success_pending_artifact_audit"
            ],
            "rollout_completed": observed_counts["completed"],
            "rollout_failed": observed_counts["failed"],
            "duplicate_submissions": 0,
            "progress": len(rollout_by_pair) / V2R3_EXPECTED_SNAPSHOTS,
            "updated_at": progress_updated_at,
        },
        "final_report": final_report,
        "metric_semantics": {
            "scope": "diagnostic_only",
            "official_evaluation": False,
            "display_label": "diagnostic positives / rate",
        },
    }


def _load_embedded_registry() -> dict[str, Any]:
    candidates = (INTERLEAVE_REGISTRY_REMOTE, INTERLEAVE_REGISTRY_LOCAL)
    registry_path = next((path for path in candidates if path.is_file()), None)
    if registry_path is None:
        raise FileNotFoundError(
            "INTERLEAVED_CORE_REGISTRY.json was not packaged with the dashboard"
        )
    return _validate_registry(_read_json(registry_path))


def _validate_registry(registry: Any) -> dict[str, Any]:
    if not isinstance(registry, dict):
        raise ValueError("interleave registry must be an object")
    raw_root = Path(str(registry["rl_raw_root"])).name
    if raw_root != INTERLEAVE_RAW_ROOT:
        raise ValueError(
            f"Registry RL root {raw_root!r} does not match dashboard root "
            f"{INTERLEAVE_RAW_ROOT!r}"
        )
    flatten_core_registry(registry)
    _validate_v2r3_diagnostic_contract(registry)
    return registry


def _latest_call_record(
    registry: dict[str, Any], stage_id: str
) -> dict[str, Any]:
    call_sources = [
        registry.get("orchestration", {}).get("calls", {}),
        registry.get("exp4", {}).get("orchestration", {}).get("calls", {}),
    ]
    for calls in call_sources:
        records = calls.get(stage_id, []) if isinstance(calls, dict) else []
        if isinstance(records, list) and records and isinstance(records[-1], dict):
            return dict(records[-1])
    return {}


def _pretrain_volume_path(value: str, *, final_endpoint: bool = False) -> str:
    normalized = "/" + value.strip().lstrip("/")
    if not normalized.startswith("/checkpoints/"):
        raise ValueError(f"pretraining path must be under /checkpoints: {value}")
    relative = normalized.removeprefix("/checkpoints/")
    if final_endpoint:
        path = Path(relative)
        if path.name != "final":
            raise ValueError(f"pretraining endpoint must end in /final: {value}")
        relative = str(path.parent)
    if ".." in Path(relative).parts:
        raise ValueError(f"unsafe pretraining path: {value}")
    return relative


def _pretrain_stage_specs(registry: dict[str, Any]) -> list[dict[str, Any]]:
    """Derive the five core pretrain legs plus explicit future extensions."""

    fixed = registry["fixed_pretraining"]
    shared = registry["shared_pretraining"]
    root = _pretrain_volume_path(str(registry["pretrain_root"]))
    leg_steps = int(fixed["p2_steps"])
    specs: list[dict[str, Any]] = [
        {
            "stage_id": "p1",
            "label": "Shared P1",
            "target": int(fixed["p1_steps"]),
            "output_path": f"{root}/p1_shared",
            "call_stage": "p1",
            "registry_status": shared["p1"].get("status", "planned"),
        },
        {
            "stage_id": "exp2",
            "label": "Exp 2 · monolithic",
            "target": int(fixed["monolithic_steps"]),
            "output_path": f"{root}/exp2_monolithic",
            "call_stage": "exp2",
            "registry_status": shared["exp2_monolithic"].get(
                "status", "planned"
            ),
        },
    ]

    p2_specs = [
        (
            "exp3-p2",
            "Exp 3 · P2",
            shared["exp3_p2"],
            "exp3-p2",
        )
    ]
    for arm in registry["core_arms"]:
        if arm["experiment"] != "E1":
            continue
        filter_code = str(arm["filter"]).lower()
        p2_specs.append(
            (
                f"e1-{filter_code}-p2",
                f"Exp 1 {filter_code.upper()} · P2",
                arm["p2"],
                f"e1-{filter_code}-p2",
            )
        )
    for stage_id, label, record, call_stage in p2_specs:
        endpoint = record.get("endpoint")
        specs.append(
            {
                "stage_id": stage_id,
                "label": label,
                "target": leg_steps,
                "output_path": (
                    _pretrain_volume_path(str(endpoint), final_endpoint=True)
                    if endpoint
                    else None
                ),
                "output_parent": f"{root}/p2",
                "output_prefix": f"{record['run_id']}-from-",
                "call_stage": call_stage,
                "registry_status": record.get("status", "planned"),
            }
        )

    exp4 = registry.get("exp4", {})
    if exp4 and not isinstance(exp4, dict):
        raise ValueError("registry.exp4 must be an object")
    exp4_arms = exp4.get("arms", []) if isinstance(exp4, dict) else []
    if exp4_arms and not isinstance(exp4_arms, list):
        raise ValueError("registry.exp4.arms must be a list")
    for arm in exp4_arms:
        if not isinstance(arm, dict):
            raise ValueError("registry.exp4 arm must be an object")
        pretrain = arm.get("stages", {}).get("pretrain", {})
        if not isinstance(pretrain, dict):
            raise ValueError("registry.exp4 pretrain stage must be an object")
        stage_id = str(pretrain["stage_id"])
        endpoint = pretrain.get("endpoint")
        target_value = pretrain.get("target_steps")
        if target_value is not None:
            target_value = int(target_value)
            if target_value <= 0:
                raise ValueError(f"{stage_id} target_steps must be positive")
        output_prefix = pretrain.get("output_prefix")
        specs.append(
            {
                "stage_id": stage_id,
                "label": (
                    f"Exp 4 {str(arm['filter']).upper()} · "
                    f"{str(arm['method']).upper()}"
                ),
                "target": target_value,
                "output_path": (
                    _pretrain_volume_path(str(endpoint), final_endpoint=True)
                    if endpoint
                    else None
                ),
                "recursive_output_root": (
                    _pretrain_volume_path(str(output_prefix))
                    if output_prefix
                    else None
                ),
                "call_stage": stage_id,
                "registry_status": pretrain.get(
                    "status", arm.get("status", "planned")
                ),
            }
        )

    dashboard = registry.get("dashboard", {})
    explicit = (
        dashboard.get("pretraining_stages", [])
        if isinstance(dashboard, dict)
        else []
    )
    if explicit and not isinstance(explicit, list):
        raise ValueError("dashboard.pretraining_stages must be a list")
    seen = {spec["stage_id"] for spec in specs}
    for item in explicit:
        if not isinstance(item, dict):
            raise ValueError("dashboard pretraining stage must be an object")
        stage_id = str(item["stage_id"])
        if stage_id in seen:
            raise ValueError(f"duplicate dashboard pretraining stage {stage_id}")
        seen.add(stage_id)
        output_path = item.get("output_path")
        output_prefix = item.get("output_prefix")
        if output_path:
            output_path = _pretrain_volume_path(str(output_path))
        if output_prefix:
            prefix_relative = _pretrain_volume_path(str(output_prefix))
            output_parent = str(Path(prefix_relative).parent)
            output_prefix = Path(prefix_relative).name
        else:
            output_parent = None
        specs.append(
            {
                "stage_id": stage_id,
                "label": str(item["label"]),
                "target": int(item["target_steps"]),
                "output_path": output_path,
                "output_parent": output_parent,
                "output_prefix": output_prefix,
                "call_stage": str(item.get("call_stage") or stage_id),
                "registry_status": str(item.get("status") or "planned"),
            }
        )
    return specs


def _resolve_pretrain_output(spec: dict[str, Any]) -> str | None:
    if spec.get("output_path"):
        return str(spec["output_path"])
    recursive_root = spec.get("recursive_output_root")
    if recursive_root:
        try:
            matches = sorted(
                {
                    str(Path(entry.path).parent)
                    for entry in pretrain_volume.listdir(
                        str(recursive_root),
                        recursive=True,
                    )
                    if Path(entry.path).name == "metrics.jsonl"
                }
            )
        except Exception:
            return None
        if len(matches) > 1:
            raise ValueError(
                f"ambiguous recursive output {recursive_root}: "
                f"{len(matches)} metrics streams"
            )
        return matches[0] if matches else None
    parent = spec.get("output_parent")
    prefix = spec.get("output_prefix")
    if not parent or not prefix:
        return None
    try:
        matches = sorted(
            {
                entry.path.rstrip("/")
                for entry in pretrain_volume.listdir(str(parent))
                if Path(entry.path).name.startswith(str(prefix))
            }
        )
    except Exception:
        return None
    if len(matches) > 1:
        raise ValueError(
            f"ambiguous output prefix {parent}/{prefix}: {len(matches)} matches"
        )
    return matches[0] if matches else None


def _pretrain_default_state(
    registry_status: str, call: dict[str, Any]
) -> str:
    call_state = str(call.get("modal_state") or call.get("status") or "").lower()
    if call_state in {"failure", "failed", "terminated", "cancelled", "canceled"}:
        return "failed"
    if call_state in {"running", "submitted", "pending"}:
        return "submitted"
    status = registry_status.lower()
    if "complete" in status:
        return "complete"
    if "submitted" in status or "running" in status:
        return "submitted"
    if "blocked" in status:
        return "blocked"
    return "planned"


def _pretrain_stage_status(
    registry: dict[str, Any],
    spec: dict[str, Any],
) -> dict[str, Any]:
    now = time.time()
    call = _latest_call_record(registry, str(spec["call_stage"]))
    reported_modal_state = call.get("modal_state") or call.get("status")
    target = spec.get("target")
    if target is not None:
        target = int(target)
    result: dict[str, Any] = {
        "stage_id": spec["stage_id"],
        "label": spec["label"],
        "step": 0,
        "target": target,
        "remaining": target,
        "progress": 0.0,
        "loss": None,
        "lr": None,
        "tokens_per_second": None,
        "eta_seconds": None,
        "last_update_at": None,
        "last_update_age_seconds": None,
        "runtime_provenance": None,
        "manifest_hash": None,
        "metric_records": 0,
        "resume_checkpoint_step": None,
        "state": _pretrain_default_state(
            str(spec["registry_status"]),
            call,
        ),
        "registry_status": str(spec["registry_status"]),
        "call_id": call.get("call_id"),
        "modal_state": reported_modal_state,
        "reported_modal_state": reported_modal_state,
        "modal_state_source": "call_record",
        "output_path": None,
        "validated": False,
        "errors": [],
    }
    try:
        output_relative = _resolve_pretrain_output(spec)
    except ValueError as exc:
        result["state"] = "error"
        result["errors"].append(str(exc))
        return result
    result["output_path"] = output_relative
    if output_relative is None:
        return result

    output = PRETRAIN_MOUNT / output_relative
    metrics_path = output / "metrics.jsonl"
    if not metrics_path.is_file():
        return result
    if target is None:
        result["state"] = "error"
        result["errors"].append(
            "metrics exist but registry target_steps is unresolved"
        )
        return result
    try:
        metric_mtime = metrics_path.stat().st_mtime
        metric_summary = parse_pretrain_metrics_jsonl(
            metrics_path.read_text(encoding="utf-8"),
            target_step=target,
            last_update_at=_iso_from_epoch(metric_mtime),
        )
        result.update(metric_summary)
        result["last_update_age_seconds"] = max(0.0, now - metric_mtime)
        result["validated"] = True
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        result["state"] = "error"
        result["errors"].append(f"metrics: {type(exc).__name__}: {exc}")
        return result

    state_path = output / "latest" / "trainer_state.json"
    if state_path.is_file():
        try:
            state = validate_pretrain_trainer_state(
                _read_json(state_path),
                metrics=metric_summary,
            )
            result["resume_checkpoint_step"] = state["step"]
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            result["state"] = "error"
            result["validated"] = False
            result["errors"].append(
                f"trainer state: {type(exc).__name__}: {exc}"
            )
            return result

    final = output / "final"
    final_weights = list(final.glob("model*.safetensors"))
    final_valid = (final / "config.json").is_file() and bool(final_weights)
    age_seconds = result["last_update_age_seconds"]
    if target is not None and result["step"] >= target and final_valid:
        result["state"] = "complete"
        result["eta_seconds"] = 0.0
        # The clean HF export and terminal metric stream are stronger evidence
        # than an asynchronously refreshed call record. Preserve the reported
        # value for audit while preventing a completed stage from displaying
        # a stale "running" Modal state.
        result["modal_state"] = "success"
        result["modal_state_source"] = "durable_final"
    elif target is not None and result["step"] >= target:
        result["state"] = "finishing"
    elif isinstance(age_seconds, (int, float)) and age_seconds <= 15 * 60:
        result["state"] = "running"
    else:
        result["state"] = "stale"
    return result


def _collect_pretraining(registry: dict[str, Any]) -> dict[str, Any]:
    stages: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    try:
        specs = _pretrain_stage_specs(registry)
    except Exception as exc:
        return {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "stages": {},
            "aggregate": {},
            "errors": [f"stage specs: {type(exc).__name__}: {exc}"],
        }
    for spec in specs:
        try:
            status = _pretrain_stage_status(registry, spec)
        except Exception as exc:
            status = {
                "stage_id": spec["stage_id"],
                "label": spec["label"],
                "step": 0,
                "target": spec.get("target"),
                "progress": 0.0,
                "state": "error",
                "errors": [f"{type(exc).__name__}: {exc}"],
            }
        stages[str(spec["stage_id"])] = status
        errors.extend(
            f"{spec['stage_id']}: {message}"
            for message in status.get("errors", [])
        )
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stages": stages,
        "aggregate": {
            "stage_count": len(stages),
            "running": sum(
                item["state"] == "running" for item in stages.values()
            ),
            "complete": sum(
                item["state"] == "complete" for item in stages.values()
            ),
            "error": sum(item["state"] == "error" for item in stages.values()),
        },
        "errors": errors,
    }


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


class _RegistrySourceChanged(RuntimeError):
    pass


def _validate_registry_source(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("registry source must be an object")
    if value.get("schema") != "interleave-dashboard-registry-source-v1":
        raise ValueError("registry source has unsupported schema")
    registry = _validate_registry(value.get("registry"))
    digest = canonical_json_sha256(registry)
    if value.get("registry_sha256") != digest:
        raise ValueError("registry source SHA-256 mismatch")
    published_at = _parse_iso(value.get("published_at"))
    if published_at is None or published_at.tzinfo is None:
        raise ValueError("registry source timestamp is invalid")
    return dict(value)


def _publish_registry_source(registry: dict[str, Any]) -> dict[str, Any]:
    registry = _validate_registry(registry)
    source = {
        "schema": "interleave-dashboard-registry-source-v1",
        "published_at": datetime.now(timezone.utc).isoformat(),
        "registry_sha256": canonical_json_sha256(registry),
        "registry": registry,
    }
    root = LIVE_FEED_ROOT / "registry"
    version = root / "feeds" / f"{source['registry_sha256']}.json"
    if not version.is_file():
        _atomic_json(version, source)
    _atomic_json(root / "latest.json", source)
    eval_volume.commit()
    dashboard_state["registry_source"] = source
    return source


def _read_registry_source() -> dict[str, Any] | None:
    try:
        source = dashboard_state.get("registry_source")
    except Exception:
        source = None
    if source is not None:
        return _validate_registry_source(source)
    path = LIVE_FEED_ROOT / "registry" / "latest.json"
    if path.is_file():
        return _validate_registry_source(_read_json(path))
    return None


def _persist_live_feed(feed: dict[str, Any]) -> None:
    validated = validate_live_feed(feed)
    source_before = _read_registry_source()
    if (
        source_before is not None
        and source_before["registry_sha256"] != validated["registry_sha256"]
    ):
        raise _RegistrySourceChanged(
            "canonical registry advanced before live-feed publish"
        )
    version_path = (
        LIVE_FEED_ROOT
        / "feeds"
        / f"{validated['payload_sha256']}.json"
    )
    if not version_path.is_file():
        _atomic_json(version_path, validated)
    _atomic_json(LIVE_FEED_POINTER, validated)
    eval_volume.commit()
    source_after = _read_registry_source()
    if (
        source_after is not None
        and source_after["registry_sha256"] != validated["registry_sha256"]
    ):
        raise _RegistrySourceChanged(
            "canonical registry advanced during live-feed publish"
        )
    dashboard_state["control_plane"] = validated


def _read_live_feed() -> tuple[dict[str, Any] | None, str | None]:
    if not LIVE_FEED_POINTER.is_file():
        return None, None
    try:
        feed = validate_live_feed(_read_json(LIVE_FEED_POINTER))
        version_path = (
            LIVE_FEED_ROOT / "feeds" / f"{feed['payload_sha256']}.json"
        )
        if not version_path.is_file():
            raise ValueError("content-addressed live feed object is missing")
        version = validate_live_feed(_read_json(version_path))
        if version["payload_sha256"] != feed["payload_sha256"]:
            raise ValueError("live feed pointer/version digest mismatch")
        return feed, None
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return None, f"live feed rejected: {type(exc).__name__}: {exc}"


def _refresh_live_control_plane(
    registry_override: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    can_publish = True
    if registry_override is not None:
        registry = _validate_registry(registry_override)
    else:
        source = _read_registry_source()
        if source is not None:
            registry = _validate_registry(dict(source["registry"]))
        else:
            previous, feed_error = _read_live_feed()
            if feed_error:
                errors.append(feed_error)
                can_publish = False
            if previous is not None:
                registry = _validate_registry(dict(previous["registry"]))
            else:
                registry = _load_embedded_registry()

    feed = build_live_feed(registry, _collect_pretraining(registry))
    if not can_publish:
        return feed, errors
    for _ in range(3):
        try:
            _persist_live_feed(feed)
            return feed, errors
        except _RegistrySourceChanged:
            source = _read_registry_source()
            if source is None:
                continue
            registry = _validate_registry(dict(source["registry"]))
            feed = build_live_feed(registry, _collect_pretraining(registry))
        except Exception as exc:
            errors.append(
                f"live feed publish: {type(exc).__name__}: {exc}"
            )
            return feed, errors
    errors.append("live feed publish: canonical registry changed repeatedly")
    return feed, errors


def _interleave_training_statuses(
    stages: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, list[int]], list[str]]:
    statuses: dict[str, dict[str, Any]] = {}
    checkpoint_steps: dict[str, list[int]] = {}
    errors: list[str] = []
    now = time.time()

    for stage in stages:
        run_name = stage["run_name"]
        root = TRAINING_MOUNT / INTERLEAVE_RAW_ROOT / run_name
        volume_root = str(root.relative_to(TRAINING_MOUNT))
        entries: list[Any] = []
        try:
            entries = list(training_volume.listdir(volume_root))
        except Exception as exc:
            # Missing roots are normal while an arm is blocked on an upstream
            # phase; retain the registry state rather than calling it an error.
            if root.exists():
                errors.append(f"{run_name} checkpoint listing: {type(exc).__name__}")

        step_entries: list[tuple[int, float]] = []
        for entry in entries:
            match = re.fullmatch(r"iter_0*(\d+)", Path(entry.path).name)
            if match:
                step_entries.append((int(match.group(1)), float(entry.mtime)))
        step_entries.sort()
        steps = [step for step, _ in step_entries]
        checkpoint_steps[run_name] = steps
        latest_step = steps[-1] if steps else 0
        latest_mtime = step_entries[-1][1] if step_entries else None

        validated = False
        validation_reason = "checkpoint unavailable"
        if latest_step:
            validated, validation_reason = _checkpoint_is_valid(root, latest_step)
            if not validated:
                errors.append(
                    f"{run_name} step {latest_step}: {validation_reason}"
                )

        age_seconds = (
            max(0.0, now - latest_mtime) if latest_mtime is not None else None
        )
        if latest_step >= int(stage["target_step"]) and validated:
            state = "complete"
        elif validated and age_seconds is not None and age_seconds <= 45 * 60:
            state = "running"
        elif latest_step:
            state = "stale"
        else:
            state = stage["registry_status"]

        statuses[run_name] = {
            **stage,
            "step": latest_step,
            "target": int(stage["target_step"]),
            "effective_step": (
                int(stage["effective_step_offset"]) + latest_step
            ),
            "progress": (
                min(1.0, latest_step / int(stage["target_step"]))
                if int(stage["target_step"])
                else 0.0
            ),
            "state": state,
            "validated": validated,
            "validation_reason": validation_reason,
            "last_checkpoint_at": _iso_from_epoch(latest_mtime),
            "checkpoint_age_seconds": age_seconds,
            "checkpoint_count": len(steps),
        }

    return statuses, checkpoint_steps, errors


def _interleave_eval_jobs(
    stages: list[dict[str, Any]],
) -> tuple[dict[tuple[str, int], dict[str, Any]], list[str]]:
    run_names = {stage["run_name"] for stage in stages}
    grouped: dict[
        tuple[str, int, str], dict[str, dict[str, Any]]
    ] = {}
    errors: list[str] = []
    entries_by_path: dict[str, Any] = {}

    try:
        for entry in eval_volume.listdir("v1", recursive=True):
            entries_by_path[entry.path] = entry
    except Exception as exc:
        errors.append(f"interleave v1 eval listing: {type(exc).__name__}: {exc}")
    # A dedicated namespace is also accepted so the forthcoming interleave
    # queue can remain isolated without requiring another dashboard revision.
    try:
        for entry in eval_volume.listdir("interleave_v1", recursive=True):
            entries_by_path[entry.path] = entry
    except Exception:
        pass

    for entry in entries_by_path.values():
        parsed = parse_interleave_marker_path(entry.path)
        if parsed is None or parsed["run_name"] not in run_names:
            continue
        if parsed["step"] % INTERLEAVE_EVAL_INTERVAL != 0:
            continue
        key = (parsed["run_name"], parsed["step"], parsed["profile"])
        grouped.setdefault(key, {})[parsed["marker"]] = {
            "path": entry.path,
            "mtime": float(entry.mtime),
            "profile": parsed["profile"],
        }

    candidates: dict[tuple[str, int], list[dict[str, Any]]] = {}
    state_rank = {"success": 4, "failed": 3, "running": 2, "queued": 1}
    for (run_name, step, profile), markers in grouped.items():
        selected = select_terminal_marker(markers)
        if selected is None:
            continue
        state, marker_info = selected
        candidates.setdefault((run_name, step), []).append(
            {
                "state": state,
                "marker": marker_info,
                "profile": profile,
            }
        )

    jobs: dict[tuple[str, int], dict[str, Any]] = {}
    for key, job_candidates in candidates.items():
        selected_job = max(
            job_candidates,
            key=lambda job: (
                state_rank[job["state"]],
                float(job["marker"]["mtime"]),
            ),
        )
        state = selected_job["state"]
        marker_info = selected_job["marker"]
        job: dict[str, Any] = {
            "state": state,
            "profile": selected_job["profile"],
            "result_marker_mtime": marker_info["mtime"],
            "metrics": None,
        }
        if state != "queued":
            try:
                marker = _read_json(EVAL_MOUNT / marker_info["path"])
                job.update(
                    {
                        "started_at": marker.get("started_at"),
                        "finished_at": marker.get("finished_at"),
                        "duration_seconds": marker.get("duration_seconds"),
                        "error": marker.get("error"),
                    }
                )
            except (OSError, json.JSONDecodeError) as exc:
                errors.append(
                    f"{key[0]} step {key[1]} marker: {type(exc).__name__}"
                )
        if state == "success":
            metrics_path = (
                EVAL_MOUNT
                / Path(marker_info["path"]).parent
                / "output"
                / "eval"
                / "generations"
                / "metrics.json"
            )
            try:
                job["metrics"] = summarize_metrics(_read_json(metrics_path))
            except (OSError, json.JSONDecodeError) as exc:
                errors.append(
                    f"{key[0]} step {key[1]} metrics: {type(exc).__name__}"
                )
        jobs[key] = job

    return jobs, errors


def _interleave_statuses(
    live_feed: dict[str, Any],
    feed_errors: list[str] | None = None,
) -> dict[str, Any]:
    validated_feed = validate_live_feed(live_feed)
    registry = _validate_registry(dict(validated_feed["registry"]))
    core_stages = flatten_core_registry(registry)
    exp4_stages = flatten_exp4_rl_registry(registry)
    stages = core_stages + exp4_stages
    training, checkpoint_steps, training_errors = (
        _interleave_training_statuses(stages)
    )
    eval_jobs, eval_errors = _interleave_eval_jobs(stages)
    rows = build_result_rows(stages, checkpoint_steps, eval_jobs)

    for stage in stages:
        run_name = stage["run_name"]
        stage_jobs = [
            job
            for (job_run, _), job in eval_jobs.items()
            if job_run == run_name
        ]
        counts = {
            state: sum(job["state"] == state for job in stage_jobs)
            for state in ("success", "running", "failed", "queued")
        }
        successful_steps = [
            step
            for (job_run, step), job in eval_jobs.items()
            if job_run == run_name and job["state"] == "success"
        ]
        training[run_name]["evaluation"] = {
            "target_checkpoints": int(stage["target_step"])
            // INTERLEAVE_EVAL_INTERVAL,
            "discovered": len(stage_jobs),
            "counts": counts,
            "latest_success_step": (
                max(successful_steps) if successful_steps else 0
            ),
        }

    eval_counts = {
        state: sum(job["state"] == state for job in eval_jobs.values())
        for state in ("success", "running", "failed", "queued")
    }
    v2r3_diagnostics = _validate_v2r3_diagnostic_contract(registry)
    return {
        "schema_version": 2,
        "experiment_version": registry["experiment_version"],
        "model": registry["model_id"],
        "eval_interval": INTERLEAVE_EVAL_INTERVAL,
        "live_sync": {
            "schema": validated_feed["schema"],
            "generated_at": validated_feed["generated_at"],
            "registry_sha256": validated_feed["registry_sha256"],
            "pretraining_sha256": validated_feed["pretraining_sha256"],
            "payload_sha256": validated_feed["payload_sha256"],
            "source": "modal_volume_live_feed",
        },
        "registry": {
            "fixed_pretraining": registry.get("fixed_pretraining", {}),
            "shared_pretraining": registry.get("shared_pretraining", {}),
            "fixed_rl": registry.get("fixed_rl", {}),
            "orchestration": registry.get("orchestration", {}),
            "exp4": registry.get("exp4", {}),
            "artifact_publication": registry.get(
                "artifact_publication", {}
            ),
        },
        "pretraining": validated_feed["pretraining"],
        "v2r3_diagnostics": v2r3_diagnostics,
        "stages": training,
        "rows": rows,
        "aggregate": {
            "stage_count": len(stages),
            "core_stage_count": len(core_stages),
            "exp4_registered_stage_count": len(exp4_stages),
            "target_evaluation_checkpoints": sum(
                int(stage["target_step"]) // INTERLEAVE_EVAL_INTERVAL
                for stage in stages
            ),
            "running_stages": sum(
                item["state"] == "running" for item in training.values()
            ),
            "complete_stages": sum(
                item["state"] == "complete" for item in training.values()
            ),
            "checkpoint_count": sum(
                item["checkpoint_count"] for item in training.values()
            ),
            "evaluation": eval_counts,
            "pretraining": validated_feed["pretraining"].get("aggregate", {}),
        },
        "errors": (
            list(feed_errors or [])
            + list(validated_feed["pretraining"].get("errors", []))
            + training_errors
            + eval_errors
        ),
    }


def _evaluation_statuses(
    previous: dict[str, Any] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    per_run: dict[str, dict[str, Any]] = {}
    all_durations: list[float] = []
    recent_results: list[dict[str, Any]] = []
    marker_pattern = re.compile(
        r"^v1/(?P<run>[^/]+)/global_step_(?P<step>\d+)/"
        rf"production_{re.escape(PRODUCTION_FINGERPRINT[:12])}/"
        r"(?P<marker>_(?:SUCCESS|FAILED|RUNNING|QUEUED)\.json)$"
    )
    markers_by_job: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    listing_error: str | None = None
    try:
        # A single recursive Volume listing avoids thousands of FUSE exists/glob
        # operations. Queued marker contents are not read because their path is
        # sufficient to establish state.
        for entry in eval_volume.listdir("v1", recursive=True):
            match = marker_pattern.fullmatch(entry.path)
            if not match:
                continue
            marker_name = match.group("marker")
            step = int(match.group("step"))
            if step % EVAL_INTERVAL != 0:
                continue
            markers_by_job.setdefault(
                (match.group("run"), step), {}
            )[marker_name] = {
                "path": entry.path,
                "mtime": float(entry.mtime),
            }
    except Exception as exc:
        listing_error = f"result listing: {type(exc).__name__}: {exc}"

    for run_key, spec in RUNS.items():
        jobs: list[dict[str, Any]] = []
        series: list[dict[str, Any]] = []
        errors: list[str] = [listing_error] if listing_error else []
        cached_by_step = {
            int(record["step"]): record
            for record in (previous or {}).get(run_key, {}).get("series", [])
            if isinstance(record, dict) and "step" in record
        }

        run_jobs = sorted(
            (
                (step, markers)
                for (marker_run, step), markers in markers_by_job.items()
                if marker_run == run_key
            ),
            key=lambda item: item[0],
        )
        for step, markers in run_jobs:
            # Marker precedence matters: queued/running markers are intentionally
            # retained after a terminal marker has been written.
            if "_SUCCESS.json" in markers:
                state = "success"
                marker_info = markers["_SUCCESS.json"]
            elif "_FAILED.json" in markers:
                state = "failed"
                marker_info = markers["_FAILED.json"]
            elif "_RUNNING.json" in markers:
                state = "running"
                marker_info = markers["_RUNNING.json"]
            elif "_QUEUED.json" in markers:
                state = "queued"
                marker_info = markers["_QUEUED.json"]
            else:
                continue
            marker_relative = marker_info["path"]

            cached_record = cached_by_step.get(step) if state == "success" else None
            cache_matches = (
                cached_record is not None
                and cached_record.get("result_marker_mtime") == marker_info["mtime"]
            )
            if cache_matches:
                duration = cached_record.get("duration_seconds")
                jobs.append(
                    {
                        "step": step,
                        "state": state,
                        "queued_at": None,
                        "started_at": cached_record.get("started_at"),
                        "finished_at": cached_record.get("finished_at"),
                        "duration_seconds": duration,
                        "error": None,
                    }
                )
                if isinstance(duration, (int, float)):
                    all_durations.append(float(duration))
                series.append(cached_record)
                recent_results.append(
                    {
                        "run_key": run_key,
                        "short": spec["short"],
                        "color": spec["color"],
                        **cached_record,
                    }
                )
                continue

            marker: dict[str, Any] = {}
            if state != "queued":
                try:
                    marker = _read_json(EVAL_MOUNT / marker_relative)
                except (OSError, json.JSONDecodeError) as exc:
                    errors.append(f"step {step} marker: {type(exc).__name__}")

            job = {
                "step": step,
                "state": state,
                "queued_at": marker.get("queued_at"),
                "started_at": marker.get("started_at"),
                "finished_at": marker.get("finished_at"),
                "duration_seconds": marker.get("duration_seconds"),
                "error": marker.get("error"),
            }
            jobs.append(job)

            if state != "success":
                continue
            if marker.get("profile") != "production":
                continue
            if marker.get("fingerprint") != PRODUCTION_FINGERPRINT:
                errors.append(f"step {step} fingerprint mismatch")
                continue
            if (
                marker.get("actual_rows") != EXPECTED_ROWS
                or marker.get("expected_rows") != EXPECTED_ROWS
            ):
                errors.append(f"step {step} row count mismatch")
                continue

            profile_relative = str(Path(marker_relative).parent)
            metrics_path = (
                EVAL_MOUNT
                / profile_relative
                / "output"
                / "eval"
                / "generations"
                / "metrics.json"
            )
            try:
                metrics = _read_json(metrics_path)
            except (OSError, json.JSONDecodeError) as exc:
                errors.append(f"step {step} metrics: {type(exc).__name__}")
                continue

            if "val-core/test_B1/reward/mean@16" not in metrics:
                errors.append(f"step {step} is not a full n=16 result")
                continue

            benchmarks = {
                benchmark: _metric_record(metrics, benchmark)
                for benchmark in ("B1", "B2", "B3", "B4", "B5")
            }
            means = [
                value["mean"]
                for value in benchmarks.values()
                if isinstance(value["mean"], (int, float))
            ]
            bests = [
                value["best"]
                for value in benchmarks.values()
                if isinstance(value["best"], (int, float))
            ]
            duration = marker.get("duration_seconds")
            if isinstance(duration, (int, float)):
                all_durations.append(float(duration))
            record = {
                "step": step,
                "result_marker_mtime": marker_info["mtime"],
                "started_at": marker.get("started_at"),
                "finished_at": marker.get("finished_at"),
                "duration_seconds": duration,
                "rows": marker.get("actual_rows"),
                "macro_mean": statistics.fmean(means) if means else None,
                "macro_best": statistics.fmean(bests) if bests else None,
                "benchmarks": benchmarks,
            }
            series.append(record)
            recent_results.append(
                {
                    "run_key": run_key,
                    "short": spec["short"],
                    "color": spec["color"],
                    **record,
                }
            )

        jobs.sort(key=lambda item: item["step"])
        series.sort(key=lambda item: item["step"])
        counts = {
            state: sum(job["state"] == state for job in jobs)
            for state in ("success", "running", "failed", "queued")
        }
        running_jobs = [job for job in jobs if job["state"] == "running"]
        successful_steps = [job["step"] for job in jobs if job["state"] == "success"]
        best_record = max(
            series,
            key=lambda record: (
                record["macro_mean"] if record["macro_mean"] is not None else -1
            ),
            default=None,
        )
        per_run[run_key] = {
            "run_key": run_key,
            "short": spec["short"],
            "color": spec["color"],
            "target_checkpoints": int(spec["target"]) // EVAL_INTERVAL,
            "discovered": len(jobs),
            "counts": counts,
            "latest_success_step": max(successful_steps) if successful_steps else 0,
            "running_steps": [job["step"] for job in running_jobs],
            "best_step": best_record["step"] if best_record else None,
            "best_macro_mean": (
                best_record["macro_mean"] if best_record else None
            ),
            "series": series,
            "errors": errors,
        }

    total_target = sum(item["target_checkpoints"] for item in per_run.values())
    totals = {
        "target": total_target,
        "discovered": sum(item["discovered"] for item in per_run.values()),
        "success": sum(item["counts"]["success"] for item in per_run.values()),
        "running": sum(item["counts"]["running"] for item in per_run.values()),
        "failed": sum(item["counts"]["failed"] for item in per_run.values()),
        "queued": sum(item["counts"]["queued"] for item in per_run.values()),
    }
    totals["not_discovered"] = max(0, total_target - totals["discovered"])
    totals["progress"] = totals["success"] / total_target if total_target else 0.0
    totals["median_duration_seconds"] = (
        statistics.median(all_durations) if all_durations else None
    )
    totals["mean_duration_seconds"] = (
        statistics.fmean(all_durations) if all_durations else None
    )
    remaining = max(0, total_target - totals["success"])
    active_workers = totals["running"]
    effective_workers = min(
        EVAL_WORKER_CEILING,
        max(1, active_workers),
    )
    totals["active_workers"] = active_workers
    totals["configured_worker_ceiling"] = EVAL_WORKER_CEILING
    totals["effective_workers_for_eta"] = effective_workers
    totals["estimated_wall_seconds"] = (
        remaining * totals["median_duration_seconds"] / effective_workers
        if totals["median_duration_seconds"]
        else None
    )

    recent_results.sort(
        key=lambda item: item.get("finished_at") or "", reverse=True
    )
    aggregate = {
        "totals": totals,
        "recent_results": recent_results[:40],
        "settings": {
            "benchmarks": ["B1", "B2", "B3", "B4", "B5"],
            "response_length": 2_560,
            "temperature": 1,
            "samples_per_prompt": 16,
            "rows_per_checkpoint": EXPECTED_ROWS,
            "eval_interval": EVAL_INTERVAL,
            "workers": active_workers,
            "active_workers": active_workers,
            "configured_worker_ceiling": EVAL_WORKER_CEILING,
            "gpu": "H200",
            "fingerprint": PRODUCTION_FINGERPRINT,
        },
        "modal_url": (
            "https://modal.com/apps/modal-labs/leon-dev/"
            "ap-mWyIUes7i1599iVW0pJWXG"
        ),
    }
    return per_run, aggregate


def _collect_snapshot(force_hf: bool = False) -> dict[str, Any]:
    errors: list[str] = []
    try:
        training_volume.reload()
    except Exception as exc:
        errors.append(f"training volume refresh: {type(exc).__name__}: {exc}")
    try:
        eval_volume.reload()
    except Exception as exc:
        errors.append(f"evaluation volume refresh: {type(exc).__name__}: {exc}")
    try:
        pretrain_volume.reload()
    except Exception as exc:
        errors.append(f"pretrain volume refresh: {type(exc).__name__}: {exc}")

    training = {
        run_key: _training_status(run_key, spec)
        for run_key, spec in RUNS.items()
    }
    hf = _hf_statuses(force=force_hf)
    previous_snapshot: dict[str, Any] | None = None
    previous_evaluation: dict[str, Any] | None = None
    previous_endpoint_evaluations: dict[str, Any] | None = None
    try:
        previous_snapshot = dashboard_state.get("snapshot")
        previous_fingerprint = (
            previous_snapshot.get("evaluation_aggregate", {})
            .get("settings", {})
            .get("fingerprint")
        )
        if previous_fingerprint == PRODUCTION_FINGERPRINT:
            previous_evaluation = previous_snapshot.get("evaluation")
        previous_endpoint_evaluations = (
            previous_snapshot.get("interleave", {}).get(
                "endpoint_evaluations"
            )
        )
    except Exception:
        previous_snapshot = None
        previous_evaluation = None
        previous_endpoint_evaluations = None
    evaluation, evaluation_aggregate = _evaluation_statuses(previous_evaluation)
    endpoint_evaluations = _cached_endpoint_evaluations(
        previous_endpoint_evaluations
    )
    if endpoint_evaluations is None:
        try:
            endpoint_evaluations = _collect_endpoint_evaluations()
        except Exception as exc:
            endpoint_evaluations = {
                "schema_version": 1,
                "namespace": ENDPOINT_NAMESPACE,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "endpoints": {},
                "aggregate": {
                    "expected": len(ENDPOINT_SUMMARY_SPECS),
                    "complete": 0,
                    "partial": 0,
                    "missing": len(ENDPOINT_SUMMARY_SPECS),
                },
                "errors": [
                    f"endpoint summary: {type(exc).__name__}: {exc}"
                ],
                "warnings": [],
                "cache_status": "live_scan_with_errors",
            }
        endpoint_evaluations = _retain_endpoint_summary_on_listing_failure(
            endpoint_evaluations,
            previous_endpoint_evaluations,
        )
    try:
        live_feed, live_feed_errors = _refresh_live_control_plane()
        interleave = _interleave_statuses(live_feed, live_feed_errors)
    except Exception as exc:
        interleave = {
            "schema_version": 2,
            "experiment_version": "mix10b_sft90k_3072_v1_20260730",
            "model": "interleave_47m_qwen3",
            "eval_interval": INTERLEAVE_EVAL_INTERVAL,
            "pretraining": {
                "schema_version": 1,
                "stages": {},
                "aggregate": {},
                "errors": [],
            },
            "stages": {},
            "rows": [],
            "aggregate": {
                "stage_count": 0,
                "running_stages": 0,
                "complete_stages": 0,
                "checkpoint_count": 0,
                "evaluation": {
                    state: 0
                    for state in ("success", "running", "failed", "queued")
                },
            },
            "errors": [
                f"interleave snapshot: {type(exc).__name__}: {exc}"
            ],
        }
    interleave["endpoint_evaluations"] = endpoint_evaluations
    interleave.setdefault("aggregate", {})["endpoint_evaluations"] = (
        endpoint_evaluations.get("aggregate", {})
    )
    interleave.setdefault("errors", []).extend(
        endpoint_evaluations.get("errors", [])
    )

    for run_key in RUNS:
        training[run_key]["hf"] = hf[run_key]
        training[run_key]["evaluation"] = {
            key: value
            for key, value in evaluation[run_key].items()
            if key != "series"
        }

    total_step = sum(item["step"] for item in training.values())
    total_target = sum(item["target"] for item in training.values())
    state_counts = {
        state: sum(item["state"] == state for item in training.values())
        for state in (
            "running",
            "complete",
            "stale",
            "unknown",
        )
    }
    training_aggregate = {
        "step": total_step,
        "target": total_target,
        "remaining": max(0, total_target - total_step),
        "progress": total_step / total_target if total_target else 0.0,
        "states": state_counts,
        "hf_uploaded": sum(item["count"] for item in hf.values()),
        "hf_expected": sum(
            int(spec["target"]) // CHECKPOINT_UPLOAD_INTERVAL
            for spec in RUNS.values()
        ),
    }

    return {
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "refresh_seconds": 30,
        "training": training,
        "training_aggregate": training_aggregate,
        "evaluation": evaluation,
        "evaluation_aggregate": evaluation_aggregate,
        "interleave": interleave,
        "errors": errors,
    }


def _overlay_live_control_plane(
    snapshot: dict[str, Any],
    live_feed: dict[str, Any],
) -> dict[str, Any]:
    """Overlay a newer validated feed onto a persisted dashboard snapshot."""

    feed = validate_live_feed(live_feed)
    current = snapshot.get("interleave", {}).get("live_sync", {})
    current_at = _parse_iso(current.get("generated_at"))
    feed_at = _parse_iso(feed["generated_at"])
    if current_at is not None and feed_at is not None and feed_at <= current_at:
        return snapshot
    registry = _validate_registry(dict(feed["registry"]))
    value = json.loads(json.dumps(snapshot))
    interleave = value.setdefault("interleave", {})
    interleave["experiment_version"] = registry["experiment_version"]
    interleave["model"] = registry["model_id"]
    interleave["live_sync"] = {
        "schema": feed["schema"],
        "generated_at": feed["generated_at"],
        "registry_sha256": feed["registry_sha256"],
        "pretraining_sha256": feed["pretraining_sha256"],
        "payload_sha256": feed["payload_sha256"],
        "source": "modal_volume_live_feed",
    }
    interleave["registry"] = {
        "fixed_pretraining": registry.get("fixed_pretraining", {}),
        "shared_pretraining": registry.get("shared_pretraining", {}),
        "fixed_rl": registry.get("fixed_rl", {}),
        "orchestration": registry.get("orchestration", {}),
        "exp4": registry.get("exp4", {}),
        "artifact_publication": registry.get(
            "artifact_publication", {}
        ),
    }
    interleave["pretraining"] = feed["pretraining"]
    interleave["v2r3_diagnostics"] = (
        _validate_v2r3_diagnostic_contract(registry)
    )
    aggregate = interleave.setdefault("aggregate", {})
    aggregate["pretraining"] = feed["pretraining"].get("aggregate", {})
    previous_errors = [
        item
        for item in interleave.get("errors", [])
        if not str(item).startswith(
            tuple(
                f"{stage_id}:"
                for stage_id in feed["pretraining"].get("stages", {})
            )
        )
    ]
    interleave["errors"] = previous_errors + list(
        feed["pretraining"].get("errors", [])
    )
    return value


def get_snapshot(force: bool = False) -> dict[str, Any]:
    global _snapshot, _snapshot_at
    now = time.time()
    with _snapshot_lock:
        if (
            not force
            and _snapshot is not None
            and now - _snapshot_at < _SNAPSHOT_TTL_SECONDS
        ):
            return _snapshot
        try:
            value = _collect_snapshot(force_hf=force)
        except Exception as exc:
            if _snapshot is not None:
                stale = dict(_snapshot)
                stale["stale"] = True
                stale["snapshot_error"] = f"{type(exc).__name__}: {exc}"
                return stale
            raise
        _snapshot = value
        _snapshot_at = now
        return value


@app.function(
    image=dashboard_image,
    timeout=5 * 60,
    max_containers=1,
    volumes={
        str(EVAL_MOUNT): eval_volume,
        str(PRETRAIN_MOUNT): pretrain_volume,
    },
)
def publish_live_control_plane(registry_json: str) -> dict[str, Any]:
    """Publish the canonical local registry with fresh pretrain telemetry."""

    if not isinstance(registry_json, str) or not registry_json.strip():
        raise ValueError("registry_json must be a non-empty JSON string")
    if len(registry_json.encode("utf-8")) > 5 * 1024 * 1024:
        raise ValueError("registry_json exceeds the 5 MiB control-plane limit")
    registry = json.loads(registry_json)
    _validate_registry(registry)
    eval_volume.reload()
    pretrain_volume.reload()
    source = _publish_registry_source(registry)
    feed, errors = _refresh_live_control_plane(registry)
    if errors:
        raise RuntimeError("; ".join(errors))
    return {
        "schema": feed["schema"],
        "generated_at": feed["generated_at"],
        "registry_sha256": feed["registry_sha256"],
        "registry_published_at": source["published_at"],
        "pretraining_sha256": feed["pretraining_sha256"],
        "payload_sha256": feed["payload_sha256"],
        "pretrain_stage_count": len(feed["pretraining"].get("stages", {})),
    }


@app.function(
    image=dashboard_image,
    timeout=5 * 60,
    max_containers=1,
    schedule=modal.Period(minutes=1),
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={
        str(TRAINING_MOUNT): training_volume,
        str(EVAL_MOUNT): eval_volume,
        str(PRETRAIN_MOUNT): pretrain_volume,
    },
)
def refresh_snapshot() -> dict[str, Any]:
    """Refresh the persistent snapshot used for low-latency web responses."""
    value = _collect_snapshot(force_hf=False)
    dashboard_state["snapshot"] = value
    return {
        "generated_at": value["generated_at"],
        "training_step": value["training_aggregate"]["step"],
        "evaluated": value["evaluation_aggregate"]["totals"]["success"],
        "interleave_checkpoints": value["interleave"]["aggregate"][
            "checkpoint_count"
        ],
        "interleave_pretrain": value["interleave"]["aggregate"].get(
            "pretraining", {}
        ),
        "interleave_endpoints": value["interleave"]["aggregate"].get(
            "endpoint_evaluations", {}
        ),
        "live_feed": value["interleave"].get("live_sync", {}),
    }


DASHBOARD_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="Live training and evaluation dashboard for Chess RL r6 experiments.">
  <title>Chess RL · Live Control Room</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #07110f;
      --panel: #0d1a17;
      --panel-2: #10221e;
      --edge: rgba(184, 231, 216, .13);
      --text: #ecf7f3;
      --muted: #8ca59d;
      --green: #57d6b5;
      --blue: #72a7ff;
      --purple: #c18cff;
      --orange: #ffb45c;
      --red: #ff7185;
      --yellow: #f6d56a;
      --shadow: 0 18px 50px rgba(0, 0, 0, .22);
    }
    * { box-sizing: border-box; }
    html { background: var(--bg); }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--text);
      background:
        radial-gradient(circle at 85% -10%, rgba(87, 214, 181, .13), transparent 32rem),
        radial-gradient(circle at 5% 35%, rgba(114, 167, 255, .08), transparent 28rem),
        linear-gradient(180deg, #081310 0%, #07110f 100%);
      font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: -.012em;
    }
    a { color: inherit; }
    button, select { font: inherit; }
    .shell { width: min(1480px, calc(100% - 40px)); margin: 0 auto; padding: 34px 0 70px; }
    .topbar { display: flex; justify-content: space-between; gap: 24px; align-items: flex-start; margin-bottom: 28px; }
    .eyebrow {
      display: flex; align-items: center; gap: 9px; color: var(--green);
      text-transform: uppercase; letter-spacing: .13em; font-size: 11px; font-weight: 750;
    }
    .pulse { width: 8px; height: 8px; border-radius: 50%; background: var(--green); box-shadow: 0 0 0 0 rgba(87,214,181,.6); animation: pulse 2s infinite; }
    @keyframes pulse { 70% { box-shadow: 0 0 0 8px rgba(87,214,181,0); } 100% { box-shadow: 0 0 0 0 rgba(87,214,181,0); } }
    h1 { margin: 8px 0 5px; font-size: clamp(31px, 4vw, 52px); line-height: 1; letter-spacing: -.05em; font-weight: 690; }
    .subtitle { color: var(--muted); font-size: 14px; max-width: 660px; line-height: 1.55; }
    .refresh-wrap { display: flex; gap: 10px; align-items: center; color: var(--muted); font-size: 12px; padding-top: 4px; }
    .refresh {
      border: 1px solid var(--edge); color: var(--text); background: rgba(255,255,255,.035);
      border-radius: 10px; padding: 9px 13px; cursor: pointer; transition: .18s ease;
    }
    .refresh:hover { border-color: rgba(87,214,181,.45); background: rgba(87,214,181,.08); }
    .refresh:disabled { opacity: .5; cursor: wait; }
    .summary-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 28px; }
    .summary-card, .card {
      background: linear-gradient(145deg, rgba(17,35,30,.94), rgba(10,24,20,.94));
      border: 1px solid var(--edge); border-radius: 16px; box-shadow: var(--shadow);
    }
    .summary-card { padding: 17px 18px 16px; min-height: 112px; position: relative; overflow: hidden; }
    .summary-card:after { content: ""; position: absolute; inset: auto -20px -45px auto; width: 110px; height: 110px; border-radius: 50%; background: var(--accent, var(--green)); opacity: .06; filter: blur(2px); }
    .summary-label { color: var(--muted); text-transform: uppercase; letter-spacing: .1em; font-size: 10px; font-weight: 720; }
    .summary-value { font-variant-numeric: tabular-nums; font-size: 29px; letter-spacing: -.045em; font-weight: 660; margin: 9px 0 4px; }
    .summary-note { color: var(--muted); font-size: 12px; }
    .section-head { display: flex; justify-content: space-between; align-items: end; gap: 15px; margin: 26px 0 12px; }
    .section-head h2 { margin: 0; font-size: 19px; letter-spacing: -.025em; font-weight: 640; }
    .section-head p { margin: 4px 0 0; color: var(--muted); font-size: 12px; }
    .train-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
    .train-card { padding: 17px; position: relative; overflow: hidden; }
    .train-card:before { content: ""; position: absolute; left: 0; top: 0; width: 100%; height: 2px; background: var(--run); opacity: .9; }
    .card-title-row { display: flex; justify-content: space-between; gap: 12px; align-items: center; }
    .run-title { font-size: 15px; font-weight: 640; }
    .state { display: inline-flex; align-items: center; gap: 6px; color: var(--muted); font-size: 10px; text-transform: uppercase; letter-spacing: .07em; }
    .state-dot { width: 6px; height: 6px; border-radius: 50%; background: currentColor; }
    .state.running, .state.complete { color: var(--green); }
    .state.submitted, .state.finishing { color: var(--blue); }
    .state.recent, .state.stale, .state.partial { color: var(--yellow); }
    .state.inactive, .state.failed, .state.error { color: var(--red); }
    .step-row { display: flex; align-items: baseline; gap: 6px; margin: 22px 0 9px; font-variant-numeric: tabular-nums; }
    .step { font-size: 26px; letter-spacing: -.04em; font-weight: 650; }
    .target { color: var(--muted); font-size: 12px; }
    .progress-track { height: 7px; width: 100%; background: rgba(255,255,255,.06); border-radius: 10px; overflow: hidden; }
    .progress-fill { height: 100%; width: 0; border-radius: inherit; background: var(--run); transition: width .6s cubic-bezier(.2,.8,.2,1); }
    .card-meta { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-top: 16px; }
    .meta-cell { min-width: 0; }
    .meta-value { font-size: 13px; font-weight: 620; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; font-variant-numeric: tabular-nums; }
    .meta-label { color: var(--muted); font-size: 9px; text-transform: uppercase; letter-spacing: .08em; margin-top: 3px; }
    .card-links { display: flex; gap: 13px; margin-top: 17px; padding-top: 13px; border-top: 1px solid var(--edge); }
    .card-links a { color: var(--muted); font-size: 11px; text-decoration: none; }
    .card-links a:hover { color: var(--run); }
    .eval-layout { display: grid; grid-template-columns: minmax(0, 1.7fr) minmax(320px, .72fr); gap: 12px; }
    .chart-card { padding: 17px 17px 12px; min-height: 465px; }
    .chart-toolbar { display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 12px; margin-bottom: 15px; }
    .segmented { display: inline-flex; padding: 3px; border: 1px solid var(--edge); border-radius: 10px; background: rgba(0,0,0,.12); gap: 2px; }
    .segmented button {
      border: 0; border-radius: 7px; padding: 7px 10px; color: var(--muted);
      background: transparent; cursor: pointer; font-size: 11px; transition: .15s ease;
    }
    .segmented button.active { color: var(--text); background: rgba(255,255,255,.08); }
    select {
      border: 1px solid var(--edge); color: var(--text); background: #10201c;
      border-radius: 9px; padding: 8px 31px 8px 10px; font-size: 11px; outline: none;
    }
    .chart-wrap { position: relative; height: 350px; width: 100%; }
    #chart { width: 100%; height: 100%; overflow: visible; }
    .axis-label { fill: #789189; font-size: 10px; font-variant-numeric: tabular-nums; }
    .grid-line { stroke: rgba(184,231,216,.09); stroke-width: 1; }
    .chart-line { fill: none; stroke-width: 2.25; vector-effect: non-scaling-stroke; }
    .chart-point { stroke: #0d1a17; stroke-width: 1.5; cursor: crosshair; vector-effect: non-scaling-stroke; }
    .legend { display: flex; flex-wrap: wrap; gap: 15px; color: var(--muted); font-size: 11px; padding: 4px 4px 0 49px; }
    .legend-item { display: inline-flex; gap: 6px; align-items: center; }
    .legend-dot { width: 7px; height: 7px; border-radius: 50%; }
    .tooltip {
      position: absolute; display: none; min-width: 170px; padding: 10px 11px;
      border: 1px solid var(--edge); background: rgba(7,17,15,.96); border-radius: 10px;
      box-shadow: 0 12px 30px rgba(0,0,0,.35); pointer-events: none; font-size: 11px; z-index: 5;
    }
    .tooltip-title { font-weight: 650; margin-bottom: 7px; }
    .tooltip-row { display: flex; justify-content: space-between; gap: 18px; color: var(--muted); padding: 2px 0; }
    .tooltip-row span:last-child { color: var(--text); font-variant-numeric: tabular-nums; }
    .queue-card { padding: 17px; }
    .queue-total { font-size: 39px; letter-spacing: -.055em; font-weight: 660; margin: 17px 0 4px; font-variant-numeric: tabular-nums; }
    .queue-sub { color: var(--muted); font-size: 12px; }
    .queue-bar { display: flex; height: 9px; margin: 20px 0; overflow: hidden; border-radius: 9px; background: rgba(255,255,255,.06); }
    .queue-segment { height: 100%; transition: width .5s ease; }
    .queue-list { display: grid; gap: 11px; }
    .queue-row { display: flex; justify-content: space-between; align-items: center; font-size: 12px; }
    .queue-name { color: var(--muted); display: inline-flex; align-items: center; gap: 8px; }
    .queue-swatch { width: 7px; height: 7px; border-radius: 50%; }
    .queue-count { font-variant-numeric: tabular-nums; font-weight: 620; }
    .queue-foot { margin-top: 20px; padding-top: 14px; border-top: 1px solid var(--edge); color: var(--muted); font-size: 11px; line-height: 1.55; }
    .run-eval-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
    .run-eval { padding: 15px 16px; }
    .eval-stat-row { display: flex; justify-content: space-between; align-items: baseline; gap: 10px; margin-top: 14px; }
    .eval-stat-big { font-size: 22px; font-weight: 650; font-variant-numeric: tabular-nums; }
    .eval-stat-note { color: var(--muted); font-size: 10px; }
    .mini-bar { height: 5px; background: rgba(255,255,255,.06); border-radius: 6px; overflow: hidden; margin-top: 11px; }
    .mini-bar > div { height: 100%; background: var(--run); border-radius: inherit; }
    .table-card { padding: 5px 16px 10px; overflow: hidden; }
    .table-scroll { overflow-x: auto; }
    table { width: 100%; border-collapse: collapse; min-width: 850px; font-size: 11px; }
    th { color: var(--muted); text-align: right; text-transform: uppercase; letter-spacing: .08em; font-size: 9px; font-weight: 650; padding: 12px 10px; border-bottom: 1px solid var(--edge); }
    th:first-child, td:first-child { text-align: left; }
    td { text-align: right; padding: 11px 10px; border-bottom: 1px solid rgba(184,231,216,.07); font-variant-numeric: tabular-nums; }
    tr:last-child td { border-bottom: 0; }
    .run-chip { display: inline-flex; gap: 7px; align-items: center; font-weight: 590; }
    .run-chip i { width: 7px; height: 7px; border-radius: 50%; }
    .results-card { min-height: 465px; }
    .results-card table { min-width: 620px; font-size: 12px; }
    .results-card th:nth-child(2), .results-card td:nth-child(2) { width: 110px; }
    .results-card th:nth-child(3), .results-card td:nth-child(3) { width: 190px; }
    .model-name { overflow-wrap: anywhere; }
    .empty { height: 310px; display: grid; place-items: center; color: var(--muted); font-size: 13px; }
    .error-banner { display: none; margin-bottom: 12px; border: 1px solid rgba(255,113,133,.35); background: rgba(255,113,133,.08); color: #ffc2ca; border-radius: 12px; padding: 11px 14px; font-size: 12px; }
    .footer { color: var(--muted); display: flex; justify-content: space-between; gap: 20px; font-size: 10px; margin-top: 24px; line-height: 1.55; }
    .skeleton { animation: shimmer 1.3s infinite linear; background: linear-gradient(90deg, rgba(255,255,255,.04) 25%, rgba(255,255,255,.09) 50%, rgba(255,255,255,.04) 75%); background-size: 200% 100%; color: transparent; border-radius: 5px; }
    @keyframes shimmer { to { background-position: -200% 0; } }
    @media (max-width: 1100px) {
      .summary-grid, .train-grid, .run-eval-grid { grid-template-columns: repeat(2, 1fr); }
      .eval-layout { grid-template-columns: 1fr; }
    }
    @media (max-width: 650px) {
      .shell { width: min(100% - 24px, 1480px); padding-top: 22px; }
      .topbar { flex-direction: column; }
      .summary-grid, .train-grid, .run-eval-grid { grid-template-columns: 1fr; }
      .summary-card { min-height: 96px; }
      .chart-card { padding-left: 10px; padding-right: 10px; }
      .chart-toolbar { align-items: flex-start; }
      #run-tabs { display: grid; grid-template-columns: repeat(2, 1fr); width: 100%; }
      .legend { padding-left: 44px; gap: 10px; }
      .footer { flex-direction: column; }
    }
  </style>
</head>
<body>
  <main class="shell">
    <header class="topbar">
      <div>
        <div class="eyebrow"><span class="pulse"></span> live research telemetry</div>
        <h1>Chess RL control room</h1>
        <div class="subtitle">Durable Modal training checkpoints and full B1–B5 evaluation results for the interleaved 47.245M experiment and four r6 policy runs.</div>
      </div>
      <div class="refresh-wrap">
        <span id="updated">connecting…</span>
        <button id="refresh" class="refresh">Refresh now</button>
      </div>
    </header>

    <div id="error" class="error-banner"></div>
    <section id="summary" class="summary-grid">
      <article class="summary-card"><div class="summary-label">Training</div><div class="summary-value skeleton">loading</div></article>
      <article class="summary-card"><div class="summary-label">Trainers live</div><div class="summary-value skeleton">loading</div></article>
      <article class="summary-card"><div class="summary-label">Evaluated</div><div class="summary-value skeleton">loading</div></article>
      <article class="summary-card"><div class="summary-label">HF checkpoints</div><div class="summary-value skeleton">loading</div></article>
    </section>

    <div class="section-head">
      <div><h2>Interleaved 47.245M experiment</h2><p>Registry-aware E1–E3 status. E1 RL2 reports effective RL step = phase step + 1,500.</p></div>
      <a class="refresh" href="/api/interleave-results.csv" download>Download interleave CSV</a>
    </div>
    <section class="card table-card results-card">
      <div class="chart-toolbar" style="padding:12px 0 0">
        <div>
          <div class="run-title">Training and evaluation phases</div>
          <div id="interleave-note" class="eval-stat-note" style="margin-top:4px">Loading interleave registry…</div>
          <div id="interleave-contract" class="eval-stat-note" style="margin-top:4px"></div>
        </div>
      </div>
      <div class="run-title" style="padding:8px 0 3px">Pretraining stages</div>
      <div class="eval-stat-note" style="margin-bottom:5px">Strict local metrics: step, loss, LR, token throughput, ETA, freshness, and runtime identity.</div>
      <div class="table-scroll"><table>
        <thead><tr><th>Stage</th><th>Step</th><th>State</th><th>Loss</th><th>LR</th><th>Tok/s</th><th>ETA</th><th>Last update</th><th>Runtime provenance</th></tr></thead>
        <tbody id="interleave-pretraining"></tbody>
      </table></div>
      <div class="run-title" style="padding:19px 0 3px">Pretraining endpoint evaluations</div>
      <div id="interleave-endpoint-note" class="eval-stat-note" style="margin-bottom:5px">Loading immutable endpoint results…</div>
      <div class="table-scroll"><table style="min-width:1450px">
        <thead><tr><th>Endpoint</th><th>Checkpoint SHA</th><th>PT loss</th><th>Pass@1</th><th>Avg reward</th><th>B1 P@1 / R</th><th>B2 P@1 / R</th><th>B3 P@1 / R</th><th>B4 P@1 / R</th><th>B5 P@1 / R</th><th>B3–B4</th><th>Result hashes</th><th>State</th></tr></thead>
        <tbody id="interleave-endpoints"></tbody>
      </table></div>
      <div class="run-title" style="padding:19px 0 3px">Artifact publication</div>
      <div id="interleave-artifact-note" class="eval-stat-note" style="margin-bottom:5px">Loading immutable Hugging Face publication ledger…</div>
      <div class="table-scroll"><table style="min-width:1250px">
        <thead><tr><th>Artifact</th><th>Repository</th><th>Published</th><th>HEAD / manifest</th><th>Modal call</th><th>State</th><th>Audit note</th></tr></thead>
        <tbody id="interleave-artifacts"></tbody>
      </table></div>
      <div class="run-title" style="padding:19px 0 3px">v2r3 trajectory-weight diagnostics</div>
      <div id="interleave-v2r3-note" class="eval-stat-note" style="margin-bottom:5px">Loading diagnostic-only contract…</div>
      <div class="table-scroll"><table style="min-width:1250px">
        <thead><tr><th>SFT loss weight</th><th>Training call</th><th>Training state</th><th>Authenticated progress</th><th>Terminal</th><th>Snapshot inventory</th><th>Authenticated bytes</th></tr></thead>
        <tbody id="interleave-v2r3-trajectories"></tbody>
      </table></div>
      <div class="run-title" style="padding:19px 0 3px">v2r3 snapshot rollout audits</div>
      <div class="eval-stat-note" style="margin-bottom:5px">Diagnostic positives and rates below are protocol/reward probes only; they are never official Pass@1 or benchmark results.</div>
      <div class="table-scroll"><table style="min-width:1650px">
        <thead><tr><th>Weight / snapshot</th><th>Rollout call</th><th>State</th><th>Snapshot identity</th><th>Protocol rows / groups</th><th>Diagnostic positives / rate</th><th>Variance groups / rate</th><th>Responses</th><th>Snapshot inventory</th></tr></thead>
        <tbody id="interleave-v2r3-rollouts"></tbody>
      </table></div>
      <div id="interleave-v2r3-report" class="eval-stat-note" style="margin:7px 0 2px">Final immutable diagnostic report: loading…</div>
      <div class="run-title" style="padding:19px 0 3px">RL phases</div>
      <div class="table-scroll"><table>
        <thead><tr><th>Model / arm</th><th>Filter</th><th>Phase</th><th>Latest phase step</th><th>Effective RL step</th><th>Training</th><th>Eval completed</th></tr></thead>
        <tbody id="interleave-stages"></tbody>
      </table></div>
      <div class="run-title" style="padding:19px 0 3px">Exp 4 positive-rollout pipelines</div>
      <div id="interleave-exp4-note" class="eval-stat-note" style="margin-bottom:5px"></div>
      <div class="table-scroll"><table>
        <thead><tr><th>Arm</th><th>Filter</th><th>Method</th><th>Pipeline</th><th>Extract</th><th>Validate</th><th>Pretrain</th><th>Final RL</th></tr></thead>
        <tbody id="interleave-exp4"></tbody>
      </table></div>
    </section>
    <section class="card table-card results-card" style="margin-top:12px">
      <div class="chart-toolbar" style="padding:12px 0 0">
        <div>
          <div class="run-title">Interleave checkpoint results</div>
          <div class="eval-stat-note" style="margin-top:4px">Pass@1 and reward are B1–B5 macros; B3–B4 is reported separately.</div>
        </div>
      </div>
      <div class="table-scroll"><table>
        <thead><tr><th>Model / arm</th><th>Filter</th><th>Phase</th><th>Phase step</th><th>Effective RL step</th><th>Pass@1</th><th>Avg reward</th><th>B3–B4 avg</th><th>Training</th><th>Eval</th></tr></thead>
        <tbody id="interleave-results"></tbody>
      </table></div>
    </section>

    <div class="section-head">
      <div><h2>Training</h2><p>Validated durable checkpoints; “running” means the checkpoint heartbeat is less than 45 minutes old.</p></div>
    </div>
    <section id="training" class="train-grid">
      <article class="card train-card"><div class="run-title skeleton">Loading run</div><div class="step-row"><span class="step skeleton">0000</span></div></article>
      <article class="card train-card"><div class="run-title skeleton">Loading run</div><div class="step-row"><span class="step skeleton">0000</span></div></article>
      <article class="card train-card"><div class="run-title skeleton">Loading run</div><div class="step-row"><span class="step skeleton">0000</span></div></article>
      <article class="card train-card"><div class="run-title skeleton">Loading run</div><div class="step-row"><span class="step skeleton">0000</span></div></article>
    </section>

    <div class="section-head">
      <div><h2>Evaluation</h2><p>Every 40 training steps: B1–B5, 16 samples per prompt, 23,680 trajectories per checkpoint.</p></div>
      <a id="eval-modal-link" class="refresh" target="_blank" rel="noreferrer">Open eval app ↗</a>
    </div>
    <section class="eval-layout">
      <article class="card table-card results-card">
        <div class="chart-toolbar" style="padding:12px 0 0">
          <div>
            <div class="run-title">Completed checkpoint results</div>
            <div id="results-note" class="eval-stat-note" style="margin-top:4px">Loading results…</div>
          </div>
          <a class="refresh" href="/api/results.csv" download>Download CSV</a>
        </div>
        <div class="table-scroll"><table>
          <thead><tr><th>Model</th><th>Step</th><th>Pass@1 avg reward</th><th>B3–B4 avg reward</th></tr></thead>
          <tbody id="results"></tbody>
        </table></div>
      </article>
      <article id="queue" class="card queue-card"></article>
    </section>

    <div class="section-head">
      <div><h2>Evaluation by run</h2><p>Terminal state takes precedence over retained running and queued markers.</p></div>
    </div>
    <section id="eval-runs" class="run-eval-grid"></section>

    <footer class="footer">
      <span>Pass@1 is the B1–B5 macro; B3–B4 avg is the mean of canonical B3 and B4 reward mean@16 · Auto-refreshes every 30 seconds</span>
      <span id="fingerprint"></span>
    </footer>
  </main>

  <script>
    const RUN_ORDER = [
      "C6p5e18_32m_alpha0.200_beta0.013",
      "C6p5e18_32m_alpha0.400_beta0.013",
      "C6p5e18_410m_alpha0.750_beta0.148",
      "C6p5e18_410m_alpha1.000_beta0.148"
    ];
    const BENCHMARKS = ["B1", "B2", "B3", "B4", "B5"];
    const B_COLORS = {B1:"#57d6b5", B2:"#72a7ff", B3:"#c18cff", B4:"#ffb45c", B5:"#ff7185"};
    let snapshot = null;
    let busy = false;

    const esc = value => String(value ?? "").replace(/[&<>"']/g, char => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[char]));
    const pct = value => `${(100 * (Number(value) || 0)).toFixed(1)}%`;
    const score = value => value == null ? "—" : Number(value).toFixed(3);
    const reward = value => value == null ? "—" : Number(value).toFixed(6);
    const diagnosticRate = value => value == null ? "—" : Number(value).toFixed(6);
    const loss = value => value == null ? "—" : Number(value).toFixed(6);
    const sci = value => value == null ? "—" : Number(value).toExponential(2);
    const int = value => Number(value || 0).toLocaleString();
    const storage = value => {
      if (value == null) return "—";
      const bytes = Number(value);
      if (bytes < 1024) return `${int(bytes)} B`;
      if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KiB`;
      if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MiB`;
      return `${(bytes / 1024 ** 3).toFixed(2)} GiB`;
    };
    const duration = seconds => {
      if (seconds == null) return "—";
      seconds = Math.max(0, Number(seconds));
      if (seconds < 90) return `${Math.round(seconds)}s`;
      const h = Math.floor(seconds / 3600);
      const m = Math.round((seconds % 3600) / 60);
      return h ? `${h}h ${m}m` : `${m}m`;
    };
    const ago = iso => {
      if (!iso) return "unknown";
      const seconds = Math.max(0, (Date.now() - new Date(iso).getTime()) / 1000);
      if (seconds < 60) return "just now";
      if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
      if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
      return `${Math.floor(seconds / 86400)}d ago`;
    };
    const shortDate = iso => {
      if (!iso) return "—";
      return new Intl.DateTimeFormat(undefined, {month:"short", day:"numeric", hour:"numeric", minute:"2-digit"}).format(new Date(iso));
    };

    function renderSummary() {
      const t = snapshot.training_aggregate;
      const e = snapshot.evaluation_aggregate.totals;
      const cards = [
        ["Training", pct(t.progress), `${int(t.step)} / ${int(t.target)} durable steps`, "#57d6b5"],
        ["Fresh trainers", `${t.states.running} / 4`, `${t.states.running} checkpointing · ${t.states.complete} complete`, "#72a7ff"],
        ["Evaluated", `${int(e.success)} / ${int(e.target)}`, `${e.running} running · ${e.failed} failed`, "#c18cff"],
        ["HF checkpoints", `${int(t.hf_uploaded)} / ${int(t.hf_expected)}`, `${int(e.discovered)} discovered by eval`, "#ffb45c"],
      ];
      document.querySelector("#summary").innerHTML = cards.map(([label,value,note,color]) => `
        <article class="summary-card" style="--accent:${color}">
          <div class="summary-label">${label}</div>
          <div class="summary-value">${value}</div>
          <div class="summary-note">${note}</div>
        </article>`).join("");
    }

    function renderInterleave() {
      const interleave = snapshot.interleave || {stages:{}, rows:[], aggregate:{evaluation:{}}};
      const stages = Object.values(interleave.stages || {});
      const rows = interleave.rows || [];
      const agg = interleave.aggregate || {};
      const evalAgg = agg.evaluation || {};
      const registry = interleave.registry || {};
      const pretraining = interleave.pretraining || {stages:{}, aggregate:{}};
      const pretrainStages = Object.values(pretraining.stages || {});
      const endpointEvaluations = interleave.endpoint_evaluations || {endpoints:{}, aggregate:{}};
      const endpointRows = Object.values(endpointEvaluations.endpoints || {});
      const endpointAggregate = endpointEvaluations.aggregate || {};
      const v2r3 = interleave.v2r3_diagnostics || {present:false, trajectories:[], rollouts:[], aggregate:{}, final_report:{}};
      const liveSync = interleave.live_sync || {};
      const exp4 = registry.exp4 || {};
      const exp4Arms = exp4.arms || [];
      const exp4Calls = ((exp4.orchestration || {}).calls || {});
      const shared = registry.shared_pretraining || {};
      const calls = ((registry.orchestration || {}).calls || {});
      const latestCall = key => {
        const records = calls[key] || [];
        return records.length ? records[records.length - 1] : {};
      };
      const p1Call = latestCall("p1");
      const exp2Call = latestCall("exp2");
      const contract = ((registry.fixed_pretraining || {}).production_contract || {});
      document.querySelector("#interleave-note").textContent =
        `${int(agg.running_stages)} RL running · ${int(agg.complete_stages)} RL complete · ${int(agg.checkpoint_count)} durable checkpoints · ${int(evalAgg.success)} evaluated · live registry ${String(liveSync.registry_sha256 || "—").slice(0,12)} (${ago(liveSync.generated_at)})`;
      document.querySelector("#interleave-contract").textContent =
        `P1 ${shared.p1?.status || "unknown"} ${p1Call.call_id || ""} · Exp2 ${shared.exp2_monolithic?.status || "unknown"} ${exp2Call.call_id || ""} · contract ${contract.attention_backend || "—"} / compile ${contract.torch_compile_mode || "—"} / ${int(contract.token_positions_per_second)} tok/s`;
      document.querySelector("#interleave-pretraining").innerHTML = pretrainStages.length ? pretrainStages.map(r => {
        const runtime = r.runtime_provenance || {};
        const runtimeText = runtime.attention_backend
          ? `${runtime.attention_backend} · compile ${runtime.torch_compile_mode} · torch ${runtime.torch_version} · transformers ${runtime.transformers_version} · workers ${int(runtime.data_num_workers)}`
          : "—";
        const path = r.output_path ? `<div class="eval-stat-note" title="${esc(r.output_path)}">${esc(r.output_path.split("/").slice(-2).join("/"))}</div>` : "";
        return `<tr>
          <td><span class="model-name">${esc(r.label)}</span>${path}<div class="eval-stat-note">${esc(r.call_id || "")}</div></td>
          <td>${int(r.step)} / ${int(r.target)}<div class="eval-stat-note">${pct(r.progress)}</div></td>
          <td><span class="state ${esc(r.state)}"><i class="state-dot"></i>${esc(r.state)}</span></td>
          <td>${loss(r.loss)}</td>
          <td>${sci(r.lr)}</td>
          <td>${r.tokens_per_second == null ? "—" : int(Math.round(r.tokens_per_second))}</td>
          <td>${r.state === "complete" ? "done" : duration(r.eta_seconds)}</td>
          <td title="${esc(r.last_update_at || "")}">${ago(r.last_update_at)}</td>
          <td><span class="eval-stat-note" title="${esc(runtimeText)}">${esc(runtimeText)}</span></td>
        </tr>`;
      }).join("") : `<tr><td colspan="9" style="text-align:center;color:var(--muted);padding:28px">No pretraining telemetry is available.</td></tr>`;
      document.querySelector("#interleave-endpoint-note").textContent =
        `${int(endpointAggregate.complete)} / ${int(endpointAggregate.expected)} immutable endpoints complete · loss and chess result hashes are content-verified`;
      const endpointBenchmark = (chess, name) => {
        const item = ((chess || {}).metrics || {}).benchmarks?.[name] || {};
        return `${reward(item.pass_at_1)} / ${reward(item.avg_reward)}`;
      };
      document.querySelector("#interleave-endpoints").innerHTML = endpointRows.length ? endpointRows.map(r => {
        const lossResult = r.loss || {};
        const lossMetrics = lossResult.metrics || {};
        const chessResult = r.chess || {};
        const chessMetrics = chessResult.metrics || {};
        const lossHash = (r.result_hashes || {}).losses;
        const chessHash = (r.result_hashes || {}).chess;
        const checkpoint = r.checkpoint_sha256;
        const hashes = [
          lossHash ? `loss ${String(lossHash).slice(0,12)}` : null,
          chessHash ? `chess ${String(chessHash).slice(0,12)}` : null,
        ].filter(Boolean).join(" · ") || "—";
        const hashTitle = [
          lossHash ? `loss ${lossHash}` : null,
          chessHash ? `chess ${chessHash}` : null,
        ].filter(Boolean).join(" · ");
        return `<tr>
          <td><span class="model-name">${esc(r.label || r.endpoint_id)}</span><div class="eval-stat-note">${esc(r.endpoint_id)}</div></td>
          <td title="${esc(checkpoint || "")}">${checkpoint ? esc(String(checkpoint).slice(0,12)) : "—"}</td>
          <td>${loss(lossMetrics.heldout_pretrain_loss)}</td>
          <td>${reward(chessMetrics.pass_at_1)}</td>
          <td>${reward(chessMetrics.avg_reward)}</td>
          <td>${endpointBenchmark(chessResult, "B1")}</td>
          <td>${endpointBenchmark(chessResult, "B2")}</td>
          <td>${endpointBenchmark(chessResult, "B3")}</td>
          <td>${endpointBenchmark(chessResult, "B4")}</td>
          <td>${endpointBenchmark(chessResult, "B5")}</td>
          <td>${reward(chessMetrics.b3_b4_avg)}</td>
          <td><span class="eval-stat-note" title="${esc(hashTitle)}">${esc(hashes)}</span></td>
          <td><span class="state ${esc(r.state)}"><i class="state-dot"></i>${esc(r.state)}</span></td>
        </tr>`;
      }).join("") : `<tr><td colspan="13" style="text-align:center;color:var(--muted);padding:20px">No endpoint evaluation summaries are available.</td></tr>`;

      const publication = registry.artifact_publication || {};
      const publicationRows = [];
      const corrected = publication.corrected_r4_pretraining || {};
      Object.entries(corrected.runs || {}).forEach(([name, run]) => {
        publicationRows.push({
          artifact: `Corrected r4 · ${name}`,
          repo: run.tagged_repo,
          published: `${int(corrected.saved_checkpoint_count_per_run)} / ${int(corrected.saved_checkpoint_count_per_run)} checkpoints`,
          head: run.tagged_head,
          manifest: run.final_model_safetensors_sha256,
          call: "",
          state: corrected.status || "unknown",
          note: "immutable tagged repo",
        });
        publicationRows.push({
          artifact: `Corrected legacy alias · ${name}`,
          repo: run.authorized_legacy_repo,
          published: `${int(corrected.saved_checkpoint_count_per_run)} / ${int(corrected.saved_checkpoint_count_per_run)} checkpoints`,
          head: run.authorized_legacy_head,
          manifest: run.final_model_safetensors_sha256,
          call: "",
          state: corrected.status || "unknown",
          note: "explicitly authorized corrected-r4 replacement",
        });
      });
      const endpointPublication = publication.v1_interleave_pretraining_endpoints || {};
      Object.entries(endpointPublication.endpoints || {}).forEach(([name, endpoint]) => {
        publicationRows.push({
          artifact: `V1 endpoint · ${name}`,
          repo: endpoint.repo,
          published: `${int(endpointPublication.payload_file_count_per_repo)} payload files`,
          head: endpoint.head,
          manifest: endpoint.manifest_sha256,
          call: endpoint.verification_call_id,
          state: endpoint.status || endpointPublication.status || "unknown",
          note: `checkpoint ${String(endpoint.checkpoint_fingerprint || "").slice(0,12)}`,
        });
      });
      const rlPublication = publication.e2_rl_checkpoints || {};
      Object.entries(rlPublication.runs || {}).forEach(([filter, run]) => {
        publicationRows.push({
          artifact: `E2 RL · ${filter}`,
          repo: run.repo,
          published: `${int(run.uploaded_count)} / ${int(rlPublication.expected_checkpoint_count_per_run)} checkpoints · tracker ${int(run.tracker_step)}`,
          head: run.head,
          manifest: run.publication_contract_sha256,
          call: run.uploader_call_id,
          state: run.status || rlPublication.status || "unknown",
          note: `steps ${(run.uploaded_steps || []).join(", ") || "none yet"}`,
        });
      });
      const superseded = rlPublication.superseded_attempt || {};
      Object.entries((superseded.repos || {})).forEach(([filter, run]) => {
        publicationRows.push({
          artifact: `Superseded E2 audit · ${filter}`,
          repo: run.repo,
          published: `${int((run.published_steps || []).length)} audit-only checkpoints`,
          head: run.head,
          manifest: "",
          call: (superseded.calls || {})[filter],
          state: superseded.status || "superseded",
          note: superseded.reason || "",
        });
      });
      const futureV2r2 = publication.future_v2r2_pretraining_endpoints || {};
      if (futureV2r2.function) {
        publicationRows.push({
          artifact: "Future full v2r2 endpoint path",
          repo: futureV2r2.repo,
          published: "0 · not invoked",
          head: futureV2r2.app_source_sha256,
          manifest: futureV2r2.contract_version,
          call: futureV2r2.call_id,
          state: futureV2r2.status || "unknown",
          note: futureV2r2.function,
        });
      }
      const firstEndpointAttempt = (endpointPublication.publication_attempts || {});
      document.querySelector("#interleave-artifact-note").textContent =
        publication.updated_at
          ? `${publicationRows.length} immutable publication records · registry ${ago(publication.updated_at)} · ${firstEndpointAttempt.initial_error || "all recorded remote verification checks passed"}`
          : "Artifact publication has not been added to the live registry.";
      document.querySelector("#interleave-artifacts").innerHTML = publicationRows.length ? publicationRows.map(r => {
        const repo = r.repo || "";
        const repoCell = repo
          ? `<a href="https://huggingface.co/${esc(repo)}" target="_blank" rel="noreferrer">${esc(repo)}</a>`
          : "—";
        const identity = [
          r.head ? `HEAD ${String(r.head).slice(0,12)}` : null,
          r.manifest ? `identity ${String(r.manifest).slice(0,12)}` : null,
        ].filter(Boolean).join(" · ") || "—";
        const identityTitle = [r.head, r.manifest].filter(Boolean).join(" · ");
        return `<tr>
          <td><span class="model-name">${esc(r.artifact)}</span></td>
          <td>${repoCell}</td>
          <td>${esc(r.published)}</td>
          <td title="${esc(identityTitle)}">${esc(identity)}</td>
          <td><span class="eval-stat-note">${esc(r.call || "—")}</span></td>
          <td><span class="state ${esc(r.state)}"><i class="state-dot"></i>${esc(r.state)}</span></td>
          <td><span class="eval-stat-note" title="${esc(r.note || "")}">${esc(r.note || "—")}</span></td>
        </tr>`;
      }).join("") : `<tr><td colspan="7" style="text-align:center;color:var(--muted);padding:20px">No artifact publication records are available.</td></tr>`;

      const diagnosticWeight = value => {
        const number = Number(value);
        return Number.isInteger(number) ? int(number) : number.toFixed(9);
      };
      const v2r3Aggregate = v2r3.aggregate || {};
      const v2r3Trajectories = v2r3.trajectories || [];
      const v2r3Rollouts = v2r3.rollouts || [];
      document.querySelector("#interleave-v2r3-note").textContent = v2r3.present
        ? `DIAGNOSTIC ONLY · NON-AUTHORIZING · ${esc(v2r3.version)} · ${int(v2r3Aggregate.training_call_count)} training calls · ${int(v2r3Aggregate.authenticated_snapshot_count)} / ${int(v2r3Aggregate.expected_snapshot_count)} authenticated snapshots · ${int(v2r3Aggregate.rollout_completed)} inspected · ${int(v2r3Aggregate.rollout_running_or_queued)} running/retrying`
        : "No v2r3 diagnostic contract is registered. This does not authorize production, P1, Exp2, or RL.";
      document.querySelector("#interleave-v2r3-trajectories").innerHTML = v2r3Trajectories.length ? v2r3Trajectories.map(r => {
        const terminalClass = r.terminal ? (r.training_status === "success_authenticated" ? "complete" : "stale") : "running";
        return `<tr>
          <td><span class="model-name">${diagnosticWeight(r.sft_loss_weight)}</span></td>
          <td><span class="eval-stat-note">${esc(r.training_call_id || "—")}</span></td>
          <td><span class="state ${terminalClass}"><i class="state-dot"></i>${esc(r.training_status)}</span></td>
          <td>${int(r.progress_step)} / ${int(r.target_step)}<div class="eval-stat-note">${pct(r.progress)} from authenticated snapshots</div></td>
          <td>${r.terminal ? "yes" : "no"}</td>
          <td>${int(r.authenticated_snapshot_count)} / ${int((r.snapshot_steps || []).length)}<div class="eval-stat-note">${(r.snapshot_steps || []).map(int).join(", ")}</div></td>
          <td>${storage(r.snapshot_total_bytes)}<div class="eval-stat-note">${int(r.snapshot_file_count)} files</div></td>
        </tr>`;
      }).join("") : `<tr><td colspan="7" style="text-align:center;color:var(--muted);padding:20px">No v2r3 trajectories are registered.</td></tr>`;
      document.querySelector("#interleave-v2r3-rollouts").innerHTML = v2r3Rollouts.length ? v2r3Rollouts.map(r => {
        const inspector = r.inspector || {};
        const statusClass = r.state === "complete" ? "complete" : r.state === "running" ? "running" : r.state === "failed" ? "stale" : "";
        const snapshotIdentity = r.hf_identity_sha256
          ? `HF ${String(r.hf_identity_sha256).slice(0,12)} · resume ${String(r.resume_identity_sha256).slice(0,12)}`
          : "awaiting authenticated snapshot";
        const snapshotTitle = [
          r.snapshot_marker_sha256,
          r.trainer_state_sha256,
          r.hf_identity_sha256,
          r.resume_identity_sha256,
        ].filter(Boolean).join(" · ");
        return `<tr>
          <td><span class="model-name">${diagnosticWeight(r.sft_loss_weight)} · step ${int(r.step)}</span><div class="eval-stat-note">${esc(r.run_name || "not launched")}</div></td>
          <td><span class="eval-stat-note">${esc(r.rollout_call_id || "—")}</span></td>
          <td><span class="state ${statusClass}"><i class="state-dot"></i>${esc(r.status)}</span></td>
          <td title="${esc(snapshotTitle)}"><span class="eval-stat-note">${esc(snapshotIdentity)}</span></td>
          <td>${inspector.joint_valid_protocol_rows == null ? "—" : `${int(inspector.joint_valid_protocol_rows)} / ${int(inspector.joint_valid_protocol_groups)}`}<div class="eval-stat-note">rate ${diagnosticRate(inspector.p_protocol)}</div></td>
          <td>${inspector.positive_samples == null ? "—" : int(inspector.positive_samples)}<div class="eval-stat-note">diagnostic rate ${diagnosticRate(inspector.diagnostic_positive_rate)}</div></td>
          <td>${inspector.nonzero_variance_groups == null ? "—" : int(inspector.nonzero_variance_groups)}<div class="eval-stat-note">rate ${diagnosticRate(inspector.variance_rate)}</div></td>
          <td>${inspector.rollout_rows == null ? "—" : int(inspector.rollout_rows)}<div class="eval-stat-note">${inspector.prompt_groups == null ? "—" : `${int(inspector.prompt_groups)} groups × ${int(inspector.samples_per_group)}`}</div></td>
          <td>${storage(r.snapshot_total_bytes)}<div class="eval-stat-note">${int(r.snapshot_file_count)} files</div></td>
        </tr>`;
      }).join("") : `<tr><td colspan="9" style="text-align:center;color:var(--muted);padding:20px">No v2r3 snapshot rollout inventory is registered.</td></tr>`;
      const v2r3Report = v2r3.final_report || {};
      document.querySelector("#interleave-v2r3-report").textContent = v2r3Report.immutable
        ? `Final immutable diagnostic report authenticated · ${int(v2r3Report.record_count)} / ${int(v2r3Aggregate.expected_snapshot_count)} records · SHA-256 ${String(v2r3Report.report_sha256 || "").slice(0,12)} · ${v2r3Report.audit_call_id || ""} · ${v2r3Report.path || ""}`
        : `Final immutable diagnostic report pending · ${v2r3Report.path || "no report path registered"} · pending reports grant no authorization`;

      document.querySelector("#interleave-stages").innerHTML = stages.length ? stages.map(r => {
        const evaluation = r.evaluation || {counts:{}, target_checkpoints:0};
        const counts = evaluation.counts || {};
        return `<tr>
          <td><span class="model-name">${esc(r.model)} / ${esc(r.arm)}</span><div class="eval-stat-note">${esc(r.run_name)}</div></td>
          <td>${esc(r.filter_mode)}</td>
          <td>${esc(r.phase)}</td>
          <td>${int(r.step)} / ${int(r.target)}</td>
          <td>${int(r.effective_step)}</td>
          <td><span class="state ${esc(r.state)}"><i class="state-dot"></i>${esc(r.state)}</span></td>
          <td>${int(counts.success)} / ${int(evaluation.target_checkpoints)}${counts.running ? ` · ${int(counts.running)} live` : ""}</td>
        </tr>`;
      }).join("") : `<tr><td colspan="7" style="text-align:center;color:var(--muted);padding:28px">Interleave registry/status is unavailable.</td></tr>`;

      const exp4Stage = stage => {
        stage = stage || {};
        const records = exp4Calls[stage.stage_id] || [];
        const call = records.length ? records[records.length - 1] : {};
        const state = call.modal_state || call.status || stage.status || "planned";
        return `<span class="state ${esc(state)}"><i class="state-dot"></i>${esc(state)}</span><div class="eval-stat-note">${esc(call.call_id || stage.call_id || "")}</div>`;
      };
      document.querySelector("#interleave-exp4-note").textContent = exp4.version
        ? `${exp4.version} · review ${exp4.review?.status || "unknown"} · ${int(exp4Arms.length)} registered arms`
        : "Exp 4 registration is not present in the live registry.";
      document.querySelector("#interleave-exp4").innerHTML = exp4Arms.length ? exp4Arms.map(arm => {
        const pipeline = arm.stages || {};
        return `<tr>
          <td><span class="model-name">${esc(arm.arm_id)}</span></td>
          <td>${esc(arm.filter)}</td>
          <td>${esc(arm.method)}</td>
          <td><span class="state ${esc(arm.status)}"><i class="state-dot"></i>${esc(arm.status)}</span></td>
          <td>${exp4Stage(pipeline.extract)}</td>
          <td>${exp4Stage(pipeline.validate)}</td>
          <td>${exp4Stage(pipeline.pretrain)}</td>
          <td>${exp4Stage(pipeline.rl)}</td>
        </tr>`;
      }).join("") : `<tr><td colspan="8" style="text-align:center;color:var(--muted);padding:20px">No Exp 4 arms registered yet.</td></tr>`;

      document.querySelector("#interleave-results").innerHTML = rows.length ? rows.map(r => {
        const evalClass = r.eval_status === "success" ? "complete" : r.eval_status === "running" ? "running" : r.eval_status === "failed" ? "stale" : "";
        return `<tr>
          <td><span class="model-name">${esc(r.model)} / ${esc(r.arm)}</span><div class="eval-stat-note">${esc(r.run_name)}</div></td>
          <td>${esc(r.filter_mode)}</td>
          <td>${esc(r.phase)}</td>
          <td>${int(r.phase_step)}</td>
          <td>${int(r.effective_rl_step)}</td>
          <td>${reward(r.pass_at_1)}</td>
          <td>${reward(r.avg_reward)}</td>
          <td>${reward(r.b3_b4_avg)}</td>
          <td><span class="state ${r.training_status === "checkpointed" ? "complete" : ""}"><i class="state-dot"></i>${esc(r.training_status)}</span></td>
          <td><span class="state ${evalClass}"><i class="state-dot"></i>${esc(r.eval_status)}</span></td>
        </tr>`;
      }).join("") : `<tr><td colspan="10" style="text-align:center;color:var(--muted);padding:28px">No interleave RL checkpoints have been observed yet.</td></tr>`;
    }

    function renderTraining() {
      document.querySelector("#training").innerHTML = RUN_ORDER.map(key => {
        const r = snapshot.training[key];
        const hf = r.hf;
        const speed = r.steps_per_hour ? `${Math.round(r.steps_per_hour)}/h` : "—";
        const eta = r.state === "complete" ? "done" : duration(r.eta_seconds);
        const stateLabel = r.state === "running" ? "running" : r.state;
        return `<article class="card train-card" style="--run:${r.color}">
          <div class="card-title-row">
            <div class="run-title">${esc(r.short)}</div>
            <div class="state ${esc(r.state)}"><i class="state-dot"></i>${esc(stateLabel)}</div>
          </div>
          <div class="step-row"><span class="step">${int(r.step)}</span><span class="target">/ ${int(r.target)} steps · ${pct(r.progress)}</span></div>
          <div class="progress-track"><div class="progress-fill" style="width:${pct(r.progress)}"></div></div>
          <div class="card-meta">
            <div class="meta-cell"><div class="meta-value">${int(hf.count)}</div><div class="meta-label">HF uploaded</div></div>
            <div class="meta-cell"><div class="meta-value">${speed}</div><div class="meta-label">Recent rate</div></div>
            <div class="meta-cell"><div class="meta-value">${eta}</div><div class="meta-label">Est. remaining</div></div>
          </div>
          <div class="card-links">
            <a href="${esc(r.modal_url)}" target="_blank" rel="noreferrer">Modal app ↗</a>
            <a href="${esc(hf.url)}" target="_blank" rel="noreferrer">Hugging Face ↗</a>
            <span style="color:var(--muted);font-size:11px;margin-left:auto">${ago(r.last_checkpoint_at)}</span>
          </div>
        </article>`;
      }).join("");
    }

    function renderQueue() {
      const e = snapshot.evaluation_aggregate.totals;
      const total = e.target || 1;
      const segments = [
        ["success", e.success, "#57d6b5", "Completed"],
        ["running", e.running, "#72a7ff", "Running"],
        ["failed", e.failed, "#ff7185", "Failed"],
        ["queued", e.queued, "#c18cff", "Queued"],
        ["not_discovered", e.not_discovered, "rgba(255,255,255,.10)", "Awaiting upload"],
      ];
      const median = duration(e.median_duration_seconds);
      const wall = duration(e.estimated_wall_seconds);
      document.querySelector("#queue").innerHTML = `
        <div class="summary-label">Evaluation fleet</div>
        <div class="queue-total">${int(e.success)} <span style="color:var(--muted);font-size:16px;font-weight:500">/ ${int(e.target)}</span></div>
        <div class="queue-sub">${pct(e.progress)} fully evaluated</div>
        <div class="queue-bar">${segments.map(([key,value,color]) => `<div class="queue-segment" title="${key}" style="width:${100*value/total}%;background:${color}"></div>`).join("")}</div>
        <div class="queue-list">${segments.map(([key,value,color,label]) => `<div class="queue-row"><span class="queue-name"><i class="queue-swatch" style="background:${color}"></i>${label}</span><span class="queue-count">${int(value)}</span></div>`).join("")}</div>
        <div class="queue-foot">Median checkpoint: <strong style="color:var(--text)">${median}</strong><br>Projected remaining wall time at ${int(e.active_workers)} active workers: <strong style="color:var(--text)">${wall}</strong>.<br>Configured ceiling: <strong style="color:var(--text)">${int(e.configured_worker_ceiling)} H200 workers</strong>.</div>`;
    }

    function renderEvalRuns() {
      document.querySelector("#eval-runs").innerHTML = RUN_ORDER.map(key => {
        const r = snapshot.evaluation[key];
        const c = r.counts;
        const target = r.target_checkpoints || 1;
        const active = c.running ? ` · running ${r.running_steps.join(", ")}` : "";
        return `<article class="card run-eval" style="--run:${r.color}">
          <div class="card-title-row"><div class="run-title">${esc(r.short)}</div><div class="state ${c.failed ? "stale" : c.running ? "running" : ""}"><i class="state-dot"></i>${c.failed ? `${c.failed} failed` : c.running ? `${c.running} live` : "queued"}</div></div>
          <div class="eval-stat-row"><span class="eval-stat-big">${int(c.success)} / ${int(target)}</span><span class="eval-stat-note">${pct(c.success/target)}</span></div>
          <div class="mini-bar"><div style="width:${pct(c.success/target)}"></div></div>
          <div class="eval-stat-note" style="margin-top:10px">latest step ${int(r.latest_success_step)} · best macro ${score(r.best_macro_mean)} @ ${r.best_step ?? "—"}${active}</div>
        </article>`;
      }).join("");
    }

    function renderResults() {
      const records = RUN_ORDER.flatMap(key =>
        snapshot.evaluation[key].series.map(record => ({
          model: key,
          color: snapshot.training[key].color,
          ...record
        }))
      );
      document.querySelector("#results-note").textContent =
        `${int(records.length)} completed checkpoints · canonical reward mean@16`;
      document.querySelector("#results").innerHTML = records.length ? records.map(r => `
        <tr>
          <td><span class="run-chip model-name"><i style="background:${r.color}"></i>${esc(r.model)}</span></td>
          <td>${int(r.step)}</td>
          <td>${reward(r.macro_mean)}</td>
          <td>${reward((r.benchmarks.B3.mean + r.benchmarks.B4.mean) / 2)}</td>
        </tr>`).join("") : `<tr><td colspan="4" style="text-align:center;color:var(--muted);padding:28px">No complete production evaluations yet.</td></tr>`;
    }

    function render() {
      renderSummary();
      renderInterleave();
      renderTraining();
      renderQueue();
      renderEvalRuns();
      renderResults();
      document.querySelector("#updated").textContent = `updated ${ago(snapshot.generated_at)}`;
      document.querySelector("#updated").title = new Date(snapshot.generated_at).toLocaleString();
      document.querySelector("#eval-modal-link").href = snapshot.evaluation_aggregate.modal_url;
      document.querySelector("#fingerprint").textContent = `eval fingerprint ${snapshot.evaluation_aggregate.settings.fingerprint.slice(0,12)}`;
      const problems = [
        ...(snapshot.errors || []),
        ...(snapshot.snapshot_error ? [snapshot.snapshot_error] : []),
        ...((snapshot.interleave && snapshot.interleave.errors) || []),
        ...RUN_ORDER.flatMap(key => snapshot.training[key].hf.error ? [`${snapshot.training[key].short} HF: ${snapshot.training[key].hf.error}`] : [])
      ];
      const banner = document.querySelector("#error");
      banner.style.display = problems.length ? "block" : "none";
      banner.textContent = problems.join(" · ");
    }

    async function load(force=false) {
      if (busy) return;
      busy = true;
      const button = document.querySelector("#refresh");
      button.disabled = true;
      button.textContent = "Refreshing…";
      try {
        const response = await fetch(`/api/snapshot${force ? "?force=1" : ""}`, {cache:"no-store"});
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        snapshot = await response.json();
        render();
        if (force && snapshot.refresh_requested) {
          setTimeout(() => load(false), 15_000);
        }
      } catch (error) {
        const banner = document.querySelector("#error");
        banner.style.display = "block";
        banner.textContent = `Could not refresh dashboard: ${error.message}`;
      } finally {
        busy = false;
        button.disabled = false;
        button.textContent = "Refresh now";
      }
    }

    document.querySelector("#refresh").addEventListener("click", () => load(true));
    load();
    setInterval(() => load(false), 30_000);
  </script>
</body>
</html>
"""


@app.function(
    image=dashboard_image,
    timeout=60,
    max_containers=2,
    scaledown_window=5 * 60,
)
@modal.asgi_app()
def web():
    import csv
    import io

    from fastapi import FastAPI, HTTPException, Query
    from fastapi.responses import HTMLResponse, JSONResponse, Response
    from starlette.middleware.gzip import GZipMiddleware

    api = FastAPI(
        title="Chess RL live dashboard",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    api.add_middleware(GZipMiddleware, minimum_size=1_000)

    @api.get("/", response_class=HTMLResponse)
    def dashboard() -> HTMLResponse:
        return HTMLResponse(
            DASHBOARD_HTML,
            headers={"Cache-Control": "public, max-age=60"},
        )

    @api.get("/api/snapshot", response_class=JSONResponse)
    def snapshot_endpoint(
        force: bool = Query(False, description="Force source refresh")
    ) -> JSONResponse:
        value: dict[str, Any] | None = None
        if not force:
            try:
                value = dashboard_state.get("snapshot")
            except Exception:
                value = None
        else:
            try:
                value = dashboard_state.get("snapshot")
            except Exception:
                value = None
            refresh_snapshot.spawn()
        if value is None:
            refresh_snapshot.spawn()
            raise HTTPException(
                status_code=503,
                detail="Dashboard snapshot is initializing; retry shortly.",
            )
        try:
            live_feed = dashboard_state.get("control_plane")
            if live_feed is not None:
                value = _overlay_live_control_plane(value, live_feed)
        except Exception as exc:
            value = json.loads(json.dumps(value))
            value.setdefault("errors", []).append(
                f"live control-plane overlay rejected: "
                f"{type(exc).__name__}: {exc}"
            )
        if force:
            value = {**value, "refresh_requested": True}
        return JSONResponse(
            value,
            headers={
                "Cache-Control": "no-store",
                "Access-Control-Allow-Origin": "*",
            },
        )

    @api.get("/api/results.csv")
    def results_csv() -> Response:
        try:
            value = dashboard_state.get("snapshot")
        except Exception:
            value = None
        if value is None:
            refresh_snapshot.spawn()
            raise HTTPException(
                status_code=503,
                detail="Dashboard snapshot is initializing; retry shortly.",
            )

        output = io.StringIO(newline="")
        writer = csv.writer(output)
        writer.writerow(
            ("model", "step", "pass@1_avg_reward", "b3_b4_avg_reward")
        )
        evaluation = value.get("evaluation", {})
        for run_key in RUNS:
            records = evaluation.get(run_key, {}).get("series", [])
            for record in sorted(records, key=lambda item: int(item["step"])):
                macro_mean = record.get("macro_mean")
                if not isinstance(macro_mean, (int, float)):
                    continue
                benchmarks = record.get("benchmarks", {})
                b3 = benchmarks.get("B3", {}).get("mean")
                b4 = benchmarks.get("B4", {}).get("mean")
                if not isinstance(b3, (int, float)) or not isinstance(
                    b4, (int, float)
                ):
                    continue
                writer.writerow(
                    (
                        run_key,
                        int(record["step"]),
                        repr(macro_mean),
                        repr((b3 + b4) / 2),
                    )
                )
        return Response(
            output.getvalue(),
            media_type="text/csv; charset=utf-8",
            headers={
                "Cache-Control": "no-store",
                "Content-Disposition": (
                    'attachment; filename="chess_rl_pass_at_1_avg_reward.csv"'
                ),
                "Access-Control-Allow-Origin": "*",
            },
        )

    @api.get("/api/interleave-results.csv")
    def interleave_results_csv() -> Response:
        try:
            value = dashboard_state.get("snapshot")
        except Exception:
            value = None
        if value is None:
            refresh_snapshot.spawn()
            raise HTTPException(
                status_code=503,
                detail="Dashboard snapshot is initializing; retry shortly.",
            )

        output = io.StringIO(newline="")
        writer = csv.writer(output)
        writer.writerow(
            (
                "model",
                "experiment",
                "arm",
                "filter_mode",
                "phase",
                "run_name",
                "phase_step",
                "effective_rl_step",
                "pass_at_1",
                "avg_reward",
                "b3_b4_avg",
                "training_status",
                "eval_status",
                "pass_at_1_semantics",
            )
        )
        rows = value.get("interleave", {}).get("rows", [])
        for row in rows:
            writer.writerow(
                (
                    row.get("model"),
                    row.get("experiment"),
                    row.get("arm"),
                    row.get("filter_mode"),
                    row.get("phase"),
                    row.get("run_name"),
                    row.get("phase_step"),
                    row.get("effective_rl_step"),
                    row.get("pass_at_1"),
                    row.get("avg_reward"),
                    row.get("b3_b4_avg"),
                    row.get("training_status"),
                    row.get("eval_status"),
                    row.get("pass_at_1_semantics"),
                )
            )
        return Response(
            output.getvalue(),
            media_type="text/csv; charset=utf-8",
            headers={
                "Cache-Control": "no-store",
                "Content-Disposition": (
                    'attachment; filename="interleave_47m_results.csv"'
                ),
                "Access-Control-Allow-Origin": "*",
            },
        )

    @api.get("/healthz", response_class=JSONResponse)
    def health() -> JSONResponse:
        return JSONResponse({"ok": True, "service": APP_NAME})

    return api
