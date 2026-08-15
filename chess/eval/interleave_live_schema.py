"""Pure validation helpers for live interleave dashboard telemetry."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any


LIVE_FEED_SCHEMA = "interleave-dashboard-live-feed-v1"
PRETRAIN_METRICS_SCHEMA = "interleaved-local-metrics-v1"
PRETRAIN_TOKEN_POSITIONS_PER_STEP = 21 * 8 * 3_072


def canonical_json_sha256(value: Any) -> str:
    """Hash JSON using one deterministic, whitespace-free representation."""

    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _finite_number(value: Any, field: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    if positive and result <= 0:
        raise ValueError(f"{field} must be positive")
    return result


def _integer(value: Any, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    if value < minimum:
        raise ValueError(f"{field} must be >= {minimum}")
    return value


def validate_runtime_provenance(value: Any) -> dict[str, Any]:
    """Validate the runtime identity emitted with every pretraining metric."""

    if not isinstance(value, Mapping):
        raise ValueError("runtime_provenance must be an object")
    required_strings = (
        "attention_backend",
        "torch_compile_mode",
        "torch_version",
        "transformers_version",
    )
    result = dict(value)
    for field in required_strings:
        item = result.get(field)
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"runtime_provenance.{field} must be non-empty")
    flash_version = result.get("flash_attention_version")
    if flash_version is not None and (
        not isinstance(flash_version, str) or not flash_version.strip()
    ):
        raise ValueError(
            "runtime_provenance.flash_attention_version must be null or non-empty"
        )
    _integer(
        result.get("data_num_workers"),
        "runtime_provenance.data_num_workers",
    )
    return result


def parse_pretrain_metrics_jsonl(
    text: str,
    *,
    target_step: int,
    last_update_at: str | None,
    token_positions_per_step: int = PRETRAIN_TOKEN_POSITIONS_PER_STEP,
) -> dict[str, Any]:
    """Strictly parse one append-only pretraining metric stream.

    Every non-empty line must be complete and valid. A malformed tail is
    reported as an error instead of silently presenting stale telemetry.
    """

    target = _integer(target_step, "target_step", minimum=1)
    positions_per_step = _integer(
        token_positions_per_step,
        "token_positions_per_step",
        minimum=1,
    )
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("metrics.jsonl is empty")

    last_step = 0
    manifest_hash: str | None = None
    runtime: dict[str, Any] | None = None
    latest: dict[str, Any] | None = None
    for index, line in enumerate(lines, start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"metrics line {index} is malformed JSON") from exc
        if not isinstance(record, Mapping):
            raise ValueError(f"metrics line {index} must be an object")
        if record.get("schema") != PRETRAIN_METRICS_SCHEMA:
            raise ValueError(f"metrics line {index} has unsupported schema")
        step = _integer(
            record.get("step"),
            f"metrics line {index}.step",
            minimum=1,
        )
        if step <= last_step:
            raise ValueError("metric steps must be strictly increasing")
        if step > target:
            raise ValueError(f"metric step {step} exceeds target {target}")
        last_step = step

        record_manifest = record.get("manifest_hash")
        if (
            not isinstance(record_manifest, str)
            or re.fullmatch(r"[0-9a-f]{64}", record_manifest) is None
        ):
            raise ValueError(f"metrics line {index} has invalid manifest_hash")
        if manifest_hash is None:
            manifest_hash = record_manifest
        elif record_manifest != manifest_hash:
            raise ValueError("manifest_hash changed inside metrics.jsonl")

        record_runtime = validate_runtime_provenance(
            record.get("runtime_provenance")
        )
        if runtime is None:
            runtime = record_runtime
        elif canonical_json_sha256(record_runtime) != canonical_json_sha256(
            runtime
        ):
            raise ValueError("runtime_provenance changed inside metrics.jsonl")

        metrics = record.get("metrics")
        if not isinstance(metrics, Mapping):
            raise ValueError(f"metrics line {index}.metrics must be an object")
        loss = _finite_number(metrics.get("train/loss"), "train/loss")
        if loss < 0:
            raise ValueError("train/loss must be non-negative")
        lr = _finite_number(metrics.get("train/lr"), "train/lr", positive=True)
        tokens_per_second = _finite_number(
            metrics.get("train/token_positions_per_second"),
            "train/token_positions_per_second",
            positive=True,
        )
        manifest_cursor = _integer(
            metrics.get("train/manifest_cursor"),
            "train/manifest_cursor",
        )
        global_valid_tokens = _integer(
            metrics.get("train/global_valid_tokens"),
            "train/global_valid_tokens",
            minimum=1,
        )
        latest = {
            "step": step,
            "loss": loss,
            "lr": lr,
            "tokens_per_second": tokens_per_second,
            "manifest_cursor": manifest_cursor,
            "global_valid_tokens": global_valid_tokens,
        }

    assert latest is not None and manifest_hash is not None and runtime is not None
    remaining_steps = max(0, target - latest["step"])
    return {
        **latest,
        "target": target,
        "remaining": remaining_steps,
        "progress": latest["step"] / target,
        "eta_seconds": (
            remaining_steps * positions_per_step / latest["tokens_per_second"]
        ),
        "token_positions_per_step": positions_per_step,
        "last_update_at": last_update_at,
        "manifest_hash": manifest_hash,
        "runtime_provenance": runtime,
        "metric_records": len(lines),
    }


def validate_pretrain_trainer_state(
    value: Any,
    *,
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a durable resume checkpoint against the live metric stream."""

    if not isinstance(value, Mapping):
        raise ValueError("trainer_state.json must be an object")
    if value.get("schema_version") != 1:
        raise ValueError("trainer_state.json has unsupported schema_version")
    step = _integer(value.get("global_step"), "trainer_state.global_step")
    if step > int(metrics["step"]):
        raise ValueError("trainer state is ahead of the metric stream")
    if value.get("manifest_hash") != metrics["manifest_hash"]:
        raise ValueError("trainer state manifest_hash does not match metrics")
    if _integer(value.get("local_batch_size"), "local_batch_size", minimum=1) != 21:
        raise ValueError("trainer state local_batch_size is not 21")
    if _integer(value.get("world_size"), "world_size", minimum=1) != 8:
        raise ValueError("trainer state world_size is not 8")
    if (
        _integer(
            value.get("gradient_accumulation_steps"),
            "gradient_accumulation_steps",
            minimum=1,
        )
        != 1
    ):
        raise ValueError("trainer state gradient accumulation is not 1")
    runtime = validate_runtime_provenance(value.get("runtime_provenance"))
    if canonical_json_sha256(runtime) != canonical_json_sha256(
        metrics["runtime_provenance"]
    ):
        raise ValueError("trainer state runtime provenance does not match metrics")
    return {
        "step": step,
        "manifest_cursor": _integer(
            value.get("manifest_cursor"),
            "trainer_state.manifest_cursor",
        ),
        "runtime_provenance": runtime,
    }


def build_live_feed(
    registry: Mapping[str, Any],
    pretraining: Mapping[str, Any],
    *,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Create a content-addressed registry + pretraining telemetry envelope."""

    timestamp = generated_at or datetime.now(timezone.utc).isoformat()
    payload = {
        "registry": dict(registry),
        "pretraining": dict(pretraining),
    }
    return {
        "schema": LIVE_FEED_SCHEMA,
        "generated_at": timestamp,
        "registry_sha256": canonical_json_sha256(payload["registry"]),
        "pretraining_sha256": canonical_json_sha256(payload["pretraining"]),
        "payload_sha256": canonical_json_sha256(payload),
        **payload,
    }


def validate_live_feed(value: Any) -> dict[str, Any]:
    """Fail closed unless all live-feed content hashes and timestamp validate."""

    if not isinstance(value, Mapping):
        raise ValueError("live feed must be an object")
    if value.get("schema") != LIVE_FEED_SCHEMA:
        raise ValueError("live feed has unsupported schema")
    generated_at = value.get("generated_at")
    if not isinstance(generated_at, str):
        raise ValueError("live feed generated_at must be an ISO timestamp")
    try:
        parsed = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("live feed generated_at is malformed") from exc
    if parsed.tzinfo is None:
        raise ValueError("live feed generated_at must include a timezone")
    registry = value.get("registry")
    pretraining = value.get("pretraining")
    if not isinstance(registry, Mapping) or not isinstance(pretraining, Mapping):
        raise ValueError("live feed registry and pretraining must be objects")
    expected = {
        "registry_sha256": canonical_json_sha256(registry),
        "pretraining_sha256": canonical_json_sha256(pretraining),
        "payload_sha256": canonical_json_sha256(
            {"registry": dict(registry), "pretraining": dict(pretraining)}
        ),
    }
    for field, digest in expected.items():
        if value.get(field) != digest:
            raise ValueError(f"live feed {field} mismatch")
    return dict(value)
