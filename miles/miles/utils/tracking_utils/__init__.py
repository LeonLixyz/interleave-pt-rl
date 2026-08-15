import hashlib
import json
import logging
import math
import numbers
import os
from pathlib import Path
from typing import Any

from .base import TrackingManager

logger = logging.getLogger(__name__)
_manager = TrackingManager()


def _gate_metric_scalar(name: str, value: Any) -> int | float:
    if hasattr(value, "numel") and hasattr(value, "item"):
        if int(value.numel()) != 1:
            raise RuntimeError(
                f"Precision-gate metric {name!r} is not scalar"
            )
        value = value.item()
    elif hasattr(value, "item") and not isinstance(value, numbers.Real):
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise RuntimeError(
            f"Precision-gate metric {name!r} has unsupported value {value!r}"
        )
    normalized = float(value)
    if not math.isfinite(normalized):
        raise RuntimeError(
            f"Precision-gate metric {name!r} is nonfinite: {value!r}"
        )
    return int(value) if isinstance(value, int) else normalized


def _record_precision_gate_metrics(
    metrics: dict[str, Any],
    *,
    step_key: str,
) -> dict[str, Any] | None:
    """Append authenticated finite scalar metrics for the two-process gate."""

    leg = os.environ.get("CHESS_RL_MILES_PRECISION_GATE_LEG")
    artifact_root = os.environ.get("CHESS_RL_MILES_ARTIFACT_ROOT")
    if leg is None:
        return None
    if leg not in {"1", "2"} or not artifact_root:
        raise RuntimeError(
            "Precision-gate metrics require leg 1/2 and an artifact root"
        )
    normalized = {
        str(name): _gate_metric_scalar(str(name), value)
        for name, value in sorted(metrics.items())
    }
    if step_key not in normalized:
        raise RuntimeError(
            f"Precision-gate metric event lacks step key {step_key!r}"
        )
    core = {
        "schema": "miles-precision-gate-metric-v1",
        "leg": int(leg),
        "step_key": str(step_key),
        "step": normalized[step_key],
        "metrics": normalized,
    }
    digest = hashlib.sha256(
        json.dumps(
            core,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    payload = {**core, "evidence_sha256": digest}
    output = (
        Path(artifact_root)
        / "precision_gate"
        / f"leg_{leg}_metrics.jsonl"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, allow_nan=False, sort_keys=True) + "\n"
    ).encode()
    descriptor = os.open(
        output,
        os.O_WRONLY | os.O_CREAT | os.O_APPEND,
        0o644,
    )
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return payload


def init_tracking(args, primary: bool = True, **kwargs):
    _manager.init(args, primary=primary, **kwargs)


def log(args, metrics, step_key: str):
    step = metrics.get(step_key)
    _record_precision_gate_metrics(metrics, step_key=step_key)
    _manager.log(metrics, step=step)


def finish_tracking():
    _manager.finish()
