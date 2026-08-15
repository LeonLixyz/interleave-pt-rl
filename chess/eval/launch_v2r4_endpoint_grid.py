"""Exact-once launcher for the three v2r4 P1 snapshot endpoint evaluations."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import modal


APP_NAME = "chess-interleave-endpoint-eval-v2r1"
LEDGER_PATH = (
    Path(__file__).resolve().parents[1]
    / "INTERLEAVED_V2R4_ENDPOINT_LAUNCH_LEDGER.json"
)
SNAPSHOT_ROOT = (
    "/pretrain-checkpoints/interleave_50m/pretrain/"
    "mix10b_sft90k_3072_v2r3_diagnostic_20260730/"
    "p1_w4067c60eaba84b1e/snapshots"
)
ENDPOINT_EVAL_SOURCE_SHA256 = (
    "80caa51691611ad89a2496e2ca89f1c4039777d1d1a9b663fc550a26cff585f0"
)
CANDIDATES = {
    6_000: {
        "checkpoint_path": f"{SNAPSHOT_ROOT}/step_6000/hf",
        "checkpoint_sha256": (
            "17acd19dd1e89390c609a3f0f6c72ab543b8869f2d2ffd10528c8fe84cb20690"
        ),
    },
    8_000: {
        "checkpoint_path": f"{SNAPSHOT_ROOT}/step_8000/hf",
        "checkpoint_sha256": (
            "e1006a970b5b7c9c9e5aefdbae3c716740e69970c0bcb4bb32b4cbab7af43634"
        ),
    },
    9_920: {
        "checkpoint_path": f"{SNAPSHOT_ROOT}/step_9920/hf",
        "checkpoint_sha256": (
            "9a89d52a60b87b0f27108e5b08e33395757e374a4b59a592babb9435edb4b1c8"
        ),
    },
}


def _canonical_hash(value: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()


def _write_ledger(value: dict[str, object]) -> None:
    core = {key: item for key, item in value.items() if key != "ledger_sha256"}
    value["ledger_sha256"] = _canonical_hash(core)
    temporary = LEDGER_PATH.with_name(f".{LEDGER_PATH.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, LEDGER_PATH)


def main() -> None:
    if LEDGER_PATH.exists():
        raise FileExistsError(
            f"exact-once endpoint ledger already exists: {LEDGER_PATH}"
        )
    loss_function = modal.Function.from_name(APP_NAME, "eval_losses_one")
    chess_function = modal.Function.from_name(APP_NAME, "eval_chess_one")
    ledger: dict[str, object] = {
        "schema": "interleaved-v2r4-endpoint-launch-ledger-v1",
        "app_name": APP_NAME,
        "endpoint_eval_source_sha256": ENDPOINT_EVAL_SOURCE_SHA256,
        "state": "launching",
        "expected_call_count": 6,
        "calls": [],
    }
    _write_ledger(ledger)
    for step, candidate in sorted(CANDIDATES.items()):
        endpoint = {
            "endpoint_id": f"v2r4-s{step}",
            "experiment": "V2R4",
            "phase": "P1",
            "filter": None,
            "method": "production-gate-candidate",
            "checkpoint_path": candidate["checkpoint_path"],
            "training_state": {
                "p2_consumed": False,
                "snapshot_step": step,
                "trajectory": "p1_w4067c60eaba84b1e",
            },
            "completion_marker": None,
            "declared_checkpoint_sha256": candidate["checkpoint_sha256"],
        }
        for component, function in (
            ("losses", loss_function),
            ("chess", chess_function),
        ):
            try:
                call = function.spawn(
                    endpoint,
                    candidate["checkpoint_sha256"],
                )
            except Exception as exc:
                ledger["state"] = "launch_failed"
                ledger["failed_cell"] = {
                    "step": step,
                    "component": component,
                }
                ledger["launch_error"] = f"{type(exc).__name__}: {exc}"
                _write_ledger(ledger)
                raise
            calls = ledger["calls"]
            assert isinstance(calls, list)
            calls.append(
                {
                    "step": step,
                    "component": component,
                    "endpoint_id": endpoint["endpoint_id"],
                    "checkpoint_sha256": candidate["checkpoint_sha256"],
                    "function_call_id": call.object_id,
                }
            )
            _write_ledger(ledger)
            print(
                f"SPAWNED {step}/{component}: {call.object_id}",
                flush=True,
            )
    ledger["state"] = "launched_all"
    _write_ledger(ledger)
    print(json.dumps(ledger, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
