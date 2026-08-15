"""Launch the two omitted context-2048 RL runs after all four evaluations.

The active evaluation pipeline was submitted with only the two staged-training
checkpoints in its RL launch list.  This detached controller waits until that
pipeline has evaluated and filtered all four checkpoints and has launched its
two RL calls.  It then launches the two mixed-training checkpoints exactly
once, with otherwise identical RL settings.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal


APP_NAME = "chess-context2048-launch-remaining-rl"
VERSION = "context2048-vocab-mixing-pass16-native2048-bos-v1-20260812"
RESULTS_VOLUME_NAME = "chess-rl-eval-results-r6"
RESULTS_MOUNT = Path("/results")
RESULTS_ROOT = RESULTS_MOUNT / VERSION
PIPELINE_PATH = RESULTS_ROOT / "pipeline.json"
LEDGER_PATH = RESULTS_ROOT / "remaining_rl_launch.json"

RL_APP_NAME = "chess-interleave-rl"
RL_FUNCTION_NAME = "train_hf"
RL_PROJECT = "chess-47m-context2048-rl"
# The already-running controller captured this group at submission time.  Keep
# the two added arms in the same W&B group; the group can be renamed as one unit.
ACTIVE_PIPELINE_GROUP = "staged-tokenizer-comparison-filtered-lr1e5"

ALL_CHECKPOINTS = (
    "vocab81_then_sft3",
    "vocab85_then_sft3",
    "mixed_sft1",
    "mixed_sft3",
)
ORIGINAL_RL_CHECKPOINTS = (
    "vocab81_then_sft3",
    "vocab85_then_sft3",
)
REMAINING_RL: dict[str, dict[str, str]] = {
    "mixed_sft1": {
        "checkpoint": (
            "/pretrain-checkpoints/context2048_vocab_mixing_v3_20260812/"
            "mixed_sft1/mixed/final"
        ),
        "fingerprint": (
            "f456a8ef8bbd1b3b642d3fd305bbb1f6f523b4dfa08017282a16858b257990be"
        ),
        "run_name": "ctx2048-mixed-sft1-filtered-lr1e5-rl1500",
    },
    "mixed_sft3": {
        "checkpoint": (
            "/pretrain-checkpoints/context2048_vocab_mixing_v3_20260812/"
            "mixed_sft3/mixed/final"
        ),
        "fingerprint": (
            "6c6372296b19594b99e09ae7b0d500c5ac49ed3a95279fecdbf95b2bb1a11968"
        ),
        "run_name": "ctx2048-mixed-sft3-filtered-lr1e5-rl1500",
    },
}


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temp, path)


def _load_authenticated_filter(key: str) -> dict[str, Any]:
    path = RESULTS_ROOT / key / "filter/success.json"
    summary = json.loads(path.read_text())
    stored_sha = summary.get("filter_sha256")
    core = {name: value for name, value in summary.items() if name != "filter_sha256"}
    if stored_sha != canonical_sha256(core):
        raise ValueError(f"filter marker hash mismatch for {key}")
    if summary.get("version") != VERSION or summary.get("checkpoint") != key:
        raise ValueError(f"filter marker identity mismatch for {key}")
    filtered = summary.get("filtered_parquet", {})
    if not str(filtered.get("path", "")).startswith("/data/chess-rl-data/"):
        raise ValueError(f"unexpected filtered parquet path for {key}")
    if len(str(filtered.get("sha256", ""))) != 64 or int(filtered.get("rows", 0)) <= 0:
        raise ValueError(f"invalid filtered parquet record for {key}")
    return summary


def _rl_kwargs(key: str, filter_summary: dict[str, Any]) -> dict[str, Any]:
    spec = REMAINING_RL[key]
    filtered = filter_summary["filtered_parquet"]
    return {
        "hf_checkpoint": spec["checkpoint"],
        "run_name": spec["run_name"],
        "num_rollout": 1_500,
        "dynamic_filter": False,
        "rollout_seed": 42,
        "save_interval": 40,
        "eval_interval": 0,
        "model_id": "context2048_47m_qwen3",
        "resume_if_available": True,
        "wandb_project": RL_PROJECT,
        "wandb_group": ACTIVE_PIPELINE_GROUP,
        "max_tokens_per_gpu": 131_072,
        "sglang_server_concurrency": 128,
        "deterministic_inference": False,
        "rollout_only": False,
        "canary": False,
        "train_file": filtered["path"],
        "train_file_sha256": filtered["sha256"],
        "lr": "1e-5",
        "kl_loss_type": "low_var_kl",
        "rollout_max_prompt_len": 512,
        "rollout_max_response_len": 1_536,
        "rollout_max_context_len": 2_048,
    }


results_volume = modal.Volume.from_name(
    RESULTS_VOLUME_NAME, create_if_missing=False
)
app = modal.App(APP_NAME, image=modal.Image.debian_slim(python_version="3.11"))


@app.function(
    cpu=1.0,
    memory=1024,
    timeout=60 * 60 * 30,
    retries=0,
    volumes={str(RESULTS_MOUNT): results_volume},
)
def launch_after_all_filters() -> dict[str, Any]:
    """Wait for the original pipeline, then launch only its two omitted arms."""
    while True:
        results_volume.reload()
        if PIPELINE_PATH.is_file():
            pipeline = json.loads(PIPELINE_PATH.read_text())
            if pipeline.get("state") == "rl_launched":
                break
        time.sleep(30)

    original_calls = pipeline.get("rl_calls", [])
    original_keys = tuple(item.get("checkpoint") for item in original_calls)
    if original_keys != ORIGINAL_RL_CHECKPOINTS:
        raise ValueError(
            "original pipeline RL calls changed; refusing an ambiguous launch: "
            f"{original_keys}"
        )

    filters = {key: _load_authenticated_filter(key) for key in ALL_CHECKPOINTS}
    if any(key in original_keys for key in REMAINING_RL):
        raise ValueError("original pipeline already contains a remaining RL arm")

    if LEDGER_PATH.is_file():
        ledger = json.loads(LEDGER_PATH.read_text())
        if ledger.get("state") == "launched":
            return ledger
    else:
        ledger = {
            "schema": "context2048-remaining-rl-launch-v1",
            "version": VERSION,
            "state": "launching",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "original_pipeline_rl_calls": original_calls,
            "rl_calls": [],
        }
        atomic_json(LEDGER_PATH, ledger)
        results_volume.commit()

    launched_keys = {item["checkpoint"] for item in ledger.get("rl_calls", [])}
    unknown = launched_keys - set(REMAINING_RL)
    if unknown:
        raise ValueError(f"unexpected checkpoint in launch ledger: {sorted(unknown)}")

    rl_function = modal.Function.from_name(RL_APP_NAME, RL_FUNCTION_NAME)
    for key in REMAINING_RL:
        if key in launched_keys:
            continue
        kwargs = _rl_kwargs(key, filters[key])
        call = rl_function.spawn(**kwargs)
        ledger["rl_calls"].append(
            {
                "checkpoint": key,
                "checkpoint_fingerprint": REMAINING_RL[key]["fingerprint"],
                "run_name": kwargs["run_name"],
                "function_call_id": call.object_id,
                "wandb_project": RL_PROJECT,
                "wandb_group": ACTIVE_PIPELINE_GROUP,
                "filtered_parquet": filters[key]["filtered_parquet"],
            }
        )
        atomic_json(LEDGER_PATH, ledger)
        results_volume.commit()

    ledger["state"] = "launched"
    ledger["finished_at"] = datetime.now(timezone.utc).isoformat()
    ledger["ledger_sha256"] = canonical_sha256(ledger)
    atomic_json(LEDGER_PATH, ledger)
    results_volume.commit()
    return ledger


@app.local_entrypoint()
def main() -> None:
    call = launch_after_all_filters.spawn()
    print(
        json.dumps(
            {
                "controller_call_id": call.object_id,
                "version": VERSION,
                "waits_for": list(ALL_CHECKPOINTS),
                "launches": list(REMAINING_RL),
                "wandb_project": RL_PROJECT,
            },
            indent=2,
            sort_keys=True,
        )
    )
