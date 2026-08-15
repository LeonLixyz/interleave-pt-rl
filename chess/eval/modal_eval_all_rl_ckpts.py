"""Resumable Modal evaluation queues for r6 and interleaved RL checkpoints.

The worker runs the repository's ``verl_eval.sh`` unchanged in evaluation
semantics (B1-B5, response length 2560, temperature 1, 16 samples). Results,
logs, metrics, and durable success/failure markers are written to a Modal
Volume. A lightweight watcher discovers new checkpoints while RL training and
Hugging Face uploads or interleaved Miles training are still in progress.

Historical r6 jobs continue to use their original Hugging Face discovery and
``v1`` result namespace.  Interleaved jobs discover raw ``iter_*`` Miles
checkpoints from the training Volume, convert only the requested checkpoint to
temporary HF format on the worker, and write to the isolated
``interleave_v1`` namespace.

Examples:
    modal run modal_eval_all_rl_ckpts.py --mode smoke
    modal run modal_eval_all_rl_ckpts.py --mode pilot
    modal run --detach modal_eval_all_rl_ckpts.py --mode launch
    modal run --detach modal_eval_all_rl_ckpts.py --mode launch --lease-hours 0
    modal run modal_eval_all_rl_ckpts.py --mode status
    modal run modal_eval_all_rl_ckpts.py --mode interleave-dry-run
    modal run --detach modal_eval_all_rl_ckpts.py --mode interleave-launch
    modal run modal_eval_all_rl_ckpts.py --mode interleave-status
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal

try:
    from .interleave_eval_queue import (
        INTERLEAVE_NAMESPACE,
        build_interleave_dry_run_plan,
        cadence_steps,
        final_table_metrics,
        flatten_interleave_eval_registry,
        parse_raw_checkpoint_step,
    )
    from .interleave_exp4_eval_queue import flatten_exp4_rl_registry
    from .interleave_live_schema import validate_live_feed
except ImportError:
    from interleave_eval_queue import (
        INTERLEAVE_NAMESPACE,
        build_interleave_dry_run_plan,
        cadence_steps,
        final_table_metrics,
        flatten_interleave_eval_registry,
        parse_raw_checkpoint_step,
    )
    from interleave_exp4_eval_queue import flatten_exp4_rl_registry
    from interleave_live_schema import validate_live_feed


HERE = Path(__file__).resolve().parent
VERL_ROOT = HERE / "pre2post-chess" / "rl"
EVAL_DATA_ROOT = HERE / "test_data"
WORKSPACE_ROOT = HERE.parent
INTERLEAVE_REGISTRY_LOCAL = WORKSPACE_ROOT / "INTERLEAVED_CORE_REGISTRY.json"
INTERLEAVE_QUEUE_LOCAL = HERE / "interleave_eval_queue.py"
INTERLEAVE_DASHBOARD_SCHEMA_LOCAL = HERE / "interleave_dashboard_schema.py"
INTERLEAVE_EXP4_QUEUE_LOCAL = HERE / "interleave_exp4_eval_queue.py"
INTERLEAVE_LIVE_SCHEMA_LOCAL = HERE / "interleave_live_schema.py"
MILES_CONVERTER_LOCAL = WORKSPACE_ROOT / "miles" / "tools" / "convert_fsdp_to_hf.py"

APP_NAME = "rl-eval"
RESULTS_VOLUME_NAME = "chess-rl-eval-results-r6"
RESULTS_ROOT = "/results"
REMOTE_VERL_ROOT = "/root/verl"
REMOTE_EVAL_DATA = "/eval-data"
INTERLEAVE_REGISTRY_REMOTE = Path(
    "/opt/chess-eval/INTERLEAVED_CORE_REGISTRY.json"
)
INTERLEAVE_QUEUE_REMOTE = "/root/interleave_eval_queue.py"
INTERLEAVE_DASHBOARD_SCHEMA_REMOTE = "/root/interleave_dashboard_schema.py"
INTERLEAVE_EXP4_QUEUE_REMOTE = "/root/interleave_exp4_eval_queue.py"
INTERLEAVE_LIVE_SCHEMA_REMOTE = "/root/interleave_live_schema.py"
INTERLEAVE_LIVE_FEED_POINTER = (
    Path(RESULTS_ROOT) / "dashboard_live" / "interleave" / "latest.json"
)
MILES_CONVERTER_REMOTE = "/opt/chess-eval/convert_fsdp_to_hf.py"
INTERLEAVE_RAW_VOLUME_NAME = "chess-rl-miles-checkpoints"
INTERLEAVE_HF_VOLUME_NAME = "rl-reasoning-checkpoints"
INTERLEAVE_RAW_MOUNT = "/rl-checkpoints"
INTERLEAVE_HF_MOUNT = "/pretrain-checkpoints"
MILES_CONVERTER_SHA256 = (
    "610f84d754892b32a80dffaea36793913bc48fbad4aa87f0d6f96083757a7489"
)
INTERLEAVE_QUEUE_SHA256 = (
    "e9edf1f090ca84711c95c62c30c750a67349f5af6be9d5fe74b912cfd4dcc7db"
)
INTERLEAVE_DASHBOARD_SCHEMA_SHA256 = (
    "c56ec7691267e6705a18d562b2f0e6abb509679f3c418b5b5dbd07d10e2618e2"
)
INTERLEAVE_EXP4_QUEUE_SHA256 = (
    "d18aa831fecfb048da0a2652b85ff327589634f96136d81d229f3fffaad4b567"
)
INTERLEAVE_LIVE_SCHEMA_SHA256 = (
    "b410fb04bda1e8da3ae9d317552b587a10e41da4971b6af5e637244a40fbb5a3"
)
HF_PREFIX = "miles_sglang_grpo_r6"
EVAL_INTERVAL = 40

UPSTREAM_GIT_SHA = "40f04428a0a446ca319c8429bda8c0cff15b5e5a"
EVAL_CODE_SHA256 = {
    "verl_eval.sh": "9fb799d4078a073b5e08f09b11caac93a18c2facd9cb726336ac85a18eb6136c",
    "ray_trainer.py": "f79680682fc5ef8fef2d3a71500953c26a37b5b8853858743128be20a394c894",
    "fsdp_workers.py": "099f0996a7ac1f11fe5d1650c9ae4ab73fae665767909664e065d843a525381a",
}
DATA_SHA256 = {
    "test_B1_multi_turn": "3ac5df0af21b395c23f864dd75b6a64335e3fe681c2b774f1485b276c6893c78",
    "test_B2_multi_turn": "9b315fe82a676b9b817ae77f96f7987be04ab34ec18513e3d42544896a133c3f",
    "test_B3_multi_turn": "8e41e0cf7c17babf6ae9a17a5b51607eef5674788dd09042e7dbbf90a945a5b9",
    "test_B4_multi_turn": "9583e4f6621ffee456eefc3e9d9de15800ec24226d20b882ff4805e82c4a985b",
    "test_B5_multi_turn": "927d62a4994d39e61ffb6719f85961ba14dbd55f365c539477fe6db72288c5cc",
}
DATASETS = list(DATA_SHA256)

RUNS = {
    "C6p5e18_32m_alpha0.200_beta0.013": {
        "repo_id": "Pre-to-Post-2/rl_C6p5e18_32m_alpha0.200_beta0.013",
        "target_step": 4000,
    },
    "C6p5e18_32m_alpha0.400_beta0.013": {
        "repo_id": "Pre-to-Post-2/rl_C6p5e18_32m_alpha0.400_beta0.013",
        "target_step": 4000,
    },
    "C6p5e18_410m_alpha0.750_beta0.148": {
        "repo_id": "Pre-to-Post-2/rl_C6p5e18_410m_alpha0.750_beta0.148",
        "target_step": 3000,
    },
    "C6p5e18_410m_alpha1.000_beta0.148": {
        "repo_id": "Pre-to-Post-2/rl_C6p5e18_410m_alpha1.000_beta0.148",
        "target_step": 3000,
    },
}

PRODUCTION_SETTINGS = {
    "datasets": DATASETS,
    "response_length": 2560,
    "temperature": 1,
    "n_samples": 16,
    "rollout": "vllm",
    "model_impl": "vllm",
    "multi_turn": True,
    "thinking": True,
    "tts": False,
    "seed": 0,
    "max_num_seqs": 2048,
    "max_num_batched_tokens": 131072,
    "gpu_memory": 0.85,
    "enforce_eager": False,
    "free_cache_engine": True,
}

PILOT_SETTINGS = {
    **PRODUCTION_SETTINGS,
    "datasets": ["test_B1_multi_turn"],
}

SMOKE_SETTINGS = {
    **PRODUCTION_SETTINGS,
    "datasets": ["test_B1_smoke"],
    "response_length": 128,
    "n_samples": 1,
    "max_num_seqs": 32,
    "max_num_batched_tokens": 8192,
}


def _fingerprint(settings: dict) -> str:
    payload = {
        "schema": 1,
        "upstream_git_sha": UPSTREAM_GIT_SHA,
        "eval_code_sha256": EVAL_CODE_SHA256,
        "data_sha256": DATA_SHA256,
        "settings": settings,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load_interleave_registry() -> dict[str, Any]:
    registry_path = next(
        (
            path
            for path in (
                INTERLEAVE_REGISTRY_REMOTE,
                INTERLEAVE_REGISTRY_LOCAL,
            )
            if path.is_file()
        ),
        None,
    )
    if registry_path is None:
        raise FileNotFoundError(
            "INTERLEAVED_CORE_REGISTRY.json was not packaged with the evaluator"
        )
    return json.loads(registry_path.read_text())


def _interleave_fingerprint(
    settings: dict[str, Any],
    stages: list[dict[str, Any]],
) -> str:
    topology = [
        {
            key: stage[key]
            for key in (
                "experiment_version",
                "model",
                "experiment",
                "arm",
                "filter",
                "phase",
                "run_name",
                "target_step",
                "effective_step_offset",
            )
        }
        for stage in stages
    ]
    payload = {
        "schema": 1,
        "namespace": INTERLEAVE_NAMESPACE,
        "upstream_git_sha": UPSTREAM_GIT_SHA,
        "eval_code_sha256": EVAL_CODE_SHA256,
        "data_sha256": DATA_SHA256,
        "miles_converter_sha256": MILES_CONVERTER_SHA256,
        "interleave_queue_sha256": INTERLEAVE_QUEUE_SHA256,
        "interleave_dashboard_schema_sha256": (
            INTERLEAVE_DASHBOARD_SCHEMA_SHA256
        ),
        "settings": settings,
        "topology": topology,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


INTERLEAVE_REGISTRY = _load_interleave_registry()
INTERLEAVE_STAGES = flatten_interleave_eval_registry(INTERLEAVE_REGISTRY)
INTERLEAVE_RUNS = {stage["run_name"]: stage for stage in INTERLEAVE_STAGES}

PRODUCTION_FINGERPRINT = _fingerprint(PRODUCTION_SETTINGS)
PILOT_FINGERPRINT = _fingerprint(PILOT_SETTINGS)
SMOKE_FINGERPRINT = _fingerprint(SMOKE_SETTINGS)
INTERLEAVE_PRODUCTION_FINGERPRINT = _interleave_fingerprint(
    PRODUCTION_SETTINGS, INTERLEAVE_STAGES
)
INTERLEAVE_PILOT_FINGERPRINT = _interleave_fingerprint(
    PILOT_SETTINGS, INTERLEAVE_STAGES
)
INTERLEAVE_SMOKE_FINGERPRINT = _interleave_fingerprint(
    SMOKE_SETTINGS, INTERLEAVE_STAGES
)


def _exp4_stage_fingerprint(
    settings: dict[str, Any],
    stage: dict[str, Any],
) -> str:
    """Return a stable stage-scoped fingerprint for a live Exp4 RL run."""

    topology_keys = (
        "experiment_version",
        "exp4_version",
        "model",
        "experiment",
        "arm",
        "filter",
        "filter_mode",
        "method",
        "phase",
        "stage_id",
        "run_name",
        "target_step",
        "effective_step_offset",
        "rollout_seed",
        "dynamic_filter",
        "conversion_origin_hf",
        "origin_hf_sha256",
        "method_plan_sha256",
        "call_contract_sha256",
    )
    payload = {
        "schema": "interleave-exp4-stage-eval-fingerprint-v1",
        "namespace": INTERLEAVE_NAMESPACE,
        "upstream_git_sha": UPSTREAM_GIT_SHA,
        "eval_code_sha256": EVAL_CODE_SHA256,
        "data_sha256": DATA_SHA256,
        "miles_converter_sha256": MILES_CONVERTER_SHA256,
        "core_queue_sha256": INTERLEAVE_QUEUE_SHA256,
        "exp4_queue_sha256": INTERLEAVE_EXP4_QUEUE_SHA256,
        "live_schema_sha256": INTERLEAVE_LIVE_SCHEMA_SHA256,
        "dashboard_schema_sha256": INTERLEAVE_DASHBOARD_SCHEMA_SHA256,
        "settings": settings,
        "stage": {key: stage[key] for key in topology_keys},
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("curl", "git", "vim", "htop")
    .pip_install(
        "wheel==0.46.3",
        "packaging==24.1",
        "ninja==1.13.0",
        "setuptools==71.1.0",
    )
    .pip_install(
        "torch==2.6.0",
        "torchvision==0.21.0",
        "torchaudio==2.6.0",
        "torchdata==0.11.0",
        "triton==3.2.0",
        "accelerate==1.13.0",
        "transformers==4.51.3",
        "tokenizers==0.21.4",
        "datasets==4.8.4",
        "huggingface_hub==0.36.2",
        "safetensors==0.7.0",
        "sentencepiece==0.2.1",
        "pyarrow==23.0.1",
        "pandas==2.3.3",
        "numpy==2.2.6",
        "scipy==1.17.1",
        "hydra-core==1.3.2",
        "omegaconf==2.3.0",
        "wandb==0.26.0",
        "mlflow==3.11.1",
        "tensordict==0.6.2",
        "ray[default]==2.47.1",
        "vllm==0.8.5",
        "xformers==0.0.29.post2",
        "peft==0.19.1",
        "dill==0.4.1",
        "codetiming==1.4.0",
        "pylatexenc==2.10",
        "pybind11==3.0.4",
        "chess==1.11.2",
        "pydantic==2.13.3",
        "openai==2.32.0",
        "compressed-tensors==0.9.3",
        "xgrammar==0.1.18",
        "outlines==0.1.11",
        "lm-format-enforcer==0.10.12",
    )
    .pip_install("flash-attn==2.7.4.post1", extra_options="--no-build-isolation")
    .add_local_file(
        str(INTERLEAVE_REGISTRY_LOCAL),
        remote_path=str(INTERLEAVE_REGISTRY_REMOTE),
        copy=True,
    )
    .add_local_file(
        str(INTERLEAVE_QUEUE_LOCAL),
        remote_path=INTERLEAVE_QUEUE_REMOTE,
        copy=True,
    )
    .add_local_file(
        str(INTERLEAVE_DASHBOARD_SCHEMA_LOCAL),
        remote_path=INTERLEAVE_DASHBOARD_SCHEMA_REMOTE,
        copy=True,
    )
    .add_local_file(
        str(INTERLEAVE_EXP4_QUEUE_LOCAL),
        remote_path=INTERLEAVE_EXP4_QUEUE_REMOTE,
        copy=True,
    )
    .add_local_file(
        str(INTERLEAVE_LIVE_SCHEMA_LOCAL),
        remote_path=INTERLEAVE_LIVE_SCHEMA_REMOTE,
        copy=True,
    )
    .add_local_file(
        str(MILES_CONVERTER_LOCAL),
        remote_path=MILES_CONVERTER_REMOTE,
        copy=True,
    )
    # Non-copy local mounts must be last: any later image build step would
    # make Modal reject the image definition.
    .add_local_dir(str(VERL_ROOT), remote_path=REMOTE_VERL_ROOT)
    .add_local_dir(str(EVAL_DATA_ROOT), remote_path=REMOTE_EVAL_DATA)
)
control_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub==0.36.2")
    .add_local_file(
        str(INTERLEAVE_REGISTRY_LOCAL),
        remote_path=str(INTERLEAVE_REGISTRY_REMOTE),
        copy=True,
    )
    .add_local_file(
        str(INTERLEAVE_QUEUE_LOCAL),
        remote_path=INTERLEAVE_QUEUE_REMOTE,
        copy=True,
    )
    .add_local_file(
        str(INTERLEAVE_DASHBOARD_SCHEMA_LOCAL),
        remote_path=INTERLEAVE_DASHBOARD_SCHEMA_REMOTE,
        copy=True,
    )
    .add_local_file(
        str(INTERLEAVE_EXP4_QUEUE_LOCAL),
        remote_path=INTERLEAVE_EXP4_QUEUE_REMOTE,
        copy=True,
    )
    .add_local_file(
        str(INTERLEAVE_LIVE_SCHEMA_LOCAL),
        remote_path=INTERLEAVE_LIVE_SCHEMA_REMOTE,
        copy=True,
    )
)

results_volume = modal.Volume.from_name(RESULTS_VOLUME_NAME, create_if_missing=True)
interleave_raw_volume = modal.Volume.from_name(
    INTERLEAVE_RAW_VOLUME_NAME, create_if_missing=False
)
interleave_hf_volume = modal.Volume.from_name(
    INTERLEAVE_HF_VOLUME_NAME, create_if_missing=False
)
app = modal.App(
    APP_NAME,
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _load_live_interleave_registry(
    pointer: Path = INTERLEAVE_LIVE_FEED_POINTER,
) -> dict[str, Any]:
    """Load and content-verify the registry published by the autopilot."""

    if not pointer.is_file():
        raise FileNotFoundError(f"live interleave feed is missing: {pointer}")
    feed = validate_live_feed(json.loads(pointer.read_text()))
    version = (
        pointer.parent / "feeds" / f"{feed['payload_sha256']}.json"
    )
    if not version.is_file():
        raise ValueError("content-addressed live interleave feed is missing")
    version_feed = validate_live_feed(json.loads(version.read_text()))
    if version_feed["payload_sha256"] != feed["payload_sha256"]:
        raise ValueError("live interleave pointer/version digest mismatch")
    registry = feed["registry"]
    if not isinstance(registry, dict):
        raise ValueError("live interleave registry is not an object")
    return registry


def _current_interleave_stages(
    pointer: Path = INTERLEAVE_LIVE_FEED_POINTER,
) -> tuple[list[dict[str, Any]], int]:
    """Merge immutable core stages with fully registered live Exp4 stages."""

    registry = _load_live_interleave_registry(pointer)
    exp4_stages = flatten_exp4_rl_registry(registry)
    core_run_names = set(INTERLEAVE_RUNS)
    collisions = sorted(
        stage["run_name"]
        for stage in exp4_stages
        if stage["run_name"] in core_run_names
    )
    if collisions:
        raise ValueError(
            "Exp4 RL run names collide with immutable core runs: "
            + ", ".join(collisions)
        )
    exp4 = registry.get("exp4", {})
    arms = exp4.get("arms", []) if isinstance(exp4, dict) else []
    if not isinstance(arms, list):
        raise ValueError("registry.exp4.arms must be a list")
    return INTERLEAVE_STAGES + exp4_stages, len(arms)


def _resolve_interleave_stage(run_name: str) -> dict[str, Any]:
    if run_name in INTERLEAVE_RUNS:
        return INTERLEAVE_RUNS[run_name]
    stages, _ = _current_interleave_stages()
    matches = [stage for stage in stages if stage["run_name"] == run_name]
    if len(matches) != 1:
        raise ValueError(f"Unknown or ambiguous interleave run: {run_name}")
    return matches[0]


def _run_root(run_key: str, step: int, profile: str, fingerprint: str) -> Path:
    return (
        Path(RESULTS_ROOT)
        / "v1"
        / run_key
        / f"global_step_{step}"
        / f"{profile}_{fingerprint[:12]}"
    )


def _model_etag(repo_id: str, step: int) -> str:
    from huggingface_hub import get_hf_file_metadata, hf_hub_url

    filename = f"{HF_PREFIX}/global_step_{step}/model.safetensors"
    metadata = get_hf_file_metadata(hf_hub_url(repo_id=repo_id, filename=filename))
    return str(metadata.etag)


def _valid_success(path: Path, fingerprint: str, model_etag: str) -> bool:
    if not path.exists():
        return False
    try:
        marker = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (
        marker.get("fingerprint") == fingerprint
        and marker.get("model_etag") == model_etag
        and marker.get("actual_rows") == marker.get("expected_rows")
    )


def _interleave_run_root(
    run_name: str,
    step: int,
    profile: str,
    fingerprint: str,
) -> Path:
    return (
        Path(RESULTS_ROOT)
        / INTERLEAVE_NAMESPACE
        / run_name
        / f"global_step_{step}"
        / f"{profile}_{fingerprint[:12]}"
    )


def _profile_settings(
    profile: str,
    *,
    interleave: bool,
) -> tuple[dict[str, Any], str]:
    if profile == "production":
        return (
            PRODUCTION_SETTINGS,
            (
                INTERLEAVE_PRODUCTION_FINGERPRINT
                if interleave
                else PRODUCTION_FINGERPRINT
            ),
        )
    if profile == "b1_pilot":
        return (
            PILOT_SETTINGS,
            (
                INTERLEAVE_PILOT_FINGERPRINT
                if interleave
                else PILOT_FINGERPRINT
            ),
        )
    if profile == "smoke":
        return (
            SMOKE_SETTINGS,
            (
                INTERLEAVE_SMOKE_FINGERPRINT
                if interleave
                else SMOKE_FINGERPRINT
            ),
        )
    raise ValueError(f"Unknown profile: {profile}")


def _interleave_stage_profile_settings(
    profile: str,
    stage: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    settings, legacy_fingerprint = _profile_settings(
        profile,
        interleave=True,
    )
    if stage.get("experiment") == "E4":
        return settings, _exp4_stage_fingerprint(settings, stage)
    return settings, legacy_fingerprint


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _raw_checkpoint_identity(
    source: Path,
    *,
    run_name: str,
    step: int,
) -> str:
    """Build a stable identity after validating a complete Miles checkpoint."""

    required = ("model", "optimizer", "lr_scheduler", "rng.pt", "meta.json")
    missing = [name for name in required if not (source / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Incomplete Miles checkpoint {source}: missing {', '.join(missing)}"
        )
    try:
        metadata = json.loads((source / "meta.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid checkpoint meta.json: {source}") from exc
    if int(metadata.get("iteration", -1)) != step:
        raise RuntimeError(
            f"Checkpoint iteration mismatch at {source}: "
            f"{metadata.get('iteration')} != {step}"
        )

    dist_metadata = source / "model" / ".metadata"
    if not dist_metadata.is_file():
        raise FileNotFoundError(
            f"Missing distributed model metadata: {dist_metadata}"
        )
    model_files = [
        {
            "path": str(path.relative_to(source / "model")),
            "bytes": path.stat().st_size,
        }
        for path in sorted((source / "model").rglob("*"))
        if path.is_file()
    ]
    payload = {
        "schema": 1,
        "run_name": run_name,
        "step": step,
        "meta_sha256": _sha256_file(source / "meta.json"),
        "dist_metadata_sha256": _sha256_file(dist_metadata),
        "model_files": model_files,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _valid_interleave_success(
    path: Path,
    *,
    fingerprint: str,
    checkpoint_identity: str,
) -> bool:
    if not path.exists():
        return False
    try:
        marker = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    table = marker.get("table_metrics")
    required_metrics = (
        "pass_at_1",
        "avg_reward",
        "b3_avg",
        "b4_avg",
        "b3_b4_avg",
    )
    return (
        marker.get("namespace") == INTERLEAVE_NAMESPACE
        and marker.get("fingerprint") == fingerprint
        and marker.get("checkpoint_identity") == checkpoint_identity
        and marker.get("actual_rows") == marker.get("expected_rows")
        and isinstance(table, dict)
        and all(isinstance(table.get(key), (int, float)) for key in required_metrics)
    )


@app.function(
    gpu="H200",
    cpu=16.0,
    memory=300 * 1024,
    timeout=60 * 60 * 8,
    retries=modal.Retries(initial_delay=5.0, max_retries=2),
    max_containers=128,
    volumes={RESULTS_ROOT: results_volume},
)
def eval_one(
    run_key: str,
    step: int,
    profile: str = "production",
) -> dict:
    """Evaluate one HF-format checkpoint and persist an auditable result."""
    from huggingface_hub import snapshot_download

    if run_key not in RUNS:
        raise ValueError(f"Unknown run: {run_key}")
    if profile not in {"production", "b1_pilot", "smoke"}:
        raise ValueError(f"Unknown profile: {profile}")

    spec = RUNS[run_key]
    repo_id = spec["repo_id"]
    if profile == "production":
        settings = PRODUCTION_SETTINGS
        fingerprint = PRODUCTION_FINGERPRINT
    elif profile == "b1_pilot":
        settings = PILOT_SETTINGS
        fingerprint = PILOT_FINGERPRINT
    else:
        settings = SMOKE_SETTINGS
        fingerprint = SMOKE_FINGERPRINT
    result_root = _run_root(run_key, step, profile, fingerprint)
    success_path = result_root / "_SUCCESS.json"
    failure_path = result_root / "_FAILED.json"

    results_volume.reload()
    model_etag = _model_etag(repo_id, step)
    if _valid_success(success_path, fingerprint, model_etag):
        print(f"[skip] already complete: {run_key} step {step} ({profile})")
        return json.loads(success_path.read_text())

    result_root.mkdir(parents=True, exist_ok=True)
    started_at = _utc_now()
    started_clock = time.monotonic()
    _atomic_json(
        result_root / "_RUNNING.json",
        {
            "run_key": run_key,
            "repo_id": repo_id,
            "step": step,
            "profile": profile,
            "fingerprint": fingerprint,
            "model_etag": model_etag,
            "started_at": started_at,
        },
    )
    results_volume.commit()

    checkpoint_download = Path("/tmp/rl_eval_checkpoint")
    if checkpoint_download.exists():
        shutil.rmtree(checkpoint_download)
    checkpoint_download.mkdir(parents=True)

    prefix = f"{HF_PREFIX}/global_step_{step}"
    print(f"[download] {repo_id}/{prefix}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(checkpoint_download),
        allow_patterns=[f"{prefix}/*", f"{prefix}/**"],
    )
    model_path = checkpoint_download / prefix
    if not (model_path / "config.json").exists() or not (model_path / "model.safetensors").exists():
        raise FileNotFoundError(f"Incomplete checkpoint download: {model_path}")

    eval_data_dir = REMOTE_EVAL_DATA
    # Four of the 1,484 raw rows exceed the evaluator's 512-token prompt cap
    # and are removed by data.filter_overlong_prompts=True.
    expected_prompts = 1480
    if profile == "b1_pilot":
        expected_prompts = 308
    if profile == "smoke":
        import pyarrow.parquet as pq

        smoke_dir = Path("/tmp/smoke_eval_data")
        if smoke_dir.exists():
            shutil.rmtree(smoke_dir)
        smoke_dir.mkdir(parents=True)
        source = Path(REMOTE_EVAL_DATA) / "test_B1_multi_turn.parquet"
        # VERL constructs a drop-last train loader even in val-only mode, so
        # keep one complete 64-row train batch in the smoke fixture.
        table = pq.read_table(source).slice(0, 64)
        pq.write_table(table, smoke_dir / "test_B1_smoke.parquet")
        eval_data_dir = str(smoke_dir)
        expected_prompts = 64

    output_dir = result_root / "output"
    # A forced requeue may restart a worker that was interrupted mid-write.
    # The old app must be stopped before requeueing, after which this partial
    # output is safe to replace. Terminal successes return above and are never
    # touched.
    if output_dir.exists():
        shutil.rmtree(output_dir)
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": f"{REMOTE_VERL_ROOT}:{env.get('PYTHONPATH', '')}",
            "PYTHONUNBUFFERED": "1",
            "CUDA_VISIBLE_DEVICES": "0",
            "GPUS": "0",
            "N_GPUS": "1",
            "MODEL_PATH": str(model_path),
            "OUTPUT_DIR": str(output_dir),
            "EXPERIMENT_NAME": "eval",
            "EVAL_DATA_DIR": eval_data_dir,
            "EVAL_DATASETS": ",".join(settings["datasets"]),
            "RES_LENGTH": str(settings["response_length"]),
            "TEMPERATURE": str(settings["temperature"]),
            "N_SAMPLES": str(settings["n_samples"]),
            "ROLLOUT_NAME": settings["rollout"],
            "VLLM_MODEL_IMPL": settings["model_impl"],
            "MULTI_TURN": str(settings["multi_turn"]),
            "THINKING": str(settings["thinking"]),
            "TTS": str(settings["tts"]),
            "SEED": str(settings["seed"]),
            "MAX_NUM_SEQS": str(settings["max_num_seqs"]),
            "MAX_NUM_BATCHED_TOKENS": str(settings["max_num_batched_tokens"]),
            "GPU_MEMORY": str(settings["gpu_memory"]),
            "MICRO_BATCH_SIZE": "32",
            "ENFORCE_EAGER": str(settings["enforce_eager"]),
            "FREE_CACHE_ENGINE": str(settings["free_cache_engine"]),
            "DEBUG": "False",
            "TOKENIZERS_PARALLELISM": "true",
            "NCCL_DEBUG": "WARN",
            "VLLM_LOGGING_LEVEL": "WARN",
        }
    )

    command = ["bash", f"{REMOTE_VERL_ROOT}/verl/eval_bash/verl_eval.sh"]
    print(f"[eval] {run_key} step {step} ({profile})")
    process = subprocess.Popen(
        command,
        cwd=REMOTE_VERL_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        sys.stdout.write(line)
    return_code = process.wait()

    generations = output_dir / "eval" / "generations" / "0.jsonl"
    metrics = output_dir / "eval" / "generations" / "metrics.json"
    actual_rows = 0
    if generations.exists():
        with generations.open() as handle:
            actual_rows = sum(1 for _ in handle)
    expected_rows = expected_prompts * int(settings["n_samples"])

    common = {
        "run_key": run_key,
        "repo_id": repo_id,
        "hf_prefix": HF_PREFIX,
        "step": step,
        "profile": profile,
        "fingerprint": fingerprint,
        "model_etag": model_etag,
        "upstream_git_sha": UPSTREAM_GIT_SHA,
        "settings": settings,
        "started_at": started_at,
        "finished_at": _utc_now(),
        "duration_seconds": round(time.monotonic() - started_clock, 3),
        "return_code": return_code,
        "expected_rows": expected_rows,
        "actual_rows": actual_rows,
        "generations": str(generations),
        "metrics": str(metrics),
    }

    if return_code == 0 and actual_rows == expected_rows and metrics.exists():
        _atomic_json(success_path, common)
        failure_path.unlink(missing_ok=True)
        (result_root / "_RUNNING.json").unlink(missing_ok=True)
        results_volume.commit()
        print(f"[success] {run_key} step {step}: {actual_rows} rows")
        return common

    _atomic_json(
        failure_path,
        {
            **common,
            "error": "evaluation failed or output validation did not match",
            "metrics_exists": metrics.exists(),
        },
    )
    results_volume.commit()
    raise RuntimeError(
        f"{run_key} step {step} failed: rc={return_code}, "
        f"rows={actual_rows}/{expected_rows}, metrics={metrics.exists()}"
    )


@app.function(
    gpu="H200",
    cpu=16.0,
    memory=300 * 1024,
    timeout=60 * 60 * 8,
    retries=modal.Retries(initial_delay=5.0, max_retries=2),
    max_containers=128,
    volumes={
        RESULTS_ROOT: results_volume,
        INTERLEAVE_RAW_MOUNT: interleave_raw_volume,
        INTERLEAVE_HF_MOUNT: interleave_hf_volume,
    },
)
def eval_interleave_one(
    run_name: str,
    step: int,
    profile: str = "production",
) -> dict[str, Any]:
    """Convert and evaluate one raw Miles checkpoint from an interleave phase."""

    results_volume.reload()
    interleave_raw_volume.reload()
    interleave_hf_volume.reload()
    stage = _resolve_interleave_stage(run_name)
    settings, fingerprint = _interleave_stage_profile_settings(profile, stage)
    if profile == "production" and step not in cadence_steps(
        int(stage["target_step"])
    ):
        raise ValueError(
            f"{run_name} step {step} is not an in-range {EVAL_INTERVAL}-step "
            "production checkpoint"
        )

    source = Path(stage["raw_checkpoint_root"]) / f"iter_{step:07d}"
    origin_hf = Path(stage["conversion_origin_hf"])
    if not origin_hf.is_dir():
        raise FileNotFoundError(
            f"Missing conversion origin for {run_name}: {origin_hf}"
        )
    checkpoint_identity = _raw_checkpoint_identity(
        source, run_name=run_name, step=step
    )

    result_root = _interleave_run_root(
        run_name, step, profile, fingerprint
    )
    success_path = result_root / "_SUCCESS.json"
    failure_path = result_root / "_FAILED.json"
    running_path = result_root / "_RUNNING.json"
    if _valid_interleave_success(
        success_path,
        fingerprint=fingerprint,
        checkpoint_identity=checkpoint_identity,
    ):
        print(
            f"[skip] interleave already complete: {run_name} step {step} "
            f"({profile})"
        )
        return json.loads(success_path.read_text())

    result_root.mkdir(parents=True, exist_ok=True)
    success_path.unlink(missing_ok=True)
    failure_path.unlink(missing_ok=True)
    started_at = _utc_now()
    started_unix = time.time()
    started_clock = time.monotonic()
    stage_fields = {
        key: stage[key]
        for key in (
            "experiment_version",
            "model",
            "experiment",
            "arm",
            "filter",
            "filter_mode",
            "phase",
            "run_name",
            "target_step",
            "effective_step_offset",
        )
    }
    stage_fields.update(
        {
            key: stage[key]
            for key in (
                "exp4_version",
                "method",
                "stage_id",
                "origin_hf_sha256",
                "method_plan_sha256",
                "call_contract_sha256",
            )
            if key in stage
        }
    )
    running_marker = {
        "namespace": INTERLEAVE_NAMESPACE,
        **stage_fields,
        "step": step,
        "effective_rl_step": int(stage["effective_step_offset"]) + step,
        "profile": profile,
        "fingerprint": fingerprint,
        "checkpoint_identity": checkpoint_identity,
        "raw_checkpoint": str(source),
        "conversion_origin_hf": str(origin_hf),
        "conversion_origin_fallback": stage["conversion_origin_fallback"],
        "started_at": started_at,
        "unix_time": started_unix,
    }
    _atomic_json(running_path, running_marker)
    results_volume.commit()

    actual_rows = 0
    expected_prompts = 1480
    expected_rows = expected_prompts * int(settings["n_samples"])
    return_code: int | None = None
    generations: Path | None = None
    metrics_path: Path | None = None
    table: dict[str, Any] | None = None
    try:
        scratch = Path("/tmp/interleave_rl_eval_checkpoint")
        if scratch.exists():
            shutil.rmtree(scratch)
        model_path = scratch / "hf"
        scratch.mkdir(parents=True)
        convert_command = [
            sys.executable,
            MILES_CONVERTER_REMOTE,
            "--input-dir",
            str(source),
            "--origin-hf-dir",
            str(origin_hf),
            "--output-dir",
            str(model_path),
            "--force",
        ]
        print(
            f"[convert] {run_name} step {step}: {source} -> {model_path}",
            flush=True,
        )
        conversion = subprocess.run(
            convert_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if conversion.stdout:
            print(conversion.stdout[-20000:], flush=True)
        if conversion.returncode:
            raise RuntimeError(
                f"FSDP-to-HF conversion failed with exit "
                f"{conversion.returncode}"
            )
        if not (model_path / "config.json").is_file() or not list(
            model_path.glob("*.safetensors")
        ):
            raise FileNotFoundError(
                f"Conversion produced an incomplete HF checkpoint: {model_path}"
            )

        eval_data_dir = REMOTE_EVAL_DATA
        if profile == "b1_pilot":
            expected_prompts = 308
        elif profile == "smoke":
            import pyarrow.parquet as pq

            smoke_dir = Path("/tmp/smoke_eval_data")
            if smoke_dir.exists():
                shutil.rmtree(smoke_dir)
            smoke_dir.mkdir(parents=True)
            source_data = (
                Path(REMOTE_EVAL_DATA) / "test_B1_multi_turn.parquet"
            )
            table_data = pq.read_table(source_data).slice(0, 64)
            pq.write_table(table_data, smoke_dir / "test_B1_smoke.parquet")
            eval_data_dir = str(smoke_dir)
            expected_prompts = 64
        expected_rows = expected_prompts * int(settings["n_samples"])

        output_dir = result_root / "output"
        if output_dir.exists():
            shutil.rmtree(output_dir)
        env = os.environ.copy()
        env.update(
            {
                "PYTHONPATH": (
                    f"{REMOTE_VERL_ROOT}:{env.get('PYTHONPATH', '')}"
                ),
                "PYTHONUNBUFFERED": "1",
                "CUDA_VISIBLE_DEVICES": "0",
                "GPUS": "0",
                "N_GPUS": "1",
                "MODEL_PATH": str(model_path),
                "OUTPUT_DIR": str(output_dir),
                "EXPERIMENT_NAME": "eval",
                "EVAL_DATA_DIR": eval_data_dir,
                "EVAL_DATASETS": ",".join(settings["datasets"]),
                "RES_LENGTH": str(settings["response_length"]),
                "TEMPERATURE": str(settings["temperature"]),
                "N_SAMPLES": str(settings["n_samples"]),
                "ROLLOUT_NAME": settings["rollout"],
                "VLLM_MODEL_IMPL": settings["model_impl"],
                "MULTI_TURN": str(settings["multi_turn"]),
                "THINKING": str(settings["thinking"]),
                "TTS": str(settings["tts"]),
                "SEED": str(settings["seed"]),
                "MAX_NUM_SEQS": str(settings["max_num_seqs"]),
                "MAX_NUM_BATCHED_TOKENS": str(
                    settings["max_num_batched_tokens"]
                ),
                "GPU_MEMORY": str(settings["gpu_memory"]),
                "MICRO_BATCH_SIZE": "32",
                "ENFORCE_EAGER": str(settings["enforce_eager"]),
                "FREE_CACHE_ENGINE": str(settings["free_cache_engine"]),
                "DEBUG": "False",
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "WARN",
            }
        )
        command = ["bash", f"{REMOTE_VERL_ROOT}/verl/eval_bash/verl_eval.sh"]
        print(f"[eval] interleave {run_name} step {step} ({profile})")
        process = subprocess.Popen(
            command,
            cwd=REMOTE_VERL_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
        return_code = process.wait()

        generations = output_dir / "eval" / "generations" / "0.jsonl"
        metrics_path = (
            output_dir / "eval" / "generations" / "metrics.json"
        )
        if generations.exists():
            with generations.open() as handle:
                actual_rows = sum(1 for _ in handle)
        if return_code:
            raise RuntimeError(f"VERL evaluation exited {return_code}")
        if actual_rows != expected_rows:
            raise RuntimeError(
                f"Evaluation row mismatch: {actual_rows}/{expected_rows}"
            )
        if not metrics_path.is_file():
            raise FileNotFoundError(
                f"Evaluation did not write metrics: {metrics_path}"
            )
        metrics_payload = json.loads(metrics_path.read_text())
        table = final_table_metrics(metrics_payload, stage, step)

        common = {
            **running_marker,
            "finished_at": _utc_now(),
            "duration_seconds": round(time.monotonic() - started_clock, 3),
            "return_code": return_code,
            "expected_rows": expected_rows,
            "actual_rows": actual_rows,
            "generations": str(generations),
            "metrics": str(metrics_path),
            "table_metrics": table,
        }
        _atomic_json(success_path, common)
        failure_path.unlink(missing_ok=True)
        running_path.unlink(missing_ok=True)
        results_volume.commit()
        print(
            f"[success] interleave {run_name} step {step}: "
            f"{actual_rows} rows, pass@1={table['pass_at_1']:.6f}, "
            f"avg_reward={table['avg_reward']:.6f}, "
            f"B3-B4={table['b3_b4_avg']:.6f}"
        )
        return common
    except Exception as exc:
        failure = {
            **running_marker,
            "finished_at": _utc_now(),
            "duration_seconds": round(time.monotonic() - started_clock, 3),
            "return_code": return_code,
            "expected_rows": expected_rows,
            "actual_rows": actual_rows,
            "generations": str(generations) if generations else None,
            "metrics": str(metrics_path) if metrics_path else None,
            "table_metrics": table,
            "error": f"{type(exc).__name__}: {exc}",
        }
        _atomic_json(failure_path, failure)
        running_path.unlink(missing_ok=True)
        results_volume.commit()
        raise


def _discover_steps(repo_id: str) -> list[int]:
    from huggingface_hub import HfApi

    api = HfApi()
    pattern = re.compile(rf"^{re.escape(HF_PREFIX)}/global_step_(\d+)$")
    steps: list[int] = []
    for entry in api.list_repo_tree(
        repo_id=repo_id,
        path_in_repo=HF_PREFIX,
        recursive=False,
        expand=False,
    ):
        match = pattern.match(entry.path)
        if match:
            steps.append(int(match.group(1)))
    return sorted(set(steps))


def _lease_is_fresh(path: Path, ttl_seconds: int) -> bool:
    if not path.exists():
        return False
    try:
        timestamp = float(json.loads(path.read_text())["unix_time"])
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False
    return time.time() - timestamp < ttl_seconds


@app.function(
    image=control_image,
    cpu=1.0,
    memory=2048,
    timeout=60 * 60 * 47,
    retries=modal.Retries(initial_delay=10.0, max_retries=3),
    volumes={RESULTS_ROOT: results_volume},
)
def watch_and_enqueue(
    poll_seconds: int = 120,
    max_hours: int = 72,
    lease_hours: int = 12,
) -> dict:
    """Discover uploaded checkpoints, enqueue each once, and watch for new ones."""
    deadline = time.monotonic() + max_hours * 3600
    submitted_this_call: set[tuple[str, int]] = set()
    all_expected_seen = False

    while time.monotonic() < deadline:
        results_volume.reload()
        discovered_total = 0
        enqueued_now = 0
        expected_total = 0
        all_expected_seen = True
        pending: list[tuple[str, int]] = []

        for run_key, spec in RUNS.items():
            all_steps = _discover_steps(spec["repo_id"])
            steps = [step for step in all_steps if step % EVAL_INTERVAL == 0]
            discovered_total += len(steps)
            expected = set(
                range(EVAL_INTERVAL, int(spec["target_step"]) + 1, EVAL_INTERVAL)
            )
            expected_total += len(expected)
            if not expected.issubset(set(steps)):
                all_expected_seen = False

            for step in steps:
                task_key = (run_key, step)
                if task_key in submitted_this_call:
                    continue
                root = _run_root(run_key, step, "production", PRODUCTION_FINGERPRINT)
                success = root / "_SUCCESS.json"
                lease = root / "_QUEUED.json"
                if success.exists() or _lease_is_fresh(lease, lease_hours * 3600):
                    continue

                _atomic_json(
                    lease,
                    {
                        "run_key": run_key,
                        "step": step,
                        "profile": "production",
                        "fingerprint": PRODUCTION_FINGERPRINT,
                        "queued_at": _utc_now(),
                        "unix_time": time.time(),
                    },
                )
                pending.append(task_key)

        # Commit all leases as one durable transaction before any GPU jobs are
        # spawned. This avoids hundreds of serial Volume commits at startup.
        if pending:
            results_volume.commit()
        for run_key, step in pending:
            eval_one.spawn(run_key, step, "production")
            submitted_this_call.add((run_key, step))
            enqueued_now += 1

        print(
            f"[watch] discovered={discovered_total}/{expected_total} "
            f"enqueued_now={enqueued_now} submitted_session={len(submitted_this_call)} "
            f"all_expected_seen={all_expected_seen}"
        )
        if all_expected_seen:
            break
        time.sleep(max(30, poll_seconds))

    return {
        "submitted": len(submitted_this_call),
        "all_expected_seen": all_expected_seen,
        "finished_at": _utc_now(),
    }


def _discover_interleave_steps(stage: dict[str, Any]) -> list[int]:
    root = Path(stage["raw_checkpoint_root"])
    if not root.is_dir():
        return []
    allowed = set(cadence_steps(int(stage["target_step"])))
    steps: list[int] = []
    for entry in root.iterdir():
        step = parse_raw_checkpoint_step(entry.name)
        if step is None or step not in allowed or not entry.is_dir():
            continue
        required = (
            entry / "model" / ".metadata",
            entry / "optimizer",
            entry / "lr_scheduler",
            entry / "rng.pt",
            entry / "meta.json",
        )
        if all(path.exists() for path in required):
            steps.append(step)
    return sorted(set(steps))


@app.function(
    image=control_image,
    cpu=2.0,
    memory=4096,
    timeout=60 * 60 * 47,
    retries=modal.Retries(initial_delay=10.0, max_retries=3),
    volumes={
        RESULTS_ROOT: results_volume,
        INTERLEAVE_RAW_MOUNT: interleave_raw_volume,
    },
)
def watch_and_enqueue_interleave(
    poll_seconds: int = 120,
    max_hours: int = 72,
    lease_hours: int = 12,
) -> dict[str, Any]:
    """Watch immutable core plus live-registered Exp4 step-40 checkpoints."""

    deadline = time.monotonic() + max_hours * 3600
    submitted_this_call: set[tuple[str, int]] = set()
    all_expected_seen = False
    current_stages = list(INTERLEAVE_STAGES)

    while time.monotonic() < deadline:
        results_volume.reload()
        interleave_raw_volume.reload()
        try:
            current_stages, expected_exp4_stages = (
                _current_interleave_stages()
            )
        except Exception as exc:
            print(
                "[interleave-watch] live registry rejected; submitting "
                f"nothing this poll: {type(exc).__name__}: {exc}",
                flush=True,
            )
            time.sleep(max(30, poll_seconds))
            continue
        registered_exp4_stages = sum(
            stage.get("experiment") == "E4" for stage in current_stages
        )
        discovered_total = 0
        expected_total = 0
        enqueued_now = 0
        all_expected_seen = (
            registered_exp4_stages == expected_exp4_stages
        )
        pending: list[tuple[str, int]] = []

        for stage in current_stages:
            run_name = stage["run_name"]
            _, stage_fingerprint = _interleave_stage_profile_settings(
                "production",
                stage,
            )
            steps = _discover_interleave_steps(stage)
            discovered_total += len(steps)
            expected = set(cadence_steps(int(stage["target_step"])))
            expected_total += len(expected)
            if not expected.issubset(set(steps)):
                all_expected_seen = False

            for step in steps:
                task_key = (run_name, step)
                if task_key in submitted_this_call:
                    continue
                root = _interleave_run_root(
                    run_name,
                    step,
                    "production",
                    stage_fingerprint,
                )
                success = root / "_SUCCESS.json"
                running = root / "_RUNNING.json"
                queued = root / "_QUEUED.json"
                if _lease_is_fresh(
                    running, lease_hours * 3600
                ) or _lease_is_fresh(queued, lease_hours * 3600):
                    continue

                source = (
                    Path(stage["raw_checkpoint_root"])
                    / f"iter_{step:07d}"
                )
                identity = _raw_checkpoint_identity(
                    source, run_name=run_name, step=step
                )
                if _valid_interleave_success(
                    success,
                    fingerprint=stage_fingerprint,
                    checkpoint_identity=identity,
                ):
                    continue

                root.mkdir(parents=True, exist_ok=True)
                success.unlink(missing_ok=True)
                (root / "_FAILED.json").unlink(missing_ok=True)
                running.unlink(missing_ok=True)
                _atomic_json(
                    queued,
                    {
                        "namespace": INTERLEAVE_NAMESPACE,
                        "run_name": run_name,
                        "experiment": stage["experiment"],
                        "arm": stage["arm"],
                        "filter": stage["filter"],
                        "phase": stage["phase"],
                        "step": step,
                        "effective_rl_step": (
                            int(stage["effective_step_offset"]) + step
                        ),
                        "profile": "production",
                        "fingerprint": stage_fingerprint,
                        "checkpoint_identity": identity,
                        "queued_at": _utc_now(),
                        "unix_time": time.time(),
                    },
                )
                pending.append(task_key)

        # Lease every checkpoint durably before spawning any H200 job.  This
        # keeps concurrent/restarted watchers idempotent.
        if pending:
            results_volume.commit()
        for run_name, step in pending:
            eval_interleave_one.spawn(run_name, step, "production")
            submitted_this_call.add((run_name, step))
            enqueued_now += 1

        print(
            f"[interleave-watch] discovered={discovered_total}/{expected_total} "
            f"exp4_registered={registered_exp4_stages}/"
            f"{expected_exp4_stages} "
            f"enqueued_now={enqueued_now} "
            f"submitted_session={len(submitted_this_call)} "
            f"all_expected_seen={all_expected_seen}"
        )
        if all_expected_seen:
            break
        time.sleep(max(30, poll_seconds))

    return {
        "namespace": INTERLEAVE_NAMESPACE,
        "stage_count": len(current_stages),
        "submitted": len(submitted_this_call),
        "all_expected_seen": all_expected_seen,
        "core_fingerprint": INTERLEAVE_PRODUCTION_FINGERPRINT,
        "exp4_fingerprint_scope": "stage",
        "finished_at": _utc_now(),
    }


@app.function(
    image=control_image,
    cpu=2.0,
    memory=4096,
    timeout=60 * 30,
    volumes={
        RESULTS_ROOT: results_volume,
        INTERLEAVE_RAW_MOUNT: interleave_raw_volume,
    },
)
def status_interleave() -> dict[str, Any]:
    """Report raw-checkpoint and production-evaluation state for all phases."""

    results_volume.reload()
    interleave_raw_volume.reload()
    current_stages, expected_exp4_stages = _current_interleave_stages()
    report: dict[str, dict[str, Any]] = {}
    for stage in current_stages:
        run_name = stage["run_name"]
        _, stage_fingerprint = _interleave_stage_profile_settings(
            "production",
            stage,
        )
        steps = _discover_interleave_steps(stage)
        expected = cadence_steps(int(stage["target_step"]))
        root = (
            Path(RESULTS_ROOT) / INTERLEAVE_NAMESPACE / run_name
        )
        profile = (
            f"production_{stage_fingerprint[:12]}"
        )
        succeeded = sorted(
            root.glob(f"global_step_*/{profile}/_SUCCESS.json")
        )
        failed = sorted(root.glob(f"global_step_*/{profile}/_FAILED.json"))
        queued = sorted(root.glob(f"global_step_*/{profile}/_QUEUED.json"))
        running = sorted(root.glob(f"global_step_*/{profile}/_RUNNING.json"))
        latest_table = None
        if succeeded:
            latest_marker = max(
                succeeded,
                key=lambda path: int(
                    path.parents[1].name.removeprefix("global_step_")
                ),
            )
            try:
                latest_table = json.loads(
                    latest_marker.read_text()
                ).get("table_metrics")
            except (OSError, json.JSONDecodeError):
                latest_table = None
        report[run_name] = {
            "experiment": stage["experiment"],
            "arm": stage["arm"],
            "filter": stage["filter"],
            "phase": stage["phase"],
            "target_step": stage["target_step"],
            "effective_step_offset": stage["effective_step_offset"],
            "fingerprint": stage_fingerprint,
            "eval_interval": EVAL_INTERVAL,
            "expected_checkpoints": len(expected),
            "raw_checkpoints": len(steps),
            "raw_latest_step": max(steps) if steps else None,
            "succeeded": len(succeeded),
            "failed_markers": len(failed),
            "queued_markers": len(queued),
            "running_markers": len(running),
            "latest_table_metrics": latest_table,
        }
    return {
        "namespace": INTERLEAVE_NAMESPACE,
        "core_fingerprint": INTERLEAVE_PRODUCTION_FINGERPRINT,
        "exp4_fingerprint_scope": "stage",
        "worker_ceiling": 128,
        "stage_count": len(current_stages),
        "exp4_registered_stage_count": sum(
            stage.get("experiment") == "E4" for stage in current_stages
        ),
        "exp4_expected_stage_count": expected_exp4_stages,
        "checkpoint_count": sum(
            len(cadence_steps(int(stage["target_step"])))
            for stage in current_stages
        ),
        "runs": report,
    }


@app.function(
    image=control_image,
    cpu=1.0,
    memory=2048,
    timeout=60 * 30,
    volumes={RESULTS_ROOT: results_volume},
)
def status() -> dict:
    results_volume.reload()
    report: dict[str, dict] = {}
    for run_key, spec in RUNS.items():
        all_steps = _discover_steps(spec["repo_id"])
        steps = [step for step in all_steps if step % EVAL_INTERVAL == 0]
        root = Path(RESULTS_ROOT) / "v1" / run_key
        def selected(paths: list[Path]) -> list[Path]:
            return [
                path
                for path in paths
                if int(path.parents[1].name.removeprefix("global_step_"))
                % EVAL_INTERVAL
                == 0
            ]

        succeeded = selected(list(root.glob("global_step_*/production_*/_SUCCESS.json")))
        failed = selected(list(root.glob("global_step_*/production_*/_FAILED.json")))
        queued = selected(list(root.glob("global_step_*/production_*/_QUEUED.json")))
        running = selected(list(root.glob("global_step_*/production_*/_RUNNING.json")))
        report[run_key] = {
            "hf_checkpoints": len(steps),
            "hf_total_checkpoints": len(all_steps),
            "hf_latest_step": max(steps) if steps else None,
            "eval_interval": EVAL_INTERVAL,
            "target_step": spec["target_step"],
            "succeeded": len(succeeded),
            "failed_markers": len(failed),
            "queued_markers": len(queued),
            "running_markers": len(running),
        }
    return report


@app.local_entrypoint()
def main(
    mode: str = "status",
    run_key: str = "C6p5e18_32m_alpha0.200_beta0.013",
    step: int = 20,
    interleave_run_name: str = "core-e1-u-rl1-seed42",
    interleave_step: int = 40,
    poll_seconds: int = 120,
    max_hours: int = 47,
    lease_hours: int = 12,
) -> None:
    if mode == "smoke":
        print(json.dumps(eval_one.remote(run_key, step, "smoke"), indent=2, sort_keys=True))
    elif mode == "pilot":
        print(json.dumps(eval_one.remote(run_key, step, "b1_pilot"), indent=2, sort_keys=True))
    elif mode == "launch":
        call = watch_and_enqueue.spawn(poll_seconds, max_hours, lease_hours)
        print(f"watcher_call_id={call.object_id}")
    elif mode == "status":
        print(json.dumps(status.remote(), indent=2, sort_keys=True))
    elif mode == "interleave-dry-run":
        plan = build_interleave_dry_run_plan(INTERLEAVE_REGISTRY)
        plan.update(
            {
                "fingerprint": INTERLEAVE_PRODUCTION_FINGERPRINT,
                "worker_gpu": "H200",
                "worker_ceiling": 128,
                "settings": PRODUCTION_SETTINGS,
            }
        )
        print(json.dumps(plan, indent=2, sort_keys=True))
    elif mode == "interleave-smoke":
        print(
            json.dumps(
                eval_interleave_one.remote(
                    interleave_run_name, interleave_step, "smoke"
                ),
                indent=2,
                sort_keys=True,
            )
        )
    elif mode == "interleave-pilot":
        print(
            json.dumps(
                eval_interleave_one.remote(
                    interleave_run_name, interleave_step, "b1_pilot"
                ),
                indent=2,
                sort_keys=True,
            )
        )
    elif mode == "interleave-launch":
        call = watch_and_enqueue_interleave.spawn(
            poll_seconds, max_hours, lease_hours
        )
        print(f"interleave_watcher_call_id={call.object_id}")
    elif mode == "interleave-status":
        print(
            json.dumps(
                status_interleave.remote(), indent=2, sort_keys=True
            )
        )
    else:
        raise ValueError(
            "mode must be one of: smoke, pilot, launch, status, "
            "interleave-dry-run, interleave-smoke, interleave-pilot, "
            "interleave-launch, interleave-status"
        )
