"""Dependency-aware Modal evaluation for every interleaved pretrain endpoint.

This app is intentionally separate from both the frozen pretraining launcher
and the RL-checkpoint evaluator.  It watches for complete HF exports and runs:

1. deterministic held-out pretraining loss/perplexity/token accuracy;
2. an explicit unavailable result for same-source held-out masked SFT; and
3. the existing immutable B1--B5 chess evaluation.

Examples:
    modal run modal_eval_interleaved_endpoints.py --mode prep
    modal run modal_eval_interleaved_endpoints.py --mode dry-run
    modal run --detach modal_eval_interleaved_endpoints.py --mode launch
    modal run modal_eval_interleaved_endpoints.py --mode status
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
from functools import wraps
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal

try:
    from .interleave_endpoint_eval import (
        CHESS_DATA_SHA256,
        ENDPOINT_NAMESPACE,
        ENDPOINT_RESULT_SCHEMA,
        EXPERIMENT_VERSION,
        PT_HOLDOUT_RECORDS,
        PT_HOLDOUT_TARGET_TOKENS,
        SEQUENCE_LENGTH,
        V2_DATA_ARTIFACT_VERSION,
        V2_EXPERIMENT_VERSION,
        build_pt_holdout_manifest,
        build_sft_holdout_audit,
        build_v2_sft_holdout_audit,
        canonical_json,
        checkpoint_files,
        checkpoint_fingerprint,
        content_hash,
        discover_endpoints,
        safe_perplexity,
        sha256_file,
        summarize_chess_metrics,
        validate_pt_holdout_manifest,
        validate_sft_holdout_audit,
    )
except ImportError:
    from interleave_endpoint_eval import (
        CHESS_DATA_SHA256,
        ENDPOINT_NAMESPACE,
        ENDPOINT_RESULT_SCHEMA,
        EXPERIMENT_VERSION,
        PT_HOLDOUT_RECORDS,
        PT_HOLDOUT_TARGET_TOKENS,
        SEQUENCE_LENGTH,
        V2_DATA_ARTIFACT_VERSION,
        V2_EXPERIMENT_VERSION,
        build_pt_holdout_manifest,
        build_sft_holdout_audit,
        build_v2_sft_holdout_audit,
        canonical_json,
        checkpoint_files,
        checkpoint_fingerprint,
        content_hash,
        discover_endpoints,
        safe_perplexity,
        sha256_file,
        summarize_chess_metrics,
        validate_pt_holdout_manifest,
        validate_sft_holdout_audit,
    )


HERE = Path(__file__).resolve().parent
ENDPOINT_HELPER_LOCAL = HERE / "interleave_endpoint_eval.py"
ENDPOINT_HELPER_REMOTE = "/root/interleave_endpoint_eval.py"
V2_ENTRYPOINT_LOCAL = HERE / "modal_eval_interleaved_endpoints_v2.py"
V2_ENTRYPOINT_REMOTE = "/root/modal_eval_interleaved_endpoints_v2.py"
VERL_ROOT = HERE / "pre2post-chess" / "rl"
EVAL_DATA_ROOT = HERE / "test_data"
REMOTE_VERL_ROOT = "/root/verl"
REMOTE_EVAL_DATA = "/eval-data"

UPSTREAM_GIT_SHA = "40f04428a0a446ca319c8429bda8c0cff15b5e5a"
EVAL_CODE_SHA256 = {
    "verl_eval.sh": (
        "9fb799d4078a073b5e08f09b11caac93a18c2facd9cb726336ac85a18eb6136c"
    ),
    "ray_trainer.py": (
        "f79680682fc5ef8fef2d3a71500953c26a37b5b8853858743128be20a394c894"
    ),
    "fsdp_workers.py": (
        "099f0996a7ac1f11fe5d1650c9ae4ab73fae665767909664e065d843a525381a"
    ),
}
PRODUCTION_SETTINGS = {
    "datasets": list(CHESS_DATA_SHA256),
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

ENDPOINT_EVAL_PROFILE = os.environ.get(
    "CHESS_INTERLEAVE_ENDPOINT_EVAL_PROFILE", "v1"
).strip().lower()
if ENDPOINT_EVAL_PROFILE not in {"v1", "v2"}:
    raise ValueError(
        "CHESS_INTERLEAVE_ENDPOINT_EVAL_PROFILE must be 'v1' or 'v2'"
    )

DATA_VOLUME_NAME = "rl-reasoning-training-data"
CHECKPOINT_VOLUME_NAME = "rl-reasoning-checkpoints"
RESULTS_VOLUME_NAME = "chess-rl-eval-results-r6"
DATA_MOUNT = Path("/data")
CHECKPOINT_MOUNT = Path("/pretrain-checkpoints")
RESULTS_MOUNT = Path("/results")
SOURCE_ROOT = DATA_MOUNT / "pretrain_v1_20b"

if ENDPOINT_EVAL_PROFILE == "v1":
    APP_NAME = "chess-interleave-endpoint-eval"
    EXPERIMENT_VERSION = "mix10b_sft90k_3072_v1_20260730"
    ENDPOINT_NAMESPACE = "endpoint_v1"
    DATA_ARTIFACT_ROOT = (
        DATA_MOUNT / "50m_interleaved_mix10b_sft90k_v1"
    )
    EVAL_ARTIFACT_ROOT = DATA_ARTIFACT_ROOT / "endpoint_eval_v1"
    PT_HOLDOUT_PATH = EVAL_ARTIFACT_ROOT / "pt_holdout.json"
    SFT_AUDIT_PATH = EVAL_ARTIFACT_ROOT / "sft_holdout_audit.json"
    PT_HOLDOUT_REUSE_MARKER_PATH: Path | None = None
    REUSE_AUTHENTICATED_PT_HOLDOUT = False
    USE_V2_SFT_AUDIT = False
    INCLUDE_EXP4 = True
elif ENDPOINT_EVAL_PROFILE == "v2":
    APP_NAME = "chess-interleave-endpoint-eval-v2r1"
    EXPERIMENT_VERSION = V2_EXPERIMENT_VERSION
    ENDPOINT_NAMESPACE = "endpoint_v2r1_weighted_clean"
    DATA_ARTIFACT_ROOT = (
        DATA_MOUNT / f"50m_interleaved_{V2_DATA_ARTIFACT_VERSION}"
    )
    EVAL_ARTIFACT_ROOT = DATA_ARTIFACT_ROOT / "endpoint_eval_v2r1"
    # Reuse the already authenticated PT split by reference.  V2 prep only
    # reads and revalidates this artifact; it is never allowed to create or
    # modify anything under the v1 evaluator root.
    PT_HOLDOUT_PATH = (
        DATA_MOUNT
        / "50m_interleaved_mix10b_sft90k_v1"
        / "endpoint_eval_v1"
        / "pt_holdout.json"
    )
    SFT_AUDIT_PATH = EVAL_ARTIFACT_ROOT / "sft_holdout_audit.json"
    PT_HOLDOUT_REUSE_MARKER_PATH = (
        EVAL_ARTIFACT_ROOT / "pt_holdout_reuse.json"
    )
    REUSE_AUTHENTICATED_PT_HOLDOUT = True
    USE_V2_SFT_AUDIT = True
    # The current Exp4 root is owned by v1.  Scanning it here would relabel v1
    # checkpoints as v2, so v2 discovers only its collision-safe fixed paths.
    INCLUDE_EXP4 = False

SOURCE_MANIFEST_PATH = DATA_ARTIFACT_ROOT / "source_manifest.json"
TRAIN_SELECTION_PATH = DATA_ARTIFACT_ROOT / "pretrain_selection.json"
MANIFEST_SET_PATH = DATA_ARTIFACT_ROOT / "manifest_set.json"
P1_METADATA_PATH = DATA_ARTIFACT_ROOT / "legs/p1/metadata.json"
P2_METADATA_PATH = DATA_ARTIFACT_ROOT / "legs/p2/metadata.json"
P1_ORDER_PATH = DATA_ARTIFACT_ROOT / "legs/p1/order.npy"
P2_ORDER_PATH = DATA_ARTIFACT_ROOT / "legs/p2/order.npy"
SFT_CACHE_METADATA_PATH = DATA_ARTIFACT_ROOT / "sft_cache/metadata.json"

REMOTE_PROFILE_ENV_KEYS = {
    "profile": "CHESS_INTERLEAVE_ENDPOINT_EVAL_PROFILE",
    "app_name": "CHESS_INTERLEAVE_ENDPOINT_EVAL_APP_NAME",
    "namespace": "CHESS_INTERLEAVE_ENDPOINT_EVAL_NAMESPACE",
    "experiment_version": (
        "CHESS_INTERLEAVE_ENDPOINT_EVAL_EXPERIMENT_VERSION"
    ),
    "data_artifact_root": (
        "CHESS_INTERLEAVE_ENDPOINT_EVAL_DATA_ARTIFACT_ROOT"
    ),
    "results_root": "CHESS_INTERLEAVE_ENDPOINT_EVAL_RESULTS_ROOT",
    "source_sha256": "CHESS_INTERLEAVE_ENDPOINT_EVAL_SOURCE_SHA256",
}

LOSS_BATCH_SIZE = 64
LOSS_ATTENTION_BACKEND = "sdpa"
LOSS_TORCH_VERSION = "2.9.0"
LOSS_TRANSFORMERS_VERSION = "4.57.0"
LOSS_NUMPY_VERSION = "2.2.6"
WATCH_POLL_SECONDS = 120
QUEUE_LEASE_SECONDS = 4 * 60 * 60
RUNNING_LEASE_SECONDS = 10 * 60 * 60
FAILURE_RETRY_DELAY_SECONDS = 15 * 60
MAX_COMPONENT_ATTEMPTS = 4
WATCH_MAX_HOURS = 47
EXPECTED_FIXED_ENDPOINTS = frozenset(
    {"p1", "e2-final", "e3-p2", "e1-u-p2", "e1-d-p2"}
)
EXPECTED_EXP4_CELLS = (
    frozenset(
        (setting, method)
        for setting in ("U", "D")
        for method in ("hard-sft", "soft-kl", "scratch-replay")
    )
    if INCLUDE_EXP4
    else frozenset()
)
_SAFE_ENDPOINT_RE = re.compile(r"[a-z0-9][a-z0-9.-]{0,127}")


def _source_digest(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: str(item)):
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


ENDPOINT_EVAL_SOURCE_PATHS = [Path(__file__).resolve(), ENDPOINT_HELPER_LOCAL]
if ENDPOINT_EVAL_PROFILE == "v2":
    ENDPOINT_EVAL_SOURCE_PATHS.append(V2_ENTRYPOINT_LOCAL)
ENDPOINT_EVAL_COMMON_SOURCE_SHA256 = _source_digest(
    ENDPOINT_EVAL_SOURCE_PATHS
)
ENDPOINT_EVAL_SOURCE_SHA256 = hashlib.sha256(
    canonical_json(
        {
            "schema": 1,
            "common_source_sha256": ENDPOINT_EVAL_COMMON_SOURCE_SHA256,
            "profile": ENDPOINT_EVAL_PROFILE,
            "app_name": APP_NAME,
            "endpoint_namespace": ENDPOINT_NAMESPACE,
            "experiment_version": EXPERIMENT_VERSION,
            "data_artifact_root": str(DATA_ARTIFACT_ROOT),
            "pt_holdout_path": str(PT_HOLDOUT_PATH),
            "include_exp4": INCLUDE_EXP4,
        }
    )
).hexdigest()
REMOTE_PROFILE_CONTRACT = {
    "profile": ENDPOINT_EVAL_PROFILE,
    "app_name": APP_NAME,
    "namespace": ENDPOINT_NAMESPACE,
    "experiment_version": EXPERIMENT_VERSION,
    "data_artifact_root": str(DATA_ARTIFACT_ROOT),
    "results_root": str(RESULTS_MOUNT / ENDPOINT_NAMESPACE),
    "source_sha256": ENDPOINT_EVAL_SOURCE_SHA256,
}
REMOTE_PROFILE_ENV = {
    REMOTE_PROFILE_ENV_KEYS[key]: value
    for key, value in REMOTE_PROFILE_CONTRACT.items()
}
CHESS_SETTINGS = dict(PRODUCTION_SETTINGS)
CHESS_FINGERPRINT_PAYLOAD = {
    "schema": 1,
    "component": "endpoint-chess-b1-b5",
    "endpoint_eval_source_sha256": ENDPOINT_EVAL_SOURCE_SHA256,
    "upstream_git_sha": UPSTREAM_GIT_SHA,
    "eval_code_sha256": EVAL_CODE_SHA256,
    "data_sha256": CHESS_DATA_SHA256,
    "settings": CHESS_SETTINGS,
}
CHESS_EVAL_FINGERPRINT = hashlib.sha256(
    canonical_json(CHESS_FINGERPRINT_PAYLOAD)
).hexdigest()


loss_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-runtime-ubuntu22.04", add_python="3.11"
    )
    .pip_install(
        f"torch=={LOSS_TORCH_VERSION}",
        f"transformers=={LOSS_TRANSFORMERS_VERSION}",
        f"numpy=={LOSS_NUMPY_VERSION}",
        "safetensors==0.6.2",
    )
    .env(
        {
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": "/root",
            **REMOTE_PROFILE_ENV,
        }
    )
    .add_local_file(
        str(ENDPOINT_HELPER_LOCAL),
        remote_path=ENDPOINT_HELPER_REMOTE,
        copy=True,
    )
    .add_local_file(
        str(V2_ENTRYPOINT_LOCAL),
        remote_path=V2_ENTRYPOINT_REMOTE,
        copy=True,
    )
)
chess_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11"
    )
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
    .pip_install(
        "flash-attn==2.7.4.post1",
        extra_options="--no-build-isolation",
    )
    .env(
        {
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": "/root",
            **REMOTE_PROFILE_ENV,
        }
    )
    .add_local_file(
        str(ENDPOINT_HELPER_LOCAL),
        remote_path=ENDPOINT_HELPER_REMOTE,
        copy=True,
    )
    .add_local_file(
        str(V2_ENTRYPOINT_LOCAL),
        remote_path=V2_ENTRYPOINT_REMOTE,
        copy=True,
    )
    # Runtime-only local mounts must remain last.
    .add_local_dir(str(VERL_ROOT), remote_path=REMOTE_VERL_ROOT)
    .add_local_dir(str(EVAL_DATA_ROOT), remote_path=REMOTE_EVAL_DATA)
)
control_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(f"numpy=={LOSS_NUMPY_VERSION}")
    .env(
        {
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": "/root",
            **REMOTE_PROFILE_ENV,
        }
    )
    .add_local_file(
        str(ENDPOINT_HELPER_LOCAL),
        remote_path=ENDPOINT_HELPER_REMOTE,
        copy=True,
    )
    .add_local_file(
        str(V2_ENTRYPOINT_LOCAL),
        remote_path=V2_ENTRYPOINT_REMOTE,
        copy=True,
    )
)

data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=False)
checkpoint_volume = modal.Volume.from_name(
    CHECKPOINT_VOLUME_NAME, create_if_missing=False
)
results_volume = modal.Volume.from_name(
    RESULTS_VOLUME_NAME, create_if_missing=True
)
app = modal.App(
    APP_NAME,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _immutable_json(path: Path, payload: dict[str, Any]) -> None:
    if path.is_file():
        existing = _read_json(path)
        if existing != payload:
            raise ValueError(f"immutable JSON differs: {path}")
        return
    _atomic_json(path, payload)


def _assert_remote_profile_contract() -> dict[str, str]:
    """Fail before remote reads/writes if its baked profile did not survive."""

    observed = {
        key: os.environ.get(environment_key)
        for key, environment_key in REMOTE_PROFILE_ENV_KEYS.items()
    }
    mismatches = {
        key: {
            "expected": expected,
            "observed": observed[key],
            "environment_key": REMOTE_PROFILE_ENV_KEYS[key],
        }
        for key, expected in REMOTE_PROFILE_CONTRACT.items()
        if observed[key] != expected
    }
    if mismatches:
        raise RuntimeError(
            "remote endpoint-evaluator profile contract mismatch; refusing "
            "all artifact access: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return dict(REMOTE_PROFILE_CONTRACT)


def _remote_profile_guard(function):
    """Run the remote profile assertion outside all mutation/error wrappers."""

    @wraps(function)
    def guarded(*args, **kwargs):
        _assert_remote_profile_contract()
        return function(*args, **kwargs)

    return guarded


def _discover_configured_endpoints(
    checkpoint_mount: Path,
) -> list[dict[str, Any]]:
    return discover_endpoints(
        checkpoint_mount,
        experiment_version=EXPERIMENT_VERSION,
        include_exp4=INCLUDE_EXP4,
    )


def _safe_endpoint(endpoint: dict[str, Any]) -> tuple[str, Path]:
    endpoint_id = str(endpoint.get("endpoint_id", ""))
    if not _SAFE_ENDPOINT_RE.fullmatch(endpoint_id):
        raise ValueError(f"unsafe endpoint_id: {endpoint_id!r}")
    checkpoint = Path(str(endpoint.get("checkpoint_path", ""))).resolve()
    mount = CHECKPOINT_MOUNT.resolve()
    if not checkpoint.is_relative_to(mount):
        raise ValueError(f"endpoint is outside checkpoint mount: {checkpoint}")
    checkpoint_files(checkpoint)
    return endpoint_id, checkpoint


def _endpoint_root(endpoint_id: str, checkpoint_sha256: str) -> Path:
    if not _SAFE_ENDPOINT_RE.fullmatch(endpoint_id):
        raise ValueError(f"unsafe endpoint_id: {endpoint_id!r}")
    if not re.fullmatch(r"[0-9a-f]{64}", checkpoint_sha256):
        raise ValueError("checkpoint_sha256 must be a full digest")
    return (
        RESULTS_MOUNT
        / ENDPOINT_NAMESPACE
        / endpoint_id
        / checkpoint_sha256
    )


def configured_endpoint_root(
    endpoint_id: str, checkpoint_sha256: str
) -> Path:
    """Expose the profile-specific immutable result root for dry-run audits."""

    return _endpoint_root(endpoint_id, checkpoint_sha256)


def _loss_fingerprint(
    holdout: dict[str, Any], sft_audit: dict[str, Any]
) -> str:
    payload = {
        "schema": 1,
        "component": "endpoint-deterministic-losses",
        "endpoint_eval_source_sha256": ENDPOINT_EVAL_SOURCE_SHA256,
        "attention_backend": LOSS_ATTENTION_BACKEND,
        "torch": LOSS_TORCH_VERSION,
        "transformers": LOSS_TRANSFORMERS_VERSION,
        "numpy": LOSS_NUMPY_VERSION,
        "batch_size": LOSS_BATCH_SIZE,
        "pt_holdout_hash": holdout["holdout_hash"],
        "sft_audit_hash": sft_audit["audit_hash"],
    }
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def _component_dir(
    endpoint_id: str,
    checkpoint_sha256: str,
    component: str,
    fingerprint: str,
) -> Path:
    if component not in {"losses", "chess"}:
        raise ValueError(f"unsupported endpoint component: {component}")
    if not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
        raise ValueError("component fingerprint must be a full digest")
    return (
        _endpoint_root(endpoint_id, checkpoint_sha256)
        / f"{component}_{fingerprint[:12]}"
    )


def _valid_success(
    path: Path,
    *,
    endpoint_id: str,
    checkpoint_sha256: str,
    component: str,
    fingerprint: str,
) -> bool:
    if not path.is_file():
        return False
    try:
        value = _read_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return (
        value.get("schema") == ENDPOINT_RESULT_SCHEMA
        and value.get("state") == "complete"
        and value.get("endpoint_id") == endpoint_id
        and value.get("checkpoint_sha256") == checkpoint_sha256
        and value.get("component") == component
        and value.get("eval_fingerprint") == fingerprint
    )


def _lease_is_fresh(path: Path, ttl_seconds: int) -> bool:
    if not path.is_file():
        return False
    try:
        value = _read_json(path)
        timestamp = float(value["unix_time"])
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False
    return time.time() - timestamp < ttl_seconds


def _record_component_failure(component: str):
    """Persist terminal-attempt evidence without defeating Modal retries."""

    if component not in {"losses", "chess"}:
        raise ValueError(f"unsupported endpoint component: {component}")

    def decorate(function):
        @wraps(function)
        def wrapped(
            endpoint: dict[str, Any],
            expected_checkpoint_sha256: str,
        ):
            try:
                return function(endpoint, expected_checkpoint_sha256)
            except Exception as exc:
                endpoint_id = str(endpoint.get("endpoint_id", ""))
                if (
                    _SAFE_ENDPOINT_RE.fullmatch(endpoint_id)
                    and re.fullmatch(
                        r"[0-9a-f]{64}", expected_checkpoint_sha256
                    )
                ):
                    root = _endpoint_root(
                        endpoint_id, expected_checkpoint_sha256
                    )
                    for component_root in root.glob(f"{component}_*"):
                        running = component_root / "_RUNNING.json"
                        queued_path = component_root / "_QUEUED.json"
                        if not running.is_file() and not queued_path.is_file():
                            continue
                        try:
                            marker = (
                                _read_json(running)
                                if running.is_file()
                                else _read_json(queued_path)
                            )
                        except Exception:
                            marker = {}
                        try:
                            queued = _read_json(queued_path)
                            attempt = int(queued.get("attempt", 1))
                        except Exception:
                            attempt = 1
                        _atomic_json(
                            component_root / "_FAILED.json",
                            {
                                **marker,
                                "schema": ENDPOINT_RESULT_SCHEMA,
                                "state": "failed",
                                "component": component,
                                "endpoint_id": endpoint_id,
                                "checkpoint_sha256": (
                                    expected_checkpoint_sha256
                                ),
                                "failed_at": _utc_now(),
                                "error": f"{type(exc).__name__}: {exc}",
                                "attempt": attempt,
                                "max_attempts": MAX_COMPONENT_ATTEMPTS,
                                "retry_after_unix": (
                                    time.time()
                                    + FAILURE_RETRY_DELAY_SECONDS
                                ),
                            },
                        )
                        running.unlink(missing_ok=True)
                        queued_path.unlink(missing_ok=True)
                    # Modal's own retry starts within seconds and writes a new
                    # RUNNING marker. If all built-in retries fail, the
                    # watcher waits a finite backoff and submits the next
                    # explicitly numbered attempt.
                    try:
                        results_volume.commit()
                    except Exception:
                        pass
                raise

        return wrapped

    return decorate


def _checkpoint_inventory(checkpoint: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path.relative_to(checkpoint)),
            "bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in checkpoint_files(checkpoint)
    ]


def _cached_checkpoint_fingerprint(
    endpoint: dict[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    endpoint_id, checkpoint = _safe_endpoint(endpoint)
    inventory = _checkpoint_inventory(checkpoint)
    identity_path = RESULTS_MOUNT / ENDPOINT_NAMESPACE / endpoint_id / "_IDENTITY.json"
    if identity_path.is_file():
        identity = _read_json(identity_path)
        if (
            identity.get("schema") == "interleaved-endpoint-identity-v1"
            and identity.get("checkpoint_path") == str(checkpoint)
            and identity.get("file_inventory") == inventory
            and re.fullmatch(
                r"[0-9a-f]{64}", str(identity.get("checkpoint_sha256", ""))
            )
        ):
            declared = endpoint.get("declared_checkpoint_sha256")
            if (
                declared is not None
                and declared != identity["checkpoint_sha256"]
            ):
                raise ValueError(
                    f"declared checkpoint digest mismatch for {endpoint_id}: "
                    f"{declared} != {identity['checkpoint_sha256']}"
                )
            return str(identity["checkpoint_sha256"]), inventory
        raise ValueError(f"endpoint identity changed after publication: {endpoint_id}")
    digest = checkpoint_fingerprint(checkpoint)
    declared = endpoint.get("declared_checkpoint_sha256")
    if declared is not None and declared != digest:
        raise ValueError(
            f"declared checkpoint digest mismatch for {endpoint_id}: "
            f"{declared} != {digest}"
        )
    payload = {
        "schema": "interleaved-endpoint-identity-v1",
        "endpoint_id": endpoint_id,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": digest,
        "declared_checkpoint_sha256": declared,
        "file_inventory": inventory,
        "recorded_at": _utc_now(),
    }
    _immutable_json(identity_path, payload)
    return digest, inventory


@app.function(
    image=control_image,
    cpu=8.0,
    memory=32 * 1024,
    timeout=60 * 60,
    retries=modal.Retries(initial_delay=5.0, max_retries=2),
    volumes={str(DATA_MOUNT): data_volume},
)
@_remote_profile_guard
def prepare_holdouts() -> dict[str, Any]:
    """Materialize and authenticate the immutable PT/SFT evaluation contract."""

    import numpy as np

    data_volume.reload()
    source_manifest = _read_json(SOURCE_MANIFEST_PATH)
    selection = _read_json(TRAIN_SELECTION_PATH)
    manifest_set = _read_json(MANIFEST_SET_PATH)
    sft_cache = _read_json(SFT_CACHE_METADATA_PATH)
    p1_metadata = _read_json(P1_METADATA_PATH)
    p2_metadata = _read_json(P2_METADATA_PATH)

    if REUSE_AUTHENTICATED_PT_HOLDOUT:
        if not PT_HOLDOUT_PATH.is_file():
            raise FileNotFoundError(
                "v2 requires the authenticated v1 PT holdout and will not "
                f"create it: {PT_HOLDOUT_PATH}"
            )
        holdout = _read_json(PT_HOLDOUT_PATH)
    else:
        holdout = build_pt_holdout_manifest(
            source_manifest,
            selection,
            shard_sha256=lambda relative: sha256_file(
                SOURCE_ROOT / relative
            ),
            num_records=PT_HOLDOUT_RECORDS,
        )
    for shard in holdout["shards"]:
        path = SOURCE_ROOT / shard["relative_path"]
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if (
            array.ndim != 1
            or len(array) != int(shard["num_tokens"])
            or array.dtype != np.dtype(shard["dtype"])
        ):
            raise ValueError(f"held-out NPY structure drifted: {path}")
    validate_pt_holdout_manifest(
        holdout, source_manifest, selection, source_root=SOURCE_ROOT
    )

    p1_order = np.load(P1_ORDER_PATH, mmap_mode="r", allow_pickle=False)
    p2_order = np.load(P2_ORDER_PATH, mmap_mode="r", allow_pickle=False)
    if USE_V2_SFT_AUDIT:
        audit = build_v2_sft_holdout_audit(
            manifest_set,
            sft_cache,
            p1_metadata,
            p2_metadata,
            p1_order,
            p2_order,
            p1_metadata_file_sha256=sha256_file(P1_METADATA_PATH),
            p2_metadata_file_sha256=sha256_file(P2_METADATA_PATH),
            p1_order_file_sha256=sha256_file(P1_ORDER_PATH),
            p2_order_file_sha256=sha256_file(P2_ORDER_PATH),
        )
    else:
        audit = build_sft_holdout_audit(
            manifest_set,
            sft_cache,
            p1_metadata,
            p2_metadata,
            p1_order,
            p2_order,
        )
    validate_sft_holdout_audit(audit)

    EVAL_ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    if REUSE_AUTHENTICATED_PT_HOLDOUT:
        if PT_HOLDOUT_REUSE_MARKER_PATH is None:
            raise AssertionError("v2 PT holdout reuse marker is not configured")
        reuse_marker: dict[str, Any] = {
            "schema": "interleaved-pt-holdout-reuse-v1",
            "schema_version": 1,
            "profile": ENDPOINT_EVAL_PROFILE,
            "experiment_version": EXPERIMENT_VERSION,
            "source_path": str(PT_HOLDOUT_PATH),
            "holdout_hash": holdout["holdout_hash"],
            "source_manifest_hash": source_manifest["manifest_hash"],
            "training_selection_hash": selection["selection_hash"],
            "validation": "full-manifest-and-source-content-revalidation",
        }
        reuse_marker["reuse_hash"] = content_hash(
            reuse_marker, "reuse_hash"
        )
        _immutable_json(PT_HOLDOUT_REUSE_MARKER_PATH, reuse_marker)
    else:
        _immutable_json(PT_HOLDOUT_PATH, holdout)
    _immutable_json(SFT_AUDIT_PATH, audit)
    data_volume.commit()
    return {
        "state": "ready",
        "pt_holdout_path": str(PT_HOLDOUT_PATH),
        "pt_holdout_hash": holdout["holdout_hash"],
        "pt_records": holdout["num_records"],
        "pt_target_tokens": holdout["target_tokens"],
        "sft_audit_path": str(SFT_AUDIT_PATH),
        "sft_audit_hash": audit["audit_hash"],
        "sft_status": audit["status"],
    }


@app.function(
    image=loss_image,
    gpu="H200",
    cpu=16.0,
    memory=128 * 1024,
    timeout=60 * 60 * 4,
    retries=modal.Retries(initial_delay=5.0, max_retries=2),
    max_containers=32,
    volumes={
        str(DATA_MOUNT): data_volume,
        str(CHECKPOINT_MOUNT): checkpoint_volume,
        str(RESULTS_MOUNT): results_volume,
    },
)
@_remote_profile_guard
@_record_component_failure("losses")
def eval_losses_one(
    endpoint: dict[str, Any],
    expected_checkpoint_sha256: str,
) -> dict[str, Any]:
    """Evaluate deterministic next-token loss on the immutable PT holdout."""

    import numpy as np
    import torch
    import torch.nn.functional as F
    import transformers
    from transformers import AutoModelForCausalLM

    started_at = _utc_now()
    started_clock = time.monotonic()
    data_volume.reload()
    checkpoint_volume.reload()
    results_volume.reload()
    endpoint_id, checkpoint = _safe_endpoint(endpoint)
    actual_checkpoint_sha256 = checkpoint_fingerprint(checkpoint)
    if actual_checkpoint_sha256 != expected_checkpoint_sha256:
        raise ValueError(
            f"checkpoint identity changed: {actual_checkpoint_sha256} != "
            f"{expected_checkpoint_sha256}"
        )
    holdout = _read_json(PT_HOLDOUT_PATH)
    sft_audit = _read_json(SFT_AUDIT_PATH)
    source_manifest = _read_json(SOURCE_MANIFEST_PATH)
    selection = _read_json(TRAIN_SELECTION_PATH)
    validate_pt_holdout_manifest(
        holdout, source_manifest, selection, source_root=SOURCE_ROOT
    )
    validate_sft_holdout_audit(sft_audit)
    fingerprint = _loss_fingerprint(holdout, sft_audit)
    component_root = _component_dir(
        endpoint_id,
        expected_checkpoint_sha256,
        "losses",
        fingerprint,
    )
    success_path = component_root / "_SUCCESS.json"
    if _valid_success(
        success_path,
        endpoint_id=endpoint_id,
        checkpoint_sha256=expected_checkpoint_sha256,
        component="losses",
        fingerprint=fingerprint,
    ):
        return _read_json(success_path)
    component_root.mkdir(parents=True, exist_ok=True)
    _atomic_json(
        component_root / "_RUNNING.json",
        {
            "schema": ENDPOINT_RESULT_SCHEMA,
            "state": "running",
            "component": "losses",
            "endpoint_id": endpoint_id,
            "checkpoint_sha256": expected_checkpoint_sha256,
            "eval_fingerprint": fingerprint,
            "started_at": started_at,
            "unix_time": time.time(),
        },
    )
    results_volume.commit()

    model = AutoModelForCausalLM.from_pretrained(
        str(checkpoint),
        torch_dtype=torch.bfloat16,
        attn_implementation=LOSS_ATTENTION_BACKEND,
    ).to("cuda")
    model.eval()
    arrays: dict[str, np.ndarray] = {}
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    records = list(holdout["records"])
    with torch.inference_mode():
        for batch_start in range(0, len(records), LOSS_BATCH_SIZE):
            batch_records = records[batch_start : batch_start + LOSS_BATCH_SIZE]
            raw_rows = []
            for record in batch_records:
                relative = str(record["relative_path"])
                array = arrays.get(relative)
                if array is None:
                    array = np.load(
                        SOURCE_ROOT / relative,
                        mmap_mode="r",
                        allow_pickle=False,
                    )
                    arrays[relative] = array
                start, stop = int(record["start"]), int(record["stop"])
                row = np.asarray(array[start:stop], dtype=np.int64)
                if len(row) != SEQUENCE_LENGTH + 1:
                    raise ValueError(f"short held-out record: {record}")
                raw_rows.append(row)
            raw = torch.from_numpy(np.stack(raw_rows)).to(
                "cuda", non_blocking=True
            )
            input_ids = raw[:, :-1]
            labels = raw[:, 1:]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(input_ids=input_ids, use_cache=False).logits
            batch_loss = F.cross_entropy(
                logits.float().reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                reduction="sum",
            )
            total_loss += float(batch_loss.item())
            total_correct += int(
                logits.argmax(dim=-1).eq(labels).sum().item()
            )
            total_tokens += int(labels.numel())
    if total_tokens != int(holdout["target_tokens"]):
        raise ValueError(
            f"held-out target count drifted: {total_tokens} != "
            f"{holdout['target_tokens']}"
        )
    mean_loss = total_loss / total_tokens
    payload = {
        "schema": ENDPOINT_RESULT_SCHEMA,
        "schema_version": 1,
        "state": "complete",
        "component": "losses",
        "namespace": ENDPOINT_NAMESPACE,
        "experiment_version": EXPERIMENT_VERSION,
        "endpoint_id": endpoint_id,
        "endpoint": endpoint,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": expected_checkpoint_sha256,
        "eval_fingerprint": fingerprint,
        "runtime": {
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "attention_backend": LOSS_ATTENTION_BACKEND,
            "dtype": "bfloat16",
            "batch_size": LOSS_BATCH_SIZE,
        },
        "datasets": {
            "pretraining": {
                "manifest_path": str(PT_HOLDOUT_PATH),
                "holdout_hash": holdout["holdout_hash"],
                "records": holdout["num_records"],
                "target_tokens": holdout["target_tokens"],
                "source": holdout["source"],
                "non_overlap_proof": holdout["non_overlap_proof"],
            },
            "masked_sft": {
                "audit_path": str(SFT_AUDIT_PATH),
                "audit_hash": sft_audit["audit_hash"],
                "status": sft_audit["status"],
                "reason": sft_audit["reason"],
                "source": sft_audit["source"],
                "coverage_proof": sft_audit["coverage_proof"],
            },
        },
        "metrics": {
            "heldout_pretrain_loss": mean_loss,
            "heldout_pretrain_perplexity": safe_perplexity(mean_loss),
            "heldout_pretrain_token_accuracy": total_correct / total_tokens,
            "heldout_pretrain_correct_tokens": total_correct,
            "heldout_pretrain_target_tokens": total_tokens,
            **sft_audit["metrics"],
            "masked_sft_status": sft_audit["status"],
        },
        "started_at": started_at,
        "finished_at": _utc_now(),
        "duration_seconds": round(time.monotonic() - started_clock, 3),
    }
    payload["result_hash"] = content_hash(payload, "result_hash")
    _immutable_json(success_path, payload)
    (component_root / "_RUNNING.json").unlink(missing_ok=True)
    (component_root / "_QUEUED.json").unlink(missing_ok=True)
    (component_root / "_FAILED.json").unlink(missing_ok=True)
    results_volume.commit()
    print(
        f"[loss-success] {endpoint_id}: loss={mean_loss:.8f}, "
        f"ppl={payload['metrics']['heldout_pretrain_perplexity']:.8f}, "
        f"accuracy={payload['metrics']['heldout_pretrain_token_accuracy']:.8f}",
        flush=True,
    )
    return payload


@app.function(
    image=chess_image,
    gpu="H200",
    cpu=16.0,
    memory=300 * 1024,
    timeout=60 * 60 * 8,
    retries=modal.Retries(initial_delay=5.0, max_retries=2),
    max_containers=32,
    volumes={
        str(CHECKPOINT_MOUNT): checkpoint_volume,
        str(RESULTS_MOUNT): results_volume,
    },
)
@_remote_profile_guard
@_record_component_failure("chess")
def eval_chess_one(
    endpoint: dict[str, Any],
    expected_checkpoint_sha256: str,
) -> dict[str, Any]:
    """Run the exact existing production B1--B5 evaluator on one endpoint."""

    started_at = _utc_now()
    started_clock = time.monotonic()
    checkpoint_volume.reload()
    results_volume.reload()
    endpoint_id, checkpoint = _safe_endpoint(endpoint)
    actual_checkpoint_sha256 = checkpoint_fingerprint(checkpoint)
    if actual_checkpoint_sha256 != expected_checkpoint_sha256:
        raise ValueError(
            f"checkpoint identity changed: {actual_checkpoint_sha256} != "
            f"{expected_checkpoint_sha256}"
        )
    component_root = _component_dir(
        endpoint_id,
        expected_checkpoint_sha256,
        "chess",
        CHESS_EVAL_FINGERPRINT,
    )
    success_path = component_root / "_SUCCESS.json"
    if _valid_success(
        success_path,
        endpoint_id=endpoint_id,
        checkpoint_sha256=expected_checkpoint_sha256,
        component="chess",
        fingerprint=CHESS_EVAL_FINGERPRINT,
    ):
        return _read_json(success_path)
    component_root.mkdir(parents=True, exist_ok=True)
    _atomic_json(
        component_root / "_RUNNING.json",
        {
            "schema": ENDPOINT_RESULT_SCHEMA,
            "state": "running",
            "component": "chess",
            "endpoint_id": endpoint_id,
            "checkpoint_sha256": expected_checkpoint_sha256,
            "eval_fingerprint": CHESS_EVAL_FINGERPRINT,
            "started_at": started_at,
            "unix_time": time.time(),
        },
    )
    results_volume.commit()

    output_dir = component_root / "output"
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
            "MODEL_PATH": str(checkpoint),
            "OUTPUT_DIR": str(output_dir),
            "EXPERIMENT_NAME": "eval",
            "EVAL_DATA_DIR": REMOTE_EVAL_DATA,
            "EVAL_DATASETS": ",".join(CHESS_SETTINGS["datasets"]),
            "RES_LENGTH": str(CHESS_SETTINGS["response_length"]),
            "TEMPERATURE": str(CHESS_SETTINGS["temperature"]),
            "N_SAMPLES": str(CHESS_SETTINGS["n_samples"]),
            "ROLLOUT_NAME": CHESS_SETTINGS["rollout"],
            "VLLM_MODEL_IMPL": CHESS_SETTINGS["model_impl"],
            "MULTI_TURN": str(CHESS_SETTINGS["multi_turn"]),
            "THINKING": str(CHESS_SETTINGS["thinking"]),
            "TTS": str(CHESS_SETTINGS["tts"]),
            "SEED": str(CHESS_SETTINGS["seed"]),
            "MAX_NUM_SEQS": str(CHESS_SETTINGS["max_num_seqs"]),
            "MAX_NUM_BATCHED_TOKENS": str(
                CHESS_SETTINGS["max_num_batched_tokens"]
            ),
            "GPU_MEMORY": str(CHESS_SETTINGS["gpu_memory"]),
            "MICRO_BATCH_SIZE": "32",
            "ENFORCE_EAGER": str(CHESS_SETTINGS["enforce_eager"]),
            "FREE_CACHE_ENGINE": str(CHESS_SETTINGS["free_cache_engine"]),
            "DEBUG": "False",
            "TOKENIZERS_PARALLELISM": "true",
            "NCCL_DEBUG": "WARN",
            "VLLM_LOGGING_LEVEL": "WARN",
        }
    )
    command = ["bash", f"{REMOTE_VERL_ROOT}/verl/eval_bash/verl_eval.sh"]
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

    generations = output_dir / "eval/generations/0.jsonl"
    metrics_path = output_dir / "eval/generations/metrics.json"
    actual_rows = 0
    if generations.is_file():
        with generations.open() as handle:
            actual_rows = sum(1 for _ in handle)
    expected_rows = 1_480 * int(CHESS_SETTINGS["n_samples"])
    if return_code != 0:
        raise RuntimeError(f"VERL chess evaluation exited {return_code}")
    if actual_rows != expected_rows:
        raise ValueError(
            f"chess evaluation row mismatch: {actual_rows}/{expected_rows}"
        )
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    raw_metrics = _read_json(metrics_path)
    summary = summarize_chess_metrics(raw_metrics)
    payload = {
        "schema": ENDPOINT_RESULT_SCHEMA,
        "schema_version": 1,
        "state": "complete",
        "component": "chess",
        "namespace": ENDPOINT_NAMESPACE,
        "experiment_version": EXPERIMENT_VERSION,
        "endpoint_id": endpoint_id,
        "endpoint": endpoint,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": expected_checkpoint_sha256,
        "eval_fingerprint": CHESS_EVAL_FINGERPRINT,
        "settings": CHESS_SETTINGS,
        "dataset_sha256": CHESS_DATA_SHA256,
        "upstream_git_sha": UPSTREAM_GIT_SHA,
        "eval_code_sha256": EVAL_CODE_SHA256,
        "expected_rows": expected_rows,
        "actual_rows": actual_rows,
        "generations": str(generations),
        "raw_metrics_path": str(metrics_path),
        "metrics": summary,
        "started_at": started_at,
        "finished_at": _utc_now(),
        "duration_seconds": round(time.monotonic() - started_clock, 3),
    }
    payload["result_hash"] = content_hash(payload, "result_hash")
    _immutable_json(success_path, payload)
    (component_root / "_RUNNING.json").unlink(missing_ok=True)
    (component_root / "_QUEUED.json").unlink(missing_ok=True)
    (component_root / "_FAILED.json").unlink(missing_ok=True)
    results_volume.commit()
    print(
        f"[chess-success] {endpoint_id}: "
        f"pass@1={summary['pass_at_1']:.8f}, "
        f"avg_reward={summary['avg_reward']:.8f}",
        flush=True,
    )
    return payload


def _component_state(
    endpoint_id: str,
    checkpoint_sha256: str,
    component: str,
    fingerprint: str,
) -> str:
    root = _component_dir(
        endpoint_id, checkpoint_sha256, component, fingerprint
    )
    if _valid_success(
        root / "_SUCCESS.json",
        endpoint_id=endpoint_id,
        checkpoint_sha256=checkpoint_sha256,
        component=component,
        fingerprint=fingerprint,
    ):
        return "complete"
    if _lease_is_fresh(root / "_RUNNING.json", RUNNING_LEASE_SECONDS):
        return "running"
    if _lease_is_fresh(root / "_QUEUED.json", QUEUE_LEASE_SECONDS):
        return "queued"
    if (root / "_FAILED.json").is_file():
        try:
            failure = _read_json(root / "_FAILED.json")
            attempt = int(failure.get("attempt", 1))
            retry_after = float(failure.get("retry_after_unix", 0))
        except Exception:
            return "failed_terminal"
        if attempt >= MAX_COMPONENT_ATTEMPTS:
            return "failed_terminal"
        if time.time() < retry_after:
            return "retry_wait"
        return "failed_retryable"
    return "not_queued"


@app.function(
    image=control_image,
    cpu=4.0,
    memory=16 * 1024,
    timeout=60 * 60 * WATCH_MAX_HOURS,
    retries=modal.Retries(initial_delay=10.0, max_retries=2),
    volumes={
        str(DATA_MOUNT): data_volume,
        str(CHECKPOINT_MOUNT): checkpoint_volume,
        str(RESULTS_MOUNT): results_volume,
    },
)
@_remote_profile_guard
def watch_and_enqueue(
    poll_seconds: int = WATCH_POLL_SECONDS,
    max_hours: int = WATCH_MAX_HOURS,
) -> dict[str, Any]:
    """Watch dependencies and enqueue each loss/chess component exactly once."""

    deadline = time.monotonic() + max_hours * 3600
    submitted: set[tuple[str, str, str]] = set()
    submission_count = 0
    while time.monotonic() < deadline:
        data_volume.reload()
        checkpoint_volume.reload()
        results_volume.reload()
        if not PT_HOLDOUT_PATH.is_file() or not SFT_AUDIT_PATH.is_file():
            raise FileNotFoundError("run prepare_holdouts before the watcher")
        holdout = _read_json(PT_HOLDOUT_PATH)
        sft_audit = _read_json(SFT_AUDIT_PATH)
        validate_sft_holdout_audit(sft_audit)
        loss_fingerprint = _loss_fingerprint(holdout, sft_audit)
        endpoints = _discover_configured_endpoints(CHECKPOINT_MOUNT)
        pending: list[tuple[str, dict[str, Any], str, str]] = []
        rows: list[dict[str, Any]] = []
        for endpoint in endpoints:
            endpoint_id = str(endpoint["endpoint_id"])
            checkpoint_sha256, inventory = _cached_checkpoint_fingerprint(endpoint)
            for component, fingerprint in (
                ("losses", loss_fingerprint),
                ("chess", CHESS_EVAL_FINGERPRINT),
            ):
                state = _component_state(
                    endpoint_id,
                    checkpoint_sha256,
                    component,
                    fingerprint,
                )
                task = (endpoint_id, checkpoint_sha256, component)
                if state in {
                    "complete",
                    "running",
                    "queued",
                    "retry_wait",
                    "failed_terminal",
                }:
                    continue
                component_root = _component_dir(
                    endpoint_id,
                    checkpoint_sha256,
                    component,
                    fingerprint,
                )
                component_root.mkdir(parents=True, exist_ok=True)
                failed_path = component_root / "_FAILED.json"
                previous_attempt = 0
                if failed_path.is_file():
                    try:
                        previous_attempt = int(
                            _read_json(failed_path).get("attempt", 0)
                        )
                    except Exception:
                        previous_attempt = MAX_COMPONENT_ATTEMPTS
                attempt = previous_attempt + 1
                if attempt > MAX_COMPONENT_ATTEMPTS:
                    continue
                _atomic_json(
                    component_root / "_QUEUED.json",
                    {
                        "schema": ENDPOINT_RESULT_SCHEMA,
                        "state": "queued",
                        "component": component,
                        "endpoint_id": endpoint_id,
                        "checkpoint_sha256": checkpoint_sha256,
                        "eval_fingerprint": fingerprint,
                        "file_inventory": inventory,
                        "attempt": attempt,
                        "max_attempts": MAX_COMPONENT_ATTEMPTS,
                        "queued_at": _utc_now(),
                        "unix_time": time.time(),
                    },
                )
                pending.append(
                    (component, endpoint, checkpoint_sha256, fingerprint)
                )
            rows.append(
                {
                    "endpoint_id": endpoint_id,
                    "checkpoint_sha256": checkpoint_sha256,
                    "losses": _component_state(
                        endpoint_id,
                        checkpoint_sha256,
                        "losses",
                        loss_fingerprint,
                    ),
                    "chess": _component_state(
                        endpoint_id,
                        checkpoint_sha256,
                        "chess",
                        CHESS_EVAL_FINGERPRINT,
                    ),
                }
            )
        if pending:
            results_volume.commit()
        for component, endpoint, checkpoint_sha256, _ in pending:
            if component == "losses":
                eval_losses_one.spawn(endpoint, checkpoint_sha256)
            else:
                eval_chess_one.spawn(endpoint, checkpoint_sha256)
            submitted.add(
                (str(endpoint["endpoint_id"]), checkpoint_sha256, component)
            )
            submission_count += 1

        discovered_fixed = {
            str(endpoint["endpoint_id"])
            for endpoint in endpoints
            if endpoint["experiment"] != "E4"
        }
        discovered_exp4 = {
            (str(endpoint["filter"]), str(endpoint["method"]))
            for endpoint in endpoints
            if endpoint["experiment"] == "E4"
        }
        all_dependencies_seen = (
            EXPECTED_FIXED_ENDPOINTS.issubset(discovered_fixed)
            and EXPECTED_EXP4_CELLS.issubset(discovered_exp4)
        )
        all_complete = bool(rows) and all(
            row["losses"] == "complete" and row["chess"] == "complete"
            for row in rows
        )
        print(
            f"[endpoint-watch] discovered={len(endpoints)} "
            f"enqueued_now={len(pending)} submitted={len(submitted)} "
            f"all_dependencies_seen={all_dependencies_seen} "
            f"all_complete={all_complete}",
            flush=True,
        )
        if all_dependencies_seen and all_complete:
            return {
                "state": "complete",
                "endpoints": rows,
                "submitted": len(submitted),
                "submission_count": submission_count,
                "finished_at": _utc_now(),
            }
        time.sleep(max(30, int(poll_seconds)))
    return {
        "state": "watch_timeout",
        "submitted": len(submitted),
        "submission_count": submission_count,
        "finished_at": _utc_now(),
    }


@app.function(
    image=control_image,
    cpu=4.0,
    memory=16 * 1024,
    timeout=60 * 60,
    volumes={
        str(DATA_MOUNT): data_volume,
        str(CHECKPOINT_MOUNT): checkpoint_volume,
        str(RESULTS_MOUNT): results_volume,
    },
)
@_remote_profile_guard
def status_endpoints() -> dict[str, Any]:
    data_volume.reload()
    checkpoint_volume.reload()
    results_volume.reload()
    holdout = _read_json(PT_HOLDOUT_PATH) if PT_HOLDOUT_PATH.is_file() else None
    audit = _read_json(SFT_AUDIT_PATH) if SFT_AUDIT_PATH.is_file() else None
    loss_fingerprint = (
        _loss_fingerprint(holdout, audit)
        if holdout is not None and audit is not None
        else None
    )
    rows = []
    for endpoint in _discover_configured_endpoints(CHECKPOINT_MOUNT):
        endpoint_id = str(endpoint["endpoint_id"])
        checkpoint_sha256, _ = _cached_checkpoint_fingerprint(endpoint)
        rows.append(
            {
                **endpoint,
                "checkpoint_sha256": checkpoint_sha256,
                "losses": (
                    _component_state(
                        endpoint_id,
                        checkpoint_sha256,
                        "losses",
                        str(loss_fingerprint),
                    )
                    if loss_fingerprint
                    else "blocked_on_holdout_prep"
                ),
                "chess": _component_state(
                    endpoint_id,
                    checkpoint_sha256,
                    "chess",
                    CHESS_EVAL_FINGERPRINT,
                ),
            }
        )
    return {
        "profile": ENDPOINT_EVAL_PROFILE,
        "namespace": ENDPOINT_NAMESPACE,
        "experiment_version": EXPERIMENT_VERSION,
        "data_artifact_root": str(DATA_ARTIFACT_ROOT),
        "evaluation_artifact_root": str(EVAL_ARTIFACT_ROOT),
        "reuses_authenticated_pt_holdout": (
            REUSE_AUTHENTICATED_PT_HOLDOUT
        ),
        "pt_holdout": (
            {
                "path": str(PT_HOLDOUT_PATH),
                "hash": holdout["holdout_hash"],
                "records": holdout["num_records"],
                "target_tokens": holdout["target_tokens"],
            }
            if holdout
            else {"status": "not_prepared"}
        ),
        "masked_sft": (
            {
                "path": str(SFT_AUDIT_PATH),
                "hash": audit["audit_hash"],
                "status": audit["status"],
            }
            if audit
            else {"status": "not_prepared"}
        ),
        "chess": {
            "eval_fingerprint": CHESS_EVAL_FINGERPRINT,
            "dataset_sha256": CHESS_DATA_SHA256,
            "settings": CHESS_SETTINGS,
        },
        "endpoints": rows,
        "expected_fixed_endpoints": sorted(EXPECTED_FIXED_ENDPOINTS),
        "expected_exp4_cells": [
            {"filter": setting, "method": method}
            for setting, method in sorted(EXPECTED_EXP4_CELLS)
        ],
        "generated_at": _utc_now(),
    }


@app.local_entrypoint()
def main(
    mode: str = "dry-run",
    poll_seconds: int = WATCH_POLL_SECONDS,
    max_hours: int = WATCH_MAX_HOURS,
) -> None:
    normalized = mode.strip().lower()
    if normalized == "prep":
        print(json.dumps(prepare_holdouts.remote(), indent=2, sort_keys=True))
    elif normalized == "dry-run":
        print(
            json.dumps(
                {
                    "app": APP_NAME,
                    "profile": ENDPOINT_EVAL_PROFILE,
                    "namespace": ENDPOINT_NAMESPACE,
                    "experiment_version": EXPERIMENT_VERSION,
                    "data_artifact_root": str(DATA_ARTIFACT_ROOT),
                    "evaluation_artifact_root": str(EVAL_ARTIFACT_ROOT),
                    "pt_holdout_path": str(PT_HOLDOUT_PATH),
                    "results_root": str(RESULTS_MOUNT / ENDPOINT_NAMESPACE),
                    "reuses_authenticated_pt_holdout": (
                        REUSE_AUTHENTICATED_PT_HOLDOUT
                    ),
                    "pt_holdout_records": PT_HOLDOUT_RECORDS,
                    "pt_holdout_target_tokens": PT_HOLDOUT_TARGET_TOKENS,
                    "sft_status": "unavailable_no_heldout",
                    "fixed_endpoints": sorted(EXPECTED_FIXED_ENDPOINTS),
                    "exp4_cells": [
                        {"filter": setting, "method": method}
                        for setting, method in sorted(EXPECTED_EXP4_CELLS)
                    ],
                    "chess_fingerprint": CHESS_EVAL_FINGERPRINT,
                    "chess_data_sha256": CHESS_DATA_SHA256,
                    "chess_settings": CHESS_SETTINGS,
                    "source_sha256": ENDPOINT_EVAL_SOURCE_SHA256,
                },
                indent=2,
                sort_keys=True,
            )
        )
    elif normalized == "launch":
        prepared = prepare_holdouts.remote()
        print(json.dumps(prepared, indent=2, sort_keys=True))
        call = watch_and_enqueue.spawn(
            poll_seconds=max(30, int(poll_seconds)),
            max_hours=max(1, int(max_hours)),
        )
        print(f"watcher_call_id={call.object_id}")
    elif normalized == "status":
        print(json.dumps(status_endpoints.remote(), indent=2, sort_keys=True))
    else:
        raise ValueError("mode must be one of prep, dry-run, launch, status")
