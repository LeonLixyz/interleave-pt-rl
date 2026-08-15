"""Modal entrypoints for the 47.245M interleaved pretrain/RL experiment.

This module intentionally does not use ``modal_train.train_one``: these runs
start from an arbitrary mixed pretrain+SFT Hugging Face checkpoint, rather than
from one of the historical SFT specs.  The rollout/training command still goes
through the same verified ``run_chess_miles`` adapter and pinned Miles/SGLang
image.
"""

from __future__ import annotations

import builtins
import ctypes
import errno
import fcntl
import functools
import hashlib
import json
import math
import os
import re
import shutil
import secrets
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import modal

# Modal mounts this launcher itself at ``/root/modal_interleave.py`` while the
# package tree is mounted separately. Ensure the package is importable before
# importing any ``chess_rl_miles`` module (the runtime subprocess environment
# is configured later, after module import).
_REMOTE_PROJECT_DIR = Path("/root/chess-rl-miles")
if _REMOTE_PROJECT_DIR.is_dir() and str(_REMOTE_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_REMOTE_PROJECT_DIR))

from chess_rl_miles.data import DEFAULT_TRAIN_FILE_SHA256
from chess_rl_miles.provenance import (
    directory_identity,
    runtime_identity,
    source_tree_identity,
    write_run_provenance,
)
from chess_rl_miles.scripts import modal_train as base
from chess_rl_miles.tokenizer_contract import (
    EXPECTED_RL_VOCAB_85,
    EXPECTED_RL_VOCAB_MAPPING_SHA256,
    validate_exact_rl_vocab,
    vocab_mapping_sha256,
)
from chess_rl_miles.v2r4_contract_binding import (
    EXPECTED_CONTRACT_FILE_SHA256 as V2R4_EXPECTED_CONTRACT_FILE_SHA256,
)
from chess_rl_miles.v2r4_contract_binding import (
    EXPECTED_CONTRACT_SHA256 as V2R4_EXPECTED_CONTRACT_SHA256,
)


APP_NAME = "chess-interleave-rl-fp32-master-v3"
MODEL_ID = "interleave_47m_qwen3"
CONTEXT2048_MODEL_ID = "context2048_47m_qwen3"
POLICY_UPDATE_PROFILE = "small-model-h200"
SMALL_MODEL_MAX_TOKENS_ALLOWED = frozenset({65_536, 131_072})
SMALL_MODEL_MAX_TOKENS_DEFAULT = 131_072
SGLANG_SERVER_CONCURRENCY_ALLOWED = frozenset({128, 256})
SGLANG_SERVER_CONCURRENCY_DEFAULT = 128
# KL estimator. "low_var_kl" (= k3) matches the original verl chess RL setup;
# Miles' own default is "k1", which is signed and higher-variance.
KL_LOSS_TYPE_DEFAULT = "low_var_kl"
KL_LOSS_TYPE_ALLOWED = frozenset({"k1", "k2", "k3", "low_var_kl"})
SMALL_MODEL_HOST_MEMORY_GB = 192
SMALL_MODEL_HOST_MEMORY_MB = SMALL_MODEL_HOST_MEMORY_GB * 1024
ROLLOUT_MAX_PROMPT_LEN_DEFAULT = 512
ROLLOUT_MAX_RESPONSE_LEN_DEFAULT = 2_560
ROLLOUT_MAX_CONTEXT_LEN_DEFAULT = 3_072
CHECKPOINT_COMMIT_POLL_SECONDS = 5.0
CHECKPOINT_COMMIT_MARKER = "COMMITTED.json"
PRECISION_FINALIZER_MAX_RETRIES = 2
PRECISION_FINALIZER_RETRY_DELAY_SECONDS = 5.0
RAW_RL_ROOT = "/rl-checkpoints/chess-rl-miles-interleave-fp32-master-v3"
PRECISION_RESUME_GATE_VERSION = "bf16-fp32-master-resume-v3"
PRECISION_RESUME_GATE_ROOT = f"{RAW_RL_ROOT}/_precision_resume_gates"
PRECISION_RESUME_GATE_WANDB_GROUP = "bf16_fp32_master_resume_gate_v3"
WANDB_ENTITY = "jingyanshen-new-york-university"
PRETRAIN_CKPT_ROOT = "/pretrain-checkpoints"
HF_EXPORT_ROOT = f"{PRETRAIN_CKPT_ROOT}/interleave_50m/rl_hf_fp32_master_v3"
EXPECTED_RL_TOKEN_IDS = {
    token: EXPECTED_RL_VOCAB_85[token]
    for token in (
        "<bos>",
        "<eos>",
        "<unk>",
        "<T>",
        "</T>",
        "<sep>",
        "<call_env>",
    )
}
EXPECTED_RL_VOCAB_SIZE = len(EXPECTED_RL_VOCAB_85)
PRODUCTION_LAUNCH_CLAIM_DICT_NAME = (
    "chess-interleave-rl-production-launch-claims-fp32-master-v3"
)
PRODUCTION_LAUNCH_CLAIM_SCHEMA = (
    "chess-rl-miles-production-launch-claim-v1"
)
PRODUCTION_LAUNCH_ATTEMPT_SCHEMA = (
    "chess-rl-miles-production-launch-attempt-v1"
)
PRODUCTION_LAUNCH_EXECUTION_SCHEMA = (
    "chess-rl-miles-production-launch-execution-v1"
)
PRODUCTION_GENERATION_RESOLUTION_SCHEMA = (
    "chess-rl-miles-production-generation-resolution-v1"
)
PRODUCTION_TERMINAL_CALL_EVIDENCE_SCHEMA = (
    "chess-rl-miles-authoritative-terminal-call-result-v2"
)
PRODUCTION_DURABLE_ANCHOR_SCHEMA = (
    "chess-rl-miles-durable-production-launch-anchor-v1"
)
PRODUCTION_DURABLE_COMPLETION_SCHEMA = (
    "chess-rl-miles-durable-production-launch-completion-v1"
)
MAX_PRODUCTION_LAUNCH_GENERATIONS = 1_000
PRODUCTION_LEASE_REFRESH_SECONDS = 5 * 60
CHECKPOINT_VOLUME_COMMIT_LOCK_SUFFIX = ".volume-commit.lock"
LOCAL_PRODUCTION_LAUNCH_RECOVERY_ROOT = (
    base.WORKSPACE_LOCAL.parent
    / ".modal-launch-recovery"
    / APP_NAME
)


PRODUCTION_RL_TRAINING_TERMINAL_MARKER = (
    "CHESS_RL_PRODUCTION_TRAINING_TERMINAL_V1"
)
PRODUCTION_RL_DISPATCHER_TERMINAL_MARKER = (
    "CHESS_RL_PRODUCTION_DISPATCHER_TERMINAL_V1"
)


def _wrap_deployed_terminal_failure(marker: str):
    """Prevent a user-raised TimeoutError from looking like a poll timeout."""

    def decorate(function):
        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            try:
                return function(*args, **kwargs)
            except Exception as exc:
                raise RuntimeError(marker) from exc

        return wrapped

    return decorate
BALANCED_TRAIN_FILE = (
    "/data/chess-rl-data/train_v4_dataset_balanced_multi_turn.parquet"
)
BALANCED_TRAIN_SHA256 = DEFAULT_TRAIN_FILE_SHA256
DEFAULT_DATA_SOURCE_PATH = (
    "miles.rollout.data_source.RolloutDataSourceWithBuffer"
)
STRICT_GATE_DATA_SOURCE_PATH = (
    "chess_rl_miles.gate_data_source.StrictEpochRolloutDataSource"
)
PRECISION_RESUME_DATA_SOURCE_PATH = (
    "chess_rl_miles.precision_resume_data_source."
    "PrecisionResumeRolloutDataSource"
)
V2R4_GATE_VERSION = "v2r4a_production_gate_20260730"
V2R4_GATE_CONTRACT_SCHEMA = (
    "interleaved-v2r4a-production-gate-runtime-contract-v1"
)
V2R4_GATE_BINDING_RELATIVE_PATH = (
    "chess_rl_miles/v2r4_contract_binding.py"
)
V2R4_GATE_CONTRACT_MANIFEST = (
    "/data/50m_interleaved_mix10b_sft90k_v2r1_clean_verify_gate/"
    "v2r4a_production_gate_20260730/runtime_contract.json"
)
V2R4_GATE_PROMPT_MANIFEST = (
    "/data/50m_interleaved_mix10b_sft90k_v2r1_clean_verify_gate/"
    "v2r4_production_gate_20260730/prompt_batches_manifest.json"
)
V2R4_GATE_PROMPT_MANIFEST_SHA256 = (
    "8ce046f9a560c7227ad33cc5f2baecc79d210e6f703c73e895469c1d566c6af5"
)
V2R4_GATE_PROMPT_MANIFEST_FILE_SHA256 = (
    "a01bb692dd2f129c2463df91aab7006e4762a0ac55e6471f0708ef4db34ba126"
)
V2R4_GATE_BATCHES = {
    "A": {
        "path": (
            "/data/50m_interleaved_mix10b_sft90k_v2r1_clean_verify_gate/"
            "v2r4_production_gate_20260730/batch_a.parquet"
        ),
        "sha256": (
            "9002d22fd567a91de9d7a3a7ba2119d0a5e812a74d473d82dce2508c2eefd01d"
        ),
        "rows": 1_024,
        "rollout_seed": 1_567_877_051,
        "prompt_set_sha256": (
            "8d2f389ba1df4aa1594d8abb894941723158b4c4d072e1b54a9681ac8a7b89a2"
        ),
        "epoch0_prompt_order_sha256": (
            "502dd02b274ef964b49d7e8b8fc187d12d9b8e27f90d5df6de03a7091d1c55d4"
        ),
    },
    "B": {
        "path": (
            "/data/50m_interleaved_mix10b_sft90k_v2r1_clean_verify_gate/"
            "v2r4_production_gate_20260730/batch_b.parquet"
        ),
        "sha256": (
            "1f9031efe2ea071c18d4beccb0c6394d1c3e10a4d962bfc2773ac5ad20d3c79e"
        ),
        "rows": 1_024,
        "rollout_seed": 923_570_888,
        "prompt_set_sha256": (
            "ceabf3581a9ea0bfbbe61430c22d24849d459aad03e17cbac356ee2c71ca9d74"
        ),
        "epoch0_prompt_order_sha256": (
            "663d0a51dfab347900dab5fe32bd55adc5b98d424306cd4108ab605e4b6c6eb2"
        ),
    },
}
_V2R4_SNAPSHOT_ROOT = (
    "/pretrain-checkpoints/interleave_50m/pretrain/"
    "mix10b_sft90k_3072_v2r3_diagnostic_20260730/"
    "p1_w4067c60eaba84b1e/snapshots"
)
V2R4_GATE_CANDIDATES = {
    6_000: {
        "hf_path": f"{_V2R4_SNAPSHOT_ROOT}/step_6000/hf",
        "hf_directory_manifest_sha256": (
            "3285baeb7c6ca4de2a320522906b031f6538c75106cb13fddc68194c96d23d70"
        ),
        "endpoint_checkpoint_sha256": (
            "17acd19dd1e89390c609a3f0f6c72ab543b8869f2d2ffd10528c8fe84cb20690"
        ),
        "original_p1_eligible": False,
    },
    8_000: {
        "hf_path": f"{_V2R4_SNAPSHOT_ROOT}/step_8000/hf",
        "hf_directory_manifest_sha256": (
            "13fde44ba75511e8cd7d23a9e73db507bade841fcc682dc2851261690e918758"
        ),
        "endpoint_checkpoint_sha256": (
            "e1006a970b5b7c9c9e5aefdbae3c716740e69970c0bcb4bb32b4cbab7af43634"
        ),
        "original_p1_eligible": False,
    },
    9_920: {
        "hf_path": f"{_V2R4_SNAPSHOT_ROOT}/step_9920/hf",
        "hf_directory_manifest_sha256": (
            "49fe6fe87d78ba58ebd96cf154567bd1526b6c12a4193809652b875a7af5d186"
        ),
        "endpoint_checkpoint_sha256": (
            "9a89d52a60b87b0f27108e5b08e33395757e374a4b59a592babb9435edb4b1c8"
        ),
        "original_p1_eligible": True,
    },
}
V2R4_GATE_SEMANTICS = {
    "debug_rollout_only": True,
    "num_rollout": 4,
    "rollout_batch_size": 256,
    "samples_per_prompt": 8,
    "total_prompt_groups": 1_024,
    "total_rows": 8_192,
    "dynamic_filter": False,
    "partial_rollout": False,
    "policy_updates": False,
    "checkpoint_saves": False,
    "automatic_retries": 0,
    "deterministic_inference": True,
    "sampling_seed_rule": (
        "rollout_seed_plus_global_sample_index_0_to_8191"
    ),
    "data_source_path": STRICT_GATE_DATA_SOURCE_PATH,
    "no_wrap": True,
    "no_requeue": True,
    "task_exception_policy": "fail_without_prompt_replacement",
}
V2R4_GATE_LAUNCH_LEDGER = "INTERLEAVED_V2R4A_GATE_LAUNCH_LEDGER.json"

pretrain_ckpt_vol = modal.Volume.from_name(
    "rl-reasoning-checkpoints", create_if_missing=True
)
production_launch_claims = modal.Dict.from_name(
    PRODUCTION_LAUNCH_CLAIM_DICT_NAME,
    create_if_missing=True,
)

app = modal.App(
    APP_NAME,
    image=base.image,
    secrets=[
        # Keep the dependency graph identical when Modal imports this module
        # locally and inside a worker. base.runtime_secrets may conditionally
        # contain Secret.from_dict(...) when a local .env was loaded, which
        # makes queued containers fail before train_hf starts because that file
        # is absent during the remote import.
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("wandb-interleave-pt-rl"),
    ],
)


@app.function(cpu=0.25, memory=256, timeout=60)
def deployment_dependency_preflight() -> dict[str, object]:
    """Verify that the deployed app can hydrate its named secrets remotely."""

    return {
        "app": APP_NAME,
        "wandb_api_key_present": bool(os.environ.get("WANDB_API_KEY")),
        "wandb_entity_present": bool(os.environ.get("WANDB_ENTITY")),
        "huggingface_token_present": bool(
            os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        ),
    }


def _safe_component(value: str, *, name: str) -> str:
    if not value or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
        raise ValueError(
            f"{name} must be one non-empty path component containing only "
            "letters, digits, '.', '_' or '-'."
        )
    return value


def _validated_logical_file_path(value: str, *, name: str) -> str:
    """Validate an absolute file path without replacing its mount alias.

    Modal exposes a Volume through a stable path such as ``/data`` while
    ``Path.resolve()`` expands it to an internal ``/__modal/volumes/...`` path.
    Precision-gate contracts intentionally bind the stable path.  Validate the
    target through the resolved path, but return the original absolute path so
    the gate and production dispatcher hash the same identity.
    """

    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{name} must be an absolute path: {path}")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return str(path)


def _validate_hf_checkpoint(path: str | Path) -> Path:
    checkpoint = Path(path)
    if not checkpoint.is_absolute():
        raise ValueError(f"HF checkpoint must be an absolute path: {checkpoint}")
    if not (checkpoint / "config.json").is_file():
        raise FileNotFoundError(f"Missing HF config.json under {checkpoint}")
    weights = sorted(checkpoint.glob("*.safetensors"))
    weights.extend(sorted(checkpoint.glob("pytorch_model*.bin")))
    if not weights:
        raise FileNotFoundError(f"Missing HF model weights under {checkpoint}")
    tokenizer_markers = (
        checkpoint / "tokenizer.json",
        checkpoint / "tokenizer_config.json",
    )
    if not any(path.is_file() for path in tokenizer_markers):
        raise FileNotFoundError(f"Missing tokenizer assets under {checkpoint}")
    return checkpoint


def _authenticated_pt_sft_hf_export(checkpoint: Path) -> dict[str, object]:
    """Validate the PT/SFT immutable FP32 HF export marker in this launcher.

    The RL Modal image intentionally does not mount the PT/SFT trainer package.
    Reproducing its small public marker contract here keeps the dependency
    surface fixed while the launcher source hash binds this validator.
    """

    marker_path = checkpoint / ".complete.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if not isinstance(marker, dict):
        raise RuntimeError(f"PT/SFT HF marker is not an object: {marker_path}")
    recorded_hash = marker.get("marker_sha256")
    core = {
        key: value
        for key, value in marker.items()
        if key != "marker_sha256"
    }
    def pt_sft_sha256(value: object) -> str:
        return hashlib.sha256(
            json.dumps(
                value,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()

    if recorded_hash != pt_sft_sha256(core):
        raise RuntimeError(f"PT/SFT HF marker hash drifted: {marker_path}")
    if marker.get("schema") != "interleaved-hf-export-v1":
        raise RuntimeError(
            f"unsupported PT/SFT HF marker schema: {marker.get('schema')!r}"
        )
    state_path = checkpoint / "interleaved_training_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if not isinstance(state, dict):
        raise RuntimeError(f"PT/SFT HF state is not an object: {state_path}")
    if marker.get("trainer_state_sha256") != _sha256(state_path):
        raise RuntimeError(f"PT/SFT HF trainer-state hash drifted: {state_path}")
    if int(marker.get("global_step", -1)) != int(
        state.get("global_step", -2)
    ):
        raise RuntimeError("PT/SFT HF marker/state global step drifted")

    files: list[dict[str, object]] = []
    for path in sorted(checkpoint.rglob("*")):
        relative = path.relative_to(checkpoint).as_posix()
        if relative == ".complete.json":
            continue
        if path.is_symlink():
            target = os.readlink(path)
            target_bytes = os.fsencode(target)
            files.append(
                {
                    "path": relative,
                    "kind": "symlink",
                    "target": target,
                    "bytes": len(target_bytes),
                    "sha256": hashlib.sha256(target_bytes).hexdigest(),
                }
            )
        elif path.is_file():
            files.append(
                {
                    "path": relative,
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    identity = {
        "file_count": len(files),
        "total_bytes": sum(int(row["bytes"]) for row in files),
        "files": files,
        "manifest_sha256": pt_sft_sha256(files),
    }
    if marker.get("export_identity") != identity:
        raise RuntimeError(f"PT/SFT HF export payload drifted: {checkpoint}")
    return {
        "schema": str(marker["schema"]),
        "marker_sha256": str(recorded_hash),
        "marker_file_sha256": _sha256(marker_path),
        "global_step": int(marker["global_step"]),
        "payload_manifest_sha256": str(identity["manifest_sha256"]),
    }


def _validate_authenticated_fp32_hf_checkpoint(
    path: str | Path,
) -> tuple[Path, dict[str, object]]:
    """Require a fully authenticated canonical FP32 HF export."""

    checkpoint = _validate_hf_checkpoint(path).resolve(strict=True)
    if str(Path(base.MILES_DIR)) not in sys.path:
        sys.path.insert(0, str(Path(base.MILES_DIR)))
    from tools.convert_fsdp_to_hf import (
        validate_committed_hf_export,
        validate_safetensors_fp32,
    )

    if (checkpoint / ".complete.json").is_file():
        authentication = _authenticated_pt_sft_hf_export(checkpoint)
    elif (checkpoint / "COMMITTED.json").is_file():
        marker = validate_committed_hf_export(checkpoint)
        authentication = {
            "schema": str(marker["schema"]),
            "marker_sha256": str(marker["commit_sha256"]),
            "marker_file_sha256": _sha256(checkpoint / "COMMITTED.json"),
            "source_checkpoint": marker["source_checkpoint"],
            "payload_manifest_sha256": _canonical_json_sha256(
                marker["payload"]
            ),
        }
    else:
        raise RuntimeError(
            "production RL requires an authenticated canonical FP32 HF export; "
            f"neither .complete.json nor COMMITTED.json exists under {checkpoint}"
        )
    precision = validate_safetensors_fp32(checkpoint)
    config_path = checkpoint / "config.json"
    vocab_path = checkpoint / "vocab.json"
    tokenizer_path = checkpoint / "tokenizer.json"
    tokenizer_py_path = checkpoint / "tokenizer.py"
    tokenizer_config_path = checkpoint / "tokenizer_config.json"
    special_tokens_path = checkpoint / "special_tokens_map.json"
    # The canonical chess export is deliberately a custom slow tokenizer.
    # Its executable tokenizer is tokenizer.py and its exact WordLevel mapping
    # is vocab.json.  save_hf_tokenizer() therefore does not emit the fast-
    # tokenizer artifact tokenizer.json.  Requiring that optional file rejects
    # the real authenticated PT/SFT exports before the runtime tokenizer can be
    # exercised.  Every required file below is already content-bound by the HF
    # export marker validated above.
    required_tokenizer_files = (
        vocab_path,
        tokenizer_py_path,
        tokenizer_config_path,
        special_tokens_path,
    )
    if any(
        not path.is_file() or path.is_symlink()
        for path in required_tokenizer_files
    ):
        raise RuntimeError(
            "production RL requires regular vocab.json, tokenizer.py, "
            "tokenizer_config.json, and special_tokens_map.json"
        )
    config = json.loads(config_path.read_text())
    vocab_evidence = _validate_rl_tokenizer_vocab(vocab_path)
    if tokenizer_path.exists():
        if not tokenizer_path.is_file() or tokenizer_path.is_symlink():
            raise RuntimeError(
                "optional production tokenizer.json must be a regular file"
            )
        tokenizer_evidence = _validate_rl_tokenizer_vocab(tokenizer_path)
        if (
            tokenizer_evidence["vocab_mapping_sha256"]
            != vocab_evidence["vocab_mapping_sha256"]
        ):
            raise RuntimeError(
                "tokenizer.json disagrees with authenticated vocab.json"
            )
        vocab_evidence["tokenizer_json_sha256"] = _sha256(tokenizer_path)
    else:
        vocab_evidence["tokenizer_json_sha256"] = None
    vocab_evidence["tokenizer_py_sha256"] = _sha256(tokenizer_py_path)
    vocab_evidence["tokenizer_backend"] = "custom_slow_tokenizer"
    tokenizer_config = json.loads(tokenizer_config_path.read_text())
    _validate_rl_tokenizer_and_model_config(tokenizer_config, config)
    special_tokens = json.loads(special_tokens_path.read_text())
    expected_special_tokens = {
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "pad_token": "<bos>",
        "unk_token": "<unk>",
        "env_token": "<call_env>",
    }
    if any(
        special_tokens.get(key) != value
        for key, value in expected_special_tokens.items()
    ):
        raise RuntimeError("production special-token map drifted")
    dtype_fields = {
        key: config[key]
        for key in ("dtype", "torch_dtype")
        if key in config
    }
    if not dtype_fields or any(
        value not in {"float32", "float", "torch.float32"}
        for value in dtype_fields.values()
    ):
        raise RuntimeError(
            "canonical production HF config must advertise FP32 tensors: "
            f"{dtype_fields}"
        )
    return checkpoint, {
        **authentication,
        "precision": precision,
        "config_dtype": dtype_fields,
        "tokenizer_contract": {
            **vocab_evidence,
            "model_max_length": 2_048,
            "tokenizer_config_sha256": _sha256(tokenizer_config_path),
            "model_config_sha256": _sha256(config_path),
        },
    }


def _validate_rl_tokenizer_vocab(tokenizer_path: Path) -> dict[str, object]:
    """Validate the complete exact 85-token WordLevel vocabulary."""

    tokenizer_payload = json.loads(tokenizer_path.read_text())
    tokenizer_model = (
        tokenizer_payload.get("model")
        if isinstance(tokenizer_payload, dict)
        else None
    )
    vocab = (
        tokenizer_model.get("vocab")
        if isinstance(tokenizer_model, dict)
        else tokenizer_payload
    )
    if not isinstance(vocab, dict):
        raise RuntimeError("production tokenizer.json lacks model.vocab")
    normalized = validate_exact_rl_vocab(vocab)
    observed_ids = {
        token: normalized[token]
        for token in EXPECTED_RL_TOKEN_IDS
    }
    mapping_sha256 = vocab_mapping_sha256(normalized)
    if mapping_sha256 != EXPECTED_RL_VOCAB_MAPPING_SHA256:
        raise RuntimeError("production RL tokenizer contract digest drifted")
    return {
        "schema": "chess-rl-exact-tokenizer-contract-v1",
        "vocab_size": len(normalized),
        "token_ids": observed_ids,
        "vocab_mapping_sha256": mapping_sha256,
        "vocab_sha256": _sha256(tokenizer_path),
    }


def _validate_rl_tokenizer_and_model_config(
    tokenizer_config: dict[str, object],
    model_config: dict[str, object],
) -> None:
    expected_tokenizer_config = {
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "unk_token": "<unk>",
        "pad_token": "<bos>",
        "model_max_length": 2_048,
    }
    for key, expected in expected_tokenizer_config.items():
        if tokenizer_config.get(key) != expected:
            raise RuntimeError(
                f"production tokenizer config {key} drifted: "
                f"{tokenizer_config.get(key)!r} != {expected!r}"
            )
    expected_wrapper_config = {
        "tokenizer_class": "HFTokenizerWrapper",
        "use_fast": False,
        "lan_tokenizer_class": "LanTokenizerSFT",
        "env_token": "<call_env>",
        "env_id": EXPECTED_RL_TOKEN_IDS["<call_env>"],
    }
    for key, expected in expected_wrapper_config.items():
        if tokenizer_config.get(key) != expected:
            raise RuntimeError(
                f"production tokenizer wrapper config {key} drifted: "
                f"{tokenizer_config.get(key)!r} != {expected!r}"
            )
    auto_map = tokenizer_config.get("auto_map")
    if not isinstance(auto_map, dict) or auto_map.get("AutoTokenizer") != [
        "tokenizer.HFTokenizerWrapper",
        None,
    ]:
        raise RuntimeError("production tokenizer AutoTokenizer mapping drifted")
    lan_config = tokenizer_config.get("lan_config")
    expected_lan_config = {
        "include_move_numbers": False,
        "include_black_tripledots": False,
        "bos": "<bos>",
        "eos": "<eos>",
        "unk": "<unk>",
        "pad": "<bos>",
        "keep_result": False,
        "include_env_tokens": True,
        "include_reward_tokens": False,
    }
    if not isinstance(lan_config, dict) or any(
        lan_config.get(key) != value
        for key, value in expected_lan_config.items()
    ):
        raise RuntimeError("production LAN tokenizer configuration drifted")
    expected_model_config = {
        "vocab_size": EXPECTED_RL_VOCAB_SIZE,
        "bos_token_id": EXPECTED_RL_TOKEN_IDS["<bos>"],
        "eos_token_id": EXPECTED_RL_TOKEN_IDS["<eos>"],
        "pad_token_id": EXPECTED_RL_TOKEN_IDS["<bos>"],
        "max_position_embeddings": 2_048,
    }
    for key, expected in expected_model_config.items():
        if model_config.get(key) != expected:
            raise RuntimeError(
                f"production model config {key} drifted: "
                f"{model_config.get(key)!r} != {expected!r}"
            )


def _validate_checkpoint_context(
    checkpoint: Path,
    *,
    requested_context_len: int,
    require_exact: bool = False,
) -> int:
    """Reject rollout positions beyond the checkpoint's native context."""
    config = json.loads((checkpoint / "config.json").read_text())
    native_context_len = config.get("max_position_embeddings")
    if not isinstance(native_context_len, int) or native_context_len <= 0:
        raise ValueError(
            f"HF checkpoint config must declare a positive max_position_embeddings: {checkpoint / 'config.json'}"
        )
    if requested_context_len > native_context_len:
        raise ValueError(
            f"Requested rollout context {requested_context_len} exceeds checkpoint native context "
            f"{native_context_len}: {checkpoint}"
        )
    if require_exact and native_context_len != requested_context_len:
        raise ValueError(
            "Exact-context profile requires checkpoint native context to equal "
            f"the requested policy/SGLang context: native={native_context_len} "
            f"requested={requested_context_len} checkpoint={checkpoint}"
        )
    return native_context_len


def build_train_command(
    *,
    hf_checkpoint: str,
    run_name: str,
    model_id: str = MODEL_ID,
    num_rollout: int,
    dynamic_filter: bool,
    rollout_seed: int,
    save_interval: int = 40,
    eval_interval: int = 0,
    resume_path: str = "",
    resume_step: int = 0,
    wandb_project: str = "chess_interleave_50m",
    wandb_group: str = "core",
    wandb_run_id: str = "",
    max_tokens_per_gpu: int = SMALL_MODEL_MAX_TOKENS_DEFAULT,
    sglang_server_concurrency: int = SGLANG_SERVER_CONCURRENCY_DEFAULT,
    deterministic_inference: bool = False,
    rollout_only: bool = False,
    canary: bool = False,
    train_file: str = BALANCED_TRAIN_FILE,
    train_file_sha256: str = BALANCED_TRAIN_SHA256,
    lr: str = "1e-5",
    kl_loss_type: str = KL_LOSS_TYPE_DEFAULT,
    data_source_path: str = DEFAULT_DATA_SOURCE_PATH,
    deterministic_seed_by_sample_index: bool = False,
    fault_tolerance: bool = True,
    rollout_health_check_interval: float = 30.0,
    log_passrate: bool = True,
    rollout_max_prompt_len: int = ROLLOUT_MAX_PROMPT_LEN_DEFAULT,
    rollout_max_response_len: int = ROLLOUT_MAX_RESPONSE_LEN_DEFAULT,
    rollout_max_context_len: int = ROLLOUT_MAX_CONTEXT_LEN_DEFAULT,
    initial_adam_checkpoint: str = "",
    initial_adam_completion_sha256: str = "",
    initial_adam_source_tree_sha256: str = "",
    initial_adam_step: int = 0,
) -> list[str]:
    """Build the explicit, verified Miles/SGLang command.

    Keeping this pure makes configuration drift visible in unit tests and in
    the dry-run output before an expensive H200 allocation is requested.
    """
    _safe_component(run_name, name="run_name")
    _safe_component(model_id, name="model_id")
    if wandb_run_id:
        _safe_component(wandb_run_id, name="wandb_run_id")
    if num_rollout <= 0:
        raise ValueError("num_rollout must be positive")
    if rollout_seed < 0:
        raise ValueError("rollout_seed must be non-negative")
    if save_interval < 0 or eval_interval < 0:
        raise ValueError("save_interval and eval_interval must be non-negative")
    if bool(resume_path) != bool(resume_step):
        raise ValueError("resume_path and resume_step must be provided together")
    if deterministic_seed_by_sample_index and not deterministic_inference:
        raise ValueError(
            "sample-index deterministic seeding requires deterministic "
            "inference"
        )
    if rollout_health_check_interval <= 0:
        raise ValueError(
            "rollout_health_check_interval must be positive"
        )
    if rollout_max_prompt_len <= 0:
        raise ValueError("rollout_max_prompt_len must be positive")
    if rollout_max_response_len <= 0:
        raise ValueError("rollout_max_response_len must be positive")
    if rollout_max_context_len <= 0:
        raise ValueError("rollout_max_context_len must be positive")
    if rollout_max_prompt_len >= rollout_max_context_len:
        raise ValueError(
            "rollout_max_prompt_len must be smaller than "
            "rollout_max_context_len"
        )
    if rollout_max_response_len > rollout_max_context_len:
        raise ValueError(
            "rollout_max_response_len cannot exceed "
            "rollout_max_context_len"
        )
    if model_id == CONTEXT2048_MODEL_ID and (
        rollout_max_prompt_len,
        rollout_max_response_len,
        rollout_max_context_len,
    ) != (512, 1_536, 2_048):
        raise ValueError(
            f"{CONTEXT2048_MODEL_ID} requires the pinned rollout geometry "
            "prompt=512, response=1536, context=2048"
        )
    if base.GPU_TYPE != "H200" or base.GPUS_PER_NODE != 8:
        raise RuntimeError(
            "small-model-h200 profile requires exactly 8 H200 GPUs"
        )
    if SMALL_MODEL_HOST_MEMORY_GB != 192:
        raise RuntimeError(
            "small-model-h200 profile requires exactly 192 GB host memory"
        )
    if max_tokens_per_gpu not in SMALL_MODEL_MAX_TOKENS_ALLOWED:
        allowed = ", ".join(
            f"{value:,}" for value in sorted(SMALL_MODEL_MAX_TOKENS_ALLOWED)
        )
        raise ValueError(
            f"max_tokens_per_gpu must be one of the benchmarked profile "
            f"budgets: {allowed}"
        )
    if kl_loss_type not in KL_LOSS_TYPE_ALLOWED:
        allowed = ", ".join(sorted(KL_LOSS_TYPE_ALLOWED))
        raise ValueError(f"kl_loss_type must be one of: {allowed}")
    if sglang_server_concurrency not in SGLANG_SERVER_CONCURRENCY_ALLOWED:
        allowed = ", ".join(
            str(value)
            for value in sorted(SGLANG_SERVER_CONCURRENCY_ALLOWED)
        )
        raise ValueError(
            "sglang_server_concurrency must be one of the staged profile "
            f"values: {allowed}"
        )
    initial_adam_values = (
        initial_adam_checkpoint,
        initial_adam_completion_sha256,
        initial_adam_source_tree_sha256,
        initial_adam_step,
    )
    if any(value not in ("", 0) for value in initial_adam_values):
        if not all(value not in ("", 0) for value in initial_adam_values):
            raise ValueError(
                "initial Adam import requires checkpoint, completion SHA-256, "
                "source-tree SHA-256, and step together"
            )
        if initial_adam_step <= 0:
            raise ValueError("initial Adam source step must be positive")
        for label, digest in (
            ("completion", initial_adam_completion_sha256),
            ("source-tree", initial_adam_source_tree_sha256),
        ):
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError(
                    f"initial Adam {label} SHA-256 must be lowercase hexadecimal"
                )

    command = [
        sys.executable,
        "-m",
        "chess_rl_miles.scripts.run_chess_miles",
        "--miles-dir",
        base.MILES_DIR,
        "--project-dir",
        base.PROJECT_DIR,
        "--hf-checkpoint",
        hf_checkpoint,
        "--model-id",
        model_id,
        "--small-model-profile",
        POLICY_UPDATE_PROFILE,
        "--no-gradient-checkpointing",
        "--run-name",
        run_name,
        "--io-layout",
        "flat",
        "--data-dir",
        base.DATA_DIR,
        "--train-file",
        train_file,
        "--train-file-sha256",
        train_file_sha256,
        "--data-source-path",
        data_source_path,
        "--save-dir",
        RAW_RL_ROOT,
        "--rollout-seed",
        str(rollout_seed),
        "--num-rollout",
        str(num_rollout),
        "--num-steps-per-rollout",
        "1",
        "--save-interval",
        str(save_interval),
        "--rollout-batch-size",
        "256",
        "--n-samples-per-prompt",
        "8",
        "--over-sampling-batch-size",
        "256",
        "--global-batch-size",
        "2048",
        "--policy-loss-agg-mode",
        "token-mean",
        "--no-cispo",
        "--optim-tag",
        "adamw",
        "--lr",
        str(lr),
        "--adam-beta1",
        "0.9",
        "--adam-beta2",
        "0.999",
        "--adam-eps",
        "1e-8",
        "--weight-decay",
        "0.01",
        "--kl-loss-coef",
        "0.001",
        "--kl-loss-type",
        str(kl_loss_type),
        "--rollout-max-prompt-len",
        str(rollout_max_prompt_len),
        "--rollout-prompt-reserved-prefix-tokens",
        "1",
        "--rollout-max-response-len",
        str(rollout_max_response_len),
        "--rollout-max-context-len",
        str(rollout_max_context_len),
        "--chess-context-margin-tokens",
        "0",
        "--rollout-temperature",
        "1.0",
        "--rollout-top-p",
        "1.0",
        "--actor-num-nodes",
        "1",
        "--actor-num-gpus-per-node",
        str(base.GPUS_PER_NODE),
        "--rollout-num-gpus-per-engine",
        "1",
        "--sglang-server-concurrency",
        str(sglang_server_concurrency),
        "--sglang-dtype",
        "bfloat16",
        "--sglang-context-length",
        str(rollout_max_context_len),
        "--eval-sglang-server-concurrency",
        "16",
        "--max-tokens-per-gpu",
        str(max_tokens_per_gpu),
        "--attn-implementation",
        "flash_attention_3",
        "--batched-rollout",
        "--sglang-token-id-only",
        "--use-miles-router",
        "--rollout-health-check-interval",
        str(rollout_health_check_interval),
        "--save-rollouts",
        "--wandb-project",
        wandb_project,
        "--wandb-group",
        wandb_group,
        "--wandb-team",
        "jingyanshen-new-york-university",
    ]
    if wandb_run_id:
        command.extend(["--wandb-run-id", wandb_run_id])
    if initial_adam_checkpoint:
        command.extend(
            [
                "--initial-adam-checkpoint",
                initial_adam_checkpoint,
                "--initial-adam-completion-sha256",
                initial_adam_completion_sha256,
                "--initial-adam-source-tree-sha256",
                initial_adam_source_tree_sha256,
                "--initial-adam-step",
                str(initial_adam_step),
            ]
        )
    if dynamic_filter:
        command.append("--dynamic-filter")
    if deterministic_inference:
        command.append("--sglang-enable-deterministic-inference")
    if deterministic_seed_by_sample_index:
        command.append("--chess-deterministic-seed-by-sample-index")
    if not fault_tolerance:
        command.append("--no-use-fault-tolerance")
    if not log_passrate:
        command.append("--no-log-passrate")
    if rollout_only:
        command.append("--debug-rollout-only")
    if eval_interval:
        command.extend(["--eval-interval", str(eval_interval)])
    if resume_path:
        command.extend(["--load", resume_path])
        command.extend(
            [
                "--",
                "--ckpt-step",
                str(resume_step),
                "--start-rollout-id",
                str(resume_step),
            ]
        )
    return command


def _runtime_env(
    *,
    run_name: str,
    deterministic_seed_mode: str | None = None,
    precision_gate_leg: int | None = None,
) -> dict[str, str]:
    env = os.environ.copy()
    cpu_threads = int(base.CPU_COUNT)
    env["PYTHONPATH"] = (
        f"{base.PROJECT_DIR}:{base.MILES_DIR}:{env.get('PYTHONPATH', '')}"
    )
    env["PYTHONUNBUFFERED"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "true"
    env["OMP_NUM_THREADS"] = str(cpu_threads)
    env["RAYON_NUM_THREADS"] = str(cpu_threads)
    env["SGLANG_CPU_THREAD_POOL_SIZE"] = str(cpu_threads)
    env["MILES_DISABLE_TQDM"] = "1"
    env["TQDM_DISABLE"] = "1"
    env["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    env["RAY_DEDUP_LOGS"] = "0"
    env["MILES_EXPERIMENTAL_ROLLOUT_REFACTOR"] = "1"
    env["CHESS_RL_MILES_ARTIFACT_ROOT"] = f"{RAW_RL_ROOT}/{run_name}"
    if precision_gate_leg is not None:
        if precision_gate_leg not in {1, 2}:
            raise ValueError("precision_gate_leg must be 1 or 2")
        env["CHESS_RL_MILES_PRECISION_GATE_LEG"] = str(
            precision_gate_leg
        )
    if deterministic_seed_mode is not None:
        if deterministic_seed_mode != "sample-index":
            raise ValueError("unsupported deterministic seed mode")
        # Ray workers inherit the head process environment, not environment
        # variables added later by the adapter's Miles subprocess.  Set this
        # before starting Ray so StrictEpochRolloutDataSource and rollout
        # workers observe the same frozen sample-index seeding contract.
        env["CHESS_RL_MILES_DETERMINISTIC_SEED_MODE"] = (
            deterministic_seed_mode
        )
    env.setdefault("HF_HOME", base.HF_CACHE_DIR)
    return env


def _verify_ray_worker_gate_environment(
    run_name: str = "v2r4a-ray-env-preflight",
    ) -> dict[str, object]:
    """Prove a live Ray worker inherits the gate environment before GPU use."""

    _safe_component(run_name, name="run_name")
    env = _runtime_env(
        run_name=run_name,
        deterministic_seed_mode="sample-index",
    )
    expected_artifact_root = f"{RAW_RL_ROOT}/{run_name}"
    if (
        env.get("CHESS_RL_MILES_DETERMINISTIC_SEED_MODE")
        != "sample-index"
        or env.get("CHESS_RL_MILES_ARTIFACT_ROOT")
        != expected_artifact_root
    ):
        raise RuntimeError("gate environment was not set before Ray startup")
    base._cleanup_runtime()
    try:
        base._start_ray_head(env, cpu_threads=2, num_gpus=0)
        env["RAY_ADDRESS"] = base.RAY_ADDRESS
        probe = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import json, os, ray; "
                    "ray.init(address=os.environ['RAY_ADDRESS'], "
                    "ignore_reinit_error=True); "
                    "f=ray.remote(lambda: {"
                    "'seed_mode': os.environ.get("
                    "'CHESS_RL_MILES_DETERMINISTIC_SEED_MODE'), "
                    "'artifact_root': os.environ.get("
                    "'CHESS_RL_MILES_ARTIFACT_ROOT')}); "
                    "value=ray.get(f.remote()); "
                    "print('V2R4_RAY_ENV=' + "
                    "json.dumps(value, sort_keys=True), flush=True); "
                    "ray.shutdown()"
                ),
            ],
            cwd=base.PROJECT_DIR,
            env=env,
            text=True,
            capture_output=True,
            timeout=120,
            check=False,
        )
        if probe.returncode:
            raise RuntimeError(
                "Ray worker gate-environment preflight failed: "
                + (probe.stderr or probe.stdout)[-2_000:]
            )
        markers = [
            line.removeprefix("V2R4_RAY_ENV=")
            for line in probe.stdout.splitlines()
            if line.startswith("V2R4_RAY_ENV=")
        ]
        if len(markers) != 1:
            raise RuntimeError(
                "Ray worker gate-environment preflight emitted an "
                f"unexpected marker count: {len(markers)}"
            )
        observed = json.loads(markers[0])
        expected = {
            "seed_mode": "sample-index",
            "artifact_root": expected_artifact_root,
        }
        if observed != expected:
            raise RuntimeError(
                f"Ray worker gate environment drifted: {observed!r}"
            )
        return {
            **observed,
            "gpu_allocated": False,
        }
    finally:
        base._cleanup_runtime()


def _validated_checkpoint_commit(
    checkpoint: Path,
    *,
    expected_step: int,
) -> dict[str, object]:
    marker_path = checkpoint / CHECKPOINT_COMMIT_MARKER
    marker = json.loads(marker_path.read_text())
    core = {
        key: value
        for key, value in marker.items()
        if key != "commit_sha256"
    }
    if marker.get("commit_sha256") != _canonical_json_sha256(core):
        raise ValueError(f"Checkpoint commit self-hash mismatch: {marker_path}")
    if (
        marker.get("schema") != "miles-fsdp-checkpoint-commit-v1"
        or marker.get("iteration") != expected_step
        or marker.get("optimizer_included") is not True
        or marker.get("rng_included") is not True
        or marker.get("rollout_state_included") is not True
    ):
        raise ValueError(f"Checkpoint commit contract mismatch: {marker_path}")

    roots = [
        checkpoint / "model",
        checkpoint / "optimizer",
        checkpoint / "lr_scheduler",
    ]
    files: list[Path] = []
    for root in roots:
        if not (root / ".metadata").is_file():
            raise FileNotFoundError(root / ".metadata")
        files.extend(path for path in root.rglob("*") if path.is_file())
    world_size = marker.get("world_size")
    if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size <= 0:
        raise ValueError(f"Invalid checkpoint world size: {world_size!r}")
    expected_rng = [
        checkpoint / f"rng_rank_{rank:05d}.pt"
        for rank in range(world_size)
    ]
    if sorted(checkpoint.glob("rng_rank_*.pt")) != expected_rng:
        raise FileNotFoundError("Checkpoint distributed RNG inventory is incomplete")
    files.extend(
        [
            checkpoint / "meta.json",
            checkpoint / "rollout_state.pt",
            *expected_rng,
        ]
    )
    if not all(path.is_file() for path in files):
        raise FileNotFoundError("Checkpoint root payload is incomplete")
    observed_files: set[Path] = set()
    observed_directories: set[Path] = set()
    for path in checkpoint.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Checkpoint contains a forbidden symlink: {path}")
        if path.is_file():
            observed_files.add(path)
        elif path.is_dir():
            observed_directories.add(path)
        else:
            raise ValueError(f"Checkpoint contains an unsupported entry: {path}")
    expected_files = set(files) | {marker_path}
    if observed_files != expected_files:
        unexpected = sorted(
            path.relative_to(checkpoint).as_posix()
            for path in observed_files - expected_files
        )
        missing = sorted(
            path.relative_to(checkpoint).as_posix()
            for path in expected_files - observed_files
        )
        raise ValueError(
            "Checkpoint contains files not bound by COMMITTED.json: "
            f"unexpected={unexpected} missing={missing}"
        )
    expected_directories: set[Path] = set()
    for path in expected_files:
        parent = path.parent
        while parent != checkpoint:
            expected_directories.add(parent)
            parent = parent.parent
    if observed_directories != expected_directories:
        unexpected = sorted(
            path.relative_to(checkpoint).as_posix()
            for path in observed_directories - expected_directories
        )
        missing = sorted(
            path.relative_to(checkpoint).as_posix()
            for path in expected_directories - observed_directories
        )
        raise ValueError(
            "Checkpoint contains directories not bound by COMMITTED.json: "
            f"unexpected={unexpected} missing={missing}"
        )
    actual_payload = []
    for path in sorted(set(files)):
        size = path.stat().st_size
        if size <= 0:
            raise ValueError(f"Empty checkpoint payload: {path}")
        actual_payload.append(
            {
                "path": path.relative_to(checkpoint).as_posix(),
                "bytes": size,
                "sha256": _sha256(path),
            }
        )
    if marker.get("payload") != actual_payload:
        raise ValueError(
            f"Checkpoint payload disagrees with commit marker: {checkpoint}"
        )
    return marker


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_directory_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish an immutable directory without a race window."""

    if source.parent != destination.parent:
        raise ValueError(
            f"immutable publication requires one filesystem: {source} -> {destination}"
        )
    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    if sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        function = libc.renameat2
        function.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        function.restype = ctypes.c_int
        result = function(-100, source_bytes, -100, destination_bytes, 1)
    elif sys.platform == "darwin" and hasattr(libc, "renamex_np"):
        function = libc.renamex_np
        function.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        function.restype = ctypes.c_int
        result = function(source_bytes, destination_bytes, 0x00000004)
    else:
        raise RuntimeError(
            "platform lacks atomic no-replace directory rename; refusing unsafe publication"
        )
    if result == 0:
        return
    error = ctypes.get_errno()
    if error in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            error,
            f"refusing to replace immutable directory: {destination}",
            str(destination),
        )
    raise OSError(error, os.strerror(error), f"{source} -> {destination}")


def _quarantine_checkpoint_path(root: Path, path: Path) -> Path:
    quarantine = root / "_quarantine"
    quarantine.mkdir(parents=True, exist_ok=True)
    destination = quarantine / f"{path.name}.{os.getpid()}.{len(list(quarantine.iterdir()))}"
    os.replace(path, destination)
    _fsync_directory(quarantine)
    _fsync_directory(root)
    return destination


def _reconcile_modal_checkpoint_root(run_root: str | Path) -> int | None:
    """Select the newest authenticated commit before building resume args."""

    root = Path(run_root)
    if not root.exists():
        return None

    for staging in sorted(root.glob(".iter_*.incomplete")):
        try:
            step = int(
                staging.name.removeprefix(".iter_").removesuffix(
                    ".incomplete"
                )
            )
        except ValueError:
            _quarantine_checkpoint_path(root, staging)
            continue
        final = root / f"iter_{step:07d}"
        if final.exists():
            _quarantine_checkpoint_path(root, staging)
            continue
        try:
            _validated_checkpoint_commit(staging, expected_step=step)
        except Exception:
            _quarantine_checkpoint_path(root, staging)
        else:
            try:
                _rename_directory_noreplace(staging, final)
            except FileExistsError:
                _quarantine_checkpoint_path(root, staging)
            else:
                _fsync_directory(root)

    committed: list[int] = []
    for checkpoint in sorted(root.glob("iter_*")):
        if not checkpoint.is_dir():
            continue
        try:
            step = int(checkpoint.name.removeprefix("iter_"))
            _validated_checkpoint_commit(checkpoint, expected_step=step)
            metadata = json.loads((checkpoint / "meta.json").read_text())
            if (
                int(metadata["iteration"]) != step
                or int(metadata["next_rollout_id"]) != step
            ):
                raise ValueError("checkpoint accounting does not match its path")
        except Exception as exc:
            raise RuntimeError(
                "Published checkpoint directory is invalid and cannot be "
                f"silently quarantined or reused: {checkpoint}"
            ) from exc
        else:
            committed.append(step)

    tracker = root / "latest_checkpointed_iteration.txt"
    if not committed:
        if tracker.exists():
            raise RuntimeError(
                f"Checkpoint tracker exists without an authenticated commit: {tracker}"
            )
        return None
    latest = max(committed)
    observed = None
    if tracker.exists():
        try:
            observed = int(tracker.read_text().strip())
        except (OSError, ValueError):
            observed = None
    if observed != latest:
        temporary = tracker.with_name(f".{tracker.name}.{os.getpid()}.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(f"{latest}\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, tracker)
        _fsync_directory(root)
    return latest


def _inspect_dcp_fp32_precision(checkpoint: Path) -> dict[str, object]:
    """Inspect actual DCP tensor metadata rather than trusting JSON claims."""

    import torch
    import torch.distributed.checkpoint as dcp

    def tensor_rows(path: Path) -> list[dict[str, str]]:
        metadata = dcp.FileSystemReader(path).read_metadata()
        rows = []
        for name, storage in sorted(metadata.state_dict_metadata.items()):
            properties = getattr(storage, "properties", None)
            dtype = getattr(properties, "dtype", None)
            if dtype is not None:
                rows.append(
                    {
                        "name": str(name),
                        "dtype": str(dtype).removeprefix("torch."),
                    }
                )
        return rows

    model_rows = tensor_rows(checkpoint / "model")
    optimizer_rows = tensor_rows(checkpoint / "optimizer")
    floating_names = {
        str(dtype).removeprefix("torch.")
        for dtype in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        )
    }
    model_floating = [
        row for row in model_rows if row["dtype"] in floating_names
    ]
    optimizer_floating = [
        row for row in optimizer_rows if row["dtype"] in floating_names
    ]
    if not model_floating:
        raise RuntimeError("DCP model metadata contains no floating tensors")
    if any(row["dtype"] != "float32" for row in model_floating):
        raise RuntimeError(
            "DCP model metadata contains non-FP32 floating tensors"
        )
    if not optimizer_floating or any(
        row["dtype"] != "float32" for row in optimizer_floating
    ):
        raise RuntimeError(
            "DCP optimizer metadata must contain only FP32 floating tensors"
        )
    exp_avg = [
        row for row in optimizer_rows if row["name"].endswith(".exp_avg")
    ]
    exp_avg_sq = [
        row
        for row in optimizer_rows
        if row["name"].endswith(".exp_avg_sq")
    ]
    if not exp_avg or not exp_avg_sq:
        raise RuntimeError(
            "DCP optimizer metadata lacks Adam exp_avg/exp_avg_sq tensors"
        )
    if any(row["dtype"] != "float32" for row in exp_avg + exp_avg_sq):
        raise RuntimeError("DCP Adam moments are not all FP32")

    dtype_manifest = {
        "model": model_rows,
        "optimizer": optimizer_rows,
    }
    return {
        "model_tensor_count": len(model_rows),
        "model_floating_tensor_count": len(model_floating),
        "optimizer_tensor_count": len(optimizer_rows),
        "optimizer_floating_tensor_count": len(optimizer_floating),
        "adam_exp_avg_count": len(exp_avg),
        "adam_exp_avg_sq_count": len(exp_avg_sq),
        "all_floating_model_and_optimizer_tensors_fp32": True,
        "dtype_manifest_sha256": _canonical_json_sha256(dtype_manifest),
    }


def _latest_complete_checkpoint_step(run_root: str | Path) -> int | None:
    """Return the tracker step only after its immutable model files are ready.

    Miles writes the distributed model, RNG state, and metadata before updating
    ``latest_checkpointed_iteration.txt``.  Rechecking those completion markers
    here keeps an incremental Modal Volume commit from publishing a partially
    written checkpoint if the tracker is observed during the final rank
    barrier.
    """

    root = Path(run_root)
    tracker = root / "latest_checkpointed_iteration.txt"
    try:
        step = int(tracker.read_text().strip())
    except (FileNotFoundError, OSError, ValueError):
        return None
    if step <= 0:
        return None

    checkpoint = root / f"iter_{step:07d}"
    metadata_path = checkpoint / "meta.json"
    required_files = (
        checkpoint / "model" / ".metadata",
        checkpoint / "optimizer" / ".metadata",
        checkpoint / "lr_scheduler" / ".metadata",
        metadata_path,
        checkpoint / CHECKPOINT_COMMIT_MARKER,
    )
    if not all(path.is_file() for path in required_files):
        return None
    try:
        metadata = json.loads(metadata_path.read_text())
        metadata_step = int(metadata["iteration"])
        next_rollout_id = int(metadata["next_rollout_id"])
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if metadata_step != step or next_rollout_id != step:
        return None
    try:
        _validated_checkpoint_commit(
            checkpoint,
            expected_step=step,
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return step


def _commit_new_checkpoint_if_ready(
    *,
    run_root: str | Path,
    volume,
    published_through_step: int,
) -> int:
    """Commit a newly completed checkpoint without interrupting training.

    A transient Volume API failure is retried on the next poll.  It must not
    terminate a healthy multi-hour GPU job; the existing final commit remains
    authoritative and will still fail the Modal function if publication never
    recovers.
    """

    root = Path(run_root)
    root.parent.mkdir(parents=True, exist_ok=True)
    lock_path = (
        root.parent / f".{root.name}{CHECKPOINT_VOLUME_COMMIT_LOCK_SUFFIX}"
    )
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(lock_path, flags, 0o644)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        tracker = root / "latest_checkpointed_iteration.txt"
        try:
            advertised_step = int(tracker.read_text().strip())
        except (FileNotFoundError, OSError, ValueError):
            return published_through_step
        if advertised_step <= published_through_step:
            return published_through_step
        # The Miles writer holds this same lock for the entire distributed
        # staging/publication interval. A leftover staging tree means a writer
        # died; never snapshot it into the durable Volume.
        if any(root.glob(".iter_*.incomplete")):
            return published_through_step
        step = _latest_complete_checkpoint_step(root)
        if step is None or step <= published_through_step:
            return published_through_step
        try:
            volume.commit()
        except Exception as exc:
            print(
                "[interleave-rl] checkpoint-publish-retry "
                f"step={step} error={type(exc).__name__}: {exc}",
                flush=True,
            )
            return published_through_step
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)
    print(
        f"[interleave-rl] checkpoint-published step={step} root={run_root}",
        flush=True,
    )
    return step


def _run_training_with_checkpoint_commits(
    command: list[str],
    *,
    env: dict[str, str],
    cwd: str | Path,
    run_root: str | Path,
    volume,
    initial_published_step: int = 0,
    poll_seconds: float = CHECKPOINT_COMMIT_POLL_SECONDS,
    lease_heartbeat: Callable[[], None] | None = None,
    lease_refresh_seconds: float = PRODUCTION_LEASE_REFRESH_SECONDS,
    runtime_cleanup: Callable[[], None] | None = None,
) -> tuple[int, int]:
    """Run Miles while making each periodic checkpoint visible cross-container."""

    if poll_seconds <= 0:
        raise ValueError("poll_seconds must be positive")
    if lease_refresh_seconds <= 0:
        raise ValueError("lease_refresh_seconds must be positive")
    process: Any = None
    try:
        process = subprocess.Popen(
            command,
            env=env,
            cwd=cwd,
            start_new_session=True,
        )
        published_through_step = initial_published_step
        last_lease_refresh = time.monotonic()
        while True:
            try:
                returncode = process.wait(timeout=poll_seconds)
            except subprocess.TimeoutExpired:
                published_through_step = _commit_new_checkpoint_if_ready(
                    run_root=run_root,
                    volume=volume,
                    published_through_step=published_through_step,
                )
                now = time.monotonic()
                if (
                    lease_heartbeat is not None
                    and now - last_lease_refresh >= lease_refresh_seconds
                ):
                    lease_heartbeat()
                    last_lease_refresh = now
                continue
            if returncode:
                # A failed child may have left a .incomplete tree. Never
                # publish arbitrary failed-process state with a mount-wide
                # Volume commit.
                return returncode, published_through_step
            published_through_step = _commit_new_checkpoint_if_ready(
                run_root=run_root,
                volume=volume,
                published_through_step=published_through_step,
            )
            return returncode, published_through_step
    except BaseException:
        if process is not None:
            pid = getattr(process, "pid", None)
            try:
                if isinstance(pid, int) and pid > 0:
                    os.killpg(pid, signal.SIGTERM)
                else:
                    process.terminate()
            except (AttributeError, ProcessLookupError):
                pass
            try:
                process.wait(timeout=30.0)
            except subprocess.TimeoutExpired:
                try:
                    if isinstance(pid, int) and pid > 0:
                        os.killpg(pid, signal.SIGKILL)
                    else:
                        process.kill()
                except (AttributeError, ProcessLookupError):
                    pass
                try:
                    process.wait(timeout=30.0)
                except (subprocess.TimeoutExpired, ChildProcessError):
                    pass
            except ChildProcessError:
                pass
        raise
    finally:
        if runtime_cleanup is not None:
            had_active_exception = sys.exc_info()[0] is not None
            try:
                runtime_cleanup()
            except Exception as cleanup_exc:
                if not had_active_exception:
                    raise
                print(
                    "[interleave-rl] runtime cleanup also failed: "
                    f"{type(cleanup_exc).__name__}: {cleanup_exc}",
                    flush=True,
                )


def _run_provenance_identity(
    *,
    checkpoint: Path,
    run_name: str,
    model_id: str,
    num_rollout: int,
    lr: str,
    kl_loss_type: str,
    train_file: str,
    train_file_sha256: str,
    dynamic_filter: bool,
    rollout_seed: int,
    save_interval: int,
    eval_interval: int,
    max_tokens_per_gpu: int,
    sglang_server_concurrency: int,
    deterministic_inference: bool,
    resume_if_available: bool,
    rollout_only: bool,
    canary: bool,
    wandb_project: str,
    wandb_group: str,
    wandb_run_id: str,
    rollout_max_prompt_len: int,
    rollout_max_response_len: int,
    rollout_max_context_len: int,
    data_source_path: str,
    fault_tolerance: bool,
    initial_adam_checkpoint: str = "",
    initial_adam_completion_sha256: str = "",
    initial_adam_source_tree_sha256: str = "",
    initial_adam_step: int = 0,
) -> dict[str, object]:
    return {
        "kind": "chess_rl_miles_interleave_run",
        "run": {
            "app_name": APP_NAME,
            "run_name": run_name,
            "model_id": model_id,
            "num_rollout": num_rollout,
            "dynamic_filter": dynamic_filter,
            "rollout_seed": rollout_seed,
            "save_interval": save_interval,
            "eval_interval": eval_interval,
            "canary": canary,
            "deterministic_inference": deterministic_inference,
            "resume_if_available": resume_if_available,
            "rollout_only": rollout_only,
            "wandb_entity": WANDB_ENTITY,
            "wandb_project": wandb_project,
            "wandb_group": wandb_group,
            "wandb_run_id": wandb_run_id or None,
        },
        "policy_update_profile": {
            "name": POLICY_UPDATE_PROFILE,
            "max_tokens_per_gpu": max_tokens_per_gpu,
            "gradient_checkpointing": False,
            "train_backend": "fsdp",
            "actor_num_nodes": 1,
            "actor_num_gpus_per_node": 8,
            "gpu_type": base.GPU_TYPE,
            "host_memory_gb": SMALL_MODEL_HOST_MEMORY_GB,
            "sglang_server_concurrency": sglang_server_concurrency,
            "master_parameter_dtype": "float32",
            "optimizer_state_dtype": "float32",
            "forward_backward_dtype": "bfloat16",
            "gradient_reduction_dtype": "float32",
            "sglang_inference_dtype": "bfloat16",
        },
        "fixed_rl_semantics": {
            "policy_update_mode": (
                "disabled_rollout_only"
                if rollout_only
                else "one_or_more_optimizer_updates"
            ),
            "rollout_batch_size": 256,
            "samples_per_prompt": 8,
            "global_batch_size": 2_048,
            "policy_loss_agg_mode": "token-mean",
            "advantage_estimator": "grpo",
            "cispo": False,
            "optimizer": "adamw",
            "lr": str(lr),
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_eps": 1e-8,
            "weight_decay": 0.01,
            "kl_loss_coef": 0.001,
            "kl_loss_type": str(kl_loss_type),
            "rollout_max_prompt_len": rollout_max_prompt_len,
            "rollout_max_response_len": rollout_max_response_len,
            "rollout_max_context_len": rollout_max_context_len,
            "data_source_path": data_source_path,
            "fault_tolerance": fault_tolerance,
            "sampling_seed_rule": (
                "rollout_seed_plus_global_sample_index"
                if deterministic_inference
                else "backend_default"
            ),
        },
        "training_data": {
            "logical_path": str(train_file),
            "sha256": str(train_file_sha256),
        },
        "initial_optimizer_state": (
            {
                "mode": "continue_adam_moments_and_parameter_steps",
                "checkpoint": initial_adam_checkpoint,
                "completion_sha256": initial_adam_completion_sha256,
                "source_tree_sha256": initial_adam_source_tree_sha256,
                "source_step": initial_adam_step,
                "destination_hyperparameters_preserved": True,
            }
            if initial_adam_checkpoint
            else {"mode": "fresh_adam_state"}
        ),
        "checkpoint_publication": {
            "mode": "incremental_modal_volume_commit",
            "poll_seconds": CHECKPOINT_COMMIT_POLL_SECONDS,
            "readiness_markers": [
                "latest_checkpointed_iteration.txt",
                "iter_<step>/model/.metadata",
                "iter_<step>/optimizer/.metadata",
                "iter_<step>/lr_scheduler/.metadata",
                "iter_<step>/rng_rank_<rank>.pt",
                "iter_<step>/meta.json",
                "iter_<step>/COMMITTED.json",
            ],
        },
        "origin_hf": directory_identity(
            checkpoint,
            logical_path=str(checkpoint),
        ),
        "sources": {
            "chess_rl_miles": source_tree_identity(Path(base.PROJECT_DIR)),
            "miles": source_tree_identity(Path(base.MILES_DIR)),
        },
        "runtime": _hydrated_modal_runtime_identity(),
    }


def _training_source_contract(
    *,
    canary: bool,
    rollout_only: bool,
) -> tuple[str, bool]:
    """Return the data source and fault-tolerance mode used by a run."""

    if not canary and not rollout_only:
        return PRECISION_RESUME_DATA_SOURCE_PATH, False
    return DEFAULT_DATA_SOURCE_PATH, True


@app.function(
    gpu=f"{base.GPU_TYPE}:{base.GPUS_PER_NODE}",
    cpu=base.CPU_COUNT,
    memory=SMALL_MODEL_HOST_MEMORY_MB,
    timeout=60 * 60 * base.TIMEOUT_HOURS,
    retries=modal.Retries(initial_delay=0.0, max_retries=base.RETRIES),
    single_use_containers=True,
    volumes={
        "/data": base.data_vol,
        "/rl-checkpoints": base.ckpt_vol,
        PRETRAIN_CKPT_ROOT: pretrain_ckpt_vol,
        base.HF_CACHE_DIR: base.hf_cache,
    },
)
@_wrap_deployed_terminal_failure(PRODUCTION_RL_TRAINING_TERMINAL_MARKER)
def train_hf(
    hf_checkpoint: str,
    run_name: str,
    num_rollout: int,
    dynamic_filter: bool = False,
    rollout_seed: int = 42,
    save_interval: int = 40,
    eval_interval: int = 0,
    model_id: str = CONTEXT2048_MODEL_ID,
    resume_if_available: bool = True,
    wandb_project: str = "chess_interleave_50m",
    wandb_group: str = "core",
    max_tokens_per_gpu: int = SMALL_MODEL_MAX_TOKENS_DEFAULT,
    sglang_server_concurrency: int = SGLANG_SERVER_CONCURRENCY_DEFAULT,
    deterministic_inference: bool = True,
    rollout_only: bool = False,
    canary: bool = False,
    train_file: str = BALANCED_TRAIN_FILE,
    train_file_sha256: str = BALANCED_TRAIN_SHA256,
    lr: str = "1e-5",
    kl_loss_type: str = KL_LOSS_TYPE_DEFAULT,
    rollout_max_prompt_len: int = 512,
    rollout_max_response_len: int = 1_536,
    rollout_max_context_len: int = 2_048,
    production_launch_token: str = "",
    production_launch_generation: int = -1,
    initial_adam_checkpoint: str = "",
    initial_adam_completion_sha256: str = "",
    initial_adam_source_tree_sha256: str = "",
    initial_adam_step: int = 0,
) -> dict[str, object]:
    """Train one RL stage from an arbitrary HF checkpoint."""
    _safe_component(run_name, name="run_name")
    _safe_component(model_id, name="model_id")
    pretrain_ckpt_vol.reload()
    base.data_vol.reload()
    base.ckpt_vol.reload()
    hf_checkpoint = str(Path(hf_checkpoint).resolve(strict=True))
    logical_train_file = _validated_logical_file_path(
        train_file,
        name="production RL training parquet",
    )
    train_file = str(Path(logical_train_file).resolve(strict=True))
    production_binding: Mapping[str, object] | None = None
    launch_identity: Mapping[str, object] | None = None
    production_function_call_id = ""
    production_wandb_run_id = ""
    if not canary and not rollout_only:
        if model_id != CONTEXT2048_MODEL_ID or (
            rollout_max_prompt_len,
            rollout_max_response_len,
            rollout_max_context_len,
        ) != (512, 1_536, 2_048):
            raise ValueError(
                "contract-bound production RL requires model_id="
                f"{CONTEXT2048_MODEL_ID}, prompt=512, response=1536, context=2048"
            )
        if dynamic_filter:
            raise ValueError(
                "contract-bound production RL requires the offline filtered parquet, not online dynamic filtering"
            )
        if not deterministic_inference:
            raise ValueError(
                "contract-bound production RL requires deterministic sample-index seeding for resumable rollout continuity"
            )
        deployment = _current_deployment_identity()
        launch_identity = _production_launch_identity(
            hf_checkpoint=hf_checkpoint,
            run_name=run_name,
            num_rollout=num_rollout,
            dynamic_filter=dynamic_filter,
            rollout_seed=rollout_seed,
            save_interval=save_interval,
            eval_interval=eval_interval,
            model_id=model_id,
            resume_if_available=resume_if_available,
            wandb_project=wandb_project,
            wandb_group=wandb_group,
            max_tokens_per_gpu=max_tokens_per_gpu,
            sglang_server_concurrency=sglang_server_concurrency,
            deterministic_inference=deterministic_inference,
            train_file=train_file,
            train_file_sha256=train_file_sha256,
            lr=lr,
            kl_loss_type=kl_loss_type,
            rollout_max_prompt_len=rollout_max_prompt_len,
            rollout_max_response_len=rollout_max_response_len,
            rollout_max_context_len=rollout_max_context_len,
            deployment_identity=deployment,
            initial_adam_checkpoint=initial_adam_checkpoint,
            initial_adam_completion_sha256=initial_adam_completion_sha256,
            initial_adam_source_tree_sha256=initial_adam_source_tree_sha256,
            initial_adam_step=initial_adam_step,
        )
        production_wandb_run_id = str(
            dict(launch_identity["wandb"])["run_id"]
        )
        production_function_call_id = modal.current_function_call_id()
        _validate_production_durable_anchor(
            run_name=run_name,
            claim=_validate_production_claim(
                production_launch_claims.get(
                    _production_claim_key(run_name), None
                ),
                run_name=run_name,
                expected_identity=launch_identity,
                launch_token=production_launch_token,
            ),
            launch_identity=launch_identity,
            launch_token=production_launch_token,
        )
        production_binding = _begin_claimed_production_worker(
            production_launch_claims,
            run_name=run_name,
            launch_token=production_launch_token,
            expected_identity=launch_identity,
            generation=production_launch_generation,
            function_call_id=production_function_call_id,
        )
    elif production_launch_token or production_launch_generation != -1:
        raise RuntimeError(
            "production launch claim inputs are forbidden for canary/rollout-only"
        )
    checkpoint = _validate_hf_checkpoint(hf_checkpoint)
    native_context_len = _validate_checkpoint_context(
        checkpoint,
        requested_context_len=rollout_max_context_len,
        require_exact=model_id == CONTEXT2048_MODEL_ID,
    )
    run_save_path = Path(RAW_RL_ROOT) / run_name

    precision_gate_evidence = None
    if not canary and not rollout_only:
        _, production_gate_contract = _precision_gate_contract_from_inputs(
            hf_checkpoint=str(checkpoint),
            model_id=model_id,
            train_file=logical_train_file,
            train_file_sha256=train_file_sha256,
            rollout_seed=rollout_seed,
            wandb_project=wandb_project,
            max_tokens_per_gpu=max_tokens_per_gpu,
            sglang_server_concurrency=sglang_server_concurrency,
            lr=lr,
            kl_loss_type=kl_loss_type,
            rollout_max_prompt_len=rollout_max_prompt_len,
            rollout_max_response_len=rollout_max_response_len,
            rollout_max_context_len=rollout_max_context_len,
            initial_adam_checkpoint=initial_adam_checkpoint,
            initial_adam_completion_sha256=initial_adam_completion_sha256,
            initial_adam_source_tree_sha256=initial_adam_source_tree_sha256,
            initial_adam_step=initial_adam_step,
        )
        precision_gate_evidence = _require_precision_resume_gate(
            contract=production_gate_contract,
        )

    resume_path: Path | None = None
    resume_step: int | None = None
    if resume_if_available and (not canary or save_interval > 0):
        resume_step = _reconcile_modal_checkpoint_root(run_save_path)
        if resume_step is not None:
            resume_path = run_save_path

    effective_data_source_path, effective_fault_tolerance = (
        _training_source_contract(
            canary=canary,
            rollout_only=rollout_only,
        )
    )

    command = build_train_command(
        hf_checkpoint=str(checkpoint),
        run_name=run_name,
        model_id=model_id,
        num_rollout=num_rollout,
        dynamic_filter=dynamic_filter,
        rollout_seed=rollout_seed,
        save_interval=save_interval,
        eval_interval=eval_interval,
        resume_path=str(resume_path or ""),
        resume_step=int(resume_step or 0),
        wandb_project=wandb_project,
        wandb_group=wandb_group,
        wandb_run_id=production_wandb_run_id,
        max_tokens_per_gpu=max_tokens_per_gpu,
        sglang_server_concurrency=sglang_server_concurrency,
        deterministic_inference=deterministic_inference,
        deterministic_seed_by_sample_index=deterministic_inference,
        data_source_path=effective_data_source_path,
        fault_tolerance=effective_fault_tolerance,
        rollout_only=rollout_only,
        canary=canary,
        train_file=train_file,
        train_file_sha256=train_file_sha256,
        lr=lr,
        kl_loss_type=kl_loss_type,
        rollout_max_prompt_len=rollout_max_prompt_len,
        rollout_max_response_len=rollout_max_response_len,
        rollout_max_context_len=rollout_max_context_len,
        initial_adam_checkpoint=initial_adam_checkpoint,
        initial_adam_completion_sha256=initial_adam_completion_sha256,
        initial_adam_source_tree_sha256=initial_adam_source_tree_sha256,
        initial_adam_step=initial_adam_step,
    )
    print("[interleave-rl] " + " ".join(command), flush=True)
    print(
        "[interleave-rl] "
        f"init={checkpoint} run={run_name} target={num_rollout} "
        f"filter={'dynamic' if dynamic_filter else 'none'} seed={rollout_seed} "
        f"resume_step={resume_step or 0} profile={POLICY_UPDATE_PROFILE} "
        f"max_tokens_per_gpu={max_tokens_per_gpu} "
        f"sglang_server_concurrency={sglang_server_concurrency} "
        f"deterministic_inference={deterministic_inference} "
        f"rollout_only={rollout_only} "
        f"native_context_len={native_context_len} "
        "gradient_checkpointing=false",
        flush=True,
    )

    provenance = write_run_provenance(
        run_root=run_save_path,
        identity={
            **_run_provenance_identity(
                checkpoint=checkpoint,
                run_name=run_name,
                model_id=model_id,
                num_rollout=num_rollout,
                lr=lr,
                kl_loss_type=kl_loss_type,
                train_file=train_file,
                train_file_sha256=train_file_sha256,
                dynamic_filter=dynamic_filter,
                rollout_seed=rollout_seed,
                save_interval=0 if canary and save_interval <= 0 else save_interval,
                eval_interval=eval_interval,
                max_tokens_per_gpu=max_tokens_per_gpu,
                sglang_server_concurrency=sglang_server_concurrency,
                deterministic_inference=deterministic_inference,
                resume_if_available=resume_if_available,
                rollout_only=rollout_only,
                canary=canary,
                wandb_project=wandb_project,
                wandb_group=wandb_group,
                wandb_run_id=production_wandb_run_id,
                rollout_max_prompt_len=rollout_max_prompt_len,
                rollout_max_response_len=rollout_max_response_len,
                rollout_max_context_len=rollout_max_context_len,
                data_source_path=effective_data_source_path,
                fault_tolerance=effective_fault_tolerance,
                initial_adam_checkpoint=initial_adam_checkpoint,
                initial_adam_completion_sha256=initial_adam_completion_sha256,
                initial_adam_source_tree_sha256=initial_adam_source_tree_sha256,
                initial_adam_step=initial_adam_step,
            ),
            "precision_resume_gate": precision_gate_evidence,
        },
        command=command,
    )
    base.ckpt_vol.commit()
    print(
        "[interleave-rl] provenance="
        + json.dumps(provenance, sort_keys=True),
        flush=True,
    )

    env = _runtime_env(
        run_name=run_name,
        deterministic_seed_mode=(
            "sample-index" if deterministic_inference else None
        ),
    )
    base._cleanup_runtime()
    base._start_ray_head(env, cpu_threads=int(base.CPU_COUNT))
    env["RAY_ADDRESS"] = base.RAY_ADDRESS
    lease_heartbeat: Callable[[], None] | None = None
    if launch_identity is not None:
        assert production_binding is not None

        def refresh_production_lease() -> None:
            claim = _validate_production_claim(
                production_launch_claims.get(
                    _production_claim_key(run_name), None
                ),
                run_name=run_name,
                expected_identity=launch_identity,
                launch_token=production_launch_token,
            )
            _validate_production_durable_anchor(
                run_name=run_name,
                claim=claim,
                launch_identity=launch_identity,
                launch_token=production_launch_token,
            )
            refreshed = _begin_claimed_production_worker(
                production_launch_claims,
                run_name=run_name,
                launch_token=production_launch_token,
                expected_identity=launch_identity,
                generation=production_launch_generation,
                function_call_id=production_function_call_id,
            )
            if refreshed["execution"] != production_binding["execution"]:
                raise RuntimeError(
                    f"{run_name} production RL lease binding drifted"
                )

        lease_heartbeat = refresh_production_lease
    returncode, published_through_step = (
        _run_training_with_checkpoint_commits(
            command,
            env=env,
            cwd=base.PROJECT_DIR,
            run_root=run_save_path,
            volume=base.ckpt_vol,
            initial_published_step=int(resume_step or 0),
            lease_heartbeat=lease_heartbeat,
            runtime_cleanup=base._cleanup_runtime,
        )
    )
    if returncode:
        raise RuntimeError(f"RL training failed for {run_name}: exit {returncode}")
    base.ckpt_vol.commit()
    durable_completion = None
    if launch_identity is not None:
        assert production_binding is not None
        base.ckpt_vol.reload()
        checkpoint_completion = _authenticated_production_recovery_checkpoint(
            run_name=run_name,
            launch_identity=launch_identity,
        )
        if checkpoint_completion.get("state") != "complete" or int(
            checkpoint_completion.get("checkpoint_step", -1)
        ) != int(num_rollout):
            raise RuntimeError(
                f"{run_name} returned cleanly without an authenticated exact "
                f"target checkpoint at update {num_rollout}"
            )
        durable_completion = _publish_production_durable_completion(
            run_name=run_name,
            launch_identity=launch_identity,
            binding=production_binding,
            checkpoint=checkpoint_completion,
            target_updates=num_rollout,
            volume=base.ckpt_vol,
        )
    return {
        "run_name": run_name,
        "checkpoint_root": str(run_save_path),
        "num_rollout": num_rollout,
        "dynamic_filter": dynamic_filter,
        "rollout_seed": rollout_seed,
        "policy_update_profile": POLICY_UPDATE_PROFILE,
        "max_tokens_per_gpu": max_tokens_per_gpu,
        "sglang_server_concurrency": sglang_server_concurrency,
        "deterministic_inference": deterministic_inference,
        "rollout_only": rollout_only,
        "gradient_checkpointing": False,
        "incremental_checkpoint_published_through": published_through_step,
        "provenance": provenance,
        "precision_resume_gate": precision_gate_evidence,
        "durable_completion": durable_completion,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_production_launch_token(launch_token: object) -> str:
    if not isinstance(launch_token, str) or not re.fullmatch(
        r"[0-9a-f]{64}", launch_token
    ):
        raise RuntimeError(
            "production RL launch requires a 256-bit hexadecimal recovery token"
        )
    return launch_token


def _production_launch_token_sha256(launch_token: object) -> str:
    token = _require_production_launch_token(launch_token)
    return hashlib.sha256(token.encode("ascii")).hexdigest()


def _production_self_hash(
    record: Mapping[str, object],
    *,
    hash_field: str,
) -> dict[str, object]:
    core = {key: value for key, value in record.items() if key != hash_field}
    return {**core, hash_field: _canonical_json_sha256(core)}


def _validate_production_self_hash(
    record: object,
    *,
    hash_field: str,
    label: str,
) -> dict[str, object]:
    if not isinstance(record, Mapping):
        raise RuntimeError(f"{label} is missing or is not an object")
    value = dict(record)
    recorded = value.pop(hash_field, None)
    if recorded != _canonical_json_sha256(value):
        raise RuntimeError(f"{label} self hash drifted")
    return {**value, hash_field: recorded}


def _validate_production_terminal_call_evidence(
    record: object,
    *,
    expected_function_call_id: str,
    allow_success: bool = False,
) -> dict[str, object]:
    evidence = _validate_production_self_hash(
        record,
        hash_field="evidence_sha256",
        label="production RL terminal FunctionCall evidence",
    )
    if set(evidence) != {
        "schema",
        "function_call_id",
        "result_category",
        "exception_type",
        "evidence_sha256",
    }:
        raise RuntimeError("production RL terminal evidence fields drifted")
    if evidence.get("schema") != PRODUCTION_TERMINAL_CALL_EVIDENCE_SCHEMA:
        raise RuntimeError("production RL terminal evidence schema drifted")
    if evidence.get("function_call_id") != expected_function_call_id:
        raise RuntimeError("production RL terminal evidence call ID drifted")
    allowed: dict[str, set[object]] = {
        "application_failure": {
            "builtins.RuntimeError",
        },
        "function_timeout": {"modal.exception.FunctionTimeoutError"},
        "remote_terminal_failure": {"modal.exception.RemoteError"},
    }
    if allow_success:
        allowed["success"] = {None}
    category = evidence.get("result_category")
    if category not in allowed:
        raise RuntimeError("production RL FunctionCall is not terminal as required")
    if evidence.get("exception_type") not in allowed[category]:
        raise RuntimeError("production RL terminal exception type drifted")
    return evidence


def _validate_production_recovery_evidence(
    record: object,
    *,
    expected_prior_attempt_sha256: str,
    expected_function_call_id: str,
    allow_resolution_binding: bool,
) -> dict[str, object]:
    evidence = _validate_production_self_hash(
        record,
        hash_field="recovery_sha256",
        label="production RL recovery evidence",
    )
    required = {
        "schema",
        "prior_attempt_sha256",
        "terminal_call",
        "checkpoint",
        "recovery_sha256",
    }
    if allow_resolution_binding:
        required.add("generation_resolution_sha256")
    if set(evidence) != required:
        raise RuntimeError("production RL recovery evidence fields drifted")
    if evidence.get("schema") != "chess-rl-miles-production-recovery-evidence-v1":
        raise RuntimeError("production RL recovery evidence schema drifted")
    if evidence.get("prior_attempt_sha256") != expected_prior_attempt_sha256:
        raise RuntimeError("production RL recovery prior attempt drifted")
    _validate_production_terminal_call_evidence(
        evidence.get("terminal_call"),
        expected_function_call_id=expected_function_call_id,
    )
    checkpoint = evidence.get("checkpoint")
    if not isinstance(checkpoint, Mapping) or checkpoint.get("state") not in {
        "resumable",
        "complete",
    }:
        raise RuntimeError("production RL recovery checkpoint evidence drifted")
    if allow_resolution_binding and not re.fullmatch(
        r"[0-9a-f]{64}", str(evidence.get("generation_resolution_sha256") or "")
    ):
        raise RuntimeError("production RL recovery resolution binding drifted")
    return evidence


def _production_claim_key(run_name: str) -> str:
    return f"production-claim:{_safe_component(run_name, name='run_name')}"


def _production_durable_anchor_path(run_name: str) -> Path:
    _production_claim_key(run_name)
    return (
        Path(RAW_RL_ROOT)
        / "_production_launch_ledger"
        / "anchors"
        / f"{run_name}.json"
    )


def _production_durable_completion_path(run_name: str) -> Path:
    _production_claim_key(run_name)
    return (
        Path(RAW_RL_ROOT)
        / "_production_launch_ledger"
        / "completions"
        / f"{run_name}.json"
    )


def _production_anchor_intent_key(run_name: str) -> str:
    _production_claim_key(run_name)
    return f"production-anchor-intent:{run_name}"


def _production_durable_anchor_record(
    *,
    run_name: str,
    claim: Mapping[str, object],
    launch_identity: Mapping[str, object],
    publisher_recovery: Mapping[str, object] | None = None,
) -> dict[str, object]:
    run_root = str(launch_identity.get("run_root") or "")
    if run_root != str(Path(RAW_RL_ROOT) / run_name):
        raise RuntimeError(f"{run_name} durable anchor run root drifted")
    core = {
        "schema": PRODUCTION_DURABLE_ANCHOR_SCHEMA,
        "app_name": APP_NAME,
        "run_name": run_name,
        "run_root": run_root,
        "claim_sha256": claim["claim_sha256"],
        "launch_token_sha256": claim["launch_token_sha256"],
        "launch_identity_sha256": claim["launch_identity_sha256"],
        "claimed_at": claim["claimed_at"],
        "publisher_recovery": (
            dict(publisher_recovery)
            if publisher_recovery is not None
            else None
        ),
    }
    return _production_self_hash(core, hash_field="anchor_sha256")


def _validate_production_durable_anchor(
    *,
    run_name: str,
    claim: Mapping[str, object],
    launch_identity: Mapping[str, object],
    launch_token: str | None = None,
) -> dict[str, object]:
    path = _production_durable_anchor_path(run_name)
    anchor = _validate_production_self_hash(
        json.loads(path.read_text(encoding="utf-8")),
        hash_field="anchor_sha256",
        label=f"{run_name} durable production RL launch anchor",
    )
    publisher_recovery = anchor.get("publisher_recovery")
    if publisher_recovery is not None:
        recovery = _validate_production_self_hash(
            publisher_recovery,
            hash_field="recovery_sha256",
            label=f"{run_name} durable anchor publisher recovery",
        )
        if set(recovery) != {
            "schema",
            "intent_sha256",
            "prior_dispatcher_function_call_id",
            "terminal_call",
            "recovery_sha256",
        } or recovery.get("schema") != (
            "chess-rl-miles-durable-anchor-publisher-recovery-v1"
        ):
            raise RuntimeError(f"{run_name} durable anchor recovery drifted")
        _validate_production_terminal_call_evidence(
            recovery.get("terminal_call"),
            expected_function_call_id=str(
                recovery.get("prior_dispatcher_function_call_id") or ""
            ),
        )
    expected = _production_durable_anchor_record(
        run_name=run_name,
        claim=claim,
        launch_identity=launch_identity,
        publisher_recovery=(
            publisher_recovery if isinstance(publisher_recovery, Mapping) else None
        ),
    )
    if anchor != expected:
        raise RuntimeError(f"{run_name} durable production RL anchor drifted")
    if launch_token is not None and anchor["launch_token_sha256"] != (
        _production_launch_token_sha256(launch_token)
    ):
        raise RuntimeError(f"{run_name} durable production RL token drifted")
    return anchor


def _publish_production_durable_anchor(
    *,
    run_name: str,
    claim: Mapping[str, object],
    launch_identity: Mapping[str, object],
    launch_token: str,
    publisher_recovery: Mapping[str, object] | None = None,
    volume: Any = None,
) -> dict[str, object]:
    mounted_volume = base.ckpt_vol if volume is None else volume
    proposed = _production_durable_anchor_record(
        run_name=run_name,
        claim=claim,
        launch_identity=launch_identity,
        publisher_recovery=publisher_recovery,
    )
    path = _production_durable_anchor_path(run_name)
    mounted_volume.reload()
    if path.exists():
        return _validate_production_durable_anchor(
            run_name=run_name,
            claim=claim,
            launch_identity=launch_identity,
            launch_token=launch_token,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        _exclusive_json(path, proposed)
    except FileExistsError:
        mounted_volume.reload()
        return _validate_production_durable_anchor(
            run_name=run_name,
            claim=claim,
            launch_identity=launch_identity,
            launch_token=launch_token,
        )
    mounted_volume.commit()
    mounted_volume.reload()
    observed = _validate_production_durable_anchor(
        run_name=run_name,
        claim=claim,
        launch_identity=launch_identity,
        launch_token=launch_token,
    )
    if observed != proposed:
        raise RuntimeError(f"{run_name} durable anchor publication drifted")
    return observed


def _ensure_production_durable_anchor(
    store: Any,
    *,
    run_name: str,
    claim: Mapping[str, object],
    launch_identity: Mapping[str, object],
    launch_token: str,
    dispatcher_function_call_id: str,
    recovery: bool,
    volume: Any = None,
) -> dict[str, object]:
    mounted_volume = base.ckpt_vol if volume is None else volume
    mounted_volume.reload()
    path = _production_durable_anchor_path(run_name)
    if path.exists():
        return _validate_production_durable_anchor(
            run_name=run_name,
            claim=claim,
            launch_identity=launch_identity,
            launch_token=launch_token,
        )
    intent = _production_self_hash(
        {
            "schema": "chess-rl-miles-durable-anchor-intent-v1",
            "run_name": run_name,
            "claim_sha256": claim["claim_sha256"],
            "dispatcher_function_call_id": dispatcher_function_call_id,
        },
        hash_field="intent_sha256",
    )
    key = _production_anchor_intent_key(run_name)
    store.put(key, intent, skip_if_exists=True)
    observed = _validate_production_self_hash(
        store.get(key, None),
        hash_field="intent_sha256",
        label=f"{run_name} durable anchor intent",
    )
    publisher_recovery: Mapping[str, object] | None = None
    if observed != intent:
        mounted_volume.reload()
        if path.exists():
            return _validate_production_durable_anchor(
                run_name=run_name,
                claim=claim,
                launch_identity=launch_identity,
                launch_token=launch_token,
            )
        if not recovery:
            raise RuntimeError(
                f"{run_name} durable anchor is owned by another dispatcher"
            )
        prior_dispatcher = str(
            observed.get("dispatcher_function_call_id") or ""
        )
        terminal = _inspect_terminal_unsuccessful_production_call(
            prior_dispatcher
        )
        publisher_recovery = _production_self_hash(
            {
                "schema": (
                    "chess-rl-miles-durable-anchor-publisher-recovery-v1"
                ),
                "intent_sha256": observed["intent_sha256"],
                "prior_dispatcher_function_call_id": prior_dispatcher,
                "terminal_call": terminal,
            },
            hash_field="recovery_sha256",
        )
    return _publish_production_durable_anchor(
        run_name=run_name,
        claim=claim,
        launch_identity=launch_identity,
        launch_token=launch_token,
        publisher_recovery=publisher_recovery,
        volume=mounted_volume,
    )


def _production_attempt_key(run_name: str, generation: int) -> str:
    _production_claim_key(run_name)
    if isinstance(generation, bool) or not (
        0 <= int(generation) < MAX_PRODUCTION_LAUNCH_GENERATIONS
    ):
        raise ValueError(f"invalid production launch generation: {generation!r}")
    return f"production-attempt:{run_name}:{int(generation):04d}"


def _production_execution_key(run_name: str, generation: int) -> str:
    _production_attempt_key(run_name, generation)
    return f"production-execution:{run_name}:{int(generation):04d}"


def _production_resolution_key(run_name: str, generation: int) -> str:
    _production_attempt_key(run_name, generation)
    return f"production-resolution:{run_name}:{int(generation):04d}"


def _local_production_recovery_path(run_name: str) -> Path:
    _production_claim_key(run_name)
    return LOCAL_PRODUCTION_LAUNCH_RECOVERY_ROOT / f"{run_name}.json"


def _write_local_production_recovery_record(
    *,
    run_name: str,
    launch_token: str,
) -> Path:
    """Persist the sole raw recovery token before making a remote claim."""

    path = _local_production_recovery_path(run_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    core = {
        "schema": "chess-rl-miles-local-production-recovery-v1",
        "app_name": APP_NAME,
        "run_name": run_name,
        "launch_token": _require_production_launch_token(launch_token),
        "created_at": _utc_now(),
    }
    record = _production_self_hash(core, hash_field="record_sha256")
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError as exc:
        raise RuntimeError(
            f"production RL recovery record already exists: {path}; use "
            "--recover-production-launch"
        ) from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(path.parent)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def _read_local_production_recovery_record(run_name: str) -> dict[str, object]:
    path = _local_production_recovery_path(run_name)
    value = json.loads(path.read_text(encoding="utf-8"))
    record = _validate_production_self_hash(
        value,
        hash_field="record_sha256",
        label=f"{run_name} local production recovery record",
    )
    expected = {
        "schema": "chess-rl-miles-local-production-recovery-v1",
        "app_name": APP_NAME,
        "run_name": run_name,
    }
    for key, expected_value in expected.items():
        if record.get(key) != expected_value:
            raise RuntimeError(f"{run_name} local recovery {key} drifted")
    _require_production_launch_token(record.get("launch_token"))
    return record


def _new_production_claim(
    *,
    run_name: str,
    launch_token: str,
    launch_identity: Mapping[str, object],
    claimed_at: str | None = None,
) -> dict[str, object]:
    identity = dict(launch_identity)
    core = {
        "schema": PRODUCTION_LAUNCH_CLAIM_SCHEMA,
        "run_name": run_name,
        "launch_token_sha256": _production_launch_token_sha256(launch_token),
        "launch_identity": identity,
        "launch_identity_sha256": _canonical_json_sha256(identity),
        "claimed_at": claimed_at or _utc_now(),
    }
    return _production_self_hash(core, hash_field="claim_sha256")


def _validate_production_claim(
    record: object,
    *,
    run_name: str,
    expected_identity: Mapping[str, object],
    launch_token: str | None = None,
) -> dict[str, object]:
    claim = _validate_production_self_hash(
        record,
        hash_field="claim_sha256",
        label=f"{run_name} production RL launch claim",
    )
    required = {
        "schema",
        "run_name",
        "launch_token_sha256",
        "launch_identity",
        "launch_identity_sha256",
        "claimed_at",
        "claim_sha256",
    }
    if set(claim) != required:
        raise RuntimeError(f"{run_name} production RL claim fields drifted")
    if claim.get("schema") != PRODUCTION_LAUNCH_CLAIM_SCHEMA:
        raise RuntimeError(f"{run_name} production RL claim schema drifted")
    if claim.get("run_name") != run_name:
        raise RuntimeError(f"{run_name} production RL claim run name drifted")
    identity = claim.get("launch_identity")
    if not isinstance(identity, Mapping):
        raise RuntimeError(f"{run_name} production RL launch identity is missing")
    if claim.get("launch_identity_sha256") != _canonical_json_sha256(identity):
        raise RuntimeError(f"{run_name} production RL launch identity hash drifted")
    if dict(identity) != dict(expected_identity):
        raise RuntimeError(
            f"{run_name} has a claim for different production RL semantics"
        )
    if launch_token is not None and claim.get("launch_token_sha256") != (
        _production_launch_token_sha256(launch_token)
    ):
        raise RuntimeError(f"{run_name} production RL recovery token does not match")
    return claim


def _acquire_production_claim(
    store: Any,
    *,
    run_name: str,
    launch_token: str,
    launch_identity: Mapping[str, object],
    claimed_at: str | None = None,
) -> dict[str, object]:
    """Atomically acquire one immutable claim; claims are never age-stolen."""

    key = _production_claim_key(run_name)
    existing = store.get(key, None)
    if existing is not None:
        return {
            "outcome": "existing_claim",
            "claim": _validate_production_claim(
                existing,
                run_name=run_name,
                expected_identity=launch_identity,
            ),
        }
    proposed = _new_production_claim(
        run_name=run_name,
        launch_token=launch_token,
        launch_identity=launch_identity,
        claimed_at=claimed_at,
    )
    won = bool(store.put(key, proposed, skip_if_exists=True))
    observed = _validate_production_claim(
        store.get(key, None),
        run_name=run_name,
        expected_identity=launch_identity,
    )
    if won and observed != proposed:
        raise RuntimeError(f"{run_name} production RL claim CAS winner drifted")
    return {
        "outcome": "acquired" if won else "existing_claim",
        "claim": observed,
    }


def _new_production_attempt(
    *,
    run_name: str,
    claim: Mapping[str, object],
    generation: int,
    dispatcher_function_call_id: str,
    recovery_evidence: Mapping[str, object] | None,
    created_at: str | None = None,
) -> dict[str, object]:
    _production_attempt_key(run_name, generation)
    if not isinstance(dispatcher_function_call_id, str) or not re.fullmatch(
        r"fc-[0-9A-Za-z]+", dispatcher_function_call_id
    ):
        raise RuntimeError(
            "production RL dispatcher lacks a valid Modal FunctionCall ID"
        )
    if generation == 0 and recovery_evidence is not None:
        raise RuntimeError("initial production RL attempt cannot be recovery")
    if generation > 0 and not isinstance(recovery_evidence, Mapping):
        raise RuntimeError("production RL recovery attempt lacks terminal evidence")
    core = {
        "schema": PRODUCTION_LAUNCH_ATTEMPT_SCHEMA,
        "run_name": run_name,
        "generation": generation,
        "claim_sha256": claim["claim_sha256"],
        "launch_token_sha256": claim["launch_token_sha256"],
        "launch_identity_sha256": claim["launch_identity_sha256"],
        "dispatcher_function_call_id": dispatcher_function_call_id,
        "recovery_evidence": (
            dict(recovery_evidence) if recovery_evidence is not None else None
        ),
        "created_at": created_at or _utc_now(),
    }
    return _production_self_hash(core, hash_field="attempt_sha256")


def _validate_production_attempt(
    record: object,
    *,
    run_name: str,
    generation: int,
    claim: Mapping[str, object],
) -> dict[str, object]:
    attempt = _validate_production_self_hash(
        record,
        hash_field="attempt_sha256",
        label=f"{run_name} production RL generation {generation} attempt",
    )
    expected = {
        "schema": PRODUCTION_LAUNCH_ATTEMPT_SCHEMA,
        "run_name": run_name,
        "generation": generation,
        "claim_sha256": claim["claim_sha256"],
        "launch_token_sha256": claim["launch_token_sha256"],
        "launch_identity_sha256": claim["launch_identity_sha256"],
    }
    for key, expected_value in expected.items():
        if attempt.get(key) != expected_value:
            raise RuntimeError(f"{run_name} production RL attempt {key} drifted")
    dispatcher = attempt.get("dispatcher_function_call_id")
    if not isinstance(dispatcher, str) or not re.fullmatch(
        r"fc-[0-9A-Za-z]+", dispatcher
    ):
        raise RuntimeError(f"{run_name} production RL attempt dispatcher drifted")
    recovery = attempt.get("recovery_evidence")
    if generation == 0 and recovery is not None:
        raise RuntimeError(f"{run_name} initial attempt has recovery evidence")
    if generation > 0 and not isinstance(recovery, Mapping):
        raise RuntimeError(f"{run_name} recovery attempt lacks evidence")
    required = {
        *expected,
        "dispatcher_function_call_id",
        "recovery_evidence",
        "created_at",
        "attempt_sha256",
    }
    if set(attempt) != required:
        raise RuntimeError(f"{run_name} production RL attempt fields drifted")
    return attempt


def _current_production_attempt(
    store: Any,
    *,
    run_name: str,
    claim: Mapping[str, object],
) -> dict[str, object] | None:
    current = None
    for generation in range(MAX_PRODUCTION_LAUNCH_GENERATIONS):
        record = store.get(_production_attempt_key(run_name, generation), None)
        if record is None:
            return current
        current = _validate_production_attempt(
            record,
            run_name=run_name,
            generation=generation,
            claim=claim,
        )
    raise RuntimeError(f"{run_name} exceeded production RL launch generations")


def _acquire_production_attempt(
    store: Any,
    *,
    run_name: str,
    claim: Mapping[str, object],
    generation: int,
    dispatcher_function_call_id: str,
    recovery_evidence: Mapping[str, object] | None,
) -> dict[str, object]:
    proposed = _new_production_attempt(
        run_name=run_name,
        claim=claim,
        generation=generation,
        dispatcher_function_call_id=dispatcher_function_call_id,
        recovery_evidence=recovery_evidence,
    )
    key = _production_attempt_key(run_name, generation)
    won = bool(store.put(key, proposed, skip_if_exists=True))
    observed = _validate_production_attempt(
        store.get(key, None),
        run_name=run_name,
        generation=generation,
        claim=claim,
    )
    if won and observed != proposed:
        raise RuntimeError(f"{run_name} production RL attempt CAS winner drifted")
    return {
        "outcome": "attempt_acquired" if won else "attempt_exists",
        "attempt": observed,
    }


def _new_production_execution(
    *,
    run_name: str,
    claim: Mapping[str, object],
    attempt: Mapping[str, object],
    generation: int,
    function_call_id: str,
) -> dict[str, object]:
    if not isinstance(function_call_id, str) or not re.fullmatch(
        r"fc-[0-9A-Za-z]+", function_call_id
    ):
        raise RuntimeError("production RL worker lacks a valid FunctionCall ID")
    core = {
        "schema": PRODUCTION_LAUNCH_EXECUTION_SCHEMA,
        "run_name": run_name,
        "generation": generation,
        "claim_sha256": claim["claim_sha256"],
        "attempt_sha256": attempt["attempt_sha256"],
        "launch_token_sha256": claim["launch_token_sha256"],
        "function_call_id": function_call_id,
    }
    return _production_self_hash(core, hash_field="execution_sha256")


def _new_production_generation_resolution(
    *,
    run_name: str,
    claim: Mapping[str, object],
    attempt: Mapping[str, object],
    generation: int,
    decision: str,
    function_call_id: str | None = None,
    recovery_evidence: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build the single CAS decision between worker start and recovery."""

    if decision not in {"worker_bound", "recovery_closed"}:
        raise RuntimeError(f"invalid production generation decision: {decision!r}")
    if decision == "worker_bound":
        if not isinstance(function_call_id, str) or not re.fullmatch(
            r"fc-[0-9A-Za-z]+", function_call_id
        ):
            raise RuntimeError("worker resolution requires a valid FunctionCall ID")
        if recovery_evidence is not None:
            raise RuntimeError("worker resolution cannot carry recovery evidence")
    else:
        dispatcher_function_call_id = attempt.get(
            "dispatcher_function_call_id"
        )
        if function_call_id != dispatcher_function_call_id:
            raise RuntimeError(
                "recovery closure must bind the exact dispatcher FunctionCall "
                "for the attempt"
            )
        if not isinstance(recovery_evidence, Mapping):
            raise RuntimeError("recovery closure requires authenticated evidence")
        _validate_production_recovery_evidence(
            recovery_evidence,
            expected_prior_attempt_sha256=str(attempt["attempt_sha256"]),
            expected_function_call_id=str(dispatcher_function_call_id),
            allow_resolution_binding=False,
        )
    core = {
        "schema": PRODUCTION_GENERATION_RESOLUTION_SCHEMA,
        "run_name": run_name,
        "generation": generation,
        "claim_sha256": claim["claim_sha256"],
        "attempt_sha256": attempt["attempt_sha256"],
        "decision": decision,
        "function_call_id": function_call_id,
        "recovery_evidence": (
            dict(recovery_evidence) if recovery_evidence is not None else None
        ),
    }
    return _production_self_hash(core, hash_field="resolution_sha256")


def _validate_production_generation_resolution(
    record: object,
    *,
    run_name: str,
    claim: Mapping[str, object],
    attempt: Mapping[str, object],
    generation: int,
) -> dict[str, object]:
    resolution = _validate_production_self_hash(
        record,
        hash_field="resolution_sha256",
        label=f"{run_name} generation {generation} resolution",
    )
    expected = {
        "schema": PRODUCTION_GENERATION_RESOLUTION_SCHEMA,
        "run_name": run_name,
        "generation": generation,
        "claim_sha256": claim["claim_sha256"],
        "attempt_sha256": attempt["attempt_sha256"],
    }
    for key, expected_value in expected.items():
        if resolution.get(key) != expected_value:
            raise RuntimeError(f"{run_name} generation resolution {key} drifted")
    required = {
        *expected,
        "decision",
        "function_call_id",
        "recovery_evidence",
        "resolution_sha256",
    }
    if set(resolution) != required:
        raise RuntimeError(f"{run_name} generation resolution fields drifted")
    decision = resolution.get("decision")
    call_id = resolution.get("function_call_id")
    recovery_evidence = resolution.get("recovery_evidence")
    if decision == "worker_bound":
        if not isinstance(call_id, str) or not re.fullmatch(
            r"fc-[0-9A-Za-z]+", call_id
        ):
            raise RuntimeError(f"{run_name} worker resolution call ID drifted")
        if recovery_evidence is not None:
            raise RuntimeError(f"{run_name} worker resolution evidence drifted")
    elif decision == "recovery_closed":
        if (
            call_id != attempt.get("dispatcher_function_call_id")
            or not isinstance(recovery_evidence, Mapping)
        ):
            raise RuntimeError(f"{run_name} recovery closure evidence drifted")
        _validate_production_recovery_evidence(
            recovery_evidence,
            expected_prior_attempt_sha256=str(attempt["attempt_sha256"]),
            expected_function_call_id=str(call_id),
            allow_resolution_binding=False,
        )
    else:
        raise RuntimeError(f"{run_name} generation resolution decision drifted")
    return resolution


def _resolve_generation_for_worker(
    store: Any,
    *,
    run_name: str,
    claim: Mapping[str, object],
    attempt: Mapping[str, object],
    generation: int,
    function_call_id: str,
) -> dict[str, object]:
    proposed = _new_production_generation_resolution(
        run_name=run_name,
        claim=claim,
        attempt=attempt,
        generation=generation,
        decision="worker_bound",
        function_call_id=function_call_id,
    )
    key = _production_resolution_key(run_name, generation)
    won = bool(store.put(key, proposed, skip_if_exists=True))
    observed = _validate_production_generation_resolution(
        store.get(key, None),
        run_name=run_name,
        claim=claim,
        attempt=attempt,
        generation=generation,
    )
    if observed != proposed:
        if observed.get("decision") == "recovery_closed":
            raise RuntimeError(
                f"{run_name} generation {generation} was closed for recovery; "
                "this worker cannot train"
            )
        raise RuntimeError(
            f"{run_name} generation {generation} is bound to a different "
            "Modal FunctionCall"
        )
    return {**observed, "new_resolution": won}


def _close_or_observe_generation_for_recovery(
    store: Any,
    *,
    run_name: str,
    claim: Mapping[str, object],
    attempt: Mapping[str, object],
    generation: int,
    recovery_evidence: Mapping[str, object],
) -> dict[str, object]:
    """Atomically close an unbound generation or observe its worker winner."""

    proposed = _new_production_generation_resolution(
        run_name=run_name,
        claim=claim,
        attempt=attempt,
        generation=generation,
        decision="recovery_closed",
        function_call_id=str(attempt["dispatcher_function_call_id"]),
        recovery_evidence=recovery_evidence,
    )
    key = _production_resolution_key(run_name, generation)
    won = bool(store.put(key, proposed, skip_if_exists=True))
    observed = _validate_production_generation_resolution(
        store.get(key, None),
        run_name=run_name,
        claim=claim,
        attempt=attempt,
        generation=generation,
    )
    if won and observed != proposed:
        raise RuntimeError(f"{run_name} generation closure CAS winner drifted")
    return {
        "outcome": (
            "closed"
            if observed["decision"] == "recovery_closed"
            else "worker_bound"
        ),
        "new_resolution": won,
        "resolution": observed,
    }


def _bind_recovery_evidence_to_generation_resolution(
    recovery_evidence: Mapping[str, object],
    *,
    resolution: Mapping[str, object],
) -> dict[str, object]:
    """Bind the next attempt to the immutable decision for its predecessor."""

    terminal_call = recovery_evidence.get("terminal_call")
    if not isinstance(terminal_call, Mapping):
        raise RuntimeError("production RL recovery terminal evidence is missing")
    validated = _validate_production_recovery_evidence(
        recovery_evidence,
        expected_prior_attempt_sha256=str(
            recovery_evidence.get("prior_attempt_sha256") or ""
        ),
        expected_function_call_id=str(
            terminal_call.get("function_call_id") or ""
        ),
        allow_resolution_binding=False,
    )
    if "generation_resolution_sha256" in validated:
        raise RuntimeError(
            "production RL recovery evidence is already bound to a generation "
            "resolution"
        )
    resolution_sha256 = resolution.get("resolution_sha256")
    if not isinstance(resolution_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", resolution_sha256
    ):
        raise RuntimeError("production RL generation resolution hash drifted")
    core = {
        key: value
        for key, value in validated.items()
        if key != "recovery_sha256"
    }
    core["generation_resolution_sha256"] = resolution_sha256
    return _production_self_hash(core, hash_field="recovery_sha256")


def _validate_production_execution(
    record: object,
    *,
    run_name: str,
    claim: Mapping[str, object],
    attempt: Mapping[str, object],
    generation: int,
) -> dict[str, object]:
    execution = _validate_production_self_hash(
        record,
        hash_field="execution_sha256",
        label=f"{run_name} production RL generation {generation} execution",
    )
    expected = {
        "schema": PRODUCTION_LAUNCH_EXECUTION_SCHEMA,
        "run_name": run_name,
        "generation": generation,
        "claim_sha256": claim["claim_sha256"],
        "attempt_sha256": attempt["attempt_sha256"],
        "launch_token_sha256": claim["launch_token_sha256"],
    }
    for key, expected_value in expected.items():
        if execution.get(key) != expected_value:
            raise RuntimeError(f"{run_name} production RL execution {key} drifted")
    call_id = execution.get("function_call_id")
    if not isinstance(call_id, str) or not re.fullmatch(
        r"fc-[0-9A-Za-z]+", call_id
    ):
        raise RuntimeError(f"{run_name} production RL execution call ID drifted")
    if set(execution) != {*expected, "function_call_id", "execution_sha256"}:
        raise RuntimeError(f"{run_name} production RL execution fields drifted")
    return execution


def _begin_claimed_production_worker(
    store: Any,
    *,
    run_name: str,
    launch_token: str,
    expected_identity: Mapping[str, object],
    generation: int,
    function_call_id: str,
) -> dict[str, object]:
    """Allow retries of one Modal call while rejecting every other call."""

    claim = _validate_production_claim(
        store.get(_production_claim_key(run_name), None),
        run_name=run_name,
        expected_identity=expected_identity,
        launch_token=launch_token,
    )
    attempt = _current_production_attempt(
        store,
        run_name=run_name,
        claim=claim,
    )
    if attempt is None or int(attempt["generation"]) != generation:
        raise RuntimeError(f"{run_name} production RL worker generation is not current")
    resolution = _resolve_generation_for_worker(
        store,
        run_name=run_name,
        claim=claim,
        attempt=attempt,
        generation=generation,
        function_call_id=function_call_id,
    )
    proposed = _new_production_execution(
        run_name=run_name,
        claim=claim,
        attempt=attempt,
        generation=generation,
        function_call_id=function_call_id,
    )
    key = _production_execution_key(run_name, generation)
    won = bool(store.put(key, proposed, skip_if_exists=True))
    observed = _validate_production_execution(
        store.get(key, None),
        run_name=run_name,
        claim=claim,
        attempt=attempt,
        generation=generation,
    )
    if observed != proposed:
        raise RuntimeError(
            f"{run_name} production RL generation {generation} is bound to a "
            "different Modal FunctionCall"
        )
    return {
        "claim": claim,
        "attempt": attempt,
        "resolution": resolution,
        "execution": observed,
        "new_binding": won,
    }


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _exclusive_json(path: Path, value: dict[str, object]) -> None:
    """Create an intent exactly once; never replace an existing controller."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o644,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        # fdopen owns the descriptor after it succeeds.
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def _normalized_source_identity(
    root: Path,
    *,
    excluded_relatives: tuple[str, ...] = (),
) -> dict[str, object]:
    identity = source_tree_identity(
        root,
        excluded_relatives=excluded_relatives,
    )
    identity.pop("logical_root", None)
    return identity


def _precision_gate_source_identity() -> dict[str, object]:
    sources = {
        "chess_rl_miles": _normalized_source_identity(
            Path(base.PROJECT_DIR)
        ),
        "miles": _normalized_source_identity(Path(base.MILES_DIR)),
    }
    return {
        "sources": sources,
        "source_sha256": _canonical_json_sha256(sources),
    }


def _local_precision_gate_source_identity() -> dict[str, object]:
    """Hash the local trees that Modal deploy will mount at remote paths."""

    sources = {
        "chess_rl_miles": _normalized_source_identity(
            Path(base.PROJECT_LOCAL)
        ),
        "miles": _normalized_source_identity(Path(base.MILES_LOCAL)),
    }
    return {
        "sources": sources,
        "source_sha256": _canonical_json_sha256(sources),
    }


def _hydrated_modal_runtime_identity() -> dict[str, object]:
    """Bind the gate and production launch to one hydrated Modal deployment."""

    try:
        image_id = base.image.object_id
        app_id = app.app_id
    except Exception as exc:
        raise RuntimeError(
            "Modal app/image handles are not hydrated; runtime identity cannot be authenticated"
        ) from exc
    if not isinstance(image_id, str) or not image_id.startswith("im-"):
        raise RuntimeError(f"invalid hydrated Modal image ID: {image_id!r}")
    if not isinstance(app_id, str) or not app_id.startswith("ap-"):
        raise RuntimeError(f"invalid hydrated Modal app ID: {app_id!r}")
    base_runtime = runtime_identity(image=base.MILES_IMAGE)
    os_inventory: list[dict[str, str]] = []
    dpkg = shutil.which("dpkg-query")
    if dpkg:
        result = subprocess.run(
            [dpkg, "-W", "-f=${Package}\t${Version}\n"],
            check=True,
            text=True,
            capture_output=True,
        )
        for line in sorted(result.stdout.splitlines()):
            package, separator, version = line.partition("\t")
            if not separator or not package or not version:
                raise RuntimeError("invalid dpkg runtime inventory row")
            os_inventory.append({"package": package, "version": version})
    return {
        **base_runtime,
        "modal_app_name": APP_NAME,
        "modal_app_id": app_id,
        "modal_image_id": image_id,
        "os_package_count": len(os_inventory),
        "os_package_inventory_sha256": _canonical_json_sha256(os_inventory),
    }


def _validate_hydrated_runtime_identity(recorded: object) -> None:
    current = _hydrated_modal_runtime_identity()
    if recorded != current:
        raise RuntimeError("authenticated Modal app/image/runtime identity drifted")


def _production_launch_identity(
    *,
    hf_checkpoint: str,
    run_name: str,
    num_rollout: int,
    dynamic_filter: bool,
    rollout_seed: int,
    save_interval: int,
    eval_interval: int,
    model_id: str,
    resume_if_available: bool,
    wandb_project: str,
    wandb_group: str,
    max_tokens_per_gpu: int,
    sglang_server_concurrency: int,
    deterministic_inference: bool,
    train_file: str,
    train_file_sha256: str,
    lr: str,
    kl_loss_type: str,
    rollout_max_prompt_len: int,
    rollout_max_response_len: int,
    rollout_max_context_len: int,
    deployment_identity: Mapping[str, object],
    initial_adam_checkpoint: str = "",
    initial_adam_completion_sha256: str = "",
    initial_adam_source_tree_sha256: str = "",
    initial_adam_step: int = 0,
) -> dict[str, object]:
    """Bind the exact production spawn to immutable model/data/runtime inputs."""

    checkpoint = Path(hf_checkpoint).resolve(strict=True)
    train_path = Path(train_file).resolve(strict=True)
    if _sha256(train_path) != train_file_sha256:
        raise RuntimeError("production RL training parquet SHA256 drifted")
    origin_hf = directory_identity(
        checkpoint,
        logical_path=str(Path(hf_checkpoint)),
    )
    runtime = deployment_identity.get("runtime")
    source_sha256 = deployment_identity.get("source_sha256")
    if not isinstance(runtime, Mapping) or not isinstance(source_sha256, str):
        raise RuntimeError("deployed production RL identity is incomplete")
    effective_data_source_path, effective_fault_tolerance = (
        _training_source_contract(canary=False, rollout_only=False)
    )
    command_without_wandb_id = build_train_command(
        hf_checkpoint=str(checkpoint),
        run_name=run_name,
        model_id=model_id,
        num_rollout=num_rollout,
        dynamic_filter=dynamic_filter,
        rollout_seed=rollout_seed,
        save_interval=save_interval,
        eval_interval=eval_interval,
        wandb_project=wandb_project,
        wandb_group=wandb_group,
        max_tokens_per_gpu=max_tokens_per_gpu,
        sglang_server_concurrency=sglang_server_concurrency,
        deterministic_inference=deterministic_inference,
        deterministic_seed_by_sample_index=deterministic_inference,
        data_source_path=effective_data_source_path,
        fault_tolerance=effective_fault_tolerance,
        rollout_only=False,
        canary=False,
        train_file=str(train_path),
        train_file_sha256=train_file_sha256,
        lr=lr,
        kl_loss_type=kl_loss_type,
        rollout_max_prompt_len=rollout_max_prompt_len,
        rollout_max_response_len=rollout_max_response_len,
        rollout_max_context_len=rollout_max_context_len,
        initial_adam_checkpoint=initial_adam_checkpoint,
        initial_adam_completion_sha256=initial_adam_completion_sha256,
        initial_adam_source_tree_sha256=initial_adam_source_tree_sha256,
        initial_adam_step=initial_adam_step,
    )
    wandb_run_id = "prod" + _canonical_json_sha256(
        {
            "schema": "chess-rl-miles-production-wandb-run-id-v1",
            "command_without_wandb_id": command_without_wandb_id,
            "entity": WANDB_ENTITY,
            "project": wandb_project,
            "group": wandb_group,
            "deployment_source_sha256": source_sha256,
        }
    )[:28]
    initial_command = build_train_command(
        hf_checkpoint=str(checkpoint),
        run_name=run_name,
        model_id=model_id,
        num_rollout=num_rollout,
        dynamic_filter=dynamic_filter,
        rollout_seed=rollout_seed,
        save_interval=save_interval,
        eval_interval=eval_interval,
        wandb_project=wandb_project,
        wandb_group=wandb_group,
        wandb_run_id=wandb_run_id,
        max_tokens_per_gpu=max_tokens_per_gpu,
        sglang_server_concurrency=sglang_server_concurrency,
        deterministic_inference=deterministic_inference,
        deterministic_seed_by_sample_index=deterministic_inference,
        data_source_path=effective_data_source_path,
        fault_tolerance=effective_fault_tolerance,
        rollout_only=False,
        canary=False,
        train_file=str(train_path),
        train_file_sha256=train_file_sha256,
        lr=lr,
        kl_loss_type=kl_loss_type,
        rollout_max_prompt_len=rollout_max_prompt_len,
        rollout_max_response_len=rollout_max_response_len,
        rollout_max_context_len=rollout_max_context_len,
        initial_adam_checkpoint=initial_adam_checkpoint,
        initial_adam_completion_sha256=initial_adam_completion_sha256,
        initial_adam_source_tree_sha256=initial_adam_source_tree_sha256,
        initial_adam_step=initial_adam_step,
    )
    sources = deployment_identity.get("sources")
    if not isinstance(sources, Mapping):
        raise RuntimeError("deployed production RL source identity is incomplete")
    identity = {
        "schema": "chess-rl-miles-production-launch-identity-v1",
        "app_name": APP_NAME,
        "run_name": run_name,
        "run_root": str(Path(RAW_RL_ROOT) / run_name),
        "origin_hf": origin_hf,
        "training_data": {
            "logical_path": str(train_path),
            "sha256": train_file_sha256,
        },
        "initial_optimizer_state": (
            {
                "mode": "continue_adam_moments_and_parameter_steps",
                "checkpoint": initial_adam_checkpoint,
                "completion_sha256": initial_adam_completion_sha256,
                "source_tree_sha256": initial_adam_source_tree_sha256,
                "source_step": initial_adam_step,
                "destination_hyperparameters_preserved": True,
            }
            if initial_adam_checkpoint
            else {"mode": "fresh_adam_state"}
        ),
        "semantics": {
            "target_updates": num_rollout,
            "dynamic_filter": dynamic_filter,
            "rollout_seed": rollout_seed,
            "save_interval": save_interval,
            "eval_interval": eval_interval,
            "model_id": model_id,
            "resume_if_available": resume_if_available,
            "lr": str(lr),
            "kl_loss_type": kl_loss_type,
            "kl_loss_coef": 0.001,
            "rollout_batch_size": 256,
            "samples_per_prompt": 8,
            "global_batch_size": 2_048,
            "max_tokens_per_gpu": max_tokens_per_gpu,
            "sglang_server_concurrency": sglang_server_concurrency,
            "deterministic_inference": deterministic_inference,
            "rollout_max_prompt_len": rollout_max_prompt_len,
            "rollout_max_response_len": rollout_max_response_len,
            "rollout_max_context_len": rollout_max_context_len,
            "policy_loss_agg_mode": "token-mean",
            "master_parameter_dtype": "float32",
            "optimizer_state_dtype": "float32",
            "forward_backward_dtype": "bfloat16",
        },
        "wandb": {
            "entity": WANDB_ENTITY,
            "project": wandb_project,
            "group": wandb_group,
            "run_id": wandb_run_id,
        },
        "deployment": {
            "source_sha256": source_sha256,
            "sources": dict(sources),
            "runtime": dict(runtime),
        },
        "initial_command_sha256": _canonical_json_sha256(initial_command),
    }
    return {
        **identity,
        "identity_sha256": _canonical_json_sha256(identity),
    }


def _production_exception_type_name(exc: BaseException) -> str:
    exception_type = type(exc)
    return f"{exception_type.__module__}.{exception_type.__qualname__}"


def _authoritative_production_call_evidence(
    get_result: Callable[..., object],
    *,
    function_call_id: str,
    allow_success: bool,
) -> dict[str, object]:
    """Poll one retained call result; ambiguous client states fail closed."""

    if not isinstance(function_call_id, str) or not re.fullmatch(
        r"fc-[0-9A-Za-z]+", function_call_id
    ):
        raise RuntimeError("invalid prior production RL FunctionCall ID")
    result_category: str
    exception_type: str | None
    try:
        get_result(timeout=0)
    except modal.exception.OutputExpiredError as exc:
        raise RuntimeError(
            f"production RL FunctionCall {function_call_id} output expired; "
            "its terminal result cannot be authenticated"
        ) from exc
    except modal.exception.FunctionTimeoutError as exc:
        result_category = "function_timeout"
        exception_type = _production_exception_type_name(exc)
    except builtins.TimeoutError as exc:
        raise RuntimeError(
            f"production RL FunctionCall {function_call_id} is still pending"
        ) from exc
    except modal.exception.RemoteError as exc:
        result_category = "remote_terminal_failure"
        exception_type = _production_exception_type_name(exc)
    except modal.exception.Error as exc:
        raise RuntimeError(
            f"production RL FunctionCall {function_call_id} terminal state is "
            f"ambiguous ({_production_exception_type_name(exc)})"
        ) from exc
    except RuntimeError as exc:
        if exc.args not in {
            (PRODUCTION_RL_TRAINING_TERMINAL_MARKER,),
            (PRODUCTION_RL_DISPATCHER_TERMINAL_MARKER,),
        }:
            raise RuntimeError(
                f"production RL FunctionCall {function_call_id} returned an "
                "unknown exception type "
                f"({_production_exception_type_name(exc)})"
            ) from exc
        result_category = "application_failure"
        exception_type = _production_exception_type_name(exc)
    except Exception as exc:
        raise RuntimeError(
            f"production RL FunctionCall {function_call_id} returned an "
            "unknown exception type "
            f"({_production_exception_type_name(exc)})"
        ) from exc
    else:
        if not allow_success:
            raise RuntimeError(
                f"production RL FunctionCall {function_call_id} completed "
                "successfully; recovery is forbidden"
            )
        result_category = "success"
        exception_type = None
    return _production_self_hash(
        {
            "schema": PRODUCTION_TERMINAL_CALL_EVIDENCE_SCHEMA,
            "function_call_id": function_call_id,
            "result_category": result_category,
            "exception_type": exception_type,
        },
        hash_field="evidence_sha256",
    )


def _inspect_terminal_unsuccessful_production_call(
    function_call_id: str,
) -> dict[str, object]:
    if not isinstance(function_call_id, str) or not re.fullmatch(
        r"fc-[0-9A-Za-z]+", function_call_id
    ):
        raise RuntimeError("invalid prior production RL FunctionCall ID")
    function_call = modal.FunctionCall.from_id(function_call_id)
    return _authoritative_production_call_evidence(
        function_call.get,
        function_call_id=function_call_id,
        allow_success=False,
    )


def _inspect_terminal_completed_production_call(
    function_call_id: str,
) -> dict[str, object]:
    """Require a terminal worker before accepting its complete checkpoint."""

    if not isinstance(function_call_id, str) or not re.fullmatch(
        r"fc-[0-9A-Za-z]+", function_call_id
    ):
        raise RuntimeError("invalid prior production RL FunctionCall ID")
    function_call = modal.FunctionCall.from_id(function_call_id)
    return _authoritative_production_call_evidence(
        function_call.get,
        function_call_id=function_call_id,
        allow_success=True,
    )


def _authenticated_production_recovery_checkpoint(
    *,
    run_name: str,
    launch_identity: Mapping[str, object],
) -> dict[str, object]:
    """Require absent output or a committed checkpoint with matching provenance."""

    run_root = Path(RAW_RL_ROOT) / run_name
    if not run_root.exists():
        return {"state": "absent", "run_root": str(run_root)}
    step = _reconcile_modal_checkpoint_root(run_root)
    if step is None:
        raise RuntimeError(
            f"{run_name} has output but no authenticated committed checkpoint"
        )
    root_provenance = run_root / "run_provenance.json"
    if not root_provenance.is_file() or root_provenance.is_symlink():
        raise RuntimeError(f"{run_name} committed checkpoint lacks run provenance")
    provenance = json.loads(root_provenance.read_text(encoding="utf-8"))
    if provenance.get("identity_sha256") != _canonical_json_sha256(
        provenance.get("identity")
    ):
        raise RuntimeError(f"{run_name} run provenance self identity drifted")
    expected = dict(launch_identity)
    if provenance.get("identity", {}).get("run", {}).get("run_name") != run_name:
        raise RuntimeError(f"{run_name} checkpoint provenance run name drifted")
    recorded = provenance.get("identity")
    if not isinstance(recorded, Mapping):
        raise RuntimeError(f"{run_name} checkpoint provenance identity is missing")
    fixed = recorded.get("fixed_rl_semantics")
    run = recorded.get("run")
    data = recorded.get("training_data")
    origin = recorded.get("origin_hf")
    semantics = expected["semantics"]
    expected_fields = {
        "model_id": semantics["model_id"],
        "num_rollout": semantics["target_updates"],
        "rollout_seed": semantics["rollout_seed"],
        "dynamic_filter": semantics["dynamic_filter"],
        "save_interval": semantics["save_interval"],
        "eval_interval": semantics["eval_interval"],
        "deterministic_inference": semantics["deterministic_inference"],
        "resume_if_available": semantics["resume_if_available"],
        "wandb_entity": expected["wandb"]["entity"],
        "wandb_project": expected["wandb"]["project"],
        "wandb_group": expected["wandb"]["group"],
        "wandb_run_id": expected["wandb"]["run_id"],
    }
    if not isinstance(run, Mapping) or any(
        run.get(key) != value for key, value in expected_fields.items()
    ):
        raise RuntimeError(f"{run_name} checkpoint run semantics drifted")
    if not isinstance(fixed, Mapping) or any(
        fixed.get(key) != value
        for key, value in {
            "lr": semantics["lr"],
            "kl_loss_type": semantics["kl_loss_type"],
            "kl_loss_coef": 0.001,
            "rollout_batch_size": 256,
            "samples_per_prompt": 8,
            "global_batch_size": 2_048,
            "rollout_max_prompt_len": semantics["rollout_max_prompt_len"],
            "rollout_max_response_len": semantics["rollout_max_response_len"],
            "rollout_max_context_len": semantics["rollout_max_context_len"],
        }.items()
    ):
        raise RuntimeError(f"{run_name} checkpoint RL semantics drifted")
    profile = recorded.get("policy_update_profile")
    if not isinstance(profile, Mapping) or any(
        profile.get(key) != value
        for key, value in {
            "max_tokens_per_gpu": semantics["max_tokens_per_gpu"],
            "sglang_server_concurrency": semantics[
                "sglang_server_concurrency"
            ],
            "master_parameter_dtype": "float32",
            "optimizer_state_dtype": "float32",
            "forward_backward_dtype": "bfloat16",
            "gradient_reduction_dtype": "float32",
        }.items()
    ):
        raise RuntimeError(f"{run_name} checkpoint update profile drifted")
    if data != expected["training_data"] or origin != expected["origin_hf"]:
        raise RuntimeError(f"{run_name} checkpoint model/data identity drifted")
    deployment = expected["deployment"]
    recorded_sources = recorded.get("sources")
    normalized_recorded_sources = None
    if isinstance(recorded_sources, Mapping):
        normalized_recorded_sources = {
            name: {
                key: value
                for key, value in dict(source).items()
                if key != "logical_root"
            }
            for name, source in recorded_sources.items()
            if isinstance(source, Mapping)
        }
    if (
        normalized_recorded_sources != deployment["sources"]
        or recorded.get("runtime") != deployment["runtime"]
    ):
        raise RuntimeError(f"{run_name} checkpoint source/runtime identity drifted")
    if provenance.get("initial_command_sha256") != expected[
        "initial_command_sha256"
    ]:
        raise RuntimeError(f"{run_name} checkpoint initial command drifted")
    state = (
        "complete"
        if step >= int(semantics["target_updates"])
        else "resumable"
    )
    return {
        "state": state,
        "run_root": str(run_root),
        "checkpoint_step": step,
        "checkpoint_marker_sha256": _sha256(
            run_root / f"iter_{step:07d}" / CHECKPOINT_COMMIT_MARKER
        ),
        "run_provenance_sha256": _sha256(root_provenance),
    }


def _validate_production_durable_completion(
    *,
    run_name: str,
    launch_identity: Mapping[str, object],
) -> dict[str, object]:
    completion = _validate_production_self_hash(
        json.loads(
            _production_durable_completion_path(run_name).read_text(
                encoding="utf-8"
            )
        ),
        hash_field="completion_sha256",
        label=f"{run_name} durable production RL completion",
    )
    if set(completion) != {
        "schema",
        "app_name",
        "run_name",
        "launch_identity_sha256",
        "anchor_sha256",
        "claim_sha256",
        "attempt_sha256",
        "resolution_sha256",
        "execution_sha256",
        "function_call_id",
        "target_updates",
        "checkpoint",
        "completion_sha256",
    } or completion.get("schema") != PRODUCTION_DURABLE_COMPLETION_SCHEMA:
        raise RuntimeError(f"{run_name} durable completion fields drifted")
    if completion.get("app_name") != APP_NAME or completion.get(
        "run_name"
    ) != run_name:
        raise RuntimeError(f"{run_name} durable completion identity drifted")
    if completion.get("launch_identity_sha256") != launch_identity.get(
        "identity_sha256"
    ):
        raise RuntimeError(f"{run_name} durable completion launch identity drifted")
    target = int(dict(launch_identity["semantics"])["target_updates"])
    checkpoint = completion.get("checkpoint")
    if (
        int(completion.get("target_updates", -1)) != target
        or not isinstance(checkpoint, Mapping)
        or checkpoint.get("state") != "complete"
        or int(checkpoint.get("checkpoint_step", -1)) != target
    ):
        raise RuntimeError(f"{run_name} durable completion target drifted")
    return completion


def _publish_production_durable_completion(
    *,
    run_name: str,
    launch_identity: Mapping[str, object],
    binding: Mapping[str, object],
    checkpoint: Mapping[str, object],
    target_updates: int,
    volume: Any = None,
) -> dict[str, object]:
    """Publish immutable completion only after exact checkpoint readback."""

    mounted_volume = base.ckpt_vol if volume is None else volume
    mounted_volume.reload()
    claim = dict(binding["claim"])
    attempt = dict(binding["attempt"])
    resolution = dict(binding["resolution"])
    execution = dict(binding["execution"])
    anchor = _validate_production_durable_anchor(
        run_name=run_name,
        claim=claim,
        launch_identity=launch_identity,
    )
    if checkpoint.get("state") != "complete" or int(
        checkpoint.get("checkpoint_step", -1)
    ) != int(target_updates):
        raise RuntimeError(f"{run_name} completion checkpoint is not exact")
    proposed = _production_self_hash(
        {
            "schema": PRODUCTION_DURABLE_COMPLETION_SCHEMA,
            "app_name": APP_NAME,
            "run_name": run_name,
            "launch_identity_sha256": launch_identity["identity_sha256"],
            "anchor_sha256": anchor["anchor_sha256"],
            "claim_sha256": claim["claim_sha256"],
            "attempt_sha256": attempt["attempt_sha256"],
            "resolution_sha256": resolution["resolution_sha256"],
            "execution_sha256": execution["execution_sha256"],
            "function_call_id": execution["function_call_id"],
            "target_updates": int(target_updates),
            "checkpoint": dict(checkpoint),
        },
        hash_field="completion_sha256",
    )
    path = _production_durable_completion_path(run_name)
    if path.exists():
        observed = _validate_production_durable_completion(
            run_name=run_name,
            launch_identity=launch_identity,
        )
        if observed != proposed:
            raise RuntimeError(f"{run_name} durable completion already differs")
        return observed
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        _exclusive_json(path, proposed)
    except FileExistsError:
        mounted_volume.reload()
        observed = _validate_production_durable_completion(
            run_name=run_name,
            launch_identity=launch_identity,
        )
        if observed != proposed:
            raise RuntimeError(f"{run_name} durable completion already differs")
        return observed
    mounted_volume.commit()
    mounted_volume.reload()
    observed = _validate_production_durable_completion(
        run_name=run_name,
        launch_identity=launch_identity,
    )
    if observed != proposed:
        raise RuntimeError(f"{run_name} durable completion publication drifted")
    return observed


@app.function(cpu=1.0, memory=1024, timeout=5 * 60)
def deployment_identity() -> dict[str, object]:
    """Identify the persistent deployment that will execute every run."""

    return _current_deployment_identity()


def _current_deployment_identity() -> dict[str, object]:
    """Build the exact identity used by deployment preflight and workers."""

    source = _precision_gate_source_identity()
    return {
        "schema": "chess-rl-miles-modal-deployment-identity-v1",
        "precision_resume_gate_version": PRECISION_RESUME_GATE_VERSION,
        **source,
        "runtime": _hydrated_modal_runtime_identity(),
    }


DEPLOYED_FUNCTION_NAMES = frozenset(
    {
        "deployment_identity",
        "deployment_dependency_preflight",
        "dispatch_production_train",
        "train_hf",
        "precision_resume_gate_leg",
        "finalize_precision_resume_gate",
        "convert_rl_to_hf",
        "v2r4_gate_preflight",
        "v2r4_gate_rollout",
    }
)


def _deployment_command() -> list[str]:
    return ["modal", "deploy", str(Path(__file__).resolve())]


def _deployed_function(name: str):
    """Resolve work only from the stable, named Modal deployment."""

    if name not in DEPLOYED_FUNCTION_NAMES:
        raise ValueError(f"unknown deployed function: {name}")
    return modal.Function.from_name(APP_NAME, name)


def _require_matching_deployment() -> dict[str, object]:
    """Refuse an ephemeral or stale app before allocating any GPU."""

    try:
        identity = _deployed_function("deployment_identity").remote()
    except Exception as exc:
        raise RuntimeError(
            "No matching persistent Modal deployment is available. Run "
            f"{' '.join(_deployment_command())} once, then retry this action."
        ) from exc
    if not isinstance(identity, dict):
        raise RuntimeError("deployed identity response is not an object")
    local_source = _local_precision_gate_source_identity()
    expected = {
        "schema": "chess-rl-miles-modal-deployment-identity-v1",
        "precision_resume_gate_version": PRECISION_RESUME_GATE_VERSION,
        **local_source,
    }
    for key, value in expected.items():
        if identity.get(key) != value:
            raise RuntimeError(
                "persistent Modal deployment does not match local launch "
                f"source for {key}; redeploy with "
                f"{' '.join(_deployment_command())}"
            )
    runtime = identity.get("runtime")
    if not isinstance(runtime, dict):
        raise RuntimeError("persistent deployment lacks runtime identity")
    if runtime.get("modal_app_name") != APP_NAME:
        raise RuntimeError("persistent deployment app name drifted")
    if runtime.get("image") != base.MILES_IMAGE or "@sha256:" not in str(
        runtime.get("image", "")
    ):
        raise RuntimeError("persistent deployment image digest drifted")
    if not str(runtime.get("modal_app_id", "")).startswith("ap-"):
        raise RuntimeError("persistent deployment has no hydrated app ID")
    if not str(runtime.get("modal_image_id", "")).startswith("im-"):
        raise RuntimeError("persistent deployment has no hydrated image ID")
    if not re.fullmatch(
        r"[0-9a-f]{64}",
        str(runtime.get("installed_packages_sha256", "")),
    ):
        raise RuntimeError(
            "persistent deployment lacks an authenticated Python package inventory"
        )
    if not re.fullmatch(
        r"[0-9a-f]{64}",
        str(runtime.get("os_package_inventory_sha256", "")),
    ):
        raise RuntimeError(
            "persistent deployment lacks an authenticated OS package inventory"
        )
    return identity


@app.function(
    cpu=2.0,
    memory=8 * 1024,
    timeout=30 * 60,
    retries=0,
    max_containers=8,
    volumes={
        "/data": base.data_vol,
        "/rl-checkpoints": base.ckpt_vol,
        PRETRAIN_CKPT_ROOT: pretrain_ckpt_vol,
        base.HF_CACHE_DIR: base.hf_cache,
    },
)
@_wrap_deployed_terminal_failure(PRODUCTION_RL_DISPATCHER_TERMINAL_MARKER)
def dispatch_production_train(
    *,
    hf_checkpoint: str,
    run_name: str,
    num_rollout: int,
    dynamic_filter: bool,
    rollout_seed: int,
    save_interval: int,
    eval_interval: int,
    model_id: str,
    resume_if_available: bool,
    wandb_project: str,
    wandb_group: str,
    max_tokens_per_gpu: int,
    sglang_server_concurrency: int,
    deterministic_inference: bool,
    train_file: str,
    train_file_sha256: str,
    lr: str,
    kl_loss_type: str,
    rollout_max_prompt_len: int,
    rollout_max_response_len: int,
    rollout_max_context_len: int,
    production_launch_token: str,
    recovery: bool,
    initial_adam_checkpoint: str = "",
    initial_adam_completion_sha256: str = "",
    initial_adam_source_tree_sha256: str = "",
    initial_adam_step: int = 0,
) -> dict[str, object]:
    """Claim, authenticate, and spawn one production GPU call exactly once."""

    _safe_component(run_name, name="run_name")
    _require_production_launch_token(production_launch_token)
    if not isinstance(num_rollout, int) or isinstance(num_rollout, bool) or num_rollout <= 0:
        raise RuntimeError("production RL target updates must be positive")
    if not resume_if_available:
        raise RuntimeError("production RL requires authenticated checkpoint resume")
    if model_id != CONTEXT2048_MODEL_ID or (
        rollout_max_prompt_len,
        rollout_max_response_len,
        rollout_max_context_len,
    ) != (512, 1_536, 2_048):
        raise RuntimeError("production RL dispatcher requires native context 2048")
    if dynamic_filter or not deterministic_inference:
        raise RuntimeError(
            "production RL dispatcher requires offline filtering and "
            "deterministic sample-index seeding"
        )
    pretrain_ckpt_vol.reload()
    base.data_vol.reload()
    base.ckpt_vol.reload()
    hf_checkpoint = str(Path(hf_checkpoint).resolve(strict=True))
    logical_train_file = _validated_logical_file_path(
        train_file,
        name="production RL training parquet",
    )
    resolved_train_file = str(Path(logical_train_file).resolve(strict=True))
    _, gate_contract = _precision_gate_contract_from_inputs(
        hf_checkpoint=hf_checkpoint,
        model_id=model_id,
        train_file=logical_train_file,
        train_file_sha256=train_file_sha256,
        rollout_seed=rollout_seed,
        wandb_project=wandb_project,
        max_tokens_per_gpu=max_tokens_per_gpu,
        sglang_server_concurrency=sglang_server_concurrency,
        lr=lr,
        kl_loss_type=kl_loss_type,
        rollout_max_prompt_len=rollout_max_prompt_len,
        rollout_max_response_len=rollout_max_response_len,
        rollout_max_context_len=rollout_max_context_len,
        initial_adam_checkpoint=initial_adam_checkpoint,
        initial_adam_completion_sha256=initial_adam_completion_sha256,
        initial_adam_source_tree_sha256=initial_adam_source_tree_sha256,
        initial_adam_step=initial_adam_step,
    )
    gate_evidence = _require_precision_resume_gate(contract=gate_contract)
    deployment = _current_deployment_identity()
    launch_identity = _production_launch_identity(
        hf_checkpoint=hf_checkpoint,
        run_name=run_name,
        num_rollout=num_rollout,
        dynamic_filter=dynamic_filter,
        rollout_seed=rollout_seed,
        save_interval=save_interval,
        eval_interval=eval_interval,
        model_id=model_id,
        resume_if_available=resume_if_available,
        wandb_project=wandb_project,
        wandb_group=wandb_group,
        max_tokens_per_gpu=max_tokens_per_gpu,
        sglang_server_concurrency=sglang_server_concurrency,
        deterministic_inference=deterministic_inference,
        train_file=resolved_train_file,
        train_file_sha256=train_file_sha256,
        lr=lr,
        kl_loss_type=kl_loss_type,
        rollout_max_prompt_len=rollout_max_prompt_len,
        rollout_max_response_len=rollout_max_response_len,
        rollout_max_context_len=rollout_max_context_len,
        deployment_identity=deployment,
        initial_adam_checkpoint=initial_adam_checkpoint,
        initial_adam_completion_sha256=initial_adam_completion_sha256,
        initial_adam_source_tree_sha256=initial_adam_source_tree_sha256,
        initial_adam_step=initial_adam_step,
    )
    store = production_launch_claims
    dispatcher_call_id = modal.current_function_call_id()
    recorded_claim = store.get(_production_claim_key(run_name), None)
    durable_anchor_path = _production_durable_anchor_path(run_name)
    run_root = Path(RAW_RL_ROOT) / run_name
    if recorded_claim is None and durable_anchor_path.exists():
        raise RuntimeError(
            f"{run_name} durable launch anchor exists but its Modal Dict "
            "claim expired or is missing; refusing a fresh claim"
        )
    if recorded_claim is None and run_root.exists():
        raise RuntimeError(
            f"{run_name} has an existing output root without a current "
            "launch claim; refusing a fresh production launch"
        )
    if recorded_claim is None:
        acquired = _acquire_production_claim(
            store,
            run_name=run_name,
            launch_token=production_launch_token,
            launch_identity=launch_identity,
        )
        claim = acquired["claim"]
        if acquired["outcome"] != "acquired":
            return {
                "outcome": "claim_lost",
                "spawned": False,
                "run_name": run_name,
            }
    else:
        claim = _validate_production_claim(
            recorded_claim,
            run_name=run_name,
            expected_identity=launch_identity,
            launch_token=production_launch_token,
        )
    assert isinstance(claim, Mapping)
    _ensure_production_durable_anchor(
        store,
        run_name=run_name,
        claim=claim,
        launch_identity=launch_identity,
        launch_token=production_launch_token,
        dispatcher_function_call_id=dispatcher_call_id,
        recovery=recovery,
    )
    current = _current_production_attempt(
        store,
        run_name=run_name,
        claim=claim,
    )
    recovery_evidence: Mapping[str, object] | None = None
    if current is None:
        generation = 0
    else:
        generation = int(current["generation"])
        same_dispatcher = (
            str(current["dispatcher_function_call_id"]) == dispatcher_call_id
        )
        resolution_record = store.get(
            _production_resolution_key(run_name, generation),
            None,
        )
        execution_record = store.get(
            _production_execution_key(run_name, generation),
            None,
        )
        execution = None
        if resolution_record is not None:
            resolution = _validate_production_generation_resolution(
                resolution_record,
                run_name=run_name,
                claim=claim,
                attempt=current,
                generation=generation,
            )
        else:
            resolution = None
        if execution_record is not None:
            execution = _validate_production_execution(
                execution_record,
                run_name=run_name,
                claim=claim,
                attempt=current,
                generation=generation,
            )
            if resolution is None or (
                resolution.get("decision") != "worker_bound"
                or resolution.get("function_call_id")
                != execution.get("function_call_id")
            ):
                raise RuntimeError(
                    f"{run_name} execution exists without its exact worker "
                    "generation resolution"
                )
            if same_dispatcher:
                return {
                    "outcome": "already_spawned_by_same_dispatcher",
                    "spawned": False,
                    "run_name": run_name,
                    "generation": generation,
                    "function_call_id": execution["function_call_id"],
                }
        if resolution is not None and resolution["decision"] == "worker_bound":
            worker_call_id = str(resolution["function_call_id"])
            if same_dispatcher:
                return {
                    "outcome": "already_spawned_by_same_dispatcher",
                    "spawned": False,
                    "run_name": run_name,
                    "generation": generation,
                    "function_call_id": worker_call_id,
                }
            if not recovery:
                return {
                    "outcome": "existing_active_or_unreconciled_call",
                    "spawned": False,
                    "run_name": run_name,
                    "generation": generation,
                    "function_call_id": worker_call_id,
                }
        elif resolution is not None and not recovery:
            return {
                "outcome": "generation_closed_for_recovery",
                "spawned": False,
                "run_name": run_name,
                "generation": generation,
            }
        elif resolution is None and same_dispatcher:
            raise RuntimeError(
                f"{run_name} dispatcher replay found its unbound generation "
                f"{generation}; recovery must authenticate this dispatcher "
                "as terminal unsuccessful"
            )
        elif resolution is None and not recovery:
            return {
                "outcome": "existing_unbound_attempt",
                "spawned": False,
                "run_name": run_name,
                "generation": generation,
            }

        if recovery and not same_dispatcher:
            terminal_call: Mapping[str, object]
            if resolution is None:
                dispatcher_terminal = (
                    _inspect_terminal_unsuccessful_production_call(
                        str(current["dispatcher_function_call_id"])
                    )
                )
                # The checkpoint read must happen after the prior dispatcher
                # is authoritatively terminal. Reading it before this poll is
                # a TOCTOU race with a worker the dispatcher may still spawn.
                base.ckpt_vol.reload()
                checkpoint_evidence = (
                    _authenticated_production_recovery_checkpoint(
                        run_name=run_name,
                        launch_identity=launch_identity,
                    )
                )
                if checkpoint_evidence["state"] not in {
                    "resumable",
                    "complete",
                }:
                    raise RuntimeError(
                        "production RL recovery requires an authenticated "
                        "committed checkpoint and provenance; absent output "
                        "cannot authorize a different FunctionCall"
                    )
                proposed_recovery = _production_self_hash(
                    {
                        "schema": (
                            "chess-rl-miles-production-recovery-evidence-v1"
                        ),
                        "prior_attempt_sha256": current["attempt_sha256"],
                        "terminal_call": dispatcher_terminal,
                        "checkpoint": checkpoint_evidence,
                    },
                    hash_field="recovery_sha256",
                )
                decision = _close_or_observe_generation_for_recovery(
                    store,
                    run_name=run_name,
                    claim=claim,
                    attempt=current,
                    generation=generation,
                    recovery_evidence=proposed_recovery,
                )
                resolution = decision["resolution"]
                if resolution["decision"] == "recovery_closed":
                    sealed = dict(resolution["recovery_evidence"])
                    terminal_call = dict(sealed["terminal_call"])
                    base.ckpt_vol.reload()
                    checkpoint_evidence = (
                        _authenticated_production_recovery_checkpoint(
                            run_name=run_name,
                            launch_identity=launch_identity,
                        )
                    )
                    if checkpoint_evidence != sealed["checkpoint"]:
                        raise RuntimeError(
                            f"{run_name} checkpoint changed after generation "
                            f"{generation} was closed for recovery"
                        )
                else:
                    # The worker won the exact same CAS while recovery was
                    # checking the dispatcher. Authenticate that worker's
                    # terminal result, then discard the earlier checkpoint
                    # read and authenticate a new Volume snapshot.
                    terminal_call = (
                        _inspect_terminal_completed_production_call(
                            str(resolution["function_call_id"])
                        )
                    )
                    base.ckpt_vol.reload()
                    checkpoint_evidence = (
                        _authenticated_production_recovery_checkpoint(
                            run_name=run_name,
                            launch_identity=launch_identity,
                        )
                    )
            elif resolution["decision"] == "worker_bound":
                terminal_call = _inspect_terminal_completed_production_call(
                    str(resolution["function_call_id"])
                )
                base.ckpt_vol.reload()
                checkpoint_evidence = (
                    _authenticated_production_recovery_checkpoint(
                        run_name=run_name,
                        launch_identity=launch_identity,
                    )
                )
            else:
                sealed = dict(resolution["recovery_evidence"])
                terminal_call = dict(sealed["terminal_call"])
                base.ckpt_vol.reload()
                checkpoint_evidence = (
                    _authenticated_production_recovery_checkpoint(
                        run_name=run_name,
                        launch_identity=launch_identity,
                    )
                )
                if checkpoint_evidence != sealed["checkpoint"]:
                    raise RuntimeError(
                        f"{run_name} checkpoint changed after generation "
                        f"{generation} was closed for recovery"
                    )

            terminal_call = _validate_production_terminal_call_evidence(
                terminal_call,
                expected_function_call_id=str(resolution["function_call_id"]),
                allow_success=True,
            )
            checkpoint_state = checkpoint_evidence.get("state")
            if checkpoint_state == "complete":
                target_updates = int(
                    dict(launch_identity["semantics"])["target_updates"]
                )
                if int(checkpoint_evidence.get("checkpoint_step", -1)) != (
                    target_updates
                ):
                    raise RuntimeError(
                        f"{run_name} recovered completion is not the exact "
                        f"target update {target_updates}"
                    )
                completion_path = _production_durable_completion_path(run_name)
                if completion_path.exists():
                    durable_completion = (
                        _validate_production_durable_completion(
                            run_name=run_name,
                            launch_identity=launch_identity,
                        )
                    )
                    if durable_completion["checkpoint"] != checkpoint_evidence:
                        raise RuntimeError(
                            f"{run_name} durable completion checkpoint drifted"
                        )
                elif resolution["decision"] == "worker_bound":
                    execution_record = store.get(
                        _production_execution_key(run_name, generation),
                        None,
                    )
                    if execution_record is None:
                        raise RuntimeError(
                            f"{run_name} exact checkpoint lacks its durable "
                            "completion and worker execution binding"
                        )
                    execution = _validate_production_execution(
                        execution_record,
                        run_name=run_name,
                        claim=claim,
                        attempt=current,
                        generation=generation,
                    )
                    durable_completion = (
                        _publish_production_durable_completion(
                            run_name=run_name,
                            launch_identity=launch_identity,
                            binding={
                                "claim": claim,
                                "attempt": current,
                                "resolution": resolution,
                                "execution": execution,
                            },
                            checkpoint=checkpoint_evidence,
                            target_updates=target_updates,
                            volume=base.ckpt_vol,
                        )
                    )
                else:
                    raise RuntimeError(
                        f"{run_name} exact checkpoint lacks an immutable "
                        "durable completion record"
                    )
                return {
                    "outcome": "authenticated_completion",
                    "spawned": False,
                    "run_name": run_name,
                    "checkpoint": checkpoint_evidence,
                    "terminal_call": terminal_call,
                    "durable_completion": durable_completion,
                    "generation_resolution_sha256": resolution[
                        "resolution_sha256"
                    ],
                }
            if checkpoint_state != "resumable":
                raise RuntimeError(
                    "production RL recovery requires an authenticated committed "
                    "checkpoint and provenance; absent output cannot authorize "
                    "a different FunctionCall"
                )
            if terminal_call["result_category"] == "success":
                raise RuntimeError(
                    f"{run_name} worker returned success before the exact "
                    "target checkpoint was durably committed"
                )
            if resolution["decision"] == "recovery_closed":
                unbound_recovery = resolution["recovery_evidence"]
            else:
                unbound_recovery = _production_self_hash(
                    {
                        "schema": (
                            "chess-rl-miles-production-recovery-evidence-v1"
                        ),
                        "prior_attempt_sha256": current["attempt_sha256"],
                        "terminal_call": terminal_call,
                        "checkpoint": checkpoint_evidence,
                    },
                    hash_field="recovery_sha256",
                )
            recovery_evidence = (
                _bind_recovery_evidence_to_generation_resolution(
                    unbound_recovery,
                    resolution=resolution,
                )
            )
            generation += 1

    if current is None or generation != int(current["generation"]):
        acquired_attempt = _acquire_production_attempt(
            store,
            run_name=run_name,
            claim=claim,
            generation=generation,
            dispatcher_function_call_id=dispatcher_call_id,
            recovery_evidence=recovery_evidence,
        )
        if acquired_attempt["outcome"] != "attempt_acquired":
            return {
                "outcome": "attempt_lost",
                "spawned": False,
                "run_name": run_name,
                "generation": generation,
            }

    worker_kwargs = {
        "hf_checkpoint": hf_checkpoint,
        "run_name": run_name,
        "num_rollout": num_rollout,
        "dynamic_filter": dynamic_filter,
        "rollout_seed": rollout_seed,
        "save_interval": save_interval,
        "eval_interval": eval_interval,
        "model_id": model_id,
        "resume_if_available": resume_if_available,
        "wandb_project": wandb_project,
        "wandb_group": wandb_group,
        "max_tokens_per_gpu": max_tokens_per_gpu,
        "sglang_server_concurrency": sglang_server_concurrency,
        "deterministic_inference": deterministic_inference,
        "rollout_only": False,
        "canary": False,
        "train_file": logical_train_file,
        "train_file_sha256": train_file_sha256,
        "lr": lr,
        "kl_loss_type": kl_loss_type,
        "rollout_max_prompt_len": rollout_max_prompt_len,
        "rollout_max_response_len": rollout_max_response_len,
        "rollout_max_context_len": rollout_max_context_len,
        "production_launch_token": production_launch_token,
        "production_launch_generation": generation,
        "initial_adam_checkpoint": initial_adam_checkpoint,
        "initial_adam_completion_sha256": initial_adam_completion_sha256,
        "initial_adam_source_tree_sha256": initial_adam_source_tree_sha256,
        "initial_adam_step": initial_adam_step,
    }
    call = _deployed_function("train_hf").spawn(**worker_kwargs)
    binding = _begin_claimed_production_worker(
        store,
        run_name=run_name,
        launch_token=production_launch_token,
        expected_identity=launch_identity,
        generation=generation,
        function_call_id=call.object_id,
    )
    return {
        "outcome": "recovery_spawned" if generation else "spawned",
        "spawned": True,
        "run_name": run_name,
        "generation": generation,
        "function_call_id": call.object_id,
        "claim_sha256": claim["claim_sha256"],
        "execution_sha256": binding["execution"]["execution_sha256"],
        "precision_gate": gate_evidence,
    }


def _precision_gate_paths(
    contract_sha256: str,
    *,
    run_name: str,
) -> tuple[Path, Path]:
    if not re.fullmatch(r"[0-9a-f]{64}", contract_sha256):
        raise ValueError("precision gate contract SHA256 is invalid")
    _safe_component(run_name, name="run_name")
    run_root = Path(RAW_RL_ROOT) / run_name
    gate_root = Path(PRECISION_RESUME_GATE_ROOT) / contract_sha256
    return run_root, gate_root


def _self_hashed_payload(
    core: dict[str, object],
    *,
    hash_key: str,
) -> dict[str, object]:
    return {**core, hash_key: _canonical_json_sha256(core)}


def _load_self_hashed_json(
    path: Path,
    *,
    hash_key: str,
) -> dict[str, object]:
    value = json.loads(path.read_text())
    core = {key: item for key, item in value.items() if key != hash_key}
    if value.get(hash_key) != _canonical_json_sha256(core):
        raise ValueError(f"Authenticated JSON self-hash mismatch: {path}")
    return value


def _validate_published_precision_gate_files(
    gate_root: Path,
    *,
    contract: dict[str, object],
    success: dict[str, object],
) -> None:
    """Authenticate the complete immutable two-file gate publication."""

    if not gate_root.is_dir() or gate_root.is_symlink():
        raise RuntimeError(f"precision-gate publication is not a directory: {gate_root}")
    entries = sorted(path.name for path in gate_root.iterdir())
    if entries != ["CONTRACT.json", "PASSED.json"]:
        raise RuntimeError(
            f"precision-gate publication inventory drifted: {entries}"
        )
    if any(path.is_symlink() or not path.is_file() for path in gate_root.iterdir()):
        raise RuntimeError("precision-gate publication contains a non-file entry")
    if _canonical_json_sha256(
        json.loads((gate_root / "CONTRACT.json").read_text())
    ) != _canonical_json_sha256(contract):
        raise RuntimeError("published precision-gate contract drifted")
    existing = _load_self_hashed_json(
        gate_root / "PASSED.json",
        hash_key="success_sha256",
    )
    if _canonical_json_sha256(existing) != _canonical_json_sha256(success):
        raise RuntimeError("published precision-gate success evidence drifted")


def _publish_precision_gate_result(
    gate_root: Path,
    *,
    contract: dict[str, object],
    success: dict[str, object],
) -> str:
    """Publish CONTRACT then PASSED last under one output-scoped lock."""

    gate_root.parent.mkdir(parents=True, exist_ok=True)
    lock_path = gate_root.parent / f".{gate_root.name}.publication.lock"
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(lock_path, flags, 0o644)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        if gate_root.exists() or gate_root.is_symlink():
            if (
                not gate_root.is_symlink()
                and gate_root.is_dir()
                and (gate_root / "PASSED.json").is_file()
            ):
                _validate_published_precision_gate_files(
                    gate_root,
                    contract=contract,
                    success=success,
                )
                return "authenticated-existing"
            quarantine = gate_root.with_name(
                f".{gate_root.name}.quarantine.{time.time_ns()}"
            )
            os.replace(gate_root, quarantine)
            reason = quarantine.with_name(f"{quarantine.name}.reason.txt")
            _atomic_text(
                reason,
                "precision-gate writer terminated before PASSED.json\n",
            )

        recovered_staging = False
        # Preserve every interrupted legacy staging tree. A complete identical
        # one proves prior work reached publication, but Modal Volumes cannot
        # rename that directory with RENAME_NOREPLACE, so recreate the two-file
        # final under the lock and retain the old tree as superseded evidence.
        for prior in sorted(
            gate_root.parent.glob(f".{gate_root.name}.*.incomplete")
        ):
            try:
                _validate_published_precision_gate_files(
                    prior,
                    contract=contract,
                    success=success,
                )
            except Exception as exc:
                quarantine = prior.with_name(
                    f"{prior.name}.quarantine.{time.time_ns()}"
                )
                os.replace(prior, quarantine)
                reason = quarantine.with_name(f"{quarantine.name}.reason.txt")
                _atomic_text(
                    reason,
                    "precision-gate staging was not an authenticated complete "
                    f"publication: {type(exc).__name__}: {exc}\n",
                )
            else:
                superseded = prior.with_name(
                    f"{prior.name}.superseded.{time.time_ns()}"
                )
                os.replace(prior, superseded)
                recovered_staging = True

        # Modal Volumes reject renameat2(RENAME_NOREPLACE) for directories.
        # Claim the final namespace directly. Readers accept it only after the
        # authenticated PASSED.json completion marker is written last.
        gate_root.mkdir(parents=False, exist_ok=False)
        try:
            _exclusive_json(gate_root / "CONTRACT.json", contract)
            _fsync_directory(gate_root)
            _fsync_directory(gate_root.parent)
            _exclusive_json(gate_root / "PASSED.json", success)
            _fsync_directory(gate_root)
            _fsync_directory(gate_root.parent)
            _validate_published_precision_gate_files(
                gate_root,
                contract=contract,
                success=success,
            )
        except BaseException:
            # Leave marker-less output in place after a hard interruption. A
            # retry holds this same lock and quarantines it before proceeding.
            raise
        return "recovered-staging" if recovered_staging else "published"
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _small_evidence_manifest(run_root: Path) -> list[dict[str, object]]:
    evidence_files = []
    for path in sorted((run_root / "precision_gate").rglob("*")):
        if (
            not path.is_file()
            or path.name == "PASSED.json"
            or path.suffix not in {".json", ".jsonl"}
        ):
            continue
        evidence_files.append(
            {
                "path": path.relative_to(run_root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    if not evidence_files:
        raise RuntimeError("Precision gate produced no persisted evidence")
    return evidence_files


def _require_precision_resume_gate(
    *,
    contract: dict[str, object],
) -> dict[str, object]:
    """Require a passing gate for the complete immutable runtime contract."""

    contract_sha256 = str(contract["contract_sha256"])
    run_root, gate_root = _precision_gate_paths(
        contract_sha256,
        run_name=str(contract["run_name"]),
    )
    passed_path = gate_root / "PASSED.json"
    if not passed_path.is_file():
        raise RuntimeError(
            "Production RL is blocked until the two-process BF16/FP32 "
            f"precision-resume gate passes for contract {contract_sha256}: "
            f"{passed_path}"
        )
    passed = _load_self_hashed_json(
        passed_path,
        hash_key="success_sha256",
    )
    if (
        passed.get("schema")
        != "chess-rl-miles-precision-resume-gate-success-v1"
        or passed.get("version") != PRECISION_RESUME_GATE_VERSION
        or passed.get("contract_sha256") != contract_sha256
        or passed.get("passed") is not True
    ):
        raise RuntimeError("Precision-resume gate success contract drifted")
    recorded_contract = json.loads((gate_root / "CONTRACT.json").read_text())
    if recorded_contract != contract:
        raise RuntimeError("Precision-resume gate contract artifact drifted")
    _validate_published_precision_gate_files(
        gate_root,
        contract=contract,
        success=passed,
    )
    recorded_files = passed.get("evidence_files")
    if not isinstance(recorded_files, list):
        raise RuntimeError("Precision-resume gate lacks an evidence manifest")
    for row in recorded_files:
        path = Path(str(row.get("absolute_path")))
        try:
            path.resolve().relative_to(run_root.resolve())
        except ValueError as exc:
            raise RuntimeError("Unsafe precision-gate evidence path") from exc
        if (
            not path.is_file()
            or path.stat().st_size != row.get("bytes")
            or _sha256(path) != row.get("sha256")
        ):
            raise RuntimeError(
                f"Precision-resume gate evidence drifted: {path}"
            )
    checkpoint_records = dict(passed.get("checkpoints") or {})
    for step in (1, 2):
        marker = _validated_checkpoint_commit(
            run_root / f"iter_{step:07d}",
            expected_step=step,
        )
        if marker.get("commit_sha256") != dict(
            checkpoint_records.get(str(step)) or {}
        ).get("commit_sha256"):
            raise RuntimeError(
                f"Precision-resume checkpoint {step} drifted after gate success"
            )
    export_record = dict(passed.get("export") or {})
    export_path = Path(str(export_record.get("path") or ""))
    if str(Path(base.MILES_DIR)) not in sys.path:
        sys.path.insert(0, str(Path(base.MILES_DIR)))
    from tools.convert_fsdp_to_hf import validate_committed_hf_export

    export_marker = validate_committed_hf_export(export_path)
    if (
        export_marker.get("commit_sha256")
        != export_record.get("marker_commit_sha256")
    ):
        raise RuntimeError("Precision-resume FP32 HF export drifted")
    return {
        "path": str(passed_path),
        "contract_sha256": contract_sha256,
        "success_sha256": passed["success_sha256"],
        "wandb": passed.get("wandb"),
    }


def _reuse_authenticated_precision_gate_result(
    *,
    gate_root: Path,
    contract: dict[str, object],
) -> dict[str, object] | None:
    """Return an immutable prior finalizer result after full authentication."""

    passed_path = gate_root / "PASSED.json"
    if not passed_path.is_file():
        return None
    # Do not trust existence alone. This revalidates the contract, evidence,
    # checkpoints, and canonical FP32 export before a retry returns success.
    _require_precision_resume_gate(contract=contract)
    existing = _load_self_hashed_json(
        passed_path,
        hash_key="success_sha256",
    )
    return {**existing, "passed_path": str(passed_path)}


def _precision_gate_contract(
    *,
    checkpoint: Path,
    origin_authentication: dict[str, object],
    model_id: str,
    train_file: str,
    train_file_sha256: str,
    rollout_seed: int,
    wandb_project: str,
    max_tokens_per_gpu: int,
    sglang_server_concurrency: int,
    lr: str,
    kl_loss_type: str,
    rollout_max_prompt_len: int,
    rollout_max_response_len: int,
    rollout_max_context_len: int,
    initial_adam_checkpoint: str = "",
    initial_adam_completion_sha256: str = "",
    initial_adam_source_tree_sha256: str = "",
    initial_adam_step: int = 0,
) -> dict[str, object]:
    if model_id != CONTEXT2048_MODEL_ID or (
        rollout_max_prompt_len,
        rollout_max_response_len,
        rollout_max_context_len,
    ) != (512, 1_536, 2_048):
        raise ValueError(
            "precision-resume gate requires the exact native-2048 contract: "
            f"model_id={CONTEXT2048_MODEL_ID}, prompt=512, response=1536, context=2048"
        )
    train_path = Path(train_file)
    if not train_path.is_file():
        raise FileNotFoundError(train_path)
    actual_train_sha256 = _sha256(train_path)
    if actual_train_sha256 != train_file_sha256:
        raise ValueError(
            "Precision-gate training parquet SHA256 mismatch: "
            f"expected={train_file_sha256} actual={actual_train_sha256}"
        )
    source = _precision_gate_source_identity()
    source_sha256 = str(source["source_sha256"])
    origin_hf_identity = directory_identity(
        checkpoint,
        logical_path=str(checkpoint),
    )
    initial_adam_values = (
        initial_adam_checkpoint,
        initial_adam_completion_sha256,
        initial_adam_source_tree_sha256,
        initial_adam_step,
    )
    if any(value not in ("", 0) for value in initial_adam_values) and not all(
        value not in ("", 0) for value in initial_adam_values
    ):
        raise ValueError("precision gate received an incomplete initial Adam identity")
    initial_adam_identity = (
        {
            "mode": "continue_adam_moments_and_parameter_steps",
            "checkpoint": initial_adam_checkpoint,
            "completion_sha256": initial_adam_completion_sha256,
            "source_tree_sha256": initial_adam_source_tree_sha256,
            "source_step": initial_adam_step,
            "destination_hyperparameters_preserved": True,
        }
        if initial_adam_checkpoint
        else {"mode": "fresh_adam_state"}
    )
    semantic_core = {
        "schema": "chess-rl-miles-precision-resume-gate-contract-v1",
        "version": PRECISION_RESUME_GATE_VERSION,
        "source_sha256": source_sha256,
        **source,
        "origin_hf": origin_hf_identity,
        "origin_hf_authentication": origin_authentication,
        "training_data": {
            "logical_path": train_file,
            "sha256": train_file_sha256,
        },
        "initial_optimizer_state": initial_adam_identity,
        "semantics": {
            "target_updates": 2,
            "first_call": "fresh rollout 0, optimizer update 1, checkpoint 1",
            "second_call": (
                "new process, explicit checkpoint 1 resume, rollout 1, "
                "optimizer update 2, checkpoint 2"
            ),
            "model_id": model_id,
            "rollout_seed": rollout_seed,
            "dynamic_filter": False,
            "rollout_batch_size": 256,
            "samples_per_prompt": 8,
            "global_batch_size": 2_048,
            "num_steps_per_rollout": 1,
            "lr": str(lr),
            "kl_loss_type": str(kl_loss_type),
            "kl_loss_coef": 0.001,
            "max_tokens_per_gpu": max_tokens_per_gpu,
            "sglang_server_concurrency": sglang_server_concurrency,
            "rollout_max_prompt_len": rollout_max_prompt_len,
            "rollout_max_response_len": rollout_max_response_len,
            "rollout_max_context_len": rollout_max_context_len,
            "sglang_dtype": "bfloat16",
            "miles_fp16": False,
            "training_compute_dtype": "bfloat16",
            "master_parameter_dtype": "float32",
            "gradient_reduction_dtype": "float32",
            "optimizer_state_dtype": "float32",
            "data_source_path": PRECISION_RESUME_DATA_SOURCE_PATH,
            "production_and_gate_share_data_source": True,
            "fault_tolerance": False,
            "minimum_admitted_prompts": 512,
            "one_prompt_request_per_process": 256,
            "no_dataset_wrap": True,
            "no_aborted_sample_requeue": True,
            "deterministic_inference": True,
            "deterministic_seed_rule": "rollout_seed_plus_global_sample_index",
            "prompt_reserved_prefix_tokens": 1,
            "chess_context_margin_tokens": 0,
            "save_interval": 1,
            "automatic_retries": 0,
        },
        "runtime": _hydrated_modal_runtime_identity(),
    }
    semantic_sha256 = _canonical_json_sha256(semantic_core)
    run_name = f"precision-resume-{semantic_sha256[:20]}"
    run_root = Path(RAW_RL_ROOT) / run_name
    wandb_run_id = f"prgate{semantic_sha256[:20]}"
    draft = {
        **semantic_core,
        "semantic_sha256": semantic_sha256,
        "run_name": run_name,
        "run_root": str(run_root),
        "wandb": {
            "entity": WANDB_ENTITY,
            "project": wandb_project,
            "group": PRECISION_RESUME_GATE_WANDB_GROUP,
            "run_id": wandb_run_id,
        },
    }
    first, second = _precision_gate_commands(draft)
    full_core = {
        **draft,
        "commands": {
            "leg_1": first,
            "leg_2": second,
            "leg_1_sha256": _canonical_json_sha256(first),
            "leg_2_sha256": _canonical_json_sha256(second),
        },
    }
    return {
        **full_core,
        "contract_sha256": _canonical_json_sha256(full_core),
    }


def _precision_gate_commands(
    contract: dict[str, object],
) -> tuple[list[str], list[str]]:
    semantics = dict(contract["semantics"])
    training_data = dict(contract["training_data"])
    origin_hf = dict(contract["origin_hf"])
    wandb = dict(contract["wandb"])
    common = {
        "hf_checkpoint": str(origin_hf["logical_path"]),
        "run_name": str(contract["run_name"]),
        "model_id": str(semantics["model_id"]),
        "dynamic_filter": False,
        "rollout_seed": int(semantics["rollout_seed"]),
        "save_interval": 1,
        "eval_interval": 0,
        "wandb_project": str(wandb["project"]),
        "wandb_group": str(wandb["group"]),
        "wandb_run_id": str(wandb["run_id"]),
        "max_tokens_per_gpu": int(semantics["max_tokens_per_gpu"]),
        "sglang_server_concurrency": int(
            semantics["sglang_server_concurrency"]
        ),
        "deterministic_inference": True,
        "rollout_only": False,
        "canary": True,
        "train_file": str(training_data["logical_path"]),
        "train_file_sha256": str(training_data["sha256"]),
        "lr": str(semantics["lr"]),
        "kl_loss_type": str(semantics["kl_loss_type"]),
        "fault_tolerance": False,
        "data_source_path": str(semantics["data_source_path"]),
        "deterministic_seed_by_sample_index": True,
        "rollout_max_prompt_len": int(
            semantics["rollout_max_prompt_len"]
        ),
        "rollout_max_response_len": int(
            semantics["rollout_max_response_len"]
        ),
        "rollout_max_context_len": int(
            semantics["rollout_max_context_len"]
        ),
        "initial_adam_checkpoint": str(
            dict(contract["initial_optimizer_state"]).get("checkpoint", "")
        ),
        "initial_adam_completion_sha256": str(
            dict(contract["initial_optimizer_state"]).get(
                "completion_sha256", ""
            )
        ),
        "initial_adam_source_tree_sha256": str(
            dict(contract["initial_optimizer_state"]).get(
                "source_tree_sha256", ""
            )
        ),
        "initial_adam_step": int(
            dict(contract["initial_optimizer_state"]).get("source_step", 0)
        ),
    }
    first = build_train_command(
        **common,
        num_rollout=1,
    )
    second = build_train_command(
        **common,
        num_rollout=2,
        resume_path=str(contract["run_root"]),
        resume_step=1,
    )
    return first, second


def _validate_precision_gate_contract(
    contract: dict[str, object],
) -> str:
    if not isinstance(contract, dict):
        raise TypeError("precision gate contract must be an object")
    core = {
        key: value
        for key, value in contract.items()
        if key != "contract_sha256"
    }
    actual = _canonical_json_sha256(core)
    if contract.get("contract_sha256") != actual:
        raise ValueError("precision gate contract self-hash mismatch")
    if (
        contract.get("schema")
        != "chess-rl-miles-precision-resume-gate-contract-v1"
        or contract.get("version") != PRECISION_RESUME_GATE_VERSION
    ):
        raise ValueError("precision gate contract schema/version mismatch")
    image = str(dict(contract["runtime"])["image"])
    if "@sha256:" not in image:
        raise ValueError(
            "precision gate requires an immutable OCI image digest"
        )
    _validate_hydrated_runtime_identity(contract["runtime"])
    first, second = _precision_gate_commands(contract)
    commands = dict(contract.get("commands") or {})
    expected_commands = {
        "leg_1": first,
        "leg_2": second,
        "leg_1_sha256": _canonical_json_sha256(first),
        "leg_2_sha256": _canonical_json_sha256(second),
    }
    if commands != expected_commands:
        raise ValueError("precision gate command binding drifted")
    for command in (first, second):
        if "--fp16" in command:
            raise ValueError("precision gate must not select FP16 compute")
        values = [
            command[index + 1]
            for index, item in enumerate(command[:-1])
            if item == "--sglang-dtype"
        ]
        if values != ["bfloat16"]:
            raise ValueError("precision gate rollout dtype must be bfloat16")
    return actual


def _precision_gate_contract_from_inputs(
    *,
    hf_checkpoint: str,
    model_id: str,
    train_file: str,
    train_file_sha256: str,
    rollout_seed: int,
    wandb_project: str,
    max_tokens_per_gpu: int,
    sglang_server_concurrency: int,
    lr: str,
    kl_loss_type: str,
    rollout_max_prompt_len: int,
    rollout_max_response_len: int,
    rollout_max_context_len: int,
    initial_adam_checkpoint: str = "",
    initial_adam_completion_sha256: str = "",
    initial_adam_source_tree_sha256: str = "",
    initial_adam_step: int = 0,
) -> tuple[Path, dict[str, object]]:
    checkpoint, origin_authentication = (
        _validate_authenticated_fp32_hf_checkpoint(hf_checkpoint)
    )
    _validate_checkpoint_context(
        checkpoint,
        requested_context_len=rollout_max_context_len,
        require_exact=model_id == CONTEXT2048_MODEL_ID,
    )
    contract = _precision_gate_contract(
        checkpoint=checkpoint,
        origin_authentication=origin_authentication,
        model_id=model_id,
        train_file=train_file,
        train_file_sha256=train_file_sha256,
        rollout_seed=rollout_seed,
        wandb_project=wandb_project,
        max_tokens_per_gpu=max_tokens_per_gpu,
        sglang_server_concurrency=sglang_server_concurrency,
        lr=lr,
        kl_loss_type=kl_loss_type,
        rollout_max_prompt_len=rollout_max_prompt_len,
        rollout_max_response_len=rollout_max_response_len,
        rollout_max_context_len=rollout_max_context_len,
        initial_adam_checkpoint=initial_adam_checkpoint,
        initial_adam_completion_sha256=initial_adam_completion_sha256,
        initial_adam_source_tree_sha256=initial_adam_source_tree_sha256,
        initial_adam_step=initial_adam_step,
    )
    _validate_precision_gate_contract(contract)
    return checkpoint, contract


def _read_authenticated_evidence_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"evidence is not an object: {path}")
    core = {
        key: item
        for key, item in value.items()
        if key != "evidence_sha256"
    }
    if value.get("evidence_sha256") != _canonical_json_sha256(core):
        raise ValueError(f"evidence self-hash mismatch: {path}")
    return value


def _read_authenticated_evidence_jsonl(
    path: Path,
) -> list[dict[str, object]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid evidence JSON at {path}:{line_number}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(
                    f"evidence row is not an object at {path}:{line_number}"
                )
            core = {
                key: item
                for key, item in value.items()
                if key != "evidence_sha256"
            }
            if value.get("evidence_sha256") != _canonical_json_sha256(core):
                raise ValueError(
                    f"evidence self-hash mismatch at {path}:{line_number}"
                )
            rows.append(value)
    if not rows:
        raise RuntimeError(f"evidence file is empty: {path}")
    return rows


def _validate_precision_checkpoint(
    run_root: Path,
    *,
    step: int,
) -> dict[str, object]:
    checkpoint = run_root / f"iter_{step:07d}"
    marker = _validated_checkpoint_commit(
        checkpoint,
        expected_step=step,
    )
    metadata = json.loads((checkpoint / "meta.json").read_text())
    expected_forward_dtypes = {
        "actor": "bfloat16",
        "actor_train": "bfloat16",
        "ref": "bfloat16",
    }
    if (
        metadata.get("iteration") != step
        or metadata.get("global_step") != step
        or metadata.get("next_rollout_id") != step
        or metadata.get("weight_version_at_save") != step
        or metadata.get("next_weight_version") != step + 1
        or metadata.get("forward_output_dtypes")
        != expected_forward_dtypes
        or metadata.get("required_forward_paths")
        != sorted(expected_forward_dtypes)
    ):
        raise ValueError(
            f"precision checkpoint accounting/dtype evidence drifted: {checkpoint}"
        )
    cursor = dict(metadata.get("rollout_state_summary") or {})
    expected_cursor = {
        "schema": "miles-rollout-data-source-v1",
        "rollout_id": step - 1,
        "next_rollout_id": step,
        "sample_offset": step * 256,
        "sample_group_index": step * 256,
        "sample_index": step * 2_048,
    }
    for key, expected in expected_cursor.items():
        if cursor.get(key) != expected:
            raise ValueError(
                f"precision checkpoint rollout cursor drifted at {key}: "
                f"expected={expected!r} actual={cursor.get(key)!r}"
            )
    return {
        "checkpoint": str(checkpoint),
        "commit_sha256": marker["commit_sha256"],
        "metadata_sha256": _sha256(checkpoint / "meta.json"),
        "dcp_precision": _inspect_dcp_fp32_precision(checkpoint),
        "cursor": cursor,
        "initial_adam_import": metadata.get("initial_adam_import"),
        "initial_adam_step_progression": metadata.get(
            "initial_adam_step_progression"
        ),
    }


def _validate_precision_runtime_evidence(
    run_root: Path,
    *,
    contract: dict[str, object],
    rollout_evidence: dict[str, object],
) -> dict[str, object]:
    evidence_root = run_root / "precision_gate"
    sglang_records: dict[str, list[dict[str, object]]] = {}
    metric_records: dict[str, list[dict[str, object]]] = {}
    weight_records: dict[str, list[dict[str, object]]] = {}
    for leg in (1, 2):
        sglang_paths = sorted(
            (evidence_root / "sglang_runtime").glob(
                f"leg_{leg}_engine_*_*.json"
            )
        )
        if len(sglang_paths) != 8:
            raise RuntimeError(
                f"precision gate leg {leg} requires eight live SGLang engines; "
                f"found {len(sglang_paths)}"
            )
        records = [
            _read_authenticated_evidence_json(path)
            for path in sglang_paths
        ]
        if (
            {int(row["engine_rank"]) for row in records} != set(range(8))
            or any(
                dict(row["actual"]).get("dtype") != "bfloat16"
                or dict(row["actual"]).get("context_length") != 2_048
                for row in records
            )
        ):
            raise RuntimeError(
                f"precision gate leg {leg} SGLang runtime evidence drifted"
            )
        sglang_records[str(leg)] = records

        metrics = _read_authenticated_evidence_jsonl(
            evidence_root / f"leg_{leg}_metrics.jsonl"
        )
        expected_train_step = leg - 1
        train_events = [
            row
            for row in metrics
            if row.get("step_key") == "train/step"
            and row.get("step") == expected_train_step
        ]
        if len(train_events) != 1:
            raise RuntimeError(
                f"precision gate leg {leg} lacks exactly one train step {expected_train_step}"
            )
        required_metrics = {
            "train/loss",
            "train/ppo_kl",
            "train/entropy_loss",
            "train/grad_norm",
        }
        if not required_metrics.issubset(
            dict(train_events[0]["metrics"])
        ):
            raise RuntimeError(
                f"precision gate leg {leg} lacks required finite training metrics"
            )
        expected_rollout_step = leg - 1
        rollout_events = [
            row
            for row in metrics
            if row.get("step_key") == "rollout/step"
            and row.get("step") == expected_rollout_step
        ]
        expected_outcome = dict(
            dict(rollout_evidence["per_rollout"])[str(expected_rollout_step)]
        )
        required_rollout_metrics = {
            "rollout/entropy": None,
            "rollout/zero_std/all_zero_percentage": expected_outcome[
                "all_zero_percentage"
            ],
        }
        for metric, expected in required_rollout_metrics.items():
            values = [
                dict(row["metrics"])[metric]
                for row in rollout_events
                if metric in dict(row["metrics"])
            ]
            if len(values) != 1 or not math.isfinite(float(values[0])):
                raise RuntimeError(
                    f"precision gate leg {leg} lacks one finite {metric} event"
                )
            if expected is not None and float(values[0]) != float(expected):
                raise RuntimeError(
                    f"precision gate leg {leg} {metric} disagrees with rollout rows"
                )
        metric_records[str(leg)] = metrics

        weights = _read_authenticated_evidence_jsonl(
            evidence_root / f"leg_{leg}_weight_versions.jsonl"
        )
        expected_pairs = (
            {(0, 1), (1, 2)} if leg == 1 else {(1, 2), (2, 3)}
        )
        observed_pairs = {
            (
                int(row["actor_global_step"]),
                int(row["expected_weight_version"]),
            )
            for row in weights
        }
        if observed_pairs != expected_pairs or any(
            not row["engine_weight_versions"]
            or set(row["engine_weight_versions"])
            != {str(row["expected_weight_version"])}
            for row in weights
        ):
            raise RuntimeError(
                f"precision gate leg {leg} weight-version progression drifted"
            )
        weight_records[str(leg)] = weights

        wandb_evidence = _read_authenticated_evidence_json(
            evidence_root / f"leg_{leg}_wandb.json"
        )
        wandb_contract = dict(contract["wandb"])
        expected_wandb = {
            "leg": leg,
            "entity": wandb_contract["entity"],
            "project": wandb_contract["project"],
            "group": wandb_contract["group"],
            "run_id": wandb_contract["run_id"],
        }
        if any(wandb_evidence.get(key) != value for key, value in expected_wandb.items()):
            raise RuntimeError("precision gate W&B leg identity drifted")
    return {
        "sglang": sglang_records,
        "metrics": metric_records,
        "weight_versions": weight_records,
    }


def _validate_precision_rollout_evidence(
    run_root: Path,
    *,
    rollout_seed: int,
    max_prompt_len: int,
    max_context_len: int,
) -> dict[str, object]:
    from argparse import Namespace

    from miles.utils.types import Sample

    from chess_rl_miles.reward import _score_sample

    from collections import Counter

    total_rows = 0
    total_model_tokens = 0
    total_env_tokens = 0
    total_env_calls = 0
    reward_agreements = 0
    artifacts: list[dict[str, object]] = []
    per_rollout: dict[str, dict[str, object]] = {}
    for rollout_id in (0, 1):
        path = (
            run_root
            / "rollouts"
            / "training"
            / f"rollout_{rollout_id}.jsonl"
        )
        with path.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle]
        if len(rows) != 2_048:
            raise RuntimeError(
                f"precision rollout {rollout_id} has {len(rows)} rows, expected 2,048"
            )
        artifacts.append(
            {
                "absolute_path": str(path),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
        expected_samples = set(
            range(rollout_id * 2_048, (rollout_id + 1) * 2_048)
        )
        observed_sample_order = [row.get("sample_index") for row in rows]
        if (
            set(observed_sample_order) != expected_samples
            or observed_sample_order != sorted(expected_samples)
        ):
            raise RuntimeError(
                f"precision rollout {rollout_id} sample identity drifted"
            )
        group_counts: Counter[int] = Counter()
        group_prompt_identities: dict[int, str] = {}
        rollout_rewards: list[float] = []
        rollout_model_tokens = 0
        rollout_env_tokens = 0
        rollout_env_calls = 0
        for row in rows:
            metadata = dict(row.get("metadata") or {})
            prompt_ids = list(row.get("prompt_token_ids") or [])
            response_ids = list(row.get("response_token_ids") or [])
            response_mask = list(row.get("response_loss_mask") or [])
            bos_id = metadata.get("chess_prompt_bos_token_id")
            if (
                not prompt_ids
                or prompt_ids[0] != bos_id
                or prompt_ids.count(bos_id) != 1
                or metadata.get("chess_prompt_bos_count") != 1
                or metadata.get("chess_prompt_token_count") != len(prompt_ids)
                or metadata.get("chess_prompt_first_token_id") != prompt_ids[0]
                or len(prompt_ids) > max_prompt_len
                or len(prompt_ids) + len(response_ids) > max_context_len
            ):
                raise RuntimeError("precision rollout BOS/context contract drifted")
            if (
                not response_ids
                or len(response_mask) != len(response_ids)
                or any(value not in {0, 1} for value in response_mask)
                or sum(response_mask) <= 0
                or sum(response_mask) != metadata.get("model_token_count")
                or response_mask.count(0) != metadata.get("env_token_count")
            ):
                raise RuntimeError("precision rollout supervised-token mask drifted")
            # Prompt tokens are never supervised. Constructing this complete
            # mask makes that implicit storage convention explicit in the gate.
            complete_mask = [0] * len(prompt_ids) + response_mask
            if any(complete_mask[: len(prompt_ids)]):
                raise AssertionError("precision rollout prompt mask is nonzero")
            sample_index = int(row["sample_index"])
            expected_group_index = sample_index // 8
            expected_sibling_index = sample_index % 8
            group_counts[int(row.get("group_index", -1))] += 1
            prompt_identity = _canonical_json_sha256(
                {
                    "input": row.get("input"),
                    "label": row.get("label"),
                    "prompt_token_ids": prompt_ids,
                    "source": {
                        key: metadata.get(key)
                        for key in (
                            "source_row_index",
                            "source_row_fingerprint",
                            "PuzzleId",
                            "FEN",
                            "Moves",
                            "difficulty",
                            "Rating",
                        )
                        if key in metadata
                    },
                }
            )
            prior_prompt_identity = group_prompt_identities.setdefault(
                expected_group_index,
                prompt_identity,
            )
            if prior_prompt_identity != prompt_identity:
                raise RuntimeError(
                    f"precision rollout group {expected_group_index} mixes prompt/source identities"
                )
            if (
                row.get("rollout_id") != rollout_id
                or row.get("group_index") != expected_group_index
                or metadata.get("sampling_seed_sibling_index")
                != expected_sibling_index
                or row.get("sampling_seed")
                != rollout_seed + sample_index
                or metadata.get("sampling_seed")
                != rollout_seed + sample_index
                or metadata.get("sampling_seed_mode") != "sample-index"
                or set(row.get("weight_versions") or [])
                != {str(rollout_id + 1)}
            ):
                raise RuntimeError("precision rollout seed/weight identity drifted")
            expected_token_hash = hashlib.sha256(
                json.dumps(
                    [prompt_ids, response_ids],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            if (
                row.get("token_artifact_schema") != 1
                or row.get("token_ids_sha256") != expected_token_hash
                or row.get("status") not in {"completed", "truncated"}
                or row.get("response_length") != len(response_ids)
                or row.get("effective_response_length") != sum(response_mask)
            ):
                raise RuntimeError(
                    "precision rollout terminal/token artifact contract drifted"
                )

            sample = Sample(
                prompt=row.get("input") or "",
                response=row.get("output") or "",
                label=row.get("label"),
                metadata=metadata,
            )
            rescored = _score_sample(
                Namespace(
                    chess_reward_model_type="RULE_BASED",
                    chess_multiturn=True,
                    chess_difficulty_threshold=1500.0,
                ),
                sample,
            )
            recorded_reward = row.get("reward")
            if not isinstance(recorded_reward, dict) or recorded_reward != rescored:
                raise RuntimeError(
                    f"online/offline full reward mismatch for sample {sample_index}"
                )
            if any(row.get(key) != value for key, value in rescored.items()):
                raise RuntimeError(
                    f"flattened reward fields drifted for sample {sample_index}"
                )
            score = row.get("score")
            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(float(score))
                or float(score) not in {0.0, 1.0}
                or float(rescored["score"]) != float(score)
            ):
                raise RuntimeError(
                    f"invalid binary reward for sample {sample_index}"
                )
            rollout_rewards.append(float(score))
            reward_agreements += 1
            total_rows += 1
            total_model_tokens += int(metadata["model_token_count"])
            total_env_tokens += int(metadata["env_token_count"])
            total_env_calls += int(metadata.get("n_env_calls", 0))
            rollout_model_tokens += int(metadata["model_token_count"])
            rollout_env_tokens += int(metadata["env_token_count"])
            rollout_env_calls += int(metadata.get("n_env_calls", 0))
        expected_groups = set(range(rollout_id * 256, (rollout_id + 1) * 256))
        if set(group_counts) != expected_groups or set(group_counts.values()) != {8}:
            raise RuntimeError(
                f"precision rollout {rollout_id} lacks exactly 256 groups x 8 samples"
            )
        grouped_rewards = [
            rollout_rewards[index : index + 8]
            for index in range(0, len(rollout_rewards), 8)
        ]
        all_zero_percentage = sum(
            all(reward == 0.0 for reward in group)
            for group in grouped_rewards
        ) / 256
        if rollout_env_tokens <= 0 or rollout_env_calls <= 0:
            raise RuntimeError(
                f"precision rollout {rollout_id} lacks nonvacuous environment-token evidence"
            )
        per_rollout[str(rollout_id)] = {
            "reward_mean": sum(rollout_rewards) / len(rollout_rewards),
            "all_zero_percentage": all_zero_percentage,
            "pass_at_1": sum(rollout_rewards) / len(rollout_rewards),
            "pass_at_8": sum(any(group) for group in grouped_rewards) / 256,
            "groups": 256,
            "samples_per_group": 8,
            "model_tokens": rollout_model_tokens,
            "env_tokens": rollout_env_tokens,
            "env_calls": rollout_env_calls,
        }
    if total_rows != 4_096 or total_env_tokens <= 0 or total_env_calls <= 0:
        raise RuntimeError(
            "precision rollout evidence is vacuous or incomplete: "
            f"rows={total_rows} env_tokens={total_env_tokens} env_calls={total_env_calls}"
        )
    return {
        "rows": total_rows,
        "model_tokens": total_model_tokens,
        "env_tokens": total_env_tokens,
        "env_calls": total_env_calls,
        "reward_agreements": reward_agreements,
        "per_rollout": per_rollout,
        "artifacts": artifacts,
        "exactly_one_bos_per_row": True,
        "prompt_mask_all_zero": True,
    }


def _validate_precision_wandb_history(
    contract: dict[str, object],
    *,
    rollout_evidence: dict[str, object],
    poll_attempts: int = 24,
    poll_seconds: float = 5.0,
) -> dict[str, object]:
    import wandb

    if poll_attempts <= 0 or poll_seconds < 0:
        raise ValueError("invalid W&B precision-gate polling budget")
    wandb_contract = dict(contract["wandb"])
    path = (
        f"{wandb_contract['entity']}/{wandb_contract['project']}/"
        f"{wandb_contract['run_id']}"
    )
    train_keys = [
        "train/step",
        "train/loss",
        "train/ppo_kl",
        "train/entropy_loss",
        "train/grad_norm",
    ]
    rollout_metrics = (
        "rollout/entropy",
        "rollout/zero_std/all_zero_percentage",
        "passrate/pass@1",
        "passrate/pass@8",
    )

    def finite_scalar(value: object) -> bool:
        return (
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
        )

    last_error: Exception | None = None
    for attempt in range(poll_attempts):
        try:
            run = wandb.Api(timeout=120).run(path)
            train_rows = list(
                run.scan_history(keys=train_keys, page_size=1000)
            )
            accepted_train: dict[str, dict[str, float]] = {}
            for step in (0, 1):
                candidates = []
                for row in train_rows:
                    if row.get("train/step") != step:
                        continue
                    values = {key: row.get(key) for key in train_keys[1:]}
                    if all(finite_scalar(value) for value in values.values()):
                        candidates.append(
                            {key: float(value) for key, value in values.items()}
                        )
                if len(candidates) != 1:
                    raise RuntimeError(
                        f"W&B requires one finite train event at step {step}; found {len(candidates)}"
                    )
                accepted_train[str(step)] = candidates[0]

            accepted_rollout: dict[str, dict[str, float]] = {
                "0": {},
                "1": {},
            }
            for metric in rollout_metrics:
                rows = list(
                    run.scan_history(
                        keys=["rollout/step", metric],
                        page_size=1000,
                    )
                )
                for step in (0, 1):
                    values = [
                        float(row[metric])
                        for row in rows
                        if row.get("rollout/step") == step
                        and finite_scalar(row.get(metric))
                    ]
                    if len(values) != 1:
                        raise RuntimeError(
                            f"W&B requires one finite value for {metric} at rollout {step}; "
                            f"found values={values}"
                        )
                    accepted_rollout[str(step)][metric] = values[0]

            for step in (0, 1):
                expected = dict(
                    dict(rollout_evidence["per_rollout"])[str(step)]
                )
                observed = accepted_rollout[str(step)]
                expected_matches = {
                    "rollout/zero_std/all_zero_percentage": expected[
                        "all_zero_percentage"
                    ],
                    "passrate/pass@1": expected["pass_at_1"],
                    "passrate/pass@8": expected["pass_at_8"],
                }
                if any(
                    not math.isclose(
                        observed[key],
                        float(value),
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                    for key, value in expected_matches.items()
                ):
                    raise RuntimeError(
                        f"W&B rollout outcomes disagree with authenticated rows at step {step}"
                    )
            return {
                "path": path,
                "url": getattr(run, "url", None),
                "train_steps": accepted_train,
                "rollout_steps": accepted_rollout,
            }
        except Exception as exc:
            last_error = exc
            if attempt + 1 < poll_attempts:
                time.sleep(poll_seconds)
    raise RuntimeError(
        f"W&B precision-gate evidence did not converge after {poll_attempts} polls for {path}: "
        f"{type(last_error).__name__}: {last_error}"
    ) from last_error


@app.function(
    gpu=f"{base.GPU_TYPE}:{base.GPUS_PER_NODE}",
    cpu=base.CPU_COUNT,
    memory=SMALL_MODEL_HOST_MEMORY_MB,
    timeout=60 * 60 * base.TIMEOUT_HOURS,
    volumes={
        "/data": base.data_vol,
        "/rl-checkpoints": base.ckpt_vol,
        PRETRAIN_CKPT_ROOT: pretrain_ckpt_vol,
        base.HF_CACHE_DIR: base.hf_cache,
    },
)
def precision_resume_gate_leg(
    *,
    leg: int,
    hf_checkpoint: str,
    expected_contract_sha256: str = "",
    model_id: str = CONTEXT2048_MODEL_ID,
    train_file: str = BALANCED_TRAIN_FILE,
    train_file_sha256: str = BALANCED_TRAIN_SHA256,
    rollout_seed: int = 42,
    wandb_project: str = "chess_interleave_50m",
    max_tokens_per_gpu: int = SMALL_MODEL_MAX_TOKENS_DEFAULT,
    sglang_server_concurrency: int = SGLANG_SERVER_CONCURRENCY_DEFAULT,
    lr: str = "1e-5",
    kl_loss_type: str = KL_LOSS_TYPE_DEFAULT,
    rollout_max_prompt_len: int = 512,
    rollout_max_response_len: int = 1_536,
    rollout_max_context_len: int = 2_048,
    initial_adam_checkpoint: str = "",
    initial_adam_completion_sha256: str = "",
    initial_adam_source_tree_sha256: str = "",
    initial_adam_step: int = 0,
) -> dict[str, object]:
    """Run exactly one gate leg; leg 2 is a new Modal FunctionCall."""

    if leg not in {1, 2}:
        raise ValueError("precision gate leg must be 1 or 2")
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("named Modal secret lacks WANDB_API_KEY")
    base.data_vol.reload()
    base.ckpt_vol.reload()
    pretrain_ckpt_vol.reload()
    checkpoint, contract = _precision_gate_contract_from_inputs(
        hf_checkpoint=hf_checkpoint,
        model_id=model_id,
        train_file=train_file,
        train_file_sha256=train_file_sha256,
        rollout_seed=rollout_seed,
        wandb_project=wandb_project,
        max_tokens_per_gpu=max_tokens_per_gpu,
        sglang_server_concurrency=sglang_server_concurrency,
        lr=lr,
        kl_loss_type=kl_loss_type,
        rollout_max_prompt_len=rollout_max_prompt_len,
        rollout_max_response_len=rollout_max_response_len,
        rollout_max_context_len=rollout_max_context_len,
        initial_adam_checkpoint=initial_adam_checkpoint,
        initial_adam_completion_sha256=initial_adam_completion_sha256,
        initial_adam_source_tree_sha256=initial_adam_source_tree_sha256,
        initial_adam_step=initial_adam_step,
    )
    contract_sha256 = str(contract["contract_sha256"])
    if expected_contract_sha256 and expected_contract_sha256 != contract_sha256:
        raise RuntimeError(
            "precision gate contract changed between independent FunctionCalls"
        )
    if leg == 2 and not expected_contract_sha256:
        raise ValueError("leg 2 requires the contract SHA256 returned by leg 1")
    run_root, gate_root = _precision_gate_paths(
        contract_sha256,
        run_name=str(contract["run_name"]),
    )
    contract_path = run_root / "precision_gate" / "CONTRACT.json"
    if leg == 1:
        if run_root.exists() or gate_root.exists():
            raise FileExistsError(
                f"precision gate identity already exists: {contract_sha256}"
            )
        run_root.mkdir(parents=True, exist_ok=False)
        _atomic_json(contract_path, contract)
    else:
        observed_contract = json.loads(contract_path.read_text())
        if observed_contract != contract:
            raise RuntimeError("precision gate contract changed before resume")
        if _reconcile_modal_checkpoint_root(run_root) != 1:
            raise RuntimeError("precision gate leg 2 requires committed checkpoint 1")

    command = list(dict(contract["commands"])[f"leg_{leg}"])
    write_run_provenance(
        run_root=run_root,
        identity={
            "kind": "chess_rl_miles_precision_resume_gate",
            "contract": contract,
        },
        command=command,
    )
    env = _runtime_env(
        run_name=str(contract["run_name"]),
        deterministic_seed_mode="sample-index",
        precision_gate_leg=leg,
    )
    base._cleanup_runtime()
    try:
        base._start_ray_head(env, cpu_threads=int(base.CPU_COUNT))
        env["RAY_ADDRESS"] = base.RAY_ADDRESS
        returncode = subprocess.call(
            command,
            env=env,
            cwd=base.PROJECT_DIR,
        )
    finally:
        base._cleanup_runtime()
    if returncode:
        raise RuntimeError(
            f"precision resume gate leg {leg} failed: exit {returncode}"
        )
    if _reconcile_modal_checkpoint_root(run_root) != leg:
        raise RuntimeError(
            f"precision gate leg {leg} did not produce exactly checkpoint {leg}"
        )
    checkpoint_evidence = _validate_precision_checkpoint(
        run_root,
        step=leg,
    )
    result_core = {
        "schema": "chess-rl-miles-precision-resume-gate-leg-v1",
        "leg": leg,
        "contract_sha256": contract_sha256,
        "run_root": str(run_root),
        "checkpoint": checkpoint_evidence,
    }
    result = _self_hashed_payload(result_core, hash_key="evidence_sha256")
    _atomic_json(
        run_root / "precision_gate" / f"LEG_{leg}_PASSED.json",
        result,
    )
    # Publish only after the child exits successfully and the checkpoint has
    # been authenticated. A failed child cannot publish its local Volume view.
    base.ckpt_vol.commit()
    return {**result, "contract": contract}


@app.function(
    gpu=f"{base.GPU_TYPE}:1",
    cpu=16.0,
    memory=64 * 1024,
    timeout=60 * 60 * 4,
    retries=modal.Retries(
        initial_delay=PRECISION_FINALIZER_RETRY_DELAY_SECONDS,
        max_retries=PRECISION_FINALIZER_MAX_RETRIES,
    ),
    volumes={
        "/data": base.data_vol,
        "/rl-checkpoints": base.ckpt_vol,
        PRETRAIN_CKPT_ROOT: pretrain_ckpt_vol,
        base.HF_CACHE_DIR: base.hf_cache,
    },
)
def finalize_precision_resume_gate(
    *,
    hf_checkpoint: str,
    expected_contract_sha256: str,
    model_id: str = CONTEXT2048_MODEL_ID,
    train_file: str = BALANCED_TRAIN_FILE,
    train_file_sha256: str = BALANCED_TRAIN_SHA256,
    rollout_seed: int = 42,
    wandb_project: str = "chess_interleave_50m",
    max_tokens_per_gpu: int = SMALL_MODEL_MAX_TOKENS_DEFAULT,
    sglang_server_concurrency: int = SGLANG_SERVER_CONCURRENCY_DEFAULT,
    lr: str = "1e-5",
    kl_loss_type: str = KL_LOSS_TYPE_DEFAULT,
    rollout_max_prompt_len: int = 512,
    rollout_max_response_len: int = 1_536,
    rollout_max_context_len: int = 2_048,
    initial_adam_checkpoint: str = "",
    initial_adam_completion_sha256: str = "",
    initial_adam_source_tree_sha256: str = "",
    initial_adam_step: int = 0,
) -> dict[str, object]:
    """Independently authenticate both calls, W&B, masks, and HF export."""

    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("named Modal secret lacks WANDB_API_KEY")
    base.data_vol.reload()
    base.ckpt_vol.reload()
    pretrain_ckpt_vol.reload()
    origin, contract = _precision_gate_contract_from_inputs(
        hf_checkpoint=hf_checkpoint,
        model_id=model_id,
        train_file=train_file,
        train_file_sha256=train_file_sha256,
        rollout_seed=rollout_seed,
        wandb_project=wandb_project,
        max_tokens_per_gpu=max_tokens_per_gpu,
        sglang_server_concurrency=sglang_server_concurrency,
        lr=lr,
        kl_loss_type=kl_loss_type,
        rollout_max_prompt_len=rollout_max_prompt_len,
        rollout_max_response_len=rollout_max_response_len,
        rollout_max_context_len=rollout_max_context_len,
        initial_adam_checkpoint=initial_adam_checkpoint,
        initial_adam_completion_sha256=initial_adam_completion_sha256,
        initial_adam_source_tree_sha256=initial_adam_source_tree_sha256,
        initial_adam_step=initial_adam_step,
    )
    if contract["contract_sha256"] != expected_contract_sha256:
        raise RuntimeError("precision gate contract changed before finalization")
    run_root, gate_root = _precision_gate_paths(
        expected_contract_sha256,
        run_name=str(contract["run_name"]),
    )
    passed_path = gate_root / "PASSED.json"
    reused = _reuse_authenticated_precision_gate_result(
        gate_root=gate_root,
        contract=contract,
    )
    if reused is not None:
        # A retry after publication avoids mutable W&B queries and CUDA/export
        # work, but only after the complete immutable result authenticates.
        return reused
    if _reconcile_modal_checkpoint_root(run_root) != 2:
        raise RuntimeError("precision gate finalizer requires committed checkpoint 2")
    for leg in (1, 2):
        leg_evidence = _load_self_hashed_json(
            run_root / "precision_gate" / f"LEG_{leg}_PASSED.json",
            hash_key="evidence_sha256",
        )
        if (
            leg_evidence.get("leg") != leg
            or leg_evidence.get("contract_sha256")
            != expected_contract_sha256
        ):
            raise RuntimeError(f"precision gate leg {leg} evidence drifted")

    checkpoints = {
        str(step): _validate_precision_checkpoint(run_root, step=step)
        for step in (1, 2)
    }
    initial_optimizer_state = dict(contract["initial_optimizer_state"])
    if initial_optimizer_state["mode"] == "fresh_adam_state":
        if any(
            checkpoints[str(step)]["initial_adam_import"] is not None
            or checkpoints[str(step)]["initial_adam_step_progression"] is not None
            for step in (1, 2)
        ):
            raise RuntimeError(
                "fresh-Adam precision gate unexpectedly contains imported state"
            )
    else:
        expected_fields = {
            "checkpoint": initial_optimizer_state["checkpoint"],
            "completion_sha256": initial_optimizer_state["completion_sha256"],
            "source_tree_sha256": initial_optimizer_state["source_tree_sha256"],
            "source_step": initial_optimizer_state["source_step"],
            "mapping_rule": "interleaved-hf-decay-then-bias-norm-v1",
            "round_trip_full_state_verified": True,
        }
        import_records = []
        for step in (1, 2):
            record = checkpoints[str(step)]["initial_adam_import"]
            progression = checkpoints[str(step)][
                "initial_adam_step_progression"
            ]
            if not isinstance(record, Mapping) or any(
                record.get(key) != value
                for key, value in expected_fields.items()
            ):
                raise RuntimeError(
                    f"precision checkpoint {step} lacks exact initial Adam import evidence"
                )
            if progression != {
                "source_step": initial_optimizer_state["source_step"],
                "rl_global_step": step,
                "expected_adam_step": (
                    int(initial_optimizer_state["source_step"]) + step
                ),
                "parameter_count": record["parameter_count"],
                "all_parameter_steps_verified": True,
            }:
                raise RuntimeError(
                    f"precision checkpoint {step} Adam step progression drifted"
                )
            import_records.append(record)
        if import_records[0] != import_records[1]:
            raise RuntimeError(
                "initial Adam import identity changed across process-boundary resume"
            )
    rollout_evidence = _validate_precision_rollout_evidence(
        run_root,
        rollout_seed=rollout_seed,
        max_prompt_len=rollout_max_prompt_len,
        max_context_len=rollout_max_context_len,
    )
    runtime_evidence = _validate_precision_runtime_evidence(
        run_root,
        contract=contract,
        rollout_evidence=rollout_evidence,
    )
    wandb_evidence = _validate_precision_wandb_history(
        contract,
        rollout_evidence=rollout_evidence,
    )

    if str(Path(base.MILES_DIR)) not in sys.path:
        sys.path.insert(0, str(Path(base.MILES_DIR)))
    from tools.convert_fsdp_to_hf import (
        convert_atomically,
        inspect_committed_dcp_fp32,
        validate_bf16_cuda_forward,
        validate_committed_hf_export,
        validate_committed_source,
    )

    source = run_root / "iter_0000002"
    dcp_export_source = inspect_committed_dcp_fp32(source)
    export = (
        Path(HF_EXPORT_ROOT)
        / f"precision_resume_{expected_contract_sha256[:20]}_step2_fp32"
    )
    authenticated_source = validate_committed_source(source)
    source_marker = dict(authenticated_source["marker"])
    expected_export_source = {
        "iteration": source_marker["iteration"],
        "commit_sha256": source_marker["commit_sha256"],
        "marker_sha256": authenticated_source["marker_sha256"],
        "payload_sha256": _canonical_json_sha256(source_marker["payload"]),
    }
    if export.exists() or export.is_symlink():
        export_marker = validate_committed_hf_export(export)
        if export_marker.get("source_checkpoint") != expected_export_source:
            raise RuntimeError(
                "existing immutable precision-gate export belongs to a different checkpoint"
            )
        conversion = {
            "precision": export_marker["precision"],
            "source_checkpoint": expected_export_source,
            "export_commit_sha256": export_marker["commit_sha256"],
        }
    else:
        conversion = convert_atomically(
            origin,
            source,
            export,
            force=False,
        )
    export_marker = validate_committed_hf_export(export)
    if export_marker.get("source_checkpoint") != expected_export_source:
        raise RuntimeError("precision-gate export source identity drifted")
    bf16_forward = validate_bf16_cuda_forward(
        export,
        sequence_length=4,
    )
    pretrain_ckpt_vol.commit()

    evidence_paths = [
        path
        for path in sorted((run_root / "precision_gate").rglob("*"))
        if path.is_file() and path.name != "PASSED.json"
    ]
    evidence_files = [
        {
            "absolute_path": str(path),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in evidence_paths
    ]
    evidence_files.extend(list(rollout_evidence["artifacts"]))
    success_core = {
        "schema": "chess-rl-miles-precision-resume-gate-success-v1",
        "version": PRECISION_RESUME_GATE_VERSION,
        "contract_sha256": expected_contract_sha256,
        "source_sha256": contract["source_sha256"],
        "passed": True,
        "checkpoints": checkpoints,
        "runtime_evidence": runtime_evidence,
        "rollout_evidence": rollout_evidence,
        "wandb": wandb_evidence,
        "export": {
            "path": str(export),
            "dcp_source": dcp_export_source,
            "conversion": conversion,
            "marker_commit_sha256": export_marker["commit_sha256"],
            "bf16_cuda_forward": bf16_forward,
        },
        "evidence_files": evidence_files,
    }
    success = _self_hashed_payload(
        success_core,
        hash_key="success_sha256",
    )
    _publish_precision_gate_result(
        gate_root,
        contract=contract,
        success=success,
    )
    base.ckpt_vol.commit()
    return {**success, "passed_path": str(passed_path)}


def _v2r4_gate_run_name(candidate_step: int, batch_label: str) -> str:
    return (
        f"v2r4a-gate-w190-s{candidate_step}-"
        f"batch-{batch_label.lower()}"
    )


def _v2r4_contract_static() -> dict[str, object]:
    cells = [
        {
            "candidate_step": candidate_step,
            "batch_label": batch_label,
            "run_name": _v2r4_gate_run_name(candidate_step, batch_label),
        }
        for candidate_step in sorted(V2R4_GATE_CANDIDATES)
        for batch_label in sorted(V2R4_GATE_BATCHES)
    ]
    return {
        "schema": V2R4_GATE_CONTRACT_SCHEMA,
        "version": V2R4_GATE_VERSION,
        "model_id": MODEL_ID,
        "cells": cells,
        "candidates": {
            str(step): dict(value)
            for step, value in sorted(V2R4_GATE_CANDIDATES.items())
        },
        "prompt_batches": {
            label: dict(value)
            for label, value in sorted(V2R4_GATE_BATCHES.items())
        },
        "prompt_manifest": {
            "path": V2R4_GATE_PROMPT_MANIFEST,
            "manifest_sha256": V2R4_GATE_PROMPT_MANIFEST_SHA256,
            "file_sha256": V2R4_GATE_PROMPT_MANIFEST_FILE_SHA256,
        },
        "semantics": dict(V2R4_GATE_SEMANTICS),
        "runtime": {
            "miles_image": base.MILES_IMAGE,
            "gpu_type": base.GPU_TYPE,
            "gpus_per_node": base.GPUS_PER_NODE,
            "host_memory_gb": SMALL_MODEL_HOST_MEMORY_GB,
            "max_tokens_per_gpu": SMALL_MODEL_MAX_TOKENS_DEFAULT,
            "sglang_server_concurrency": (
                SGLANG_SERVER_CONCURRENCY_DEFAULT
            ),
        },
    }


def _require_frozen_v2r4_contract_digest(requested_sha256: str) -> None:
    if requested_sha256 != V2R4_EXPECTED_CONTRACT_SHA256:
        raise ValueError(
            "v2r4 launch requires the exact frozen contract SHA256"
        )
    if not re.fullmatch(r"[0-9a-f]{64}", requested_sha256):
        raise RuntimeError("v2r4 contract binding is not frozen")


def _load_v2r4_gate_contract(
    requested_sha256: str,
) -> dict[str, object]:
    _require_frozen_v2r4_contract_digest(requested_sha256)
    if not re.fullmatch(
        r"[0-9a-f]{64}", V2R4_EXPECTED_CONTRACT_FILE_SHA256
    ):
        raise RuntimeError("v2r4 contract file binding is not frozen")

    path = Path(V2R4_GATE_CONTRACT_MANIFEST)
    if not path.is_file():
        raise FileNotFoundError(path)
    if _sha256(path) != V2R4_EXPECTED_CONTRACT_FILE_SHA256:
        raise ValueError("v2r4 runtime-contract file SHA256 drifted")
    contract = json.loads(path.read_text())
    if not isinstance(contract, dict):
        raise ValueError("v2r4 runtime contract is not an object")
    embedded_sha256 = contract.pop("contract_sha256", None)
    if (
        embedded_sha256 != V2R4_EXPECTED_CONTRACT_SHA256
        or _canonical_json_sha256(contract)
        != V2R4_EXPECTED_CONTRACT_SHA256
    ):
        raise ValueError("v2r4 runtime-contract self-hash drifted")

    static = _v2r4_contract_static()
    for key, expected in static.items():
        if contract.get(key) != expected:
            raise ValueError(
                f"v2r4 runtime-contract field drifted: {key}"
            )

    sources = contract.get("sources")
    if not isinstance(sources, dict):
        raise ValueError("v2r4 runtime contract lacks source identities")
    actual_project = _normalized_source_identity(
        Path(base.PROJECT_DIR),
        excluded_relatives=(V2R4_GATE_BINDING_RELATIVE_PATH,),
    )
    actual_miles = _normalized_source_identity(Path(base.MILES_DIR))
    if sources.get("chess_rl_miles") != actual_project:
        raise ValueError("v2r4 chess-rl-miles source identity drifted")
    if sources.get("miles") != actual_miles:
        raise ValueError("v2r4 Miles source identity drifted")

    plan = contract.get("plan")
    endpoint_evaluators = contract.get("endpoint_evaluators")
    if (
        not isinstance(plan, dict)
        or not re.fullmatch(r"[0-9a-f]{64}", str(plan.get("sha256", "")))
        or not isinstance(endpoint_evaluators, dict)
        or set(endpoint_evaluators) != {"pt_b1_b5", "p2_sft_at_p1"}
        or any(
            not re.fullmatch(r"[0-9a-f]{64}", str(value))
            for value in endpoint_evaluators.values()
        )
    ):
        raise ValueError(
            "v2r4 runtime contract lacks frozen plan/evaluator identities"
        )
    return {
        **contract,
        "contract_sha256": embedded_sha256,
        "contract_file_sha256": V2R4_EXPECTED_CONTRACT_FILE_SHA256,
        "contract_path": str(path),
    }


def _v2r4_prompt_fingerprint(row: dict[str, object]) -> str:
    return _canonical_json_sha256(
        {
            "input": str(row.get("input") or ""),
            "FEN": str(row.get("FEN") or ""),
            "PuzzleId": str(row.get("PuzzleId") or ""),
            "ground_truth": str(row.get("ground_truth") or ""),
        }
    )


def _validate_v2r4_gate_artifacts(
    *,
    run_root: Path,
    batch_manifest: dict[str, object],
    rollout_seed: int,
) -> list[dict[str, object]]:
    """Authenticate exact prompt/order/sampling shape without reading rewards."""

    quarters = batch_manifest.get("rollout_quarters")
    if not isinstance(quarters, list) or len(quarters) != 4:
        raise ValueError("v2r4 prompt manifest must contain four quarters")
    records: list[dict[str, object]] = []
    observed_global_groups: set[int] = set()
    observed_global_samples: set[int] = set()
    for rollout_id, quarter in enumerate(quarters):
        if not isinstance(quarter, dict):
            raise ValueError("v2r4 rollout quarter is not an object")
        expected_prompts = quarter.get("ordered_prompt_fingerprints")
        if (
            not isinstance(expected_prompts, list)
            or len(expected_prompts) != 256
            or any(
                not isinstance(item, str) or len(item) != 64
                for item in expected_prompts
            )
        ):
            raise ValueError("v2r4 rollout quarter prompt inventory drifted")
        path = (
            run_root
            / "rollouts"
            / "training"
            / f"rollout_{rollout_id}.jsonl"
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        temporary = path.with_suffix(path.suffix + ".tmp")
        if temporary.exists():
            raise RuntimeError(f"partial atomic rollout artifact remains: {temporary}")
        rows: list[dict[str, object]] = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid rollout JSON at {path}:{line_number}"
                    ) from exc
                if not isinstance(value, dict):
                    raise ValueError(
                        f"rollout row is not an object at {path}:{line_number}"
                    )
                rows.append(value)
        if len(rows) != 2_048:
            raise ValueError(
                f"{path} must contain exactly 2,048 rows, got {len(rows)}"
            )

        observed_prompts: list[str] = []
        for local_group in range(256):
            group_index = rollout_id * 256 + local_group
            group_rows = rows[local_group * 8 : (local_group + 1) * 8]
            fingerprints = {
                _v2r4_prompt_fingerprint(row) for row in group_rows
            }
            if len(fingerprints) != 1:
                raise ValueError(
                    f"prompt identity changed inside group {group_index}"
                )
            observed_prompts.append(next(iter(fingerprints)))
            for sibling_index, row in enumerate(group_rows):
                sample_index = group_index * 8 + sibling_index
                if row.get("rollout_id") != rollout_id:
                    raise ValueError("rollout_id identity drifted")
                if row.get("group_index") != group_index:
                    raise ValueError("global group_index identity drifted")
                if row.get("sample_index") != sample_index:
                    raise ValueError("global sample_index identity drifted")
                if row.get("sampling_seed_sibling_index") != sibling_index:
                    raise ValueError("sampling sibling identity drifted")
                if row.get("sampling_seed") != rollout_seed + sample_index:
                    raise ValueError("sample-index deterministic seed drifted")
                metadata = row.get("metadata")
                if not isinstance(metadata, dict):
                    raise ValueError("rollout row lacks metadata")
                if metadata.get("sampling_seed") != rollout_seed + sample_index:
                    raise ValueError("metadata sampling seed drifted")
                if (
                    metadata.get("sampling_seed_sibling_index")
                    != sibling_index
                ):
                    raise ValueError("metadata sibling identity drifted")
                if metadata.get("sampling_seed_mode") != "sample-index":
                    raise ValueError("sampling seed mode drifted")
                if row.get("status") not in {"completed", "truncated"}:
                    raise ValueError("rollout row has a disallowed status")
                observed_global_groups.add(group_index)
                observed_global_samples.add(sample_index)
        if observed_prompts != expected_prompts:
            raise ValueError(
                f"rollout_{rollout_id} prompt order differs from manifest"
            )
        records.append(
            {
                "rollout_id": rollout_id,
                "path": str(path),
                "rows": len(rows),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
                "prompt_order_sha256": _canonical_json_sha256(
                    observed_prompts
                ),
            }
        )
    if observed_global_groups != set(range(1_024)):
        raise ValueError("v2r4 global prompt-group inventory is incomplete")
    if observed_global_samples != set(range(8_192)):
        raise ValueError("v2r4 global sample inventory is incomplete")
    return records


@app.function(
    cpu=2.0,
    memory=8 * 1024,
    timeout=10 * 60,
    volumes={
        "/data": base.data_vol,
        "/rl-checkpoints": base.ckpt_vol,
    },
)
def v2r4_gate_preflight(contract_sha256: str) -> dict[str, object]:
    """Prove the exact contract and all six canonical roots before spawning."""

    base.data_vol.reload()
    base.ckpt_vol.reload()
    contract = _load_v2r4_gate_contract(contract_sha256)
    run_roots = [
        str(Path(RAW_RL_ROOT) / str(cell["run_name"]))
        for cell in contract["cells"]
    ]
    existing = [path for path in run_roots if Path(path).exists()]
    if existing:
        raise FileExistsError(
            "v2r4 preflight found canonical run roots: "
            + ", ".join(existing)
        )
    ray_worker_environment = _verify_ray_worker_gate_environment()
    return {
        "schema": "interleaved-v2r4-gate-preflight-v1",
        "version": V2R4_GATE_VERSION,
        "contract_sha256": contract_sha256,
        "contract_file_sha256": contract["contract_file_sha256"],
        "all_six_canonical_roots_absent": True,
        "ray_worker_environment": ray_worker_environment,
        "run_roots": run_roots,
    }


@app.function(
    gpu=f"{base.GPU_TYPE}:{base.GPUS_PER_NODE}",
    cpu=base.CPU_COUNT,
    memory=SMALL_MODEL_HOST_MEMORY_MB,
    timeout=60 * 60 * base.TIMEOUT_HOURS,
    volumes={
        "/data": base.data_vol,
        "/rl-checkpoints": base.ckpt_vol,
        PRETRAIN_CKPT_ROOT: pretrain_ckpt_vol,
        base.HF_CACHE_DIR: base.hf_cache,
    },
)
def v2r4_gate_rollout(
    *,
    candidate_step: int,
    batch_label: str,
    contract_sha256: str,
) -> dict[str, object]:
    """Run one exact 1,024x8 prompt-disjoint rollout-gate cell."""

    if candidate_step not in V2R4_GATE_CANDIDATES:
        raise ValueError("candidate_step is outside the frozen v2r4 grid")
    normalized_batch = str(batch_label).strip().upper()
    if normalized_batch not in V2R4_GATE_BATCHES:
        raise ValueError("batch_label must be A or B")
    base.data_vol.reload()
    base.ckpt_vol.reload()
    pretrain_ckpt_vol.reload()
    contract = _load_v2r4_gate_contract(contract_sha256)

    candidate = dict(V2R4_GATE_CANDIDATES[candidate_step])
    batch = dict(V2R4_GATE_BATCHES[normalized_batch])
    run_name = _v2r4_gate_run_name(candidate_step, normalized_batch)
    authorized_cells = [
        cell
        for cell in contract["cells"]
        if (
            cell["candidate_step"] == candidate_step
            and cell["batch_label"] == normalized_batch
            and cell["run_name"] == run_name
        )
    ]
    if len(authorized_cells) != 1:
        raise ValueError("v2r4 runtime contract does not authorize this cell")

    checkpoint = _validate_hf_checkpoint(str(candidate["hf_path"]))
    checkpoint_identity = directory_identity(
        checkpoint, logical_path=str(checkpoint)
    )
    if (
        checkpoint_identity["manifest_sha256"]
        != candidate["hf_directory_manifest_sha256"]
    ):
        raise ValueError("v2r4 candidate HF directory identity drifted")

    prompt_path = Path(str(batch["path"]))
    if not prompt_path.is_file():
        raise FileNotFoundError(prompt_path)
    if _sha256(prompt_path) != batch["sha256"]:
        raise ValueError("v2r4 prompt parquet SHA256 drifted")
    import pyarrow.parquet as pq

    if pq.ParquetFile(prompt_path).metadata.num_rows != batch["rows"]:
        raise ValueError("v2r4 prompt parquet row count drifted")

    prompt_manifest_path = Path(V2R4_GATE_PROMPT_MANIFEST)
    if (
        not prompt_manifest_path.is_file()
        or _sha256(prompt_manifest_path)
        != V2R4_GATE_PROMPT_MANIFEST_FILE_SHA256
    ):
        raise ValueError("v2r4 prompt-batch manifest file drifted")
    prompt_manifest = json.loads(prompt_manifest_path.read_text())
    embedded_manifest_sha256 = prompt_manifest.pop(
        "manifest_sha256", None
    )
    if (
        embedded_manifest_sha256 != V2R4_GATE_PROMPT_MANIFEST_SHA256
        or _canonical_json_sha256(prompt_manifest)
        != V2R4_GATE_PROMPT_MANIFEST_SHA256
    ):
        raise ValueError("v2r4 prompt-batch manifest self-hash drifted")
    batch_manifest = prompt_manifest.get("batches", {}).get(
        normalized_batch
    )
    if not isinstance(batch_manifest, dict):
        raise ValueError("v2r4 prompt manifest lacks the requested batch")
    if (
        batch_manifest.get("file_sha256") != batch["sha256"]
        or batch_manifest.get("logical_path") != batch["path"]
        or batch_manifest.get("prompt_set_sha256")
        != batch["prompt_set_sha256"]
        or batch_manifest.get("epoch0_prompt_order_sha256")
        != batch["epoch0_prompt_order_sha256"]
        or batch_manifest.get("rollout_seed") != batch["rollout_seed"]
    ):
        raise ValueError("v2r4 prompt-batch manifest/constants disagree")

    run_root = Path(RAW_RL_ROOT) / run_name
    if run_root.exists():
        raise FileExistsError(
            "v2r4 canonical run root already exists; retries and duplicate "
            f"launches are forbidden: {run_root}"
        )
    command = build_train_command(
        hf_checkpoint=str(checkpoint),
        run_name=run_name,
        model_id=MODEL_ID,
        num_rollout=4,
        dynamic_filter=False,
        rollout_seed=int(batch["rollout_seed"]),
        save_interval=0,
        eval_interval=0,
        resume_path="",
        resume_step=0,
        wandb_project="chess_interleave_50m",
        wandb_group="v2r4a_production_gate",
        max_tokens_per_gpu=SMALL_MODEL_MAX_TOKENS_DEFAULT,
        sglang_server_concurrency=SGLANG_SERVER_CONCURRENCY_DEFAULT,
        deterministic_inference=True,
        rollout_only=True,
        canary=False,
        train_file=str(prompt_path),
        train_file_sha256=str(batch["sha256"]),
        data_source_path=STRICT_GATE_DATA_SOURCE_PATH,
        deterministic_seed_by_sample_index=True,
        fault_tolerance=False,
        rollout_health_check_interval=1e18,
    )
    run_root.mkdir(parents=True, exist_ok=False)
    intent = {
        "schema": "interleaved-v2r4-gate-cell-intent-v1",
        "version": V2R4_GATE_VERSION,
        "contract_sha256": contract_sha256,
        "contract_file_sha256": contract["contract_file_sha256"],
        "candidate_step": candidate_step,
        "batch_label": normalized_batch,
        "run_name": run_name,
        "command_sha256": hashlib.sha256(
            json.dumps(command, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    _atomic_json(run_root / "_V2R4_GATE_INTENT.json", intent)
    identity = {
        "kind": "chess_rl_miles_v2r4_production_gate_rollout",
        "version": V2R4_GATE_VERSION,
        "contract_sha256": contract_sha256,
        "contract_file_sha256": contract["contract_file_sha256"],
        "contract_path": contract["contract_path"],
        "authorized_cell": authorized_cells[0],
        "candidate": {
            "step": candidate_step,
            **candidate,
            "directory_identity": checkpoint_identity,
        },
        "prompt_batch": {
            "label": normalized_batch,
            **batch,
            "manifest_sha256": V2R4_GATE_PROMPT_MANIFEST_SHA256,
            "manifest_file_sha256": (
                V2R4_GATE_PROMPT_MANIFEST_FILE_SHA256
            ),
        },
        "semantics": dict(V2R4_GATE_SEMANTICS),
        "sources": {
            "chess_rl_miles": _normalized_source_identity(
                Path(base.PROJECT_DIR),
                excluded_relatives=(V2R4_GATE_BINDING_RELATIVE_PATH,),
            ),
            "miles": _normalized_source_identity(Path(base.MILES_DIR)),
        },
        "runtime": runtime_identity(image=base.MILES_IMAGE),
    }
    provenance = write_run_provenance(
        run_root=run_root,
        identity=identity,
        command=command,
    )
    base.ckpt_vol.commit()
    print("[v2r4-gate] " + " ".join(command), flush=True)

    env = _runtime_env(
        run_name=run_name,
        deterministic_seed_mode="sample-index",
    )
    base._cleanup_runtime()
    base._start_ray_head(env, cpu_threads=int(base.CPU_COUNT))
    env["RAY_ADDRESS"] = base.RAY_ADDRESS
    returncode = subprocess.call(
        command,
        env=env,
        cwd=base.PROJECT_DIR,
    )
    if returncode:
        base.ckpt_vol.commit()
        raise RuntimeError(
            f"v2r4 rollout gate failed for {run_name}: exit {returncode}"
        )

    artifact_records = _validate_v2r4_gate_artifacts(
        run_root=run_root,
        batch_manifest=batch_manifest,
        rollout_seed=int(batch["rollout_seed"]),
    )
    success_core = {
        "schema": "interleaved-v2r4-gate-cell-success-v1",
        "version": V2R4_GATE_VERSION,
        "contract_sha256": contract_sha256,
        "contract_file_sha256": contract["contract_file_sha256"],
        "run_name": run_name,
        "candidate_step": candidate_step,
        "batch_label": normalized_batch,
        "provenance": provenance,
        "prompt_batch_sha256": batch["sha256"],
        "prompt_set_sha256": batch["prompt_set_sha256"],
        "rollout_seed": batch["rollout_seed"],
        "artifact_records": artifact_records,
        "shape_authenticated": True,
        "reward_metrics_inspected": False,
    }
    success = {
        **success_core,
        "success_sha256": _canonical_json_sha256(success_core),
    }
    success_path = run_root / "_V2R4_GATE_SUCCESS.json"
    _atomic_json(success_path, success)
    base.ckpt_vol.commit()
    return {
        **success,
        "success_path": str(success_path),
    }


def _export_manifest(
    *,
    source: Path,
    origin_hf: Path,
    output: Path,
    run_name: str,
    step: int,
) -> dict[str, object]:
    files = []
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.name == "interleave_handoff_manifest.json":
            continue
        relative = str(path.relative_to(output))
        record: dict[str, object] = {
            "path": relative,
            "bytes": path.stat().st_size,
        }
        if (
            path.suffix in {".json", ".safetensors"}
            or path.name.startswith("tokenizer")
        ):
            record["sha256"] = _sha256(path)
        files.append(record)
    return {
        "schema_version": 1,
        "kind": "miles_fsdp_to_hf_interleave_handoff",
        "run_name": run_name,
        "step": step,
        "source": str(source),
        "origin_hf": str(origin_hf),
        "output": str(output),
        "files": files,
    }


@app.function(
    gpu=f"{base.GPU_TYPE}:1",
    cpu=8.0,
    memory=64 * 1024,
    timeout=60 * 60 * 2,
    volumes={
        "/rl-checkpoints": base.ckpt_vol,
        PRETRAIN_CKPT_ROOT: pretrain_ckpt_vol,
    },
)
def convert_rl_to_hf(
    run_name: str,
    origin_hf: str,
    output_name: str,
    step: int = 0,
    force: bool = False,
) -> dict[str, object]:
    """Convert one raw Miles checkpoint into a pretrain-consumable HF model."""
    _safe_component(run_name, name="run_name")
    _safe_component(output_name, name="output_name")
    if step < 0:
        raise ValueError("step must be non-negative")

    base.ckpt_vol.reload()
    pretrain_ckpt_vol.reload()
    origin = _validate_hf_checkpoint(origin_hf)
    run_root = Path(RAW_RL_ROOT) / run_name
    if step == 0:
        reconciled = _reconcile_modal_checkpoint_root(run_root)
        if reconciled is None:
            raise FileNotFoundError(
                f"No authenticated committed checkpoint under {run_root}"
            )
        step = reconciled
    source = run_root / f"iter_{step:07d}"
    if not (source / "model").is_dir():
        raise FileNotFoundError(f"Incomplete Miles checkpoint: {source}")

    output = Path(HF_EXPORT_ROOT) / output_name
    if force:
        print(
            "[convert] force is accepted for CLI compatibility but committed exports are immutable",
            flush=True,
        )
    output.parent.mkdir(parents=True, exist_ok=True)

    if str(Path(base.MILES_DIR)) not in sys.path:
        sys.path.insert(0, str(Path(base.MILES_DIR)))
    from tools.convert_fsdp_to_hf import (
        convert_atomically,
        inspect_committed_dcp_fp32,
        validate_bf16_cuda_forward,
        validate_committed_hf_export,
    )

    source_evidence = inspect_committed_dcp_fp32(source)
    conversion = convert_atomically(
        origin,
        source,
        output,
        force=force,
    )
    marker = validate_committed_hf_export(output)
    bf16_forward = validate_bf16_cuda_forward(
        output,
        sequence_length=4,
    )
    pretrain_ckpt_vol.commit()
    print(f"[convert] ready: {output}", flush=True)
    return {
        "schema": "chess-rl-miles-hf-export-result-v1",
        "run_name": run_name,
        "step": step,
        "source": source_evidence,
        "origin_hf": str(origin),
        "output": str(output),
        "conversion": conversion,
        "export_commit_sha256": marker["commit_sha256"],
        "bf16_cuda_forward": bf16_forward,
    }


def _refresh_wandb_modal_secret() -> Path:
    """Refresh the stable named secret without exposing its value."""

    dotenv = base.WORKSPACE_LOCAL.parent / ".env"
    if not dotenv.is_file():
        raise FileNotFoundError(
            f"repository W&B environment file is missing: {dotenv}"
        )
    names = {
        line.split("=", 1)[0].removeprefix("export ").strip()
        for line in dotenv.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#") and "=" in line
    }
    if "WANDB_API_KEY" not in names:
        raise RuntimeError(
            f"repository environment file lacks WANDB_API_KEY: {dotenv}"
        )
    result = subprocess.run(
        [
            "modal",
            "secret",
            "create",
            "wandb-interleave-pt-rl",
            "--from-dotenv",
            str(dotenv),
            "--force",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            "failed to refresh named Modal W&B secret (output withheld to protect secret material)"
        )
    return dotenv


@app.local_entrypoint()
def main(
    action: str = "train",
    hf_checkpoint: str = "",
    run_name: str = "",
    num_rollout: int = 1,
    dynamic_filter: bool = False,
    rollout_seed: int = 42,
    save_interval: int = 40,
    eval_interval: int = 0,
    model_id: str = CONTEXT2048_MODEL_ID,
    resume_if_available: bool = True,
    wandb_project: str = "chess_interleave_50m",
    wandb_group: str = "core",
    max_tokens_per_gpu: int = SMALL_MODEL_MAX_TOKENS_DEFAULT,
    sglang_server_concurrency: int = SGLANG_SERVER_CONCURRENCY_DEFAULT,
    deterministic_inference: bool = True,
    rollout_only: bool = False,
    canary: bool = False,
    output_name: str = "",
    step: int = 0,
    candidate_step: int = 0,
    batch_label: str = "",
    contract_sha256: str = "",
    launch_ledger_path: str = V2R4_GATE_LAUNCH_LEDGER,
    precision_gate_ledger_path: str = "PRECISION_RESUME_GATE_LAUNCH_LEDGER.json",
    recover_production_launch: bool = False,
    force: bool = False,
    wait: bool = False,
    dry_run: bool = False,
    train_file: str = BALANCED_TRAIN_FILE,
    train_file_sha256: str = BALANCED_TRAIN_SHA256,
    lr: str = "1e-5",
    kl_loss_type: str = KL_LOSS_TYPE_DEFAULT,
    rollout_max_prompt_len: int = 512,
    rollout_max_response_len: int = 1_536,
    rollout_max_context_len: int = 2_048,
    initial_adam_checkpoint: str = "",
    initial_adam_completion_sha256: str = "",
    initial_adam_source_tree_sha256: str = "",
    initial_adam_step: int = 0,
) -> None:
    if action == "precision-gate":
        # This is a fixed scientific gate, not a generic smoke test.  Pin the
        # native context before any remote allocation instead of inheriting
        # the production launcher's historical 3072-token defaults.
        model_id = CONTEXT2048_MODEL_ID
        rollout_max_prompt_len = 512
        rollout_max_response_len = 1_536
        rollout_max_context_len = 2_048
    if recover_production_launch and action != "train":
        raise ValueError(
            "--recover-production-launch is valid only for --action train"
        )
    if recover_production_launch and (canary or rollout_only):
        raise ValueError(
            "--recover-production-launch is valid only for production training"
        )
    if action in {"train", "precision-gate"} and not dry_run:
        _refresh_wandb_modal_secret()
    if action == "train" and not canary and not rollout_only:
        if model_id != CONTEXT2048_MODEL_ID or (
            rollout_max_prompt_len,
            rollout_max_response_len,
            rollout_max_context_len,
        ) != (512, 1_536, 2_048):
            raise ValueError(
                "production train requires the native-2048 contract before "
                "remote allocation: model=context2048_47m_qwen3, "
                "prompt=512, response=1536, context=2048"
            )
    if not dry_run:
        _require_matching_deployment()
    if action == "train":
        if not hf_checkpoint or not run_name:
            raise ValueError("train requires --hf-checkpoint and --run-name")
        effective_data_source_path, effective_fault_tolerance = (
            _training_source_contract(
                canary=canary,
                rollout_only=rollout_only,
            )
        )
        command = build_train_command(
            hf_checkpoint=hf_checkpoint,
            run_name=run_name,
            model_id=model_id,
            num_rollout=num_rollout,
            dynamic_filter=dynamic_filter,
            rollout_seed=rollout_seed,
            save_interval=save_interval,
            eval_interval=eval_interval,
            wandb_project=wandb_project,
            wandb_group=wandb_group,
            max_tokens_per_gpu=max_tokens_per_gpu,
            sglang_server_concurrency=sglang_server_concurrency,
            deterministic_inference=deterministic_inference,
            deterministic_seed_by_sample_index=deterministic_inference,
            data_source_path=effective_data_source_path,
            fault_tolerance=effective_fault_tolerance,
            rollout_only=rollout_only,
            canary=canary,
            train_file=train_file,
            train_file_sha256=train_file_sha256,
            lr=lr,
            kl_loss_type=kl_loss_type,
            rollout_max_prompt_len=rollout_max_prompt_len,
            rollout_max_response_len=rollout_max_response_len,
            rollout_max_context_len=rollout_max_context_len,
            initial_adam_checkpoint=initial_adam_checkpoint,
            initial_adam_completion_sha256=initial_adam_completion_sha256,
            initial_adam_source_tree_sha256=initial_adam_source_tree_sha256,
            initial_adam_step=initial_adam_step,
        )
        if dry_run:
            print(" ".join(command))
            return
        train_kwargs = {
            "hf_checkpoint": hf_checkpoint,
            "run_name": run_name,
            "num_rollout": num_rollout,
            "dynamic_filter": dynamic_filter,
            "rollout_seed": rollout_seed,
            "save_interval": save_interval,
            "eval_interval": eval_interval,
            "model_id": model_id,
            "resume_if_available": resume_if_available,
            "wandb_project": wandb_project,
            "wandb_group": wandb_group,
            "max_tokens_per_gpu": max_tokens_per_gpu,
            "sglang_server_concurrency": sglang_server_concurrency,
            "deterministic_inference": deterministic_inference,
            "train_file": train_file,
            "train_file_sha256": train_file_sha256,
            "lr": lr,
            "kl_loss_type": kl_loss_type,
            "rollout_max_prompt_len": rollout_max_prompt_len,
            "rollout_max_response_len": rollout_max_response_len,
            "rollout_max_context_len": rollout_max_context_len,
            "initial_adam_checkpoint": initial_adam_checkpoint,
            "initial_adam_completion_sha256": initial_adam_completion_sha256,
            "initial_adam_source_tree_sha256": initial_adam_source_tree_sha256,
            "initial_adam_step": initial_adam_step,
        }
        if not canary and not rollout_only:
            if recover_production_launch:
                recovery = _read_local_production_recovery_record(run_name)
                launch_token = str(recovery["launch_token"])
            else:
                launch_token = secrets.token_hex(32)
                _write_local_production_recovery_record(
                    run_name=run_name,
                    launch_token=launch_token,
                )
            dispatcher = _deployed_function("dispatch_production_train").spawn(
                **train_kwargs,
                production_launch_token=launch_token,
                recovery=recover_production_launch,
            )
            dispatch_result = dispatcher.get()
            if not isinstance(dispatch_result, Mapping):
                raise RuntimeError("production RL dispatcher returned invalid evidence")
            function_call_id = str(
                dispatch_result.get("function_call_id", "") or ""
            )
            if dispatch_result.get("outcome") == "authenticated_completion":
                print(
                    "PRODUCTION RL ALREADY COMPLETE: "
                    + json.dumps(dispatch_result, sort_keys=True),
                    flush=True,
                )
                return
            if not re.fullmatch(r"fc-[0-9A-Za-z]+", function_call_id):
                raise RuntimeError(
                    "production RL dispatcher did not return an authenticated "
                    f"worker FunctionCall: {dispatch_result}"
                )
            handle = modal.FunctionCall.from_id(function_call_id)
            print(
                f"DISPATCHED train: dispatcher={dispatcher.object_id} "
                f"worker={function_call_id} outcome={dispatch_result['outcome']}",
                flush=True,
            )
        else:
            handle = _deployed_function("train_hf").spawn(
                **train_kwargs,
                rollout_only=rollout_only,
                canary=canary,
            )
    elif action == "precision-gate":
        if not hf_checkpoint:
            raise ValueError("precision-gate requires --hf-checkpoint")
        if any((run_name, output_name, step, force)):
            raise ValueError(
                "precision-gate derives immutable names; do not pass run/output/step/force"
            )
        if dry_run:
            print(
                "precision-gate: fresh Modal FunctionCall for update 1, "
                "independent resume FunctionCall for update 2, then GPU finalizer"
            )
            return
        ledger_path = Path(precision_gate_ledger_path).expanduser().resolve()
        ledger: dict[str, object] = {
            "schema": "chess-rl-miles-precision-gate-launch-ledger-v1",
            "state": "launching_leg_1",
            "expected_training_call_count": 2,
            "calls": [],
        }
        _exclusive_json(
            ledger_path,
            _self_hashed_payload(ledger, hash_key="ledger_sha256"),
        )

        def persist_precision_ledger() -> None:
            _atomic_json(
                ledger_path,
                _self_hashed_payload(
                    {
                        key: value
                        for key, value in ledger.items()
                        if key != "ledger_sha256"
                    },
                    hash_key="ledger_sha256",
                ),
            )

        common = {
            "hf_checkpoint": hf_checkpoint,
            "model_id": model_id,
            "train_file": train_file,
            "train_file_sha256": train_file_sha256,
            "rollout_seed": rollout_seed,
            "wandb_project": wandb_project,
            "max_tokens_per_gpu": max_tokens_per_gpu,
            "sglang_server_concurrency": sglang_server_concurrency,
            "lr": lr,
            "kl_loss_type": kl_loss_type,
            "rollout_max_prompt_len": rollout_max_prompt_len,
            "rollout_max_response_len": rollout_max_response_len,
            "rollout_max_context_len": rollout_max_context_len,
            "initial_adam_checkpoint": initial_adam_checkpoint,
            "initial_adam_completion_sha256": initial_adam_completion_sha256,
            "initial_adam_source_tree_sha256": initial_adam_source_tree_sha256,
            "initial_adam_step": initial_adam_step,
        }
        try:
            gate_leg = _deployed_function("precision_resume_gate_leg")
            leg_1 = gate_leg.spawn(leg=1, **common)
            cast_calls = ledger["calls"]
            assert isinstance(cast_calls, list)
            cast_calls.append(
                {"leg": 1, "function_call_id": leg_1.object_id}
            )
            persist_precision_ledger()
            leg_1_result = leg_1.get()
            contract_digest = str(leg_1_result["contract_sha256"])
            ledger["contract_sha256"] = contract_digest
            ledger["state"] = "launching_leg_2"
            persist_precision_ledger()

            leg_2 = gate_leg.spawn(
                leg=2,
                expected_contract_sha256=contract_digest,
                **common,
            )
            cast_calls.append(
                {"leg": 2, "function_call_id": leg_2.object_id}
            )
            persist_precision_ledger()
            leg_2_result = leg_2.get()
            if leg_2_result.get("contract_sha256") != contract_digest:
                raise RuntimeError("leg 2 returned a different contract")
            ledger["state"] = "finalizing"
            persist_precision_ledger()

            finalizer = _deployed_function(
                "finalize_precision_resume_gate"
            ).spawn(
                expected_contract_sha256=contract_digest,
                **common,
            )
            ledger["finalizer_call_id"] = finalizer.object_id
            persist_precision_ledger()
            result = finalizer.get()
            ledger["state"] = "passed"
            ledger["passed_path"] = result["passed_path"]
            persist_precision_ledger()
        except Exception as exc:
            ledger["state"] = "failed"
            ledger["error"] = f"{type(exc).__name__}: {exc}"
            persist_precision_ledger()
            raise
        print(json.dumps(result, sort_keys=True), flush=True)
        return
    elif action == "convert":
        if not run_name or not hf_checkpoint or not output_name:
            raise ValueError(
                "convert requires --run-name, --hf-checkpoint (the origin HF), "
                "and --output-name"
            )
        if dry_run:
            print(
                f"convert {RAW_RL_ROOT}/{run_name}/iter_{step:07d} using "
                f"{hf_checkpoint} -> {HF_EXPORT_ROOT}/{output_name}"
            )
            return
        handle = _deployed_function("convert_rl_to_hf").spawn(
            run_name=run_name,
            origin_hf=hf_checkpoint,
            output_name=output_name,
            step=step,
            force=force,
        )
    elif action == "v2r4-gate":
        normalized_batch = batch_label.strip().upper()
        if candidate_step not in V2R4_GATE_CANDIDATES:
            raise ValueError(
                "v2r4-gate requires --candidate-step 6000, 8000, or 9920"
            )
        if normalized_batch not in V2R4_GATE_BATCHES:
            raise ValueError("v2r4-gate requires --batch-label A or B")
        _require_frozen_v2r4_contract_digest(contract_sha256)
        candidate = V2R4_GATE_CANDIDATES[candidate_step]
        batch = V2R4_GATE_BATCHES[normalized_batch]
        gate_run_name = (
            f"v2r4a-gate-w190-s{candidate_step}-"
            f"batch-{normalized_batch.lower()}"
        )
        if dry_run:
            command = build_train_command(
                hf_checkpoint=str(candidate["hf_path"]),
                run_name=gate_run_name,
                model_id=MODEL_ID,
                num_rollout=4,
                dynamic_filter=False,
                rollout_seed=int(batch["rollout_seed"]),
                save_interval=0,
                eval_interval=0,
                wandb_project="chess_interleave_50m",
                wandb_group="v2r4a_production_gate",
                deterministic_inference=True,
                rollout_only=True,
                train_file=str(batch["path"]),
                train_file_sha256=str(batch["sha256"]),
                data_source_path=STRICT_GATE_DATA_SOURCE_PATH,
                deterministic_seed_by_sample_index=True,
                fault_tolerance=False,
                rollout_health_check_interval=1e18,
            )
            print(" ".join(command))
            return
        raise RuntimeError(
            "individual v2r4 production launches are forbidden; use "
            "--action v2r4-launch-all so the sole exact-once ledger owns all "
            "six FunctionCall IDs"
        )
    elif action == "v2r4-launch-all":
        if any(
            (
                candidate_step,
                bool(batch_label),
                bool(hf_checkpoint),
                bool(run_name),
                bool(output_name),
                force,
                wait,
                dry_run,
            )
        ):
            raise ValueError(
                "v2r4-launch-all accepts only --contract-sha256 and "
                "optionally --launch-ledger-path"
            )
        _require_frozen_v2r4_contract_digest(contract_sha256)
        ledger_path = Path(launch_ledger_path).expanduser().resolve()
        if ledger_path.exists():
            raise FileExistsError(
                "v2r4 exact-once launch ledger already exists; refusing any "
                f"additional spawn: {ledger_path}"
            )
        cells = list(_v2r4_contract_static()["cells"])
        ledger: dict[str, object] = {
            "schema": "interleaved-v2r4-gate-launch-ledger-v1",
            "version": V2R4_GATE_VERSION,
            "state": "launching",
            "contract_sha256": contract_sha256,
            "expected_call_count": 6,
            "calls": [],
        }

        def persist_ledger() -> None:
            core = {
                key: value
                for key, value in ledger.items()
                if key != "ledger_sha256"
            }
            ledger["ledger_sha256"] = _canonical_json_sha256(core)
            _atomic_json(ledger_path, ledger)

        # Write the intent before the first spawn.  If this process dies at
        # any later point, rerunning fails closed and requires reconciliation
        # against Modal rather than risking a duplicate.
        initial_core = dict(ledger)
        ledger["ledger_sha256"] = _canonical_json_sha256(initial_core)
        _exclusive_json(ledger_path, ledger)
        preflight_call = _deployed_function(
            "v2r4_gate_preflight"
        ).spawn(contract_sha256)
        ledger["preflight_call_id"] = preflight_call.object_id
        persist_ledger()
        try:
            preflight_result = preflight_call.get()
        except Exception as exc:
            ledger["state"] = "preflight_failed"
            ledger["launch_error"] = f"{type(exc).__name__}: {exc}"
            persist_ledger()
            raise
        if (
            preflight_result.get("contract_sha256") != contract_sha256
            or preflight_result.get("all_six_canonical_roots_absent")
            is not True
            or preflight_result.get("ray_worker_environment")
            != {
                "seed_mode": "sample-index",
                "artifact_root": (
                    f"{RAW_RL_ROOT}/v2r4a-ray-env-preflight"
                ),
                "gpu_allocated": False,
            }
            or len(preflight_result.get("run_roots", ())) != 6
        ):
            ledger["state"] = "preflight_failed"
            ledger["launch_error"] = "preflight result shape drifted"
            persist_ledger()
            raise RuntimeError("v2r4 preflight result shape drifted")
        ledger["preflight"] = preflight_result
        persist_ledger()
        for cell in cells:
            try:
                call = _deployed_function("v2r4_gate_rollout").spawn(
                    candidate_step=int(cell["candidate_step"]),
                    batch_label=str(cell["batch_label"]),
                    contract_sha256=contract_sha256,
                )
            except Exception as exc:
                ledger["state"] = "launch_failed"
                ledger["failed_cell"] = dict(cell)
                ledger["launch_error"] = (
                    f"{type(exc).__name__}: {exc}"
                )
                persist_ledger()
                raise
            call_record = {
                **cell,
                "function_call_id": call.object_id,
            }
            cast_calls = ledger["calls"]
            assert isinstance(cast_calls, list)
            cast_calls.append(call_record)
            persist_ledger()
            print(
                "SPAWNED v2r4-gate "
                f"{cell['candidate_step']}/{cell['batch_label']}: "
                f"{call.object_id}",
                flush=True,
            )
        ledger["state"] = "launched_all"
        persist_ledger()
        print(
            "SPAWNED v2r4-launch-all: "
            + json.dumps(ledger, sort_keys=True),
            flush=True,
        )
        return
    else:
        raise ValueError(
            "action must be 'train', 'precision-gate', 'convert', "
            "'v2r4-gate', or 'v2r4-launch-all'"
        )

    print(f"SPAWNED {action}: {handle.object_id}")
    if wait:
        print(handle.get())
