"""Strict HF trainer for the 50M interleaved pretraining/RL experiments.

This module intentionally has a narrower contract than ``trainer_hf.py``:

* the model is the native-context Qwen3 experiment model, with 47,243,264
  parameters at vocabulary size 81 or 47,245,312 at vocabulary size 85;
* pretraining, staged SFT, and mixed records arrive through authenticated,
  deterministically constructed manifests with stage-specific local batches;
* every optimizer event uses a globally normalized valid-token loss; and
* full resume and weights-only initialization are separate, fail-closed modes.

The data construction lives in :mod:`training.interleaved_data`.  The trainer
only consumes its public stream interface and never rebuilds or reshuffles a
manifest.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import pathlib
import re
import statistics
import sys
import sysconfig
import time
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import AutoModelForCausalLM, Qwen3Config

from llm_tokens.chess.tokenizer_factory import init_tokenizer

from .hf_tokenizer_utils import save_hf_tokenizer
from .immutable_checkpoint import (
    checkpoint_volume_commit_lock,
    checkpoint_directory,
    publish_diagnostic_snapshot_directory,
    publish_checkpoint_directory,
    publish_hf_export_directory,
    resolve_resume_checkpoint,
    temporary_checkpoint_directory,
    validate_checkpoint_run_root,
    validate_completed_checkpoint,
    validate_completed_diagnostic_snapshot,
    validate_completed_hf_export,
    write_diagnostic_snapshot_completion_marker,
    write_completion_marker,
    write_hf_export_completion_marker,
    write_latest_checkpoint_pointer,
)
from .optim_sched import build_optimizer
from .tokenizer_contract import (
    normalize_vocab_mapping,
    validate_hf_tokenizer_contract,
    validate_vocab_transition,
)


EXPECTED_PARAMETER_COUNT = 47_245_312
EXPECTED_VOCAB_SIZE = 85
EXPECTED_CONTEXT_LENGTH = 3_072
EXPECTED_LOCAL_BATCH_SIZE = 21
EXPECTED_WORLD_SIZE = 8
EXPECTED_GRAD_ACCUMULATION = 1
EXPECTED_GLOBAL_BATCH_SIZE = (
    EXPECTED_LOCAL_BATCH_SIZE * EXPECTED_WORLD_SIZE
)
EXPECTED_TOKEN_POSITIONS_PER_UPDATE = (
    EXPECTED_GLOBAL_BATCH_SIZE * EXPECTED_CONTEXT_LENGTH
)
SUPPORTED_EXPERIMENT_CONTEXT_LENGTHS = frozenset({2_048, 3_072})
RUNTIME_SITE_PACKAGES = sysconfig.get_paths()["purelib"]
SUPPORTED_EXPERIMENT_VOCAB_SIZES = frozenset({81, 85})
EXPECTED_PARAMETER_COUNTS_BY_VOCAB = {
    81: 47_243_264,
    85: 47_245_312,
}
CANONICAL_INITIALIZATION_VOCAB_SIZE = 85
DEFAULT_MODEL_INIT_SEED = 42
STATE_SCHEMA_VERSION = 2
IGNORE_INDEX = -100
MASTER_PARAMETER_DTYPE = torch.float32
COMPUTE_DTYPE = torch.bfloat16
PRECISION_CONTRACT = {
    "master_parameter_dtype": "float32",
    "optimizer_state_dtype": "float32",
    "forward_backward_dtype": "bfloat16",
    "gradient_dtype": "float32",
    "hf_export_dtype": "float32",
}
DETERMINISM_CONTRACT = {
    "deterministic_algorithms": True,
    "warn_only": False,
    "cublas_workspace_config": ":4096:8",
    "cudnn_benchmark": False,
    "cudnn_deterministic": True,
    "cuda_matmul_tf32": False,
    "cudnn_tf32": False,
    "device_specific_seed": True,
}
SUPPORTED_ATTENTION_BACKENDS = frozenset(
    {"sdpa", "flash_attention_2"}
)
SUPPORTED_COMPILE_MODES = frozenset(
    {"none", "default", "reduce-overhead", "max-autotune"}
)
PINNED_FLASH_ATTENTION_VERSION = "2.8.3"
BENCHMARK_OUTPUT_ROOT = pathlib.Path("/tmp/chess-interleave-benchmarks")
LOCAL_METRICS_SCHEMA = "interleaved-local-metrics-v1"
SAMPLE_PAD = 0
SAMPLE_PRETRAIN = 1
SAMPLE_SFT = 2
SAMPLE_POSITIVE_REPLAY = 3
SUPPORTED_SAMPLE_TYPES = frozenset(
    {SAMPLE_PAD, SAMPLE_PRETRAIN, SAMPLE_SFT, SAMPLE_POSITIVE_REPLAY}
)
INITIAL_LAUNCH_COMMAND_ENV = "INTERLEAVED_INITIAL_LAUNCH_COMMAND_JSON"
INITIAL_LAUNCH_COMMAND_SHA256_ENV = (
    "INTERLEAVED_INITIAL_LAUNCH_COMMAND_SHA256"
)


def _get(mapping: Any, key: str, default: Any = None) -> Any:
    """Read a key from a dict or OmegaConf mapping without changing it."""
    if mapping is None:
        return default
    getter = getattr(mapping, "get", None)
    if getter is not None:
        return getter(key, default)
    return getattr(mapping, key, default)


def configure_deterministic_training(
    seed: int,
    *,
    process_index: int = 0,
) -> dict[str, Any]:
    """Configure reproducible CUDA kernels and one stable seed per rank."""

    expected_workspace = str(
        DETERMINISM_CONTRACT["cublas_workspace_config"]
    )
    configured_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if configured_workspace is None:
        if torch.cuda.is_initialized():
            raise RuntimeError(
                "CUBLAS_WORKSPACE_CONFIG must be set before CUDA initialization"
            )
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = expected_workspace
    elif configured_workspace != expected_workspace:
        raise RuntimeError(
            "CUBLAS_WORKSPACE_CONFIG drifted: "
            f"{configured_workspace!r} != {expected_workspace!r}"
        )
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    set_seed(int(seed) + int(process_index), device_specific=False)
    observed = {
        "deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "warn_only": bool(
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        "cublas_workspace_config": os.environ.get(
            "CUBLAS_WORKSPACE_CONFIG"
        ),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cuda_matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32),
        "device_specific_seed": True,
    }
    if observed != DETERMINISM_CONTRACT:
        raise RuntimeError(
            f"deterministic training contract drifted: {observed}"
        )
    return observed


def _plain(value: Any) -> Any:
    """Convert OmegaConf containers to JSON-serializable Python containers."""
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(value):
            return OmegaConf.to_container(value, resolve=True)
    except ImportError:
        pass
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def assert_fp32_master_parameters(
    model: torch.nn.Module,
    *,
    where: str,
) -> None:
    """Fail if any trainable parameter is not an FP32 master parameter."""

    parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not parameters:
        raise RuntimeError(f"{where}: model has no trainable parameters")
    bad = [
        f"{name}={parameter.dtype}"
        for name, parameter in parameters
        if parameter.dtype != MASTER_PARAMETER_DTYPE
    ]
    if bad:
        preview = ", ".join(bad[:8])
        raise RuntimeError(
            f"{where}: trainable parameters must be FP32 master weights; "
            f"found {preview}"
        )


def assert_fp32_gradients(
    model: torch.nn.Module,
    *,
    where: str,
) -> None:
    """Fail if populated trainable gradients are not accumulated in FP32."""

    gradients = [
        (name, parameter.grad)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not gradients:
        raise RuntimeError(f"{where}: no trainable gradients were produced")
    bad = [
        f"{name}={gradient.dtype}"
        for name, gradient in gradients
        if gradient.dtype != torch.float32
    ]
    if bad:
        preview = ", ".join(bad[:8])
        raise RuntimeError(
            f"{where}: gradients must accumulate/reduce in FP32; found "
            f"{preview}"
        )


def _raw_optimizer(
    optimizer: torch.optim.Optimizer,
) -> torch.optim.Optimizer:
    return getattr(optimizer, "optimizer", optimizer)


def assert_fp32_optimizer(
    optimizer: torch.optim.Optimizer,
    *,
    where: str,
    require_initialized_state: bool,
) -> None:
    """Validate the optimizer parameter references and floating state."""

    raw_optimizer = _raw_optimizer(optimizer)
    parameters = [
        parameter
        for group in raw_optimizer.param_groups
        for parameter in group["params"]
        if parameter.requires_grad
    ]
    if not parameters:
        raise RuntimeError(f"{where}: optimizer has no trainable parameters")
    bad_parameters = [
        str(parameter.dtype)
        for parameter in parameters
        if parameter.dtype != MASTER_PARAMETER_DTYPE
    ]
    if bad_parameters:
        raise RuntimeError(
            f"{where}: optimizer references non-FP32 parameters: "
            f"{sorted(set(bad_parameters))}"
        )

    floating_state_count = 0
    bad_state: list[str] = []
    missing_state_parameters = [
        index
        for index, parameter in enumerate(parameters)
        if parameter not in raw_optimizer.state
    ]
    missing_adam_moments: list[str] = []
    for parameter_index, parameter in enumerate(parameters):
        state = raw_optimizer.state.get(parameter)
        if state is None:
            continue
        for moment_name in ("exp_avg", "exp_avg_sq"):
            moment = state.get(moment_name)
            if not isinstance(moment, torch.Tensor):
                missing_adam_moments.append(
                    f"parameter[{parameter_index}].{moment_name}"
                )
        for key, value in state.items():
            if not isinstance(value, torch.Tensor) or not value.is_floating_point():
                continue
            floating_state_count += 1
            if value.dtype != torch.float32:
                bad_state.append(f"{key}={value.dtype}")
    if bad_state:
        raise RuntimeError(
            f"{where}: floating optimizer state must be FP32; found "
            + ", ".join(bad_state[:8])
        )
    if require_initialized_state and floating_state_count == 0:
        raise RuntimeError(f"{where}: optimizer state is unexpectedly empty")
    if require_initialized_state and missing_state_parameters:
        raise RuntimeError(
            f"{where}: Adam state is missing for trainable parameter indices "
            f"{missing_state_parameters[:20]}"
        )
    if require_initialized_state and missing_adam_moments:
        raise RuntimeError(
            f"{where}: Adam moment state is incomplete: "
            + ", ".join(missing_adam_moments[:20])
        )


def assert_fp32_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    *,
    where: str,
) -> None:
    """Validate that every floating tensor destined for export is FP32."""

    floating = [
        (name, tensor)
        for name, tensor in state_dict.items()
        if isinstance(tensor, torch.Tensor) and tensor.is_floating_point()
    ]
    if not floating:
        raise RuntimeError(f"{where}: state dict has no floating tensors")
    bad = [
        f"{name}={tensor.dtype}"
        for name, tensor in floating
        if tensor.dtype != torch.float32
    ]
    if bad:
        raise RuntimeError(
            f"{where}: exported model tensors must be actual FP32 values; "
            f"found {', '.join(bad[:8])}"
        )


def assert_finite_gradient_norm(gradient_norm: Any, *, where: str) -> float:
    """Fail before the optimizer step when clipping observes NaN or Inf."""

    if gradient_norm is None:
        raise FloatingPointError(f"{where}: gradient norm was not returned")
    if isinstance(gradient_norm, torch.Tensor):
        if gradient_norm.numel() != 1:
            raise FloatingPointError(
                f"{where}: gradient norm must be scalar, got {gradient_norm.shape}"
            )
        value = float(gradient_norm.detach().float().item())
    else:
        value = float(gradient_norm)
    if not math.isfinite(value):
        raise FloatingPointError(
            f"{where}: gradient norm is non-finite ({value}); refusing optimizer step"
        )
    return value


def register_bf16_output_head_assertion(
    model: torch.nn.Module,
) -> tuple[dict[str, Any], Any]:
    """Prove the first real output-head computation happens in BF16.

    Accelerate converts autocast model outputs back to FP32 at the outer model
    boundary.  A hook on the unprepared LM head observes the inner activation
    and logits before that conversion.
    """

    get_output_embeddings = getattr(model, "get_output_embeddings", None)
    if get_output_embeddings is None:
        raise TypeError("model does not expose get_output_embeddings()")
    output_head = get_output_embeddings()
    if not isinstance(output_head, torch.nn.Module):
        raise TypeError("model output head is not a torch module")
    evidence: dict[str, Any] = {
        "validated": False,
        "input_dtype": None,
        "output_dtype": None,
    }
    handle_ref: dict[str, Any] = {}

    def assert_first_output_head_dtype(
        _module: torch.nn.Module,
        inputs: tuple[Any, ...],
        output: Any,
    ) -> None:
        handle = handle_ref.get("handle")
        if handle is not None:
            handle.remove()
        activation = inputs[0] if inputs else None
        evidence["input_dtype"] = (
            str(activation.dtype).removeprefix("torch.")
            if isinstance(activation, torch.Tensor)
            else None
        )
        evidence["output_dtype"] = (
            str(output.dtype).removeprefix("torch.")
            if isinstance(output, torch.Tensor)
            else None
        )
        if not isinstance(activation, torch.Tensor) or not isinstance(
            output, torch.Tensor
        ):
            raise RuntimeError(
                "first output-head call did not expose tensor input/output"
            )
        if output.dtype != COMPUTE_DTYPE:
            raise RuntimeError(
                "first output-head computation must produce actual BF16 logits "
                "before Accelerate converts the model output to FP32; got "
                f"input={activation.dtype}, output={output.dtype}"
            )
        evidence["validated"] = True

    handle = output_head.register_forward_hook(assert_first_output_head_dtype)
    handle_ref["handle"] = handle
    return evidence, handle


def sha256_file(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_runtime_package_versions(
    expected: Mapping[str, Any],
) -> dict[str, str]:
    """Resolve installed distributions and require exact configured versions."""

    observed: dict[str, str] = {}
    for distribution, expected_version in sorted(expected.items()):
        name = str(distribution)
        wanted = str(expected_version)
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                f"required runtime package is unavailable: {name}=={wanted}"
            ) from exc
        if actual != wanted:
            raise RuntimeError(
                f"runtime package drift: {name}=={actual}, expected {wanted}"
            )
        observed[name] = actual
    if not observed:
        raise ValueError("expected runtime package mapping is empty")
    return observed


def runtime_distribution_identity() -> dict[str, Any]:
    """Hash the complete installed Python distribution name/version inventory."""

    inventory: dict[str, str] = {}
    for distribution in importlib.metadata.distributions(
        path=[RUNTIME_SITE_PACKAGES]
    ):
        raw_name = distribution.metadata.get("Name")
        if not raw_name:
            raise RuntimeError(
                "installed distribution lacks a canonical metadata Name"
            )
        name = re.sub(r"[-_.]+", "-", str(raw_name)).lower()
        version = str(distribution.version)
        previous = inventory.setdefault(name, version)
        if previous != version:
            raise RuntimeError(
                f"multiple installed versions for {name}: {previous}, {version}"
            )
    if not inventory:
        raise RuntimeError("installed Python distribution inventory is empty")
    return {
        "distribution_count": len(inventory),
        "inventory_sha256": hashlib.sha256(
            json.dumps(
                inventory,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest(),
    }


def _directory_file_identity(root: pathlib.Path) -> dict[str, Any]:
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not files:
        raise RuntimeError(f"snapshot directory is empty: {root}")
    return {
        "file_count": len(files),
        "total_bytes": sum(int(row["bytes"]) for row in files),
        "files": files,
        "manifest_sha256": hashlib.sha256(
            json.dumps(
                files,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest(),
    }


def new_diagnostic_ce_cumulative(*, through_step: int = 0) -> dict[str, Any]:
    if (
        isinstance(through_step, bool)
        or not isinstance(through_step, int)
        or through_step < 0
    ):
        raise ValueError("through_step must be a non-negative integer")
    return {
        "schema": "interleaved-diagnostic-ce-cumulative-v1",
        "through_step": through_step,
        "pretrain_loss_sum": 0.0,
        "pretrain_token_count": 0,
        "pretrain_contributing_steps": 0,
        "sft_loss_sum": 0.0,
        "sft_token_count": 0,
        "sft_contributing_steps": 0,
    }


def add_diagnostic_ce_step(
    cumulative: Mapping[str, Any],
    *,
    step: int,
    pretrain_loss_sum: float,
    pretrain_token_count: int,
    sft_loss_sum: float,
    sft_token_count: int,
) -> dict[str, Any]:
    expected_step = int(cumulative.get("through_step", -1)) + 1
    if isinstance(step, bool) or not isinstance(step, int) or step != expected_step:
        raise ValueError(
            f"diagnostic CE step must be contiguous: {step} != {expected_step}"
        )
    values = (pretrain_loss_sum, sft_loss_sum)
    if any(not math.isfinite(float(value)) or float(value) < 0 for value in values):
        raise ValueError("diagnostic CE loss sums must be finite/non-negative")
    counts = (pretrain_token_count, sft_token_count)
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in counts
    ):
        raise ValueError("diagnostic CE token counts must be non-negative ints")
    result = dict(cumulative)
    result["through_step"] = step
    result["pretrain_loss_sum"] = (
        float(result["pretrain_loss_sum"]) + float(pretrain_loss_sum)
    )
    result["pretrain_token_count"] = (
        int(result["pretrain_token_count"]) + pretrain_token_count
    )
    result["pretrain_contributing_steps"] = int(
        result["pretrain_contributing_steps"]
    ) + int(pretrain_token_count > 0)
    result["sft_loss_sum"] = (
        float(result["sft_loss_sum"]) + float(sft_loss_sum)
    )
    result["sft_token_count"] = (
        int(result["sft_token_count"]) + sft_token_count
    )
    result["sft_contributing_steps"] = int(
        result["sft_contributing_steps"]
    ) + int(sft_token_count > 0)
    return result


def diagnostic_ce_interval(
    base: Mapping[str, Any],
    cumulative: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        base.get("schema") != "interleaved-diagnostic-ce-cumulative-v1"
        or cumulative.get("schema")
        != "interleaved-diagnostic-ce-cumulative-v1"
    ):
        raise ValueError("invalid diagnostic CE cumulative schema")
    start_step = int(base.get("through_step", -1)) + 1
    end_step = int(cumulative.get("through_step", -1))
    if end_step < start_step:
        raise ValueError("diagnostic CE interval is empty or reversed")
    result: dict[str, Any] = {
        "schema": "interleaved-diagnostic-ce-interval-v1",
        "measurement_semantics": (
            "token_weighted_training_stream_pre_update_batch_logits"
        ),
        "held_out": False,
        "endpoint_checkpoint_evaluation": False,
        "start_step": start_step,
        "end_step": end_step,
        "optimizer_steps": end_step - start_step + 1,
    }
    for prefix in ("pretrain", "sft"):
        loss_sum = float(cumulative[f"{prefix}_loss_sum"]) - float(
            base[f"{prefix}_loss_sum"]
        )
        token_count = int(cumulative[f"{prefix}_token_count"]) - int(
            base[f"{prefix}_token_count"]
        )
        contributing_steps = int(
            cumulative[f"{prefix}_contributing_steps"]
        ) - int(base[f"{prefix}_contributing_steps"])
        if (
            not math.isfinite(loss_sum)
            or loss_sum < 0
            or token_count <= 0
            or contributing_steps <= 0
        ):
            raise ValueError(
                f"diagnostic CE interval has no valid {prefix} mass"
            )
        result[f"{prefix}_loss_sum"] = loss_sum
        result[f"{prefix}_token_count"] = token_count
        result[f"{prefix}_contributing_steps"] = contributing_steps
        result[f"{prefix}_token_ce"] = loss_sum / token_count
    return result


def validate_diagnostic_ce_resume_state(
    state: Mapping[str, Any],
    *,
    global_step: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Authenticate the CE accumulator boundary stored in a resume state."""

    cumulative = state.get("diagnostic_ce_cumulative")
    interval_base = state.get("diagnostic_ce_interval_base")
    last_interval_base = state.get("diagnostic_last_ce_interval_base")
    last_interval = state.get("diagnostic_last_ce_interval")
    values = (
        cumulative,
        interval_base,
        last_interval_base,
        last_interval,
    )
    if not all(isinstance(value, Mapping) for value in values):
        raise ValueError(
            "diagnostic resume lacks CE cumulative/base/interval"
        )
    if (
        cumulative.get("through_step") != global_step
        or interval_base != cumulative
        or last_interval.get("end_step") != global_step
    ):
        raise ValueError("diagnostic CE resume boundary/state mismatch")
    if diagnostic_ce_interval(last_interval_base, cumulative) != last_interval:
        raise ValueError("diagnostic CE resume interval delta mismatch")
    return (
        dict(cumulative),
        dict(interval_base),
        dict(last_interval_base),
        dict(last_interval),
    )


def build_interleaved_qwen_config(
    *,
    vocab_size: int,
    bos_token_id: int | None,
    eos_token_id: int | None,
    pad_token_id: int | None,
    context_length: int = EXPECTED_CONTEXT_LENGTH,
) -> Qwen3Config:
    """Return the one architecture admitted by this experiment trainer."""
    vocab_size = int(vocab_size)
    context_length = int(context_length)
    if vocab_size not in SUPPORTED_EXPERIMENT_VOCAB_SIZES:
        raise ValueError(
            "Interleaved Qwen requires vocab in "
            f"{sorted(SUPPORTED_EXPERIMENT_VOCAB_SIZES)}, got {vocab_size}"
        )
    if context_length not in SUPPORTED_EXPERIMENT_CONTEXT_LENGTHS:
        raise ValueError(
            "Interleaved Qwen requires native context in "
            f"{sorted(SUPPORTED_EXPERIMENT_CONTEXT_LENGTHS)}, "
            f"got {context_length}"
        )
    return Qwen3Config(
        attention_bias=False,
        attention_dropout=0.0,
        head_dim=128,
        hidden_act="silu",
        hidden_size=512,
        initializer_range=0.02,
        intermediate_size=1536,
        max_position_embeddings=context_length,
        max_window_layers=12,
        num_attention_heads=8,
        num_hidden_layers=12,
        num_key_value_heads=4,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000.0,
        sliding_window=None,
        tie_word_embeddings=True,
        # Keep the canonical model and the optimizer-facing parameters in
        # FP32.  Accelerator's BF16 autocast policy below supplies BF16
        # forward/backward compute without quantizing the master parameters.
        torch_dtype="float32",
        use_cache=False,
        use_sliding_window=False,
        vocab_size=vocab_size,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )


def normalize_attention_backend(value: Any) -> str:
    backend = str(value or "sdpa").strip().lower()
    if backend not in SUPPORTED_ATTENTION_BACKENDS:
        raise ValueError(
            f"model.attn_implementation must be one of "
            f"{sorted(SUPPORTED_ATTENTION_BACKENDS)}, got {value!r}"
        )
    return backend


def normalize_compile_mode(value: Any) -> str:
    if value is True:
        return "default"
    if value in (False, None, "", "false", "False", "none", "None"):
        return "none"
    mode = str(value).strip().lower()
    if mode not in SUPPORTED_COMPILE_MODES:
        raise ValueError(
            f"training.torch_compile must be one of "
            f"{sorted(SUPPORTED_COMPILE_MODES)}, got {value!r}"
        )
    return mode


def _require_flash_attention_version(expected_version: str) -> str:
    try:
        installed = importlib.metadata.version("flash-attn")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(
            "flash_attention_2 was requested but the pinned flash-attn "
            f"{expected_version} package is unavailable"
        ) from error
    if installed != expected_version:
        raise RuntimeError(
            "flash_attention_2 package drift: "
            f"expected flash-attn=={expected_version}, got {installed}"
        )
    return installed


def build_interleaved_qwen_model(
    tokenizer: Any,
    *,
    attn_implementation: str = "sdpa",
    flash_attention_version: str = PINNED_FLASH_ATTENTION_VERSION,
    context_length: int = EXPECTED_CONTEXT_LENGTH,
    expected_parameter_count: int | None = None,
    model_init_seed: int = DEFAULT_MODEL_INIT_SEED,
) -> torch.nn.Module:
    """Build the fixed model from one vocabulary-invariant initialization.

    Both tokenizer experiments must start from the same random model.  Building
    Qwen directly at vocabulary size 81 or 85 does not provide that property:
    the different embedding shape advances PyTorch's random stream by a
    different amount before later layers are initialized.  Always initialize
    the canonical 85-row model under an isolated, explicitly seeded RNG, then
    shrink its tied embedding to 81 rows when requested.  This makes every
    non-vocabulary tensor and embedding rows 0:81 bitwise identical across the
    two models, while rows 81:85 are deterministic for later vocabulary
    expansion.

    The isolated RNG also prevents dataloader construction or call order from
    changing model initialization, and prevents model construction from
    advancing the caller's CPU RNG state.
    """

    backend = normalize_attention_backend(attn_implementation)
    if backend == "flash_attention_2":
        _require_flash_attention_version(str(flash_attention_version))
    vocab_size = len(tokenizer.get_vocab())
    if vocab_size not in SUPPORTED_EXPERIMENT_VOCAB_SIZES:
        raise ValueError(
            "Interleaved Qwen requires vocab in "
            f"{sorted(SUPPORTED_EXPERIMENT_VOCAB_SIZES)}, got {vocab_size}"
        )
    bos_token_id = tokenizer.bos_id()
    eos_token_id = tokenizer.eos_id()
    pad_token_id = tokenizer.pad_id()
    # LanTokenizer historically reports no padding ID, while
    # LanTokenizerSFT reports <bos> (ID 0).  The experiment model contract uses
    # <bos> for padding in both cases, so normalize that interface difference
    # before constructing the canonical model.  Without this normalization,
    # Qwen zeroes embedding row 0 for vocab 85 but randomly initializes it for
    # vocab 81, violating the shared-initialization invariant.
    if pad_token_id is None:
        pad_token_id = bos_token_id
    if (bos_token_id, eos_token_id, pad_token_id) != (0, 1, 0):
        raise ValueError(
            "Interleaved Qwen requires bos/eos/pad token IDs (0, 1, 0), got "
            f"{(bos_token_id, eos_token_id, pad_token_id)}"
        )
    config = build_interleaved_qwen_config(
        vocab_size=CANONICAL_INITIALIZATION_VOCAB_SIZE,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        context_length=context_length,
    )
    with torch.random.fork_rng(devices=[]):
        # Seed only the CPU generator used for CPU model construction.  Calling
        # torch.manual_seed here would also mutate CUDA generator state even
        # though fork_rng(devices=[]) intentionally snapshots only the CPU RNG.
        torch.default_generator.manual_seed(int(model_init_seed))
        model = AutoModelForCausalLM.from_config(
            config,
            attn_implementation=backend,
        )
        if vocab_size != CANONICAL_INITIALIZATION_VOCAB_SIZE:
            model.resize_token_embeddings(
                vocab_size,
                mean_resizing=False,
            )
            model.config.vocab_size = vocab_size
            model.tie_weights()

    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    if tuple(input_embeddings.weight.shape) != (vocab_size, 512):
        raise RuntimeError(
            "Interleaved Qwen embedding shape drifted: expected "
            f"({vocab_size}, 512), got {tuple(input_embeddings.weight.shape)}"
        )
    if output_embeddings is None or (
        input_embeddings.weight.data_ptr()
        != output_embeddings.weight.data_ptr()
    ):
        raise RuntimeError(
            "Interleaved Qwen input and output embeddings must remain tied"
        )
    actual_backend = str(
        getattr(model.config, "_attn_implementation", "")
    )
    if actual_backend != backend:
        raise RuntimeError(
            f"Transformers selected attention backend {actual_backend!r}, "
            f"but {backend!r} was required"
        )
    # Public custom fields survive save_pretrained and make backend provenance
    # visible without relying on Transformers' private config attributes.
    model.config.interleaved_attention_backend = backend
    model.config.interleaved_flash_attention_version = (
        str(flash_attention_version)
        if backend == "flash_attention_2"
        else None
    )
    model.config.interleaved_model_init_seed = int(model_init_seed)
    model.config.interleaved_initialization_vocab_size = (
        CANONICAL_INITIALIZATION_VOCAB_SIZE
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if expected_parameter_count is None:
        expected_parameter_count = EXPECTED_PARAMETER_COUNTS_BY_VOCAB.get(
            vocab_size
        )
    if expected_parameter_count is None:
        raise RuntimeError(
            f"No pinned parameter count for vocabulary size {vocab_size}"
        )
    if parameter_count != int(expected_parameter_count):
        raise RuntimeError(
            "Interleaved Qwen architecture drifted: "
            f"expected {int(expected_parameter_count):,}, "
            f"got {parameter_count:,}"
        )
    assert_fp32_master_parameters(
        model,
        where="model construction",
    )
    return model


def validate_topology(
    training_cfg: Any,
    *,
    world_size: int,
    allow_topology_override: bool = False,
) -> None:
    """Fail before training if the production optimization topology drifted."""
    local_batch = int(
        _get(
            training_cfg,
            "local_batch_size",
            _get(training_cfg, "batch_size", EXPECTED_LOCAL_BATCH_SIZE),
        )
    )
    grad_accum = int(
        _get(
            training_cfg,
            "gradient_accumulation_steps",
            EXPECTED_GRAD_ACCUMULATION,
        )
    )
    mixed_precision = str(_get(training_cfg, "mixed_precision", "bf16")).lower()
    if not allow_topology_override and local_batch != EXPECTED_LOCAL_BATCH_SIZE:
        raise ValueError(
            f"local_batch_size must be {EXPECTED_LOCAL_BATCH_SIZE}, got {local_batch}"
        )
    if local_batch <= 0:
        raise ValueError("local_batch_size must be positive")
    if grad_accum != EXPECTED_GRAD_ACCUMULATION:
        raise ValueError("Interleaved training does not use gradient accumulation")
    if mixed_precision != "bf16":
        raise ValueError(
            "Interleaved training requires accuracy-first BF16 mixed "
            "precision: FP32 master parameters and optimizer state with "
            "BF16 forward/backward compute"
        )
    if not allow_topology_override and int(world_size) != EXPECTED_WORLD_SIZE:
        raise ValueError(
            f"production topology requires {EXPECTED_WORLD_SIZE} ranks, "
            f"got {world_size}"
        )


def resolve_arc_steps(training_cfg: Any) -> tuple[int, ...]:
    """Resolve one or two exact optimizer arcs from the training config."""
    scheduler_cfg = _get(training_cfg, "scheduler", {})
    floor_tail_steps = int(_get(training_cfg, "floor_tail_steps", 0))
    if floor_tail_steps < 0:
        raise ValueError("training.floor_tail_steps must be non-negative")
    raw = _get(training_cfg, "arc_steps", None)
    if raw is None:
        raw = _get(scheduler_cfg, "arc_steps", None)
    if raw is None:
        total_steps = _get(training_cfg, "total_steps", None)
        if total_steps is None:
            raise ValueError("training.arc_steps or training.total_steps is required")
        raw = [int(total_steps) - floor_tail_steps]
    elif isinstance(raw, (int, float)):
        raw = [int(raw)]

    arc_steps = tuple(int(step) for step in raw)
    if len(arc_steps) not in (1, 2):
        raise ValueError(f"expected one or two scheduler arcs, got {arc_steps}")
    if any(step <= 0 for step in arc_steps):
        raise ValueError(f"all arc lengths must be positive, got {arc_steps}")

    configured_total = _get(training_cfg, "total_steps", None)
    if (
        configured_total is not None
        and int(configured_total) != sum(arc_steps) + floor_tail_steps
    ):
        raise ValueError(
            f"training.total_steps={configured_total} does not equal "
            f"sum(arc_steps)+floor_tail_steps="
            f"{sum(arc_steps) + floor_tail_steps}"
        )
    return arc_steps


@dataclass(frozen=True)
class ArcPosition:
    arc_index: int
    local_update: int
    arc_length: int
    warmup_steps: int


class ExactArcCosine:
    """Cosine schedule whose final *training update* uses ``min_lr``.

    ``completed_steps`` is the number of optimizer updates already completed.
    The LR installed in the optimizer is always the LR for the next update.
    When two arcs are configured, the second arc restarts warmup at its first
    update.  The trainer separately clears Adam moments at that boundary.
    ``floor_tail_steps`` optionally adds optimizer updates at exactly
    ``min_lr`` without stretching or restarting the cosine arcs.  Exp 4's
    FLOP-unbounded scratch-replay branch uses that tail for replay-induced
    steps after the unchanged P2 schedule.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        arc_steps: Sequence[int],
        peak_lr: float,
        min_lr: float,
        warmup_ratio: float,
        floor_tail_steps: int = 0,
    ) -> None:
        self.optimizer = optimizer
        self.arc_steps = tuple(int(value) for value in arc_steps)
        self.peak_lr = float(peak_lr)
        self.min_lr = float(min_lr)
        self.warmup_ratio = float(warmup_ratio)
        self.floor_tail_steps = int(floor_tail_steps)
        self.completed_steps = 0
        if not 0.0 <= self.warmup_ratio < 1.0:
            raise ValueError("warmup_ratio must be in [0, 1)")
        if not 0.0 < self.min_lr <= self.peak_lr:
            raise ValueError("require 0 < min_lr <= peak_lr")
        if len(self.arc_steps) not in (1, 2) or any(
            value <= 0 for value in self.arc_steps
        ):
            raise ValueError(f"invalid arc_steps={self.arc_steps}")
        if self.floor_tail_steps < 0:
            raise ValueError("floor_tail_steps must be non-negative")
        self._set_lr_for_next_update()

    @property
    def total_steps(self) -> int:
        return sum(self.arc_steps) + self.floor_tail_steps

    @property
    def cosine_steps(self) -> int:
        return sum(self.arc_steps)

    @property
    def boundaries(self) -> tuple[int, ...]:
        running = 0
        values: list[int] = []
        for length in self.arc_steps[:-1]:
            running += length
            values.append(running)
        return tuple(values)

    def position(self, update_index: int) -> ArcPosition:
        if update_index < 0 or update_index >= self.cosine_steps:
            raise IndexError(
                f"update_index {update_index} outside "
                f"cosine arcs [0, {self.cosine_steps})"
            )
        remaining = int(update_index)
        for arc_index, arc_length in enumerate(self.arc_steps):
            if remaining < arc_length:
                return ArcPosition(
                    arc_index=arc_index,
                    local_update=remaining,
                    arc_length=arc_length,
                    warmup_steps=int(arc_length * self.warmup_ratio),
                )
            remaining -= arc_length
        raise AssertionError("unreachable")

    def lr_for_update(self, update_index: int) -> float:
        if self.cosine_steps <= update_index < self.total_steps:
            return self.min_lr
        position = self.position(update_index)
        local = position.local_update
        warmup = position.warmup_steps
        length = position.arc_length

        if warmup > 0 and local < warmup:
            return self.peak_lr * float(local + 1) / float(warmup)

        decay_updates = length - warmup
        if decay_updates <= 1:
            return self.min_lr
        # The first post-warmup update starts decaying and the last update hits
        # the floor exactly. The last warmup update is exactly peak_lr.
        decay_position = local - warmup + 1
        progress = float(decay_position) / float(decay_updates)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.peak_lr - self.min_lr) * cosine

    def _set_lr_for_next_update(self) -> None:
        if self.completed_steps < self.total_steps:
            lr = self.lr_for_update(self.completed_steps)
        else:
            lr = self.min_lr
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def step(self) -> None:
        if self.completed_steps >= self.total_steps:
            raise RuntimeError("scheduler stepped past its configured arcs")
        self.completed_steps += 1
        self._set_lr_for_next_update()

    def get_last_lr(self) -> list[float]:
        return [float(group["lr"]) for group in self.optimizer.param_groups]

    def state_dict(self) -> dict[str, Any]:
        return {
            "arc_steps": list(self.arc_steps),
            "peak_lr": self.peak_lr,
            "min_lr": self.min_lr,
            "warmup_ratio": self.warmup_ratio,
            "floor_tail_steps": self.floor_tail_steps,
            "completed_steps": self.completed_steps,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected = {
            "arc_steps": list(self.arc_steps),
            "peak_lr": self.peak_lr,
            "min_lr": self.min_lr,
            "warmup_ratio": self.warmup_ratio,
            "floor_tail_steps": self.floor_tail_steps,
        }
        for key, value in expected.items():
            checkpoint_value = (
                state.get(key, 0)
                if key == "floor_tail_steps"
                else state.get(key)
            )
            if checkpoint_value != value:
                raise ValueError(
                    f"scheduler resume mismatch for {key}: "
                    f"checkpoint={checkpoint_value!r}, config={value!r}"
                )
        completed = int(state["completed_steps"])
        if not 0 <= completed <= self.total_steps:
            raise ValueError(f"invalid scheduler completed_steps={completed}")
        self.completed_steps = completed
        self._set_lr_for_next_update()


def causal_ce_sum(
    logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return CE sum for labels that are already next-token aligned.

    For a configured sequence length ``S``, a full pretraining record keeps the
    historical ``S`` source targets but replaces the arbitrary preceding
    context with BOS: ``input_ids = [BOS] + targets[:-1]`` and
    ``labels = targets``. Shifting again here would drop one target from every
    full record and break the experiment's exact token accounting. SFT cache
    records follow the same aligned-label contract, with prompt and padding
    positions set to ``IGNORE_INDEX``.
    """
    if logits.ndim != 3 or labels.ndim != 2:
        raise ValueError(
            f"expected logits [B,S,V] and labels [B,S], got "
            f"{tuple(logits.shape)} and {tuple(labels.shape)}"
        )
    if logits.shape[:2] != labels.shape:
        raise ValueError("logits and labels batch/sequence dimensions differ")
    aligned_labels = labels.contiguous()
    if attention_mask is not None:
        if attention_mask.shape != labels.shape:
            raise ValueError("attention_mask and labels shapes differ")
        aligned_labels = aligned_labels.masked_fill(
            ~attention_mask.to(dtype=torch.bool),
            IGNORE_INDEX,
        )
    valid_tokens = aligned_labels.ne(IGNORE_INDEX).sum()
    loss_sum = F.cross_entropy(
        logits.float().view(-1, logits.shape[-1]),
        aligned_labels.view(-1),
        ignore_index=IGNORE_INDEX,
        reduction="sum",
    )
    return loss_sum, valid_tokens


def normalize_sft_loss_weight(value: Any) -> float:
    """Validate the explicit mixed-objective weight for supervised SFT tokens."""

    weight = float(value)
    if not math.isfinite(weight) or weight <= 0.0:
        raise ValueError(
            "training.sft_loss_weight must be finite and strictly positive"
        )
    return weight


def weighted_causal_ce_sum(
    logits: torch.Tensor,
    labels: torch.Tensor,
    sample_type: torch.Tensor,
    *,
    sft_loss_weight: float,
    attention_mask: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Return the weighted mixed-task CE numerator and token accounting.

    The SFT coefficient is applied to both the loss numerator and its valid
    token denominator.  Therefore ``sft_loss_weight=1`` is the original raw
    valid-token objective, while a larger value changes only the relative task
    mixture—not the optimizer-step or data-exposure accounting.
    """

    weight = normalize_sft_loss_weight(sft_loss_weight)
    if logits.ndim != 3 or labels.ndim != 2:
        raise ValueError(
            f"expected logits [B,S,V] and labels [B,S], got "
            f"{tuple(logits.shape)} and {tuple(labels.shape)}"
        )
    if logits.shape[:2] != labels.shape:
        raise ValueError("logits and labels batch/sequence dimensions differ")
    if sample_type.ndim != 1 or sample_type.shape[0] != labels.shape[0]:
        raise ValueError("sample_type must have shape [B]")
    observed_types = {
        int(value)
        for value in sample_type.detach().cpu().reshape(-1).tolist()
    }
    unexpected_types = observed_types - SUPPORTED_SAMPLE_TYPES
    if unexpected_types:
        raise ValueError(
            f"unsupported sample_type values: {sorted(unexpected_types)}"
        )

    aligned_labels = labels.contiguous()
    if attention_mask is not None:
        if attention_mask.shape != labels.shape:
            raise ValueError("attention_mask and labels shapes differ")
        aligned_labels = aligned_labels.masked_fill(
            ~attention_mask.to(dtype=torch.bool),
            IGNORE_INDEX,
        )
    valid = aligned_labels.ne(IGNORE_INDEX)
    pad_rows = sample_type.eq(SAMPLE_PAD).unsqueeze(1)
    if (valid & pad_rows).any():
        raise ValueError("padding records must not contain supervised labels")

    losses = F.cross_entropy(
        logits.float().view(-1, logits.shape[-1]),
        aligned_labels.view(-1),
        ignore_index=IGNORE_INDEX,
        reduction="none",
    ).view_as(aligned_labels)
    row_weights = torch.ones(
        sample_type.shape[0],
        device=losses.device,
        dtype=losses.dtype,
    )
    row_weights = row_weights.masked_fill(
        sample_type.to(device=losses.device).eq(SAMPLE_PAD),
        0.0,
    )
    row_weights = row_weights.masked_fill(
        sample_type.to(device=losses.device).eq(SAMPLE_SFT),
        weight,
    )
    token_weights = row_weights.unsqueeze(1) * valid.to(losses.dtype)
    weighted_loss_sum = (losses * token_weights).sum()
    weighted_valid_tokens = token_weights.sum()
    raw_valid_tokens = valid.sum()
    pretrain_valid_mask = (
        valid
        & sample_type.to(device=valid.device).eq(SAMPLE_PRETRAIN).unsqueeze(1)
    )
    sft_valid_mask = (
        valid
        & sample_type.to(device=valid.device).eq(SAMPLE_SFT).unsqueeze(1)
    )
    pretrain_loss_sum = (
        losses * pretrain_valid_mask.to(losses.dtype)
    ).sum()
    sft_loss_sum = (losses * sft_valid_mask.to(losses.dtype)).sum()
    return (
        weighted_loss_sum,
        weighted_valid_tokens,
        raw_valid_tokens,
        pretrain_valid_mask.sum(),
        sft_valid_mask.sum(),
        pretrain_loss_sum,
        sft_loss_sum,
    )


def globally_normalized_backward_loss(
    local_loss_sum: torch.Tensor,
    *,
    global_valid_tokens: torch.Tensor | int,
    world_size: int,
) -> torch.Tensor:
    """Scale a local CE sum so DDP's gradient average is a global token mean."""
    if isinstance(global_valid_tokens, torch.Tensor):
        if global_valid_tokens.numel() != 1:
            raise ValueError("global_valid_tokens must be scalar")
        if int(global_valid_tokens.detach().item()) <= 0:
            raise ValueError("a mixed batch must contain at least one valid token")
        denominator = global_valid_tokens.to(
            device=local_loss_sum.device,
            dtype=local_loss_sum.dtype,
        )
    else:
        if int(global_valid_tokens) <= 0:
            raise ValueError("a mixed batch must contain at least one valid token")
        denominator = local_loss_sum.new_tensor(int(global_valid_tokens))
    return local_loss_sum * int(world_size) / denominator


def _extract_scalar(value: Any, *, name: str) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            raise ValueError(f"empty batch metadata field {name}")
        values = value.detach().cpu().reshape(-1).tolist()
        if any(item != values[0] for item in values[1:]):
            raise ValueError(f"inconsistent batch metadata field {name}: {values}")
        return values[0]
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError(f"empty batch metadata field {name}")
        if any(item != value[0] for item in value[1:]):
            raise ValueError(f"inconsistent batch metadata field {name}: {value}")
        return value[0]
    return value


def validate_resume_state(
    state: Mapping[str, Any],
    *,
    manifest_hash: str,
    arc_steps: Sequence[int],
    floor_tail_steps: int = 0,
    local_batch_size: int,
    world_size: int,
    attention_backend: str | None = None,
    torch_compile_mode: str | None = None,
    sft_loss_weight: float = 1.0,
    configured_provenance: Mapping[str, Any] | None = None,
) -> None:
    if int(state.get("schema_version", -1)) != STATE_SCHEMA_VERSION:
        raise ValueError("unsupported interleaved trainer state schema")
    expected = {
        "manifest_hash": str(manifest_hash),
        "arc_steps": list(arc_steps),
        "floor_tail_steps": int(floor_tail_steps),
        "local_batch_size": int(local_batch_size),
        "world_size": int(world_size),
        "gradient_accumulation_steps": EXPECTED_GRAD_ACCUMULATION,
        "sft_loss_weight": normalize_sft_loss_weight(sft_loss_weight),
        "precision_contract": dict(PRECISION_CONTRACT),
    }
    if attention_backend is not None:
        expected["attention_backend"] = str(attention_backend)
    if torch_compile_mode is not None:
        expected["torch_compile_mode"] = str(torch_compile_mode)
    if configured_provenance is not None:
        expected["configured_provenance"] = _plain(
            configured_provenance
        )
    for key, value in expected.items():
        # Schema-v2 checkpoints from before the optional floor tail omit the
        # field; omission is exactly equivalent to a zero-length tail.
        if key == "floor_tail_steps":
            checkpoint_value = state.get(key, 0)
        elif key == "sft_loss_weight":
            # Older v2 checkpoints omitted this field and are exactly the
            # unweighted objective.
            checkpoint_value = float(state.get(key, 1.0))
        else:
            checkpoint_value = state.get(key)
        if checkpoint_value != value:
            raise ValueError(
                f"resume mismatch for {key}: checkpoint={checkpoint_value!r}, "
                f"current={value!r}"
            )
    total_steps = sum(arc_steps) + int(floor_tail_steps)
    global_step = int(state.get("global_step", -1))
    if not 0 <= global_step <= total_steps:
        raise ValueError(f"invalid resume global_step={global_step}")
    cursor = int(state.get("manifest_cursor", -1))
    if cursor < 0:
        raise ValueError(f"invalid resume manifest_cursor={cursor}")
    if cursor != global_step:
        raise ValueError(
            f"resume cursor/global-step mismatch: {cursor} != {global_step}"
        )


def _candidate_weight_files(path: pathlib.Path) -> tuple[pathlib.Path, ...]:
    """Return authenticated unsharded or indexed safetensors weight files."""

    if not path.is_dir():
        raise RuntimeError(
            "weights-only initialization requires an authenticated immutable "
            f"HF export directory, got {path}"
        )
    validate_completed_hf_export(path)
    unsharded = path / "model.safetensors"
    index_path = path / "model.safetensors.index.json"
    if unsharded.is_file():
        if index_path.exists():
            raise RuntimeError(
                f"HF export contains both unsharded weights and an index: {path}"
            )
        return (unsharded,)
    if not index_path.is_file():
        raise FileNotFoundError(
            "authenticated HF export has neither model.safetensors nor "
            f"model.safetensors.index.json: {path}"
        )
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map") if isinstance(index, Mapping) else None
    if not isinstance(weight_map, Mapping) or not weight_map:
        raise RuntimeError(f"invalid HF safetensors index: {index_path}")
    filenames: set[str] = set()
    for tensor_name, filename in weight_map.items():
        if not isinstance(tensor_name, str) or not isinstance(filename, str):
            raise RuntimeError(f"invalid HF safetensors weight map: {index_path}")
        relative = pathlib.Path(filename)
        if (
            relative.is_absolute()
            or len(relative.parts) != 1
            or relative.suffix != ".safetensors"
        ):
            raise RuntimeError(
                f"unsafe safetensors shard path {filename!r} in {index_path}"
            )
        filenames.add(filename)
    files = tuple(path / filename for filename in sorted(filenames))
    missing = [file for file in files if not file.is_file()]
    if missing:
        raise FileNotFoundError(f"indexed safetensors shards are missing: {missing}")
    return files


def _canonical_launch_command_from_environment() -> dict[str, Any]:
    """Authenticate the exact initial outer command supplied by the launcher."""

    encoded = os.environ.get(INITIAL_LAUNCH_COMMAND_ENV)
    recorded_sha256 = os.environ.get(INITIAL_LAUNCH_COMMAND_SHA256_ENV)
    if not encoded or not recorded_sha256:
        raise RuntimeError(
            "authenticated staged initialization requires "
            f"{INITIAL_LAUNCH_COMMAND_ENV} and "
            f"{INITIAL_LAUNCH_COMMAND_SHA256_ENV}"
        )
    try:
        command = json.loads(encoded)
    except json.JSONDecodeError as exc:
        raise RuntimeError("initial launch command is not valid JSON") from exc
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(item, str) and item for item in command)
    ):
        raise RuntimeError("initial launch command must be a nonempty string list")
    canonical = json.dumps(
        command,
        ensure_ascii=True,
        separators=(",", ":"),
    )
    observed_sha256 = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if recorded_sha256 != observed_sha256:
        raise RuntimeError(
            "initial launch command SHA-256 drifted: "
            f"{recorded_sha256!r} != {observed_sha256!r}"
        )
    return {
        "schema": "interleaved-initial-launch-command-v1",
        "argv": command,
        "sha256": observed_sha256,
    }


def authenticated_weights_only_identity(
    path: os.PathLike[str] | str,
    *,
    destination_vocab: Mapping[str, Any],
    allow_vocab_expansion: bool,
    context_length: int,
) -> dict[str, Any]:
    """Bind a fresh stage to one exact authenticated HF parent export."""

    export_path = pathlib.Path(path)
    validated = validate_completed_hf_export(export_path)
    marker = validated["marker"]
    source_state = validated["state"]
    model_config_path = validated["export"] / "config.json"
    try:
        model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"weights-only source has invalid model config: {model_config_path}"
        ) from exc
    source_vocab_size = model_config.get("vocab_size")
    if isinstance(source_vocab_size, bool) or not isinstance(
        source_vocab_size, int
    ):
        raise RuntimeError("weights-only source config lacks integer vocab_size")
    source_tokenizer = validate_hf_tokenizer_contract(
        validated["export"],
        expected_vocab_size=source_vocab_size,
        expected_context_length=int(context_length),
    )
    transition = validate_vocab_transition(
        source_tokenizer["vocab_mapping"],
        destination_vocab,
        allow_vocab_expansion=bool(allow_vocab_expansion),
    )
    source_provenance = source_state.get("configured_provenance")
    if not isinstance(source_provenance, Mapping):
        raise RuntimeError(
            "weights-only source trainer state lacks configured provenance"
        )
    export_identity = marker.get("export_identity")
    if not isinstance(export_identity, Mapping) or not isinstance(
        export_identity.get("manifest_sha256"), str
    ):
        raise RuntimeError("weights-only source marker lacks file-manifest identity")
    for key in ("marker_sha256", "trainer_state_sha256"):
        if not isinstance(marker.get(key), str):
            raise RuntimeError(f"weights-only source marker lacks {key}")
    source_seed = source_provenance.get("seed")
    source_model_init_seed = source_state.get("model_init_seed")
    if isinstance(source_seed, bool) or not isinstance(source_seed, int):
        raise RuntimeError("weights-only source provenance lacks integer seed")
    if isinstance(source_model_init_seed, bool) or not isinstance(
        source_model_init_seed, int
    ):
        raise RuntimeError(
            "weights-only source trainer state lacks integer model_init_seed"
        )
    if model_config.get("interleaved_model_init_seed") != source_model_init_seed:
        raise RuntimeError(
            "weights-only source model config/trainer state model-init seed "
            "mismatch"
        )
    return {
        "schema": "interleaved-authenticated-parent-v1",
        "mode": "weights-only",
        "source_export_path": str(validated["export"]),
        "source_experiment": source_provenance.get("experiment"),
        "source_stage": source_provenance.get("stage"),
        "source_experiment_version": source_provenance.get(
            "experiment_version"
        ),
        "source_global_step": int(source_state.get("global_step", -1)),
        "source_seed": int(source_seed),
        "source_model_init_seed": int(source_model_init_seed),
        "source_marker_sha256": marker["marker_sha256"],
        "source_trainer_state_sha256": marker["trainer_state_sha256"],
        "source_export_manifest_sha256": export_identity[
            "manifest_sha256"
        ],
        "source_tokenizer_contract": source_tokenizer,
        "tokenizer_transition": transition,
    }


def load_weights_only(
    model: torch.nn.Module,
    path: os.PathLike[str] | str,
    *,
    allow_vocab_expansion: bool = False,
    destination_vocab: Mapping[str, Any],
    context_length: int,
) -> int | None:
    """Strictly load model tensors, optionally expanding only tied vocab rows.

    The controlled tokenizer experiment initializes the four new 85-token
    embedding rows from the destination model's seeded initialization and
    copies the 81 historical rows byte-for-byte.  No other tensor mismatch is
    accepted.
    """
    export_path = pathlib.Path(path)
    source_identity = authenticated_weights_only_identity(
        export_path,
        destination_vocab=destination_vocab,
        allow_vocab_expansion=allow_vocab_expansion,
        context_length=context_length,
    )
    transition = source_identity["tokenizer_transition"]
    expected_expansion = transition["transition"] == "81-to-85"
    if expected_expansion != bool(allow_vocab_expansion):
        raise RuntimeError(
            "vocabulary expansion setting disagrees with authenticated "
            f"tokenizer transition: {transition}"
        )
    weight_files = _candidate_weight_files(export_path)
    from safetensors.torch import load_file

    state_dict: dict[str, torch.Tensor] = {}
    for weight_file in weight_files:
        shard = load_file(str(weight_file), device="cpu")
        duplicates = sorted(set(state_dict) & set(shard))
        if duplicates:
            raise RuntimeError(
                f"duplicate tensors across HF shards: {duplicates[:20]}"
            )
        state_dict.update(shard)
    if len(weight_files) > 1:
        index = json.loads(
            (export_path / "model.safetensors.index.json").read_text(
                encoding="utf-8"
            )
        )
        weight_map = index["weight_map"]
        observed_map = {
            name: weight_file.name
            for weight_file in weight_files
            for name in load_file(str(weight_file), device="cpu")
        }
        if weight_map != observed_map:
            raise RuntimeError(
                "loaded safetensors shards disagree with their authenticated index"
            )
    if isinstance(state_dict, Mapping):
        for wrapper_key in ("model", "state_dict"):
            wrapped = state_dict.get(wrapper_key)
            if isinstance(wrapped, Mapping):
                state_dict = wrapped
                break
    if not isinstance(state_dict, Mapping):
        raise TypeError(f"invalid state dict in {export_path}")
    if state_dict and all(str(key).startswith("module.") for key in state_dict):
        state_dict = {
            str(key)[len("module.") :]: value
            for key, value in state_dict.items()
        }
    else:
        state_dict = dict(state_dict)
    assert_fp32_state_dict(
        state_dict,
        where=f"authenticated weights-only source {export_path}",
    )

    expanded_from: int | None = None
    if allow_vocab_expansion:
        destination_state = model.state_dict()
        expandable = (
            "model.embed_tokens.weight",
            "lm_head.weight",
        )
        expanded_keys: list[str] = []
        for key in expandable:
            if key not in state_dict or key not in destination_state:
                continue
            source = state_dict[key]
            destination = destination_state[key]
            if tuple(source.shape) == tuple(destination.shape):
                continue
            if (
                source.ndim != 2
                or destination.ndim != 2
                or source.shape[1:] != destination.shape[1:]
                or int(source.shape[0]) != 81
                or int(destination.shape[0]) != 85
            ):
                raise ValueError(
                    f"unsupported vocabulary expansion for {key}: "
                    f"{tuple(source.shape)} -> {tuple(destination.shape)}"
                )
            expanded = destination.detach().clone()
            expanded[: source.shape[0]].copy_(
                source.to(device=expanded.device, dtype=expanded.dtype)
            )
            state_dict[key] = expanded
            expanded_keys.append(key)
        if not expanded_keys:
            raise ValueError(
                "allow_vocab_expansion was requested but no 81->85 "
                "embedding tensor required expansion"
            )
        expanded_from = 81
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    allowed_missing: set[str] = set()
    config = getattr(model, "config", None)
    if (
        bool(getattr(config, "tie_word_embeddings", False))
        and "lm_head.weight" in missing
        and "model.embed_tokens.weight" in state_dict
    ):
        allowed_missing.add("lm_head.weight")
        tie_weights = getattr(model, "tie_weights", None)
        if tie_weights is None:
            raise TypeError("tied-embedding model does not expose tie_weights()")
        tie_weights()
    missing = [key for key in missing if key not in allowed_missing]
    if missing or unexpected:
        raise ValueError(
            f"weights-only architecture mismatch: missing={missing}, "
            f"unexpected={unexpected}"
        )
    return expanded_from


def _stream_attribute(stream: Any, name: str, default: Any = None) -> Any:
    if hasattr(stream, name):
        value = getattr(stream, name)
        return value() if callable(value) else value
    dataset = getattr(stream, "dataset", None)
    if dataset is not None and hasattr(dataset, name):
        value = getattr(dataset, name)
        return value() if callable(value) else value
    return default


def _make_interleaved_stream(
    data_cfg: Any,
    tokenizer: Any,
    *,
    rank: int,
    world_size: int,
    local_batch_size: int,
    start_cursor: int,
) -> Iterable[MutableMapping[str, Any]]:
    """Build the rank-local view of the immutable global manifest."""
    stream_cfg = _get(data_cfg, "interleaved", data_cfg)
    manifest_path = _get(
        stream_cfg,
        "leg_manifest_path",
        _get(stream_cfg, "manifest_path", None),
    )
    stream_kind = str(_get(stream_cfg, "kind", "")).lower()
    manifest_schema = None
    if manifest_path:
        path = pathlib.Path(str(manifest_path))
        if path.is_file():
            try:
                with path.open("r", encoding="utf-8") as handle:
                    manifest_schema = json.load(handle).get("schema")
            except (AttributeError, json.JSONDecodeError):
                manifest_schema = None
    if (
        stream_kind in {"scratch_replay", "exp4_scratch_replay"}
        or manifest_schema == "interleaved-scratch-replay-manifest-v1"
    ):
        from .scratch_replay import create_scratch_replay_dataloader

        factory = create_scratch_replay_dataloader
    else:
        from . import interleaved_data

        factory = getattr(
            interleaved_data, "create_interleaved_dataloader", None
        )
        if factory is None:
            raise AttributeError(
                "training.interleaved_data must expose "
                "create_interleaved_dataloader"
            )

    available = {
        "cfg": data_cfg,
        "data_cfg": data_cfg,
        "tokenizer": tokenizer,
        "source_root": _get(stream_cfg, "source_root", None),
        "source_manifest_path": _get(
            stream_cfg, "source_manifest_path", None
        ),
        "selection_manifest_path": _get(
            stream_cfg, "selection_manifest_path", None
        ),
        "sft_cache_dir": _get(stream_cfg, "sft_cache_dir", None),
        "leg_manifest_path": manifest_path,
        "pad_token_id": tokenizer.pad_id(),
        "bos_token_id": tokenizer.bos_id(),
        "rank": int(rank),
        "world_size": int(world_size),
        "local_batch_size": int(local_batch_size),
        "batch_size": int(local_batch_size),
        "start_cursor": int(start_cursor),
        "cursor": int(start_cursor),
        "num_workers": int(_get(stream_cfg, "num_workers", 0)),
        "max_open_shards": int(_get(stream_cfg, "max_open_shards", 64)),
    }
    signature = inspect.signature(factory)
    kwargs = {
        name: available[name]
        for name in signature.parameters
        if name in available
    }
    missing = [
        name
        for name, parameter in signature.parameters.items()
        if parameter.default is inspect.Parameter.empty
        and name not in kwargs
        and parameter.kind
        not in (
            inspect.Parameter.VAR_KEYWORD,
            inspect.Parameter.VAR_POSITIONAL,
        )
    ]
    if missing:
        raise ValueError(
            "interleaved data config cannot satisfy required factory "
            f"arguments: {missing}"
        )
    none_required = [
        name
        for name, value in kwargs.items()
        if value is None
        and signature.parameters[name].default is inspect.Parameter.empty
    ]
    if none_required:
        raise ValueError(
            f"missing required interleaved data paths: {none_required}"
        )
    stream = factory(**kwargs)
    return stream


class InterleavedHFTrainer:
    """Production trainer for a single pretraining arc or two matched arcs."""

    def __init__(
        self,
        cfg: Any,
        *,
        run_config_path: str | None = None,
        accelerator: Accelerator | None = None,
    ) -> None:
        self.cfg = cfg
        self.run_config_path = run_config_path
        self.mcfg = _get(cfg, "model", {})
        self.tcfg = _get(cfg, "training", {})
        self.dcfg = _get(cfg, "data", {})
        self.tokcfg = _get(cfg, "tokenizer", {})
        self.logging_cfg = _get(cfg, "logging", {})
        self.context_length = int(
            _get(self.mcfg, "block_size", EXPECTED_CONTEXT_LENGTH)
        )
        if self.context_length not in SUPPORTED_EXPERIMENT_CONTEXT_LENGTHS:
            raise ValueError(
                "model.block_size must be one of "
                f"{sorted(SUPPORTED_EXPERIMENT_CONTEXT_LENGTHS)}, "
                f"got {self.context_length}"
            )
        configured_sequence_length = int(
            _get(self.dcfg, "sequence_length", self.context_length)
        )
        if configured_sequence_length != self.context_length:
            raise ValueError(
                "data.sequence_length must equal the native model context: "
                f"{configured_sequence_length} != {self.context_length}"
            )
        configured_provenance = _plain(_get(cfg, "provenance", {}))
        if not isinstance(configured_provenance, Mapping):
            raise ValueError("configured provenance must be a mapping")
        self.configured_provenance = {
            str(key): value
            for key, value in configured_provenance.items()
        }
        self.attention_backend = normalize_attention_backend(
            _get(self.mcfg, "attn_implementation", "sdpa")
        )
        self.flash_attention_version = str(
            _get(
                self.mcfg,
                "flash_attention_version",
                PINNED_FLASH_ATTENTION_VERSION,
            )
        )
        self.torch_compile_mode = normalize_compile_mode(
            _get(self.tcfg, "torch_compile", "none")
        )
        self.sft_loss_weight = normalize_sft_loss_weight(
            _get(self.tcfg, "sft_loss_weight", 1.0)
        )
        stream_cfg = _get(self.dcfg, "interleaved", self.dcfg)
        self.data_num_workers = int(_get(stream_cfg, "num_workers", 0))
        if self.data_num_workers < 0:
            raise ValueError("data.num_workers must be non-negative")
        self.benchmark_only = bool(
            _get(self.tcfg, "benchmark_only", False)
        )
        self.benchmark_warmup_steps = int(
            _get(self.tcfg, "benchmark_warmup_steps", 0)
        )
        if self.benchmark_warmup_steps < 0:
            raise ValueError(
                "training.benchmark_warmup_steps must be non-negative"
            )
        self._benchmark_records: list[dict[str, Any]] = []

        mixed_precision = str(_get(self.tcfg, "mixed_precision", "bf16")).lower()
        backend = _get(self.logging_cfg, "backend", None)
        if backend in ("none", "null", ""):
            backend = None
        self.acc = accelerator or Accelerator(
            gradient_accumulation_steps=EXPECTED_GRAD_ACCUMULATION,
            mixed_precision=mixed_precision,
            log_with=backend,
        )
        actual_mixed_precision = str(
            getattr(self.acc, "mixed_precision", "")
        ).lower()
        if actual_mixed_precision != "bf16":
            raise RuntimeError(
                "Accelerator must use BF16 autocast while the model keeps "
                "FP32 master parameters; got "
                f"mixed_precision={actual_mixed_precision!r}"
            )
        self.local_batch_size = int(
            _get(
                self.tcfg,
                "local_batch_size",
                _get(self.tcfg, "batch_size", EXPECTED_LOCAL_BATCH_SIZE),
            )
        )
        validate_topology(
            self.tcfg,
            world_size=self.acc.num_processes,
            allow_topology_override=bool(
                _get(
                    self.tcfg,
                    "allow_topology_override",
                    _get(self.tcfg, "allow_world_size_override", False),
                )
            ),
        )
        self.arc_steps = resolve_arc_steps(self.tcfg)
        configured_floor_tail = _get(self.tcfg, "floor_tail_steps", None)
        if (
            configured_floor_tail is not None
            and int(configured_floor_tail) < 0
        ):
            raise ValueError("training.floor_tail_steps must be non-negative")

        configured_resume_path = _get(self.tcfg, "resume", None)
        self.resume_path = (
            str(resolve_resume_checkpoint(pathlib.Path(str(configured_resume_path))))
            if configured_resume_path
            else None
        )
        self.weights_only_path = _get(
            self.tcfg,
            "weights_only",
            _get(self.tcfg, "pretrained_weights", None),
        )
        if self.resume_path and self.weights_only_path:
            raise ValueError("full resume and weights-only init are mutually exclusive")

        self.output_dir = pathlib.Path(
            _get(
                self.tcfg,
                "output_dir",
                _get(self.tcfg, "save_dir", "checkpoints/interleaved_50m"),
            )
        )
        if self.benchmark_only:
            resolved_output = self.output_dir.resolve(strict=False)
            benchmark_root = BENCHMARK_OUTPUT_ROOT.resolve(strict=False)
            if not resolved_output.is_relative_to(benchmark_root):
                raise ValueError(
                    "benchmark-only outputs must remain under "
                    f"{BENCHMARK_OUTPUT_ROOT}, got {resolved_output}"
                )
            if backend is not None:
                raise ValueError("benchmark-only runs must disable trackers")
            if self.resume_path or self.weights_only_path:
                raise ValueError(
                    "benchmark-only runs require deterministic random init"
                )
        run_name = str(_get(self.tcfg, "run_name", "interleaved_50m"))
        self.resume_state = self._read_resume_state(self.resume_path)

        self.tokenizer = init_tokenizer(
            name=str(_get(self.tokcfg, "name", "LanTokenizerSFT")),
            config=self.tokcfg,
        )
        destination_vocab = normalize_vocab_mapping(self.tokenizer.get_vocab())
        self.vocab_size = len(destination_vocab)
        configured_vocab_size = int(
            _get(self.mcfg, "vocab_size", self.vocab_size)
        )
        if configured_vocab_size != self.vocab_size:
            raise ValueError(
                "model.vocab_size must match the tokenizer: "
                f"{configured_vocab_size} != {self.vocab_size}"
            )
        if self.vocab_size not in SUPPORTED_EXPERIMENT_VOCAB_SIZES:
            raise ValueError(
                "tokenizer vocabulary size must be one of "
                f"{sorted(SUPPORTED_EXPERIMENT_VOCAB_SIZES)}; "
                f"got {self.vocab_size}"
            )
        self.training_seed = int(_get(self.tcfg, "seed", 42))
        if _get(self.tcfg, "deterministic_algorithms", True) is not True:
            raise ValueError(
                "production interleaved training requires deterministic_algorithms=true"
            )
        self.determinism_contract = configure_deterministic_training(
            self.training_seed,
            process_index=self.acc.process_index,
        )
        if self.resume_state is not None and self.resume_state.get(
            "determinism_contract"
        ) != self.determinism_contract:
            raise ValueError(
                "resume mismatch for deterministic training contract"
            )
        configured_seed = self.configured_provenance.get("seed")
        configured_initialization = self.configured_provenance.get(
            "initialization_identity"
        )
        self.weights_only_identity: dict[str, Any] | None = None
        if self.weights_only_path or (
            isinstance(configured_initialization, Mapping)
            and configured_initialization.get("mode") == "weights-only"
        ):
            if not isinstance(configured_initialization, Mapping):
                raise RuntimeError(
                    "weights-only initialization lacks authenticated parent "
                    "identity in configured provenance"
                )
            if configured_seed != self.training_seed:
                raise RuntimeError(
                    "staged initialization seed drifted: "
                    f"{configured_seed!r} != {self.training_seed}"
                )
            if configured_initialization.get("destination_seed") != self.training_seed:
                raise RuntimeError(
                    "authenticated parent identity is not bound to the "
                    f"destination seed {self.training_seed}"
                )
            source_path = self.weights_only_path or configured_initialization.get(
                "source_export_path"
            )
            if not source_path:
                raise RuntimeError(
                    "authenticated parent identity lacks source_export_path"
                )
            actual_initialization = authenticated_weights_only_identity(
                source_path,
                destination_vocab=destination_vocab,
                allow_vocab_expansion=bool(
                    _get(self.tcfg, "allow_vocab_expansion", False)
                ),
                context_length=self.context_length,
            )
            actual_initialization["destination_seed"] = self.training_seed
            if dict(configured_initialization) != actual_initialization:
                raise RuntimeError(
                    "configured weights-only parent identity does not match "
                    f"the authenticated source export: configured="
                    f"{dict(configured_initialization)!r}, actual="
                    f"{actual_initialization!r}"
                )
            self.weights_only_identity = actual_initialization
            self.configured_provenance["initialization_identity"] = dict(
                actual_initialization
            )
        elif isinstance(configured_initialization, Mapping):
            expected_random_initialization = {
                "schema": "interleaved-random-initialization-v1",
                "mode": "random",
                "destination_seed": self.training_seed,
            }
            if dict(configured_initialization) != expected_random_initialization:
                raise RuntimeError(
                    "random initialization identity drifted: "
                    f"{dict(configured_initialization)!r} != "
                    f"{expected_random_initialization!r}"
                )
        launch_env_present = bool(os.environ.get(INITIAL_LAUNCH_COMMAND_ENV))
        launch_hash_env_present = bool(
            os.environ.get(INITIAL_LAUNCH_COMMAND_SHA256_ENV)
        )
        if launch_env_present != launch_hash_env_present:
            raise RuntimeError("initial launch command environment is incomplete")
        if launch_env_present:
            self.configured_provenance["initial_launch_command"] = (
                _canonical_launch_command_from_environment()
            )
        elif self.weights_only_identity is not None:
            raise RuntimeError(
                "weights-only initialization requires an authenticated "
                "stable initial launch command"
            )

        self.manifest_hash = self._resolve_manifest_hash()
        start_cursor = (
            int(self.resume_state["manifest_cursor"])
            if self.resume_state is not None
            else int(_get(self.dcfg, "start_cursor", 0))
        )
        self.stream = _make_interleaved_stream(
            self.dcfg,
            self.tokenizer,
            rank=self.acc.process_index,
            world_size=self.acc.num_processes,
            local_batch_size=self.local_batch_size,
            start_cursor=start_cursor,
        )
        stream_hash = _stream_attribute(self.stream, "manifest_hash", None)
        if stream_hash is not None and str(stream_hash) != self.manifest_hash:
            raise ValueError(
                f"stream manifest hash {stream_hash} != {self.manifest_hash}"
            )
        stream_baseline_steps = _stream_attribute(
            self.stream, "baseline_cosine_steps", None
        )
        if (
            stream_baseline_steps is not None
            and int(stream_baseline_steps) != sum(self.arc_steps)
        ):
            raise ValueError(
                f"scratch baseline contains {stream_baseline_steps} cosine "
                f"steps but config arcs contain {sum(self.arc_steps)}"
            )
        stream_floor_tail = _stream_attribute(
            self.stream, "floor_tail_steps", None
        )
        if stream_floor_tail is not None:
            stream_floor_tail = int(stream_floor_tail)
            if (
                configured_floor_tail is not None
                and int(configured_floor_tail) != stream_floor_tail
            ):
                raise ValueError(
                    f"configured floor_tail_steps={configured_floor_tail} "
                    f"differs from manifest={stream_floor_tail}"
                )
            self.floor_tail_steps = stream_floor_tail
        else:
            self.floor_tail_steps = int(configured_floor_tail or 0)
        self.total_steps = sum(self.arc_steps) + self.floor_tail_steps
        self.max_steps = int(_get(self.tcfg, "max_steps", self.total_steps))
        if not 0 < self.max_steps <= self.total_steps:
            raise ValueError(
                f"training.max_steps must be in [1, {self.total_steps}]"
            )
        stream_total_steps = _stream_attribute(self.stream, "total_steps", None)
        if stream_total_steps is not None and int(stream_total_steps) != self.total_steps:
            raise ValueError(
                f"manifest contains {stream_total_steps} steps but schedule "
                f"contains {self.total_steps}"
            )
        if self.resume_state is not None:
            validate_resume_state(
                self.resume_state,
                manifest_hash=self.manifest_hash,
                arc_steps=self.arc_steps,
                floor_tail_steps=self.floor_tail_steps,
                local_batch_size=self.local_batch_size,
                world_size=self.acc.num_processes,
                attention_backend=self.attention_backend,
                torch_compile_mode=self.torch_compile_mode,
                sft_loss_weight=self.sft_loss_weight,
                configured_provenance=self.configured_provenance,
            )
        if self.resume_state is not None:
            data_state = self.resume_state.get("data_state")
            if not isinstance(data_state, Mapping):
                raise ValueError("full resume checkpoint has no data_state")
            load_data_state = getattr(self.stream, "load_state_dict", None)
            if load_data_state is None:
                raise TypeError("interleaved stream cannot restore data state")
            load_data_state(data_state)

        self.manifest_cursor = start_cursor
        self._pending_cursor_end: int | None = None
        self.global_step = (
            int(self.resume_state["global_step"])
            if self.resume_state is not None
            else 0
        )
        self.optimizer_resets_completed = {
            int(value)
            for value in (
                self.resume_state.get("optimizer_resets_completed", [])
                if self.resume_state
                else []
            )
        }

        requires_random_init = bool(
            _stream_attribute(self.stream, "requires_random_init", False)
        )
        if requires_random_init and self.weights_only_path:
            raise ValueError(
                "Exp4 scratch replay requires random initialization; "
                "weights-only initialization is forbidden"
            )
        configured_seed = int(_get(self.tcfg, "seed", DEFAULT_MODEL_INIT_SEED))
        self.model_init_seed = configured_seed
        if requires_random_init:
            stream_model_init_seed = int(
                _stream_attribute(self.stream, "model_init_seed", 42)
            )
            if configured_seed != stream_model_init_seed:
                raise ValueError(
                    f"training.seed={configured_seed} differs from immutable "
                    f"scratch model_init_seed={stream_model_init_seed}"
                )
            self.model_init_seed = stream_model_init_seed
        self.model = build_interleaved_qwen_model(
            self.tokenizer,
            attn_implementation=self.attention_backend,
            flash_attention_version=self.flash_attention_version,
            context_length=self.context_length,
            expected_parameter_count=EXPECTED_PARAMETER_COUNTS_BY_VOCAB[
                self.vocab_size
            ],
            model_init_seed=self.model_init_seed,
        )
        self.vocab_expanded_from: int | None = (
            81
            if self.weights_only_identity is not None
            and self.weights_only_identity["tokenizer_transition"]["transition"]
            == "81-to-85"
            else None
        )
        if self.weights_only_path:
            loaded_expanded_from = load_weights_only(
                self.model,
                self.weights_only_path,
                allow_vocab_expansion=bool(
                    _get(self.tcfg, "allow_vocab_expansion", False)
                ),
                destination_vocab=destination_vocab,
                context_length=self.context_length,
            )
            if loaded_expanded_from != self.vocab_expanded_from:
                raise RuntimeError(
                    "loaded vocabulary transition disagrees with authenticated "
                    f"parent identity: {loaded_expanded_from!r} != "
                    f"{self.vocab_expanded_from!r}"
                )
        assert_fp32_master_parameters(
            self.model,
            where="before optimizer construction",
        )
        (
            self._forward_precision_evidence,
            self._forward_precision_hook,
        ) = register_bf16_output_head_assertion(self.model)
        self.optimizer = self._build_optimizer()
        assert_fp32_optimizer(
            self.optimizer,
            where="before Accelerator.prepare",
            require_initialized_state=False,
        )
        if self.torch_compile_mode != "none":
            compile_kwargs: dict[str, Any] = {
                "fullgraph": False,
                "dynamic": False,
            }
            if self.torch_compile_mode != "default":
                compile_kwargs["mode"] = self.torch_compile_mode
            self.model = torch.compile(self.model, **compile_kwargs)
        scheduler_cfg = _get(self.tcfg, "scheduler", {})
        self.scheduler = ExactArcCosine(
            self.optimizer,
            arc_steps=self.arc_steps,
            peak_lr=float(_get(_get(self.tcfg, "optimizer", {}), "lr", 1e-3)),
            min_lr=float(_get(scheduler_cfg, "eta_min", 1e-5)),
            warmup_ratio=float(_get(scheduler_cfg, "warmup_ratio", 0.05)),
            floor_tail_steps=self.floor_tail_steps,
        )

        self.model, self.optimizer = self.acc.prepare(
            self.model,
            self.optimizer,
        )
        assert_fp32_master_parameters(
            self.model,
            where="after Accelerator.prepare",
        )
        assert_fp32_optimizer(
            self.optimizer,
            where="after Accelerator.prepare",
            require_initialized_state=False,
        )
        self.acc.register_for_checkpointing(self.scheduler)

        if self.resume_path:
            self.acc.load_state(str(self.resume_path))
            assert_fp32_master_parameters(
                self.model,
                where="after full-state resume",
            )
            assert_fp32_optimizer(
                self.optimizer,
                where="after full-state resume",
                require_initialized_state=self.global_step > 0,
            )
            if self.scheduler.completed_steps != self.global_step:
                raise ValueError(
                    "scheduler/global-step mismatch after full resume: "
                    f"{self.scheduler.completed_steps} != {self.global_step}"
                )
        self._gradient_precision_validated = False
        self._optimizer_state_precision_validated = self.global_step > 0

        self.save_interval = int(_get(self.tcfg, "save_interval", 500))
        self.export_interval = int(_get(self.tcfg, "export_interval", 0))
        raw_snapshot_steps = _plain(
            _get(self.tcfg, "snapshot_steps", [])
        )
        if not isinstance(raw_snapshot_steps, list) or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in raw_snapshot_steps
        ):
            raise ValueError(
                "training.snapshot_steps must be a list of integers"
            )
        self.snapshot_steps = tuple(raw_snapshot_steps)
        if (
            tuple(sorted(set(self.snapshot_steps))) != self.snapshot_steps
            or any(
                step <= 0 or step > self.max_steps
                for step in self.snapshot_steps
            )
        ):
            raise ValueError(
                "training.snapshot_steps must be unique, increasing, and "
                f"inside [1, {self.max_steps}]"
            )
        if self.resume_state is not None and list(self.snapshot_steps) != list(
            self.resume_state.get("snapshot_steps", [])
        ):
            raise ValueError(
                "snapshot_steps changed across a full-state resume"
            )
        if self.snapshot_steps:
            if self.resume_state is None:
                self.diagnostic_ce_cumulative = (
                    new_diagnostic_ce_cumulative()
                )
                self.diagnostic_ce_interval_base = (
                    new_diagnostic_ce_cumulative()
                )
                self.diagnostic_last_ce_interval_base = None
                self.diagnostic_last_ce_interval = None
            else:
                (
                    self.diagnostic_ce_cumulative,
                    self.diagnostic_ce_interval_base,
                    self.diagnostic_last_ce_interval_base,
                    self.diagnostic_last_ce_interval,
                ) = validate_diagnostic_ce_resume_state(
                    self.resume_state,
                    global_step=self.global_step,
                )
        else:
            self.diagnostic_ce_cumulative = None
            self.diagnostic_ce_interval_base = None
            self.diagnostic_last_ce_interval_base = None
            self.diagnostic_last_ce_interval = None
        self.log_interval = int(_get(self.tcfg, "log_interval", 10))
        self.max_grad_norm = float(_get(self.tcfg, "max_grad_norm", 1.0))
        if self.benchmark_only:
            if self.max_steps <= self.benchmark_warmup_steps:
                raise ValueError(
                    "benchmark max_steps must exceed benchmark_warmup_steps"
                )
            if self.save_interval != 0 or self.export_interval != 0:
                raise ValueError(
                    "benchmark-only runs must set save_interval=0 and "
                    "export_interval=0"
                )
        self._last_log_time = time.monotonic()
        self._last_log_step = self.global_step
        saved_runtime_provenance = (
            self.resume_state.get("runtime_provenance", {})
            if self.resume_state is not None
            else {}
        )
        configured_runtime_packages = self.configured_provenance.get(
            "runtime_package_versions"
        )
        if configured_runtime_packages is None:
            runtime_package_versions = None
        elif not isinstance(configured_runtime_packages, Mapping):
            raise ValueError(
                "provenance.runtime_package_versions must be a mapping"
            )
        else:
            runtime_package_versions = validate_runtime_package_versions(
                configured_runtime_packages
            )
        distribution_identity = runtime_distribution_identity()
        configured_distribution_hash = self.configured_provenance.get(
            "runtime_distribution_inventory_sha256"
        )
        configured_distribution_count = self.configured_provenance.get(
            "runtime_distribution_count"
        )
        if configured_distribution_hash is not None and (
            str(configured_distribution_hash)
            != distribution_identity["inventory_sha256"]
            or int(configured_distribution_count)
            != distribution_identity["distribution_count"]
        ):
            raise RuntimeError(
                "complete installed Python distribution identity drifted: "
                f"{distribution_identity}"
            )
        self.runtime_provenance = {
            "attention_backend": self.attention_backend,
            "torch_compile_mode": self.torch_compile_mode,
            "torch_version": str(torch.__version__),
            "transformers_version": importlib.metadata.version("transformers"),
            "flash_attention_version": (
                _require_flash_attention_version(
                    self.flash_attention_version
                )
                if self.attention_backend == "flash_attention_2"
                else None
            ),
            "data_num_workers": self.data_num_workers,
            "sft_loss_weight": self.sft_loss_weight,
            "context_length": self.context_length,
            "vocab_size": self.vocab_size,
            "vocab_expanded_from": self.vocab_expanded_from,
            "model_init_seed": self.model_init_seed,
            "precision_contract": dict(PRECISION_CONTRACT),
            "determinism_contract": dict(self.determinism_contract),
            "runtime_package_versions": runtime_package_versions,
            "runtime_distribution_count": distribution_identity[
                "distribution_count"
            ],
            "runtime_distribution_inventory_sha256": distribution_identity[
                "inventory_sha256"
            ],
            "python_version": sys.version,
            "cuda_runtime_version": torch.version.cuda,
            "modal_app_id": self.configured_provenance.get("modal_app_id"),
            "modal_image_id": self.configured_provenance.get("modal_image_id"),
            "modal_app_name": self.configured_provenance.get("modal_app_name"),
            "modal_base_image": self.configured_provenance.get("modal_base_image"),
            "modal_client_version": self.configured_provenance.get(
                "modal_client_version"
            ),
            "canary_sample_evidence": saved_runtime_provenance.get(
                "canary_sample_evidence"
            ),
            "first_forward_output_head_input_dtype": saved_runtime_provenance.get(
                "first_forward_output_head_input_dtype"
            ),
            "first_forward_output_head_dtype": saved_runtime_provenance.get(
                "first_forward_output_head_dtype"
            ),
            "configured_provenance": dict(self.configured_provenance),
            "process_argv": list(sys.argv),
            "process_argv_sha256": hashlib.sha256(
                json.dumps(
                    list(sys.argv),
                    ensure_ascii=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
        }
        self.local_metrics_path = self.output_dir / "metrics.jsonl"
        self._last_local_metric_step = -1
        self._committed_checkpoint_paths: dict[
            tuple[pathlib.Path, int], pathlib.Path
        ] = {}
        self._volume_commit_guard_depth = 0

        self._validate_output_root_before_write()
        with self._checkpoint_volume_write_guard():
            if self.acc.is_main_process:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                self._write_config_snapshot()
                self._prepare_local_metrics()
        self._init_trackers(run_name)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        optimizer_cfg = _get(self.tcfg, "optimizer", {})
        # Peak LR is stage-configurable (for example, staged SFT warms to
        # 3e-4); everything else stays a frozen contract unless an explicit
        # guarded override below permits it.
        required = {
            "name": "adamw",
            "weight_decay": 0.1,
            "betas": [0.9, 0.95],
        }
        actual = {
            "name": str(_get(optimizer_cfg, "name", "adamw")).lower(),
            "lr": float(_get(optimizer_cfg, "lr", 1e-3)),
            "weight_decay": float(_get(optimizer_cfg, "weight_decay", 0.1)),
            "betas": list(_get(optimizer_cfg, "betas", [0.9, 0.95])),
        }
        lr = actual["lr"]
        if not math.isfinite(lr) or lr <= 0.0:
            raise ValueError(f"interleaved optimizer lr must be positive, got {lr}")
        allow_weight_decay_override = bool(
            _get(self.tcfg, "allow_weight_decay_override", False)
        )
        expected_without_lr = {k: v for k, v in required.items()}
        actual_without_lr = {k: v for k, v in actual.items() if k != "lr"}
        if allow_weight_decay_override:
            if actual["weight_decay"] not in {0.01, 0.1}:
                raise ValueError(
                    "controlled weight-decay override must be 0.01 or 0.1"
                )
            expected_without_lr["weight_decay"] = actual["weight_decay"]
        if actual_without_lr != expected_without_lr:
            raise ValueError(
                f"interleaved optimizer must match {expected_without_lr} "
                f"(plus a positive"
                f" lr), got {actual}"
            )
        return build_optimizer(
            self.model,
            {
                **actual,
                "exclude_bias_and_norm": True,
                "fused": bool(_get(optimizer_cfg, "fused", False)),
            },
        )

    @staticmethod
    def _read_resume_state(
        resume_path: str | None,
    ) -> dict[str, Any] | None:
        if not resume_path:
            return None
        checkpoint = resolve_resume_checkpoint(pathlib.Path(resume_path))
        return dict(validate_completed_checkpoint(checkpoint)["state"])

    def _validate_output_root_before_write(self) -> None:
        """Do not reuse an output identity without an authenticated resume."""

        if not self.output_dir.exists() or not any(self.output_dir.iterdir()):
            return
        if not self.resume_path:
            raise FileExistsError(
                "refusing to start a fresh run in a nonempty output root: "
                f"{self.output_dir}"
            )
        allowed_directories: set[str] = set()
        final = self.output_dir / "final"
        if final.exists():
            validated_final = validate_completed_hf_export(final)
            if self.resume_state is None or validated_final["state"] != self.resume_state:
                raise RuntimeError(
                    "completed final export does not match the requested resume "
                    f"state: {final}"
                )
            allowed_directories.add("final")
        snapshots = self.output_dir / "snapshots"
        if snapshots.exists():
            if not snapshots.is_dir() or snapshots.is_symlink():
                raise RuntimeError(f"snapshot root is not a regular directory: {snapshots}")
            for child in sorted(snapshots.iterdir()):
                if not child.is_dir() or child.name.startswith("."):
                    raise RuntimeError(
                        f"incomplete or unknown snapshot entry: {child}"
                    )
                validated_snapshot = validate_completed_diagnostic_snapshot(child)
                if int(validated_snapshot["state"].get("global_step", -1)) > int(
                    self.resume_state.get("global_step", -1)
                ):
                    raise RuntimeError(
                        "diagnostic snapshot is newer than the requested resume: "
                        f"{child}"
                    )
            allowed_directories.add("snapshots")
        latest = validate_checkpoint_run_root(
            self.output_dir,
            allowed_root_directories=frozenset(allowed_directories),
        )
        requested = pathlib.Path(self.resume_path).resolve(strict=True)
        if latest != requested:
            raise RuntimeError(
                "output root latest checkpoint differs from requested resume: "
                f"{latest} != {requested}"
            )

    @contextlib.contextmanager
    def _checkpoint_volume_write_guard(
        self,
        root: os.PathLike[str] | str | None = None,
    ):
        """Let every rank mutate the run tree under one launcher-visible lock."""

        depth = int(getattr(self, "_volume_commit_guard_depth", 0))
        if depth:
            self._volume_commit_guard_depth += 1
            try:
                yield
            finally:
                self._volume_commit_guard_depth -= 1
            return
        self._volume_commit_guard_depth = 1
        lock_root = pathlib.Path(
            getattr(self, "output_dir", root) if root is None else root
        )
        lock = (
            checkpoint_volume_commit_lock(lock_root)
            if self.acc.is_main_process
            else contextlib.nullcontext()
        )
        try:
            with lock:
                # Non-main ranks wait here while rank zero waits for a Modal
                # commit already holding the lock.  Once released, every rank
                # performs the checkpoint mutation before rank zero unlocks.
                self.acc.wait_for_everyone()
                try:
                    yield
                finally:
                    self.acc.wait_for_everyone()
        finally:
            self._volume_commit_guard_depth = 0

    @contextlib.contextmanager
    def _main_process_volume_write_guard(self):
        """Guard a mutation performed exclusively by the main process."""

        if not self.acc.is_main_process or int(
            getattr(self, "_volume_commit_guard_depth", 0)
        ):
            yield
            return
        with checkpoint_volume_commit_lock(self.output_dir):
            yield

    def _resolve_manifest_hash(self) -> str:
        stream_cfg = _get(self.dcfg, "interleaved", self.dcfg)
        expected = _get(
            stream_cfg,
            "expected_manifest_hash",
            _get(stream_cfg, "manifest_hash", None),
        )
        manifest_path = _get(
            stream_cfg,
            "leg_manifest_path",
            _get(stream_cfg, "manifest_path", None),
        )
        actual = None
        if manifest_path:
            path = pathlib.Path(str(manifest_path))
            if not path.is_file():
                raise FileNotFoundError(f"manifest does not exist: {path}")
            actual = sha256_file(path)
        if expected is None and actual is None:
            raise ValueError(
                "data.expected_manifest_hash or data.manifest_path is required"
            )
        if expected is not None and actual is not None and str(expected) != actual:
            raise ValueError(
                f"manifest SHA-256 mismatch: expected={expected}, actual={actual}"
            )
        return str(expected or actual)

    def _init_trackers(self, run_name: str) -> None:
        backend = _get(self.logging_cfg, "backend", None)
        if backend in (None, "none", "null", ""):
            return
        project = str(_get(self.logging_cfg, "project", "chess-interleaved-50m"))
        kwargs: dict[str, Any] = {}
        if str(backend) == "wandb":
            kwargs["wandb"] = {
                "entity": _get(self.logging_cfg, "entity", None),
                "name": run_name,
                "notes": _get(self.logging_cfg, "notes", None),
                "tags": list(_get(self.logging_cfg, "tags", [])),
            }
            group = _get(self.logging_cfg, "group", None)
            job_type = _get(self.logging_cfg, "job_type", None)
            if group:
                kwargs["wandb"]["group"] = str(group)
            if job_type:
                kwargs["wandb"]["job_type"] = str(job_type)
            run_id = _get(self.logging_cfg, "id", None)
            resume = _get(self.logging_cfg, "resume", None)
            if run_id:
                kwargs["wandb"]["id"] = str(run_id)
            if resume:
                kwargs["wandb"]["resume"] = str(resume)
        self.acc.init_trackers(
            project,
            config={
                "arc_steps": list(self.arc_steps),
                "floor_tail_steps": self.floor_tail_steps,
                "total_steps": self.total_steps,
                "manifest_hash": self.manifest_hash,
                "local_batch_size": self.local_batch_size,
                "world_size": self.acc.num_processes,
                "attention_backend": self.attention_backend,
                "torch_compile_mode": self.torch_compile_mode,
                "data_num_workers": self.data_num_workers,
                "sft_loss_weight": self.sft_loss_weight,
                "context_length": self.context_length,
                "vocab_size": self.vocab_size,
                "precision_contract": dict(PRECISION_CONTRACT),
                "configured_provenance": dict(
                    self.configured_provenance
                ),
                "token_positions_per_update": (
                    self.local_batch_size
                    * self.acc.num_processes
                    * self.context_length
                ),
            },
            init_kwargs=kwargs,
        )

    def _write_config_snapshot(self) -> None:
        try:
            from omegaconf import OmegaConf

            text = OmegaConf.to_yaml(self.cfg, resolve=True)
        except (ImportError, TypeError):
            text = json.dumps(_plain(self.cfg), indent=2)
        target = self.output_dir / "config.yaml"
        if target.exists():
            # A committed run root is immutable.  The authenticated trainer
            # state governs resume compatibility; never rewrite its original
            # launch snapshot with a later process's --resume argument.
            return
        target.write_text(text, encoding="utf-8")

    def _prepare_local_metrics(self) -> None:
        """Reconcile an append-only metric log with the resumable step."""

        if not self.local_metrics_path.is_file():
            return
        lines = self.local_metrics_path.read_text(encoding="utf-8").splitlines()
        kept: list[dict[str, Any]] = []
        changed = False
        last_step = -1
        nonempty = [line for line in lines if line.strip()]
        for index, line in enumerate(nonempty):
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                # A process kill can leave only the final append incomplete.
                if index != len(nonempty) - 1:
                    raise
                changed = True
                break
            if value.get("schema") != LOCAL_METRICS_SCHEMA:
                raise ValueError(
                    f"unsupported local metric schema in "
                    f"{self.local_metrics_path}"
                )
            step = int(value.get("step", -1))
            if step <= last_step:
                raise ValueError("local metric steps are not strictly increasing")
            if step > self.global_step:
                changed = True
                continue
            kept.append(value)
            last_step = step
        self._last_local_metric_step = last_step
        if changed or len(kept) != len(nonempty):
            temporary = self.local_metrics_path.with_suffix(".jsonl.tmp")
            text = "".join(
                json.dumps(value, sort_keys=True) + "\n"
                for value in kept
            )
            temporary.write_text(text, encoding="utf-8")
            os.replace(temporary, self.local_metrics_path)

    def _append_local_metrics(self, metrics: Mapping[str, Any]) -> None:
        if not self.acc.is_main_process:
            return
        if self.global_step <= self._last_local_metric_step:
            raise ValueError(
                f"refusing duplicate local metric step {self.global_step}"
            )
        record = {
            "schema": LOCAL_METRICS_SCHEMA,
            "step": self.global_step,
            "manifest_hash": self.manifest_hash,
            "runtime_provenance": dict(self.runtime_provenance),
            "metrics": _plain(metrics),
        }
        with self._main_process_volume_write_guard():
            with self.local_metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        self._last_local_metric_step = self.global_step

    def _validate_canary_sample_contract(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
        sample_type: torch.Tensor,
    ) -> None:
        """Prove the real canary update contains its required sample types."""

        contract = self.configured_provenance.get("canary_sample_contract")
        if contract is None:
            return
        contract = str(contract)
        if contract not in {"pt-only", "sft-only", "mixed-pt-sft"}:
            raise ValueError(f"unsupported canary sample contract: {contract!r}")
        nonpad_types = set(
            int(value)
            for value in sample_type[sample_type.ne(SAMPLE_PAD)].tolist()
        )
        expected_types = {
            "pt-only": {SAMPLE_PRETRAIN},
            "sft-only": {SAMPLE_SFT},
            "mixed-pt-sft": {SAMPLE_PRETRAIN, SAMPLE_SFT},
        }[contract]
        if nonpad_types != expected_types:
            raise RuntimeError(
                "real canary batch has the wrong sample types: "
                f"contract={contract} observed={sorted(nonpad_types)}"
            )

        if bool((labels[attention_mask.eq(0)] != IGNORE_INDEX).any().item()):
            raise RuntimeError("canary batch supervises right-padding positions")
        bos_id = int(self.tokenizer.bos_id())
        local_pt_rows = 0
        local_pt_supervised_tokens = 0
        for row_index in sample_type.eq(SAMPLE_PRETRAIN).nonzero().flatten().tolist():
            active_length = int(attention_mask[row_index].sum().item())
            if active_length <= 0:
                raise RuntimeError("canary PT row is empty")
            if int(input_ids[row_index, 0].item()) != bos_id:
                raise RuntimeError(
                    "canary PT row must start with the explicit BOS token"
                )
            active_labels = labels[row_index, :active_length]
            if int(active_labels[0].item()) == IGNORE_INDEX:
                raise RuntimeError(
                    "canary PT BOS context does not predict an active target"
                )
            supervised_tokens = int(active_labels.ne(IGNORE_INDEX).sum().item())
            if supervised_tokens <= 0:
                raise RuntimeError("canary PT row contains no supervised targets")
            local_pt_rows += 1
            local_pt_supervised_tokens += supervised_tokens

        local_sft_rows = 0
        local_sft_supervised_tokens = 0
        for row_index in sample_type.eq(SAMPLE_SFT).nonzero().flatten().tolist():
            active_length = int(attention_mask[row_index].sum().item())
            if active_length <= 1:
                raise RuntimeError("canary SFT row is empty or has no target")
            active_ids = input_ids[row_index, :active_length]
            active_labels = labels[row_index, :active_length]
            if (
                int(active_ids[0].item()) != bos_id
                or int(active_ids.eq(bos_id).sum().item()) != 1
            ):
                raise RuntimeError(
                    "canary SFT row must start with exactly one BOS token"
                )
            if int(active_labels[0].item()) != IGNORE_INDEX:
                raise RuntimeError("canary SFT prompt/BOS target is not masked")
            supervised = active_labels.ne(IGNORE_INDEX)
            supervised_tokens = int(supervised.sum().item())
            if supervised_tokens <= 0:
                raise RuntimeError("canary SFT row has no supervised response tokens")
            aligned = supervised[:-1]
            if bool(
                active_labels[:-1][aligned]
                .ne(active_ids[1:][aligned])
                .any()
                .item()
            ):
                raise RuntimeError(
                    "canary SFT labels are not next-token aligned after masking"
                )
            local_sft_rows += 1
            local_sft_supervised_tokens += supervised_tokens

        counts = torch.tensor(
            [
                local_pt_rows,
                local_sft_rows,
                local_pt_supervised_tokens,
                local_sft_supervised_tokens,
            ],
            dtype=torch.long,
            device=self.acc.device,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        global_counts = [int(value) for value in counts.cpu().tolist()]
        if contract == "pt-only" and not (
            global_counts[0] > 0
            and global_counts[1] == 0
            and global_counts[2] > 0
            and global_counts[3] == 0
        ):
            raise RuntimeError(f"PT canary token evidence is invalid: {global_counts}")
        if contract == "sft-only" and not (
            global_counts[0] == 0
            and global_counts[1] > 0
            and global_counts[2] == 0
            and global_counts[3] > 0
        ):
            raise RuntimeError(f"SFT canary token evidence is invalid: {global_counts}")
        if contract == "mixed-pt-sft" and not all(
            value > 0 for value in global_counts
        ):
            raise RuntimeError(
                f"mixed canary did not supervise both PT and SFT: {global_counts}"
            )
        self.runtime_provenance["canary_sample_evidence"] = {
            "contract": contract,
            "global_pretrain_rows": global_counts[0],
            "global_sft_rows": global_counts[1],
            "global_pretrain_supervised_tokens": global_counts[2],
            "global_sft_supervised_tokens": global_counts[3],
            "pt_leading_bos_validated": global_counts[0] > 0,
            "sft_bos_and_mask_validated": global_counts[1] > 0,
        }

    def _validate_batch(
        self,
        batch: MutableMapping[str, Any],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        for key in ("input_ids", "labels", "attention_mask", "sample_type"):
            if key not in batch:
                raise KeyError(f"interleaved batch missing {key}")
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        attention_mask = batch["attention_mask"]
        sample_type = batch["sample_type"]
        if not all(
            isinstance(value, torch.Tensor)
            for value in (input_ids, labels, attention_mask, sample_type)
        ):
            raise TypeError(
                "input_ids, labels, attention_mask and sample_type "
                "must be tensors"
            )
        expected_shape = (
            self.local_batch_size,
            self.context_length,
        )
        if tuple(input_ids.shape) != expected_shape:
            raise ValueError(
                f"local mixed batch must have shape {expected_shape}, "
                f"got {tuple(input_ids.shape)}"
            )
        if labels.shape != input_ids.shape or attention_mask.shape != input_ids.shape:
            raise ValueError("labels/attention_mask must match input_ids")
        if tuple(sample_type.shape) != (self.local_batch_size,):
            raise ValueError(
                "sample_type must have one value per local mixed record"
            )
        unexpected_types = set(
            int(value) for value in sample_type.detach().cpu().tolist()
        ) - SUPPORTED_SAMPLE_TYPES
        if unexpected_types:
            raise ValueError(
                f"unsupported sample_type values: {sorted(unexpected_types)}"
            )
        if input_ids.min().item() < 0 or input_ids.max().item() >= self.vocab_size:
            raise ValueError(
                "batch contains token IDs outside the configured "
                f"{self.vocab_size}-token vocabulary"
            )
        invalid_labels = labels.ne(IGNORE_INDEX) & (
            labels.lt(0) | labels.ge(self.vocab_size)
        )
        if invalid_labels.any():
            raise ValueError("batch contains invalid unmasked label IDs")
        pretrain_rows = sample_type.eq(SAMPLE_PRETRAIN)
        if bool(pretrain_rows.any().item()):
            bos_token_id = int(self.tokenizer.bos_id())
            if bool(input_ids[pretrain_rows, 0].ne(bos_token_id).any().item()):
                raise ValueError(
                    "every packed PT sequence must begin with the explicit BOS token"
                )
            if bool(
                (
                    labels[pretrain_rows, 0].eq(IGNORE_INDEX)
                    | attention_mask[pretrain_rows, 0].eq(0)
                )
                .any()
                .item()
            ):
                raise ValueError(
                    "packed PT BOS context must predict an active first target"
                )

        batch_hash = batch.get("manifest_hash")
        if batch_hash is not None:
            batch_hash = str(_extract_scalar(batch_hash, name="manifest_hash"))
            if batch_hash != self.manifest_hash:
                raise ValueError(
                    f"batch manifest hash {batch_hash} != {self.manifest_hash}"
                )
        cursor_start = batch.get("cursor_start")
        cursor_end = batch.get("cursor_end")
        if cursor_start is None or cursor_end is None:
            raise KeyError(
                "interleaved batches must include cursor_start and cursor_end"
            )
        cursor_start = int(_extract_scalar(cursor_start, name="cursor_start"))
        cursor_end = int(_extract_scalar(cursor_end, name="cursor_end"))
        if cursor_start != self.manifest_cursor:
            raise ValueError(
                f"manifest cursor discontinuity: expected {self.manifest_cursor}, "
                f"got {cursor_start}"
            )
        if cursor_end <= cursor_start:
            raise ValueError(f"invalid batch cursor range {cursor_start}:{cursor_end}")
        self._pending_cursor_end = cursor_end
        self._validate_canary_sample_contract(
            input_ids,
            labels,
            attention_mask,
            sample_type,
        )

        return (
            input_ids.to(self.acc.device, non_blocking=True),
            labels.to(self.acc.device, non_blocking=True),
            attention_mask.to(self.acc.device, non_blocking=True),
            sample_type.to(self.acc.device, non_blocking=True),
        )

    def _global_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
        sample_type: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        float,
        int,
        float,
        int,
        int,
        float | None,
        float | None,
        float,
        float,
        float,
    ]:
        (
            local_sum,
            local_weighted_count,
            local_raw_count,
            local_pretrain_count,
            local_sft_count,
            local_pretrain_sum,
            local_sft_sum,
        ) = weighted_causal_ce_sum(
            logits,
            labels,
            sample_type,
            sft_loss_weight=self.sft_loss_weight,
            attention_mask=attention_mask,
        )
        statistics = torch.stack(
            (
                local_sum.detach().float(),
                local_weighted_count.detach().float(),
                local_raw_count.detach().float(),
                local_pretrain_count.detach().float(),
                local_sft_count.detach().float(),
                local_pretrain_sum.detach().float(),
                local_sft_sum.detach().float(),
            )
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(statistics, op=dist.ReduceOp.SUM)
        (
            global_sum,
            global_weighted_count,
            global_raw_count,
            global_pretrain_count,
            global_sft_count,
            global_pretrain_sum,
            global_sft_sum,
        ) = statistics
        if (
            not torch.isfinite(global_sum).item()
            or not torch.isfinite(global_weighted_count).item()
            or global_weighted_count.item() <= 0
        ):
            raise FloatingPointError(
                "mixed objective produced a non-finite loss/count"
            )
        backward_loss = globally_normalized_backward_loss(
            local_sum,
            global_valid_tokens=global_weighted_count,
            world_size=self.acc.num_processes,
        )
        return (
            backward_loss,
            float((global_sum / global_weighted_count).item()),
            int(global_raw_count.item()),
            float(global_weighted_count.item()),
            int(global_pretrain_count.item()),
            int(global_sft_count.item()),
            (
                float(
                    (global_pretrain_sum / global_pretrain_count).item()
                )
                if global_pretrain_count.item() > 0
                else None
            ),
            (
                float((global_sft_sum / global_sft_count).item())
                if global_sft_count.item() > 0
                else None
            ),
            float(global_pretrain_sum.item()),
            float(global_sft_sum.item()),
            float(
                (
                    self.sft_loss_weight
                    * global_sft_count
                    / global_weighted_count
                ).item()
            ),
        )

    def _maybe_reset_optimizer(self) -> None:
        if self.global_step not in self.scheduler.boundaries:
            return
        if self.global_step in self.optimizer_resets_completed:
            return
        if not bool(_get(self.tcfg, "reset_optimizer_between_arcs", True)):
            self.optimizer_resets_completed.add(self.global_step)
            return
        raw_optimizer = getattr(self.optimizer, "optimizer", self.optimizer)
        raw_optimizer.state.clear()
        self._optimizer_state_precision_validated = False
        self.optimizer.zero_grad(set_to_none=True)
        self.optimizer_resets_completed.add(self.global_step)
        self.acc.print(
            f"[arc] reset AdamW state at update boundary {self.global_step}"
        )

    def train(self) -> None:
        self.model.train()
        iterator = iter(self.stream)
        while self.global_step < self.max_steps:
            self._maybe_reset_optimizer()
            step_started = time.monotonic()
            try:
                batch = next(iterator)
            except StopIteration as error:
                raise RuntimeError(
                    f"manifest stream ended at step {self.global_step}, "
                    f"before target {self.max_steps}"
                ) from error
            data_ready = time.monotonic()

            (
                input_ids,
                labels,
                attention_mask,
                sample_type,
            ) = self._validate_batch(batch)
            with self.acc.autocast():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
                if not self._forward_precision_evidence["validated"]:
                    raise RuntimeError(
                        "first real forward did not execute the registered "
                        "output-head BF16 precision assertion"
                    )
                if (
                    self.runtime_provenance[
                        "first_forward_output_head_dtype"
                    ]
                    is None
                ):
                    self.runtime_provenance[
                        "first_forward_output_head_input_dtype"
                    ] = self._forward_precision_evidence["input_dtype"]
                    self.runtime_provenance[
                        "first_forward_output_head_dtype"
                    ] = self._forward_precision_evidence["output_dtype"]
                (
                    backward_loss,
                    logged_loss,
                    valid_tokens,
                    weighted_valid_tokens,
                    pretrain_valid_tokens,
                    sft_valid_tokens,
                    pretrain_loss,
                    sft_loss,
                    pretrain_loss_sum,
                    sft_loss_sum,
                    effective_sft_share,
                ) = self._global_loss(
                    outputs.logits,
                    labels,
                    attention_mask,
                    sample_type,
                )
            self.acc.backward(backward_loss)
            if not self._gradient_precision_validated:
                assert_fp32_gradients(
                    self.model,
                    where="after first backward pass",
                )
                self._gradient_precision_validated = True
            gradient_norm = self.acc.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm
            )
            assert_finite_gradient_norm(
                gradient_norm,
                where=f"optimizer update {self.global_step + 1}",
            )
            lr_used = float(self.optimizer.param_groups[0]["lr"])
            self.optimizer.step()
            if not self._optimizer_state_precision_validated:
                assert_fp32_optimizer(
                    self.optimizer,
                    where="after first optimizer update",
                    require_initialized_state=True,
                )
                self._optimizer_state_precision_validated = True
            self.optimizer.zero_grad(set_to_none=True)
            self.scheduler.step()
            commit_step = getattr(self.stream, "commit_step", None)
            if commit_step is None:
                raise TypeError("interleaved stream must expose commit_step()")
            commit_step()
            if self._pending_cursor_end is None:
                raise AssertionError("validated batch did not set a pending cursor")
            stream_cursor = int(
                _stream_attribute(
                    self.stream,
                    "cursor",
                    self._pending_cursor_end,
                )
            )
            if stream_cursor != self._pending_cursor_end:
                raise ValueError(
                    f"stream committed cursor {stream_cursor}, expected "
                    f"{self._pending_cursor_end}"
                )
            self.manifest_cursor = self._pending_cursor_end
            self._pending_cursor_end = None
            self.global_step += 1
            if self.snapshot_steps:
                self.diagnostic_ce_cumulative = add_diagnostic_ce_step(
                    self.diagnostic_ce_cumulative,
                    step=self.global_step,
                    pretrain_loss_sum=pretrain_loss_sum,
                    pretrain_token_count=pretrain_valid_tokens,
                    sft_loss_sum=sft_loss_sum,
                    sft_token_count=sft_valid_tokens,
                )
            if self.benchmark_only:
                # One synchronization provides honest end-to-end benchmark
                # timing without imposing any production-loop overhead.
                if torch.cuda.is_available():
                    torch.cuda.synchronize(self.acc.device)
                step_finished = time.monotonic()
                self._benchmark_records.append(
                    {
                        "step": self.global_step,
                        "loss": logged_loss,
                        "valid_tokens": valid_tokens,
                        "data_seconds": data_ready - step_started,
                        "compute_seconds": step_finished - data_ready,
                        "step_seconds": step_finished - step_started,
                    }
                )

            if self.global_step % self.log_interval == 0:
                self._log(
                    loss=logged_loss,
                    valid_tokens=valid_tokens,
                    weighted_valid_tokens=weighted_valid_tokens,
                    pretrain_valid_tokens=pretrain_valid_tokens,
                    sft_valid_tokens=sft_valid_tokens,
                    pretrain_loss=pretrain_loss,
                    sft_loss=sft_loss,
                    effective_sft_share=effective_sft_share,
                    lr_used=lr_used,
                )
            if self.global_step in self.snapshot_steps:
                self.diagnostic_last_ce_interval_base = dict(
                    self.diagnostic_ce_interval_base
                )
                self.diagnostic_last_ce_interval = diagnostic_ce_interval(
                    self.diagnostic_last_ce_interval_base,
                    self.diagnostic_ce_cumulative,
                )
                self.diagnostic_ce_interval_base = dict(
                    self.diagnostic_ce_cumulative
                )
                self.save_diagnostic_snapshot()
            if self.save_interval > 0 and self.global_step % self.save_interval == 0:
                self.save_resume_checkpoint()
            if (
                self.export_interval > 0
                and self.global_step % self.export_interval == 0
            ):
                self.export_hf(self.output_dir / f"step_{self.global_step}")

        if self.benchmark_only:
            self._finish_benchmark()
            self.acc.end_training()
            return
        if self.snapshot_steps:
            if self.global_step != self.snapshot_steps[-1]:
                raise RuntimeError(
                    "diagnostic trajectory ended before its final snapshot"
                )
            self.acc.end_training()
            return
        self.save_resume_checkpoint()
        self.export_hf(self.output_dir / "final")
        self.acc.end_training()

    @staticmethod
    def _percentile(values: Sequence[float], quantile: float) -> float:
        if not values:
            raise ValueError("cannot compute a percentile of no values")
        ordered = sorted(float(value) for value in values)
        index = (len(ordered) - 1) * float(quantile)
        lower = math.floor(index)
        upper = math.ceil(index)
        if lower == upper:
            return ordered[lower]
        fraction = index - lower
        return (
            ordered[lower] * (1.0 - fraction)
            + ordered[upper] * fraction
        )

    def _benchmark_summary(self) -> dict[str, Any]:
        measured = [
            record
            for record in self._benchmark_records
            if int(record["step"]) > self.benchmark_warmup_steps
        ]
        if not measured:
            raise RuntimeError("benchmark produced no measured steps")
        step_seconds = [
            float(record["step_seconds"]) for record in measured
        ]
        data_seconds = [
            float(record["data_seconds"]) for record in measured
        ]
        compute_seconds = [
            float(record["compute_seconds"]) for record in measured
        ]
        positions_per_step = (
            self.local_batch_size
            * self.acc.num_processes
            * self.context_length
        )
        mean_step_seconds = statistics.fmean(step_seconds)
        return {
            "schema": "interleaved-throughput-benchmark-v1",
            "attention_backend": self.attention_backend,
            "torch_compile_mode": self.torch_compile_mode,
            "data_num_workers": self.data_num_workers,
            "runtime_provenance": dict(self.runtime_provenance),
            "world_size": self.acc.num_processes,
            "local_batch_size": self.local_batch_size,
            "sequence_length": self.context_length,
            "token_positions_per_step": positions_per_step,
            "warmup_steps": self.benchmark_warmup_steps,
            "measured_steps": len(measured),
            "mean_step_seconds": mean_step_seconds,
            "p50_step_seconds": self._percentile(step_seconds, 0.50),
            "p90_step_seconds": self._percentile(step_seconds, 0.90),
            "mean_data_seconds": statistics.fmean(data_seconds),
            "mean_compute_seconds": statistics.fmean(compute_seconds),
            "token_positions_per_second": (
                positions_per_step / mean_step_seconds
            ),
            "max_cuda_memory_allocated_bytes": (
                int(torch.cuda.max_memory_allocated(self.acc.device))
                if torch.cuda.is_available()
                else 0
            ),
            # Step 1 is before the first optimizer update and is the strictest
            # useful numerical-parity point across attention backends.
            "loss_trace": [
                {
                    "step": int(record["step"]),
                    "loss": float(record["loss"]),
                }
                for record in self._benchmark_records
            ],
        }

    def _finish_benchmark(self) -> None:
        summary = self._benchmark_summary()
        if self.acc.is_main_process:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            output = self.output_dir / "benchmark_result.json"
            output.write_text(
                json.dumps(summary, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            self.acc.print(
                "[benchmark-result] "
                + json.dumps(summary, sort_keys=True)
            )
        self.acc.wait_for_everyone()

    def _log(
        self,
        *,
        loss: float,
        valid_tokens: int,
        weighted_valid_tokens: float,
        pretrain_valid_tokens: int,
        sft_valid_tokens: int,
        pretrain_loss: float | None,
        sft_loss: float | None,
        effective_sft_share: float,
        lr_used: float,
    ) -> None:
        now = time.monotonic()
        elapsed = max(now - self._last_log_time, 1e-9)
        steps = self.global_step - self._last_log_step
        token_positions = (
            steps
            * self.local_batch_size
            * self.acc.num_processes
            * self.context_length
        )
        tokens_per_second = token_positions / elapsed
        self._last_log_time = now
        self._last_log_step = self.global_step
        metrics = {
            "train/loss": loss,
            "train/lr": lr_used,
            "train/global_valid_tokens": valid_tokens,
            "train/global_weighted_valid_tokens": weighted_valid_tokens,
            "train/global_pretrain_valid_tokens": pretrain_valid_tokens,
            "train/global_sft_valid_tokens": sft_valid_tokens,
            "train/sft_loss_weight": self.sft_loss_weight,
            "train/effective_sft_loss_mass_share": effective_sft_share,
            "train/token_positions_per_second": tokens_per_second,
            "train/manifest_cursor": self.manifest_cursor,
        }
        if pretrain_loss is not None:
            metrics["train/pretrain_token_loss"] = pretrain_loss
        if sft_loss is not None:
            metrics["train/sft_token_loss"] = sft_loss
        self.acc.print(
            f"step={self.global_step} loss={loss:.6f} lr={lr_used:.3e} "
            f"cursor={self.manifest_cursor} tok/s={tokens_per_second:.0f}"
        )
        self._append_local_metrics(metrics)
        if _get(self.logging_cfg, "backend", None) not in (
            None,
            "none",
            "null",
            "",
        ):
            self.acc.log(metrics, step=self.global_step)

    def _trainer_state(self) -> dict[str, Any]:
        stream_state_fn = getattr(self.stream, "state_dict", None)
        if stream_state_fn is None:
            raise TypeError("interleaved stream must expose state_dict()")
        state = {
            "schema_version": STATE_SCHEMA_VERSION,
            "global_step": self.global_step,
            "manifest_hash": self.manifest_hash,
            "manifest_cursor": self.manifest_cursor,
            "arc_steps": list(self.arc_steps),
            "floor_tail_steps": self.floor_tail_steps,
            "local_batch_size": self.local_batch_size,
            "world_size": self.acc.num_processes,
            "gradient_accumulation_steps": EXPECTED_GRAD_ACCUMULATION,
            "attention_backend": self.attention_backend,
            "torch_compile_mode": self.torch_compile_mode,
            "data_num_workers": self.data_num_workers,
            "sft_loss_weight": self.sft_loss_weight,
            "context_length": self.context_length,
            "vocab_size": self.vocab_size,
            "vocab_expanded_from": self.vocab_expanded_from,
            "weights_only_identity": (
                dict(self.weights_only_identity)
                if self.weights_only_identity is not None
                else None
            ),
            "model_init_seed": self.model_init_seed,
            "precision_contract": dict(PRECISION_CONTRACT),
            "determinism_contract": dict(self.determinism_contract),
            "configured_provenance": dict(self.configured_provenance),
            "runtime_provenance": dict(self.runtime_provenance),
            "snapshot_steps": list(self.snapshot_steps),
            "optimizer_resets_completed": sorted(
                self.optimizer_resets_completed
            ),
            "data_state": _plain(stream_state_fn()),
        }
        if self.snapshot_steps:
            if (
                self.diagnostic_ce_cumulative is None
                or self.diagnostic_ce_interval_base is None
                or self.diagnostic_last_ce_interval_base is None
                or self.diagnostic_last_ce_interval is None
            ):
                raise RuntimeError(
                    "diagnostic trainer state lacks finalized CE interval"
                )
            state.update(
                {
                    "diagnostic_ce_cumulative": dict(
                        self.diagnostic_ce_cumulative
                    ),
                    "diagnostic_ce_interval_base": dict(
                        self.diagnostic_ce_interval_base
                    ),
                    "diagnostic_last_ce_interval_base": dict(
                        self.diagnostic_last_ce_interval_base
                    ),
                    "diagnostic_last_ce_interval": dict(
                        self.diagnostic_last_ce_interval
                    ),
                }
            )
        return state

    def save_resume_checkpoint(
        self,
        output: os.PathLike[str] | str | None = None,
    ) -> pathlib.Path:
        with self._checkpoint_volume_write_guard(
            getattr(self, "output_dir", output)
        ):
            return self._save_resume_checkpoint_unlocked(output)

    def _save_resume_checkpoint_unlocked(
        self,
        output: os.PathLike[str] | str | None = None,
    ) -> pathlib.Path:
        root = pathlib.Path(output or self.output_dir)
        final = checkpoint_directory(root, self.global_step)
        temporary = temporary_checkpoint_directory(root, self.global_step)
        key = (root.resolve(strict=False), self.global_step)
        assert_fp32_master_parameters(
            self.model,
            where="before full-state checkpoint",
        )
        assert_fp32_optimizer(
            self.optimizer,
            where="before full-state checkpoint",
            require_initialized_state=self.global_step > 0,
        )
        already_committed = self._committed_checkpoint_paths.get(key)
        if already_committed is not None:
            validate_completed_checkpoint(already_committed)
            return already_committed
        if self.acc.is_main_process:
            if not final.exists():
                if temporary.exists():
                    raise FileExistsError(
                        "incomplete checkpoint publication must be diagnosed, "
                        f"not overwritten: {temporary}"
                    )
                final.parent.mkdir(parents=True, exist_ok=True)
                temporary.mkdir(parents=False, exist_ok=False)
        self.acc.wait_for_everyone()
        if final.exists():
            validated = validate_completed_checkpoint(final)
            if validated["state"] != self._trainer_state():
                raise FileExistsError(
                    "immutable checkpoint step already exists with different "
                    f"trainer state: {final}"
                )
            if self.acc.is_main_process:
                write_latest_checkpoint_pointer(root, final)
            self._committed_checkpoint_paths[key] = final
            return final
        self.acc.save_state(str(temporary))
        self.acc.wait_for_everyone()
        if self.acc.is_main_process:
            state_temporary = temporary / "trainer_state.json.tmp"
            state_final = temporary / "trainer_state.json"
            state_temporary.write_text(
                json.dumps(self._trainer_state(), indent=2, sort_keys=True),
                encoding="utf-8",
            )
            os.replace(state_temporary, state_final)
            write_completion_marker(temporary, step=self.global_step)
            publish_checkpoint_directory(
                temporary,
                final,
                allow_serialized_fallback=True,
            )
            write_latest_checkpoint_pointer(root, final)
            self.acc.print(f"[checkpoint] committed full resume state -> {final}")
        self.acc.wait_for_everyone()
        validate_completed_checkpoint(final)
        self._committed_checkpoint_paths[key] = final
        return final

    def save_diagnostic_snapshot(self) -> None:
        """Atomically publish a paired full-resume and clean-HF snapshot."""

        final = self.output_dir / "snapshots" / f"step_{self.global_step}"
        if final.exists() or final.is_symlink():
            self._save_diagnostic_snapshot_unlocked()
            return
        with self._checkpoint_volume_write_guard():
            self._save_diagnostic_snapshot_unlocked()

    def _save_diagnostic_snapshot_unlocked(self) -> None:
        """Publish a diagnostic snapshot while the Volume lock is held."""

        step = self.global_step
        snapshots_root = self.output_dir / "snapshots"
        final = snapshots_root / f"step_{step}"
        temporary = snapshots_root / f".step_{step}.tmp"
        if self.acc.is_main_process:
            if not final.exists():
                if temporary.exists() or temporary.is_symlink():
                    raise FileExistsError(
                        "incomplete diagnostic snapshot must be cleared by the "
                        f"fail-closed launcher before resume: {temporary}"
                    )
                snapshots_root.mkdir(parents=True, exist_ok=True)
                temporary.mkdir(parents=True)
        self.acc.wait_for_everyone()
        if final.exists():
            validate_completed_diagnostic_snapshot(final)
            if self.acc.is_main_process:
                self.acc.print(
                    "[snapshot] verified existing paired full-resume + clean HF -> "
                    f"{final}"
                )
            return
        resume_checkpoint = self.save_resume_checkpoint(temporary / "resume")
        self.export_hf(temporary / "hf")
        if self.acc.is_main_process:
            resume_state = json.loads(
                (resume_checkpoint / "trainer_state.json").read_text(
                    encoding="utf-8"
                )
            )
            hf_state = json.loads(
                (
                    temporary
                    / "hf"
                    / "interleaved_training_state.json"
                ).read_text(encoding="utf-8")
            )
            if resume_state != hf_state:
                raise RuntimeError(
                    "diagnostic snapshot resume and HF states differ"
                )
            if int(resume_state.get("global_step", -1)) != step:
                raise RuntimeError(
                    "diagnostic snapshot state has the wrong global step"
                )
            interval_metrics = resume_state.get(
                "diagnostic_last_ce_interval"
            )
            if (
                not isinstance(interval_metrics, Mapping)
                or interval_metrics.get("end_step") != step
                or int(interval_metrics.get("pretrain_token_count", 0)) <= 0
                or int(interval_metrics.get("sft_token_count", 0)) <= 0
            ):
                raise RuntimeError(
                    "diagnostic snapshot lacks a complete PT/SFT CE interval"
                )
            write_diagnostic_snapshot_completion_marker(
                temporary,
                global_step=step,
                interval_unweighted_ce=interval_metrics,
            )
            publish_diagnostic_snapshot_directory(
                temporary,
                final,
                allow_serialized_fallback=True,
            )
            self.acc.print(
                "[snapshot] paired full-resume + clean HF -> "
                f"{final}"
            )
        self.acc.wait_for_everyone()
        validate_completed_diagnostic_snapshot(final)

    def export_hf(self, output: os.PathLike[str] | str) -> None:
        """Publish an authenticated immutable FP32 Hugging Face export."""

        output_path = pathlib.Path(output)
        if output_path.exists() or output_path.is_symlink():
            self._export_hf_unlocked(output_path)
            return
        lock_root = getattr(self, "output_dir", output_path.parent)
        with self._checkpoint_volume_write_guard(lock_root):
            self._export_hf_unlocked(output)

    def _export_hf_unlocked(self, output: os.PathLike[str] | str) -> None:
        """Publish one HF export while the Volume lock is held."""

        output = pathlib.Path(output)
        temporary = output.with_name(f".{output.name}.tmp")
        self.acc.wait_for_everyone()
        state_dict = self.acc.get_state_dict(self.model)
        assert_fp32_state_dict(
            state_dict,
            where="before clean HF export",
        )
        if self.acc.is_main_process:
            if not output.exists():
                if temporary.exists() or temporary.is_symlink():
                    raise FileExistsError(
                        "incomplete HF export staging directory must be diagnosed, "
                        f"not overwritten: {temporary}"
                    )
                output.parent.mkdir(parents=True, exist_ok=True)
                temporary.mkdir(parents=False, exist_ok=False)
        self.acc.wait_for_everyone()
        if output.exists():
            validated = validate_completed_hf_export(output)
            if validated["state"] != self._trainer_state():
                raise FileExistsError(
                    "immutable HF export exists for different trainer state: "
                    f"{output}"
                )
            if self.acc.is_main_process:
                self.acc.print(
                    f"[export] verified existing HF checkpoint -> {output}"
                )
            return
        if self.acc.is_main_process:
            base_model = self.acc.unwrap_model(self.model)
            base_model.config.use_cache = True
            base_model.config.vocab_size = self.vocab_size
            base_model.config.max_position_embeddings = self.context_length
            call_env_id = getattr(self.tokenizer, "call_env_id", None)
            env_token_id = call_env_id() if call_env_id is not None else None
            base_model.config.env_token_id = env_token_id
            base_model.save_pretrained(
                temporary,
                state_dict=state_dict,
                safe_serialization=True,
            )
            save_hf_tokenizer(
                tokenizer=self.tokenizer,
                tokcfg=self.tokcfg,
                save_directory=temporary,
                model_max_length=self.context_length,
                env_id=env_token_id,
            )
            generation_config = {
                "bos_token_id": self.tokenizer.bos_id(),
                "eos_token_id": self.tokenizer.eos_id(),
                "pad_token_id": self.tokenizer.pad_id(),
                "max_new_tokens": self.context_length,
                "do_sample": True,
                "temperature": 1.0,
            }
            (temporary / "generation_config.json").write_text(
                json.dumps(generation_config, indent=2),
                encoding="utf-8",
            )
            (temporary / "interleaved_training_state.json").write_text(
                json.dumps(self._trainer_state(), indent=2, sort_keys=True),
                encoding="utf-8",
            )
            write_hf_export_completion_marker(
                temporary,
                global_step=self.global_step,
            )
            publish_hf_export_directory(
                temporary,
                output,
                allow_serialized_fallback=True,
            )
            self.acc.print(f"[export] clean HF checkpoint -> {output}")
        self.acc.wait_for_everyone()
        validated = validate_completed_hf_export(output)
        if validated["state"] != self._trainer_state():
            raise RuntimeError(
                f"published HF export trainer state drifted: {output}"
            )


__all__ = [
    "BENCHMARK_OUTPUT_ROOT",
    "COMPUTE_DTYPE",
    "DETERMINISM_CONTRACT",
    "EXPECTED_CONTEXT_LENGTH",
    "EXPECTED_GLOBAL_BATCH_SIZE",
    "EXPECTED_LOCAL_BATCH_SIZE",
    "EXPECTED_PARAMETER_COUNT",
    "EXPECTED_TOKEN_POSITIONS_PER_UPDATE",
    "EXPECTED_VOCAB_SIZE",
    "EXPECTED_WORLD_SIZE",
    "MASTER_PARAMETER_DTYPE",
    "PINNED_FLASH_ATTENTION_VERSION",
    "PRECISION_CONTRACT",
    "STATE_SCHEMA_VERSION",
    "SUPPORTED_ATTENTION_BACKENDS",
    "SUPPORTED_COMPILE_MODES",
    "ArcPosition",
    "ExactArcCosine",
    "InterleavedHFTrainer",
    "assert_fp32_gradients",
    "assert_finite_gradient_norm",
    "assert_fp32_master_parameters",
    "assert_fp32_optimizer",
    "assert_fp32_state_dict",
    "authenticated_weights_only_identity",
    "build_interleaved_qwen_config",
    "build_interleaved_qwen_model",
    "causal_ce_sum",
    "configure_deterministic_training",
    "globally_normalized_backward_loss",
    "load_weights_only",
    "normalize_attention_backend",
    "normalize_compile_mode",
    "normalize_sft_loss_weight",
    "register_bf16_output_head_assertion",
    "resolve_arc_steps",
    "sha256_file",
    "validate_resume_state",
    "validate_topology",
    "weighted_causal_ce_sum",
]
