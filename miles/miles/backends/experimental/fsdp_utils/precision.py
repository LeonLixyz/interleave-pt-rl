"""Precision invariants for the FSDP training backend.

FSDP2's :class:`MixedPrecisionPolicy` only controls the dtype of the
unsharded parameters used for forward/backward and gradient reduction.  The
optimizer updates the sharded parameter in its original dtype.  We therefore
keep the original parameter and Adam state in FP32 and use BF16 only through
the FSDP mixed-precision policy.
"""

from __future__ import annotations

from collections import Counter
import math
import numbers
from typing import Any

import torch


MASTER_PARAMETER_DTYPE = torch.float32
GRADIENT_REDUCTION_DTYPE = torch.float32


def compute_dtype(args) -> torch.dtype:
    """Return the forward/backward dtype selected by the FSDP CLI."""
    return torch.float16 if getattr(args, "fp16", False) else torch.bfloat16


def dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def upcast_model_to_fp32_(model: torch.nn.Module) -> None:
    """Upcast floating parameters and buffers before FSDP owns the model.

    Calling this even when ``from_pretrained`` was asked for FP32 is
    intentional.  It makes the precision contract independent of a
    checkpoint's ``config.json`` dtype and of Transformers loading defaults.
    """
    for param in model.parameters():
        if param.is_floating_point():
            param.data = param.data.to(dtype=MASTER_PARAMETER_DTYPE)

    # Floating buffers used in model math (for example rotary inv_freq) should
    # also begin in full precision. Integer/boolean bookkeeping is preserved.
    for buffer in model.buffers():
        if buffer.is_floating_point():
            buffer.data = buffer.data.to(dtype=MASTER_PARAMETER_DTYPE)


def _parameter_dtype_counts(model: torch.nn.Module) -> Counter[torch.dtype]:
    return Counter(param.dtype for param in model.parameters() if param.is_floating_point())


def assert_fp32_master_parameters(model: torch.nn.Module, *, where: str) -> None:
    """Fail if any floating model parameter is not stored in FP32."""
    mismatches = [
        (name, param.dtype)
        for name, param in model.named_parameters()
        if param.is_floating_point() and param.dtype != MASTER_PARAMETER_DTYPE
    ]
    if mismatches:
        examples = ", ".join(f"{name}={dtype_name(dtype)}" for name, dtype in mismatches[:8])
        counts = ", ".join(
            f"{dtype_name(dtype)}={count}"
            for dtype, count in sorted(_parameter_dtype_counts(model).items(), key=lambda item: str(item[0]))
        )
        raise RuntimeError(
            f"FSDP precision contract violation at {where}: optimizer-facing model parameters must be float32; "
            f"found {len(mismatches)} non-float32 tensors ({counts}). Examples: {examples}"
        )


def assert_fp32_gradients(model: torch.nn.Module, *, where: str) -> None:
    """Fail if any materialized floating gradient is not FP32."""
    gradients = [
        (name, param.grad)
        for name, param in model.named_parameters()
        if param.grad is not None and param.grad.is_floating_point()
    ]
    if not gradients:
        raise RuntimeError(
            f"FSDP precision contract violation at {where}: no floating gradients were materialized"
        )

    mismatches = [(name, grad.dtype) for name, grad in gradients if grad.dtype != GRADIENT_REDUCTION_DTYPE]
    if mismatches:
        examples = ", ".join(f"{name}={dtype_name(dtype)}" for name, dtype in mismatches[:8])
        raise RuntimeError(
            f"FSDP precision contract violation at {where}: accumulated/reduced gradients must be float32; "
            f"found {len(mismatches)} non-float32 tensors. Examples: {examples}"
        )


def assert_fp32_optimizer_state(
    optimizer: torch.optim.Optimizer,
    *,
    where: str,
    require_initialized: bool,
) -> None:
    """Fail if initialized Adam state contains non-FP32 floating tensors."""
    if require_initialized and not optimizer.state:
        raise RuntimeError(
            f"FSDP precision contract violation at {where}: expected initialized FP32 optimizer state, but it is empty"
        )

    mismatches: list[tuple[str, torch.dtype]] = []
    missing_moments: list[str] = []
    for state_index, state in enumerate(optimizer.state.values()):
        if state and ("exp_avg" not in state or "exp_avg_sq" not in state):
            missing_moments.append(str(state_index))
        for key, value in state.items():
            if isinstance(value, torch.Tensor) and value.is_floating_point() and value.dtype != torch.float32:
                mismatches.append((f"state[{state_index}].{key}", value.dtype))

    if missing_moments:
        raise RuntimeError(
            f"FSDP precision contract violation at {where}: Adam state entries are missing exp_avg/exp_avg_sq "
            f"for indices {', '.join(missing_moments[:8])}"
        )
    if mismatches:
        examples = ", ".join(f"{name}={dtype_name(dtype)}" for name, dtype in mismatches[:8])
        raise RuntimeError(
            f"FSDP precision contract violation at {where}: all floating Adam state tensors must be float32; "
            f"found {len(mismatches)} non-float32 tensors. Examples: {examples}"
        )


def assert_fp32_training_state(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    where: str,
    require_optimizer_state: bool,
) -> None:
    assert_fp32_master_parameters(model, where=where)
    assert_fp32_optimizer_state(optimizer, where=where, require_initialized=require_optimizer_state)


def assert_finite_training_value(value: Any, *, name: str, where: str) -> None:
    """Fail before an optimizer step if a loss/metric contains NaN or Inf."""

    if isinstance(value, torch.Tensor):
        if value.is_floating_point() and not bool(torch.isfinite(value).all().item()):
            raise RuntimeError(
                f"Nonfinite training value at {where}: {name} contains NaN or Inf"
            )
        return
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            raise RuntimeError(
                f"Nonfinite training value at {where}: {name}={value!r}"
            )
        return
    if isinstance(value, dict):
        for child_name, child in value.items():
            assert_finite_training_value(
                child,
                name=f"{name}.{child_name}",
                where=where,
            )
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            assert_finite_training_value(
                child,
                name=f"{name}[{index}]",
                where=where,
            )


def validate_policy_logging_wrapper(
    log_dict: dict[str, Any],
    *,
    required_metrics: tuple[str, ...] = ("ppo_kl", "entropy_loss"),
    where: str,
) -> None:
    """Validate finite policy metrics in loss_function's packed log format."""

    keys = log_dict.get("keys")
    values = log_dict.get("values")
    if not isinstance(keys, list) or not isinstance(values, torch.Tensor):
        raise RuntimeError(
            f"Malformed policy logging wrapper at {where}: expected keys list and values tensor"
        )
    if values.ndim != 1 or values.numel() != len(keys) + 1:
        raise RuntimeError(
            f"Malformed policy logging wrapper at {where}: "
            f"len(keys)={len(keys)} values_shape={tuple(values.shape)}"
        )
    duplicates = {key for key in keys if keys.count(key) > 1}
    if duplicates:
        raise RuntimeError(
            f"Duplicate policy metrics at {where}: {sorted(duplicates)}"
        )
    missing = [name for name in required_metrics if name not in keys]
    if missing:
        raise RuntimeError(
            f"Training loss did not expose required finite metrics at {where}: {missing}"
        )
    assert_finite_training_value(
        values,
        name="policy_logging_values",
        where=where,
    )


def precision_contract(args) -> dict[str, str]:
    """Serializable precision contract stored with resumable checkpoints."""
    return {
        "master_parameter_dtype": dtype_name(MASTER_PARAMETER_DTYPE),
        "optimizer_state_dtype": dtype_name(torch.float32),
        "forward_backward_dtype": dtype_name(compute_dtype(args)),
        "gradient_reduction_dtype": dtype_name(GRADIENT_REDUCTION_DTYPE),
    }


def rollout_weight_dtype(args) -> torch.dtype:
    """Return the explicitly selected in-memory SGLang weight dtype."""
    name = getattr(args, "sglang_dtype", None)
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise RuntimeError(
        "Rollout inference precision must be explicit: --sglang-dtype must be bfloat16 or float16"
    )


def cast_tensor_for_rollout(tensor: torch.Tensor, args) -> torch.Tensor:
    """Create a transient inference tensor without mutating FP32 training state."""
    if tensor.is_floating_point():
        return tensor.to(dtype=rollout_weight_dtype(args))
    return tensor


def assert_rollout_tensor_dtypes(named_tensors, args) -> None:
    expected_dtype = rollout_weight_dtype(args)
    mismatches = [
        (name, tensor.dtype)
        for name, tensor in named_tensors
        if tensor.is_floating_point() and tensor.dtype != expected_dtype
    ]
    if mismatches:
        examples = ", ".join(f"{name}={dtype}" for name, dtype in mismatches[:8])
        raise RuntimeError(
            f"Rollout weight-sync precision violation: expected all floating tensors to be {expected_dtype}; "
            f"found {len(mismatches)} mismatches. Examples: {examples}"
        )
