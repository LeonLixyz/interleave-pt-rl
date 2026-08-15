from argparse import Namespace

import pytest
import torch

from miles.backends.experimental.fsdp_utils.precision import (
    assert_rollout_tensor_dtypes,
    assert_fp32_gradients,
    assert_fp32_master_parameters,
    assert_fp32_optimizer_state,
    assert_fp32_training_state,
    cast_tensor_for_rollout,
    compute_dtype,
    precision_contract,
    rollout_weight_dtype,
    upcast_model_to_fp32_,
    validate_policy_logging_wrapper,
)


def test_upcast_restores_fp32_master_parameters_without_changing_values():
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.LayerNorm(4)).bfloat16()
    before = {name: param.detach().float().clone() for name, param in model.named_parameters()}

    upcast_model_to_fp32_(model)
    assert_fp32_master_parameters(model, where="unit test")

    for name, param in model.named_parameters():
        assert param.dtype is torch.float32
        torch.testing.assert_close(param, before[name], rtol=0, atol=0)


def test_master_parameter_assertion_rejects_bf16():
    model = torch.nn.Linear(2, 2).bfloat16()
    with pytest.raises(RuntimeError, match="optimizer-facing model parameters must be float32"):
        assert_fp32_master_parameters(model, where="unit test")


def test_gradient_assertion_accepts_fp32_and_rejects_bf16():
    model = torch.nn.Linear(2, 2).float()
    model(torch.ones(1, 2)).sum().backward()
    assert_fp32_gradients(model, where="unit test")

    bad_model = torch.nn.Linear(2, 2).bfloat16()
    bad_model(torch.ones(1, 2, dtype=torch.bfloat16)).sum().backward()
    with pytest.raises(RuntimeError, match="accumulated/reduced gradients must be float32"):
        assert_fp32_gradients(bad_model, where="unit test")


def test_adam_state_is_fp32_with_fp32_master_parameters():
    model = torch.nn.Linear(2, 2).float()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()

    assert_fp32_training_state(
        model,
        optimizer,
        where="unit test",
        require_optimizer_state=True,
    )
    assert all(state["exp_avg"].dtype is torch.float32 for state in optimizer.state.values())
    assert all(state["exp_avg_sq"].dtype is torch.float32 for state in optimizer.state.values())


def test_optimizer_state_assertion_rejects_bf16_moments():
    model = torch.nn.Linear(2, 2).float()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    first_state = next(iter(optimizer.state.values()))
    first_state["exp_avg"] = first_state["exp_avg"].bfloat16()

    with pytest.raises(RuntimeError, match="all floating Adam state tensors must be float32"):
        assert_fp32_optimizer_state(optimizer, where="unit test", require_initialized=True)


def test_optimizer_state_assertion_fails_closed_when_state_expected_but_empty():
    optimizer = torch.optim.AdamW(torch.nn.Linear(2, 2).parameters(), lr=1e-5)
    with pytest.raises(RuntimeError, match="expected initialized FP32 optimizer state"):
        assert_fp32_optimizer_state(optimizer, where="unit test", require_initialized=True)


@pytest.mark.parametrize(
    ("fp16", "expected_compute", "expected_name"),
    [(False, torch.bfloat16, "bfloat16"), (True, torch.float16, "float16")],
)
def test_precision_contract_uses_fp32_training_state_and_low_precision_compute(
    fp16,
    expected_compute,
    expected_name,
):
    args = Namespace(fp16=fp16)
    assert compute_dtype(args) is expected_compute
    assert precision_contract(args) == {
        "master_parameter_dtype": "float32",
        "optimizer_state_dtype": "float32",
        "forward_backward_dtype": expected_name,
        "gradient_reduction_dtype": "float32",
    }


@pytest.mark.parametrize(
    ("fp16", "expected"),
    [(False, torch.bfloat16), (True, torch.float16)],
)
def test_rollout_weight_cast_is_transient_and_preserves_nonfloating_tensors(fp16, expected):
    args = Namespace(sglang_dtype="float16" if fp16 else "bfloat16")
    source = torch.tensor([0.1234567], dtype=torch.float32)
    integer_buffer = torch.tensor([1, 2], dtype=torch.int64)

    inference = cast_tensor_for_rollout(source, args)

    assert rollout_weight_dtype(args) is expected
    assert inference.dtype is expected
    assert source.dtype is torch.float32
    assert cast_tensor_for_rollout(integer_buffer, args) is integer_buffer


def test_rollout_weight_dtype_assertion_rejects_fp32_sync_payload():
    with pytest.raises(RuntimeError, match="Rollout weight-sync precision violation"):
        assert_rollout_tensor_dtypes(
            [("model.weight", torch.ones(2, dtype=torch.float32))],
            Namespace(sglang_dtype="bfloat16"),
        )


def test_rollout_weight_dtype_fails_closed_without_explicit_selection():
    with pytest.raises(RuntimeError, match="must be explicit"):
        rollout_weight_dtype(Namespace())


def test_policy_logging_wrapper_requires_finite_ppo_kl_and_entropy():
    validate_policy_logging_wrapper(
        {
            "keys": ["loss", "ppo_kl", "entropy_loss"],
            "values": torch.tensor([8.0, 1.0, 0.0, 0.0]),
        },
        where="unit test",
    )


@pytest.mark.parametrize(
    "log_dict",
    [
        {
            "keys": ["loss", "entropy_loss"],
            "values": torch.tensor([8.0, 1.0, 0.0]),
        },
        {
            "keys": ["loss", "ppo_kl", "entropy_loss"],
            "values": torch.tensor([8.0, 1.0, float("nan"), 0.0]),
        },
        {
            "keys": ["loss", "ppo_kl", "entropy_loss"],
            "values": torch.tensor([8.0, 1.0, 0.0, float("inf")]),
        },
    ],
)
def test_policy_logging_wrapper_rejects_missing_or_nonfinite_metrics(log_dict):
    with pytest.raises(RuntimeError):
        validate_policy_logging_wrapper(log_dict, where="unit test")
