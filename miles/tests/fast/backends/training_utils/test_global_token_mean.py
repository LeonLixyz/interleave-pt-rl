from argparse import Namespace

import pytest
import torch

import miles.backends.training_utils.loss as loss_module
from miles.backends.training_utils.data import DataIterator, peek_supervised_token_count
from miles.backends.training_utils.loss import loss_function
from miles.backends.training_utils.parallel import GroupInfo, ParallelState, set_parallel_state


def _set_single_process_parallel_state() -> None:
    group = GroupInfo(rank=0, size=1, group=None)
    set_parallel_state(
        ParallelState(
            intra_dp=group,
            intra_dp_cp=group,
            cp=group,
            tp=group,
            pp=group,
            ep=group,
            etp=group,
        )
    )


def _args() -> Namespace:
    return Namespace(
        calculate_per_token_loss=True,
        policy_loss_agg_mode="token-mean",
        qkv_format="thd",
        recompute_loss_function=False,
        allgather_cp=False,
        use_dynamic_global_batch_size=False,
        global_batch_size=2,
        true_on_policy_mode=False,
    )


def _batch(mask: torch.Tensor) -> dict:
    token_count = mask.numel()
    return {
        "loss_masks": [mask],
        "total_lengths": [token_count + 1],
        "response_lengths": [token_count],
    }


def test_global_token_mean_gradient_is_independent_of_microbatch_partition(monkeypatch):
    _set_single_process_parallel_state()
    args = _args()

    def element_sum_loss(_args, batch, logits, _reducer):
        mask = batch["loss_masks"][0].to(logits)
        loss_sum = (logits.flatten()[: mask.numel()] * mask).sum()
        return loss_sum, {"loss": loss_sum.detach()}

    monkeypatch.setattr(loss_module, "get_loss_function", lambda _args: element_sum_loss)

    full_logits = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)
    full_mask = torch.ones(5)
    full_loss, _, _ = loss_function(
        args,
        _batch(full_mask),
        num_microbatches=1,
        logits=full_logits,
        global_token_count=torch.tensor(5.0),
    )
    full_loss.backward()
    full_gradient = full_logits.grad.clone()

    partitioned_logits = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)
    for start, end in ((0, 1), (1, 5)):
        microbatch_loss, _, _ = loss_function(
            args,
            _batch(torch.ones(end - start)),
            num_microbatches=2,
            logits=partitioned_logits[start:end],
            global_token_count=torch.tensor(5.0),
        )
        microbatch_loss.backward()

    torch.testing.assert_close(partitioned_logits.grad, full_gradient, rtol=0, atol=0)
    torch.testing.assert_close(full_gradient, torch.full((5,), 0.2), rtol=0, atol=0)


def test_token_mean_mode_alone_selects_global_supervised_token_mean(monkeypatch):
    """The explicit aggregation mode is authoritative without a legacy flag."""

    _set_single_process_parallel_state()
    args = _args()
    args.calculate_per_token_loss = False

    def element_sum_loss(_args, batch, logits, _reducer):
        mask = batch["loss_masks"][0].to(logits)
        loss_sum = (logits.flatten()[: mask.numel()] * mask).sum()
        return loss_sum, {"loss": loss_sum.detach()}

    monkeypatch.setattr(loss_module, "get_loss_function", lambda _args: element_sum_loss)
    logits = torch.ones(5, requires_grad=True)
    for start, end in ((0, 1), (1, 5)):
        loss, _, _ = loss_function(
            args,
            _batch(torch.ones(end - start)),
            num_microbatches=2,
            logits=logits[start:end],
            global_token_count=torch.tensor(5.0),
        )
        loss.backward()
    torch.testing.assert_close(logits.grad, torch.full((5,), 0.2), rtol=0, atol=0)


def test_token_mean_fails_closed_without_update_wide_denominator(monkeypatch):
    _set_single_process_parallel_state()
    args = _args()
    monkeypatch.setattr(
        loss_module,
        "get_loss_function",
        lambda _args: lambda _a, _b, logits, _r: (logits.sum(), {"loss": logits.detach().sum()}),
    )

    with pytest.raises(RuntimeError, match="complete optimizer update"):
        loss_function(args, _batch(torch.ones(2)), 1, torch.ones(2, requires_grad=True))


def test_global_token_mean_scales_each_rank_by_dp_global_denominator(monkeypatch):
    """FSDP averages DP gradients, so each local sum needs the DP-size factor."""
    _set_single_process_parallel_state()
    group = GroupInfo(rank=0, size=2, group=None)
    set_parallel_state(
        ParallelState(
            intra_dp=group,
            intra_dp_cp=group,
            cp=GroupInfo(rank=0, size=1, group=None),
            tp=GroupInfo(rank=0, size=1, group=None),
            pp=GroupInfo(rank=0, size=1, group=None),
            ep=GroupInfo(rank=0, size=1, group=None),
            etp=GroupInfo(rank=0, size=1, group=None),
        )
    )

    monkeypatch.setattr(
        loss_module,
        "get_loss_function",
        lambda _args: lambda _a, _b, logits, _r: (logits.sum(), {"loss": logits.detach().sum()}),
    )

    logits = torch.ones(3, requires_grad=True)
    loss, _, _ = loss_function(
        _args(),
        _batch(torch.ones(3)),
        1,
        logits,
        global_token_count=torch.tensor(5.0),
    )
    loss.backward()
    torch.testing.assert_close(logits.grad, torch.full((3,), 0.4), rtol=0, atol=0)


def test_token_count_peek_preserves_nonzero_iterator_offset_across_multiple_steps():
    iterator = DataIterator(
        {"loss_masks": [torch.ones(length) for length in (1, 2, 3, 4)]},
        micro_batch_size=1,
    )
    iterator.offset = 2

    assert peek_supervised_token_count(iterator, 2).item() == 7
    assert iterator.offset == 2
    next_batch = iterator.get_next(["loss_masks"])
    assert next_batch["loss_masks"][0].numel() == 3
