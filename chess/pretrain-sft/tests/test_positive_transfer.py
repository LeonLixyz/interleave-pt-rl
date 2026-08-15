import json

import pytest
import torch
import torch.nn.functional as F

from training.positive_replay import canonical_json, token_ids_sha256
from training.positive_transfer import (
    IGNORE_INDEX,
    DeterministicDistributedBatchSampler,
    PositiveReplayDataset,
    PositiveTransferConfig,
    _sha256_json,
    _transfer_contract,
    collate_positive_replay,
    forward_kl_loss_sum,
    globally_normalized_ddp_loss,
    hard_sft_loss_sum,
    validate_transfer_resume_state,
)


def _replay_row(index=0):
    prompt = [1, 2]
    response = [3, 4, 5]
    return {
        "schema_version": 1,
        "prompt": "prompt",
        "response": "response",
        "prompt_token_ids": prompt,
        "response_token_ids": response,
        "response_loss_mask": [1, 0, 1],
        "token_ids_sha256": token_ids_sha256(prompt, response),
        "prompt_response_sha256": str(index),
        "group_index": index,
        "sample_index": index,
    }


def test_collator_aligns_first_response_target_and_masks_environment_tokens():
    row = _replay_row()
    row["dataset_index"] = 7

    batch = collate_positive_replay(
        [row, {"is_padding": True}],
        pad_token_id=0,
    )

    assert batch["input_ids"][0].tolist() == [1, 2, 3, 4]
    assert batch["labels"][0].tolist() == [IGNORE_INDEX, 3, IGNORE_INDEX, 5]
    assert batch["model_owned_mask"][0].tolist() == [False, True, False, True]
    assert batch["attention_mask"][0].tolist() == [True, True, True, True]
    assert batch["dataset_indices"].tolist() == [7, -1]
    assert batch["padding_records"] == 1


def test_distributed_sampler_exposes_each_row_once_and_uses_ignore_padding():
    samplers = [
        DeterministicDistributedBatchSampler(
            13,
            local_batch_size=2,
            rank=rank,
            world_size=3,
            seed=42,
        )
        for rank in range(3)
    ]
    per_rank = [list(iter(sampler)) for sampler in samplers]
    assert all(len(batches) == 3 for batches in per_rank)

    global_order = []
    for batch_index in range(3):
        for rank in range(3):
            global_order.extend(per_rank[rank][batch_index])
    real = [index for index in global_order if index >= 0]
    assert sorted(real) == list(range(13))
    assert global_order.count(-1) == 5
    assert per_rank == [
        list(
            DeterministicDistributedBatchSampler(
                13,
                local_batch_size=2,
                rank=rank,
                world_size=3,
                seed=42,
            )
        )
        for rank in range(3)
    ]


def test_hard_sft_and_forward_kl_are_sums_over_owned_positions_only():
    torch.manual_seed(0)
    student = torch.randn(1, 3, 5, requires_grad=True)
    teacher = torch.randn(1, 3, 5)
    labels = torch.tensor([[2, IGNORE_INDEX, 4]])
    mask = labels.ne(IGNORE_INDEX)

    hard_sum, hard_count = hard_sft_loss_sum(student, labels)
    reference_hard = F.cross_entropy(
        student.reshape(-1, 5),
        labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
        reduction="sum",
    )
    assert hard_count.item() == 2
    assert torch.allclose(hard_sum, reference_hard)

    kl_sum, kl_count = forward_kl_loss_sum(student, teacher, mask)
    teacher_logp = F.log_softmax(teacher, dim=-1)
    reference_tokens = (
        teacher_logp.exp() * (teacher_logp - F.log_softmax(student, dim=-1))
    ).sum(-1)
    assert kl_count.item() == 2
    assert torch.allclose(kl_sum, reference_tokens[mask].sum(), atol=1e-6)

    zero, zero_count = forward_kl_loss_sum(student, student.detach(), mask)
    assert zero_count.item() == 2
    assert abs(zero.item()) < 1e-6


def test_ddp_scaling_compensates_for_gradient_average():
    local_sum = torch.tensor(9.0)
    scaled = globally_normalized_ddp_loss(
        local_sum,
        global_valid_tokens=12,
        world_size=4,
    )
    assert scaled.item() == 3.0
    with pytest.raises(ValueError, match="no model-owned"):
        globally_normalized_ddp_loss(
            local_sum,
            global_valid_tokens=0,
            world_size=4,
        )


def test_indexed_dataset_verifies_checksum_manifest(tmp_path):
    replay = tmp_path / "positive.jsonl"
    encoded = canonical_json(_replay_row()) + "\n"
    replay.write_text(encoded, encoding="utf-8")
    import hashlib

    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"output": {"sha256": digest, "rows": 1}}),
        encoding="utf-8",
    )

    dataset = PositiveReplayDataset(replay, manifest_path=manifest)
    assert len(dataset) == 1
    assert dataset[0]["dataset_index"] == 0
    assert dataset[-1] == {"is_padding": True}

    replay.write_text(encoded + encoded, encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        PositiveReplayDataset(replay, manifest_path=manifest)


def test_transfer_config_enforces_teacher_mode_pairing():
    common = dict(
        student_checkpoint="p1",
        replay_path="positive.jsonl",
        output_dir="out",
        learning_rate=1e-5,
    )
    with pytest.raises(ValueError, match="requires teacher"):
        PositiveTransferConfig(mode="soft_kl", **common).validate()
    with pytest.raises(ValueError, match="must not load"):
        PositiveTransferConfig(
            mode="hard_sft",
            teacher_checkpoint="teacher",
            **common,
        ).validate()


def test_sampler_can_resume_at_an_exact_batch_without_replaying_rows():
    complete = DeterministicDistributedBatchSampler(
        13,
        local_batch_size=2,
        rank=1,
        world_size=3,
        seed=42,
    )
    expected = list(complete)
    resumed = DeterministicDistributedBatchSampler(
        13,
        local_batch_size=2,
        rank=1,
        world_size=3,
        seed=42,
        start_batch=2,
    )
    assert list(resumed) == expected[2:]
    with pytest.raises(ValueError, match="start_batch"):
        resumed.set_start_batch(4)


def test_resume_state_is_bound_to_contract_cursor_and_global_counters():
    config = PositiveTransferConfig(
        mode="hard_sft",
        student_checkpoint="p1",
        replay_path="positive.jsonl",
        replay_manifest="positive.manifest.json",
        output_dir="out",
        learning_rate=1e-5,
        checkpoint_dir="latest",
        run_fingerprint="f" * 64,
    )
    contract = _transfer_contract(
        config,
        replay_sha256="a" * 64,
        replay_manifest_sha256="b" * 64,
        world_size=8,
    )
    state = {
        "schema_version": 1,
        "contract": contract,
        "contract_sha256": _sha256_json(contract),
        "completed_steps": 7,
        "next_epoch": 0,
        "next_batch": 7,
        "counters": {
            "processed_positive_examples": 1176,
            "processed_model_owned_tokens": 12345,
            "padding_records": 0,
            "teacher_forward_examples": 0,
        },
    }
    validate_transfer_resume_state(
        state,
        contract=contract,
        batches_per_epoch=10,
        target_steps=10,
    )
    changed = dict(contract)
    changed["replay_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="contract mismatch"):
        validate_transfer_resume_state(
            state,
            contract=changed,
            batches_per_epoch=10,
            target_steps=10,
        )
    bad_cursor = {**state, "next_batch": 6}
    with pytest.raises(ValueError, match="cursor mismatch"):
        validate_transfer_resume_state(
            bad_cursor,
            contract=contract,
            batches_per_epoch=10,
            target_steps=10,
        )
