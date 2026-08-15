from __future__ import annotations

import copy
import importlib
import sys
import types
from types import SimpleNamespace

import pytest


class _FakeSample:
    def __init__(self, source_row_index: int, **extra_metadata):
        self.metadata = {
            "source_row_index": source_row_index,
            **extra_metadata,
        }
        self.group_index = None
        self.index = None


class _FakeDataset:
    def __init__(self, source_row_indices, **extra_metadata):
        self.samples = [
            _FakeSample(source_row_index, **extra_metadata)
            for source_row_index in source_row_indices
        ]

    def __len__(self) -> int:
        return len(self.samples)


class _FakeRolloutDataSource:
    def __init__(self, args):
        self.args = args
        self.dataset = (
            _FakeDataset(
                args.source_row_indices,
                **getattr(args, "extra_metadata", {}),
            )
            if args.rollout_global_dataset
            else None
        )
        self.epoch_id = 0
        self.sample_offset = 0
        self.sample_group_index = 0
        self.sample_index = 0

    def get_samples(self, num_samples: int):
        if self.sample_offset + num_samples <= len(self.dataset):
            prompts = self.dataset.samples[
                self.sample_offset : self.sample_offset + num_samples
            ]
            self.sample_offset += num_samples
        else:
            prompts = self.dataset.samples[self.sample_offset :]
            remaining = num_samples - len(prompts)
            self.epoch_id += 1
            prompts += self.dataset.samples[:remaining]
            self.sample_offset = remaining

        groups = []
        for prompt in prompts:
            group = []
            for _ in range(self.args.n_samples_per_prompt):
                sample = copy.deepcopy(prompt)
                sample.group_index = self.sample_group_index
                sample.index = self.sample_index
                self.sample_index += 1
                group.append(sample)
            self.sample_group_index += 1
            groups.append(group)
        return groups


@pytest.fixture
def exhaustive_source_module(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "CHESS_RL_MILES_DETERMINISTIC_SEED_MODE",
        "sample-index",
    )
    monkeypatch.setenv(
        "CHESS_RL_MILES_ARTIFACT_ROOT",
        str(tmp_path / "artifacts"),
    )
    miles = types.ModuleType("miles")
    miles.__path__ = []
    rollout = types.ModuleType("miles.rollout")
    rollout.__path__ = []
    data_source = types.ModuleType("miles.rollout.data_source")
    data_source.RolloutDataSource = _FakeRolloutDataSource
    monkeypatch.setitem(sys.modules, "miles", miles)
    monkeypatch.setitem(sys.modules, "miles.rollout", rollout)
    monkeypatch.setitem(
        sys.modules, "miles.rollout.data_source", data_source
    )
    monkeypatch.delitem(
        sys.modules,
        "chess_rl_miles.exhaustive_data_source",
        raising=False,
    )
    return importlib.import_module(
        "chess_rl_miles.exhaustive_data_source"
    )


def _args(**overrides):
    values = {
        "rollout_global_dataset": True,
        "debug_rollout_only": True,
        "sglang_enable_deterministic_inference": True,
        "partial_rollout": False,
        "dynamic_sampling_filter_path": None,
        "use_fault_tolerance": False,
        "use_wandb": False,
        "sglang_skip_tokenizer_init": True,
        "use_miles_router": True,
        "rollout_batch_size": 2,
        "over_sampling_batch_size": 2,
        "n_samples_per_prompt": 16,
        "num_rollout": 2,
        "source_row_indices": [17, 3, 99, 101],
        "rollout_max_prompt_len": 1024,
        "rollout_max_response_len": 2560,
        "rollout_max_context_len": 3072,
        "rollout_temperature": 1.0,
        "rollout_top_p": 1.0,
        "rollout_function_path": (
            "chess_rl_miles.batched_rollout.ChessBatchedRolloutFn"
        ),
        "custom_generate_function_path": (
            "chess_rl_miles.rollout.generate"
        ),
        "custom_rm_path": "chess_rl_miles.reward.reward_func",
        "custom_rollout_log_function_path": (
            "chess_rl_miles.io.log_rollout_data"
        ),
        "extra_metadata": {},
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_exhaustive_source_consumes_exact_shard_and_assigns_global_ids(
    exhaustive_source_module,
):
    source = (
        exhaustive_source_module.StrictExhaustiveRolloutDataSource(
            _args()
        )
    )
    batches = [source.get_samples(2), source.get_samples(2)]
    groups = [group for batch in batches for group in batch]

    assert [group[0].metadata["source_row_index"] for group in groups] == [
        17,
        3,
        99,
        101,
    ]
    assert all(len(group) == 16 for group in groups)
    assert source.sample_offset == 4
    assert source.epoch_id == 0
    assert source._strict_consumed_prompts == 4
    assert source._strict_get_calls == 2

    observed_sample_ids = []
    for group in groups:
        source_row_index = group[0].metadata["source_row_index"]
        assert {sample.group_index for sample in group} == {
            source_row_index
        }
        for sample_slot, sample in enumerate(group):
            expected_sample_index = source_row_index * 16 + sample_slot
            assert sample.index == expected_sample_index
            assert (
                sample.metadata["pass_at_16_sample_slot"]
                == sample_slot
            )
            assert (
                sample.metadata["pass_at_16_sample_index"]
                == expected_sample_index
            )
            observed_sample_ids.append(sample.index)

    assert len(observed_sample_ids) == len(set(observed_sample_ids)) == 64
    with pytest.raises(RuntimeError, match="refuses dataset wrap"):
        source.get_samples(2)
    with pytest.raises(RuntimeError, match="only permits"):
        source.get_samples(1)


def test_exhaustive_source_refuses_requeue(exhaustive_source_module):
    source = (
        exhaustive_source_module.StrictExhaustiveRolloutDataSource(
            _args()
        )
    )
    source.add_samples([])
    with pytest.raises(RuntimeError, match="refuses aborted-sample requeue"):
        source.add_samples([[object()]])


@pytest.mark.parametrize(
    "overrides,error",
    [
        ({"debug_rollout_only": False}, "debug_rollout_only"),
        (
            {"sglang_enable_deterministic_inference": False},
            "deterministic_inference",
        ),
        ({"partial_rollout": True}, "partial_rollout"),
        ({"dynamic_sampling_filter_path": "filter"}, "dynamic sampling"),
        ({"use_fault_tolerance": True}, "fault tolerance"),
        ({"use_wandb": True}, "W&B telemetry"),
        (
            {"sglang_skip_tokenizer_init": False},
            "sglang_skip_tokenizer_init",
        ),
        ({"use_miles_router": False}, "use_miles_router"),
        ({"rollout_batch_size": 0}, "rollout_batch_size"),
        ({"over_sampling_batch_size": 4}, "over_sampling_batch_size"),
        ({"n_samples_per_prompt": 8}, "n_samples_per_prompt"),
        ({"num_rollout": 0}, "num_rollout"),
        (
            {"source_row_indices": [17, 3, 99]},
            "post-tokenization dataset size",
        ),
        ({"rollout_max_prompt_len": 512}, "rollout_max_prompt_len"),
        (
            {"rollout_max_response_len": 2048},
            "rollout_max_response_len",
        ),
        ({"rollout_max_context_len": 4096}, "rollout_max_context_len"),
        ({"rollout_temperature": 0.7}, "rollout_temperature"),
        ({"rollout_top_p": 0.95}, "rollout_top_p"),
        ({"rollout_function_path": "wrong"}, "rollout_function_path"),
        (
            {"custom_generate_function_path": "wrong"},
            "custom_generate_function_path",
        ),
        ({"custom_rm_path": "wrong"}, "custom_rm_path"),
        (
            {"custom_rollout_log_function_path": "wrong"},
            "custom_rollout_log_function_path",
        ),
    ],
)
def test_exhaustive_source_rejects_contract_drift(
    exhaustive_source_module,
    overrides,
    error,
):
    with pytest.raises(ValueError, match=error):
        exhaustive_source_module.StrictExhaustiveRolloutDataSource(
            _args(**overrides)
        )


def test_exhaustive_source_requires_unique_sample_seed_mode(
    exhaustive_source_module,
    monkeypatch,
):
    monkeypatch.setenv(
        "CHESS_RL_MILES_DETERMINISTIC_SEED_MODE",
        "sibling-index",
    )
    with pytest.raises(ValueError, match="seed mode must equal sample-index"):
        exhaustive_source_module.StrictExhaustiveRolloutDataSource(
            _args()
        )


def test_exhaustive_source_requires_absolute_artifact_root(
    exhaustive_source_module,
    monkeypatch,
):
    monkeypatch.setenv("CHESS_RL_MILES_ARTIFACT_ROOT", "relative")
    with pytest.raises(ValueError, match="ARTIFACT_ROOT"):
        exhaustive_source_module.StrictExhaustiveRolloutDataSource(
            _args()
        )


@pytest.mark.parametrize(
    "source_row_indices,extra_metadata,error",
    [
        ([17, 3, 17, 101], {}, "must be unique"),
        ([17, True, 99, 101], {}, "non-negative integer"),
        (
            [17, 3, 99, 101],
            {"pass_at_16_sample_slot": 0},
            "reserved pass@16 identity",
        ),
    ],
)
def test_exhaustive_source_rejects_source_identity_drift(
    exhaustive_source_module,
    source_row_indices,
    extra_metadata,
    error,
):
    with pytest.raises(ValueError, match=error):
        exhaustive_source_module.StrictExhaustiveRolloutDataSource(
            _args(
                source_row_indices=source_row_indices,
                extra_metadata=extra_metadata,
            )
        )
