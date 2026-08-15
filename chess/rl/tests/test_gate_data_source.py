from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace

import pytest


class _FakeDataset:
    def __init__(self, size: int):
        self.samples = list(range(size))

    def __len__(self) -> int:
        return len(self.samples)


class _FakeRolloutDataSource:
    def __init__(self, args):
        self.args = args
        self.dataset = (
            _FakeDataset(args.dataset_size)
            if args.rollout_global_dataset
            else None
        )
        self.epoch_id = 0
        self.sample_offset = 0

    def get_samples(self, num_samples: int):
        if self.sample_offset + num_samples <= len(self.dataset):
            values = self.dataset.samples[
                self.sample_offset : self.sample_offset + num_samples
            ]
            self.sample_offset += num_samples
            return values
        values = self.dataset.samples[self.sample_offset :]
        remaining = num_samples - len(values)
        self.epoch_id += 1
        values += self.dataset.samples[:remaining]
        self.sample_offset = remaining
        return values


@pytest.fixture
def strict_source_module(monkeypatch):
    monkeypatch.setenv(
        "CHESS_RL_MILES_DETERMINISTIC_SEED_MODE",
        "sample-index",
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
        sys.modules, "chess_rl_miles.gate_data_source", raising=False
    )
    return importlib.import_module("chess_rl_miles.gate_data_source")


def _args(**overrides):
    values = {
        "rollout_global_dataset": True,
        "debug_rollout_only": True,
        "sglang_enable_deterministic_inference": True,
        "rollout_shuffle": True,
        "partial_rollout": False,
        "dynamic_sampling_filter_path": None,
        "rollout_batch_size": 256,
        "over_sampling_batch_size": 256,
        "n_samples_per_prompt": 8,
        "num_rollout": 4,
        "dataset_size": 1024,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_strict_source_consumes_one_epoch_without_wrap(
    strict_source_module,
):
    source = strict_source_module.StrictEpochRolloutDataSource(_args())
    batches = [source.get_samples(256) for _ in range(4)]

    assert [len(batch) for batch in batches] == [256] * 4
    assert [value for batch in batches for value in batch] == list(
        range(1024)
    )
    assert source.sample_offset == 1024
    assert source.epoch_id == 0
    with pytest.raises(RuntimeError, match="refuses dataset wrap"):
        source.get_samples(256)
    with pytest.raises(RuntimeError, match="only permits"):
        source.get_samples(1)


def test_strict_source_refuses_requeue(strict_source_module):
    source = strict_source_module.StrictEpochRolloutDataSource(_args())
    source.add_samples([])
    with pytest.raises(RuntimeError, match="refuses aborted-sample requeue"):
        source.add_samples([object()])


@pytest.mark.parametrize(
    "overrides,error",
    [
        ({"debug_rollout_only": False}, "debug_rollout_only"),
        (
            {"sglang_enable_deterministic_inference": False},
            "deterministic_inference",
        ),
        ({"rollout_shuffle": False}, "rollout_shuffle"),
        ({"partial_rollout": True}, "partial_rollout"),
        ({"dynamic_sampling_filter_path": "filter"}, "dynamic sampling"),
        ({"rollout_batch_size": 128}, "rollout_batch_size"),
        ({"over_sampling_batch_size": 512}, "over_sampling_batch_size"),
        ({"n_samples_per_prompt": 4}, "n_samples_per_prompt"),
        ({"num_rollout": 0, "dataset_size": 0}, "num_rollout"),
        ({"dataset_size": 1023}, "post-tokenization dataset size"),
    ],
)
def test_strict_source_rejects_contract_drift(
    strict_source_module,
    overrides,
    error,
):
    with pytest.raises(ValueError, match=error):
        strict_source_module.StrictEpochRolloutDataSource(
            _args(**overrides)
        )


def test_strict_source_requires_unique_sample_seed_mode(
    strict_source_module,
    monkeypatch,
):
    monkeypatch.setenv(
        "CHESS_RL_MILES_DETERMINISTIC_SEED_MODE",
        "sibling-index",
    )
    with pytest.raises(ValueError, match="seed mode must equal sample-index"):
        strict_source_module.StrictEpochRolloutDataSource(_args())
