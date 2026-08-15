"""Fail-closed prompt source for fixed, paired rollout gates.

The production RL source is intentionally allowed to cycle through its prompt
dataset.  A statistical gate has a different contract: every declared prompt
must be sampled exactly once, and a failed generation task must not be
silently replaced by a different prompt.  This source enforces the data-side
half of that contract; :mod:`chess_rl_miles.batched_rollout` recognizes its
``strict_exact_once`` marker and fails on any generation-task exception.
"""

from __future__ import annotations

import os
from typing import Any

from miles.rollout.data_source import RolloutDataSource


class StrictEpochRolloutDataSource(RolloutDataSource):
    """Consume one immutable prompt parquet exactly once without replacement."""

    strict_exact_once = True

    def __init__(self, args: Any):
        super().__init__(args)

        violations: list[str] = []
        if self.dataset is None:
            violations.append("rollout_global_dataset must be enabled")
        if not bool(getattr(args, "debug_rollout_only", False)):
            violations.append("debug_rollout_only must be enabled")
        if not bool(
            getattr(args, "sglang_enable_deterministic_inference", False)
        ):
            violations.append(
                "sglang_enable_deterministic_inference must be enabled"
            )
        if (
            os.environ.get(
                "CHESS_RL_MILES_DETERMINISTIC_SEED_MODE"
            )
            != "sample-index"
        ):
            violations.append(
                "deterministic seed mode must equal sample-index"
            )
        if not bool(getattr(args, "rollout_shuffle", False)):
            violations.append("rollout_shuffle must be enabled")
        if bool(getattr(args, "partial_rollout", False)):
            violations.append("partial_rollout must be disabled")
        if getattr(args, "dynamic_sampling_filter_path", None) is not None:
            violations.append("dynamic sampling must be disabled")

        rollout_batch_size = getattr(args, "rollout_batch_size", None)
        over_sampling_batch_size = getattr(
            args, "over_sampling_batch_size", None
        )
        num_rollout = getattr(args, "num_rollout", None)
        if rollout_batch_size != 256:
            violations.append("rollout_batch_size must equal 256")
        if over_sampling_batch_size != rollout_batch_size:
            violations.append(
                "over_sampling_batch_size must equal rollout_batch_size"
            )
        if getattr(args, "n_samples_per_prompt", None) != 8:
            violations.append("n_samples_per_prompt must equal 8")
        if (
            isinstance(num_rollout, bool)
            or not isinstance(num_rollout, int)
            or num_rollout <= 0
        ):
            violations.append("num_rollout must be a positive integer")

        expected_prompts = (
            num_rollout * rollout_batch_size
            if isinstance(num_rollout, int)
            and not isinstance(num_rollout, bool)
            and isinstance(rollout_batch_size, int)
            and not isinstance(rollout_batch_size, bool)
            else None
        )
        if self.dataset is not None and (
            expected_prompts is None or len(self.dataset) != expected_prompts
        ):
            violations.append(
                "post-tokenization dataset size must equal "
                "num_rollout * rollout_batch_size"
            )

        if violations:
            raise ValueError(
                "strict exact-once rollout source rejected configuration: "
                + "; ".join(violations)
            )

        self._strict_expected_prompts = int(expected_prompts)
        self._strict_get_calls = 0

    def get_samples(self, num_samples: int):
        if num_samples != self.args.rollout_batch_size:
            raise RuntimeError(
                "strict exact-once rollout source only permits one complete "
                f"{self.args.rollout_batch_size}-prompt request"
            )
        if (
            self.sample_offset + num_samples
            > self._strict_expected_prompts
        ):
            raise RuntimeError(
                "strict exact-once rollout source refuses dataset wrap"
            )

        epoch_before = self.epoch_id
        offset_before = self.sample_offset
        samples = super().get_samples(num_samples)
        self._strict_get_calls += 1

        if self.epoch_id != epoch_before or self.epoch_id != 0:
            raise RuntimeError(
                "strict exact-once rollout source changed dataset epoch"
            )
        if self.sample_offset != offset_before + num_samples:
            raise RuntimeError(
                "strict exact-once rollout source advanced an unexpected "
                "number of prompts"
            )
        if len(samples) != num_samples:
            raise RuntimeError(
                "strict exact-once rollout source returned an incomplete batch"
            )
        return samples

    def add_samples(self, samples):
        if samples:
            raise RuntimeError(
                "strict exact-once rollout source refuses aborted-sample "
                "requeue"
            )
