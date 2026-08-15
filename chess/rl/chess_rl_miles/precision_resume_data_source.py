"""Deterministic checkpoint-aware prompt source for production and its gate.

Production and the BF16/FP32 resume gate deliberately use this same source and
failure behavior. In gate mode, leg 1 consumes shuffled epoch-0 prompts
0..255. A fresh leg-2 process must restore the authenticated checkpoint-1
cursor and consume prompts 256..511. Outside gate mode, ordinary committed
cursor checkpoint/restore supports the complete production run.
"""

from __future__ import annotations

import os
from typing import Any

from miles.rollout.data_source import RolloutDataSource


EXPECTED_ROLLOUT_BATCH_SIZE = 256
EXPECTED_SAMPLES_PER_PROMPT = 8
MINIMUM_ADMITTED_PROMPTS = 512


class PrecisionResumeRolloutDataSource(RolloutDataSource):
    """Use deterministic prompt cursors; add exact two-leg gate assertions."""

    strict_exact_once = True

    def __init__(self, args: Any):
        super().__init__(args)
        leg_text = os.environ.get("CHESS_RL_MILES_PRECISION_GATE_LEG")
        leg = int(leg_text) if leg_text in {"1", "2"} else None
        violations: list[str] = []
        if self.dataset is None:
            violations.append("rollout_global_dataset must be enabled")
        if bool(getattr(args, "debug_rollout_only", False)):
            violations.append("debug_rollout_only must be disabled")
        if not bool(
            getattr(args, "sglang_enable_deterministic_inference", False)
        ):
            violations.append(
                "sglang_enable_deterministic_inference must be enabled"
            )
        if os.environ.get("CHESS_RL_MILES_DETERMINISTIC_SEED_MODE") != "sample-index":
            violations.append("deterministic seed mode must equal sample-index")
        if not bool(getattr(args, "rollout_shuffle", False)):
            violations.append("rollout_shuffle must be enabled")
        if bool(getattr(args, "partial_rollout", False)):
            violations.append("partial_rollout must be disabled")
        if getattr(args, "dynamic_sampling_filter_path", None) is not None:
            violations.append("dynamic sampling must be disabled")
        if bool(getattr(args, "use_fault_tolerance", False)):
            violations.append("rollout fault tolerance must be disabled")
        if getattr(args, "rollout_batch_size", None) != EXPECTED_ROLLOUT_BATCH_SIZE:
            violations.append("rollout_batch_size must equal 256")
        if getattr(args, "over_sampling_batch_size", None) != EXPECTED_ROLLOUT_BATCH_SIZE:
            violations.append("over_sampling_batch_size must equal 256")
        if getattr(args, "n_samples_per_prompt", None) != EXPECTED_SAMPLES_PER_PROMPT:
            violations.append("n_samples_per_prompt must equal 8")
        if getattr(args, "num_steps_per_rollout", None) != 1:
            violations.append("num_steps_per_rollout must equal 1")
        num_rollout = getattr(args, "num_rollout", None)
        if isinstance(num_rollout, bool) or not isinstance(num_rollout, int) or num_rollout <= 0:
            violations.append("num_rollout must be a positive integer")
        if leg is not None:
            expected_num_rollout = 1 if leg == 1 else 2
            if num_rollout != expected_num_rollout:
                violations.append(
                    f"num_rollout must equal {expected_num_rollout!r} for leg {leg!r}"
                )
            expected_start = 0 if leg == 1 else 1
            observed_start = getattr(args, "start_rollout_id", None)
            if observed_start not in {None, expected_start}:
                violations.append(
                    f"start_rollout_id must equal {expected_start!r} for leg {leg!r}"
                )
        if (
            leg is not None
            and self.dataset is not None
            and len(self.dataset) < MINIMUM_ADMITTED_PROMPTS
        ):
            violations.append(
                "post-tokenization dataset must admit at least 512 prompts"
            )
        if violations:
            raise ValueError(
                "precision resume rollout source rejected configuration: "
                + "; ".join(violations)
            )

        self._precision_leg = leg
        self._precision_get_calls = 0
        self._precision_restored = False

    def _expected_before(self) -> tuple[int, int, int]:
        if self._precision_leg == 1:
            return (0, 0, 0)
        return (256, 256, 2_048)

    def get_samples(self, num_samples: int):
        if self._precision_leg is None:
            if num_samples != EXPECTED_ROLLOUT_BATCH_SIZE:
                raise RuntimeError(
                    "deterministic checkpoint source only permits complete 256-prompt requests"
                )
            return super().get_samples(num_samples)
        if self._precision_get_calls != 0:
            raise RuntimeError(
                "precision resume source permits exactly one prompt request per process"
            )
        if num_samples != EXPECTED_ROLLOUT_BATCH_SIZE:
            raise RuntimeError(
                "precision resume source only permits one 256-prompt request"
            )
        if self._precision_leg == 2 and not self._precision_restored:
            raise RuntimeError(
                "precision resume leg 2 must restore checkpoint-1 cursor before sampling"
            )
        expected_offset, expected_group, expected_sample = self._expected_before()
        observed_before = (
            self.sample_offset,
            self.sample_group_index,
            self.sample_index,
        )
        if observed_before != (expected_offset, expected_group, expected_sample):
            raise RuntimeError(
                "precision resume cursor before sampling drifted: "
                f"expected={(expected_offset, expected_group, expected_sample)} "
                f"actual={observed_before}"
            )
        epoch_before = self.epoch_id
        groups = super().get_samples(num_samples)
        self._precision_get_calls += 1
        if self.epoch_id != epoch_before or self.epoch_id != 0:
            raise RuntimeError("precision resume source changed dataset epoch")
        expected_after = (
            expected_offset + 256,
            expected_group + 256,
            expected_sample + 2_048,
        )
        observed_after = (
            self.sample_offset,
            self.sample_group_index,
            self.sample_index,
        )
        if observed_after != expected_after or len(groups) != 256:
            raise RuntimeError(
                "precision resume source advanced an unexpected cursor: "
                f"expected={expected_after} actual={observed_after} groups={len(groups)}"
            )
        return groups

    def restore_checkpoint_state(self, state_dict: dict, rollout_id: int) -> None:
        if self._precision_leg is None:
            super().restore_checkpoint_state(state_dict, rollout_id)
            return
        if self._precision_leg != 2 or rollout_id != 0:
            raise RuntimeError(
                "only precision resume leg 2 may restore checkpoint-1 cursor"
            )
        super().restore_checkpoint_state(state_dict, rollout_id)
        expected = (256, 0, 256, 2_048)
        observed = (
            self.sample_offset,
            self.epoch_id,
            self.sample_group_index,
            self.sample_index,
        )
        if observed != expected:
            raise RuntimeError(
                "precision resume checkpoint-1 cursor drifted: "
                f"expected={expected} actual={observed}"
            )
        self._precision_restored = True

    def checkpoint_state(self, rollout_id: int) -> dict:
        if self._precision_leg is None:
            return super().checkpoint_state(rollout_id)
        expected_rollout = self._precision_leg - 1
        if rollout_id != expected_rollout or self._precision_get_calls != 1:
            raise RuntimeError(
                "precision resume source can checkpoint only after its one exact prompt batch"
            )
        return super().checkpoint_state(rollout_id)

    def add_samples(self, samples):
        if samples:
            raise RuntimeError(
                "precision resume source refuses aborted-sample requeue"
            )
