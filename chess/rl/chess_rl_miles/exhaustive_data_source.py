"""Fail-closed data source for exhaustive pass@16 rollout-only evaluation.

This source is intentionally separate from both the production RL source and
the fixed statistical-gate source.  It is selected explicitly with
``--data-source-path`` and accepts only the rollout-only Miles/SGLang
configuration used for an exhaustive 16-sample evaluation.

Every input parquet row must contain a non-negative, globally unique
``extra_info.source_row_index``.  That identity is propagated to all sibling
samples so generation seeds and exported rows remain stable even when each
Modal shard is shuffled independently:

``group_index = source_row_index``
``sample_index = source_row_index * 16 + sample_slot``

The class also refuses dataset wrap and aborted-sample requeue.  Combined with
the ``strict_exact_once`` marker honored by
:mod:`chess_rl_miles.batched_rollout`, a generation-task failure terminates the
shard instead of silently substituting a different prompt.
"""

from __future__ import annotations

import os
from numbers import Integral
from typing import Any

from miles.rollout.data_source import RolloutDataSource


SOURCE_ROW_INDEX_KEY = "source_row_index"
SAMPLE_SLOT_KEY = "pass_at_16_sample_slot"
SOURCE_SAMPLE_INDEX_KEY = "pass_at_16_sample_index"

SAMPLES_PER_PROMPT = 16
MAX_PROMPT_LEN = 1_024
MAX_RESPONSE_LEN = 2_560
MAX_CONTEXT_LEN = 3_072

ROLLOUT_FUNCTION_PATH = (
    "chess_rl_miles.batched_rollout.ChessBatchedRolloutFn"
)
GENERATE_FUNCTION_PATH = "chess_rl_miles.rollout.generate"
REWARD_FUNCTION_PATH = "chess_rl_miles.reward.reward_func"
ROLLOUT_LOG_FUNCTION_PATH = "chess_rl_miles.io.log_rollout_data"


def _positive_int(value: Any) -> bool:
    return (
        isinstance(value, Integral)
        and not isinstance(value, bool)
        and int(value) > 0
    )


def _source_row_index(sample: Any) -> int:
    metadata = getattr(sample, "metadata", None)
    if not isinstance(metadata, dict):
        raise ValueError(
            "every exhaustive prompt must have dictionary metadata"
        )
    value = metadata.get(SOURCE_ROW_INDEX_KEY)
    if (
        not isinstance(value, Integral)
        or isinstance(value, bool)
        or int(value) < 0
    ):
        raise ValueError(
            "every exhaustive prompt must have a non-negative integer "
            f"metadata.{SOURCE_ROW_INDEX_KEY}"
        )
    return int(value)


class StrictExhaustiveRolloutDataSource(RolloutDataSource):
    """Consume one complete pass@16 shard exactly once without replacement."""

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
        artifact_root = os.environ.get(
            "CHESS_RL_MILES_ARTIFACT_ROOT", ""
        )
        if not artifact_root or not os.path.isabs(artifact_root):
            violations.append(
                "CHESS_RL_MILES_ARTIFACT_ROOT must be an absolute path"
            )
        if bool(getattr(args, "partial_rollout", False)):
            violations.append("partial_rollout must be disabled")
        if getattr(args, "dynamic_sampling_filter_path", None) is not None:
            violations.append("dynamic sampling must be disabled")
        if bool(getattr(args, "use_fault_tolerance", False)):
            violations.append("rollout fault tolerance must be disabled")
        if bool(getattr(args, "use_wandb", False)):
            violations.append("W&B telemetry must be disabled")
        if not bool(getattr(args, "sglang_skip_tokenizer_init", False)):
            violations.append(
                "sglang_skip_tokenizer_init must be enabled"
            )
        if not bool(getattr(args, "use_miles_router", False)):
            violations.append("use_miles_router must be enabled")

        rollout_batch_size = getattr(args, "rollout_batch_size", None)
        over_sampling_batch_size = getattr(
            args, "over_sampling_batch_size", None
        )
        num_rollout = getattr(args, "num_rollout", None)
        if not _positive_int(rollout_batch_size):
            violations.append("rollout_batch_size must be a positive integer")
        if over_sampling_batch_size != rollout_batch_size:
            violations.append(
                "over_sampling_batch_size must equal rollout_batch_size"
            )
        if getattr(args, "n_samples_per_prompt", None) != SAMPLES_PER_PROMPT:
            violations.append(
                f"n_samples_per_prompt must equal {SAMPLES_PER_PROMPT}"
            )
        if not _positive_int(num_rollout):
            violations.append("num_rollout must be a positive integer")

        fixed_settings = (
            ("rollout_max_prompt_len", MAX_PROMPT_LEN),
            ("rollout_max_response_len", MAX_RESPONSE_LEN),
            ("rollout_max_context_len", MAX_CONTEXT_LEN),
            ("rollout_temperature", 1.0),
            ("rollout_top_p", 1.0),
            ("rollout_function_path", ROLLOUT_FUNCTION_PATH),
            ("custom_generate_function_path", GENERATE_FUNCTION_PATH),
            ("custom_rm_path", REWARD_FUNCTION_PATH),
            (
                "custom_rollout_log_function_path",
                ROLLOUT_LOG_FUNCTION_PATH,
            ),
        )
        for field, expected in fixed_settings:
            observed = getattr(args, field, None)
            if observed != expected:
                violations.append(f"{field} must equal {expected!r}")

        expected_prompts = (
            int(num_rollout) * int(rollout_batch_size)
            if _positive_int(num_rollout)
            and _positive_int(rollout_batch_size)
            else None
        )
        if self.dataset is not None and (
            expected_prompts is None or len(self.dataset) != expected_prompts
        ):
            violations.append(
                "post-tokenization dataset size must equal "
                "num_rollout * rollout_batch_size"
            )

        source_indices: list[int] = []
        if self.dataset is not None:
            try:
                for sample in self.dataset.samples:
                    metadata = getattr(sample, "metadata", None)
                    source_indices.append(_source_row_index(sample))
                    if isinstance(metadata, dict) and any(
                        key in metadata
                        for key in (
                            SAMPLE_SLOT_KEY,
                            SOURCE_SAMPLE_INDEX_KEY,
                        )
                    ):
                        raise ValueError(
                            "source prompt metadata contains reserved "
                            "pass@16 identity fields"
                        )
                if len(set(source_indices)) != len(source_indices):
                    raise ValueError(
                        "metadata.source_row_index values must be unique "
                        "within an exhaustive shard"
                    )
            except ValueError as exc:
                violations.append(str(exc))

        if violations:
            raise ValueError(
                "strict exhaustive rollout source rejected configuration: "
                + "; ".join(violations)
            )

        self._strict_expected_prompts = int(expected_prompts)
        self._strict_consumed_prompts = 0
        self._strict_get_calls = 0

    def get_samples(self, num_samples: int):
        if num_samples != self.args.rollout_batch_size:
            raise RuntimeError(
                "strict exhaustive rollout source only permits one complete "
                f"{self.args.rollout_batch_size}-prompt request"
            )
        if (
            self._strict_consumed_prompts + num_samples
            > self._strict_expected_prompts
        ):
            raise RuntimeError(
                "strict exhaustive rollout source refuses dataset wrap"
            )

        epoch_before = self.epoch_id
        groups = super().get_samples(num_samples)
        if self.epoch_id != epoch_before or self.epoch_id != 0:
            raise RuntimeError(
                "strict exhaustive rollout source changed dataset epoch"
            )
        if len(groups) != num_samples:
            raise RuntimeError(
                "strict exhaustive rollout source returned an incomplete "
                "prompt batch"
            )

        for group in groups:
            if len(group) != SAMPLES_PER_PROMPT:
                raise RuntimeError(
                    "strict exhaustive rollout source returned an incomplete "
                    "pass@16 group"
                )
            source_row_index = _source_row_index(group[0])
            for sample_slot, sample in enumerate(group):
                if _source_row_index(sample) != source_row_index:
                    raise RuntimeError(
                        "strict exhaustive rollout group mixed source rows"
                    )
                source_sample_index = (
                    source_row_index * SAMPLES_PER_PROMPT + sample_slot
                )
                sample.group_index = source_row_index
                sample.index = source_sample_index
                sample.metadata[SAMPLE_SLOT_KEY] = sample_slot
                sample.metadata[
                    SOURCE_SAMPLE_INDEX_KEY
                ] = source_sample_index

        self._strict_consumed_prompts += num_samples
        self._strict_get_calls += 1
        return groups

    def add_samples(self, samples):
        if samples:
            raise RuntimeError(
                "strict exhaustive rollout source refuses aborted-sample "
                "requeue"
            )
