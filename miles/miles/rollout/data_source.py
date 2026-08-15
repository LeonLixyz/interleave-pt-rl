import abc
import copy
import logging
import os
from pathlib import Path

import torch

from miles.utils.data import Dataset
from miles.utils.misc import load_function
from miles.utils.processing_utils import load_processor, load_tokenizer
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


class DataSource(abc.ABC):
    @abc.abstractmethod
    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        """
        Return num_samples samples
        """

    @abc.abstractmethod
    def add_samples(self, samples: list[list[Sample]]):
        """
        Add samples to the data source
        """

    @abc.abstractmethod
    def save(self, rollout_id):
        """
        Save the state of the data source
        """

    @abc.abstractmethod
    def load(self, rollout_id=None):
        """
        Load the state of the data source
        """


# TODO may further refactor data-loading part later
class RolloutDataSource(DataSource):
    def __init__(self, args):
        self.args = args

        self.epoch_id = 0
        self.sample_group_index = 0
        self.sample_index = 0
        self.sample_offset = 0
        # TODO remove this
        self.metadata = {}

        if args.rollout_global_dataset:
            tokenizer = load_tokenizer(
                args.hf_checkpoint, chat_template_path=args.chat_template_path, trust_remote_code=True
            )
            processor = load_processor(args.hf_checkpoint, trust_remote_code=True)

            # TODO move (during the refactor)
            if (d := args.dump_details) is not None:
                tokenizer.save_pretrained(Path(d) / "tokenizer")
                if processor:
                    processor.save_pretrained(Path(d) / "processor")

            reserved_prefix_tokens = int(
                getattr(args, "rollout_prompt_reserved_prefix_tokens", 0)
            )
            if reserved_prefix_tokens < 0:
                raise ValueError(
                    "rollout_prompt_reserved_prefix_tokens must be non-negative"
                )
            dataset_max_length = args.rollout_max_prompt_len
            if dataset_max_length is not None:
                dataset_max_length -= reserved_prefix_tokens
                if dataset_max_length <= 0:
                    raise ValueError(
                        "rollout_max_prompt_len must exceed reserved prefix tokens"
                    )
            self.dataset = Dataset(
                args.prompt_data,
                tokenizer=tokenizer,
                processor=processor,
                max_length=dataset_max_length,
                prompt_key=args.input_key,
                multimodal_keys=args.multimodal_keys,
                label_key=args.label_key,
                metadata_key=args.metadata_key,
                tool_key=args.tool_key,
                apply_chat_template=args.apply_chat_template,
                apply_chat_template_kwargs=args.apply_chat_template_kwargs,
                seed=args.rollout_seed,
            )
            self.metadata["prompt_filter"] = {
                "configured_post_prefix_limit": args.rollout_max_prompt_len,
                "reserved_prefix_tokens": reserved_prefix_tokens,
                "prefilter_limit": dataset_max_length,
                "input_rows": self.dataset.input_row_count,
                "admitted_rows": len(self.dataset),
                "filtered_rows": self.dataset.filtered_row_count,
            }
            if self.args.rollout_shuffle:
                self.dataset.shuffle(self.epoch_id)
        else:
            self.dataset = None

    def get_samples(self, num_samples):
        # TODO further improve code
        if self.dataset is not None:
            if self.sample_offset + num_samples <= len(self.dataset):
                prompt_samples = self.dataset.samples[self.sample_offset : self.sample_offset + num_samples]
                self.sample_offset += num_samples
            else:
                prompt_samples = self.dataset.samples[self.sample_offset :]
                num_samples -= len(prompt_samples)
                self.epoch_id += 1
                if self.args.rollout_shuffle:
                    self.dataset.shuffle(self.epoch_id)
                prompt_samples += self.dataset.samples[:num_samples]
                self.sample_offset = num_samples
        else:
            prompt_samples = [Sample() for _ in range(num_samples)]

        samples = []
        for prompt_sample in prompt_samples:
            group = []
            for _ in range(self.args.n_samples_per_prompt):
                sample = copy.deepcopy(prompt_sample)
                sample.group_index = self.sample_group_index
                sample.index = self.sample_index
                self.sample_index += 1
                group.append(sample)
            self.sample_group_index += 1
            samples.append(group)
        return samples

    def add_samples(self, samples: list[list[Sample]]):
        raise RuntimeError(f"Cannot add samples to {self.__class__.__name__}. This is a read-only data source.")

    def checkpoint_state(self, rollout_id: int) -> dict:
        """Return the complete cursor needed to generate the next rollout.

        FSDP checkpoints serialize this object inside the same immutable
        ``iter_N`` transaction as the model, optimizer, scheduler, and RNG.
        Keeping the cursor in that transaction prevents a committed model
        checkpoint from being paired with a missing or stale prompt cursor.
        """

        if not self.args.rollout_global_dataset:
            raise RuntimeError(
                "rollout data-source checkpoint state requires --rollout-global-dataset"
            )
        if isinstance(rollout_id, bool) or not isinstance(rollout_id, int) or rollout_id < 0:
            raise ValueError(f"invalid rollout_id for data-source checkpoint: {rollout_id!r}")
        state = {
            "schema": "miles-rollout-data-source-v1",
            "rollout_id": rollout_id,
            "next_rollout_id": rollout_id + 1,
            "dataset_length": len(self.dataset),
            "sample_offset": self.sample_offset,
            "epoch_id": self.epoch_id,
            "sample_group_index": self.sample_group_index,
            "sample_index": self.sample_index,
            "metadata": self.metadata,
        }
        if hasattr(self, "buffer"):
            # Partial-rollout samples affect the subsequent sampling stream and
            # therefore belong to the authenticated cursor as well.
            state["buffer"] = copy.deepcopy(self.buffer)
        return state

    def restore_checkpoint_state(self, state_dict: dict, rollout_id: int) -> None:
        if not isinstance(state_dict, dict):
            raise RuntimeError("rollout data-source checkpoint must be a dictionary")
        expected = {
            "schema": "miles-rollout-data-source-v1",
            "rollout_id": rollout_id,
            "next_rollout_id": rollout_id + 1,
            "dataset_length": len(self.dataset),
        }
        mismatches = {
            key: {"expected": value, "actual": state_dict.get(key)}
            for key, value in expected.items()
            if state_dict.get(key) != value
        }
        if mismatches:
            raise RuntimeError(
                "rollout data-source checkpoint identity mismatch: "
                f"{mismatches}"
            )

        integer_fields = (
            "sample_offset",
            "epoch_id",
            "sample_group_index",
            "sample_index",
        )
        for key in integer_fields:
            value = state_dict.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise RuntimeError(
                    f"invalid rollout data-source checkpoint field {key}={value!r}"
                )
        if state_dict["sample_offset"] > len(self.dataset):
            raise RuntimeError(
                "rollout data-source sample_offset exceeds the restored dataset: "
                f"offset={state_dict['sample_offset']} length={len(self.dataset)}"
            )

        self.sample_offset = state_dict["sample_offset"]
        self.epoch_id = state_dict["epoch_id"]
        self.sample_group_index = state_dict["sample_group_index"]
        self.sample_index = state_dict["sample_index"]
        self.metadata = state_dict.get("metadata", {})
        if hasattr(self, "buffer"):
            buffer = state_dict.get("buffer")
            if not isinstance(buffer, list):
                raise RuntimeError(
                    "buffered rollout data source requires an authenticated buffer list"
                )
            self.buffer = buffer

        if self.args.rollout_shuffle:
            self.dataset.shuffle(self.epoch_id)

    def save(self, rollout_id):
        if not self.args.rollout_global_dataset:
            return

        # FSDP owns an atomic checkpoint transaction.  Its caller passes
        # checkpoint_state() into checkpoint.save(), so writing a second,
        # unauthenticated cursor here would reintroduce a crash window.
        if getattr(self.args, "train_backend", None) == "fsdp":
            raise RuntimeError(
                "FSDP rollout cursor must be committed inside iter_N; "
                "use RolloutManager.checkpoint_state instead of save"
            )

        state_dict = self.checkpoint_state(rollout_id)
        path = os.path.join(self.args.save, f"rollout/global_dataset_state_dict_{rollout_id}.pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(state_dict, path)

    def load(self, rollout_id=None):
        if not self.args.rollout_global_dataset:
            return

        if self.args.load is None:
            return

        if getattr(self.args, "train_backend", None) == "fsdp":
            from miles.backends.experimental.fsdp_utils.checkpoint import (
                load_committed_rollout_state,
            )

            state_dict = load_committed_rollout_state(
                Path(self.args.load),
                rollout_id=int(rollout_id),
            )
            self.restore_checkpoint_state(state_dict, int(rollout_id))
            logger.info(
                "loaded authenticated rollout cursor for rollout %s from iter_%07d",
                rollout_id,
                int(rollout_id) + 1,
            )
            return

        path = os.path.join(self.args.load, f"rollout/global_dataset_state_dict_{rollout_id}.pt")
        if not os.path.exists(path):
            logger.info(f"Checkpoint {path} does not exist.")
            return

        logger.info(f"load metadata from {path}")
        logger.info(f"load metadata: {self.metadata}")
        state_dict = torch.load(path, weights_only=False)
        self.restore_checkpoint_state(state_dict, int(rollout_id))


class RolloutDataSourceWithBuffer(RolloutDataSource):
    def __init__(self, args):
        super().__init__(args)
        self.buffer = []
        if self.args.buffer_filter_path is None:
            self.buffer_filter = pop_first
        else:
            self.buffer_filter = load_function(self.args.buffer_filter_path)

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        """
        Return num_samples samples
        """

        samples = self._get_samples_from_buffer(num_samples)
        num_samples -= len(samples)

        if num_samples == 0:
            return samples

        samples += super().get_samples(num_samples=num_samples)
        return samples

    def _get_samples_from_buffer(self, num_samples: int) -> list[list[Sample]]:
        if len(self.buffer) == 0 or num_samples == 0:
            return []

        samples = self.buffer_filter(self.args, None, self.buffer, num_samples)
        return samples

    def add_samples(self, samples: list[list[Sample]]):
        """
        Add a sample group to buffer.
        """
        if not samples:
            return
        assert isinstance(samples, list), f"samples must be a list, got {type(samples)}"
        assert isinstance(samples[0], list), f"the elements of samples must be list, got {type(samples[0])}"
        for i in range(0, len(samples)):
            assert (
                len(samples[i]) == self.args.n_samples_per_prompt
            ), f"the length of the elements of samples must be equal to n_samples_per_prompt, got {len(samples[i])} != {self.args.n_samples_per_prompt}"
            group = samples[i]  # type: ignore
            self.buffer.append(group)

    # TODO remove
    def update_metadata(self, metadata: dict):
        self.metadata.update(metadata)

    # TODO remove
    def get_metadata(self):
        return self.metadata

    def get_buffer_length(self):
        return len(self.buffer)


def pop_first(args, rollout_id, buffer: list[list[Sample]], num_samples: int) -> list[list[Sample]]:
    num_to_pop = min(len(buffer), num_samples)
    samples = buffer[:num_to_pop]
    del buffer[:num_to_pop]
    return samples
