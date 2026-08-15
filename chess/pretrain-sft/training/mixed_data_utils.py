"""Utilities for mixing pretraining and SFT-style batches in one dataloader."""
from __future__ import annotations

import random
from typing import Iterable, Optional, List

import torch
from torch.utils.data import IterableDataset, DataLoader

from .data_utils import create_dataloader
from .sft_data_utils import create_sft_dataloader


class _InfiniteIterator:
    """Cycle indefinitely over a dataloader."""

    def __init__(self, loader: Iterable):
        self.loader = loader

    def __iter__(self):
        while True:
            for batch in self.loader:
                yield batch


class MixedBatchIterable(IterableDataset):
    """Yield a fixed number of batches sampled from two loaders.

    Batches are normalized to a shared format with ``input_ids``, ``labels`` and
    ``attention_mask`` so a single training loop can consume both pretraining
    and SFT-formatted data. Each batch also carries a ``data_type`` field for
    debugging/monitoring.
    """

    def __init__(
        self,
        pretrain_loader: Iterable,
        sft_loader: Iterable,
        *,
        pretrain_fraction: float,
        total_batches: int,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.pretrain_loader = pretrain_loader
        self.sft_loader = sft_loader
        self.pretrain_fraction = float(pretrain_fraction)
        self.total_batches = int(total_batches)
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return self.total_batches

    def _cycle(self, loader: Iterable):
        return iter(_InfiniteIterator(loader))

    @staticmethod
    def _format_pretrain_batch(batch):
        x, y = batch
        attention_mask = torch.ones_like(x)
        return {
            "input_ids": x,
            "labels": y,
            "attention_mask": attention_mask,
            "data_type": "pretrain",
        }

    @staticmethod
    def _format_sft_batch(batch):
        # Ensure we return a mutable dict with a data type marker
        formatted = dict(batch)
        formatted.setdefault("data_type", "sft")
        return formatted

    def __iter__(self):
        pre_iter = self._cycle(self.pretrain_loader)
        sft_iter = self._cycle(self.sft_loader)

        for _ in range(self.total_batches):
            use_pretrain = self._rng.random() < self.pretrain_fraction
            if use_pretrain:
                batch = next(pre_iter)
                yield self._format_pretrain_batch(batch)
            else:
                batch = next(sft_iter)
                yield self._format_sft_batch(batch)


def create_mixed_dataloader(
    *,
    pretrain_txt_files: List[str],
    sft_files: List[str],
    tokenizer,
    seq_len: int = 512,
    batch_size: int = 64,
    sft_batch_size: Optional[int] = None,
    pretrain_fraction: float = 0.5,
    total_batches: Optional[int] = None,
    num_workers: int = 0,
    sft_num_workers: int = 0,
    cache_size: int = 1_000_000,
    dataset_shuffle: bool = False,
    num_shards: Optional[int] = None,
    prefetch_factor: Optional[int] = None,
    persistent_workers: bool = False,
    sft_prefetch_factor: int = 2,
    sft_persistent_workers: bool = True,
    sft_mask_prompt: bool = True,
    sft_pad_token_id: Optional[int] = None,
    sft_cot_field: str = "cot_format",
    sft_prompt_field: str = "pgn",
    seed: Optional[int] = None,
) -> DataLoader:
    """Create a dataloader that mixes pretraining and SFT batches.

    Args:
        pretrain_txt_files: Text or tokenized files for standard language modeling.
        sft_files: JSON/JSONL files with chat-style SFT data.
        tokenizer: Shared tokenizer instance.
        seq_len: Sequence length for the pretraining side.
        batch_size: Batch size for pretraining batches.
        sft_batch_size: Optional override for SFT batches (defaults to ``batch_size``).
        pretrain_fraction: Probability of sampling a pretraining batch at each step.
        total_batches: Number of batches to yield per epoch. If ``None``, uses the
            sum of the individual dataloader lengths.
        num_workers: Worker count for pretraining dataloader.
        sft_num_workers: Worker count for SFT dataloader.
        cache_size: Max tokens per shard for pretraining dataloader.
        dataset_shuffle: Whether to shuffle shards for pretraining dataloader.
        num_shards: Optional limit of shards for pretraining dataloader.
        prefetch_factor: Prefetch factor for pretraining workers (if any).
        persistent_workers: Whether to persist pretraining workers.
        sft_prefetch_factor: Prefetch factor for SFT dataloader workers.
        sft_persistent_workers: Whether to persist SFT dataloader workers.
        sft_mask_prompt: Whether to mask prompt tokens for SFT labels.
        sft_pad_token_id: Optional pad token override for SFT batches.
        sft_cot_field: Which CoT field to use for responses in SFT data.
        sft_prompt_field: Which field to use for prompts in SFT data.
        seed: Optional random seed for mixing choice.

    Returns:
        DataLoader yielding normalized mixed batches.
    """

    pretrain_loader = create_dataloader(
        txt_files=pretrain_txt_files,
        tokenizer=tokenizer,
        batch_size=batch_size,
        seq_len=seq_len,
        num_workers=num_workers,
        shuffle=False,
        cache_size=cache_size,
        dataset_shuffle=dataset_shuffle,
        num_shards=num_shards,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
    )

    sft_loader = create_sft_dataloader(
        data_files=sft_files,
        tokenizer=tokenizer,
        batch_size=sft_batch_size or batch_size,
        seq_len=seq_len,
        num_workers=sft_num_workers,
        shuffle=True,
        mask_prompt=sft_mask_prompt,
        pad_token_id=sft_pad_token_id,
        prefetch_factor=sft_prefetch_factor,
        persistent_workers=sft_persistent_workers,
        cot_field=sft_cot_field,
        prompt_field=sft_prompt_field,
    )

    if total_batches is None:
        total_batches = len(pretrain_loader) + len(sft_loader)

    mixed_dataset = MixedBatchIterable(
        pretrain_loader=pretrain_loader,
        sft_loader=sft_loader,
        pretrain_fraction=pretrain_fraction,
        total_batches=total_batches,
        seed=seed,
    )

    return DataLoader(mixed_dataset, batch_size=None)
