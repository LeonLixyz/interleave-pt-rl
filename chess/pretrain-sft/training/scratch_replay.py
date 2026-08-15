"""Unified P2 + positive-replay stream for Exp 4's scratch branch.

This is deliberately a *sample-level* mixture.  The builder removes only the
divisibility sentinels from the immutable P2 order, adds every extracted
positive replay row, applies one seed-pinned PCG64 shuffle, and then pads the
combined order once for the fixed 8 x 21 topology.  No P2 pretraining/SFT row
and no replay row is repeated or dropped.

The baseline P2 warmup/cosine arc is not stretched by the extra replay
records.  ``baseline_cosine_steps`` is copied from the authenticated P2
manifest; any additional optimizer steps caused by replay stay at the
1e-5 cosine floor.  The examples remain interspersed throughout the unified
order—the floor tail describes LR time, not an appended replay-only phase.

Replay examples consume the exact token IDs and response ownership masks
written by Miles and authenticated by ``positive_replay``.  Text is retained
for provenance but is never re-tokenized here.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .interleaved_data import (
    PAD_RECORD,
    SAMPLE_PAD,
    DistributedManifestBatchSampler,
    InterleavedDataStream,
    LegManifest,
    LogicalTokenSelection,
    PackedPretrainDataset,
    PretrainSelection,
    SFTCache,
    SFTCacheDataset,
    SourceShardManifest,
    UnifiedInterleavedCollator,
)
from .positive_transfer import PositiveReplayDataset, validate_replay_record

SCRATCH_REPLAY_SCHEMA_VERSION = 1
SCRATCH_REPLAY_SCHEMA = "interleaved-scratch-replay-manifest-v1"
SAMPLE_POSITIVE_REPLAY = 3
# SFT IDs occupy the small negative integers.  Replay IDs descend from this
# distant sentinel, leaving PAD_RECORD (int64 minimum) reserved for padding.
REPLAY_CODE_BASE = -(1 << 62)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _hash_dict(value: Mapping[str, Any], hash_field: str) -> str:
    return hashlib.sha256(
        _canonical_json(
            {key: item for key, item in value.items() if key != hash_field}
        )
    ).hexdigest()


def _sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_save_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npy", dir=path.parent
    )
    os.close(fd)
    try:
        np.save(temporary, np.asarray(value, dtype="<i8"), allow_pickle=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _resolve_relative(metadata_path: Path, value: str) -> Path:
    return (metadata_path.parent / value).resolve()


@dataclass(frozen=True)
class ScratchReplayManifest:
    metadata_path: Path
    order_path: Path
    base_leg_path: Path
    replay_path: Path
    replay_manifest_path: Path
    sequence_length: int
    pretrain_records: int
    sft_records: int
    replay_records: int
    padding_records: int
    world_size: int
    local_batch_size: int
    baseline_cosine_steps: int
    floor_tail_steps: int
    physical_steps: int
    total_steps: int
    shuffle_seed: int
    model_init_seed: int
    source_manifest_hash: str
    selection_hash: str
    sft_cache_hash: str
    base_leg_sha256: str
    replay_sha256: str
    replay_manifest_sha256: str
    order_sha256: str
    metadata_hash: str

    @property
    def global_batch_size(self) -> int:
        return self.world_size * self.local_batch_size

    @property
    def real_records(self) -> int:
        return self.pretrain_records + self.sft_records + self.replay_records

    @classmethod
    def load(cls, metadata_path: str | Path) -> ScratchReplayManifest:
        metadata_path = Path(metadata_path).resolve()
        with metadata_path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, dict) or value.get("schema") != SCRATCH_REPLAY_SCHEMA:
            raise ValueError(f"not a scratch-replay manifest: {metadata_path}")
        expected_hash = value.get("metadata_hash")
        actual_hash = _hash_dict(value, "metadata_hash")
        if not expected_hash or expected_hash != actual_hash:
            raise ValueError(
                f"scratch-replay metadata hash mismatch: "
                f"{expected_hash} != {actual_hash}"
            )

        order_path = _resolve_relative(
            metadata_path, value.get("order_file", "order.npy")
        )
        base_leg_path = _resolve_relative(metadata_path, value["base_leg_path"])
        replay_path = _resolve_relative(metadata_path, value["replay_path"])
        replay_manifest_path = _resolve_relative(
            metadata_path, value["replay_manifest_path"]
        )
        expected_files = (
            (order_path, value["order_sha256"], "order"),
            (base_leg_path, value["base_leg_sha256"], "base P2 manifest"),
            (replay_path, value["replay_sha256"], "positive replay"),
            (
                replay_manifest_path,
                value["replay_manifest_sha256"],
                "positive replay manifest",
            ),
        )
        for path, expected, label in expected_files:
            if not path.is_file():
                raise FileNotFoundError(path)
            actual = _sha256_file(path)
            if actual != expected:
                raise ValueError(
                    f"{label} SHA-256 mismatch: {actual} != {expected}"
                )

        base = LegManifest.load(base_leg_path)
        if base.leg != "p2":
            raise ValueError(f"scratch replay requires a P2 leg, got {base.leg!r}")
        if base.physical_steps != base.total_steps:
            raise ValueError("scratch replay requires a full physical P2 manifest")
        topology = (int(value["world_size"]), int(value["local_batch_size"]))
        if topology != (base.world_size, base.local_batch_size):
            raise ValueError("scratch topology differs from the P2 manifest")
        if int(value["baseline_cosine_steps"]) != base.total_steps:
            raise ValueError("baseline cosine steps differ from the P2 manifest")
        if (
            int(value["pretrain_records"]),
            int(value["sft_records"]),
            int(value["sequence_length"]),
        ) != (
            base.pretrain_records,
            base.sft_records,
            base.sequence_length,
        ):
            raise ValueError("scratch P2 accounting differs from the base manifest")
        for field in (
            "source_manifest_hash",
            "selection_hash",
            "sft_cache_hash",
        ):
            if value[field] != getattr(base, field):
                raise ValueError(f"scratch/base mismatch for {field}")

        replay = PositiveReplayDataset(
            replay_path,
            manifest_path=replay_manifest_path,
            context_limit=int(value["sequence_length"]),
        )
        if len(replay) != int(value["replay_records"]):
            raise ValueError("scratch replay row count differs from replay artifact")

        order = np.load(order_path, mmap_mode="r", allow_pickle=False)
        if order.ndim != 1 or order.dtype != np.dtype("int64"):
            raise ValueError("scratch order must be one-dimensional int64")
        global_batch = topology[0] * topology[1]
        if global_batch <= 0 or len(order) % global_batch:
            raise ValueError("scratch order is not divisible by global batch size")
        physical_steps = len(order) // global_batch
        if physical_steps != int(value["physical_steps"]):
            raise ValueError("scratch physical step count is inconsistent")
        if int(value["total_steps"]) != physical_steps:
            raise ValueError("scratch total_steps must equal physical_steps")
        padding = int(np.count_nonzero(order == int(PAD_RECORD)))
        pretrain = int(np.count_nonzero(order >= 0))
        sft = int(
            np.count_nonzero(
                (order < 0)
                & (order > REPLAY_CODE_BASE)
                & (order != int(PAD_RECORD))
            )
        )
        positive = int(
            np.count_nonzero(
                (order <= REPLAY_CODE_BASE) & (order != int(PAD_RECORD))
            )
        )
        actual_counts = (pretrain, sft, positive, padding)
        expected_counts = (
            int(value["pretrain_records"]),
            int(value["sft_records"]),
            int(value["replay_records"]),
            int(value["padding_records"]),
        )
        if actual_counts != expected_counts:
            raise ValueError(
                f"scratch order counts differ: {actual_counts} != {expected_counts}"
            )
        floor_tail = physical_steps - base.total_steps
        if floor_tail != int(value["floor_tail_steps"]) or floor_tail < 0:
            raise ValueError("scratch floor-tail step accounting is inconsistent")

        return cls(
            metadata_path=metadata_path,
            order_path=order_path,
            base_leg_path=base_leg_path,
            replay_path=replay_path,
            replay_manifest_path=replay_manifest_path,
            sequence_length=int(value["sequence_length"]),
            pretrain_records=int(value["pretrain_records"]),
            sft_records=int(value["sft_records"]),
            replay_records=int(value["replay_records"]),
            padding_records=int(value["padding_records"]),
            world_size=topology[0],
            local_batch_size=topology[1],
            baseline_cosine_steps=int(value["baseline_cosine_steps"]),
            floor_tail_steps=floor_tail,
            physical_steps=physical_steps,
            total_steps=physical_steps,
            shuffle_seed=int(value["shuffle_seed"]),
            model_init_seed=int(value["model_init_seed"]),
            source_manifest_hash=value["source_manifest_hash"],
            selection_hash=value["selection_hash"],
            sft_cache_hash=value["sft_cache_hash"],
            base_leg_sha256=value["base_leg_sha256"],
            replay_sha256=value["replay_sha256"],
            replay_manifest_sha256=value["replay_manifest_sha256"],
            order_sha256=value["order_sha256"],
            metadata_hash=value["metadata_hash"],
        )


def _requested_inputs(
    *,
    p2_manifest_path: Path,
    replay_path: Path,
    replay_manifest_path: Path,
    shuffle_seed: int,
    model_init_seed: int,
) -> dict[str, Any]:
    return {
        "base_leg_sha256": _sha256_file(p2_manifest_path),
        "replay_sha256": _sha256_file(replay_path),
        "replay_manifest_sha256": _sha256_file(replay_manifest_path),
        "shuffle_seed": int(shuffle_seed),
        "model_init_seed": int(model_init_seed),
    }


def build_scratch_replay_manifest(
    *,
    p2_manifest_path: str | Path,
    replay_path: str | Path,
    replay_manifest_path: str | Path,
    output_dir: str | Path,
    shuffle_seed: int = 44,
    model_init_seed: int = 42,
    validate_all_replay_rows: bool = True,
) -> ScratchReplayManifest:
    """Build the immutable, unified P2 + replay sample order."""

    p2_manifest_path = Path(p2_manifest_path).resolve()
    replay_path = Path(replay_path).resolve()
    replay_manifest_path = Path(replay_manifest_path).resolve()
    output_dir = Path(output_dir).resolve()
    for path in (p2_manifest_path, replay_path, replay_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    requested = _requested_inputs(
        p2_manifest_path=p2_manifest_path,
        replay_path=replay_path,
        replay_manifest_path=replay_manifest_path,
        shuffle_seed=shuffle_seed,
        model_init_seed=model_init_seed,
    )
    metadata_path = output_dir / "metadata.json"
    if output_dir.exists():
        if not metadata_path.is_file():
            raise FileExistsError(
                f"existing scratch-replay directory is incomplete: {output_dir}"
            )
        existing = ScratchReplayManifest.load(metadata_path)
        actual = {
            "base_leg_sha256": existing.base_leg_sha256,
            "replay_sha256": existing.replay_sha256,
            "replay_manifest_sha256": existing.replay_manifest_sha256,
            "shuffle_seed": existing.shuffle_seed,
            "model_init_seed": existing.model_init_seed,
        }
        if actual != requested:
            raise ValueError(
                f"existing scratch-replay inputs differ: {actual} != {requested}"
            )
        return existing

    base = LegManifest.load(p2_manifest_path)
    if base.leg != "p2":
        raise ValueError(f"scratch replay requires a P2 leg, got {base.leg!r}")
    if base.physical_steps != base.total_steps:
        raise ValueError("scratch replay requires a full physical P2 manifest")
    base_order = np.load(base.order_path, mmap_mode="r", allow_pickle=False)
    base_real = np.asarray(
        base_order[base_order != int(PAD_RECORD)],
        dtype="<i8",
    )
    expected_base_records = base.pretrain_records + base.sft_records
    if len(base_real) != expected_base_records:
        raise ValueError(
            f"P2 non-padding records differ: {len(base_real)} "
            f"!= {expected_base_records}"
        )

    replay = PositiveReplayDataset(
        replay_path,
        manifest_path=replay_manifest_path,
        context_limit=base.sequence_length,
    )
    if validate_all_replay_rows:
        for index in range(len(replay)):
            # PositiveReplayDataset performs checksum, structure, token-ID,
            # ownership-mask, vocabulary, and context validation.
            replay[index]
        replay.close()
    if len(replay) >= (1 << 62):
        raise ValueError("positive replay corpus is too large for int64 encoding")
    replay_codes = REPLAY_CODE_BASE - np.arange(len(replay), dtype="<i8")
    order = np.concatenate((base_real, replay_codes))
    np.random.Generator(np.random.PCG64(int(shuffle_seed))).shuffle(order)
    global_batch = base.world_size * base.local_batch_size
    padding = (-len(order)) % global_batch
    if padding:
        order = np.concatenate(
            (order, np.full(padding, PAD_RECORD, dtype="<i8"))
        )
    physical_steps = len(order) // global_batch
    floor_tail_steps = physical_steps - base.total_steps
    if floor_tail_steps < 0:
        raise AssertionError("adding replay unexpectedly shortened P2")

    output_dir.mkdir(parents=True)
    _atomic_save_npy(output_dir / "order.npy", order)
    order_sha256 = _sha256_file(output_dir / "order.npy")
    value: dict[str, Any] = {
        "schema": SCRATCH_REPLAY_SCHEMA,
        "schema_version": SCRATCH_REPLAY_SCHEMA_VERSION,
        "kind": "exp4_scratch_p2_plus_positive_replay",
        "order_file": "order.npy",
        "order_encoding": {
            "pretrain": "nonnegative P2 local packed-record index",
            "sft": "negative P2 global SFT row encoding",
            "positive_replay": f"{REPLAY_CODE_BASE}-dataset_index",
            "padding": int(PAD_RECORD),
        },
        "base_leg_path": os.path.relpath(p2_manifest_path, start=output_dir),
        "replay_path": os.path.relpath(replay_path, start=output_dir),
        "replay_manifest_path": os.path.relpath(
            replay_manifest_path, start=output_dir
        ),
        "sequence_length": base.sequence_length,
        "pretrain_records": base.pretrain_records,
        "sft_records": base.sft_records,
        "replay_records": len(replay),
        "base_nonpadding_records": expected_base_records,
        "padding_records": int(padding),
        "num_order_records": len(order),
        "world_size": base.world_size,
        "local_batch_size": base.local_batch_size,
        "global_batch_size": global_batch,
        "baseline_cosine_steps": base.total_steps,
        "floor_tail_steps": floor_tail_steps,
        "physical_steps": physical_steps,
        "total_steps": physical_steps,
        "lr_contract": {
            "warmup_cosine_steps": base.total_steps,
            "tail_steps": floor_tail_steps,
            "tail_lr": 1e-5,
            "note": (
                "Unified shuffle spans every record. Tail denotes optimizer "
                "time after the unchanged P2 cosine arc, not replay placement."
            ),
        },
        "shuffle_algorithm": "numpy-pcg64-single-sample-level-permutation-v1",
        "shuffle_seed": int(shuffle_seed),
        "model_init": "random_exact_interleaved_qwen_47_245_312",
        "model_init_seed": int(model_init_seed),
        "source_manifest_hash": base.source_manifest_hash,
        "selection_hash": base.selection_hash,
        "sft_cache_hash": base.sft_cache_hash,
        **requested,
        "order_sha256": order_sha256,
    }
    value["metadata_hash"] = _hash_dict(value, "metadata_hash")
    _atomic_json(metadata_path, value)
    return ScratchReplayManifest.load(metadata_path)


def replay_row_to_aligned_sample(
    row: Mapping[str, Any],
    *,
    record_id: int,
) -> dict[str, Any]:
    """Convert exact rollout token artifacts to the trainer's aligned CE form."""

    validate_replay_record(row)
    prompt = list(row["prompt_token_ids"])
    response = list(row["response_token_ids"])
    ownership = [bool(value) for value in row["response_loss_mask"]]
    tokens = prompt + response
    input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
    targets = tokens[1:]
    target_ownership = [False] * (len(prompt) - 1) + ownership
    labels = torch.tensor(
        [
            target if is_model_owned else -100
            for target, is_model_owned in zip(
                targets, target_ownership, strict=True
            )
        ],
        dtype=torch.long,
    )
    valid_targets = int(labels.ne(-100).sum().item())
    if valid_targets <= 0:
        raise ValueError("positive replay row has no supervised target")
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
        "sample_type": SAMPLE_POSITIVE_REPLAY,
        "record_id": int(record_id),
        "valid_targets": valid_targets,
    }


class ScratchReplayDataset(Dataset):
    """Decode one immutable combined order into PT, SFT, replay, or padding."""

    def __init__(
        self,
        *,
        pretrain: PackedPretrainDataset,
        sft: SFTCacheDataset,
        replay: PositiveReplayDataset,
        manifest: ScratchReplayManifest,
        order: np.ndarray,
    ) -> None:
        self.pretrain = pretrain
        self.sft = sft
        self.replay = replay
        self.manifest = manifest
        self.order = order

    def __len__(self) -> int:
        return len(self.order)

    def __getitem__(self, position: int) -> dict[str, Any]:
        code = int(self.order[position])
        if code == int(PAD_RECORD):
            sample = {
                "input_ids": torch.empty(0, dtype=torch.long),
                "labels": torch.empty(0, dtype=torch.long),
                "attention_mask": torch.empty(0, dtype=torch.long),
                "sample_type": SAMPLE_PAD,
                "record_id": code,
                "valid_targets": 0,
            }
        elif code >= 0:
            sample = self.pretrain[code]
        elif code > REPLAY_CODE_BASE:
            sample = self.sft[-code - 1]
        else:
            replay_index = REPLAY_CODE_BASE - code
            sample = replay_row_to_aligned_sample(
                self.replay[replay_index],
                record_id=replay_index,
            )
        sample["manifest_position"] = int(position)
        return sample


class ScratchReplayDataStream(InterleavedDataStream):
    """Interleaved stream whose state explicitly binds replay and LR tail."""

    def __init__(self, *, scratch_manifest: ScratchReplayManifest, **kwargs):
        super().__init__(**kwargs)
        self.scratch_manifest = scratch_manifest
        self.baseline_cosine_steps = scratch_manifest.baseline_cosine_steps
        self.floor_tail_steps = scratch_manifest.floor_tail_steps
        self.requires_random_init = True
        self.model_init_seed = scratch_manifest.model_init_seed
        self.replay_sha256 = scratch_manifest.replay_sha256
        self.base_leg_sha256 = scratch_manifest.base_leg_sha256

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state.update(
            {
                "scratch_replay_schema": SCRATCH_REPLAY_SCHEMA_VERSION,
                "base_leg_sha256": self.base_leg_sha256,
                "replay_sha256": self.replay_sha256,
                "baseline_cosine_steps": self.baseline_cosine_steps,
                "floor_tail_steps": self.floor_tail_steps,
                "model_init_seed": self.model_init_seed,
            }
        )
        return state

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected = {
            "scratch_replay_schema": SCRATCH_REPLAY_SCHEMA_VERSION,
            "base_leg_sha256": self.base_leg_sha256,
            "replay_sha256": self.replay_sha256,
            "baseline_cosine_steps": self.baseline_cosine_steps,
            "floor_tail_steps": self.floor_tail_steps,
            "model_init_seed": self.model_init_seed,
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise ValueError(
                    f"scratch replay resume mismatch for {key}: "
                    f"{state.get(key)!r} != {value!r}"
                )
        super().load_state_dict(state)


def create_scratch_replay_dataloader(
    *,
    source_root: str | Path,
    source_manifest_path: str | Path,
    selection_manifest_path: str | Path,
    sft_cache_dir: str | Path,
    leg_manifest_path: str | Path,
    pad_token_id: int,
    bos_token_id: int,
    rank: int,
    world_size: int,
    local_batch_size: int,
    start_cursor: int = 0,
    num_workers: int = 0,
    max_open_shards: int = 64,
) -> ScratchReplayDataStream:
    """Open a rank-local view of an immutable scratch-replay manifest."""

    manifest = ScratchReplayManifest.load(leg_manifest_path)
    source = SourceShardManifest.load(source_manifest_path)
    selection = PretrainSelection.load(selection_manifest_path)
    cache = SFTCache.load(sft_cache_dir, verify_large_files=False)
    if (
        source.manifest_hash != manifest.source_manifest_hash
        or selection.selection_hash != manifest.selection_hash
        or cache.cache_hash != manifest.sft_cache_hash
    ):
        raise ValueError("scratch replay data artifacts differ from its manifest")
    if selection.source_manifest_hash != source.manifest_hash:
        raise ValueError("selection was built from a different source manifest")
    if (int(world_size), int(local_batch_size)) != (
        manifest.world_size,
        manifest.local_batch_size,
    ):
        raise ValueError(
            "runtime topology differs from immutable scratch topology: "
            f"{(world_size, local_batch_size)} != "
            f"{(manifest.world_size, manifest.local_batch_size)}"
        )

    base = LegManifest.load(manifest.base_leg_path)
    logical = LogicalTokenSelection(
        source_root,
        source,
        selection,
        max_open_shards=max_open_shards,
    )
    pretrain = PackedPretrainDataset(
        logical,
        target_start=base.target_start,
        target_count=base.target_count,
        bos_token_id=bos_token_id,
        sequence_length=manifest.sequence_length,
    )
    sft = SFTCacheDataset(cache)
    replay = PositiveReplayDataset(
        manifest.replay_path,
        manifest_path=manifest.replay_manifest_path,
        context_limit=manifest.sequence_length,
    )
    order = np.load(manifest.order_path, mmap_mode="r", allow_pickle=False)
    dataset = ScratchReplayDataset(
        pretrain=pretrain,
        sft=sft,
        replay=replay,
        manifest=manifest,
        order=order,
    )
    sampler = DistributedManifestBatchSampler(
        num_records=len(order),
        rank=rank,
        world_size=world_size,
        local_batch_size=local_batch_size,
        start_cursor=start_cursor,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=UnifiedInterleavedCollator(
            sequence_length=manifest.sequence_length,
            pad_token_id=pad_token_id,
        ),
        num_workers=int(num_workers),
        pin_memory=True,
        persistent_workers=bool(num_workers),
    )
    return ScratchReplayDataStream(
        scratch_manifest=manifest,
        dataloader=loader,
        sampler=sampler,
        manifest=manifest,
        manifest_file_hash=_sha256_file(manifest.metadata_path),
        rank=rank,
    )


__all__ = [
    "REPLAY_CODE_BASE",
    "SAMPLE_POSITIVE_REPLAY",
    "SCRATCH_REPLAY_SCHEMA",
    "ScratchReplayDataStream",
    "ScratchReplayDataset",
    "ScratchReplayManifest",
    "build_scratch_replay_manifest",
    "create_scratch_replay_dataloader",
    "replay_row_to_aligned_sample",
]
