"""Hard-SFT and soft-KL transfer components for Exp 4 positive replay.

Both transfer modes consume the same immutable replay JSONL and the same
deterministic rank-local batches.  Replay rows retain the exact token IDs and
response loss masks emitted during Miles rollout; no text is re-tokenized.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from .positive_replay import (
    SCHEMA_VERSION,
    canonical_json,
    sha256_file,
    token_ids_sha256,
)

IGNORE_INDEX = -100
EXPECTED_PARAMETER_COUNT = 47_245_312
EXPECTED_VOCAB_SIZE = 85
EXPECTED_CONTEXT_LENGTH = 3_072
TRANSFER_STATE_SCHEMA = 1
TRANSFER_RESUME_SCHEMA = 1
SUPPORTED_ATTENTION_BACKENDS = frozenset({"sdpa", "flash_attention_2"})


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def validate_replay_record(
    row: Mapping[str, Any],
    *,
    vocab_size: int = EXPECTED_VOCAB_SIZE,
    context_limit: int = EXPECTED_CONTEXT_LENGTH,
) -> None:
    if row.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("positive replay schema version mismatch")
    for key in (
        "prompt_token_ids",
        "response_token_ids",
        "response_loss_mask",
    ):
        value = row.get(key)
        if not isinstance(value, list) or any(
            isinstance(item, bool) or not isinstance(item, int) for item in value
        ):
            raise ValueError(f"invalid replay field {key}")
    prompt_ids = row["prompt_token_ids"]
    response_ids = row["response_token_ids"]
    response_mask = row["response_loss_mask"]
    if not prompt_ids or not response_ids:
        raise ValueError("positive replay token sequences cannot be empty")
    if len(response_ids) != len(response_mask):
        raise ValueError("positive replay response mask length mismatch")
    if any(value not in (0, 1) for value in response_mask):
        raise ValueError("positive replay response mask must be binary")
    if not any(response_mask):
        raise ValueError("positive replay has no model-owned target tokens")
    if len(prompt_ids) + len(response_ids) > context_limit:
        raise ValueError("positive replay exceeds model context")
    if any(
        token_id < 0 or token_id >= vocab_size
        for token_id in (*prompt_ids, *response_ids)
    ):
        raise ValueError("positive replay token ID is outside the vocabulary")
    if row.get("token_ids_sha256") != token_ids_sha256(
        prompt_ids,
        response_ids,
    ):
        raise ValueError("positive replay token hash mismatch")


class PositiveReplayDataset(Dataset):
    """Indexed JSONL dataset with optional checksum-manifest verification."""

    def __init__(
        self,
        replay_path: os.PathLike[str] | str,
        *,
        manifest_path: os.PathLike[str] | str | None = None,
        vocab_size: int = EXPECTED_VOCAB_SIZE,
        context_limit: int = EXPECTED_CONTEXT_LENGTH,
    ) -> None:
        self.path = Path(replay_path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.vocab_size = int(vocab_size)
        self.context_limit = int(context_limit)
        self.manifest_path = (
            Path(manifest_path).expanduser().resolve()
            if manifest_path is not None
            else None
        )
        self.replay_sha256 = sha256_file(self.path)
        if self.manifest_path is not None:
            manifest = _read_json(self.manifest_path)
            output = manifest.get("output")
            if not isinstance(output, Mapping):
                raise ValueError("replay manifest has no output provenance")
            if output.get("sha256") != self.replay_sha256:
                raise ValueError("replay JSONL checksum does not match manifest")
            self.provenance = manifest
        else:
            self.provenance = None

        self._offsets: list[int] = []
        with self.path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if line.strip():
                    self._offsets.append(offset)
        if not self._offsets:
            raise ValueError("positive replay corpus is empty")
        if self.provenance is not None:
            expected_rows = int(self.provenance["output"]["rows"])
            if expected_rows != len(self._offsets):
                raise ValueError(
                    "replay row count does not match manifest: "
                    f"{len(self._offsets)} != {expected_rows}"
                )
        self._handle = None

    def __len__(self) -> int:
        return len(self._offsets)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handle"] = None
        return state

    def _open(self):
        if self._handle is None or self._handle.closed:
            self._handle = self.path.open("rb")
        return self._handle

    def close(self) -> None:
        """Close the process-local lazy JSONL handle, if it was opened."""
        if self._handle is not None and not self._handle.closed:
            self._handle.close()
        self._handle = None

    def __del__(self):
        # DataLoader worker teardown and interpreter shutdown are not
        # guaranteed to run context managers around Dataset instances.
        handle = getattr(self, "_handle", None)
        if handle is not None and not handle.closed:
            handle.close()

    def __getitem__(self, index: int) -> dict[str, Any]:
        # The distributed sampler uses -1 as an all-ignore padding sentinel.
        if index == -1:
            return {"is_padding": True}
        if index < 0 or index >= len(self._offsets):
            raise IndexError(index)
        handle = self._open()
        handle.seek(self._offsets[index])
        row = json.loads(handle.readline())
        if not isinstance(row, dict):
            raise ValueError(f"positive replay row {index} is not an object")
        validate_replay_record(
            row,
            vocab_size=self.vocab_size,
            context_limit=self.context_limit,
        )
        row["dataset_index"] = index
        return row


class DeterministicDistributedBatchSampler(Sampler[list[int]]):
    """One exact global permutation, sliced into equal DDP rank batches.

    The final global batch is padded with ``-1`` all-ignore sentinels.  No
    positive trajectory is repeated or dropped.
    """

    def __init__(
        self,
        dataset_size: int,
        *,
        local_batch_size: int,
        rank: int,
        world_size: int,
        seed: int,
        epoch: int = 0,
        start_batch: int = 0,
    ) -> None:
        if dataset_size <= 0:
            raise ValueError("dataset_size must be positive")
        if local_batch_size <= 0 or world_size <= 0:
            raise ValueError("batch and world sizes must be positive")
        if rank < 0 or rank >= world_size:
            raise ValueError("rank is outside world_size")
        self.dataset_size = int(dataset_size)
        self.local_batch_size = int(local_batch_size)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.epoch = int(epoch)
        self.set_start_batch(start_batch)

    @property
    def global_batch_size(self) -> int:
        return self.local_batch_size * self.world_size

    @property
    def padding_records(self) -> int:
        return (-self.dataset_size) % self.global_batch_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def set_start_batch(self, start_batch: int) -> None:
        value = int(start_batch)
        if value < 0 or value > len(self):
            raise ValueError(f"start_batch must be in [0, {len(self)}], got {value}")
        self.start_batch = value

    def __len__(self) -> int:
        return math.ceil(self.dataset_size / self.global_batch_size)

    def __iter__(self) -> Iterator[list[int]]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        order = torch.randperm(
            self.dataset_size,
            generator=generator,
        ).tolist()
        order.extend([-1] * self.padding_records)
        rank_offset = self.rank * self.local_batch_size
        for batch_index, start in enumerate(
            range(0, len(order), self.global_batch_size)
        ):
            if batch_index < self.start_batch:
                continue
            local_start = start + rank_offset
            yield order[local_start : local_start + self.local_batch_size]


def collate_positive_replay(
    rows: Sequence[Mapping[str, Any]],
    *,
    pad_token_id: int,
    context_limit: int = EXPECTED_CONTEXT_LENGTH,
) -> dict[str, Any]:
    """Create causal inputs without re-tokenizing or supervising env replies."""
    if not rows:
        raise ValueError("cannot collate an empty replay batch")
    encoded: list[dict[str, Any]] = []
    max_input_length = 1
    for row in rows:
        if row.get("is_padding"):
            encoded.append(
                {
                    "input_ids": [int(pad_token_id)],
                    "labels": [IGNORE_INDEX],
                    "model_owned_mask": [False],
                    # A real attention position avoids all-masked attention on
                    # ranks whose final local batch consists only of sentinels.
                    "attention_mask": [1],
                    "dataset_index": -1,
                    "is_padding": True,
                }
            )
            continue
        validate_replay_record(row, context_limit=context_limit)
        prompt = list(row["prompt_token_ids"])
        response = list(row["response_token_ids"])
        response_mask = list(row["response_loss_mask"])
        tokens = prompt + response
        # logits[t] predicts tokens[t + 1].  The first response token is
        # therefore supervised at the last prompt input position.
        target_mask = [False] * (len(prompt) - 1) + [
            bool(value) for value in response_mask
        ]
        input_ids = tokens[:-1]
        targets = tokens[1:]
        labels = [
            target if owned else IGNORE_INDEX
            for target, owned in zip(targets, target_mask, strict=True)
        ]
        if len(input_ids) > context_limit:
            raise ValueError("replay causal input exceeds context")
        max_input_length = max(max_input_length, len(input_ids))
        encoded.append(
            {
                "input_ids": input_ids,
                "labels": labels,
                "model_owned_mask": target_mask,
                "attention_mask": [1] * len(input_ids),
                "dataset_index": int(row.get("dataset_index", -1)),
                "is_padding": False,
            }
        )

    batch_size = len(encoded)
    input_tensor = torch.full(
        (batch_size, max_input_length),
        int(pad_token_id),
        dtype=torch.long,
    )
    label_tensor = torch.full(
        (batch_size, max_input_length),
        IGNORE_INDEX,
        dtype=torch.long,
    )
    attention_tensor = torch.zeros(
        (batch_size, max_input_length),
        dtype=torch.bool,
    )
    owned_tensor = torch.zeros(
        (batch_size, max_input_length),
        dtype=torch.bool,
    )
    for index, row in enumerate(encoded):
        length = len(row["input_ids"])
        input_tensor[index, :length] = torch.tensor(
            row["input_ids"],
            dtype=torch.long,
        )
        label_tensor[index, :length] = torch.tensor(
            row["labels"],
            dtype=torch.long,
        )
        attention_tensor[index, :length] = torch.tensor(
            row["attention_mask"],
            dtype=torch.bool,
        )
        owned_tensor[index, :length] = torch.tensor(
            row["model_owned_mask"],
            dtype=torch.bool,
        )
    return {
        "input_ids": input_tensor,
        "labels": label_tensor,
        "attention_mask": attention_tensor,
        "model_owned_mask": owned_tensor,
        "dataset_indices": torch.tensor(
            [row["dataset_index"] for row in encoded],
            dtype=torch.long,
        ),
        "padding_records": sum(bool(row["is_padding"]) for row in encoded),
    }


def hard_sft_loss_sum(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Masked hard cross-entropy sum over model-owned response targets."""
    if logits.ndim != 3 or labels.ndim != 2:
        raise ValueError("expected logits [B,S,V] and labels [B,S]")
    if logits.shape[:2] != labels.shape:
        raise ValueError("hard-SFT logits and labels shapes differ")
    valid_tokens = labels.ne(IGNORE_INDEX).sum()
    loss_sum = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
        reduction="sum",
    )
    return loss_sum, valid_tokens


def forward_kl_loss_sum(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    model_owned_mask: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Full-vocabulary ``KL(teacher || student)`` on model-owned positions."""
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student and teacher logits shapes differ")
    if student_logits.ndim != 3:
        raise ValueError("expected student and teacher logits [B,S,V]")
    if model_owned_mask.shape != student_logits.shape[:2]:
        raise ValueError("KL mask shape differs from logits")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    scaled_teacher = teacher_logits.detach().float() / float(temperature)
    scaled_student = student_logits.float() / float(temperature)
    teacher_log_probs = F.log_softmax(scaled_teacher, dim=-1)
    teacher_probs = teacher_log_probs.exp()
    student_log_probs = F.log_softmax(scaled_student, dim=-1)
    token_kl = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(
        dim=-1
    ) * float(temperature) ** 2
    mask = model_owned_mask.to(dtype=torch.bool)
    loss_sum = token_kl.masked_select(mask).sum()
    return loss_sum, mask.sum()


def globally_normalized_ddp_loss(
    local_loss_sum: torch.Tensor,
    *,
    global_valid_tokens: torch.Tensor | int,
    world_size: int,
) -> torch.Tensor:
    """Compensate for DDP gradient averaging to obtain a global token mean."""
    if isinstance(global_valid_tokens, torch.Tensor):
        if global_valid_tokens.numel() != 1:
            raise ValueError("global_valid_tokens must be scalar")
        count = int(global_valid_tokens.detach().item())
        denominator = global_valid_tokens.to(
            device=local_loss_sum.device,
            dtype=local_loss_sum.dtype,
        )
    else:
        count = int(global_valid_tokens)
        denominator = local_loss_sum.new_tensor(count)
    if count <= 0:
        raise ValueError("transfer batch has no model-owned response tokens")
    return local_loss_sum * int(world_size) / denominator


def validate_interleaved_50m_model(model: torch.nn.Module) -> None:
    config = getattr(model, "config", None)
    actual_parameters = sum(parameter.numel() for parameter in model.parameters())
    errors: list[str] = []
    if actual_parameters != EXPECTED_PARAMETER_COUNT:
        errors.append(
            f"parameters={actual_parameters}, expected={EXPECTED_PARAMETER_COUNT}"
        )
    if int(getattr(config, "vocab_size", -1)) != EXPECTED_VOCAB_SIZE:
        errors.append(f"vocab_size={getattr(config, 'vocab_size', None)}")
    if int(getattr(config, "max_position_embeddings", -1)) < EXPECTED_CONTEXT_LENGTH:
        errors.append(
            "max_position_embeddings="
            f"{getattr(config, 'max_position_embeddings', None)}"
        )
    expected_config = {
        "hidden_size": 512,
        "intermediate_size": 1_536,
        "num_hidden_layers": 12,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "head_dim": 128,
    }
    for name, expected in expected_config.items():
        if int(getattr(config, name, -1)) != expected:
            errors.append(f"{name}={getattr(config, name, None)}")
    if errors:
        raise ValueError(
            "checkpoint is not the 47.245M interleaved Qwen model: " + ", ".join(errors)
        )


def validate_teacher_student_pair(
    student: torch.nn.Module,
    teacher: torch.nn.Module,
) -> None:
    """Require matching token semantics and architecture, ignoring runtime flags."""
    fields = (
        "model_type",
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "max_position_embeddings",
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "env_token_id",
        "tie_word_embeddings",
        "rope_theta",
    )
    mismatches = [
        (
            name,
            getattr(student.config, name, None),
            getattr(teacher.config, name, None),
        )
        for name in fields
        if getattr(student.config, name, None) != getattr(teacher.config, name, None)
    ]
    if mismatches:
        raise ValueError(
            "teacher/student architectures or token IDs differ: "
            + ", ".join(
                f"{name}={student_value!r}/{teacher_value!r}"
                for name, student_value, teacher_value in mismatches
            )
        )


@dataclass(frozen=True)
class PositiveTransferConfig:
    mode: str
    student_checkpoint: str
    replay_path: str
    output_dir: str
    learning_rate: float
    teacher_checkpoint: str | None = None
    replay_manifest: str | None = None
    local_batch_size: int = 21
    epochs: int = 1
    seed: int = 42
    weight_decay: float = 0.1
    temperature: float = 1.0
    max_steps: int | None = None
    num_workers: int = 0
    save_interval: int = 200
    checkpoint_dir: str | None = None
    resume_path: str | None = None
    run_fingerprint: str | None = None
    attn_implementation: str = "sdpa"
    flash_attention_version: str = "2.8.3"

    def validate(self) -> None:
        if self.mode not in {"hard_sft", "soft_kl"}:
            raise ValueError("mode must be hard_sft or soft_kl")
        if self.mode == "soft_kl" and not self.teacher_checkpoint:
            raise ValueError("soft_kl requires teacher_checkpoint")
        if self.mode == "hard_sft" and self.teacher_checkpoint:
            raise ValueError("hard_sft must not load a teacher")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be explicit and positive")
        if self.local_batch_size <= 0 or self.epochs <= 0:
            raise ValueError("batch size and epochs must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.save_interval < 0:
            raise ValueError("save_interval must be non-negative")
        if bool(self.resume_path) and not self.checkpoint_dir:
            raise ValueError("resume_path requires checkpoint_dir")
        if self.run_fingerprint is not None and not _is_sha256(self.run_fingerprint):
            raise ValueError("run_fingerprint must be a lowercase SHA-256")
        if self.attn_implementation not in SUPPORTED_ATTENTION_BACKENDS:
            raise ValueError("attn_implementation must be sdpa or flash_attention_2")
        if (
            self.attn_implementation == "flash_attention_2"
            and not self.flash_attention_version
        ):
            raise ValueError("flash_attention_2 requires a pinned package version")
        if self.attn_implementation == "flash_attention_2":
            try:
                installed = importlib.metadata.version("flash-attn")
            except importlib.metadata.PackageNotFoundError as error:
                raise RuntimeError(
                    "flash_attention_2 requires the pinned flash-attn package"
                ) from error
            if installed != self.flash_attention_version:
                raise RuntimeError(
                    "flash-attn package drift: "
                    f"{installed} != {self.flash_attention_version}"
                )


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha256_json(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _transfer_contract(
    config: PositiveTransferConfig,
    *,
    replay_sha256: str,
    replay_manifest_sha256: str | None,
    world_size: int,
) -> dict[str, Any]:
    """Return the immutable contract bound into every transfer checkpoint."""

    return {
        "mode": config.mode,
        "student_checkpoint": str(
            Path(config.student_checkpoint).expanduser().resolve()
        ),
        "teacher_checkpoint": (
            str(Path(config.teacher_checkpoint).expanduser().resolve())
            if config.teacher_checkpoint
            else None
        ),
        "replay_path": str(Path(config.replay_path).expanduser().resolve()),
        "replay_sha256": replay_sha256,
        "replay_manifest_path": (
            str(Path(config.replay_manifest).expanduser().resolve())
            if config.replay_manifest
            else None
        ),
        "replay_manifest_sha256": replay_manifest_sha256,
        "learning_rate": float(config.learning_rate),
        "local_batch_size": int(config.local_batch_size),
        "world_size": int(world_size),
        "gradient_accumulation_steps": 1,
        "epochs": int(config.epochs),
        "seed": int(config.seed),
        "weight_decay": float(config.weight_decay),
        "adam_betas": [0.9, 0.95],
        "adam_eps": 1e-8,
        "scheduler": "constant",
        "temperature": float(config.temperature),
        "max_steps": (int(config.max_steps) if config.max_steps is not None else None),
        "run_fingerprint": config.run_fingerprint,
        "attn_implementation": config.attn_implementation,
        "flash_attention_version": (
            config.flash_attention_version
            if config.attn_implementation == "flash_attention_2"
            else None
        ),
    }


def validate_transfer_resume_state(
    state: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    batches_per_epoch: int,
    target_steps: int,
) -> None:
    """Fail closed when a saved transfer cursor belongs to another run."""

    if int(state.get("schema_version", -1)) != TRANSFER_RESUME_SCHEMA:
        raise ValueError("positive-transfer resume schema mismatch")
    recorded_contract = state.get("contract")
    if not isinstance(recorded_contract, Mapping):
        raise ValueError("positive-transfer resume has no contract")
    if dict(recorded_contract) != dict(contract):
        raise ValueError("positive-transfer resume contract mismatch")
    if state.get("contract_sha256") != _sha256_json(dict(contract)):
        raise ValueError("positive-transfer resume contract hash mismatch")
    completed = int(state.get("completed_steps", -1))
    next_epoch = int(state.get("next_epoch", -1))
    next_batch = int(state.get("next_batch", -1))
    if completed < 0 or completed > target_steps:
        raise ValueError("positive-transfer completed_steps is invalid")
    expected_epoch, expected_batch = divmod(completed, batches_per_epoch)
    if completed == target_steps and target_steps < (
        int(contract["epochs"]) * batches_per_epoch
    ):
        # A max_steps boundary can occur in the middle of an epoch.
        expected_epoch, expected_batch = divmod(completed, batches_per_epoch)
    if (next_epoch, next_batch) != (expected_epoch, expected_batch):
        raise ValueError(
            "positive-transfer resume cursor mismatch: "
            f"{(next_epoch, next_batch)} != "
            f"{(expected_epoch, expected_batch)}"
        )
    counters = state.get("counters")
    if not isinstance(counters, Mapping):
        raise ValueError("positive-transfer resume has no counters")
    required_counters = (
        "processed_positive_examples",
        "processed_model_owned_tokens",
        "padding_records",
        "teacher_forward_examples",
    )
    if any(int(counters.get(key, -1)) < 0 for key in required_counters):
        raise ValueError("positive-transfer resume counters are invalid")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _save_transfer_resume(
    *,
    accelerator: Any,
    checkpoint_dir: Path,
    contract: Mapping[str, Any],
    completed_steps: int,
    batches_per_epoch: int,
    counters: Mapping[str, int],
) -> None:
    """Save optimizer/model/RNG first and publish the cursor marker last."""

    accelerator.wait_for_everyone()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    accelerator.save_state(str(checkpoint_dir))
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        next_epoch, next_batch = divmod(completed_steps, batches_per_epoch)
        _atomic_json(
            checkpoint_dir / "positive_transfer_resume.json",
            {
                "schema_version": TRANSFER_RESUME_SCHEMA,
                "kind": "exp4_positive_transfer_resume",
                "contract": dict(contract),
                "contract_sha256": _sha256_json(dict(contract)),
                "completed_steps": int(completed_steps),
                "next_epoch": next_epoch,
                "next_batch": next_batch,
                "counters": {key: int(value) for key, value in counters.items()},
            },
        )
    accelerator.wait_for_everyone()


def run_positive_transfer(config: PositiveTransferConfig) -> dict[str, Any]:
    """Run hard-SFT/forward-KL with strict, same-plan resumability."""

    config.validate()
    from accelerate import Accelerator
    from transformers import AutoModelForCausalLM, AutoTokenizer

    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=1,
    )
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    dataset = PositiveReplayDataset(
        config.replay_path,
        manifest_path=config.replay_manifest,
    )
    sampler = DeterministicDistributedBatchSampler(
        len(dataset),
        local_batch_size=config.local_batch_size,
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
        seed=config.seed,
    )
    batches_per_epoch = len(sampler)
    full_steps = config.epochs * batches_per_epoch
    target_steps = min(
        full_steps,
        config.max_steps if config.max_steps is not None else full_steps,
    )
    replay_manifest_sha256 = (
        sha256_file(config.replay_manifest)
        if config.replay_manifest is not None
        else None
    )
    contract = _transfer_contract(
        config,
        replay_sha256=dataset.replay_sha256,
        replay_manifest_sha256=replay_manifest_sha256,
        world_size=accelerator.num_processes,
    )

    student = AutoModelForCausalLM.from_pretrained(
        config.student_checkpoint,
        torch_dtype=torch.float32,
        attn_implementation=config.attn_implementation,
    )
    validate_interleaved_50m_model(student)
    tokenizer = AutoTokenizer.from_pretrained(
        config.student_checkpoint,
        trust_remote_code=True,
        use_fast=False,
    )
    if len(tokenizer.get_vocab()) != EXPECTED_VOCAB_SIZE:
        raise ValueError(
            f"student tokenizer has {len(tokenizer.get_vocab())} tokens, "
            f"expected {EXPECTED_VOCAB_SIZE}"
        )
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("student checkpoint has no pad_token_id")
    if getattr(student.config, "pad_token_id", None) != pad_token_id:
        raise ValueError("student model/tokenizer pad_token_id mismatch")

    teacher = None
    if config.mode == "soft_kl":
        teacher = AutoModelForCausalLM.from_pretrained(
            config.teacher_checkpoint,
            torch_dtype=torch.float32,
            attn_implementation=config.attn_implementation,
        )
        validate_interleaved_50m_model(teacher)
        validate_teacher_student_pair(student, teacher)
        teacher_tokenizer = AutoTokenizer.from_pretrained(
            config.teacher_checkpoint,
            trust_remote_code=True,
            use_fast=False,
        )
        if teacher_tokenizer.get_vocab() != tokenizer.get_vocab():
            raise ValueError("teacher/student tokenizer ID mappings differ")
        teacher.requires_grad_(False)
        teacher.eval()

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=config.weight_decay,
    )
    student, optimizer = accelerator.prepare(student, optimizer)
    if teacher is not None:
        # A fully frozen module must not be wrapped in DDP; each rank keeps one
        # inference-only teacher replica.
        teacher.to(accelerator.device)

    completed_steps = 0
    counters = {
        "processed_positive_examples": 0,
        "processed_model_owned_tokens": 0,
        "padding_records": 0,
        "teacher_forward_examples": 0,
    }
    checkpoint_dir = (
        Path(config.checkpoint_dir).expanduser().resolve()
        if config.checkpoint_dir is not None
        else None
    )
    if config.resume_path:
        resume_path = Path(config.resume_path).expanduser().resolve()
        state_path = resume_path / "positive_transfer_resume.json"
        if not state_path.is_file():
            raise FileNotFoundError(f"positive-transfer resume requires {state_path}")
        resume_state = _read_json(state_path)
        validate_transfer_resume_state(
            resume_state,
            contract=contract,
            batches_per_epoch=batches_per_epoch,
            target_steps=target_steps,
        )
        accelerator.load_state(str(resume_path))
        completed_steps = int(resume_state["completed_steps"])
        counters = {key: int(resume_state["counters"][key]) for key in counters}

    student.train()
    stop = completed_steps >= target_steps
    start_epoch, start_batch = divmod(completed_steps, batches_per_epoch)
    for epoch in range(start_epoch, config.epochs):
        if stop:
            break
        sampler.set_epoch(epoch)
        sampler.set_start_batch(start_batch if epoch == start_epoch else 0)
        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=config.num_workers,
            collate_fn=partial(
                collate_positive_replay,
                pad_token_id=int(pad_token_id),
            ),
        )
        for batch in loader:
            input_ids = batch["input_ids"].to(accelerator.device)
            attention_mask = batch["attention_mask"].to(accelerator.device)
            labels = batch["labels"].to(accelerator.device)
            owned_mask = batch["model_owned_mask"].to(accelerator.device)
            dataset_indices = batch["dataset_indices"].to(accelerator.device)

            optimizer.zero_grad(set_to_none=True)
            with accelerator.autocast():
                student_logits = student(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                ).logits
                if config.mode == "hard_sft":
                    local_loss_sum, local_count = hard_sft_loss_sum(
                        student_logits,
                        labels,
                    )
                else:
                    assert teacher is not None
                    with torch.no_grad():
                        teacher_logits = teacher(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            use_cache=False,
                        ).logits
                    local_loss_sum, local_count = forward_kl_loss_sum(
                        student_logits,
                        teacher_logits,
                        owned_mask,
                        temperature=config.temperature,
                    )

            global_count = local_count.detach().to(
                device=accelerator.device,
                dtype=torch.long,
            )
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(global_count, op=dist.ReduceOp.SUM)
            backward_loss = globally_normalized_ddp_loss(
                local_loss_sum,
                global_valid_tokens=global_count,
                world_size=accelerator.num_processes,
            )
            accelerator.backward(backward_loss)
            optimizer.step()

            real_examples = int(dataset_indices.ge(0).sum().item())
            step_metrics = torch.tensor(
                [
                    real_examples,
                    int(local_count.detach().item()),
                    int(batch["padding_records"]),
                    real_examples if config.mode == "soft_kl" else 0,
                ],
                dtype=torch.long,
                device=accelerator.device,
            )
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(step_metrics, op=dist.ReduceOp.SUM)
            step_totals = [int(value) for value in step_metrics.detach().cpu().tolist()]
            for key, value in zip(counters, step_totals, strict=True):
                counters[key] += value
            completed_steps += 1

            if (
                checkpoint_dir is not None
                and config.save_interval > 0
                and completed_steps % config.save_interval == 0
            ):
                _save_transfer_resume(
                    accelerator=accelerator,
                    checkpoint_dir=checkpoint_dir,
                    contract=contract,
                    completed_steps=completed_steps,
                    batches_per_epoch=batches_per_epoch,
                    counters=counters,
                )
            if completed_steps >= target_steps:
                stop = True
                break
        start_batch = 0

    if checkpoint_dir is not None:
        _save_transfer_resume(
            accelerator=accelerator,
            checkpoint_dir=checkpoint_dir,
            contract=contract,
            completed_steps=completed_steps,
            batches_per_epoch=batches_per_epoch,
            counters=counters,
        )

    output_dir = Path(config.output_dir).expanduser().resolve()
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        unwrapped = accelerator.unwrap_model(student)
        unwrapped.config.use_cache = True
        state_dict = accelerator.get_state_dict(student)
        unwrapped.save_pretrained(
            output_dir,
            state_dict=state_dict,
            safe_serialization=True,
        )
        tokenizer.save_pretrained(output_dir)
        state = {
            "schema_version": TRANSFER_STATE_SCHEMA,
            "kind": "exp4_positive_transfer",
            "config": asdict(config),
            "replay_sha256": dataset.replay_sha256,
            "replay_manifest_sha256": replay_manifest_sha256,
            "contract": contract,
            "contract_sha256": _sha256_json(contract),
            "completed_steps": completed_steps,
            **counters,
        }
        _atomic_json(output_dir / "positive_transfer_state.json", state)
    accelerator.wait_for_everyone()
    return {
        "completed_steps": completed_steps,
        **counters,
        "output_dir": str(output_dir),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Exp 4 hard-SFT or soft-KL positive transfer"
    )
    parser.add_argument("--mode", choices=("hard_sft", "soft_kl"), required=True)
    parser.add_argument("--student-checkpoint", required=True)
    parser.add_argument("--teacher-checkpoint")
    parser.add_argument("--replay", required=True)
    parser.add_argument("--replay-manifest")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--local-batch-size", type=int, default=21)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-interval", type=int, default=200)
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--resume")
    parser.add_argument("--run-fingerprint")
    parser.add_argument(
        "--attn-implementation",
        choices=sorted(SUPPORTED_ATTENTION_BACKENDS),
        default="sdpa",
    )
    parser.add_argument("--flash-attention-version", default="2.8.3")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = run_positive_transfer(
        PositiveTransferConfig(
            mode=args.mode,
            student_checkpoint=args.student_checkpoint,
            teacher_checkpoint=args.teacher_checkpoint,
            replay_path=args.replay,
            replay_manifest=args.replay_manifest,
            output_dir=args.output_dir,
            learning_rate=args.learning_rate,
            local_batch_size=args.local_batch_size,
            epochs=args.epochs,
            seed=args.seed,
            weight_decay=args.weight_decay,
            temperature=args.temperature,
            max_steps=args.max_steps,
            num_workers=args.num_workers,
            save_interval=args.save_interval,
            checkpoint_dir=args.checkpoint_dir,
            resume_path=args.resume,
            run_fingerprint=args.run_fingerprint,
            attn_implementation=args.attn_implementation,
            flash_attention_version=args.flash_attention_version,
        )
    )
    print(canonical_json(result))


if __name__ == "__main__":
    main()
