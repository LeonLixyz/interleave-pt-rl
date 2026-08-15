"""Deterministic mixed pretraining/SFT data and immutable manifests.

This module intentionally does not depend on the legacy ``mixed_trainer``.
It supports configurable context lengths, target ranges, batch topology, and
training-leg layouts.  A stream record is one next-token-aligned pretraining
sequence, one independently masked SFT row, or a numerically safe all-ignore
padding record.  Pretraining target ranges include the preceding context token
without repeating a next-token target across adjacent ranges.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
import os
import random
import re
import shutil
import tempfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler


SCHEMA_VERSION = 1
DEFAULT_SEQUENCE_LENGTH = 3_072
DEFAULT_TOTAL_TARGETS = 10_000_000_000
DEFAULT_LEG_TARGETS = 5_000_000_000
DEFAULT_SFT_ROWS = 77_717
DEFAULT_WORLD_SIZE = 8
DEFAULT_LOCAL_BATCH_SIZE = 21

PAD_RECORD = np.iinfo(np.int64).min
SAMPLE_PAD = 0
SAMPLE_PRETRAIN = 1
SAMPLE_SFT = 2

_SHARD_NUMBER_RE = re.compile(r"(\d+)(?=\.npy$)")
_VERIFY_SCORE_PAIR_RE = re.compile(
    r"\s*<verify>\s*<[+-]?(?:\d+(?:\.\d+)?|\.\d+)>"
)
_VERIFY_TOKEN_RE = re.compile(r"<verify>")
SFT_RESPONSE_NORMALIZATION_STRIP_VERIFY_V1 = (
    "strip-numeric-verify-score-pairs-normalize-whitespace-v1"
)
SFT_SUPERVISED_UNK_POLICY_REJECT_V1 = "reject-supervised-unk-v1"
SFT_STRICT_AUDIT_SCHEMA_V1 = "interleaved-sft-strict-audit-v1"
SFT_SUPERVISED_DELIMITERS = (
    "<T>",
    "</T>",
    "<sep>",
    "<call_env>",
    "<eos>",
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _hash_dict(value: Mapping[str, Any], hash_field: str) -> str:
    unhashed = {key: item for key, item in value.items() if key != hash_field}
    return hashlib.sha256(_canonical_json(unhashed)).hexdigest()


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
            json.dump(value, handle, indent=2, sort_keys=True)
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
        np.save(temporary, value, allow_pickle=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _numeric_shard_key(path: Path) -> tuple[int, str]:
    match = _SHARD_NUMBER_RE.search(path.name)
    if match is None:
        raise ValueError(f"Shard name has no numeric .npy suffix: {path.name}")
    return int(match.group(1)), path.name


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _checked_metadata(path: Path, hash_field: str, schema: str) -> dict[str, Any]:
    value = _load_json(path)
    if value.get("schema") != schema:
        raise ValueError(
            f"Unexpected schema in {path}: {value.get('schema')!r}, expected {schema!r}"
        )
    expected = value.get(hash_field)
    actual = _hash_dict(value, hash_field)
    if not expected or expected != actual:
        raise ValueError(f"{hash_field} mismatch in {path}: {expected} != {actual}")
    return value


@dataclass(frozen=True)
class SourceShard:
    shard_number: int
    relative_path: str
    num_tokens: int
    dtype: str
    byte_size: int
    content_sha256: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "shard_number": self.shard_number,
            "relative_path": self.relative_path,
            "num_tokens": self.num_tokens,
            "dtype": self.dtype,
            "byte_size": self.byte_size,
            "content_sha256": self.content_sha256,
        }


@dataclass(frozen=True)
class SourceShardManifest:
    path: Path
    shards: tuple[SourceShard, ...]
    total_tokens: int
    manifest_hash: str
    content_hashes: bool

    @classmethod
    def load(cls, path: str | Path) -> "SourceShardManifest":
        path = Path(path)
        value = _checked_metadata(
            path, "manifest_hash", "interleaved-source-shards-v1"
        )
        shards = tuple(SourceShard(**item) for item in value["shards"])
        if sum(shard.num_tokens for shard in shards) != int(value["total_tokens"]):
            raise ValueError(f"Source token total is inconsistent in {path}")
        numbers = [shard.shard_number for shard in shards]
        if numbers != sorted(numbers) or len(numbers) != len(set(numbers)):
            raise ValueError(f"Source shards are not uniquely numeric-sorted in {path}")
        return cls(
            path=path,
            shards=shards,
            total_tokens=int(value["total_tokens"]),
            manifest_hash=value["manifest_hash"],
            content_hashes=bool(value["content_hashes"]),
        )


def build_source_manifest(
    source_root: str | Path,
    output_path: str | Path,
    *,
    pattern: str = "raw.*.npy",
    content_hashes: bool = False,
    trusted_npy_dtype: str | np.dtype[Any] | None = None,
    trusted_npy_header_bytes: int | None = None,
    expected_total_tokens: int | None = None,
) -> SourceShardManifest:
    """Scan one corpus, numeric-sort shards, and freeze their lengths/hash.

    The default path opens every shard with :func:`numpy.load` and is the safe
    generic behavior for arbitrary corpora.

    A caller that has *already* authenticated a pinned inventory may provide
    all three ``trusted_*``/``expected_*`` arguments.  In that mode token
    counts are inferred from the stat size as
    ``(byte_size - header_bytes) / dtype.itemsize`` without opening every
    shard.  The mode fails closed if only some arguments are supplied, if any
    payload is negative or not exactly divisible by the item size, or if the
    inferred aggregate differs from ``expected_total_tokens``.  Authenticating
    the filenames/sizes and representative NPY headers remains the caller's
    responsibility; this optimization must not be used for an unpinned tree.
    """

    source_root = Path(source_root).resolve()
    output_path = Path(output_path)
    paths = sorted(source_root.glob(pattern), key=_numeric_shard_key)
    if not paths:
        raise FileNotFoundError(f"No shards matching {pattern!r} under {source_root}")

    fast_path_values = (
        trusted_npy_dtype,
        trusted_npy_header_bytes,
        expected_total_tokens,
    )
    if any(value is not None for value in fast_path_values) and not all(
        value is not None for value in fast_path_values
    ):
        raise ValueError(
            "trusted source-manifest fast path requires trusted_npy_dtype, "
            "trusted_npy_header_bytes, and expected_total_tokens together"
        )
    use_trusted_sizes = all(value is not None for value in fast_path_values)
    trusted_dtype: np.dtype[Any] | None = None
    header_bytes: int | None = None
    if use_trusted_sizes:
        trusted_dtype = np.dtype(trusted_npy_dtype)
        if trusted_dtype.hasobject or trusted_dtype.itemsize <= 0:
            raise ValueError(
                f"trusted_npy_dtype must be a fixed-width non-object dtype, "
                f"got {trusted_dtype}"
            )
        header_bytes = int(trusted_npy_header_bytes)
        if header_bytes < 0:
            raise ValueError(
                "trusted_npy_header_bytes must be non-negative, "
                f"got {header_bytes}"
            )
        if int(expected_total_tokens) < 0:
            raise ValueError(
                "expected_total_tokens must be non-negative, "
                f"got {expected_total_tokens}"
            )

    shards: list[SourceShard] = []
    seen_numbers: set[int] = set()
    for path in paths:
        shard_number, _ = _numeric_shard_key(path)
        if shard_number in seen_numbers:
            raise ValueError(f"Duplicate numeric shard id {shard_number}")
        seen_numbers.add(shard_number)
        byte_size = path.stat().st_size
        if use_trusted_sizes:
            assert trusted_dtype is not None
            assert header_bytes is not None
            payload_bytes = byte_size - header_bytes
            if payload_bytes < 0:
                raise ValueError(
                    f"Shard is smaller than the trusted NPY header "
                    f"({byte_size} < {header_bytes}): {path}"
                )
            if payload_bytes % trusted_dtype.itemsize:
                raise ValueError(
                    f"Shard payload is not divisible by dtype item size "
                    f"({payload_bytes} % {trusted_dtype.itemsize}): {path}"
                )
            num_tokens = payload_bytes // trusted_dtype.itemsize
            dtype_string = trusted_dtype.str
        else:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if array.ndim != 1:
                raise ValueError(
                    f"Expected a 1-D token shard, got {array.shape} in {path}"
                )
            num_tokens = int(array.shape[0])
            dtype_string = np.dtype(array.dtype).str
            del array
        shards.append(
            SourceShard(
                shard_number=shard_number,
                relative_path=path.relative_to(source_root).as_posix(),
                num_tokens=num_tokens,
                dtype=dtype_string,
                byte_size=byte_size,
                content_sha256=_sha256_file(path) if content_hashes else None,
            )
        )

    total_tokens = sum(shard.num_tokens for shard in shards)
    if use_trusted_sizes and total_tokens != int(expected_total_tokens):
        raise ValueError(
            "Trusted source token total mismatch: "
            f"{total_tokens} != {int(expected_total_tokens)}"
        )

    value: dict[str, Any] = {
        "schema": "interleaved-source-shards-v1",
        "schema_version": SCHEMA_VERSION,
        "sort": "numeric-final-npy-component",
        "content_hashes": bool(content_hashes),
        "total_tokens": total_tokens,
        "shards": [shard.as_dict() for shard in shards],
    }
    value["manifest_hash"] = _hash_dict(value, "manifest_hash")
    _atomic_json(output_path, value)
    return SourceShardManifest.load(output_path)


@dataclass(frozen=True)
class TokenSpan:
    shard_number: int
    relative_path: str
    start: int
    stop: int

    @property
    def num_tokens(self) -> int:
        return self.stop - self.start

    def as_dict(self) -> dict[str, Any]:
        return {
            "shard_number": self.shard_number,
            "relative_path": self.relative_path,
            "start": self.start,
            "stop": self.stop,
        }


@dataclass(frozen=True)
class PretrainSelection:
    path: Path
    source_manifest_hash: str
    target_tokens: int
    source_tokens: int
    seed: int
    spans: tuple[TokenSpan, ...]
    selection_hash: str

    @classmethod
    def load(cls, path: str | Path) -> "PretrainSelection":
        path = Path(path)
        value = _checked_metadata(
            path, "selection_hash", "interleaved-pretrain-selection-v1"
        )
        spans = tuple(TokenSpan(**span) for span in value["spans"])
        source_tokens = sum(span.num_tokens for span in spans)
        if source_tokens != int(value["source_tokens"]):
            raise ValueError(f"Selection token total is inconsistent in {path}")
        if source_tokens != int(value["target_tokens"]) + 1:
            raise ValueError("A next-token selection must contain target_tokens + 1")
        return cls(
            path=path,
            source_manifest_hash=value["source_manifest_hash"],
            target_tokens=int(value["target_tokens"]),
            source_tokens=source_tokens,
            seed=int(value["seed"]),
            spans=spans,
            selection_hash=value["selection_hash"],
        )


def build_pretrain_selection(
    source_manifest_path: str | Path,
    output_path: str | Path,
    *,
    target_tokens: int = DEFAULT_TOTAL_TARGETS,
    seed: int = 42,
) -> PretrainSelection:
    """Select exactly ``target_tokens + 1`` source tokens without replacement."""

    source = SourceShardManifest.load(source_manifest_path)
    target_tokens = int(target_tokens)
    required = target_tokens + 1
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    if source.total_tokens < required:
        raise ValueError(
            f"Corpus has {source.total_tokens:,} tokens, need {required:,}"
        )

    rng = random.Random(int(seed))
    order = list(range(len(source.shards)))
    rng.shuffle(order)
    remaining = required
    spans: list[TokenSpan] = []
    for ordinal in order:
        shard = source.shards[ordinal]
        take = min(remaining, shard.num_tokens)
        if take == shard.num_tokens:
            start = 0
        else:
            start = rng.randrange(0, shard.num_tokens - take + 1)
        spans.append(
            TokenSpan(
                shard_number=shard.shard_number,
                relative_path=shard.relative_path,
                start=start,
                stop=start + take,
            )
        )
        remaining -= take
        if remaining == 0:
            break
    if remaining:
        raise AssertionError(f"Selection builder left {remaining} tokens")

    value: dict[str, Any] = {
        "schema": "interleaved-pretrain-selection-v1",
        "schema_version": SCHEMA_VERSION,
        "algorithm": "python-random-shard-permutation-v1",
        "source_manifest_hash": source.manifest_hash,
        "target_tokens": target_tokens,
        "source_tokens": required,
        "seed": int(seed),
        "spans": [span.as_dict() for span in spans],
    }
    value["selection_hash"] = _hash_dict(value, "selection_hash")
    _atomic_json(Path(output_path), value)
    return PretrainSelection.load(output_path)


class _MMapShardStore:
    def __init__(self, root: Path, max_open_shards: int = 64):
        self.root = Path(root)
        self.max_open_shards = max(1, int(max_open_shards))
        self._arrays: OrderedDict[str, np.ndarray] = OrderedDict()

    def get(self, relative_path: str) -> np.ndarray:
        array = self._arrays.pop(relative_path, None)
        if array is None:
            array = np.load(
                self.root / relative_path, mmap_mode="r", allow_pickle=False
            )
        self._arrays[relative_path] = array
        while len(self._arrays) > self.max_open_shards:
            self._arrays.popitem(last=False)
        return array

    def __getstate__(self):
        return {
            "root": self.root,
            "max_open_shards": self.max_open_shards,
            "_arrays": OrderedDict(),
        }


class LogicalTokenSelection:
    """Lazy random access into the concatenation of selected source spans."""

    def __init__(
        self,
        source_root: str | Path,
        source_manifest: SourceShardManifest,
        selection: PretrainSelection,
        *,
        max_open_shards: int = 64,
    ):
        if source_manifest.manifest_hash != selection.source_manifest_hash:
            raise ValueError("Selection was built from a different source manifest")
        self.source_root = Path(source_root)
        self.selection = selection
        self._ends: list[int] = []
        total = 0
        for span in selection.spans:
            total += span.num_tokens
            self._ends.append(total)
        if total != selection.source_tokens:
            raise ValueError("Selection spans do not cover the declared source tokens")
        self._store = _MMapShardStore(self.source_root, max_open_shards)

    def __len__(self) -> int:
        return self.selection.source_tokens

    def read(self, start: int, stop: int) -> np.ndarray:
        start, stop = int(start), int(stop)
        if start < 0 or stop < start or stop > len(self):
            raise IndexError(f"Invalid logical token range [{start}, {stop})")
        if start == stop:
            return np.empty(0, dtype=np.int64)
        pieces: list[np.ndarray] = []
        cursor = start
        span_index = bisect.bisect_right(self._ends, cursor)
        while cursor < stop:
            span = self.selection.spans[span_index]
            logical_span_start = 0 if span_index == 0 else self._ends[span_index - 1]
            local_start = span.start + (cursor - logical_span_start)
            take = min(stop - cursor, self._ends[span_index] - cursor)
            array = self._store.get(span.relative_path)
            pieces.append(
                np.asarray(array[local_start : local_start + take], dtype=np.int64)
            )
            cursor += take
            span_index += 1
        if len(pieces) == 1:
            return np.array(pieces[0], dtype=np.int64, copy=True)
        return np.concatenate(pieces).astype(np.int64, copy=False)


class PackedPretrainDataset(Dataset):
    """Exact PT targets with an explicit BOS at every packed-context start."""

    def __init__(
        self,
        logical_tokens: LogicalTokenSelection,
        *,
        target_start: int,
        target_count: int,
        bos_token_id: int,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    ):
        self.logical_tokens = logical_tokens
        self.target_start = int(target_start)
        self.target_count = int(target_count)
        self.bos_token_id = int(bos_token_id)
        self.sequence_length = int(sequence_length)
        if self.target_start < 0 or self.target_count <= 0:
            raise ValueError("Invalid pretraining target range")
        if self.bos_token_id < 0:
            raise ValueError("bos_token_id must be non-negative")
        if self.target_start + self.target_count + 1 > len(logical_tokens):
            raise ValueError("Pretraining target range exceeds logical selection")

    def __len__(self) -> int:
        return math.ceil(self.target_count / self.sequence_length)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        target_offset = index * self.sequence_length
        valid_targets = min(
            self.sequence_length, self.target_count - target_offset
        )
        source_start = self.target_start + target_offset
        # Preserve the historical supervised target set exactly: target i was
        # logical token source_start+i+1. Replace only the arbitrary token
        # before each packed chunk with an explicit BOS context token.
        targets = self.logical_tokens.read(
            source_start + 1, source_start + valid_targets + 1
        )
        if len(targets) != valid_targets:
            raise AssertionError("A packed record did not receive every target")
        inputs = np.empty(valid_targets, dtype=np.int64)
        inputs[0] = self.bos_token_id
        if valid_targets > 1:
            inputs[1:] = targets[:-1]
        return {
            "input_ids": torch.tensor(inputs, dtype=torch.long),
            "labels": torch.tensor(targets, dtype=torch.long),
            "attention_mask": torch.ones(valid_targets, dtype=torch.long),
            "sample_type": SAMPLE_PRETRAIN,
            "record_id": int(index),
            "valid_targets": valid_targets,
        }


def _nested_field(row: Mapping[str, Any], path: str) -> Any:
    value: Any = row
    for component in path.split("."):
        if not isinstance(value, Mapping) or component not in value:
            return None
        value = value[component]
    return value


def _rows_from_file(path: Path) -> Iterator[tuple[int, Mapping[str, Any]]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, Mapping):
                    raise ValueError(f"Non-object row {index} in {path}")
                yield index, row
        return

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if isinstance(value, Mapping):
        rows = value.get("results")
    else:
        rows = value
    if not isinstance(rows, list):
        raise ValueError(f"{path} must contain a list or an object with 'results'")
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"Non-object row {index} in {path}")
        yield index, row


def _tokenizer_vocab(tokenizer) -> Mapping[str, int]:
    if not hasattr(tokenizer, "get_vocab"):
        raise TypeError("SFT tokenizer must provide get_vocab()")
    return tokenizer.get_vocab()


def normalize_sft_response(
    response: str,
    *,
    strip_verify_scores: bool = True,
) -> str:
    """Normalize one raw SFT response before it reaches the tokenizer.

    The pinned ``trajectory_sep.cot_format_no_labels`` snapshot contains some
    residual verifier annotations such as ``<verify> <-3>`` and
    ``<verify> <+0.5>``.  They are labels, not model-visible chess actions, and
    neither token belongs in the production tokenizer vocabulary.  Remove
    every numeric verifier/score pair at this raw-text boundary so spacing
    changes cannot turn either half into ``<unk>``.

    A residual ``<verify>`` after the substitution is a malformed or
    unsupported label.  Fail closed instead of silently passing it to the
    tokenizer.  Callers intentionally training on verifier labels may disable
    this normalization explicitly.
    """

    if not isinstance(response, str):
        raise TypeError("SFT response must be a string")
    if not strip_verify_scores:
        return response.strip()

    normalized = _VERIFY_SCORE_PAIR_RE.sub(" ", response)
    if _VERIFY_TOKEN_RE.search(normalized):
        raise ValueError(
            "SFT response contains an unpaired or non-numeric <verify> label"
        )
    return " ".join(normalized.split())


def _tokenizer_unk_id(tokenizer, vocab: Mapping[str, int]) -> int:
    value = getattr(tokenizer, "unk_id", None)
    if callable(value):
        try:
            return int(value())
        except (AttributeError, TypeError, ValueError):
            pass

    value = getattr(tokenizer, "unk_token_id", None)
    if value is not None:
        try:
            return int(value)
        except (TypeError, ValueError):
            pass

    unk_token = getattr(tokenizer, "_unk", "<unk>")
    if isinstance(unk_token, str) and unk_token in vocab:
        return int(vocab[unk_token])
    if "<unk>" in vocab:
        return int(vocab["<unk>"])
    raise ValueError(
        "Tokenizer has no discoverable unknown-token ID; strict supervised "
        "<unk> validation cannot be enforced"
    )


def _optional_tokenizer_id(tokenizer, method: str, fallback: int | None) -> int | None:
    value = getattr(tokenizer, method, None)
    if callable(value):
        try:
            return int(value())
        except (AttributeError, TypeError, ValueError):
            pass
    return fallback


def _mask_multi_turn_env_responses(
    input_ids: np.ndarray,
    labels: np.ndarray,
    *,
    tokenizer,
) -> None:
    """Match ``MultiTurnSFTDataset._mask_env_responses`` exactly."""

    vocab = _tokenizer_vocab(tokenizer)
    call_env_id = _optional_tokenizer_id(
        tokenizer, "call_env_id", vocab.get("<call_env>")
    )
    if call_env_id is None:
        raise ValueError("Tokenizer has no <call_env> ID for multi-turn masking")

    move_start_ids = {
        vocab[token]
        for token in ("K", "Q", "R", "B", "N", "P", "O-O", "O-O-O")
        if token in vocab
    }
    structural_ids = {
        vocab[token]
        for token in ("<T>", "</T>", "<sep>", "<bos>", "<eos>")
        if token in vocab
    }
    env_token_ids = getattr(tokenizer, "env_token_ids", None)
    if callable(env_token_ids):
        try:
            structural_ids.update(
                int(token_id)
                for token, token_id in env_token_ids().items()
                if token != "<call_env>"
            )
        except (AttributeError, TypeError, ValueError):
            pass
    equal_id = vocab.get("=")

    in_env_response = False
    env_move_started = False
    last_was_equal = False
    for index in range(len(input_ids)):
        if int(input_ids[index]) == call_env_id:
            in_env_response = True
            env_move_started = False
            last_was_equal = False

        if in_env_response:
            label = int(labels[index])
            if label in structural_ids:
                in_env_response = False
                env_move_started = False
                last_was_equal = False
            else:
                is_move_start = label in move_start_ids and not last_was_equal
                if is_move_start:
                    if not env_move_started:
                        env_move_started = True
                        labels[index] = -100
                        last_was_equal = False
                    else:
                        in_env_response = False
                        env_move_started = False
                        last_was_equal = False
                else:
                    labels[index] = -100
                    last_was_equal = equal_id is not None and label == equal_id
        else:
            last_was_equal = False


def tokenize_masked_sft_row(
    row: Mapping[str, Any],
    tokenizer,
    *,
    cot_field: str,
    prompt_field: str = "pgn",
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    strip_verify_scores: bool = True,
    reject_supervised_unk: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Tokenize one row with the existing multi-turn SFT mask semantics.

    Unlike the legacy dataset, invalid rows are errors rather than omissions.
    A combined token sequence of length ``sequence_length + 1`` is valid
    because next-token shifting produces exactly ``sequence_length`` aligned
    input/label positions.
    """

    prompt_value = _nested_field(row, prompt_field)
    response_value = _nested_field(row, cot_field)
    if not isinstance(prompt_value, str) or not prompt_value.strip():
        raise ValueError(f"Missing/empty SFT prompt field {prompt_field!r}")
    if not isinstance(response_value, str) or not response_value.strip():
        raise ValueError(f"Missing/empty SFT response field {cot_field!r}")
    prompt = prompt_value.replace("\n", " ").strip()
    response = normalize_sft_response(
        response_value,
        strip_verify_scores=strip_verify_scores,
    )
    if not response:
        raise ValueError(
            "SFT response is empty after verifier-label normalization"
        )

    prompt_tokens = list(tokenizer.encode(prompt))
    response_tokens = list(tokenizer.encode(response))
    if len(prompt_tokens) < 1 or len(response_tokens) < 1:
        raise ValueError("Tokenizer returned an empty prompt or response encoding")
    # Match SFTDataset: remove prompt EOS and response BOS.
    prompt_tokens = prompt_tokens[:-1]
    response_tokens = response_tokens[1:]

    vocab = _tokenizer_vocab(tokenizer)
    thinking_id = vocab.get("<T>")
    response_starts_with_thinking = bool(
        response_tokens
        and thinking_id is not None
        and response_tokens[0] == thinking_id
    )
    eos_id = _optional_tokenizer_id(tokenizer, "eos_id", vocab.get("<eos>"))
    if (
        not response_starts_with_thinking
        and response_tokens
        and eos_id is not None
        and response_tokens[-1] == eos_id
    ):
        response_tokens = response_tokens[:-1]

    combined = prompt_tokens + response_tokens
    aligned_length = len(combined) - 1
    if aligned_length <= 0:
        raise ValueError("SFT row has no next-token target")
    if aligned_length > int(sequence_length):
        raise ValueError(
            f"SFT row has {aligned_length} aligned positions, exceeds "
            f"context {sequence_length}"
        )

    input_ids = np.asarray(combined[:-1], dtype=np.int32)
    labels = np.asarray(combined[1:], dtype=np.int32)
    prompt_length = len(prompt_tokens) + (
        1 if response_starts_with_thinking else 0
    )
    if prompt_length > 1:
        labels[: min(prompt_length - 1, len(labels))] = -100
    _mask_multi_turn_env_responses(input_ids, labels, tokenizer=tokenizer)
    if not np.any(labels != -100):
        raise ValueError("SFT row has no supervised target after masking")
    if reject_supervised_unk:
        unk_id = _tokenizer_unk_id(tokenizer, vocab)
        supervised_unk_positions = np.flatnonzero(labels == unk_id)
        if supervised_unk_positions.size:
            preview = ", ".join(
                str(int(position))
                for position in supervised_unk_positions[:8]
            )
            raise ValueError(
                "SFT row has "
                f"{supervised_unk_positions.size} supervised <unk> target(s) "
                f"after normalization at aligned position(s) {preview}"
            )
    return input_ids, labels


@dataclass(frozen=True)
class SFTCache:
    directory: Path
    num_rows: int
    total_positions: int
    sequence_length: int
    cot_field: str
    prompt_field: str
    cache_hash: str

    @property
    def metadata_path(self) -> Path:
        return self.directory / "metadata.json"

    @classmethod
    def load(
        cls, directory: str | Path, *, verify_large_files: bool = True
    ) -> "SFTCache":
        directory = Path(directory)
        value = _checked_metadata(
            directory / "metadata.json", "cache_hash", "interleaved-sft-cache-v1"
        )
        for filename, hash_key in (
            ("input_ids.i32", "input_ids_sha256"),
            ("labels.i32", "labels_sha256"),
            ("offsets.npy", "offsets_sha256"),
        ):
            path = directory / filename
            if not path.is_file():
                raise FileNotFoundError(path)
            is_large_flat_file = filename.endswith(".i32")
            if (verify_large_files or not is_large_flat_file) and (
                _sha256_file(path) != value[hash_key]
            ):
                raise ValueError(f"SFT cache file hash mismatch: {path}")
        expected_flat_bytes = int(value["total_positions"]) * np.dtype("<i4").itemsize
        for filename in ("input_ids.i32", "labels.i32"):
            if (directory / filename).stat().st_size != expected_flat_bytes:
                raise ValueError(f"SFT cache byte length mismatch: {filename}")
        offsets = np.load(directory / "offsets.npy", mmap_mode="r", allow_pickle=False)
        if len(offsets) != int(value["num_rows"]) + 1:
            raise ValueError("SFT offsets length does not match num_rows")
        if int(offsets[-1]) != int(value["total_positions"]):
            raise ValueError("SFT offsets do not match total_positions")
        if (
            value.get("supervised_unk_policy")
            == SFT_SUPERVISED_UNK_POLICY_REJECT_V1
            and value.get("supervised_unk_targets") != 0
        ):
            raise ValueError(
                "Strict SFT cache metadata must certify zero supervised "
                "<unk> targets"
            )
        _validate_strict_sft_cache_metadata(value)
        return cls(
            directory=directory,
            num_rows=int(value["num_rows"]),
            total_positions=int(value["total_positions"]),
            sequence_length=int(value["sequence_length"]),
            cot_field=value["cot_field"],
            prompt_field=value["prompt_field"],
            cache_hash=value["cache_hash"],
        )


def _metadata_nonnegative_int(
    value: Mapping[str, Any],
    key: str,
    *,
    positive: bool = False,
) -> int:
    item = value.get(key)
    if isinstance(item, bool) or not isinstance(item, int):
        raise ValueError(f"Strict SFT cache metadata {key!r} must be an integer")
    lower_bound = 1 if positive else 0
    if item < lower_bound:
        comparator = "positive" if positive else "non-negative"
        raise ValueError(
            f"Strict SFT cache metadata {key!r} must be {comparator}"
        )
    return int(item)


def _validate_strict_sft_cache_metadata(value: Mapping[str, Any]) -> None:
    """Validate the self-consistency of new strict cache audit metadata.

    Legacy v1 caches predate ``strict_sft_audit`` and remain loadable.  A cache
    that explicitly requires the new strict audit must carry the complete
    audit and satisfy every row/aggregate invariant.
    """

    audit = value.get("strict_sft_audit")
    strict_audit_required = value.get("strict_sft_audit_required", False)
    if not isinstance(strict_audit_required, bool):
        raise ValueError("strict_sft_audit_required must be a boolean")
    claims_strict_cleaned_cache = (
        value.get("response_normalization")
        == SFT_RESPONSE_NORMALIZATION_STRIP_VERIFY_V1
        and value.get("supervised_unk_policy")
        == SFT_SUPERVISED_UNK_POLICY_REJECT_V1
    )
    if audit is None:
        if strict_audit_required or claims_strict_cleaned_cache:
            raise ValueError(
                "Strict SFT cache requires strict_sft_audit metadata"
            )
        return
    if not isinstance(audit, Mapping):
        raise ValueError("strict_sft_audit must be an object or null")
    if not strict_audit_required:
        raise ValueError(
            "strict_sft_audit metadata requires strict_sft_audit_required=true"
        )
    if audit.get("schema") != SFT_STRICT_AUDIT_SCHEMA_V1:
        raise ValueError("Unsupported strict SFT cache audit schema")
    if (
        value.get("response_normalization")
        != SFT_RESPONSE_NORMALIZATION_STRIP_VERIFY_V1
        or value.get("supervised_unk_policy")
        != SFT_SUPERVISED_UNK_POLICY_REJECT_V1
    ):
        raise ValueError(
            "strict_sft_audit requires the cleaned-response and supervised-unk "
            "rejection policies"
        )

    num_rows = _metadata_nonnegative_int(value, "num_rows", positive=True)
    total_positions = _metadata_nonnegative_int(
        value, "total_positions", positive=True
    )
    supervised_targets = _metadata_nonnegative_int(
        value, "supervised_targets", positive=True
    )
    if supervised_targets > total_positions:
        raise ValueError(
            "Strict SFT cache supervised targets exceed total positions"
        )
    if value.get("supervised_unk_targets") != 0:
        raise ValueError("Strict SFT cache must have zero supervised <unk> targets")

    expected_targets = audit.get("expected_supervised_targets")
    if expected_targets is not None:
        if isinstance(expected_targets, bool) or not isinstance(
            expected_targets, int
        ):
            raise ValueError(
                "expected_supervised_targets must be an integer or null"
            )
        if expected_targets <= 0:
            raise ValueError("expected_supervised_targets must be positive")
        if expected_targets != supervised_targets:
            raise ValueError(
                "Strict SFT cache supervised target total differs from its "
                "preregistered expectation"
            )

    delimiter_counts = value.get("supervised_delimiter_counts")
    if not isinstance(delimiter_counts, Mapping):
        raise ValueError("supervised_delimiter_counts must be an object")
    if set(delimiter_counts) != set(SFT_SUPERVISED_DELIMITERS):
        raise ValueError(
            "supervised_delimiter_counts must contain exactly "
            f"{SFT_SUPERVISED_DELIMITERS}"
        )
    parsed_delimiters = {
        token: _metadata_nonnegative_int(delimiter_counts, token)
        for token in SFT_SUPERVISED_DELIMITERS
    }
    if sum(parsed_delimiters.values()) > supervised_targets:
        raise ValueError(
            "Aggregate supervised delimiter counts exceed supervised targets"
        )
    if parsed_delimiters["</T>"] != num_rows:
        raise ValueError(
            "Strict SFT cache must contain exactly one supervised </T> per row"
        )
    if parsed_delimiters["<call_env>"] < num_rows:
        raise ValueError(
            "Strict SFT cache must contain at least one supervised <call_env> "
            "per row"
        )

    t_end_rows = _metadata_nonnegative_int(audit, "t_end_rows_exactly_one")
    call_env_rows = _metadata_nonnegative_int(
        audit, "call_env_rows_at_least_one"
    )
    call_env_min = _metadata_nonnegative_int(
        audit, "call_env_supervised_per_row_min", positive=True
    )
    call_env_max = _metadata_nonnegative_int(
        audit, "call_env_supervised_per_row_max", positive=True
    )
    if t_end_rows != num_rows:
        raise ValueError(
            "Strict SFT cache did not certify exactly one supervised </T> "
            "for every row"
        )
    if call_env_rows != num_rows:
        raise ValueError(
            "Strict SFT cache did not certify a supervised <call_env> "
            "for every row"
        )
    if call_env_min > call_env_max:
        raise ValueError(
            "Strict SFT cache <call_env> per-row minimum exceeds maximum"
        )
    call_env_total = parsed_delimiters["<call_env>"]
    if not (
        num_rows * call_env_min
        <= call_env_total
        <= num_rows * call_env_max
    ):
        raise ValueError(
            "Strict SFT cache aggregate <call_env> count is inconsistent with "
            "its per-row range"
        )


def build_sft_cache(
    sft_files: Sequence[str | Path],
    tokenizer,
    output_dir: str | Path,
    *,
    cot_field: str = "cot_by_method.trajectory_sep.cot_format_no_labels",
    prompt_field: str = "pgn",
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    expected_rows: int = DEFAULT_SFT_ROWS,
    strip_verify_scores: bool = True,
    reject_supervised_unk: bool = True,
    strict_sft_audit: bool = True,
    expected_supervised_targets: int | None = None,
) -> SFTCache:
    """Tokenize all SFT rows once into mmap-friendly aligned flat arrays."""

    if strict_sft_audit and (
        not strip_verify_scores or not reject_supervised_unk
    ):
        raise ValueError(
            "strict_sft_audit requires strip_verify_scores=True and "
            "reject_supervised_unk=True"
        )
    if (
        not strict_sft_audit
        and strip_verify_scores
        and reject_supervised_unk
    ):
        raise ValueError(
            "The cleaned-response and supervised-unk rejection policies "
            "require strict_sft_audit=True"
        )
    if expected_supervised_targets is not None:
        if (
            isinstance(expected_supervised_targets, bool)
            or not isinstance(expected_supervised_targets, int)
            or expected_supervised_targets <= 0
        ):
            raise ValueError(
                "expected_supervised_targets must be a positive integer or None"
            )
        if not strict_sft_audit:
            raise ValueError(
                "expected_supervised_targets requires strict_sft_audit=True"
            )

    files = tuple(sorted((Path(path) for path in sft_files), key=lambda p: str(p)))
    if not files:
        raise ValueError("No SFT files supplied")
    output_dir = Path(output_dir)
    if output_dir.exists():
        metadata_path = output_dir / "metadata.json"
        if not metadata_path.is_file():
            raise FileExistsError(
                f"Existing SFT cache is incomplete (no metadata): {output_dir}"
            )
        existing = SFTCache.load(output_dir, verify_large_files=True)
        expected_contract = (
            int(expected_rows),
            int(sequence_length),
            cot_field,
            prompt_field,
        )
        actual_contract = (
            existing.num_rows,
            existing.sequence_length,
            existing.cot_field,
            existing.prompt_field,
        )
        if actual_contract != expected_contract:
            raise ValueError(
                "Existing SFT cache contract differs: "
                f"{actual_contract} != {expected_contract}"
            )
        metadata = _checked_metadata(
            metadata_path, "cache_hash", "interleaved-sft-cache-v1"
        )
        expected_policies = (
            (
                SFT_RESPONSE_NORMALIZATION_STRIP_VERIFY_V1
                if strip_verify_scores
                else "none"
            ),
            (
                SFT_SUPERVISED_UNK_POLICY_REJECT_V1
                if reject_supervised_unk
                else "allow"
            ),
        )
        actual_policies = (
            metadata.get("response_normalization"),
            metadata.get("supervised_unk_policy"),
        )
        if actual_policies != expected_policies:
            raise ValueError(
                "Existing SFT cache normalization/validation contract differs: "
                f"{actual_policies} != {expected_policies}; use a new cache directory"
            )
        actual_audit = metadata.get("strict_sft_audit")
        if strict_sft_audit:
            if not isinstance(actual_audit, Mapping):
                raise ValueError(
                    "Existing SFT cache lacks the requested strict audit"
                )
            if (
                actual_audit.get("expected_supervised_targets")
                != expected_supervised_targets
            ):
                raise ValueError(
                    "Existing SFT cache preregistered supervised-target "
                    "expectation differs; use a new cache directory"
                )
        elif actual_audit is not None:
            raise ValueError(
                "Existing SFT cache has a strict audit but this build disabled "
                "it; use a new cache directory"
            )
        supplied_sources = []
        for ordinal, path in enumerate(files):
            if not path.is_file():
                raise FileNotFoundError(path)
            supplied_sources.append(
                {
                    "ordinal": ordinal,
                    "name": path.name,
                    "byte_size": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
        if metadata.get("source_files") != supplied_sources:
            raise ValueError(
                "Existing SFT cache was built from different source files"
            )
        return existing
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        offsets = [0]
        input_digest = hashlib.sha256()
        label_digest = hashlib.sha256()
        row_digest = hashlib.sha256()
        num_rows = 0
        supervised_targets = 0
        supervised_unk_targets = 0
        supervised_delimiter_counts = {
            token: 0 for token in SFT_SUPERVISED_DELIMITERS
        }
        t_end_rows_exactly_one = 0
        call_env_rows_at_least_one = 0
        call_env_supervised_per_row_min: int | None = None
        call_env_supervised_per_row_max = 0
        vocab = _tokenizer_vocab(tokenizer)
        delimiter_ids: dict[str, int] = {}
        if strict_sft_audit:
            missing_delimiters = [
                token
                for token in SFT_SUPERVISED_DELIMITERS
                if token not in vocab
            ]
            if missing_delimiters:
                raise ValueError(
                    "Strict SFT cache tokenizer lacks delimiter token(s): "
                    f"{missing_delimiters}"
                )
            delimiter_ids = {
                token: int(vocab[token])
                for token in SFT_SUPERVISED_DELIMITERS
            }
        unk_id = (
            _tokenizer_unk_id(tokenizer, vocab)
            if strict_sft_audit
            else None
        )
        source_files = []
        with (
            (temporary / "input_ids.i32").open("wb") as input_handle,
            (temporary / "labels.i32").open("wb") as label_handle,
        ):
            for file_ordinal, path in enumerate(files):
                if not path.is_file():
                    raise FileNotFoundError(path)
                source_files.append(
                    {
                        "ordinal": file_ordinal,
                        "name": path.name,
                        "byte_size": path.stat().st_size,
                        "sha256": _sha256_file(path),
                    }
                )
                for row_index, row in _rows_from_file(path):
                    try:
                        input_ids, labels = tokenize_masked_sft_row(
                            row,
                            tokenizer,
                            cot_field=cot_field,
                            prompt_field=prompt_field,
                            sequence_length=sequence_length,
                            strip_verify_scores=strip_verify_scores,
                            reject_supervised_unk=reject_supervised_unk,
                        )
                        if strict_sft_audit:
                            valid_targets = int(
                                np.count_nonzero(labels != -100)
                            )
                            row_unk_targets = int(
                                np.count_nonzero(labels == unk_id)
                            )
                            row_delimiters = {
                                token: int(
                                    np.count_nonzero(labels == token_id)
                                )
                                for token, token_id in delimiter_ids.items()
                            }
                            if row_delimiters["</T>"] != 1:
                                raise ValueError(
                                    "expected exactly one supervised </T>, got "
                                    f"{row_delimiters['</T>']}"
                                )
                            call_env_count = row_delimiters["<call_env>"]
                            if call_env_count < 1:
                                raise ValueError(
                                    "expected at least one supervised "
                                    "<call_env>, got 0"
                                )
                    except Exception as error:
                        raise ValueError(
                            f"Invalid SFT row {path.name}:{row_index}: {error}"
                        ) from error
                    if strict_sft_audit:
                        supervised_targets += valid_targets
                        supervised_unk_targets += row_unk_targets
                        for token, count in row_delimiters.items():
                            supervised_delimiter_counts[token] += count
                        t_end_rows_exactly_one += 1
                        call_env_rows_at_least_one += 1
                        call_env_supervised_per_row_min = (
                            call_env_count
                            if call_env_supervised_per_row_min is None
                            else min(
                                call_env_supervised_per_row_min,
                                call_env_count,
                            )
                        )
                        call_env_supervised_per_row_max = max(
                            call_env_supervised_per_row_max,
                            call_env_count,
                        )
                    input_bytes = np.asarray(input_ids, dtype="<i4").tobytes()
                    label_bytes = np.asarray(labels, dtype="<i4").tobytes()
                    input_handle.write(input_bytes)
                    label_handle.write(label_bytes)
                    input_digest.update(input_bytes)
                    label_digest.update(label_bytes)
                    row_digest.update(
                        _canonical_json(
                            {
                                "file_ordinal": file_ordinal,
                                "row_index": row_index,
                                "input_sha256": hashlib.sha256(input_bytes).hexdigest(),
                                "labels_sha256": hashlib.sha256(label_bytes).hexdigest(),
                            }
                        )
                    )
                    offsets.append(offsets[-1] + len(input_ids))
                    num_rows += 1

        if num_rows != int(expected_rows):
            raise ValueError(
                f"SFT row count is {num_rows:,}; expected exactly {expected_rows:,}"
            )
        if (
            strict_sft_audit
            and expected_supervised_targets is not None
            and supervised_targets != expected_supervised_targets
        ):
            raise ValueError(
                "SFT supervised target count is "
                f"{supervised_targets:,}; expected exactly "
                f"{expected_supervised_targets:,}"
            )
        strict_audit = None
        if strict_sft_audit:
            if supervised_unk_targets != 0:
                raise AssertionError(
                    "Strict SFT cache accumulated supervised <unk> targets"
                )
            if call_env_supervised_per_row_min is None:
                raise AssertionError("Strict SFT audit saw no rows")
            strict_audit = {
                "schema": SFT_STRICT_AUDIT_SCHEMA_V1,
                "expected_supervised_targets": expected_supervised_targets,
                "t_end_rows_exactly_one": t_end_rows_exactly_one,
                "call_env_rows_at_least_one": call_env_rows_at_least_one,
                "call_env_supervised_per_row_min": (
                    call_env_supervised_per_row_min
                ),
                "call_env_supervised_per_row_max": (
                    call_env_supervised_per_row_max
                ),
            }
        offsets_array = np.asarray(offsets, dtype="<i8")
        _atomic_save_npy(temporary / "offsets.npy", offsets_array)
        value: dict[str, Any] = {
            "schema": "interleaved-sft-cache-v1",
            "schema_version": SCHEMA_VERSION,
            "num_rows": num_rows,
            "total_positions": int(offsets[-1]),
            "sequence_length": int(sequence_length),
            "cot_field": cot_field,
            "prompt_field": prompt_field,
            "response_normalization": (
                SFT_RESPONSE_NORMALIZATION_STRIP_VERIFY_V1
                if strip_verify_scores
                else "none"
            ),
            "supervised_unk_policy": (
                SFT_SUPERVISED_UNK_POLICY_REJECT_V1
                if reject_supervised_unk
                else "allow"
            ),
            "supervised_targets": (
                supervised_targets if strict_sft_audit else None
            ),
            "supervised_delimiter_counts": (
                supervised_delimiter_counts if strict_sft_audit else None
            ),
            "supervised_unk_targets": (
                supervised_unk_targets
                if strict_sft_audit
                else (0 if reject_supervised_unk else None)
            ),
            "strict_sft_audit": strict_audit,
            "strict_sft_audit_required": bool(strict_sft_audit),
            "masking": "multiturn-prompt-and-env-v1",
            "dtype": "<i4",
            "source_files": source_files,
            "rows_sha256": row_digest.hexdigest(),
            "input_ids_sha256": input_digest.hexdigest(),
            "labels_sha256": label_digest.hexdigest(),
            "offsets_sha256": _sha256_file(temporary / "offsets.npy"),
        }
        value["cache_hash"] = _hash_dict(value, "cache_hash")
        _atomic_json(temporary / "metadata.json", value)
        os.replace(temporary, output_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return SFTCache.load(output_dir, verify_large_files=True)


class SFTCacheDataset(Dataset):
    def __init__(self, cache: SFTCache):
        self.cache = cache
        self._input_ids: np.memmap | None = None
        self._labels: np.memmap | None = None
        self._offsets: np.ndarray | None = None

    def __len__(self) -> int:
        return self.cache.num_rows

    def _open(self) -> None:
        if self._offsets is None:
            self._offsets = np.load(
                self.cache.directory / "offsets.npy",
                mmap_mode="r",
                allow_pickle=False,
            )
            self._input_ids = np.memmap(
                self.cache.directory / "input_ids.i32",
                mode="r",
                dtype="<i4",
                shape=(self.cache.total_positions,),
            )
            self._labels = np.memmap(
                self.cache.directory / "labels.i32",
                mode="r",
                dtype="<i4",
                shape=(self.cache.total_positions,),
            )

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        self._open()
        assert self._offsets is not None
        assert self._input_ids is not None
        assert self._labels is not None
        start, stop = int(self._offsets[index]), int(self._offsets[index + 1])
        input_ids = np.array(self._input_ids[start:stop], dtype=np.int64, copy=True)
        labels = np.array(self._labels[start:stop], dtype=np.int64, copy=True)
        return {
            "input_ids": torch.from_numpy(input_ids),
            "labels": torch.from_numpy(labels),
            "attention_mask": torch.ones(stop - start, dtype=torch.long),
            "sample_type": SAMPLE_SFT,
            "record_id": int(index),
            "valid_targets": int(np.count_nonzero(labels != -100)),
        }

    def __getstate__(self):
        return {
            "cache": self.cache,
            "_input_ids": None,
            "_labels": None,
            "_offsets": None,
        }


@dataclass(frozen=True)
class LegManifest:
    metadata_path: Path
    order_path: Path
    leg: str
    target_start: int
    target_count: int
    sequence_length: int
    pretrain_records: int
    sft_records: int
    sft_supervised_targets: int | None
    padding_records: int
    world_size: int
    local_batch_size: int
    physical_steps: int
    total_steps: int
    source_manifest_hash: str
    selection_hash: str
    sft_cache_hash: str
    order_sha256: str
    order_provenance: dict[str, Any] | None
    metadata_hash: str

    @property
    def global_batch_size(self) -> int:
        return self.world_size * self.local_batch_size

    @classmethod
    def load(cls, metadata_path: str | Path) -> "LegManifest":
        metadata_path = Path(metadata_path)
        value = _checked_metadata(
            metadata_path, "metadata_hash", "interleaved-leg-manifest-v1"
        )
        order_path = metadata_path.parent / value.get("order_file", "order.npy")
        if not order_path.is_file():
            raise FileNotFoundError(order_path)
        order_sha256 = _sha256_file(order_path)
        if order_sha256 != value["order_sha256"]:
            raise ValueError(f"Leg order hash mismatch in {order_path}")
        order = np.load(order_path, mmap_mode="r", allow_pickle=False)
        if order.dtype != np.dtype("<i8") and order.dtype != np.dtype("int64"):
            raise ValueError(f"Leg order must be int64, got {order.dtype}")
        if order.ndim != 1 or len(order) != int(value["num_order_records"]):
            raise ValueError("Leg order shape does not match metadata")
        global_batch = int(value["world_size"]) * int(value["local_batch_size"])
        if len(order) % global_batch:
            raise ValueError("Leg order is not divisible by global batch size")
        physical_steps = len(order) // global_batch
        if physical_steps != int(value["physical_steps"]):
            raise ValueError("Leg physical step count is inconsistent")
        raw_sft_supervised_targets = value.get("sft_supervised_targets")
        if raw_sft_supervised_targets is None:
            # Compatibility for immutable v1 leg manifests created before
            # exact per-split SFT target accounting was introduced.
            sft_supervised_targets = None
        else:
            if (
                isinstance(raw_sft_supervised_targets, bool)
                or not isinstance(raw_sft_supervised_targets, int)
                or raw_sft_supervised_targets < 0
            ):
                raise ValueError(
                    "Leg sft_supervised_targets must be a non-negative integer"
                )
            sft_supervised_targets = int(raw_sft_supervised_targets)
            sft_records = int(value["sft_records"])
            if (sft_records == 0) != (sft_supervised_targets == 0):
                raise ValueError(
                    "Leg SFT records and supervised targets are inconsistent"
                )
        raw_order_provenance = value.get("order_provenance")
        if raw_order_provenance is None:
            # Compatibility for immutable v1 manifests created before order
            # construction was recorded as a structured, authenticated graph.
            order_provenance = None
        else:
            if not isinstance(raw_order_provenance, dict):
                raise ValueError("Leg order_provenance must be a JSON object")
            schema = raw_order_provenance.get("schema")
            if not isinstance(schema, str) or not schema:
                raise ValueError(
                    "Leg order_provenance must declare a non-empty schema"
                )
            # Normalize to a detached JSON value so callers cannot mutate the
            # loaded metadata through a shared object reference.
            order_provenance = json.loads(
                _canonical_json(raw_order_provenance).decode("utf-8")
            )
        return cls(
            metadata_path=metadata_path,
            order_path=order_path,
            leg=value["leg"],
            target_start=int(value["target_start"]),
            target_count=int(value["target_count"]),
            sequence_length=int(value["sequence_length"]),
            pretrain_records=int(value["pretrain_records"]),
            sft_records=int(value["sft_records"]),
            sft_supervised_targets=sft_supervised_targets,
            padding_records=int(value["padding_records"]),
            world_size=int(value["world_size"]),
            local_batch_size=int(value["local_batch_size"]),
            physical_steps=physical_steps,
            total_steps=int(value["total_steps"]),
            source_manifest_hash=value["source_manifest_hash"],
            selection_hash=value["selection_hash"],
            sft_cache_hash=value["sft_cache_hash"],
            order_sha256=order_sha256,
            order_provenance=order_provenance,
            metadata_hash=value["metadata_hash"],
        )


def _write_leg_manifest(
    output_dir: Path,
    *,
    leg: str,
    order: np.ndarray,
    target_start: int,
    target_count: int,
    sequence_length: int,
    pretrain_records: int,
    sft_records: int,
    sft_supervised_targets: int,
    padding_records: int,
    world_size: int,
    local_batch_size: int,
    total_steps: int | None,
    source_manifest_hash: str,
    selection_hash: str,
    sft_cache_hash: str,
    shuffle_seed: int | None,
    order_provenance: Mapping[str, Any] | None = None,
) -> LegManifest:
    output_dir.mkdir(parents=True, exist_ok=True)
    order = np.asarray(order, dtype="<i8")
    global_batch = int(world_size) * int(local_batch_size)
    if global_batch <= 0 or len(order) % global_batch:
        raise ValueError("Order length must be divisible by the global batch")
    physical_steps = len(order) // global_batch
    if (
        isinstance(sft_supervised_targets, bool)
        or not isinstance(sft_supervised_targets, int)
        or sft_supervised_targets < 0
    ):
        raise ValueError(
            "sft_supervised_targets must be a non-negative integer"
        )
    if (int(sft_records) == 0) != (sft_supervised_targets == 0):
        raise ValueError(
            "SFT records and supervised targets must both be zero or both "
            "be positive"
        )
    _atomic_save_npy(output_dir / "order.npy", order)
    order_sha256 = _sha256_file(output_dir / "order.npy")
    value: dict[str, Any] = {
        "schema": "interleaved-leg-manifest-v1",
        "schema_version": SCHEMA_VERSION,
        "leg": leg,
        "order_file": "order.npy",
        "order_encoding": {
            "pretrain": "nonnegative local packed-record index",
            "sft": "-(global_sft_row_index+1)",
            "padding": int(PAD_RECORD),
        },
        "target_start": int(target_start),
        "target_count": int(target_count),
        "sequence_length": int(sequence_length),
        "pretrain_records": int(pretrain_records),
        "sft_records": int(sft_records),
        "sft_supervised_targets": int(sft_supervised_targets),
        "padding_records": int(padding_records),
        "num_order_records": int(len(order)),
        "world_size": int(world_size),
        "local_batch_size": int(local_batch_size),
        "global_batch_size": global_batch,
        "physical_steps": physical_steps,
        # Canary metadata advertises the production LR arc while intentionally
        # containing just one physical smoke-test batch.
        "total_steps": int(total_steps if total_steps is not None else physical_steps),
        "source_manifest_hash": source_manifest_hash,
        "selection_hash": selection_hash,
        "sft_cache_hash": sft_cache_hash,
        "shuffle_seed": shuffle_seed,
        "order_sha256": order_sha256,
    }
    if order_provenance is not None:
        if not isinstance(order_provenance, Mapping):
            raise ValueError("order_provenance must be a mapping")
        normalized_order_provenance = json.loads(
            _canonical_json(order_provenance).decode("utf-8")
        )
        schema = normalized_order_provenance.get("schema")
        if not isinstance(schema, str) or not schema:
            raise ValueError("order_provenance must declare a non-empty schema")
        value["order_provenance"] = normalized_order_provenance
    value["metadata_hash"] = _hash_dict(value, "metadata_hash")
    _atomic_json(output_dir / "metadata.json", value)
    return LegManifest.load(output_dir / "metadata.json")


def _shuffled_leg_order(
    *,
    pretrain_records: int,
    sft_indices: np.ndarray,
    global_batch_size: int,
    seed: int,
) -> tuple[np.ndarray, int]:
    pretrain = np.arange(pretrain_records, dtype="<i8")
    sft = -(np.asarray(sft_indices, dtype="<i8") + 1)
    order = np.concatenate((pretrain, sft))
    np.random.Generator(np.random.PCG64(int(seed))).shuffle(order)
    padding = (-len(order)) % int(global_batch_size)
    if padding:
        order = np.concatenate(
            (order, np.full(padding, PAD_RECORD, dtype="<i8"))
        )
    return order, int(padding)


def _sft_supervised_targets_per_row(cache: SFTCache) -> np.ndarray:
    """Read one cache once and return exact valid-label counts by source row."""

    offsets = np.load(
        cache.directory / "offsets.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    labels = np.memmap(
        cache.directory / "labels.i32",
        mode="r",
        dtype="<i4",
        shape=(cache.total_positions,),
    )
    counts = np.empty(cache.num_rows, dtype="<i8")
    for row_index in range(cache.num_rows):
        start = int(offsets[row_index])
        stop = int(offsets[row_index + 1])
        counts[row_index] = np.count_nonzero(labels[start:stop] != -100)
    if np.any(counts <= 0):
        bad_rows = np.flatnonzero(counts <= 0)
        preview = ", ".join(str(int(index)) for index in bad_rows[:8])
        raise ValueError(
            "SFT cache contains row(s) without supervised targets: "
            f"{preview}"
        )

    metadata = _checked_metadata(
        cache.metadata_path,
        "cache_hash",
        "interleaved-sft-cache-v1",
    )
    declared_total = metadata.get("supervised_targets")
    actual_total = int(counts.sum(dtype=np.int64))
    if declared_total is not None and declared_total != actual_total:
        raise ValueError(
            "SFT cache supervised target metadata differs from labels: "
            f"{declared_total} != {actual_total}"
        )
    return counts


def build_leg_manifests(
    selection_manifest_path: str | Path,
    sft_cache_dir: str | Path,
    output_root: str | Path,
    *,
    source_manifest_path: str | Path | None = None,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    leg_target_tokens: int | None = None,
    world_size: int = DEFAULT_WORLD_SIZE,
    local_batch_size: int = DEFAULT_LOCAL_BATCH_SIZE,
    split_seed: int = 42,
    p1_seed: int = 42,
    p2_seed: int = 43,
    canary_world_size: int = 1,
    canary_local_batch_size: int = 2,
    canary_total_steps: int | None = None,
    expected_sft_supervised_targets: tuple[int, int] | None = None,
) -> dict[str, Path]:
    """Build P1/P2, their composite Exp2 descriptor, and a mixed canary."""

    selection = PretrainSelection.load(selection_manifest_path)
    cache = SFTCache.load(sft_cache_dir, verify_large_files=False)
    if selection.target_tokens % 2:
        raise ValueError("Pretraining target count must split evenly")
    if leg_target_tokens is None:
        leg_target_tokens = selection.target_tokens // 2
    leg_target_tokens = int(leg_target_tokens)
    if selection.target_tokens != 2 * leg_target_tokens:
        raise ValueError("Selection must contain exactly two equal target legs")
    if cache.sequence_length != int(sequence_length):
        raise ValueError("SFT cache context does not match requested context")
    if expected_sft_supervised_targets is not None:
        if (
            not isinstance(expected_sft_supervised_targets, tuple)
            or len(expected_sft_supervised_targets) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                for value in expected_sft_supervised_targets
            )
        ):
            raise ValueError(
                "expected_sft_supervised_targets must be a tuple of two "
                "positive integers"
            )
        expected_sft_supervised_targets = tuple(
            int(value) for value in expected_sft_supervised_targets
        )

    if source_manifest_path is not None:
        source_hash = SourceShardManifest.load(source_manifest_path).manifest_hash
        if source_hash != selection.source_manifest_hash:
            raise ValueError("Source and selection manifest hashes differ")
    else:
        source_hash = selection.source_manifest_hash

    sft_order = np.random.Generator(
        np.random.PCG64(int(split_seed))
    ).permutation(cache.num_rows)
    # Preserve every source row when the pinned corpus has an odd size: P1
    # receives floor(N/2), and P2 receives the remaining ceil(N/2).
    split_sft = cache.num_rows // 2
    sft_halves = (sft_order[:split_sft], sft_order[split_sft:])
    sft_targets_per_row = _sft_supervised_targets_per_row(cache)
    sft_supervised_target_halves = tuple(
        int(sft_targets_per_row[indices].sum(dtype=np.int64))
        for indices in sft_halves
    )
    if expected_sft_supervised_targets is not None and (
        sft_supervised_target_halves
        != expected_sft_supervised_targets
    ):
        raise ValueError(
            "SFT supervised-target split is "
            f"{sft_supervised_target_halves}; expected exactly "
            f"{expected_sft_supervised_targets}"
        )
    pretrain_records = math.ceil(leg_target_tokens / int(sequence_length))
    global_batch = int(world_size) * int(local_batch_size)
    output_root = Path(output_root)

    built: dict[str, LegManifest] = {}
    for leg_index, (leg, order_seed) in enumerate(
        (("p1", int(p1_seed)), ("p2", int(p2_seed)))
    ):
        order, padding = _shuffled_leg_order(
            pretrain_records=pretrain_records,
            sft_indices=sft_halves[leg_index],
            global_batch_size=global_batch,
            seed=order_seed,
        )
        built[leg] = _write_leg_manifest(
            output_root / leg,
            leg=leg,
            order=order,
            target_start=leg_index * leg_target_tokens,
            target_count=leg_target_tokens,
            sequence_length=sequence_length,
            pretrain_records=pretrain_records,
            sft_records=len(sft_halves[leg_index]),
            sft_supervised_targets=(
                sft_supervised_target_halves[leg_index]
            ),
            padding_records=padding,
            world_size=world_size,
            local_batch_size=local_batch_size,
            total_steps=None,
            source_manifest_hash=source_hash,
            selection_hash=selection.selection_hash,
            sft_cache_hash=cache.cache_hash,
            shuffle_seed=order_seed,
        )

    if (
        selection.target_tokens == DEFAULT_TOTAL_TARGETS
        and int(sequence_length) == DEFAULT_SEQUENCE_LENGTH
        and cache.num_rows == DEFAULT_SFT_ROWS
        and global_batch == DEFAULT_WORLD_SIZE * DEFAULT_LOCAL_BATCH_SIZE
    ):
        expected_sft = (38_858, 38_859)
        expected_padding = (97, 96)
        for leg_index, leg in enumerate((built["p1"], built["p2"])):
            if (
                leg.pretrain_records != 1_627_605
                or leg.sft_records != expected_sft[leg_index]
                or leg.padding_records != expected_padding[leg_index]
                or leg.physical_steps != 9_920
            ):
                raise AssertionError("Production leg accounting drifted")

    # Exp2 consumes the exact P1 order followed by the exact P2 order.
    exp2_dir = output_root / "exp2"
    exp2_dir.mkdir(parents=True, exist_ok=True)
    exp2_value: dict[str, Any] = {
        "schema": "interleaved-composite-manifest-v1",
        "schema_version": SCHEMA_VERSION,
        "name": "p1+p2",
        "components": [
            {
                "path": os.path.relpath(
                    built["p1"].metadata_path, start=exp2_dir
                ),
                "sha256": _sha256_file(built["p1"].metadata_path),
            },
            {
                "path": os.path.relpath(
                    built["p2"].metadata_path, start=exp2_dir
                ),
                "sha256": _sha256_file(built["p2"].metadata_path),
            },
        ],
        "total_steps": built["p1"].total_steps + built["p2"].total_steps,
        "source_manifest_hash": source_hash,
        "selection_hash": selection.selection_hash,
        "sft_cache_hash": cache.cache_hash,
    }
    exp2_value["metadata_hash"] = _hash_dict(exp2_value, "metadata_hash")
    _atomic_json(exp2_dir / "metadata.json", exp2_value)

    # A distinct one-batch order guarantees the canary sees both objectives.
    canary_sft_code = -(int(sft_halves[0][0]) + 1)
    canary_global_batch = int(canary_world_size) * int(canary_local_batch_size)
    if canary_global_batch < 2:
        raise ValueError("Canary global batch must fit one PT and one SFT record")
    canary_order = np.full(canary_global_batch, PAD_RECORD, dtype="<i8")
    canary_order[0] = 0
    canary_order[1] = canary_sft_code
    canary = _write_leg_manifest(
        output_root / "canary",
        leg="canary",
        order=canary_order,
        target_start=0,
        target_count=leg_target_tokens,
        sequence_length=sequence_length,
        pretrain_records=1,
        sft_records=1,
        sft_supervised_targets=int(
            sft_targets_per_row[int(sft_halves[0][0])]
        ),
        padding_records=canary_global_batch - 2,
        world_size=canary_world_size,
        local_batch_size=canary_local_batch_size,
        total_steps=canary_total_steps or built["p1"].total_steps,
        source_manifest_hash=source_hash,
        selection_hash=selection.selection_hash,
        sft_cache_hash=cache.cache_hash,
        shuffle_seed=None,
    )

    return {
        "p1": built["p1"].metadata_path,
        "p2": built["p2"].metadata_path,
        "p1+p2": exp2_dir / "metadata.json",
        "canary": canary.metadata_path,
    }


def build_manifest_set(
    output_path: str | Path,
    *,
    source_manifest_path: str | Path,
    selection_manifest_path: str | Path,
    sft_cache_dir: str | Path,
    legs_root: str | Path,
    experiment_version: str = "v1",
    source_repo: str | None = None,
    source_revision: str | None = None,
    sft_repo: str | None = None,
    sft_revision: str | None = None,
    pretrain_tokens: int = DEFAULT_TOTAL_TARGETS,
    sft_rows: int = DEFAULT_SFT_ROWS,
) -> dict[str, Any]:
    """Write the small provenance/index file consumed by Modal launchers."""

    output_path = Path(output_path)
    source = SourceShardManifest.load(source_manifest_path)
    selection = PretrainSelection.load(selection_manifest_path)
    cache = SFTCache.load(sft_cache_dir, verify_large_files=False)
    legs_root = Path(legs_root)
    manifest_paths = {
        "p1": legs_root / "p1" / "metadata.json",
        "p2": legs_root / "p2" / "metadata.json",
        "p1+p2": legs_root / "exp2" / "metadata.json",
        "canary": legs_root / "canary" / "metadata.json",
    }
    for path in manifest_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    value: dict[str, Any] = {
        "schema": "interleaved-manifest-set-v1",
        "schema_version": SCHEMA_VERSION,
        "experiment_version": experiment_version,
        "source_repo": source_repo,
        "source_revision": source_revision,
        "sft_repo": sft_repo,
        "sft_revision": sft_revision,
        "pretrain_tokens": int(pretrain_tokens),
        "sft_rows": int(sft_rows),
        "source_manifest_hash": source.manifest_hash,
        "selection_hash": selection.selection_hash,
        "sft_cache_hash": cache.cache_hash,
        "manifests": {
            name: {
                "path": os.path.relpath(path, start=output_path.parent),
                "sha256": _sha256_file(path),
            }
            for name, path in manifest_paths.items()
        },
    }
    value["manifest_set_hash"] = _hash_dict(value, "manifest_set_hash")
    _atomic_json(output_path, value)
    return _checked_metadata(
        output_path, "manifest_set_hash", "interleaved-manifest-set-v1"
    )


class InterleavedLegDataset(Dataset):
    def __init__(
        self,
        pretrain: PackedPretrainDataset,
        sft: SFTCacheDataset,
        manifest: LegManifest,
        order: np.ndarray,
    ):
        self.pretrain = pretrain
        self.sft = sft
        self.manifest = manifest
        self.order = order

    def __len__(self) -> int:
        return len(self.order)

    def __getitem__(self, position: int) -> dict[str, Any]:
        code = int(self.order[position])
        if code == int(PAD_RECORD):
            return {
                "input_ids": torch.empty(0, dtype=torch.long),
                "labels": torch.empty(0, dtype=torch.long),
                "attention_mask": torch.empty(0, dtype=torch.long),
                "sample_type": SAMPLE_PAD,
                "record_id": code,
                "valid_targets": 0,
                "manifest_position": int(position),
            }
        if code >= 0:
            sample = self.pretrain[code]
        else:
            sample = self.sft[-code - 1]
        sample["manifest_position"] = int(position)
        return sample


class UnifiedInterleavedCollator:
    def __init__(self, *, sequence_length: int, pad_token_id: int):
        self.sequence_length = int(sequence_length)
        self.pad_token_id = int(pad_token_id)

    def __call__(self, samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        batch_size = len(samples)
        shape = (batch_size, self.sequence_length)
        input_ids = torch.full(shape, self.pad_token_id, dtype=torch.long)
        labels = torch.full(shape, -100, dtype=torch.long)
        attention_mask = torch.zeros(shape, dtype=torch.long)
        sample_type = torch.empty(batch_size, dtype=torch.long)
        record_id = torch.empty(batch_size, dtype=torch.long)
        positions = torch.empty(batch_size, dtype=torch.long)
        valid_targets = torch.empty(batch_size, dtype=torch.long)

        for row_index, sample in enumerate(samples):
            kind = int(sample["sample_type"])
            sample_type[row_index] = kind
            record_id[row_index] = int(sample["record_id"])
            positions[row_index] = int(sample["manifest_position"])
            valid_targets[row_index] = int(sample["valid_targets"])
            if kind == SAMPLE_PAD:
                # An all-zero attention row can create all-masked softmax/NaNs.
                # Inputs are harmless pad IDs and labels remain entirely ignored.
                attention_mask[row_index].fill_(1)
                continue
            length = int(len(sample["input_ids"]))
            if length <= 0 or length > self.sequence_length:
                raise ValueError(f"Invalid mixed-record length {length}")
            input_ids[row_index, :length] = sample["input_ids"]
            labels[row_index, :length] = sample["labels"]
            attention_mask[row_index, :length] = sample["attention_mask"]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "sample_type": sample_type,
            "record_id": record_id,
            "manifest_position": positions,
            "valid_targets": valid_targets,
        }


class DistributedManifestBatchSampler(Sampler[list[int]]):
    """Rank-local slices of immutable global optimizer-step batches."""

    def __init__(
        self,
        *,
        num_records: int,
        rank: int,
        world_size: int,
        local_batch_size: int,
        start_cursor: int,
    ):
        self.num_records = int(num_records)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.local_batch_size = int(local_batch_size)
        self.global_batch_size = self.world_size * self.local_batch_size
        if not 0 <= self.rank < self.world_size:
            raise ValueError("rank is outside world_size")
        if self.global_batch_size <= 0 or self.num_records % self.global_batch_size:
            raise ValueError("Manifest records must divide into exact global batches")
        self.total_steps = self.num_records // self.global_batch_size
        self.start_cursor = int(start_cursor)
        if not 0 <= self.start_cursor <= self.total_steps:
            raise ValueError("start_cursor is outside the physical manifest")

    def __iter__(self) -> Iterator[list[int]]:
        rank_offset = self.rank * self.local_batch_size
        for step in range(self.start_cursor, self.total_steps):
            start = step * self.global_batch_size + rank_offset
            yield list(range(start, start + self.local_batch_size))

    def __len__(self) -> int:
        return self.total_steps - self.start_cursor


class InterleavedDataStream:
    """Iterable rank-local stream with an explicit committed global cursor."""

    def __init__(
        self,
        *,
        dataloader: DataLoader,
        sampler: DistributedManifestBatchSampler,
        manifest: LegManifest,
        manifest_file_hash: str,
        rank: int,
    ):
        self.dataloader = dataloader
        self.sampler = sampler
        self.leg_manifest = manifest
        self.manifest_hash = manifest_file_hash
        self.source_manifest_hash = manifest.source_manifest_hash
        self.selection_hash = manifest.selection_hash
        self.sft_cache_hash = manifest.sft_cache_hash
        self.rank = int(rank)
        self.cursor = sampler.start_cursor
        self.total_steps = manifest.total_steps
        self.physical_steps = manifest.physical_steps
        self._pending_cursor: int | None = None
        self._iteration_started = False

    @property
    def remaining_steps(self) -> int:
        return max(0, self.physical_steps - self.cursor)

    def __len__(self) -> int:
        return self.remaining_steps

    def __iter__(self) -> Iterator[dict[str, Any]]:
        if self._iteration_started:
            raise RuntimeError("An interleaved stream iterator is already active")
        self._iteration_started = True
        try:
            expected = self.cursor
            for batch in self.dataloader:
                if self._pending_cursor is not None:
                    raise RuntimeError(
                        "commit_step() is required before requesting another batch"
                    )
                if self.cursor != expected:
                    raise RuntimeError("Committed data cursor changed unexpectedly")
                batch["cursor_start"] = expected
                batch["cursor_end"] = expected + 1
                batch["manifest_hash"] = self.manifest_hash
                self._pending_cursor = expected + 1
                yield batch
                if self._pending_cursor is not None:
                    raise RuntimeError(
                        "Batch was consumed without a successful commit_step()"
                    )
                expected += 1
        finally:
            self._iteration_started = False

    def commit_step(self) -> None:
        if self._pending_cursor is None:
            raise RuntimeError("No yielded batch is pending commit")
        self.cursor = self._pending_cursor
        self._pending_cursor = None

    def state_dict(self) -> dict[str, Any]:
        if self._pending_cursor is not None:
            raise RuntimeError("Cannot checkpoint an uncommitted data batch")
        # Deliberately rank-agnostic: rank 0 writes this once and all ranks load it.
        return {
            "schema": "interleaved-stream-state-v1",
            "manifest_hash": self.manifest_hash,
            "source_manifest_hash": self.source_manifest_hash,
            "selection_hash": self.selection_hash,
            "sft_cache_hash": self.sft_cache_hash,
            "cursor": self.cursor,
            "world_size": self.sampler.world_size,
            "local_batch_size": self.sampler.local_batch_size,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if self._iteration_started or self._pending_cursor is not None:
            raise RuntimeError("Data state must be loaded before iteration")
        expected = {
            "schema": "interleaved-stream-state-v1",
            "manifest_hash": self.manifest_hash,
            "source_manifest_hash": self.source_manifest_hash,
            "selection_hash": self.selection_hash,
            "sft_cache_hash": self.sft_cache_hash,
            "world_size": self.sampler.world_size,
            "local_batch_size": self.sampler.local_batch_size,
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise ValueError(
                    f"Resume data state mismatch for {key}: "
                    f"{state.get(key)!r} != {value!r}"
                )
        cursor = int(state.get("cursor", -1))
        if cursor != self.cursor:
            raise ValueError(
                "Factory start_cursor and loaded stream cursor differ: "
                f"{self.cursor} != {cursor}"
            )


class CompositeInterleavedDataStream:
    """Sequential P1+P2 view used by Exp2's one optimizer/scheduler arc."""

    def __init__(
        self,
        *,
        streams: Sequence[InterleavedDataStream],
        manifest_hash: str,
        total_steps: int,
        start_cursor: int,
        source_manifest_hash: str,
        selection_hash: str,
        sft_cache_hash: str,
        world_size: int,
        local_batch_size: int,
    ):
        self.streams = tuple(streams)
        self.manifest_hash = manifest_hash
        self.total_steps = int(total_steps)
        self.cursor = int(start_cursor)
        self.source_manifest_hash = source_manifest_hash
        self.selection_hash = selection_hash
        self.sft_cache_hash = sft_cache_hash
        self.world_size = int(world_size)
        self.local_batch_size = int(local_batch_size)
        self._pending_stream: InterleavedDataStream | None = None
        if not 0 <= self.cursor <= self.total_steps:
            raise ValueError("Composite start cursor is outside total steps")
        if sum(stream.remaining_steps for stream in streams) != self.remaining_steps:
            raise ValueError("Remaining composite component steps are inconsistent")

    @property
    def remaining_steps(self) -> int:
        return self.total_steps - self.cursor

    def __len__(self) -> int:
        return self.remaining_steps

    def __iter__(self) -> Iterator[dict[str, Any]]:
        global_cursor = self.cursor
        for stream in self.streams:
            component_offset = global_cursor - stream.cursor
            for batch in stream:
                batch["cursor_start"] = global_cursor
                batch["cursor_end"] = global_cursor + 1
                batch["manifest_hash"] = self.manifest_hash
                self._pending_stream = stream
                yield batch
                if self._pending_stream is not None:
                    raise RuntimeError(
                        "Composite batch was consumed without commit_step()"
                    )
                global_cursor += 1
            # The offset is only an invariant aid; it is constant within a component.
            if global_cursor - stream.cursor != component_offset:
                raise AssertionError("Composite cursor accounting drifted")

    def commit_step(self) -> None:
        if self._pending_stream is None:
            raise RuntimeError("No composite batch is pending commit")
        self._pending_stream.commit_step()
        self._pending_stream = None
        self.cursor += 1

    def state_dict(self) -> dict[str, Any]:
        if self._pending_stream is not None:
            raise RuntimeError("Cannot checkpoint an uncommitted composite batch")
        return {
            "schema": "interleaved-stream-state-v1",
            "manifest_hash": self.manifest_hash,
            "source_manifest_hash": self.source_manifest_hash,
            "selection_hash": self.selection_hash,
            "sft_cache_hash": self.sft_cache_hash,
            "cursor": self.cursor,
            "world_size": self.world_size,
            "local_batch_size": self.local_batch_size,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        current = self.state_dict()
        for key, value in current.items():
            if state.get(key) != value:
                raise ValueError(
                    f"Composite resume state mismatch for {key}: "
                    f"{state.get(key)!r} != {value!r}"
                )


def _create_leg_stream(
    *,
    source_root: str | Path,
    source_manifest: SourceShardManifest,
    selection: PretrainSelection,
    cache: SFTCache,
    metadata_path: Path,
    pad_token_id: int,
    bos_token_id: int,
    rank: int,
    world_size: int,
    local_batch_size: int,
    start_cursor: int,
    num_workers: int,
    max_open_shards: int,
) -> InterleavedDataStream:
    manifest = LegManifest.load(metadata_path)
    if manifest.source_manifest_hash != source_manifest.manifest_hash:
        raise ValueError("Leg and source manifest hashes differ")
    if manifest.selection_hash != selection.selection_hash:
        raise ValueError("Leg and selection hashes differ")
    if manifest.sft_cache_hash != cache.cache_hash:
        raise ValueError("Leg and SFT cache hashes differ")
    if (manifest.world_size, manifest.local_batch_size) != (
        int(world_size),
        int(local_batch_size),
    ):
        raise ValueError(
            "Runtime topology differs from immutable leg topology: "
            f"{(world_size, local_batch_size)} != "
            f"{(manifest.world_size, manifest.local_batch_size)}"
        )
    logical = LogicalTokenSelection(
        source_root,
        source_manifest,
        selection,
        max_open_shards=max_open_shards,
    )
    pretrain = PackedPretrainDataset(
        logical,
        target_start=manifest.target_start,
        target_count=manifest.target_count,
        bos_token_id=bos_token_id,
        sequence_length=manifest.sequence_length,
    )
    sft = SFTCacheDataset(cache)
    order = np.load(manifest.order_path, mmap_mode="r", allow_pickle=False)
    dataset = InterleavedLegDataset(pretrain, sft, manifest, order)
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
    return InterleavedDataStream(
        dataloader=loader,
        sampler=sampler,
        manifest=manifest,
        manifest_file_hash=_sha256_file(metadata_path),
        rank=rank,
    )


def create_interleaved_dataloader(
    *,
    source_root: str | Path,
    source_manifest_path: str | Path,
    selection_manifest_path: str | Path,
    sft_cache_dir: str | Path,
    leg_manifest_path: str | Path,
    pad_token_id: int,
    bos_token_id: int,
    rank: int,
    world_size: int = DEFAULT_WORLD_SIZE,
    local_batch_size: int = DEFAULT_LOCAL_BATCH_SIZE,
    start_cursor: int = 0,
    num_workers: int = 0,
    max_open_shards: int = 64,
) -> InterleavedDataStream | CompositeInterleavedDataStream:
    """Open one immutable production leg or the Exp2 P1+P2 composite."""

    source = SourceShardManifest.load(source_manifest_path)
    selection = PretrainSelection.load(selection_manifest_path)
    cache = SFTCache.load(sft_cache_dir, verify_large_files=False)
    metadata_path = Path(leg_manifest_path)
    header = _load_json(metadata_path)
    schema = header.get("schema")
    if schema == "interleaved-leg-manifest-v1":
        return _create_leg_stream(
            source_root=source_root,
            source_manifest=source,
            selection=selection,
            cache=cache,
            metadata_path=metadata_path,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            rank=rank,
            world_size=world_size,
            local_batch_size=local_batch_size,
            start_cursor=start_cursor,
            num_workers=num_workers,
            max_open_shards=max_open_shards,
        )
    if schema != "interleaved-composite-manifest-v1":
        raise ValueError(f"Unsupported interleaved manifest schema: {schema!r}")

    composite = _checked_metadata(
        metadata_path, "metadata_hash", "interleaved-composite-manifest-v1"
    )
    component_headers: list[tuple[Path, LegManifest]] = []
    for item in composite["components"]:
        path = (metadata_path.parent / item["path"]).resolve()
        if _sha256_file(path) != item["sha256"]:
            raise ValueError(f"Composite component hash mismatch: {path}")
        component_headers.append((path, LegManifest.load(path)))

    streams: list[InterleavedDataStream] = []
    consumed = 0
    for path, leg in component_headers:
        local_start = min(max(int(start_cursor) - consumed, 0), leg.total_steps)
        consumed += leg.total_steps
        if local_start >= leg.total_steps:
            continue
        streams.append(
            _create_leg_stream(
                source_root=source_root,
                source_manifest=source,
                selection=selection,
                cache=cache,
                metadata_path=path,
                pad_token_id=pad_token_id,
                bos_token_id=bos_token_id,
                rank=rank,
                world_size=world_size,
                local_batch_size=local_batch_size,
                start_cursor=local_start,
                num_workers=num_workers,
                max_open_shards=max_open_shards,
            )
        )
    return CompositeInterleavedDataStream(
        streams=streams,
        manifest_hash=_sha256_file(metadata_path),
        total_steps=int(composite["total_steps"]),
        start_cursor=int(start_cursor),
        source_manifest_hash=source.manifest_hash,
        selection_hash=selection.selection_hash,
        sft_cache_hash=cache.cache_hash,
        world_size=world_size,
        local_batch_size=local_batch_size,
    )


__all__ = [
    "DEFAULT_LEG_TARGETS",
    "DEFAULT_LOCAL_BATCH_SIZE",
    "DEFAULT_SEQUENCE_LENGTH",
    "DEFAULT_SFT_ROWS",
    "DEFAULT_TOTAL_TARGETS",
    "DEFAULT_WORLD_SIZE",
    "PAD_RECORD",
    "SAMPLE_PAD",
    "SAMPLE_PRETRAIN",
    "SAMPLE_SFT",
    "SFT_RESPONSE_NORMALIZATION_STRIP_VERIFY_V1",
    "SFT_STRICT_AUDIT_SCHEMA_V1",
    "SFT_SUPERVISED_DELIMITERS",
    "SFT_SUPERVISED_UNK_POLICY_REJECT_V1",
    "CompositeInterleavedDataStream",
    "InterleavedDataStream",
    "LegManifest",
    "LogicalTokenSelection",
    "PackedPretrainDataset",
    "PretrainSelection",
    "SFTCache",
    "SFTCacheDataset",
    "SourceShardManifest",
    "UnifiedInterleavedCollator",
    "build_leg_manifests",
    "build_manifest_set",
    "build_pretrain_selection",
    "build_sft_cache",
    "build_source_manifest",
    "create_interleaved_dataloader",
    "normalize_sft_response",
    "tokenize_masked_sft_row",
]
