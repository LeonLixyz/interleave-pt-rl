"""Frozen-prompt, fixed-policy generation for the math D0/D1 experiment.

This module intentionally has two layers:

* Pure helpers for prompt identity, deterministic seed derivation, record
  validation, canonical JSONL, and fake/local generation tests.
* Thin Modal functions that freeze the SkyEasy train prompt pool and run vLLM
  against the audited pre-RL (D0) or post-RL (D1) checkpoints.

The artifact boundary is token IDs.  Decoded text is retained for verification
and human audit, but is never re-tokenized when the transfer corpus is built.

Examples (these commands are declarations only; do not run them from tests):

    modal run transfer_generate.py::freeze_prompt_manifest_remote \
        --output-path /checkpoints/transfer_data/pilot/prompts.jsonl

    modal run --detach transfer_generate.py::generate_fixed_policy_remote \
        --arm D0 \
        --prompt-manifest-path /checkpoints/transfer_data/pilot/prompts.jsonl \
        --output-dir /checkpoints/transfer_data/pilot/raw/D0

Use the exact same prompt manifest and sampling arguments for D0 and D1.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence


PROMPT_SCHEMA_VERSION = "math-transfer-prompt-v1"
GENERATION_SCHEMA_VERSION = "math-transfer-generation-v1"
MANIFEST_SCHEMA_VERSION = "math-transfer-manifest-v1"

CHECKPOINT_VOLUME_NAME = "olmo-core-checkpoints-v2"
CHECKPOINT_MOUNT = "/checkpoints"
CACHE_VOLUME_NAME = "olmo-core-cache"
CACHE_MOUNT = "/cache"

D0_MODEL_PATH = (
    f"{CHECKPOINT_MOUNT}/sft/"
    "math-1b-sft-numinamath-bs512-from-step10000"
)
D1_MODEL_PATH = f"{CHECKPOINT_MOUNT}/interleave/armB_small/rl_leg1_hf"
MODEL_PATHS = {"D0": D0_MODEL_PATH, "D1": D1_MODEL_PATH}

DEFAULT_DATASET_PATH = (
    f"{CHECKPOINT_MOUNT}/rl_data/skyeasy25k_omi2/train.parquet"
)
DEFAULT_MAX_PROMPT_TOKENS = 512
DEFAULT_EOS_TOKEN_ID = 100257
KNOWN_DIFFICULTY_BINS = ("0", "1", "2-3", "4+")

_MODEL_BUNDLE_FILES = (
    "model.safetensors",
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
)


@dataclass(frozen=True)
class SamplingConfig:
    """Every generation parameter that can change the sampled corpus."""

    samples_per_prompt: int = 16
    temperature: float = 1.0
    top_p: float = 1.0
    max_new_tokens: int = 3584
    base_seed: int = 0

    def __post_init__(self) -> None:
        if self.samples_per_prompt <= 0:
            raise ValueError("samples_per_prompt must be positive")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")


def canonical_json_bytes(value: Any) -> bytes:
    """Stable UTF-8 JSON used for hashes and byte-reproducible JSONL."""

    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def hash_model_bundle(model_path: str | os.PathLike[str]) -> dict[str, Any]:
    """Hash only inference-relevant files, excluding trainer checkpoints."""

    root = Path(model_path)
    files: list[dict[str, Any]] = []
    for name in _MODEL_BUNDLE_FILES:
        path = root / name
        if path.is_file():
            files.append(
                {
                    "path": name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    if not files or not any(row["path"] == "model.safetensors" for row in files):
        raise FileNotFoundError(f"incomplete inference model bundle: {root}")
    bundle_id = sha256_bytes(b"".join(canonical_json_bytes(row) for row in files))
    return {"bundle_sha256": bundle_id, "files": files}


def atomic_write_bytes(path: str | os.PathLike[str], payload: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with open(temporary, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def write_jsonl_atomic(path: str | os.PathLike[str], rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    payload = b"".join(canonical_json_bytes(dict(row)) for row in rows)
    atomic_write_bytes(path, payload)
    return {
        "path": Path(path).name,
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
        "rows": payload.count(b"\n"),
    }


def read_jsonl(path: str | os.PathLike[str]) -> Iterator[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield value


def write_manifest_atomic(path: str | os.PathLike[str], core: Mapping[str, Any]) -> dict[str, Any]:
    """Write a self-identifying manifest without nondeterministic timestamps."""

    document = dict(core)
    document.setdefault("schema_version", MANIFEST_SCHEMA_VERSION)
    document["content_id"] = sha256_bytes(canonical_json_bytes(document))
    atomic_write_bytes(path, canonical_json_bytes(document))
    return document


def normalize_problem_text(text: str) -> str:
    """Conservative normalization for semantic duplicate grouping."""

    normalized = unicodedata.normalize("NFKC", str(text))
    return re.sub(r"\s+", " ", normalized).strip()


def prompt_dedup_group(messages: Sequence[Mapping[str, Any]]) -> str:
    user_parts = [
        normalize_problem_text(str(message.get("content", "")))
        for message in messages
        if str(message.get("role", "")) == "user"
    ]
    if not user_parts:
        user_parts = [
            normalize_problem_text(str(message.get("content", "")))
            for message in messages
        ]
    return sha256_bytes("\n".join(user_parts).encode("utf-8"))


def make_problem_uid(dataset_sha256: str, split: str, dataset_index: int) -> str:
    identity = {
        "dataset_sha256": dataset_sha256,
        "dataset_index": int(dataset_index),
        "split": str(split),
    }
    return sha256_bytes(canonical_json_bytes(identity))


def derive_sample_seed(base_seed: int, problem_uid: str, sample_index: int) -> int:
    """Stable 31-bit seed, independent of Python hash randomization."""

    if sample_index < 0:
        raise ValueError("sample_index must be non-negative")
    material = f"{int(base_seed)}\0{problem_uid}\0{int(sample_index)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (2**31 - 1)


def _difficulty_bin(value: Any) -> str:
    if value is None:
        return "missing"
    try:
        integer = int(value)
    except (TypeError, ValueError):
        return "other"
    if integer == 0:
        return "0"
    if integer == 1:
        return "1"
    if integer <= 3:
        return "2-3"
    return "4+"


def make_prompt_record(
    *,
    dataset_sha256: str,
    split: str,
    dataset_index: int,
    data_source: str,
    messages: Sequence[Mapping[str, Any]],
    prompt_token_ids: Sequence[int],
    ground_truth: Any,
    model_difficulty: Mapping[str, Any] | None = None,
    difficulty_model: str | None = None,
) -> dict[str, Any]:
    """Create and validate one frozen prompt row from already-tokenized input."""

    canonical_messages = [
        {"role": str(message["role"]), "content": str(message["content"])}
        for message in messages
    ]
    token_ids = [int(token_id) for token_id in prompt_token_ids]
    if not canonical_messages:
        raise ValueError("prompt messages may not be empty")
    if not token_ids or min(token_ids) < 0:
        raise ValueError("prompt_token_ids must contain non-negative integers")
    difficulty_map = {
        str(key): value for key, value in sorted((model_difficulty or {}).items())
    }
    difficulty_value = difficulty_map.get(difficulty_model) if difficulty_model else None
    record = {
        "schema_version": PROMPT_SCHEMA_VERSION,
        "problem_uid": make_problem_uid(dataset_sha256, split, dataset_index),
        "dedup_group": prompt_dedup_group(canonical_messages),
        "dataset_sha256": dataset_sha256,
        "dataset_split": str(split),
        "dataset_index": int(dataset_index),
        "data_source": str(data_source),
        "messages": canonical_messages,
        "prompt_token_ids": token_ids,
        "prompt_token_count": len(token_ids),
        "ground_truth": ground_truth,
        "model_difficulty": difficulty_map,
        "difficulty_model": difficulty_model,
        "difficulty_bin": _difficulty_bin(difficulty_value),
    }
    validate_prompt_record(record)
    return record


def validate_prompt_record(record: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "problem_uid",
        "dedup_group",
        "dataset_sha256",
        "dataset_split",
        "dataset_index",
        "messages",
        "prompt_token_ids",
        "prompt_token_count",
        "ground_truth",
        "difficulty_bin",
    }
    missing = sorted(required - set(record))
    if missing:
        raise ValueError(f"prompt record missing fields: {missing}")
    if record["schema_version"] != PROMPT_SCHEMA_VERSION:
        raise ValueError(f"unsupported prompt schema: {record['schema_version']}")
    token_ids = record["prompt_token_ids"]
    if not isinstance(token_ids, list) or not token_ids:
        raise ValueError("prompt_token_ids must be a non-empty list")
    if int(record["prompt_token_count"]) != len(token_ids):
        raise ValueError("prompt_token_count does not match prompt_token_ids")
    if any(not isinstance(value, int) or value < 0 for value in token_ids):
        raise ValueError("prompt_token_ids must be non-negative integers")


def freeze_prompt_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    dataset_sha256: str,
    split: str,
    tokenize_prompt: Callable[[Sequence[Mapping[str, Any]]], Sequence[int]],
    max_prompt_tokens: int = DEFAULT_MAX_PROMPT_TOKENS,
    difficulty_model: str | None = "DeepSeek-R1-Distill-Qwen-1.5B",
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Freeze, filter, and sort a dataset using an injected tokenizer function."""

    kept: list[dict[str, Any]] = []
    stats = {"input_rows": 0, "kept_rows": 0, "overlong_rows": 0}
    seen_indices: set[int] = set()
    for ordinal, row in enumerate(rows):
        stats["input_rows"] += 1
        extra_info = row.get("extra_info") or {}
        dataset_index = int(extra_info.get("index", ordinal))
        if dataset_index in seen_indices:
            raise ValueError(f"duplicate dataset index: {dataset_index}")
        seen_indices.add(dataset_index)
        messages = row.get("prompt")
        if not isinstance(messages, list):
            raise ValueError(f"dataset index {dataset_index}: prompt is not a message list")
        prompt_token_ids = [int(value) for value in tokenize_prompt(messages)]
        if len(prompt_token_ids) > max_prompt_tokens:
            stats["overlong_rows"] += 1
            continue
        reward_model = row.get("reward_model") or {}
        kept.append(
            make_prompt_record(
                dataset_sha256=dataset_sha256,
                split=split,
                dataset_index=dataset_index,
                data_source=str(row.get("data_source", "")),
                messages=messages,
                prompt_token_ids=prompt_token_ids,
                ground_truth=reward_model.get("ground_truth"),
                model_difficulty=extra_info.get("model_difficulty"),
                difficulty_model=difficulty_model,
            )
        )
    kept.sort(key=lambda row: (row["dataset_index"], row["problem_uid"]))
    stats["kept_rows"] = len(kept)
    return kept, stats


def domain_separated_hash_rank(
    *, split_seed: int, domain: str, identities: Sequence[str]
) -> str:
    """Return a deterministic rank hash with explicit domain separation."""

    if not domain:
        raise ValueError("hash-rank domain may not be empty")
    payload = {
        "domain": str(domain),
        "identities": [str(value) for value in identities],
        "split_seed": int(split_seed),
    }
    return sha256_bytes(canonical_json_bytes(payload))


def split_semantic_prompt_pool(
    prompts: Iterable[Mapping[str, Any]],
    *,
    split_seed: int,
    generation_groups: int = 18_000,
    dev_groups: int = 2_000,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Freeze one representative per semantic group and make disjoint pools.

    Representatives and pool order use separate hash domains.  The first
    ``generation_groups`` entries in the hash-ranked permutation form the
    transfer-generation pool, the next ``dev_groups`` form the disjoint dev
    pool, and every remaining semantic group is held in reserve.

    The split operates on semantic groups, not dataset rows.  With the audited
    20,658 eligible groups, the preregistered feasible split is therefore
    18,000 generation + 2,000 dev + 658 reserve groups.
    """

    if generation_groups <= 0:
        raise ValueError("generation_groups must be positive")
    if dev_groups < 0:
        raise ValueError("dev_groups may not be negative")
    rows = [dict(prompt) for prompt in prompts]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        validate_prompt_record(row)
        grouped.setdefault(str(row["dedup_group"]), []).append(row)
    if generation_groups + dev_groups > len(grouped):
        raise ValueError(
            "requested semantic split is infeasible: "
            f"generation={generation_groups:,} + dev={dev_groups:,} > "
            f"available_groups={len(grouped):,}"
        )

    representatives: list[dict[str, Any]] = []
    for dedup_group, group_rows in sorted(grouped.items()):
        ranked_rows = []
        for row in group_rows:
            representative_rank = domain_separated_hash_rank(
                split_seed=split_seed,
                domain="math-transfer/semantic-representative/v1",
                identities=(dedup_group, str(row["problem_uid"])),
            )
            ranked_rows.append((representative_rank, str(row["problem_uid"]), row))
        representative_rank, _, representative = min(ranked_rows)
        chosen = dict(representative)
        chosen["semantic_representative_rank_sha256"] = representative_rank
        chosen["pool_split_rank_sha256"] = domain_separated_hash_rank(
            split_seed=split_seed,
            domain="math-transfer/pool-order/v1",
            identities=(dedup_group,),
        )
        representatives.append(chosen)

    representatives.sort(
        key=lambda row: (row["pool_split_rank_sha256"], row["dedup_group"])
    )
    boundaries = {
        "generation": (0, generation_groups),
        "dev": (generation_groups, generation_groups + dev_groups),
        "reserve": (generation_groups + dev_groups, len(representatives)),
    }
    pools: dict[str, list[dict[str, Any]]] = {}
    for split_name, (start, end) in boundaries.items():
        pools[split_name] = []
        for row in representatives[start:end]:
            assigned = dict(row)
            assigned["pool_split"] = split_name
            assigned["pool_split_seed"] = int(split_seed)
            pools[split_name].append(assigned)

    summary = {
        "split_seed": int(split_seed),
        "input_prompt_rows": len(rows),
        "unique_semantic_groups": len(representatives),
        "dropped_duplicate_rows": len(rows) - len(representatives),
        "counts": {name: len(split_rows) for name, split_rows in pools.items()},
        "representative_rank_domain": "math-transfer/semantic-representative/v1",
        "pool_rank_domain": "math-transfer/pool-order/v1",
    }
    return pools, summary


def write_split_prompt_artifacts(
    output_dir: str | os.PathLike[str],
    pools: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    split_summary: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Write the generation/dev/reserve pools and one canonical manifest."""

    expected_names = ("generation", "dev", "reserve")
    if set(pools) != set(expected_names):
        raise ValueError(f"split pools must be exactly {expected_names}")
    seen_groups: set[str] = set()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, dict[str, Any]] = {}
    for split_name in expected_names:
        rows = [dict(row) for row in pools[split_name]]
        for row in rows:
            validate_prompt_record(row)
            if row.get("pool_split") != split_name:
                raise ValueError(
                    f"prompt {row['problem_uid']} has pool_split={row.get('pool_split')!r}, "
                    f"expected {split_name!r}"
                )
            dedup_group = str(row["dedup_group"])
            if dedup_group in seen_groups:
                raise ValueError(f"semantic group appears in multiple pools: {dedup_group}")
            seen_groups.add(dedup_group)
        artifacts[split_name] = write_jsonl_atomic(
            destination / f"{split_name}.jsonl", rows
        )
    if len(seen_groups) != int(split_summary["unique_semantic_groups"]):
        raise ValueError("written semantic-group count does not match split summary")
    return write_manifest_atomic(
        destination / "manifest.json",
        {
            "artifact_kind": "semantic_split_prompt_pool",
            "prompt_schema_version": PROMPT_SCHEMA_VERSION,
            "split_summary": dict(split_summary),
            "provenance": dict(provenance),
            "artifacts": artifacts,
        },
    )


def select_stratified_smoke_prompts(
    prompts: Iterable[Mapping[str, Any]],
    *,
    count: int,
    selection_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select a deterministic, difficulty-balanced smoke set from a dev pool.

    Only the four preregistered known difficulty bins are eligible.  Quotas are
    allocated by deterministic water filling: bins receive equal counts while
    they have capacity, and shortages are redistributed as evenly as possible
    among the remaining bins.  Hash rank chooses prompts within each bin.
    """

    if count <= 0:
        raise ValueError("smoke selection count must be positive")
    rows = [dict(prompt) for prompt in prompts]
    by_bin: dict[str, list[dict[str, Any]]] = {
        bin_name: [] for bin_name in KNOWN_DIFFICULTY_BINS
    }
    seen_uids: set[str] = set()
    seen_groups: set[str] = set()
    excluded_unknown = 0
    for row in rows:
        validate_prompt_record(row)
        if row.get("pool_split") != "dev":
            raise ValueError(
                f"smoke selector requires a frozen dev pool; "
                f"{row['problem_uid']} has pool_split={row.get('pool_split')!r}"
            )
        uid = str(row["problem_uid"])
        dedup_group = str(row["dedup_group"])
        if uid in seen_uids:
            raise ValueError(f"duplicate problem_uid in dev pool: {uid}")
        if dedup_group in seen_groups:
            raise ValueError(f"duplicate semantic group in dev pool: {dedup_group}")
        seen_uids.add(uid)
        seen_groups.add(dedup_group)
        bin_name = str(row["difficulty_bin"])
        if bin_name not in by_bin:
            excluded_unknown += 1
            continue
        ranked = dict(row)
        ranked["smoke_selection_rank_sha256"] = domain_separated_hash_rank(
            split_seed=selection_seed,
            domain="math-transfer/smoke-prompt-rank/v1",
            identities=(bin_name, uid),
        )
        by_bin[bin_name].append(ranked)

    available = {bin_name: len(bin_rows) for bin_name, bin_rows in by_bin.items()}
    total_known = sum(available.values())
    if total_known < count:
        raise ValueError(
            "smoke selection is infeasible: "
            f"requested={count:,} but only {total_known:,} dev prompts have known difficulty"
        )
    for bin_rows in by_bin.values():
        bin_rows.sort(
            key=lambda row: (
                row["smoke_selection_rank_sha256"], row["problem_uid"]
            )
        )

    bin_priority = sorted(
        KNOWN_DIFFICULTY_BINS,
        key=lambda bin_name: (
            domain_separated_hash_rank(
                split_seed=selection_seed,
                domain="math-transfer/smoke-bin-priority/v1",
                identities=(bin_name,),
            ),
            bin_name,
        ),
    )
    quotas = {bin_name: 0 for bin_name in KNOWN_DIFFICULTY_BINS}
    for _ in range(count):
        eligible_bins = [
            bin_name
            for bin_name in bin_priority
            if quotas[bin_name] < available[bin_name]
        ]
        if not eligible_bins:  # guarded by total_known, retained as an invariant.
            raise AssertionError("smoke quota allocator exhausted available prompts")
        minimum_quota = min(quotas[bin_name] for bin_name in eligible_bins)
        chosen_bin = next(
            bin_name
            for bin_name in eligible_bins
            if quotas[bin_name] == minimum_quota
        )
        quotas[chosen_bin] += 1

    selected: list[dict[str, Any]] = []
    for bin_name in KNOWN_DIFFICULTY_BINS:
        for row in by_bin[bin_name][: quotas[bin_name]]:
            chosen = dict(row)
            chosen["smoke_selection_seed"] = int(selection_seed)
            selected.append(chosen)
    selected.sort(
        key=lambda row: (
            KNOWN_DIFFICULTY_BINS.index(str(row["difficulty_bin"])),
            row["smoke_selection_rank_sha256"],
            row["problem_uid"],
        )
    )
    if len(selected) != count:
        raise AssertionError("smoke selector returned the wrong number of prompts")
    summary = {
        "selection_seed": int(selection_seed),
        "requested_count": int(count),
        "source_count": len(rows),
        "known_difficulty_count": total_known,
        "excluded_unknown_difficulty": excluded_unknown,
        "available_bin_counts": available,
        "selected_bin_counts": quotas,
        "prompt_rank_domain": "math-transfer/smoke-prompt-rank/v1",
        "bin_priority_domain": "math-transfer/smoke-bin-priority/v1",
    }
    return selected, summary


def make_generation_record(
    *,
    arm: str,
    prompt: Mapping[str, Any],
    sample_index: int,
    sample_seed: int,
    response_token_ids: Sequence[int],
    response_text: str,
    finish_reason: str | None,
    stop_reason: Any,
    sampling_config: SamplingConfig,
    model_bundle_sha256: str,
    tokenizer_sha256: str,
) -> dict[str, Any]:
    validate_prompt_record(prompt)
    if arm not in MODEL_PATHS:
        raise ValueError(f"arm must be one of {sorted(MODEL_PATHS)}, got {arm!r}")
    response_ids = [int(token_id) for token_id in response_token_ids]
    if any(token_id < 0 for token_id in response_ids):
        raise ValueError("response_token_ids must be non-negative")
    if stop_reason is not None and not isinstance(stop_reason, (str, int, float, bool)):
        stop_reason = str(stop_reason)
    record = {
        "schema_version": GENERATION_SCHEMA_VERSION,
        "arm": arm,
        "problem_uid": prompt["problem_uid"],
        "dedup_group": prompt["dedup_group"],
        "dataset_index": prompt["dataset_index"],
        "data_source": prompt.get("data_source", ""),
        "difficulty_bin": prompt["difficulty_bin"],
        "sample_index": int(sample_index),
        "sample_seed": int(sample_seed),
        "prompt_token_ids": list(prompt["prompt_token_ids"]),
        "prompt_token_count": int(prompt["prompt_token_count"]),
        "response_token_ids": response_ids,
        "response_token_count": len(response_ids),
        "response_text": str(response_text),
        "response_sha256": sha256_bytes(canonical_json_bytes(response_ids)),
        "finish_reason": None if finish_reason is None else str(finish_reason),
        "stop_reason": stop_reason,
        "sampling_config": asdict(sampling_config),
        "model_bundle_sha256": str(model_bundle_sha256),
        "tokenizer_sha256": str(tokenizer_sha256),
    }
    validate_generation_record(record)
    return record


def validate_generation_record(record: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "arm",
        "problem_uid",
        "dedup_group",
        "sample_index",
        "sample_seed",
        "prompt_token_ids",
        "prompt_token_count",
        "response_token_ids",
        "response_token_count",
        "response_text",
        "response_sha256",
        "sampling_config",
        "model_bundle_sha256",
        "tokenizer_sha256",
    }
    missing = sorted(required - set(record))
    if missing:
        raise ValueError(f"generation record missing fields: {missing}")
    if record["schema_version"] != GENERATION_SCHEMA_VERSION:
        raise ValueError(f"unsupported generation schema: {record['schema_version']}")
    if record["arm"] not in MODEL_PATHS:
        raise ValueError(f"invalid arm: {record['arm']}")
    for prefix in ("prompt", "response"):
        token_ids = record[f"{prefix}_token_ids"]
        if not isinstance(token_ids, list):
            raise ValueError(f"{prefix}_token_ids must be a list")
        if any(not isinstance(value, int) or value < 0 for value in token_ids):
            raise ValueError(f"{prefix}_token_ids must be non-negative integers")
        if int(record[f"{prefix}_token_count"]) != len(token_ids):
            raise ValueError(f"{prefix}_token_count mismatch")
    expected_response_hash = sha256_bytes(
        canonical_json_bytes(record["response_token_ids"])
    )
    if record["response_sha256"] != expected_response_hash:
        raise ValueError("response_sha256 mismatch")


def generate_records_pure(
    prompts: Iterable[Mapping[str, Any]],
    *,
    arm: str,
    sampling_config: SamplingConfig,
    model_bundle_sha256: str,
    tokenizer_sha256: str,
    sampler: Callable[[Sequence[int], int, SamplingConfig], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Local/fake-backend generation path used by focused tests."""

    records: list[dict[str, Any]] = []
    prompt_rows = sorted(prompts, key=lambda row: row["problem_uid"])
    for prompt in prompt_rows:
        validate_prompt_record(prompt)
        for sample_index in range(sampling_config.samples_per_prompt):
            seed = derive_sample_seed(
                sampling_config.base_seed, prompt["problem_uid"], sample_index
            )
            output = sampler(prompt["prompt_token_ids"], seed, sampling_config)
            records.append(
                make_generation_record(
                    arm=arm,
                    prompt=prompt,
                    sample_index=sample_index,
                    sample_seed=seed,
                    response_token_ids=output.get("token_ids", []),
                    response_text=str(output.get("text", "")),
                    finish_reason=output.get("finish_reason"),
                    stop_reason=output.get("stop_reason"),
                    sampling_config=sampling_config,
                    model_bundle_sha256=model_bundle_sha256,
                    tokenizer_sha256=tokenizer_sha256,
                )
            )
    return records


# ---------------------------------------------------------------------------
# Modal wrappers.  They deliberately call the pure helpers above.
# ---------------------------------------------------------------------------

try:
    import modal
except ImportError:  # pragma: no cover - pure helpers remain importable without Modal.
    modal = None


if modal is not None:
    _image = (
        modal.Image.from_registry(
            "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11"
        )
        .apt_install("build-essential", "git", "curl")
        .pip_install("wheel", "packaging", "ninja", "setuptools")
        .pip_install("torch==2.8.0")
        .pip_install("huggingface_hub==0.34.4", "hf_xet==1.1.5")
        .pip_install(
            "vllm==0.11.0",
            "transformers==4.57.1",
            "flash-attn==2.8.3",
            extra_options="--no-build-isolation",
        )
        .pip_install("pyarrow==17.0.0", "numpy>=1.26")
        .env({"VLLM_ATTENTION_BACKEND": "FLASH_ATTN"})
    )
    app = modal.App("math-transfer-generate", image=_image)
    checkpoint_volume = modal.Volume.from_name(
        CHECKPOINT_VOLUME_NAME, create_if_missing=True, version=2
    )
    cache_volume = modal.Volume.from_name(CACHE_VOLUME_NAME, create_if_missing=True)

    @app.function(
        cpu=8,
        memory=32 * 1024,
        timeout=60 * 30,
        volumes={CHECKPOINT_MOUNT: checkpoint_volume, CACHE_MOUNT: cache_volume},
    )
    def freeze_prompt_manifest_remote(
        output_path: str,
        dataset_path: str = DEFAULT_DATASET_PATH,
        tokenizer_path: str = D0_MODEL_PATH,
        split: str = "train",
        max_prompt_tokens: int = DEFAULT_MAX_PROMPT_TOKENS,
        difficulty_model: str = "DeepSeek-R1-Distill-Qwen-1.5B",
    ) -> dict[str, Any]:
        """Freeze the exact eligible training prompt pool to canonical JSONL."""

        import pyarrow.parquet as pq
        from transformers import AutoTokenizer

        checkpoint_volume.reload()
        os.environ["HF_HOME"] = f"{CACHE_MOUNT}/hf"
        dataset_sha256 = sha256_file(dataset_path)
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=True, use_fast=True
        )

        def tokenize_prompt(messages: Sequence[Mapping[str, Any]]) -> Sequence[int]:
            return tokenizer.apply_chat_template(
                list(messages), tokenize=True, add_generation_prompt=True
            )

        table = pq.read_table(dataset_path)
        prompt_rows, stats = freeze_prompt_rows(
            table.to_pylist(),
            dataset_sha256=dataset_sha256,
            split=split,
            tokenize_prompt=tokenize_prompt,
            max_prompt_tokens=max_prompt_tokens,
            difficulty_model=difficulty_model,
        )
        artifact = write_jsonl_atomic(output_path, prompt_rows)
        tokenizer_file = Path(tokenizer_path) / "tokenizer.json"
        template_file = Path(tokenizer_path) / "chat_template.jinja"
        manifest = write_manifest_atomic(
            str(output_path) + ".manifest.json",
            {
                "artifact_kind": "frozen_prompt_pool",
                "prompt_schema_version": PROMPT_SCHEMA_VERSION,
                "dataset_path": dataset_path,
                "dataset_sha256": dataset_sha256,
                "dataset_split": split,
                "tokenizer_path": tokenizer_path,
                "tokenizer_sha256": sha256_file(tokenizer_file),
                "chat_template_sha256": sha256_file(template_file),
                "max_prompt_tokens": max_prompt_tokens,
                "difficulty_model": difficulty_model,
                "stats": stats,
                "artifact": artifact,
            },
        )
        checkpoint_volume.commit()
        return manifest

    @app.function(
        cpu=8,
        memory=32 * 1024,
        timeout=60 * 30,
        volumes={CHECKPOINT_MOUNT: checkpoint_volume, CACHE_MOUNT: cache_volume},
    )
    def freeze_split_prompt_manifests_remote(
        output_dir: str,
        generation_count: int = 18_000,
        dev_count: int = 2_000,
        split_seed: int = 20_260_710,
        dataset_path: str = DEFAULT_DATASET_PATH,
        tokenizer_path: str = D0_MODEL_PATH,
        split: str = "train",
        max_prompt_tokens: int = DEFAULT_MAX_PROMPT_TOKENS,
        difficulty_model: str = "DeepSeek-R1-Distill-Qwen-1.5B",
    ) -> dict[str, Any]:
        """Freeze one semantic representative per group into three prompt pools."""

        import pyarrow.parquet as pq
        from transformers import AutoTokenizer

        checkpoint_volume.reload()
        os.environ["HF_HOME"] = f"{CACHE_MOUNT}/hf"
        dataset_sha256 = sha256_file(dataset_path)
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=True, use_fast=True
        )

        def tokenize_prompt(messages: Sequence[Mapping[str, Any]]) -> Sequence[int]:
            return tokenizer.apply_chat_template(
                list(messages), tokenize=True, add_generation_prompt=True
            )

        prompt_rows, freeze_stats = freeze_prompt_rows(
            pq.read_table(dataset_path).to_pylist(),
            dataset_sha256=dataset_sha256,
            split=split,
            tokenize_prompt=tokenize_prompt,
            max_prompt_tokens=max_prompt_tokens,
            difficulty_model=difficulty_model,
        )
        pools, split_summary = split_semantic_prompt_pool(
            prompt_rows,
            split_seed=split_seed,
            generation_groups=generation_count,
            dev_groups=dev_count,
        )
        tokenizer_file = Path(tokenizer_path) / "tokenizer.json"
        template_file = Path(tokenizer_path) / "chat_template.jinja"
        manifest = write_split_prompt_artifacts(
            output_dir,
            pools,
            split_summary=split_summary,
            provenance={
                "dataset_path": dataset_path,
                "dataset_sha256": dataset_sha256,
                "dataset_split": split,
                "tokenizer_path": tokenizer_path,
                "tokenizer_sha256": sha256_file(tokenizer_file),
                "chat_template_sha256": sha256_file(template_file),
                "max_prompt_tokens": max_prompt_tokens,
                "difficulty_model": difficulty_model,
                "freeze_stats": freeze_stats,
            },
        )
        checkpoint_volume.commit()
        return manifest

    @app.function(
        cpu=4,
        memory=8 * 1024,
        timeout=60 * 10,
        volumes={CHECKPOINT_MOUNT: checkpoint_volume},
    )
    def select_smoke_prompt_manifest_remote(
        input_path: str,
        output_path: str,
        count: int = 32,
        selection_seed: int = 20_260_710,
    ) -> dict[str, Any]:
        """Select and persist a deterministic stratified smoke prompt manifest."""

        checkpoint_volume.reload()
        source_rows = list(read_jsonl(input_path))
        source_sha256 = sha256_file(input_path)
        selected, selection_summary = select_stratified_smoke_prompts(
            source_rows, count=count, selection_seed=selection_seed
        )
        artifact = write_jsonl_atomic(output_path, selected)
        manifest = write_manifest_atomic(
            str(output_path) + ".manifest.json",
            {
                "artifact_kind": "stratified_smoke_prompt_pool",
                "prompt_schema_version": PROMPT_SCHEMA_VERSION,
                "source_path": Path(input_path).name,
                "source_sha256": source_sha256,
                "source_count": len(source_rows),
                "selection_seed": int(selection_seed),
                "requested_count": int(count),
                "available_bin_counts": selection_summary["available_bin_counts"],
                "selected_bin_counts": selection_summary["selected_bin_counts"],
                "selection_summary": selection_summary,
                "artifact": artifact,
            },
        )
        checkpoint_volume.commit()
        return manifest

    @app.function(
        gpu="H200:1",
        cpu=8,
        memory=100 * 1024,
        timeout=60 * 60 * 12,
        volumes={CHECKPOINT_MOUNT: checkpoint_volume, CACHE_MOUNT: cache_volume},
    )
    def generate_fixed_policy_remote(
        arm: str,
        prompt_manifest_path: str,
        output_dir: str,
        samples_per_prompt: int = 16,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_new_tokens: int = 3584,
        base_seed: int = 0,
        request_chunk_size: int = 1024,
        gpu_memory_utilization: float = 0.85,
    ) -> dict[str, Any]:
        """Generate one arm from a frozen policy and exact prompt token IDs."""

        from vllm import LLM, SamplingParams

        arm = arm.upper()
        if arm not in MODEL_PATHS:
            raise ValueError(f"arm must be one of {sorted(MODEL_PATHS)}")
        if request_chunk_size <= 0:
            raise ValueError("request_chunk_size must be positive")

        checkpoint_volume.reload()
        os.environ["HF_HOME"] = f"{CACHE_MOUNT}/hf"
        prompts = list(read_jsonl(prompt_manifest_path))
        for prompt in prompts:
            validate_prompt_record(prompt)
        prompts.sort(key=lambda row: row["problem_uid"])
        prompt_manifest_sha256 = sha256_file(prompt_manifest_path)

        config = SamplingConfig(
            samples_per_prompt=samples_per_prompt,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            base_seed=base_seed,
        )
        model_path = MODEL_PATHS[arm]
        model_bundle = hash_model_bundle(model_path)
        tokenizer_sha256 = sha256_file(Path(model_path) / "tokenizer.json")
        llm = LLM(
            model=model_path,
            dtype="bfloat16",
            trust_remote_code=True,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=DEFAULT_MAX_PROMPT_TOKENS + max_new_tokens,
        )

        requests: list[tuple[Mapping[str, Any], int, int]] = []
        for prompt in prompts:
            for sample_index in range(samples_per_prompt):
                requests.append(
                    (
                        prompt,
                        sample_index,
                        derive_sample_seed(base_seed, prompt["problem_uid"], sample_index),
                    )
                )

        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        artifacts: list[dict[str, Any]] = []
        total_rows = 0
        for chunk_index, start in enumerate(range(0, len(requests), request_chunk_size)):
            chunk = requests[start : start + request_chunk_size]
            token_prompts = [
                {"prompt_token_ids": list(prompt["prompt_token_ids"])}
                for prompt, _, _ in chunk
            ]
            sampling_params = [
                SamplingParams(
                    n=1,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_new_tokens,
                    seed=seed,
                )
                for _, _, seed in chunk
            ]
            outputs = llm.generate(token_prompts, sampling_params, use_tqdm=True)
            if len(outputs) != len(chunk):
                raise RuntimeError(
                    f"vLLM returned {len(outputs)} outputs for {len(chunk)} requests"
                )
            rows: list[dict[str, Any]] = []
            for (prompt, sample_index, seed), request_output in zip(chunk, outputs):
                returned_prompt_ids = list(request_output.prompt_token_ids)
                if returned_prompt_ids != list(prompt["prompt_token_ids"]):
                    raise RuntimeError(
                        f"prompt token drift for {prompt['problem_uid']} sample {sample_index}"
                    )
                if len(request_output.outputs) != 1:
                    raise RuntimeError("expected exactly one completion per seeded request")
                completion = request_output.outputs[0]
                rows.append(
                    make_generation_record(
                        arm=arm,
                        prompt=prompt,
                        sample_index=sample_index,
                        sample_seed=seed,
                        response_token_ids=list(completion.token_ids),
                        response_text=completion.text,
                        finish_reason=getattr(completion, "finish_reason", None),
                        stop_reason=getattr(completion, "stop_reason", None),
                        sampling_config=config,
                        model_bundle_sha256=model_bundle["bundle_sha256"],
                        tokenizer_sha256=tokenizer_sha256,
                    )
                )
            part_path = destination / f"part-{chunk_index:05d}.jsonl"
            artifacts.append(write_jsonl_atomic(part_path, rows))
            total_rows += len(rows)
            checkpoint_volume.commit()

        manifest = write_manifest_atomic(
            destination / "manifest.json",
            {
                "artifact_kind": "fixed_policy_generations",
                "generation_schema_version": GENERATION_SCHEMA_VERSION,
                "arm": arm,
                "model_path": model_path,
                "model_bundle": model_bundle,
                "tokenizer_sha256": tokenizer_sha256,
                "prompt_manifest_path": prompt_manifest_path,
                "prompt_manifest_sha256": prompt_manifest_sha256,
                "sampling_config": asdict(config),
                "prompt_count": len(prompts),
                "generation_count": total_rows,
                "parts": artifacts,
            },
        )
        checkpoint_volume.commit()
        return manifest
