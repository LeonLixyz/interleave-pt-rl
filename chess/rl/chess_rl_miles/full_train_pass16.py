"""Pure preparation and validation helpers for exhaustive P1 pass@16."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator


SOURCE_ROWS = 53_225
SOURCE_SHA256 = (
    "bcf131d88cd9916f6ec12a12f8e85901a05e0dabc5d5a1fdebd8ea0b0ae86c30"
)
RL_PROMPT_CAP = 512
EXHAUSTIVE_PROMPT_CAP = 1_024
SAMPLES_PER_PROMPT = 16
EXPECTED_TRAJECTORIES = SOURCE_ROWS * SAMPLES_PER_PROMPT
SHARD_COUNT = 16
SHARD_LENGTHS = (3_325,) * 15 + (3_350,)
ROLLOUTS_PER_SHARD = 25
ROLLOUT_BATCH_SIZES = (133,) * 15 + (134,)
BASE_SEED = 4_242_000


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def shard_ranges() -> tuple[tuple[int, int], ...]:
    ranges: list[tuple[int, int]] = []
    start = 0
    for length in SHARD_LENGTHS:
        ranges.append((start, start + length))
        start += length
    if start != SOURCE_ROWS:
        raise AssertionError(f"invalid shard total: {start}")
    return tuple(ranges)


def checkpoint_fingerprint(checkpoint: str | Path) -> str:
    """Match Eval/interleave_endpoint_eval.py::checkpoint_fingerprint."""

    root = Path(checkpoint).resolve(strict=True)
    required = (
        root / "config.json",
        root / "interleaved_training_state.json",
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    weights = sorted(root.glob("model*.safetensors"))
    if not weights:
        raise FileNotFoundError(f"missing safetensors weights under {root}")
    files = list(weights)
    for name in (
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "interleaved_training_state.json",
    ):
        candidate = root / name
        if candidate.is_file():
            files.append(candidate)
    for pattern in (
        "tokenizer*",
        "vocab*",
        "merges*",
        "special_tokens_map.json",
        "added_tokens.json",
        "sentencepiece*",
        "spiece*",
    ):
        files.extend(path for path in root.glob(pattern) if path.is_file())
    files = sorted(set(files), key=lambda path: path.relative_to(root).as_posix())
    if not any(path.name.startswith("tokenizer") for path in files):
        raise FileNotFoundError(f"missing tokenizer under {root}")
    if not any(path.name.startswith("vocab") for path in files):
        raise FileNotFoundError(f"missing vocabulary under {root}")

    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 << 20), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _original_row_fingerprint(row: dict[str, Any]) -> str:
    return canonical_sha256(
        {
            "data_source": row.get("data_source"),
            "prompt": row.get("prompt"),
            "ability": row.get("ability"),
            "reward_model": row.get("reward_model"),
            "extra_info": row.get("extra_info"),
            "difficulty": row.get("difficulty"),
        }
    )


def prepare_shards(
    *,
    source_path: str | Path,
    tokenizer_path: str | Path,
    output_root: str | Path,
    tokenizer_revision: str,
) -> dict[str, Any]:
    """Create immutable, globally indexed parquet shards and a long-prompt canary."""

    import pyarrow as pa
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    source = Path(source_path).resolve(strict=True)
    output = Path(output_root).resolve()
    if sha256_file(source) != SOURCE_SHA256:
        raise ValueError("source parquet SHA256 drifted")
    table = pq.read_table(source)
    if table.num_rows != SOURCE_ROWS:
        raise ValueError(f"source row count drifted: {table.num_rows}")

    tokenizer = AutoTokenizer.from_pretrained(
        str(Path(tokenizer_path).resolve(strict=True)),
        trust_remote_code=True,
    )
    original_rows = table.to_pylist()
    prompts = [str(row["prompt"]) for row in original_rows]
    tokenized = tokenizer(
        prompts,
        add_special_tokens=False,
        padding=False,
        truncation=False,
    )["input_ids"]
    prompt_lengths = [len(token_ids) for token_ids in tokenized]
    if max(prompt_lengths) > EXHAUSTIVE_PROMPT_CAP:
        raise ValueError("exhaustive prompt cap no longer covers the source")

    prepared_rows: list[dict[str, Any]] = []
    fingerprints: list[str] = []
    for source_row_index, (row, prompt_tokens) in enumerate(
        zip(original_rows, prompt_lengths, strict=True)
    ):
        fingerprint = _original_row_fingerprint(row)
        fingerprints.append(fingerprint)
        metadata = dict(row.get("extra_info") or {})
        metadata.update(
            {
                "source_row_index": source_row_index,
                "source_row_fingerprint": fingerprint,
                "prompt_token_length": prompt_tokens,
                "rl_prompt_cap_eligible": prompt_tokens <= RL_PROMPT_CAP,
                "source_data_source": row.get("data_source"),
                "source_difficulty": row.get("difficulty"),
            }
        )
        prepared = dict(row)
        prepared["extra_info"] = metadata
        prepared_rows.append(prepared)
    if len(set(fingerprints)) != SOURCE_ROWS:
        raise ValueError("source contains duplicate content fingerprints")

    output.mkdir(parents=True, exist_ok=False)
    shards_root = output / "shards"
    shards_root.mkdir()
    shard_records: list[dict[str, Any]] = []
    for shard_id, ((start, stop), batch_size) in enumerate(
        zip(shard_ranges(), ROLLOUT_BATCH_SIZES, strict=True)
    ):
        shard_rows = prepared_rows[start:stop]
        for row in shard_rows:
            row["extra_info"]["full_eval_shard_id"] = shard_id
        shard_path = shards_root / f"shard_{shard_id:02d}.parquet"
        pq.write_table(
            pa.Table.from_pylist(shard_rows),
            shard_path,
            compression="zstd",
            use_dictionary=True,
        )
        shard_records.append(
            {
                "shard_id": shard_id,
                "relative_path": shard_path.relative_to(output).as_posix(),
                "source_row_start": start,
                "source_row_stop": stop,
                "rows": stop - start,
                "rollout_batch_size": batch_size,
                "num_rollout": ROLLOUTS_PER_SHARD,
                "trajectories": (stop - start) * SAMPLES_PER_PROMPT,
                "bytes": shard_path.stat().st_size,
                "sha256": sha256_file(shard_path),
            }
        )

    longest_index = max(range(SOURCE_ROWS), key=prompt_lengths.__getitem__)
    canary_indices = (0, longest_index)
    canary_path = output / "canary.parquet"
    pq.write_table(
        pa.Table.from_pylist([prepared_rows[index] for index in canary_indices]),
        canary_path,
        compression="zstd",
        use_dictionary=True,
    )

    long_rows = [
        {
            "source_row_index": index,
            "prompt_token_length": length,
            "source_row_fingerprint": fingerprints[index],
        }
        for index, length in enumerate(prompt_lengths)
        if length > RL_PROMPT_CAP
    ]
    manifest: dict[str, Any] = {
        "schema": "p1-full-rl-train-pass16-prepared-v1",
        "source": {
            "path": str(source),
            "rows": SOURCE_ROWS,
            "bytes": source.stat().st_size,
            "sha256": SOURCE_SHA256,
        },
        "tokenizer": {
            "path": str(Path(tokenizer_path).resolve(strict=True)),
            "revision": tokenizer_revision,
            "add_special_tokens": False,
        },
        "contract": {
            "samples_per_prompt": SAMPLES_PER_PROMPT,
            "expected_trajectories": EXPECTED_TRAJECTORIES,
            "rl_prompt_cap": RL_PROMPT_CAP,
            "exhaustive_prompt_cap": EXHAUSTIVE_PROMPT_CAP,
            "response_cap": 2_560,
            "context_cap": 3_072,
            "temperature": 1.0,
            "top_p": 1.0,
            "base_seed": BASE_SEED,
            "shard_count": SHARD_COUNT,
            "rollouts_per_shard": ROLLOUTS_PER_SHARD,
        },
        "prompt_lengths": {
            "min": min(prompt_lengths),
            "max": max(prompt_lengths),
            "rl_eligible_rows": sum(
                length <= RL_PROMPT_CAP for length in prompt_lengths
            ),
            "supplemental_long_rows": len(long_rows),
            "long_rows": long_rows,
            "long_rows_sha256": canonical_sha256(
                [
                    [row["source_row_index"], row["prompt_token_length"]]
                    for row in long_rows
                ]
            ),
        },
        "row_fingerprints_sha256": canonical_sha256(fingerprints),
        "canary": {
            "relative_path": canary_path.relative_to(output).as_posix(),
            "source_row_indices": list(canary_indices),
            "rows": len(canary_indices),
            "bytes": canary_path.stat().st_size,
            "sha256": sha256_file(canary_path),
        },
        "shards": shard_records,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    manifest_path = output / "prepared_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def iter_jsonl(paths: Iterable[str | Path]) -> Iterator[dict[str, Any]]:
    for path_value in paths:
        path = Path(path_value)
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if line.strip():
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"invalid JSONL at {path}:{line_number}"
                        ) from exc
                    if not isinstance(row, dict):
                        raise ValueError(
                            f"non-object JSONL at {path}:{line_number}"
                        )
                    yield row


def validate_rollout_rows(
    rows: Iterable[dict[str, Any]],
    *,
    expected_source_indices: set[int],
) -> dict[str, Any]:
    """Validate one shard and return outcome-independent and outcome metrics."""

    slots: dict[int, set[int]] = defaultdict(set)
    success_counts: Counter[int] = Counter()
    status_counts: Counter[str] = Counter()
    row_count = 0
    positive_count = 0
    seen_sample_indices: set[int] = set()
    for row in rows:
        row_count += 1
        metadata = row.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError("raw rollout row lacks metadata")
        source_index = metadata.get("source_row_index")
        slot = metadata.get("pass_at_16_sample_slot")
        sample_index = metadata.get("pass_at_16_sample_index")
        if (
            not isinstance(source_index, int)
            or isinstance(source_index, bool)
            or source_index not in expected_source_indices
        ):
            raise ValueError("raw rollout source_row_index is outside shard")
        if (
            not isinstance(slot, int)
            or isinstance(slot, bool)
            or not 0 <= slot < SAMPLES_PER_PROMPT
        ):
            raise ValueError("raw rollout sibling slot is invalid")
        expected_sample_index = source_index * SAMPLES_PER_PROMPT + slot
        if (
            sample_index != expected_sample_index
            or row.get("group_index") != source_index
            or row.get("sample_index") != expected_sample_index
        ):
            raise ValueError("raw rollout global identity drifted")
        if expected_sample_index in seen_sample_indices:
            raise ValueError("duplicate raw rollout sample identity")
        seen_sample_indices.add(expected_sample_index)
        slots[source_index].add(slot)

        score = row.get("score")
        if isinstance(score, bool) or score not in (0, 0.0, 1, 1.0):
            raise ValueError(f"non-binary rollout score: {score!r}")
        reward = row.get("reward")
        if not isinstance(reward, dict) or reward.get("score") != score:
            raise ValueError("flattened score disagrees with reward.score")
        positive_count += int(float(score) == 1.0)
        success_counts[source_index] += int(float(score) == 1.0)
        status = str(row.get("status"))
        if status not in {"completed", "truncated"}:
            raise ValueError(f"nonterminal rollout status: {status}")
        status_counts[status] += 1

    expected_rows = len(expected_source_indices) * SAMPLES_PER_PROMPT
    if row_count != expected_rows:
        raise ValueError(f"raw rollout row mismatch: {row_count}/{expected_rows}")
    if set(slots) != expected_source_indices:
        raise ValueError("raw rollout source coverage mismatch")
    full_slots = set(range(SAMPLES_PER_PROMPT))
    if any(value != full_slots for value in slots.values()):
        raise ValueError("raw rollout does not contain exactly slots 0..15")
    histogram = Counter(success_counts.get(index, 0) for index in expected_source_indices)
    return {
        "rows": row_count,
        "positive_trajectories": positive_count,
        "solved_prompts": sum(count > 0 for count in success_counts.values()),
        "status_counts": dict(sorted(status_counts.items())),
        "success_count_histogram": {
            str(count): histogram.get(count, 0)
            for count in range(SAMPLES_PER_PROMPT + 1)
        },
    }


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if trials <= 0:
        return (math.nan, math.nan)
    proportion = successes / trials
    denominator = 1 + z * z / trials
    center = (proportion + z * z / (2 * trials)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / trials
            + z * z / (4 * trials * trials)
        )
        / denominator
    )
    return center - radius, center + radius


def atomic_json(path: str | Path, value: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
