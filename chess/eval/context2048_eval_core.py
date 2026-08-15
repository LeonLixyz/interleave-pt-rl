"""Pure helpers shared by native context-2048 chess evaluators.

This module intentionally has no Modal, vLLM, pandas, or torch dependency so
the evaluation contract can be unit-tested without building a GPU image.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    """Publish one JSON object with a same-directory atomic rename."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(
            json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def frame_prompt_ids(
    unframed_ids: list[int], *, bos_id: int, prompt_cap: int
) -> list[int] | None:
    """Prepend exactly one BOS and apply a cap that includes that BOS."""

    if prompt_cap <= 0:
        raise ValueError(f"prompt_cap must be positive, got {prompt_cap}")
    raw = [int(token_id) for token_id in unframed_ids]
    if int(bos_id) in raw:
        raise RuntimeError(
            "source prompt unexpectedly contains BOS while tokenized with "
            "add_special_tokens=False"
        )
    framed = [int(bos_id), *raw]
    if len(framed) > prompt_cap:
        return None
    if framed[0] != int(bos_id) or framed.count(int(bos_id)) != 1:
        raise AssertionError("exactly-one-leading-BOS construction failed")
    return framed


def to_python_value(value: Any) -> Any:
    """Convert pandas/PyArrow containers to the production Python shapes."""

    if isinstance(value, dict):
        return {key: to_python_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_python_value(item) for item in value]
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return to_python_value(tolist())
    return value


def unbiased_pass_at_k(
    histogram: Mapping[int, int], *, n: int, k: int
) -> float:
    """Compute the standard unbiased pass@k estimator from win counts."""

    if n <= 0 or not 1 <= k <= n:
        raise ValueError(f"require n > 0 and 1 <= k <= n, got n={n}, k={k}")
    prompts = sum(int(count) for count in histogram.values())
    if prompts <= 0:
        raise ValueError("empty pass@k histogram")
    total = 0.0
    for raw_wins, raw_count in histogram.items():
        wins = int(raw_wins)
        count = int(raw_count)
        if not 0 <= wins <= n or count < 0:
            raise ValueError(
                f"invalid histogram entry wins={wins}, count={count}, n={n}"
            )
        if wins == 0:
            value = 0.0
        elif n - wins < k:
            value = 1.0
        else:
            value = 1.0 - math.comb(n - wins, k) / math.comb(n, k)
        total += value * count
    return total / prompts


def pass_at_k_curve(
    histogram: Mapping[int, int], *, n: int
) -> dict[str, float]:
    return {
        str(k): unbiased_pass_at_k(histogram, n=n, k=k)
        for k in range(1, n + 1)
    }


def deterministic_sample_seed(
    *,
    base_seed: int,
    dataset_key: str,
    row_index: int,
    sample_slot: int,
    generation_round: int,
) -> int:
    """Return a stable per-request seed shared across model checkpoints."""

    if min(row_index, sample_slot, generation_round) < 0:
        raise ValueError("row, sample slot, and generation round must be nonnegative")
    payload = {
        "base_seed": int(base_seed),
        "dataset_key": str(dataset_key),
        "generation_round": int(generation_round),
        "row_index": int(row_index),
        "sample_slot": int(sample_slot),
    }
    # vLLM accepts nonnegative signed 32-bit seeds on every supported backend.
    return int(canonical_sha256(payload)[:8], 16) & 0x7FFF_FFFF


def summarize_histogram(
    histogram: Mapping[int, int], *, n: int
) -> dict[str, Any]:
    prompts = sum(int(count) for count in histogram.values())
    if prompts <= 0:
        raise ValueError("empty result histogram")
    normalized = {wins: int(histogram.get(wins, 0)) for wins in range(n + 1)}
    return {
        "evaluated_prompts": prompts,
        "pass_at_k": pass_at_k_curve(normalized, n=n),
        "all_zero_prompts": normalized[0],
        "all_zero_percentage": normalized[0] / prompts,
        "all_one_prompts": normalized[n],
        "all_one_percentage": normalized[n] / prompts,
        "mixed_outcome_prompts": sum(normalized[wins] for wins in range(1, n)),
        "wins_histogram": {
            str(wins): normalized[wins] for wins in range(n + 1)
        },
    }
