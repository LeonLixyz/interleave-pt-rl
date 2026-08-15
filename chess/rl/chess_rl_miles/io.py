from __future__ import annotations

import json
import os
import hashlib
from collections import Counter
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import torch
from miles.utils.metric_utils import compute_rollout_step


STRICT_EXACT_ONCE_DATA_SOURCE = (
    "chess_rl_miles.gate_data_source.StrictEpochRolloutDataSource"
)
STRICT_EXHAUSTIVE_DATA_SOURCE = (
    "chess_rl_miles.exhaustive_data_source."
    "StrictExhaustiveRolloutDataSource"
)
STRICT_EXACT_ONCE_DATA_SOURCES = frozenset(
    {
        STRICT_EXACT_ONCE_DATA_SOURCE,
        STRICT_EXHAUSTIVE_DATA_SOURCE,
    }
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _artifact_root(args) -> Path | None:
    env_root = os.environ.get("CHESS_RL_MILES_ARTIFACT_ROOT")
    if env_root:
        return Path(env_root)
    dump_details = getattr(args, "dump_details", None)
    if not dump_details:
        return None
    return Path(dump_details).parent


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(value), f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def _reward_value(args, sample) -> Any:
    if sample.reward is None:
        return None
    if isinstance(sample.reward, dict):
        reward_key = getattr(args, "reward_key", None) or "score"
        return sample.reward.get(reward_key)
    return sample.reward


def _is_positive_score(value: Any) -> bool:
    """Return whether ``value`` is the exact binary success reward.

    Rollout rewards are normally Python/NumPy floating point values.  Bool is
    deliberately excluded even though ``True == 1`` in Python: the replay
    corpus contract is a numeric ``score == 1`` emitted by the reward model.
    """
    value = _jsonable(value)
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and float(value) == 1.0
    )


def _token_ids_sha256(prompt_token_ids: list[int], response_token_ids: list[int]) -> str:
    payload = json.dumps(
        [prompt_token_ids, response_token_ids],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _positive_token_artifact(sample) -> dict[str, Any]:
    """Build a lossless response-mask artifact.

    Ordinary runs persist it only for positive replay rows. The precision
    resume gate persists it for every row so prompt/BOS and environment-token
    masking can be independently checked before production is allowed.
    """
    tokens = [int(token_id) for token_id in (sample.tokens or [])]
    response_length = int(sample.response_length)
    if response_length <= 0:
        raise ValueError("positive rollout has no response tokens")
    if response_length > len(tokens):
        raise ValueError(
            "positive rollout response_length exceeds token count: "
            f"{response_length} > {len(tokens)}"
        )

    loss_mask = sample.loss_mask
    if loss_mask is None:
        raise ValueError("positive rollout is missing its response loss mask")
    response_loss_mask = [int(value) for value in loss_mask]
    if len(response_loss_mask) != response_length:
        raise ValueError(
            "positive rollout loss-mask length does not match response length: "
            f"{len(response_loss_mask)} != {response_length}"
        )
    if any(value not in (0, 1) for value in response_loss_mask):
        raise ValueError("positive rollout response loss mask must be binary")

    prompt_token_ids = tokens[:-response_length]
    response_token_ids = tokens[-response_length:]
    if not prompt_token_ids:
        raise ValueError("positive rollout has no prompt tokens")
    return {
        "token_artifact_schema": 1,
        "prompt_token_ids": prompt_token_ids,
        "response_token_ids": response_token_ids,
        "response_loss_mask": response_loss_mask,
        "token_ids_sha256": _token_ids_sha256(
            prompt_token_ids,
            response_token_ids,
        ),
    }


def _flat_rollout_row(
    args,
    sample,
    *,
    rollout_id: int,
    split: str,
    index: int,
    dataset: str | None = None,
) -> dict[str, Any]:
    step = compute_rollout_step(args, rollout_id)
    reward = sample.reward if isinstance(sample.reward, dict) else {}
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}

    row: dict[str, Any] = {
        "input": sample.prompt,
        "output": sample.response,
        "score": _reward_value(args, sample),
        "step": step,
        "rollout_id": rollout_id,
        "split": split,
        # ``index`` is retained for compatibility with existing consumers.
        # The unambiguous names below preserve both the JSONL row and Miles
        # sample identities.
        "index": index,
        "artifact_index": index,
        "group_index": sample.group_index,
        "sample_index": sample.index,
    }
    if dataset is not None:
        row["dataset"] = dataset
    row["reward"] = reward
    row.update({str(k): v for k, v in reward.items()})

    # Preserve the complete source metadata, while retaining the historical
    # flattened convenience fields used by dashboards.
    row["metadata"] = metadata
    row["train_metadata"] = sample.train_metadata
    row["label"] = sample.label
    row["session_id"] = sample.session_id
    row["weight_versions"] = sample.weight_versions
    for key, value in metadata.items():
        if key not in row:
            row[str(key)] = value

    row["status"] = sample.status.value if hasattr(sample.status, "value") else str(sample.status)
    row["response_length"] = sample.response_length
    row["effective_response_length"] = sample.effective_response_length
    if _is_positive_score(row["score"]) or os.environ.get(
        "CHESS_RL_MILES_PRECISION_GATE_LEG"
    ) in {"1", "2"}:
        row.update(_positive_token_artifact(sample))
    return row


def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    root = _artifact_root(args)
    strict_exact_once = (
        getattr(args, "data_source_path", None)
        in STRICT_EXACT_ONCE_DATA_SOURCES
    )
    if root is None:
        if strict_exact_once:
            raise RuntimeError(
                "strict exact-once rollout cannot suppress default outcome "
                "metrics without an authenticated artifact root"
            )
        return False

    rows = [
        _flat_rollout_row(args, sample, rollout_id=rollout_id, split="training", index=i)
        for i, sample in enumerate(samples)
    ]
    _write_jsonl(root / "rollouts" / "training" / f"rollout_{rollout_id}.jsonl", rows)
    # Miles interprets True as "custom logger handled everything."  Suppress
    # its reward/zero-variance metric logger only for the blinded fixed gate;
    # ordinary RL runs retain their existing metrics and tracking behavior.
    return strict_exact_once


def log_all_attempts_positive(
    rollout_id: int,
    args,
    sample_groups,
) -> int:
    """Persist successful completed attempts before dynamic filtering.

    The standard Miles rollout logger receives only groups accepted for the RL
    update.  Dynamic nonzero-variance filtering drops all-one groups, so Exp 4
    needs this separate stream built from ``all_samples`` inside the rollout
    function.  Only positive completed rows are written to control artifact
    size; each includes the lossless token artifact added by
    :func:`_flat_rollout_row`.
    """
    root = _artifact_root(args)
    if root is None:
        return 0

    output_root = root / "rollouts" / "all_attempts_positive"
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / f"rollout_{rollout_id}.jsonl"
    temporary_output_path = output_path.with_suffix(".jsonl.tmp")
    attempt_index = 0
    completed_samples = 0
    positive_completed_samples = 0
    group_success_counts: Counter[int] = Counter()
    # Stream rows to disk. A difficult-policy dynamic-filter rollout can
    # oversample far beyond the accepted 2,048 trajectories, and retaining a
    # second in-memory copy of every token array would be unnecessarily large.
    with temporary_output_path.open("w", encoding="utf-8") as handle:
        for group in sample_groups:
            group_successes = 0
            for sample in group:
                score = _reward_value(args, sample)
                status = (
                    sample.status.value
                    if hasattr(sample.status, "value")
                    else str(sample.status)
                )
                if status == "completed":
                    completed_samples += 1
                if _is_positive_score(score) and status == "completed":
                    group_successes += 1
                    positive_completed_samples += 1
                    row = _flat_rollout_row(
                        args,
                        sample,
                        rollout_id=rollout_id,
                        split="all_attempts_positive",
                        index=attempt_index,
                    )
                    row["sampling_scope"] = (
                        "all_completed_attempts_before_dynamic_filter"
                    )
                    handle.write(
                        json.dumps(
                            _jsonable(row),
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                attempt_index += 1
            group_success_counts[group_successes] += 1
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_output_path, output_path)

    _write_json(
        output_root / f"rollout_{rollout_id}.summary.json",
        {
            "schema_version": 1,
            "sampling_scope": "all_completed_attempts_before_dynamic_filter",
            "rollout_id": rollout_id,
            "step": compute_rollout_step(args, rollout_id),
            "attempted_groups": len(sample_groups),
            "attempted_samples": attempt_index,
            "completed_samples": completed_samples,
            "positive_completed_samples": positive_completed_samples,
            "group_success_count_histogram": {
                str(success_count): group_count
                for success_count, group_count in sorted(
                    group_success_counts.items()
                )
            },
        },
    )
    return positive_completed_samples


def log_eval_rollout_data(rollout_id, args, data, extra_metrics) -> bool:
    root = _artifact_root(args)
    if root is None:
        return False

    rows = []
    for dataset_name, info in data.items():
        for i, sample in enumerate(info.get("samples") or []):
            rows.append(
                _flat_rollout_row(
                    args,
                    sample,
                    rollout_id=rollout_id,
                    split="validation",
                    index=i,
                    dataset=dataset_name,
                )
            )
    if rows:
        _write_jsonl(root / "rollouts" / "validation" / f"eval_{rollout_id}.jsonl", rows)
    return False
