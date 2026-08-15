"""Stable Modal dashboard for the four context-2048 checkpoint evaluations.

Deploy with:
    modal deploy chess/eval/modal_context2048_eval_dashboard.py

The page reads authenticated summary markers and durable checkpoints from
Modal Volumes, plus read-only RL histories from the intended W&B project. It
does not scan the large generation artifacts.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal


APP_NAME = "chess-context2048-eval-dashboard"
VERSION = "context2048-fp32-master-v13-pass16-native2048-bos-v1-20260814"
EVAL_VOLUME_NAME = "chess-rl-eval-results-r6"
EVAL_MOUNT = Path("/results")
RESULTS_ROOT = EVAL_MOUNT / VERSION
PRETRAIN_VOLUME_NAME = "rl-reasoning-checkpoints"
PRETRAIN_MOUNT = Path("/pretraining")
PRETRAIN_ROOT = (
    PRETRAIN_MOUNT / "context2048_vocab_mixing_fp32_master_v13_20260813"
)
RL_CHECKPOINT_VOLUME_NAME = "chess-rl-miles-checkpoints"
RL_CHECKPOINT_MOUNT = Path("/rl-checkpoints")
RL_CHECKPOINT_ROOT = (
    RL_CHECKPOINT_MOUNT / "chess-rl-miles-interleave-fp32-master-v3"
)
VALIDATION_VERSION = "context2048-fp32-master-v13-heldout-pt-v1-20260814"
VALIDATION_ROOT = EVAL_MOUNT / VALIDATION_VERSION
VALIDATION_TARGET_TOKENS = 4_096 * 2_048
WANDB_ROOT = (
    "https://wandb.ai/jingyanshen-new-york-university/"
    "chess-47m-context2048-tokenizer-mixing-fp32-master-v13/runs"
)
RL_WANDB_ENTITY = "jingyanshen-new-york-university"
RL_WANDB_PROJECT = "chess-47m-context2048-rl"
RL_WANDB_PROJECT_PATH = f"{RL_WANDB_ENTITY}/{RL_WANDB_PROJECT}"
RL_WANDB_URL = f"https://wandb.ai/{RL_WANDB_PROJECT_PATH}"
RL_WANDB_GROUP = "all-four-context2048-checkpoints-filtered-lr1e5"
RL_ADAM_CONTINUATION_WANDB_GROUP = (
    "ctx2048-mixed-sft3-fresh-vs-continued-adam-lr1e5"
)
RL_TARGET_UPDATES = 1_500
RL_LR = "1e-5"

SOURCE_ROWS = 53_225
SOURCE_SHA256 = (
    "bcf131d88cd9916f6ec12a12f8e85901a05e0dabc5d5a1fdebd8ea0b0ae86c30"
)
N_SAMPLES = 16
SHARD_COUNT = 4
EXPECTED_EVALUATED_PROMPTS = 53_156
EXPECTED_SKIPPED_OVERLONG = 69

CHECKPOINTS: dict[str, dict[str, str]] = {
    "vocab81_then_sft3": {
        "label": (
            "9,181,735,000-token pretraining with the 81-token tokenizer, "
            "deterministic expansion to 85 tokens, then 77,717-row SFT for 3 epochs"
        ),
        "detail": "Separate pretraining and SFT; 256-sequence SFT global batch",
        "fingerprint": (
            "350e1eb7dd87e5fb0107437a3ccdb1dc42efdc034edd4cc0b502738c04de7270"
        ),
        "validation_checkpoint_path": (
            "/pretrain-checkpoints/context2048_vocab_mixing_fp32_master_v13_20260813/"
            "vocab81_then_sft3/sft/final"
        ),
        "color": "#68d5c1",
    },
    "vocab85_then_sft3": {
        "label": (
            "9,181,735,000-token pretraining with the native 85-token tokenizer, "
            "then 77,717-row SFT for 3 epochs"
        ),
        "detail": "Separate pretraining and SFT; 256-sequence SFT global batch",
        "fingerprint": (
            "0b286a1ad928c1efefb135cdd8d8bf28d867276e28a7dc682ade3684e6ee6c19"
        ),
        "validation_checkpoint_path": (
            "/pretrain-checkpoints/context2048_vocab_mixing_fp32_master_v13_20260813/"
            "vocab85_then_sft3/sft/final"
        ),
        "color": "#79a8ff",
    },
    "mixed_sft1": {
        "label": (
            "Native-85 uniformly shuffled mixed training: 9,181,735,000-token "
            "pretraining data plus one independently placed copy of every SFT row"
        ),
        "detail": "Each SFT row remains a separate right-padded sequence",
        "fingerprint": (
            "e42a2ed9a5e2b0550c5e5e06ef48e4089ff046d4415d2b4c9c28af0745c0c139"
        ),
        "validation_checkpoint_path": (
            "/pretrain-checkpoints/context2048_vocab_mixing_fp32_master_v13_20260813/"
            "mixed_sft1/mixed/final"
        ),
        "color": "#ba91ff",
    },
    "mixed_sft3": {
        "label": (
            "Native-85 uniformly shuffled mixed training: 9,181,735,000-token "
            "pretraining data plus three independently shuffled SFT copies"
        ),
        "detail": "Each SFT row remains a separate right-padded sequence",
        "fingerprint": (
            "61193269be0afc01e310705fef7ed071ea8b224da83242db52594279edf32075"
        ),
        "validation_checkpoint_path": (
            "/pretrain-checkpoints/context2048_vocab_mixing_fp32_master_v13_20260813/"
            "mixed_sft3/mixed/final"
        ),
        "color": "#ffb66d",
    },
}

TRAINING_STAGES: dict[str, tuple[dict[str, Any], ...]] = {
    "vocab81_then_sft3": (
        {
            "stage": "pt",
            "label": "PT (81-token tokenizer)",
            "relative_root": "vocab81_then_sft3/pt",
            "target_steps": 35_026,
            "manifest_hash": (
                "57b2c6e9c494cd0edad8910ba14c90984007651f6a38fe3e3195c3e23d67ee77"
            ),
            "wandb_id": (
                "context2048_vocab_mixing_fp32_master_v13_20260813-"
                "vocab81_then_sft3-pt"
            ),
            "vocab_size": 81,
        },
        {
            "stage": "sft",
            "label": "SFT ×3 (expanded to 85 tokens)",
            "relative_root": "vocab81_then_sft3/sft",
            "target_steps": 911,
            "manifest_hash": (
                "f0e682fca1426196ceafed8e765dc6b0d0d8a83431a045d4600935744c94ef6d"
            ),
            "wandb_id": (
                "context2048_vocab_mixing_fp32_master_v13_20260813-"
                "vocab81_then_sft3-sft"
            ),
            "vocab_size": 85,
        },
    ),
    "vocab85_then_sft3": (
        {
            "stage": "pt",
            "label": "PT (85-token tokenizer)",
            "relative_root": "vocab85_then_sft3/pt",
            "target_steps": 35_026,
            "manifest_hash": (
                "57b2c6e9c494cd0edad8910ba14c90984007651f6a38fe3e3195c3e23d67ee77"
            ),
            "wandb_id": (
                "context2048_vocab_mixing_fp32_master_v13_20260813-"
                "vocab85_then_sft3-pt"
            ),
            "vocab_size": 85,
        },
        {
            "stage": "sft",
            "label": "SFT ×3",
            "relative_root": "vocab85_then_sft3/sft",
            "target_steps": 911,
            "manifest_hash": (
                "f0e682fca1426196ceafed8e765dc6b0d0d8a83431a045d4600935744c94ef6d"
            ),
            "wandb_id": (
                "context2048_vocab_mixing_fp32_master_v13_20260813-"
                "vocab85_then_sft3-sft"
            ),
            "vocab_size": 85,
        },
    ),
    "mixed_sft1": (
        {
            "stage": "mixed",
            "label": "Mixed PT + one SFT copy",
            "relative_root": "mixed_sft1/mixed",
            "target_steps": 35_633,
            "manifest_hash": (
                "bb588033e7576a5ef41efb46c872092846dfcb484ea91ca6b01b17d5a741f22d"
            ),
            "wandb_id": (
                "context2048_vocab_mixing_fp32_master_v13_20260813-"
                "mixed_sft1-mixed"
            ),
            "vocab_size": 85,
        },
    ),
    "mixed_sft3": (
        {
            "stage": "mixed",
            "label": "Mixed PT + three SFT copies",
            "relative_root": "mixed_sft3/mixed",
            "target_steps": 36_848,
            "manifest_hash": (
                "82a741565fa2413268a784c5a0fa308a61cb5a92892b99dc29492180a085eb5d"
            ),
            "wandb_id": (
                "context2048_vocab_mixing_fp32_master_v13_20260813-"
                "mixed_sft3-mixed"
            ),
            "vocab_size": 85,
        },
    ),
}

RL_RUNS: dict[str, dict[str, Any]] = {
    "vocab81_then_sft3": {
        "label": f"{CHECKPOINTS['vocab81_then_sft3']['label']} · RL 1e-5",
        "run_name": (
            "ctx2048-fp32masterv13-vocab81pt-expand85-sft3-"
            "filtered-lr1e5-rl1500-r3"
        ),
        "wandb_id": "prodaad055b695a8fbbc745fa0a86671",
        "color": CHECKPOINTS["vocab81_then_sft3"]["color"],
    },
    "vocab85_then_sft3": {
        "label": f"{CHECKPOINTS['vocab85_then_sft3']['label']} · RL 1e-5",
        "run_name": (
            "ctx2048-fp32masterv13-vocab85pt-sft3-filtered-lr1e5-rl1500-r3"
        ),
        "wandb_id": "prod9a04ec3ee6a837fd2d53a5ab7f97",
        "color": CHECKPOINTS["vocab85_then_sft3"]["color"],
    },
    "mixed_sft1": {
        "label": f"{CHECKPOINTS['mixed_sft1']['label']} · RL 1e-5",
        "run_name": (
            "ctx2048-fp32masterv13-mixed-sft1-filtered-lr1e5-rl1500-r3"
        ),
        "wandb_id": "prodac9fa58fe7c3ba6c6345fa3cac3d",
        "color": CHECKPOINTS["mixed_sft1"]["color"],
    },
    "mixed_sft3": {
        "label": f"{CHECKPOINTS['mixed_sft3']['label']} · RL 1e-5",
        "run_name": (
            "ctx2048-fp32masterv13-mixed-sft3-filtered-lr1e5-rl1500-r3"
        ),
        "wandb_id": "prod1d227d32f9abe5ef0bd7b3d2b10a",
        "color": CHECKPOINTS["mixed_sft3"]["color"],
    },
    "mixed_sft3_adam_continuation": {
        "label": (
            f"{CHECKPOINTS['mixed_sft3']['label']} · RL 1e-5 · "
            "continued FP32 Adam moments and parameter step 36,848"
        ),
        "run_name": (
            "ctx2048-fp32masterv13-mixed-pt-plus-sft3-continue-adam36848-"
            "filtered-lr1e5-rl1500"
        ),
        "wandb_id": "prod3fb2066bcca31ed09fe4f7a4cc80",
        "wandb_group": RL_ADAM_CONTINUATION_WANDB_GROUP,
        "color": "#ff668a",
        "fixed_launch": {
            "function_call_id": "fc-01M021JAAA1PCWZN36PETQWZMN",
            "launched_at": "2026-08-15T06:25:32.411495+00:00",
            "claim_sha256": (
                "e22b31d20e6e447704e3165d54eab38b869d1ace180b8f26f1d9cd3b5f27a179"
            ),
            "execution_sha256": (
                "e1a37c8bee23a9fa88ada42ec349560f36a962ea10b6d11836a567e97f539fc4"
            ),
            "resolution_sha256": (
                "06b5f7de6f1d3253f455cbdc0bd6210ba8aad2d7551e369f244d0f50c7474ef6"
            ),
            "initial_adam_step": 36_848,
        },
    },
}

RL_WANDB_METRICS: dict[str, str] = {
    "reward": "rollout/raw_reward",
    "pass_at_1": "passrate/pass@1",
    "pass_at_2": "passrate/pass@2",
    "pass_at_4": "passrate/pass@4",
    "pass_at_8": "passrate/pass@8",
    "entropy": "rollout/entropy",
    "grad_norm": "train/grad_norm",
    "ppo_kl": "train/ppo_kl",
    "all_zero_percentage": "rollout/zero_std/all_zero_percentage",
}


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object at {path}")
    return value


def _pass_at_k(histogram: dict[int, int], *, n: int, k: int) -> float:
    total = sum(histogram.values())
    if total <= 0 or not 1 <= k <= n:
        raise ValueError("invalid pass@k inputs")
    result = 0.0
    for successes, count in histogram.items():
        if not 0 <= successes <= n or count < 0:
            raise ValueError("invalid wins histogram")
        failures = n - successes
        miss = math.comb(failures, k) / math.comb(n, k) if failures >= k else 0.0
        result += count * (1.0 - miss)
    return result / total


def _verified_shard(root: Path, key: str, shard_id: int) -> dict[str, Any] | None:
    path = root / key / "n16" / f"shard-{shard_id:02d}" / "success.json"
    if not path.is_file():
        return None
    summary = _read_json(path)
    recorded = summary.get("summary_sha256")
    core = {name: value for name, value in summary.items() if name != "summary_sha256"}
    if recorded != _canonical_sha256(core):
        raise ValueError(f"summary hash mismatch for {key}/shard-{shard_id:02d}")
    spec = CHECKPOINTS[key]
    if (
        summary.get("schema") != "context2048-pass16-shard-summary-v1"
        or summary.get("version") != VERSION
        or summary.get("checkpoint") != key
        or summary.get("checkpoint_fingerprint") != spec["fingerprint"]
        or summary.get("shard_id") != shard_id
        or summary.get("n_samples") != N_SAMPLES
        or summary.get("source_sha256") != SOURCE_SHA256
    ):
        raise ValueError(f"summary identity mismatch for {key}/shard-{shard_id:02d}")
    generation = summary.get("generation", {})
    if generation.get("bos_prepended_exactly_once_by_evaluator") is not True:
        raise ValueError(f"BOS contract missing for {key}/shard-{shard_id:02d}")
    if (
        generation.get("prompt_cap_including_bos") != 512
        or generation.get("dataset_prefilter_cap_excluding_bos") != 511
        or generation.get("response_budget") != 1_536
        or generation.get("model_context") != 2_048
        or generation.get("context_margin") != 0
    ):
        raise ValueError(f"context contract mismatch for {key}/shard-{shard_id:02d}")
    return summary


def _verified_filter(root: Path, key: str) -> dict[str, Any] | None:
    path = root / key / "filter" / "success.json"
    if not path.is_file():
        return None
    summary = _read_json(path)
    recorded = summary.get("filter_sha256")
    core = {name: value for name, value in summary.items() if name != "filter_sha256"}
    if recorded != _canonical_sha256(core):
        raise ValueError(f"filter hash mismatch for {key}")
    if (
        summary.get("schema") != "context2048-mixed-outcome-filter-v1"
        or summary.get("version") != VERSION
        or summary.get("checkpoint") != key
        or summary.get("checkpoint_fingerprint") != CHECKPOINTS[key]["fingerprint"]
        or summary.get("rule") != "1 <= success_count <= 15 from exactly 16 samples"
    ):
        raise ValueError(f"filter identity mismatch for {key}")
    record = summary.get("filtered_parquet", {})
    if (
        not isinstance(record.get("rows"), int)
        or record["rows"] <= 0
        or len(str(record.get("sha256", ""))) != 64
    ):
        raise ValueError(f"invalid filtered parquet record for {key}")
    return summary


def _verified_common_filter(root: Path) -> dict[str, Any] | None:
    path = root / "common_filter" / "success.json"
    if not path.is_file():
        return None
    summary = _read_json(path)
    recorded = summary.get("filter_sha256")
    core = {name: value for name, value in summary.items() if name != "filter_sha256"}
    if recorded != _canonical_sha256(core):
        raise ValueError("common-filter hash mismatch")
    source = summary.get("source", {})
    record = summary.get("filtered_parquet", {})
    if (
        summary.get("schema") != "context2048-common-mixed-outcome-filter-v1"
        or summary.get("version") != VERSION
        or summary.get("n_samples_per_checkpoint") != N_SAMPLES
        or summary.get("comparison_contract")
        != "all four RL runs must use this exact parquet and SHA-256"
        or source.get("rows") != SOURCE_ROWS
        or source.get("sha256") != SOURCE_SHA256
        or not isinstance(record.get("rows"), int)
        or record["rows"] <= 0
        or len(str(record.get("sha256", ""))) != 64
    ):
        raise ValueError("common-filter identity mismatch")
    return summary


def _aggregate_checkpoint(root: Path, key: str) -> dict[str, Any]:
    spec = CHECKPOINTS[key]
    shards = [_verified_shard(root, key, shard_id) for shard_id in range(SHARD_COUNT)]
    complete_shards = [summary for summary in shards if summary is not None]
    result: dict[str, Any] = {
        "key": key,
        **spec,
        "state": "complete" if len(complete_shards) == SHARD_COUNT else "running",
        "shards_complete": len(complete_shards),
        "shards_total": SHARD_COUNT,
        "metrics": None,
        "filter": None,
    }
    if len(complete_shards) != SHARD_COUNT:
        return result

    histogram: dict[int, int] = {}
    evaluated = skipped = trajectories = 0
    format_hits = 0.0
    finished_at: list[str] = []
    for summary in complete_shards:
        for wins, count in summary["wins_histogram"].items():
            integer_wins = int(wins)
            histogram[integer_wins] = histogram.get(integer_wins, 0) + int(count)
        evaluated += int(summary["evaluated_prompts"])
        skipped += len(summary["skipped_overlong"])
        trajectories += int(summary["trajectories"])
        format_hits += float(summary["format_rate"]) * int(summary["trajectories"])
        finished_at.append(str(summary["finished_at"]))

    if evaluated != EXPECTED_EVALUATED_PROMPTS or skipped != EXPECTED_SKIPPED_OVERLONG:
        raise ValueError(
            f"coverage mismatch for {key}: {evaluated} evaluated, {skipped} skipped"
        )
    if evaluated + skipped != SOURCE_ROWS or trajectories != evaluated * N_SAMPLES:
        raise ValueError(f"trajectory accounting mismatch for {key}")
    mixed_outcome = sum(count for wins, count in histogram.items() if 0 < wins < N_SAMPLES)
    metrics = {
        "evaluated_prompts": evaluated,
        "skipped_overlong": skipped,
        "trajectories": trajectories,
        "pass_at_1": _pass_at_k(histogram, n=N_SAMPLES, k=1),
        "pass_at_16": _pass_at_k(histogram, n=N_SAMPLES, k=16),
        "format_rate": format_hits / trajectories,
        "mixed_outcome_prompts": mixed_outcome,
        "mixed_outcome_share": mixed_outcome / evaluated,
        "wins_histogram": {str(wins): histogram.get(wins, 0) for wins in range(17)},
        "finished_at": max(finished_at),
    }
    filter_summary = _verified_filter(root, key)
    if filter_summary is not None:
        filtered = filter_summary["filtered_parquet"]
        if int(filtered["rows"]) != mixed_outcome:
            raise ValueError(f"filtered row count disagrees with wins histogram for {key}")
        result["filter"] = {
            "state": "complete",
            "rows": int(filtered["rows"]),
            "sha256": str(filtered["sha256"]),
            "created_at": filter_summary["created_at"],
        }
    else:
        result["filter"] = {
            "state": "pending",
            "rows": mixed_outcome,
            "sha256": None,
            "created_at": None,
        }
    result["metrics"] = metrics
    return result


def _finite_metric(value: Any, field: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < minimum:
        raise ValueError(f"{field} must be finite and >= {minimum}")
    return number


def _read_training_stage(root: Path, key: str, spec: dict[str, Any]) -> dict[str, Any]:
    stage_root = root / str(spec["relative_root"])
    metrics_path = stage_root / "metrics.jsonl"
    state_path = stage_root / "final" / "interleaved_training_state.json"
    state = _read_json(state_path)
    if int(state.get("global_step", -1)) != int(spec["target_steps"]):
        raise ValueError(f"final training step mismatch for {key}/{spec['stage']}")
    provenance = state.get("configured_provenance", {})
    if (
        provenance.get("experiment_version")
        != "context2048_vocab_mixing_fp32_master_v13_20260813"
        or provenance.get("experiment") != key
        or provenance.get("stage") != spec["stage"]
        or int(provenance.get("context_length", -1)) != 2_048
        or int(provenance.get("vocab_size", -1)) != int(spec["vocab_size"])
    ):
        raise ValueError(f"training provenance mismatch for {key}/{spec['stage']}")

    points: list[dict[str, Any]] = []
    previous_step = 0
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            record = json.loads(line)
            if (
                record.get("schema") != "interleaved-local-metrics-v1"
                or record.get("manifest_hash") != spec["manifest_hash"]
            ):
                raise ValueError(
                    f"training metric identity mismatch for {key}/{spec['stage']} "
                    f"at line {line_number}"
                )
            step = int(record.get("step", -1))
            if step <= previous_step or step > int(spec["target_steps"]):
                raise ValueError(f"non-monotonic training step for {key}/{spec['stage']}")
            previous_step = step
            metrics = record.get("metrics", {})
            point: dict[str, Any] = {
                "step": step,
                "loss": _finite_metric(metrics.get("train/loss"), "train/loss"),
                "lr": _finite_metric(metrics.get("train/lr"), "train/lr"),
                "pretrain_token_loss": None,
                "sft_token_loss": None,
                "pretrain_valid_tokens": int(
                    metrics.get("train/global_pretrain_valid_tokens", 0)
                ),
                "sft_valid_tokens": int(metrics.get("train/global_sft_valid_tokens", 0)),
                "effective_sft_loss_mass_share": _finite_metric(
                    metrics.get("train/effective_sft_loss_mass_share", 0.0),
                    "train/effective_sft_loss_mass_share",
                ),
                "token_positions_per_second": _finite_metric(
                    metrics.get("train/token_positions_per_second"),
                    "train/token_positions_per_second",
                ),
            }
            if "train/pretrain_token_loss" in metrics:
                point["pretrain_token_loss"] = _finite_metric(
                    metrics["train/pretrain_token_loss"], "train/pretrain_token_loss"
                )
            if "train/sft_token_loss" in metrics:
                point["sft_token_loss"] = _finite_metric(
                    metrics["train/sft_token_loss"], "train/sft_token_loss"
                )
            points.append(point)
    if not points:
        raise ValueError(f"training metrics are empty for {key}/{spec['stage']}")

    last = points[-1]
    tail = points[-10:]
    return {
        "stage": spec["stage"],
        "label": spec["label"],
        "state": "complete",
        "target_step": int(spec["target_steps"]),
        "last_logged_step": int(last["step"]),
        "logged_points": len(points),
        "wandb_url": f"{WANDB_ROOT}/{spec['wandb_id']}",
        "last": last,
        "tail_10_mean": {
            name: (
                statistics.fmean(
                    float(point[name]) for point in tail if point[name] is not None
                )
                if any(point[name] is not None for point in tail)
                else None
            )
            for name in ("loss", "pretrain_token_loss", "sft_token_loss", "lr")
        },
        "series": points,
    }


def _collect_training(root: Path = PRETRAIN_ROOT) -> dict[str, Any]:
    experiments: dict[str, Any] = {}
    errors: list[str] = []
    for key, stage_specs in TRAINING_STAGES.items():
        stages: list[dict[str, Any]] = []
        for spec in stage_specs:
            try:
                stages.append(_read_training_stage(root, key, spec))
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                stages.append(
                    {
                        "stage": spec["stage"],
                        "label": spec["label"],
                        "state": "error",
                        "target_step": int(spec["target_steps"]),
                        "series": [],
                    }
                )
                errors.append(f"training {key}/{spec['stage']}: {type(exc).__name__}: {exc}")
        experiments[key] = {"stages": stages}
    return {
        "status": "training_only_no_validation_logged",
        "validation_loss_logged": False,
        "validation_note": (
            "The trainer recorded only train/* metrics. All 77,717 SFT rows "
            "were consumed by training, so same-source SFT validation loss is unavailable."
        ),
        "wandb_project": (
            "https://wandb.ai/jingyanshen-new-york-university/"
            "chess-47m-context2048-tokenizer-mixing-fp32-master-v13"
        ),
        "experiments": experiments,
        "errors": errors,
    }


def _read_heldout_validation(root: Path, key: str) -> dict[str, Any]:
    path = root / key / "success.json"
    if not path.is_file():
        return {"state": "pending", "metrics": None}
    result = _read_json(path)
    recorded = result.get("result_sha256")
    core = {name: value for name, value in result.items() if name != "result_sha256"}
    if recorded != _canonical_sha256(core):
        raise ValueError(f"held-out validation result hash mismatch for {key}")
    if (
        result.get("schema") != "context2048-heldout-pt-result-v1"
        or result.get("version") != VALIDATION_VERSION
        or result.get("state") != "complete"
        or result.get("checkpoint") != key
        or result.get("checkpoint_path")
        != CHECKPOINTS[key]["validation_checkpoint_path"]
        or result.get("checkpoint_fingerprint") != CHECKPOINTS[key]["fingerprint"]
    ):
        raise ValueError(f"held-out validation identity mismatch for {key}")
    metrics = result.get("metrics", {})
    loss = _finite_metric(metrics.get("heldout_pretrain_loss"), "heldout_pretrain_loss")
    perplexity = _finite_metric(
        metrics.get("heldout_pretrain_perplexity"),
        "heldout_pretrain_perplexity",
        minimum=1.0,
    )
    accuracy = _finite_metric(
        metrics.get("heldout_pretrain_token_accuracy"),
        "heldout_pretrain_token_accuracy",
    )
    targets = int(metrics.get("heldout_pretrain_target_tokens", -1))
    correct = int(metrics.get("heldout_pretrain_correct_tokens", -1))
    if targets != VALIDATION_TARGET_TOKENS or not 0 <= correct <= targets or accuracy > 1.0:
        raise ValueError(f"held-out validation accounting mismatch for {key}")
    return {
        "state": "complete",
        "metrics": {
            "heldout_pretrain_loss": loss,
            "heldout_pretrain_perplexity": perplexity,
            "heldout_pretrain_token_accuracy": accuracy,
            "heldout_pretrain_correct_tokens": correct,
            "heldout_pretrain_target_tokens": targets,
        },
        "holdout_hash": result.get("holdout_hash"),
        "finished_at": result.get("finished_at"),
    }


def _read_authenticated_rl_ledger(
    path: Path, *, expected_schema: str
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    ledger = _read_json(path)
    recorded = ledger.get("ledger_sha256")
    core = {name: value for name, value in ledger.items() if name != "ledger_sha256"}
    if not isinstance(recorded, str) or recorded != _canonical_sha256(core):
        raise ValueError(f"RL launch ledger hash mismatch at {path.name}")
    if (
        ledger.get("schema") != expected_schema
        or ledger.get("version") != VERSION
    ):
        raise ValueError(f"RL launch ledger identity mismatch at {path.name}")
    return ledger


def _collect_rl_launch_records(root: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    records: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    ledger_specs = (
        (root / "pipeline.json", "context2048-pass16-rl-pipeline-v1"),
        (root / "remaining_rl_launch.json", "context2048-remaining-rl-launch-v1"),
    )
    for path, schema in ledger_specs:
        try:
            ledger = _read_authenticated_rl_ledger(path, expected_schema=schema)
            if ledger is None:
                continue
            calls = ledger.get("rl_calls", [])
            if not isinstance(calls, list):
                raise ValueError("rl_calls must be an array")
            for raw in calls:
                if not isinstance(raw, dict):
                    raise ValueError("RL call record must be an object")
                key = str(raw.get("checkpoint", ""))
                spec = RL_RUNS.get(key)
                if spec is None:
                    raise ValueError(f"unknown RL checkpoint {key!r}")
                if key in records:
                    raise ValueError(f"duplicate RL call for {key}")
                call_id = str(raw.get("function_call_id", ""))
                if not re.fullmatch(r"fc-[0-9A-Z]+", call_id):
                    raise ValueError(f"invalid Modal FunctionCall id for {key}")
                if (
                    raw.get("run_name") != spec["run_name"]
                    or raw.get("wandb_project") != RL_WANDB_PROJECT
                    or raw.get("wandb_group")
                    != spec.get("wandb_group", RL_WANDB_GROUP)
                ):
                    raise ValueError(f"RL launch settings mismatch for {key}")
                filter_summary = _verified_common_filter(root)
                if filter_summary is None:
                    raise ValueError("authenticated common RL filter missing")
                expected_filter = filter_summary["filtered_parquet"]
                observed_filter = raw.get("filtered_parquet", {})
                if any(
                    observed_filter.get(field) != expected_filter.get(field)
                    for field in ("path", "sha256", "rows")
                ):
                    raise ValueError(f"RL filtered dataset mismatch for {key}")
                records[key] = {
                    "function_call_id": call_id,
                    "modal_url": f"https://modal.com/id/{call_id}",
                    "filtered_parquet": {
                        field: expected_filter.get(field)
                        for field in ("path", "sha256", "rows", "bytes")
                    },
                    "launched_at": ledger.get("finished_at") or ledger.get("created_at"),
                }
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"RL ledger {path.name}: {type(exc).__name__}: {exc}")

    # The fifth arm was launched later through the authenticated production
    # exact-once claim path, rather than the original four-run evaluation
    # ledger. Bind its immutable claim/execution/resolution hashes and worker
    # ID in this dashboard deployment so it appears in the same comparison.
    filter_summary = _verified_common_filter(root)
    if filter_summary is not None:
        expected_filter = filter_summary["filtered_parquet"]
        for key, spec in RL_RUNS.items():
            fixed = spec.get("fixed_launch")
            if key in records or not isinstance(fixed, dict):
                continue
            call_id = str(fixed.get("function_call_id", ""))
            hashes = (
                fixed.get("claim_sha256"),
                fixed.get("execution_sha256"),
                fixed.get("resolution_sha256"),
            )
            if (
                not re.fullmatch(r"fc-[0-9A-Z]+", call_id)
                or any(
                    not isinstance(value, str)
                    or not re.fullmatch(r"[0-9a-f]{64}", value)
                    for value in hashes
                )
            ):
                errors.append(f"fixed RL launch identity is invalid for {key}")
                continue
            records[key] = {
                "function_call_id": call_id,
                "modal_url": f"https://modal.com/id/{call_id}",
                "filtered_parquet": {
                    field: expected_filter.get(field)
                    for field in ("path", "sha256", "rows", "bytes")
                },
                "launched_at": fixed.get("launched_at"),
                "claim_sha256": fixed["claim_sha256"],
                "execution_sha256": fixed["execution_sha256"],
                "resolution_sha256": fixed["resolution_sha256"],
                "initial_adam_step": fixed.get("initial_adam_step"),
            }
    return records, errors


def _history_points(row_groups: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    by_step: dict[int, dict[str, Any]] = {}
    reverse_metrics = {source: field for field, source in RL_WANDB_METRICS.items()}
    for rows in row_groups:
        for row in rows:
            raw_step = row.get("rollout/step", row.get("train/step"))
            if isinstance(raw_step, bool) or not isinstance(raw_step, (int, float)):
                continue
            if not math.isfinite(float(raw_step)):
                continue
            step = int(raw_step)
            if step < 0 or step >= RL_TARGET_UPDATES:
                continue
            point = by_step.setdefault(step, {"step": step})
            for source, field in reverse_metrics.items():
                value = row.get(source)
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                number = float(value)
                if not math.isfinite(number):
                    continue
                if field in {
                    "reward",
                    "pass_at_1",
                    "pass_at_2",
                    "pass_at_4",
                    "pass_at_8",
                    "all_zero_percentage",
                } and not 0.0 <= number <= 1.0:
                    continue
                if field in {"entropy", "grad_norm"} and number < 0.0:
                    continue
                point[field] = number
    return [by_step[step] for step in sorted(by_step)]


def _fetch_rl_wandb_history(run_name: str) -> dict[str, Any]:
    import wandb

    api = wandb.Api(timeout=20)
    run_spec = next(
        (spec for spec in RL_RUNS.values() if spec["run_name"] == run_name), None
    )
    if run_spec is None:
        raise ValueError(f"unknown RL W&B run {run_name}")
    display_names = [run_name, f"{run_name}_miles"]
    intended_url = f"{RL_WANDB_URL}/runs/{run_spec['wandb_id']}"
    try:
        run = api.run(f"{RL_WANDB_PROJECT_PATH}/{run_spec['wandb_id']}")
    except Exception:
        return {
            "state": "pending",
            "run_id": run_spec["wandb_id"],
            "run_name": f"{run_name}_miles",
            "run_url": intended_url,
            "wandb_state": None,
            "points": [],
            "note": "The W&B run has not appeared in the intended project yet.",
        }
    if run.id != run_spec["wandb_id"] or run.name not in display_names:
        raise ValueError(f"W&B run identity mismatch for {run_name}")
    query_groups = (
        ["rollout/step", "rollout/raw_reward", "rollout/entropy"],
        [
            "rollout/step",
            "passrate/pass@1",
            "passrate/pass@2",
            "passrate/pass@4",
            "passrate/pass@8",
        ],
        ["train/step", "train/grad_norm", "train/ppo_kl"],
        ["rollout/step", "rollout/zero_std/all_zero_percentage"],
    )
    row_groups = [
        list(run.scan_history(keys=list(keys), page_size=2_000))
        for keys in query_groups
    ]
    points = _history_points(row_groups)
    return {
        "state": "reporting" if points else "pending",
        "run_id": run.id,
        "run_name": run.name,
        "run_url": run.url or intended_url,
        "wandb_state": run.state,
        "points": points,
        "note": None if points else "The W&B run exists but has no RL metrics yet.",
    }


_wandb_cache_lock = threading.Lock()
_wandb_cache: dict[str, tuple[float, dict[str, Any]]] = {}
WANDB_CACHE_SECONDS = 45


def _cached_rl_wandb_history(run_name: str) -> dict[str, Any]:
    now = time.monotonic()
    with _wandb_cache_lock:
        cached = _wandb_cache.get(run_name)
        if cached is not None and now - cached[0] < WANDB_CACHE_SECONDS:
            return cached[1]
    try:
        value = _fetch_rl_wandb_history(run_name)
    except Exception as exc:  # A missing/private project must not break the page.
        run_spec = next(
            (spec for spec in RL_RUNS.values() if spec["run_name"] == run_name),
            None,
        )
        run_id = run_spec.get("wandb_id") if run_spec else None
        value = {
            "state": "pending",
            "run_id": run_id,
            "run_name": f"{run_name}_miles",
            "run_url": f"{RL_WANDB_URL}/runs/{run_id}" if run_id else None,
            "wandb_state": None,
            "points": [],
            "note": f"W&B is not reporting yet ({type(exc).__name__}).",
        }
    with _wandb_cache_lock:
        _wandb_cache[run_name] = (now, value)
    return value


def _find_modal_call_status(nodes: list[Any], call_id: str) -> str | None:
    for node in nodes:
        node_id = getattr(node, "function_call_id", None)
        status = getattr(node, "status", None)
        if node_id == call_id:
            name = getattr(status, "name", None)
            return str(name or status).lower()
        found = _find_modal_call_status(list(getattr(node, "children", []) or []), call_id)
        if found is not None:
            return found
    return None


def _fetch_modal_call_status(call_id: str) -> dict[str, Any]:
    call = modal.FunctionCall.from_id(call_id)
    status = _find_modal_call_status(call.get_call_graph(), call_id)
    return {
        "state": status or "unknown",
        "url": call.get_dashboard_url(),
    }


def _bounded_modal_call_status(call_id: str, *, timeout: float = 5.0) -> dict[str, Any]:
    """Return control-plane status without letting it delay dashboard metrics."""
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        future = pool.submit(_fetch_modal_call_status, call_id)
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            future.cancel()
            return {
                "state": "unknown",
                "url": f"https://modal.com/id/{call_id}",
                "note": "Modal status lookup timed out; W&B metrics remain live.",
            }
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def _rl_checkpoint_status(root: Path, run_name: str) -> dict[str, Any]:
    run_root = root / run_name
    tracker = run_root / "latest_checkpointed_iteration.txt"
    if not tracker.is_file():
        return {"state": "pending", "step": 0, "validated": False}
    try:
        step = int(tracker.read_text().strip())
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid RL checkpoint tracker for {run_name}") from exc
    checkpoint = run_root / f"iter_{step:07d}"
    if step <= 0 or not checkpoint.is_dir():
        raise ValueError(f"incomplete RL checkpoint {run_name}/iter_{step:07d}")
    marker_path = checkpoint / "COMMITTED.json"
    if not marker_path.is_file():
        raise ValueError(f"uncommitted RL checkpoint {run_name}/iter_{step:07d}")
    marker = _read_json(marker_path)
    marker_core = {
        key: value for key, value in marker.items() if key != "commit_sha256"
    }
    commit_sha256 = marker.get("commit_sha256")
    if commit_sha256 != _canonical_sha256(marker_core):
        raise ValueError(f"RL checkpoint commit hash mismatch for {run_name}")
    if (
        marker.get("schema") != "miles-fsdp-checkpoint-commit-v1"
        or marker.get("iteration") != step
        or marker.get("optimizer_included") is not True
        or marker.get("rng_included") is not True
        or marker.get("rollout_state_included") is not True
    ):
        raise ValueError(f"RL checkpoint commit contract mismatch for {run_name}")

    world_size = marker.get("world_size")
    if (
        isinstance(world_size, bool)
        or not isinstance(world_size, int)
        or world_size <= 0
    ):
        raise ValueError(f"invalid RL checkpoint world size for {run_name}")
    payload = marker.get("payload")
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"invalid RL checkpoint payload for {run_name}")
    payload_by_path: dict[str, dict[str, Any]] = {}
    for entry in payload:
        if not isinstance(entry, dict):
            raise ValueError(f"invalid RL checkpoint payload entry for {run_name}")
        relative = entry.get("path")
        size = entry.get("bytes")
        sha256 = entry.get("sha256")
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or relative in payload_by_path
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
            or not isinstance(sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", sha256) is None
        ):
            raise ValueError(f"invalid RL checkpoint payload entry for {run_name}")
        payload_by_path[relative] = entry

    required_payload = {
        "model/.metadata",
        "optimizer/.metadata",
        "lr_scheduler/.metadata",
        "meta.json",
        "rollout_state.pt",
        *(f"rng_rank_{rank:05d}.pt" for rank in range(world_size)),
    }
    if not required_payload.issubset(payload_by_path):
        raise ValueError(f"incomplete RL checkpoint inventory for {run_name}")

    observed_files: set[str] = set()
    observed_directories: set[str] = set()
    for path in checkpoint.rglob("*"):
        relative = path.relative_to(checkpoint).as_posix()
        if path.is_symlink():
            raise ValueError(f"RL checkpoint contains a symlink: {run_name}/{relative}")
        if path.is_file():
            observed_files.add(relative)
        elif path.is_dir():
            observed_directories.add(relative)
        else:
            raise ValueError(f"unsupported RL checkpoint entry: {run_name}/{relative}")
    expected_files = set(payload_by_path) | {"COMMITTED.json"}
    if observed_files != expected_files:
        raise ValueError(f"RL checkpoint inventory mismatch for {run_name}")
    expected_directories: set[str] = set()
    for relative in expected_files:
        parent = Path(relative).parent
        while parent != Path("."):
            expected_directories.add(parent.as_posix())
            parent = parent.parent
    if observed_directories != expected_directories:
        raise ValueError(f"RL checkpoint directory inventory mismatch for {run_name}")
    for relative, entry in payload_by_path.items():
        path = checkpoint / relative
        if not path.is_file() or path.stat().st_size != entry["bytes"]:
            raise ValueError(f"RL checkpoint payload size mismatch for {run_name}")

    metadata_path = checkpoint / "meta.json"
    metadata = _read_json(metadata_path)
    if (
        int(metadata.get("iteration", -1)) != step
        or int(metadata.get("next_rollout_id", -1)) != step
    ):
        raise ValueError(f"RL checkpoint metadata mismatch for {run_name}")
    return {
        "state": "complete" if step >= RL_TARGET_UPDATES else "checkpointed",
        "step": step,
        "validated": True,
        "commit_sha256": commit_sha256,
    }


def _derive_rl_state(
    *, launched: bool, modal_state: str, checkpoint_step: int, history: dict[str, Any]
) -> str:
    points = history.get("points", [])
    latest_rollout = max((int(point["step"]) for point in points), default=-1)
    if checkpoint_step >= RL_TARGET_UPDATES or latest_rollout >= RL_TARGET_UPDATES - 1:
        return "complete"
    if modal_state in {"failure", "init_failure", "terminated", "timeout"}:
        return "failed"
    if modal_state == "success":
        return "stopped early"
    if checkpoint_step > 0 or points or history.get("wandb_state") == "running":
        return "running"
    if launched:
        return "queued"
    return "awaiting launch"


def _collect_rl(
    root: Path,
    checkpoint_root: Path,
    *,
    history_fetcher=None,
    modal_status_fetcher=None,
) -> dict[str, Any]:
    history_fetcher = history_fetcher or _cached_rl_wandb_history
    modal_status_fetcher = modal_status_fetcher or _bounded_modal_call_status
    launch_records, errors = _collect_rl_launch_records(root)
    histories: dict[str, dict[str, Any]] = {}
    if launch_records:
        with ThreadPoolExecutor(max_workers=len(launch_records)) as pool:
            futures = {
                pool.submit(history_fetcher, RL_RUNS[key]["run_name"]): key
                for key in launch_records
            }
            for future in as_completed(futures):
                key = futures[future]
                try:
                    histories[key] = future.result()
                except Exception as exc:
                    histories[key] = {
                        "state": "pending",
                        "points": [],
                        "note": f"W&B is not reporting yet ({type(exc).__name__}).",
                    }

    runs: list[dict[str, Any]] = []
    for key, spec in RL_RUNS.items():
        launch = launch_records.get(key)
        history = histories.get(
            key,
            {
                "state": "pending",
                "run_id": None,
                "run_name": f"{spec['run_name']}_miles",
                "run_url": None,
                "wandb_state": None,
                "points": [],
                "note": "RL has not been launched yet.",
            },
        )
        try:
            checkpoint = _rl_checkpoint_status(checkpoint_root, spec["run_name"])
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            checkpoint = {"state": "error", "step": 0, "validated": False}
            errors.append(f"RL checkpoint {key}: {type(exc).__name__}: {exc}")
        modal_status = {"state": "not_launched", "url": None}
        if launch is not None:
            if history.get("wandb_state") == "running":
                modal_status = {
                    "state": "running",
                    "url": launch["modal_url"],
                    "note": "W&B is actively reporting this run.",
                }
            else:
                try:
                    modal_status = modal_status_fetcher(launch["function_call_id"])
                except Exception as exc:
                    modal_status = {
                        "state": "unknown",
                        "url": launch["modal_url"],
                        "note": f"Modal status unavailable ({type(exc).__name__}).",
                    }
        points = history.get("points", [])
        latest_rollout = max((int(point["step"]) for point in points), default=-1)
        runs.append(
            {
                "key": key,
                **spec,
                "state": _derive_rl_state(
                    launched=launch is not None,
                    modal_state=str(modal_status.get("state", "unknown")),
                    checkpoint_step=int(checkpoint["step"]),
                    history=history,
                ),
                "target_updates": RL_TARGET_UPDATES,
                "learning_rate": RL_LR,
                "kl_loss_type": "low_var_kl",
                "data_label": "mixed-outcome filtered (1–15/16)",
                "launch": launch,
                "modal": modal_status,
                "checkpoint": checkpoint,
                "wandb": {name: value for name, value in history.items() if name != "points"},
                "latest_rollout_index": latest_rollout if latest_rollout >= 0 else None,
                "updates_observed": latest_rollout + 1 if latest_rollout >= 0 else 0,
                "series": points,
            }
        )
    return {
        "project": RL_WANDB_PROJECT,
        "project_url": RL_WANDB_URL,
        "entity": RL_WANDB_ENTITY,
        "group": RL_WANDB_GROUP,
        "target_updates": RL_TARGET_UPDATES,
        "learning_rate": RL_LR,
        "runs": runs,
        "aggregate": {
            "launched": sum(run["launch"] is not None for run in runs),
            "reporting": sum(bool(run["series"]) for run in runs),
            "running": sum(run["state"] == "running" for run in runs),
            "complete": sum(run["state"] == "complete" for run in runs),
            "total": len(runs),
        },
        "errors": errors,
    }


def _collect_snapshot(
    root: Path = RESULTS_ROOT,
    training_root: Path = PRETRAIN_ROOT,
    validation_root: Path = VALIDATION_ROOT,
    rl_checkpoint_root: Path = RL_CHECKPOINT_ROOT,
) -> dict[str, Any]:
    errors: list[str] = []
    checkpoints: list[dict[str, Any]] = []
    for key in CHECKPOINTS:
        try:
            checkpoint = _aggregate_checkpoint(root, key)
            checkpoint["heldout_pretrain"] = _read_heldout_validation(
                validation_root, key
            )
            checkpoints.append(checkpoint)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            spec = CHECKPOINTS[key]
            checkpoints.append(
                {
                    "key": key,
                    **spec,
                    "state": "error",
                    "shards_complete": 0,
                    "shards_total": SHARD_COUNT,
                    "metrics": None,
                    "filter": None,
                    "heldout_pretrain": {"state": "error", "metrics": None},
                }
            )
            errors.append(f"{key}: {type(exc).__name__}: {exc}")

    pipeline: dict[str, Any] | None = None
    pipeline_path = root / "pipeline.json"
    if pipeline_path.is_file():
        try:
            raw = _read_json(pipeline_path)
            pipeline = {
                "state": raw.get("state"),
                "created_at": raw.get("created_at"),
                "finished_at": raw.get("finished_at"),
                "rl_calls": raw.get("rl_calls", []),
            }
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"pipeline: {type(exc).__name__}: {exc}")

    complete = sum(item["state"] == "complete" for item in checkpoints)
    filters_complete = sum(
        item.get("filter", {}).get("state") == "complete"
        for item in checkpoints
        if isinstance(item.get("filter"), dict)
    )
    shards_complete = sum(int(item["shards_complete"]) for item in checkpoints)
    total_trajectories = sum(
        int(item["metrics"]["trajectories"])
        for item in checkpoints
        if item.get("metrics") is not None
    )
    training = _collect_training(training_root)
    errors.extend(training["errors"])
    rl = _collect_rl(root, rl_checkpoint_root)
    errors.extend(rl["errors"])
    heldout_complete = sum(
        item.get("heldout_pretrain", {}).get("state") == "complete"
        for item in checkpoints
    )
    common_filter: dict[str, Any] | None = None
    try:
        common_summary = _verified_common_filter(root)
        if common_summary is not None:
            record = common_summary["filtered_parquet"]
            common_filter = {
                "state": "complete",
                "rows": int(record["rows"]),
                "sha256": str(record["sha256"]),
                "path": str(record["path"]),
                "created_at": common_summary.get("created_at"),
            }
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"common filter: {type(exc).__name__}: {exc}")
    return {
        "schema": "context2048-checkpoint-eval-dashboard-v3",
        "version": VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "state": "complete" if complete == len(CHECKPOINTS) else "running",
        "aggregate": {
            "checkpoints_complete": complete,
            "checkpoints_total": len(CHECKPOINTS),
            "shards_complete": shards_complete,
            "shards_total": len(CHECKPOINTS) * SHARD_COUNT,
            "filters_complete": filters_complete,
            "filters_total": len(CHECKPOINTS),
            "completed_trajectories": total_trajectories,
            "heldout_pretrain_complete": heldout_complete,
            "heldout_pretrain_total": len(CHECKPOINTS),
            "rl_launched": rl["aggregate"]["launched"],
            "rl_reporting": rl["aggregate"]["reporting"],
            "rl_complete": rl["aggregate"]["complete"],
            "rl_total": rl["aggregate"]["total"],
        },
        "evaluation": {
            "source_rows": SOURCE_ROWS,
            "samples_per_prompt": N_SAMPLES,
            "prompt_cap_including_bos": 512,
            "dataset_prefilter_cap_excluding_bos": 511,
            "response_budget": 1_536,
            "model_context": 2_048,
            "bos_prepended_exactly_once": True,
            "pass_at_k_estimator": "unbiased estimator from 16 samples per prompt",
            "filter_rule": "1–15 successes among exactly 16 samples",
        },
        "pipeline": pipeline,
        "common_filter": common_filter,
        "training": training,
        "rl": rl,
        "checkpoints": checkpoints,
        "errors": errors,
    }


EVAL_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Context-2048 checkpoint evaluation</title>
  <style>
    :root { --bg:#0a0d12; --panel:#111722; --line:#263143; --text:#edf3fa; --muted:#91a0b5; --good:#68d5c1; }
    * { box-sizing:border-box; }
    body { margin:0; background:radial-gradient(circle at 20% 0,#152134 0,transparent 35%),var(--bg); color:var(--text); font-family:Inter,ui-sans-serif,system-ui,-apple-system,sans-serif; }
    main { width:min(1180px,calc(100% - 32px)); margin:0 auto; padding:42px 0 64px; }
    header { display:flex; gap:20px; justify-content:space-between; align-items:flex-start; margin-bottom:26px; }
    h1 { font-size:clamp(28px,4vw,46px); letter-spacing:-.035em; margin:0 0 10px; }
    .subtitle { color:var(--muted); max-width:760px; line-height:1.55; }
    .stamp { color:var(--muted); font-size:13px; white-space:nowrap; padding-top:10px; }
    .cards { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:18px 0 28px; }
    .card,.panel { background:color-mix(in srgb,var(--panel) 94%,transparent); border:1px solid var(--line); border-radius:14px; box-shadow:0 18px 60px #0005; }
    .card { padding:17px 18px; }
    .card .label { color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.1em; }
    .card .value { font-size:24px; font-weight:700; margin-top:7px; }
    .panel { padding:20px; margin-top:14px; overflow:hidden; }
    .panel h2 { font-size:17px; margin:0 0 16px; }
    .panel-head { display:flex; align-items:flex-start; justify-content:space-between; gap:16px; margin-bottom:14px; }
    .panel-head h2 { margin:0; }
    .panel-head p { margin:5px 0 0; color:var(--muted); font-size:12px; line-height:1.45; }
    table { width:100%; border-collapse:collapse; min-width:860px; }
    th { color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.08em; font-weight:600; text-align:right; padding:10px 12px; border-bottom:1px solid var(--line); }
    th:first-child,td:first-child { text-align:left; }
    td { text-align:right; padding:15px 12px; border-bottom:1px solid #202a39; font-variant-numeric:tabular-nums; }
    tr:last-child td { border-bottom:0; }
    .model { display:flex; align-items:flex-start; gap:10px; }
    .dot { width:9px; height:9px; border-radius:50%; margin-top:6px; flex:0 0 auto; }
    .model small { display:block; color:var(--muted); margin-top:3px; line-height:1.35; }
    .status { display:inline-flex; border:1px solid var(--line); border-radius:99px; padding:4px 8px; font-size:12px; color:var(--muted); }
    .status.complete { color:var(--good); border-color:#32665f; background:#102a27; }
    .status.running { color:#9fc1ff; border-color:#395a86; background:#10213a; }
    .status.failed,.status.stopped-early { color:#ffc1c1; border-color:#8a4141; background:#37191d; }
    .bars { display:grid; gap:15px; }
    .bar-row { display:grid; grid-template-columns:minmax(230px,1.4fr) 3fr 72px 3fr 72px; gap:12px; align-items:center; }
    .bar-label { font-size:13px; }
    .track { height:12px; border-radius:99px; background:#202a38; overflow:hidden; }
    .fill { height:100%; border-radius:99px; width:0; transition:width .5s ease; }
    .bar-value { text-align:right; font-variant-numeric:tabular-nums; font-size:13px; }
    .legend { display:grid; grid-template-columns:minmax(230px,1.4fr) 3fr 72px 3fr 72px; gap:12px; margin-bottom:8px; color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.08em; }
    .legend span:nth-child(2),.legend span:nth-child(4) { text-align:center; }
    .chart { width:100%; height:250px; display:block; background:#0d121a; border:1px solid #202a39; border-radius:10px; }
    .chart-grid { display:grid; grid-template-columns:1fr 1fr; gap:14px; }
    .chart-box h3 { margin:0 0 8px; color:var(--muted); font-size:12px; font-weight:600; letter-spacing:.05em; text-transform:uppercase; }
    .chart-legend { display:flex; flex-wrap:wrap; gap:9px 15px; margin-top:10px; color:var(--muted); font-size:12px; }
    .chart-legend span { display:inline-flex; gap:6px; align-items:center; }
    .chart-legend i { width:8px; height:8px; border-radius:50%; }
    .controls { display:flex; gap:10px; align-items:center; color:var(--muted); font-size:12px; }
    select { color:var(--text); background:#0d121a; border:1px solid var(--line); border-radius:8px; padding:7px 9px; }
    .rl-summary { display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin:12px 0 16px; }
    .rl-summary .card { box-shadow:none; }
    .rl-charts { grid-template-columns:repeat(2,minmax(0,1fr)); }
    .mono { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:11px; }
    a { color:#9fc1ff; }
    .note { color:var(--muted); font-size:13px; line-height:1.55; margin-top:18px; }
    .error { display:none; padding:13px 15px; border:1px solid #8a4141; background:#37191d; color:#ffc1c1; border-radius:10px; margin:14px 0; white-space:pre-wrap; }
    @media(max-width:820px) { .cards,.rl-summary{grid-template-columns:repeat(2,1fr)} header{display:block}.stamp{margin-top:8px}.bar-row,.legend{grid-template-columns:160px 1fr 58px}.bar-row .track:nth-of-type(2),.bar-row .bar-value:last-child,.legend span:nth-child(n+4){display:none}.chart-grid,.rl-charts{grid-template-columns:1fr}.panel-head{display:block}.controls{margin-top:10px} }
  </style>
</head>
<body><main>
  <header><div><h1>Context-2048 checkpoint evaluation</h1><div class="subtitle">Four completed PT/SFT checkpoints evaluated on the full RL training source. Each eligible prompt has 16 sampled trajectories. BOS is prepended exactly once.</div></div><div class="stamp" id="updated">loading…</div></header>
  <div class="error" id="error"></div>
  <section class="cards">
    <div class="card"><div class="label">Checkpoints</div><div class="value" id="checkpoints">—</div></div>
    <div class="card"><div class="label">Evaluation shards</div><div class="value" id="shards">—</div></div>
    <div class="card"><div class="label">Completed trajectories</div><div class="value" id="trajectories">—</div></div>
    <div class="card"><div class="label">Filtered datasets</div><div class="value" id="filters">—</div></div>
  </section>
  <section class="panel"><div class="panel-head"><div><h2>Pre-RL full-dataset Pass@k</h2><p>These measurements were made on the four PT/SFT checkpoints before RL training. Pass@16 here is not an RL-training curve.</p></div></div><div class="legend"><span></span><span>Pre-RL Pass@1</span><span></span><span>Pre-RL Pass@16</span><span></span></div><div class="bars" id="bars"></div></section>
  <section class="panel"><h2>Pre-RL evaluation accounting</h2><div style="overflow:auto"><table><thead><tr><th>Checkpoint</th><th>Status</th><th>Pass@1</th><th>Pass@16</th><th>Format</th><th>Evaluated</th><th>Overlength</th><th>Mixed-outcome filtered</th></tr></thead><tbody id="results"></tbody></table></div></section>
  <section class="panel"><div class="panel-head"><div><h2>Common RL comparison data</h2><p>The production RL comparison uses the intersection of prompts that have 1–15 successes for every checkpoint. All four RL runs therefore receive one byte-identical parquet.</p></div><span class="status" id="common-filter-status">pending</span></div><div class="note mono" id="common-filter-detail">waiting for offline filtering</div></section>
  <section class="panel" id="rl-training"><div class="panel-head"><div><h2>RL training results</h2><p>All four runs use the exact same common mixed-outcome filtered (1–15/16) parquet, 1e-5 learning rate, low_var_kl, native 2,048-token context, and 1,500 updates. Curves are read from W&amp;B; checkpoint progress is read from the persistent Modal Volume.</p></div><div class="controls"><label for="rl-smoothing">Curves</label><select id="rl-smoothing"><option value="0">Raw</option><option value="10">Raw + MA(10)</option><option value="25" selected>Raw + MA(25)</option><option value="50">Raw + MA(50)</option></select><a id="rl-wandb-project" target="_blank" rel="noreferrer">W&amp;B project ↗</a></div></div>
    <div class="rl-summary"><div class="card"><div class="label">Launched</div><div class="value" id="rl-launched">—</div></div><div class="card"><div class="label">W&amp;B reporting</div><div class="value" id="rl-reporting">—</div></div><div class="card"><div class="label">Running</div><div class="value" id="rl-running">—</div></div><div class="card"><div class="label">Complete</div><div class="value" id="rl-complete">—</div></div></div>
    <div style="overflow:auto"><table><thead><tr><th>RL run</th><th>Status</th><th>Observed updates</th><th>Checkpoint</th><th>Latest reward</th><th>Latest Pass@1</th><th>W&amp;B</th><th>Modal call</th></tr></thead><tbody id="rl-results"></tbody></table></div>
    <div class="chart-grid rl-charts" style="margin-top:18px">
      <div class="chart-box"><h3>Mean raw reward</h3><svg class="chart rl-chart" data-field="reward" viewBox="0 0 560 250" preserveAspectRatio="none"></svg></div>
      <div class="chart-box"><h3>Pass@1</h3><svg class="chart rl-chart" data-field="pass_at_1" viewBox="0 0 560 250" preserveAspectRatio="none"></svg></div>
      <div class="chart-box"><h3>Pass@2</h3><svg class="chart rl-chart" data-field="pass_at_2" viewBox="0 0 560 250" preserveAspectRatio="none"></svg></div>
      <div class="chart-box"><h3>Pass@4</h3><svg class="chart rl-chart" data-field="pass_at_4" viewBox="0 0 560 250" preserveAspectRatio="none"></svg></div>
      <div class="chart-box"><h3>Pass@8</h3><svg class="chart rl-chart" data-field="pass_at_8" viewBox="0 0 560 250" preserveAspectRatio="none"></svg></div>
      <div class="chart-box"><h3>Policy entropy</h3><svg class="chart rl-chart" data-field="entropy" viewBox="0 0 560 250" preserveAspectRatio="none"></svg></div>
      <div class="chart-box"><h3>Gradient norm</h3><svg class="chart rl-chart" data-field="grad_norm" viewBox="0 0 560 250" preserveAspectRatio="none"></svg></div>
      <div class="chart-box"><h3>PPO KL</h3><svg class="chart rl-chart" data-field="ppo_kl" viewBox="0 0 560 250" preserveAspectRatio="none"></svg></div>
      <div class="chart-box"><h3>All-zero prompt groups</h3><svg class="chart rl-chart" data-field="all_zero_percentage" viewBox="0 0 560 250" preserveAspectRatio="none"></svg></div>
    </div><div class="chart-legend" id="rl-legend"></div><p class="note">Raw lines contain the recorded per-update W&amp;B values. Moving averages are trailing windows and do not replace the raw data. Pass@1/2/4/8 are computed within each RL rollout batch; the pre-RL Pass@16 panel above is a separate full-dataset checkpoint evaluation.</p>
  </section>
  <section class="panel"><div class="panel-head"><div><h2>True held-out pretraining metrics</h2><p>Deterministic CE on 4,096 × 2,048 targets from complete source shards absent from the training selection.</p></div><span class="status" id="validation-status">pending</span></div><div style="overflow:auto"><table><thead><tr><th>Checkpoint</th><th>Validation CE</th><th>Perplexity</th><th>Token accuracy</th><th>Target tokens</th></tr></thead><tbody id="validation-results"></tbody></table></div></section>
  <section class="panel"><div class="panel-head"><div><h2>Recorded training token cross-entropy</h2><p>These are training metrics, not validation loss. Staged PT and SFT counters reset; mixed-run SFT points are noisier because some logged batches contain few SFT targets.</p></div><a id="wandb-project" target="_blank" rel="noreferrer">W&amp;B project ↗</a></div><div class="chart-grid"><div class="chart-box"><h3>Pretraining token CE</h3><svg class="chart" id="pt-chart" viewBox="0 0 560 250" preserveAspectRatio="none"></svg></div><div class="chart-box"><h3>SFT token CE</h3><svg class="chart" id="sft-chart" viewBox="0 0 560 250" preserveAspectRatio="none"></svg></div></div><div class="chart-legend" id="training-legend"></div><div style="overflow:auto;margin-top:16px"><table><thead><tr><th>Training stage</th><th>Last logged step</th><th>Final train CE</th><th>Final PT CE</th><th>Final SFT CE</th><th>Tail-10 PT CE</th><th>Tail-10 SFT CE</th></tr></thead><tbody id="training-results"></tbody></table></div></section>
  <div class="note">The pre-RL Pass@1 and Pass@16 values use the unbiased pass@k estimator over 16 samples per prompt. Checkpoint-specific mixed-outcome filters are retained for audit; the four-run RL comparison uses their common intersection. Rows above the 512-token post-BOS prompt cap are counted as overlength and excluded consistently.</div>
</main>
<script>
  const esc = s => String(s ?? "").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
  const pct = v => typeof v === "number" ? `${(100*v).toFixed(2)}%` : "—";
  const num = v => typeof v === "number" ? v.toLocaleString() : "—";
  const loss = v => typeof v === "number" ? v.toFixed(5) : "—";
  function plot(svg, series, field) {
    const width=560,height=250,pad={l:48,r:14,t:16,b:30};
    const rows=series.map(s=>({...s,points:s.points.filter(p=>typeof p[field]==="number")})).filter(s=>s.points.length);
    if(!rows.length){svg.innerHTML=`<text x="280" y="130" text-anchor="middle" fill="#91a0b5">No recorded series</text>`;return;}
    const values=rows.flatMap(s=>s.points.map(p=>p[field])); let ymin=Math.min(...values),ymax=Math.max(...values); const spread=Math.max(ymax-ymin,.01); ymin-=spread*.08;ymax+=spread*.08;
    const sx=(step,max)=>pad.l+(width-pad.l-pad.r)*(step/max), sy=v=>pad.t+(height-pad.t-pad.b)*(1-(v-ymin)/(ymax-ymin));
    let out=""; for(let i=0;i<5;i++){const y=pad.t+(height-pad.t-pad.b)*i/4;const v=ymax-(ymax-ymin)*i/4;out+=`<line x1="${pad.l}" y1="${y}" x2="${width-pad.r}" y2="${y}" stroke="#202a39"/><text x="${pad.l-7}" y="${y+4}" text-anchor="end" fill="#718197" font-size="10">${v.toFixed(3)}</text>`;}
    rows.forEach(s=>{const max=Math.max(...s.points.map(p=>p.step));const d=s.points.map((p,i)=>`${i?"L":"M"}${sx(p.step,max).toFixed(2)},${sy(p[field]).toFixed(2)}`).join(" ");out+=`<path d="${d}" fill="none" stroke="${s.color}" stroke-width="2" vector-effect="non-scaling-stroke" opacity=".92"/>`;});
    out+=`<line x1="${pad.l}" y1="${height-pad.b}" x2="${width-pad.r}" y2="${height-pad.b}" stroke="#526075"/><text x="${(pad.l+width-pad.r)/2}" y="${height-8}" text-anchor="middle" fill="#718197" font-size="10">fraction of stage updates</text>`;svg.innerHTML=out;
  }
  const boundedRlFields=new Set(["reward","pass_at_1","pass_at_2","pass_at_4","pass_at_8","all_zero_percentage"]);
  function movingAverage(points,field,window) {
    if(window<=1)return points.filter(p=>typeof p[field]==="number");
    const values=[];return points.filter(p=>typeof p[field]==="number").map(p=>{values.push(p[field]);if(values.length>window)values.shift();return {...p,[field]:values.reduce((a,b)=>a+b,0)/values.length};});
  }
  function plotRl(svg,runs,field,window) {
    const width=560,height=250,pad={l:48,r:14,t:16,b:30},target=1500;
    const rows=runs.map(r=>({...r,raw:(r.series||[]).filter(p=>typeof p[field]==="number")})).filter(r=>r.raw.length);
    if(!rows.length){svg.innerHTML=`<text x="280" y="126" text-anchor="middle" fill="#91a0b5">Pending W&amp;B metrics</text><text x="280" y="145" text-anchor="middle" fill="#718197" font-size="10">This panel will populate after the run reports.</text>`;return;}
    const values=rows.flatMap(r=>r.raw.map(p=>p[field]));let ymin=Math.min(...values),ymax=Math.max(...values);
    if(boundedRlFields.has(field)){ymin=0;ymax=1}else{const spread=Math.max(ymax-ymin,Math.max(Math.abs(ymax),.01)*.08);ymin-=spread*.1;ymax+=spread*.1;}
    const sx=step=>pad.l+(width-pad.l-pad.r)*(step/(target-1)),sy=v=>pad.t+(height-pad.t-pad.b)*(1-(v-ymin)/(ymax-ymin));
    let out="";for(let i=0;i<5;i++){const y=pad.t+(height-pad.t-pad.b)*i/4,v=ymax-(ymax-ymin)*i/4;out+=`<line x1="${pad.l}" y1="${y}" x2="${width-pad.r}" y2="${y}" stroke="#202a39"/><text x="${pad.l-7}" y="${y+4}" text-anchor="end" fill="#718197" font-size="10">${v.toFixed(3)}</text>`;}
    rows.forEach(r=>{const path=points=>points.map((p,i)=>`${i?"L":"M"}${sx(p.step).toFixed(2)},${sy(p[field]).toFixed(2)}`).join(" ");out+=`<path d="${path(r.raw)}" fill="none" stroke="${r.color}" stroke-width="1.2" vector-effect="non-scaling-stroke" opacity="${window>1?".22":".9"}"/>`;if(window>1){const smooth=movingAverage(r.raw,field,window);out+=`<path d="${path(smooth)}" fill="none" stroke="${r.color}" stroke-width="2.2" vector-effect="non-scaling-stroke" opacity=".96"/>`;}});
    out+=`<line x1="${pad.l}" y1="${height-pad.b}" x2="${width-pad.r}" y2="${height-pad.b}" stroke="#526075"/><text x="${pad.l}" y="${height-8}" fill="#718197" font-size="10">0</text><text x="${width-pad.r}" y="${height-8}" text-anchor="end" fill="#718197" font-size="10">1,499 rollout index</text>`;svg.innerHTML=out;
  }
  function latestMetric(series,field){for(let i=(series||[]).length-1;i>=0;i--)if(typeof series[i][field]==="number")return series[i][field];return null;}
  let latestRlData=null;
  function renderRl(rl){
    latestRlData=rl;const a=rl.aggregate;
    document.querySelector("#rl-launched").textContent=`${a.launched} / ${a.total}`;document.querySelector("#rl-reporting").textContent=`${a.reporting} / ${a.total}`;document.querySelector("#rl-running").textContent=`${a.running} / ${a.total}`;document.querySelector("#rl-complete").textContent=`${a.complete} / ${a.total}`;document.querySelector("#rl-wandb-project").href=rl.project_url;
    document.querySelector("#rl-results").innerHTML=rl.runs.map(r=>{const stateClass=r.state.replaceAll(" ","-");const wb=r.wandb||{},launch=r.launch||{},modal=r.modal||{},ck=r.checkpoint||{};const wbLink=wb.run_url?`<a href="${esc(wb.run_url)}" target="_blank" rel="noreferrer">run ↗</a>`:`<span title="${esc(wb.note||"")}">pending</span>`;const modalLink=launch.function_call_id?`<a class="mono" href="${esc(modal.url||launch.modal_url)}" target="_blank" rel="noreferrer">${esc(launch.function_call_id.slice(0,14))}… ↗</a><small>${esc(modal.state||"unknown")}</small>`:"—";return `<tr><td><div class="model"><i class="dot" style="background:${r.color}"></i><div>${esc(r.label)}<small>${num((launch.filtered_parquet||{}).rows)} filtered prompts</small></div></div></td><td><span class="status ${stateClass}">${esc(r.state)}</span></td><td>${num(r.updates_observed)} / ${num(r.target_updates)}</td><td>${num(ck.step||0)} / ${num(r.target_updates)}</td><td>${pct(latestMetric(r.series,"reward"))}</td><td>${pct(latestMetric(r.series,"pass_at_1"))}</td><td>${wbLink}</td><td>${modalLink}</td></tr>`;}).join("");
    document.querySelector("#rl-legend").innerHTML=rl.runs.map(r=>`<span><i style="background:${r.color}"></i>${esc(r.label)}</span>`).join("");const window=Number(document.querySelector("#rl-smoothing").value);document.querySelectorAll(".rl-chart").forEach(svg=>plotRl(svg,rl.runs,svg.dataset.field,window));
  }
  function render(data) {
    const a=data.aggregate;
    document.querySelector("#checkpoints").textContent=`${a.checkpoints_complete} / ${a.checkpoints_total}`;
    document.querySelector("#shards").textContent=`${a.shards_complete} / ${a.shards_total}`;
    document.querySelector("#trajectories").textContent=num(a.completed_trajectories);
    document.querySelector("#filters").textContent=`${a.filters_complete} / ${a.filters_total}`;
    document.querySelector("#updated").textContent=`updated ${new Date(data.generated_at).toLocaleString()}`;
    document.querySelector("#bars").innerHTML=data.checkpoints.map(r=>{
      const m=r.metrics||{}; return `<div class="bar-row"><div class="bar-label">${esc(r.label)}</div><div class="track"><div class="fill" style="width:${100*(m.pass_at_1||0)}%;background:${r.color}"></div></div><div class="bar-value">${pct(m.pass_at_1)}</div><div class="track"><div class="fill" style="width:${100*(m.pass_at_16||0)}%;background:${r.color}"></div></div><div class="bar-value">${pct(m.pass_at_16)}</div></div>`}).join("");
    document.querySelector("#results").innerHTML=data.checkpoints.map(r=>{const m=r.metrics;const f=r.filter;return `<tr><td><div class="model"><i class="dot" style="background:${r.color}"></i><div>${esc(r.label)}<small>${esc(r.detail)}</small></div></div></td><td><span class="status ${r.state}">${r.state==="complete"?"complete":`${r.shards_complete}/${r.shards_total} shards`}</span></td><td>${m?pct(m.pass_at_1):"—"}</td><td>${m?pct(m.pass_at_16):"—"}</td><td>${m?pct(m.format_rate):"—"}</td><td>${m?num(m.evaluated_prompts):"—"}</td><td>${m?num(m.skipped_overlong):"—"}</td><td>${f?`${num(f.rows)}${f.state==="pending"?" (pending write)":""}`:"—"}</td></tr>`}).join("");
    const cf=data.common_filter;const cfStatus=document.querySelector("#common-filter-status");cfStatus.textContent=cf?"complete":"pending";cfStatus.className=`status ${cf?"complete":""}`;document.querySelector("#common-filter-detail").textContent=cf?`${num(cf.rows)} prompts · SHA-256 ${cf.sha256}`:"waiting for offline filtering";
    document.querySelector("#validation-status").textContent=`${a.heldout_pretrain_complete} / ${a.heldout_pretrain_total} complete`;document.querySelector("#validation-status").className=`status ${a.heldout_pretrain_complete===a.heldout_pretrain_total?"complete":""}`;
    document.querySelector("#validation-results").innerHTML=data.checkpoints.map(r=>{const h=r.heldout_pretrain||{};const m=h.metrics;return `<tr><td><div class="model"><i class="dot" style="background:${r.color}"></i><div>${esc(r.label)}</div></div></td><td>${m?loss(m.heldout_pretrain_loss):"pending"}</td><td>${m?m.heldout_pretrain_perplexity.toFixed(4):"—"}</td><td>${m?pct(m.heldout_pretrain_token_accuracy):"—"}</td><td>${m?num(m.heldout_pretrain_target_tokens):"—"}</td></tr>`}).join("");
    const trainingSeries=[];const trainingRows=[];data.checkpoints.forEach(r=>{const stages=(data.training.experiments[r.key]||{}).stages||[];stages.forEach((s,index)=>{const label=`${r.label} · ${s.label}`;trainingSeries.push({label,color:r.color,points:s.series||[]});if(s.state==="complete")trainingRows.push({label,color:r.color,...s});});});
    plot(document.querySelector("#pt-chart"),trainingSeries,"pretrain_token_loss");plot(document.querySelector("#sft-chart"),trainingSeries,"sft_token_loss");
    document.querySelector("#training-legend").innerHTML=trainingSeries.map(s=>`<span><i style="background:${s.color}"></i>${esc(s.label)}</span>`).join("");document.querySelector("#wandb-project").href=data.training.wandb_project;
    document.querySelector("#training-results").innerHTML=trainingRows.map(s=>`<tr><td><div class="model"><i class="dot" style="background:${s.color}"></i><div><a href="${s.wandb_url}" target="_blank" rel="noreferrer">${esc(s.label)} ↗</a></div></div></td><td>${num(s.last_logged_step)} / ${num(s.target_step)}</td><td>${loss(s.last.loss)}</td><td>${loss(s.last.pretrain_token_loss)}</td><td>${loss(s.last.sft_token_loss)}</td><td>${loss(s.tail_10_mean.pretrain_token_loss)}</td><td>${loss(s.tail_10_mean.sft_token_loss)}</td></tr>`).join("");
    renderRl(data.rl);
    const banner=document.querySelector("#error"); banner.style.display=data.errors.length?"block":"none"; banner.textContent=data.errors.join("\n");
  }
  document.querySelector("#rl-smoothing").addEventListener("change",()=>{if(latestRlData)renderRl(latestRlData)});
  async function load(){try{const r=await fetch("/api/results",{cache:"no-store"});if(!r.ok)throw new Error(`HTTP ${r.status}`);const data=await r.json();render(data);if(data.state!=="complete"||data.aggregate.filters_complete<data.aggregate.filters_total||data.aggregate.heldout_pretrain_complete<data.aggregate.heldout_pretrain_total||data.aggregate.rl_complete<data.aggregate.rl_total)setTimeout(load,15000);}catch(e){const b=document.querySelector("#error");b.style.display="block";b.textContent=`Could not load results: ${e.message}`;setTimeout(load,15000)}}
  load();
</script></body></html>"""


dashboard_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "fastapi==0.116.1",
    "wandb==0.28.0",
)
eval_volume = modal.Volume.from_name(EVAL_VOLUME_NAME, create_if_missing=False)
pretrain_volume = modal.Volume.from_name(PRETRAIN_VOLUME_NAME, create_if_missing=False)
rl_checkpoint_volume = modal.Volume.from_name(
    RL_CHECKPOINT_VOLUME_NAME, create_if_missing=False
)
wandb_secret = modal.Secret.from_name("wandb-interleave-pt-rl")
app = modal.App(APP_NAME)

_cache_lock = threading.Lock()
_cache_value: dict[str, Any] | None = None
_cache_at = 0.0
CACHE_SECONDS = 12


def _snapshot(*, force: bool = False) -> dict[str, Any]:
    global _cache_at, _cache_value
    now = time.monotonic()
    with _cache_lock:
        if not force and _cache_value is not None and now - _cache_at < CACHE_SECONDS:
            return _cache_value
        eval_volume.reload()
        pretrain_volume.reload()
        rl_checkpoint_volume.reload()
        value = _collect_snapshot()
        _cache_value = value
        _cache_at = now
        return value


@app.function(
    image=dashboard_image,
    timeout=60,
    max_containers=2,
    scaledown_window=5 * 60,
    secrets=[wandb_secret],
    volumes={
        str(EVAL_MOUNT): eval_volume,
        str(PRETRAIN_MOUNT): pretrain_volume,
        str(RL_CHECKPOINT_MOUNT): rl_checkpoint_volume,
    },
)
@modal.asgi_app()
def web():
    from fastapi import FastAPI, Query
    from fastapi.responses import HTMLResponse, JSONResponse
    from starlette.middleware.gzip import GZipMiddleware

    api = FastAPI(
        title="Context-2048 checkpoint evaluation",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    api.add_middleware(GZipMiddleware, minimum_size=1_000)

    @api.get("/", response_class=HTMLResponse)
    def dashboard() -> HTMLResponse:
        return HTMLResponse(EVAL_HTML, headers={"Cache-Control": "public, max-age=60"})

    @api.get("/api/results", response_class=JSONResponse)
    def results(force: bool = Query(False)) -> JSONResponse:
        return JSONResponse(
            _snapshot(force=force),
            headers={
                "Cache-Control": "no-store",
                "Access-Control-Allow-Origin": "*",
            },
        )

    @api.get("/healthz", response_class=JSONResponse)
    def health() -> JSONResponse:
        return JSONResponse({"ok": True, "service": APP_NAME})

    return api
