"""Load and validate the context-2,048 PT/SFT/trace/RL experiment matrix.

The YAML files are the scientific source of truth.  This module resolves the
shared contract, applies stage defaults, checks the complete stage graph, and
returns a canonical SHA-256 that launch claims can bind to.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

import yaml


SCHEMA = "context2048-pt-sft-trace-rl-experiment-v1"
SHARED_SCHEMA = "context2048-pt-sft-trace-rl-shared-v1"
STAGE_KINDS = frozenset(
    {
        "pt_sft_mixed",
        "rl",
        "trace_supervised",
        "pt_sft_trace_mixed",
    }
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be an object: {path}")
    return value


def _deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in overlay.items():
        if (
            key in result
            and isinstance(result[key], Mapping)
            and isinstance(value, Mapping)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _require_number(value: Any, *, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    if not math.isfinite(float(value)):
        raise ValueError(f"{label} must be finite")
    return float(value)


def _validate_shared(shared: Mapping[str, Any]) -> None:
    if shared.get("schema") != SHARED_SCHEMA:
        raise ValueError("shared contract schema drifted")
    model = shared.get("model", {})
    if model.get("context_length") != 2_048 or model.get("vocab_size") != 85:
        raise ValueError("model must use context 2048 and the 85-token vocabulary")
    tokenizer = shared.get("tokenizer", {})
    expected_ids = {
        "<bos>": 0,
        "<eos>": 1,
        "<unk>": 2,
        "<T>": 81,
        "</T>": 82,
        "<sep>": 83,
        "<call_env>": 84,
    }
    if tokenizer.get("special_token_ids") != expected_ids:
        raise ValueError("complete special-token ID contract drifted")
    if tokenizer.get("exactly_one_bos") is not True:
        raise ValueError("exactly-one-BOS must be enabled")
    precision = shared.get("precision", {})
    expected_precision = {
        "master_parameters": "float32",
        "adam_moments": "float32",
        "forward_backward": "bfloat16",
        "gradient_reduction": "float32",
        "resumable_checkpoint": "float32",
        "hf_export": "float32",
        "inference_in_memory": "bfloat16",
    }
    if precision != expected_precision:
        raise ValueError("FP32-master/BF16-compute precision contract drifted")
    sft = shared.get("sft", {})
    if sft.get("rows") != 77_717 or sft.get("copies") != 3:
        raise ValueError("SFT×3 identity drifted")
    if sft.get("mixing") != "uniform_stable_binary_placement_without_replacement":
        raise ValueError("SFT must be uniformly interleaved with PT")
    rl = shared.get("rl", {})
    if rl.get("learning_rate") != 1e-5 or rl.get("low_var_kl_coefficient") != 0.001:
        raise ValueError("RL learning-rate/KL contract drifted")
    if (rl.get("prompts_per_update"), rl.get("samples_per_prompt")) != (256, 8):
        raise ValueError("RL batch geometry drifted")
    parquet = rl.get("train_parquet", {})
    if parquet.get("rows") != 28_419 or not SHA256_RE.fullmatch(
        str(parquet.get("sha256", ""))
    ):
        raise ValueError("RL parquet identity is incomplete")


def _resolve_stages(shared: Mapping[str, Any], raw_stages: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_stages, list) or not raw_stages:
        raise ValueError("experiment stages must be a nonempty list")
    defaults = shared.get("stage_defaults", {})
    if not isinstance(defaults, Mapping):
        raise ValueError("shared stage defaults are missing")
    stages: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_stages):
        if not isinstance(raw, Mapping):
            raise ValueError(f"stage {index} must be an object")
        kind = str(raw.get("kind", ""))
        if kind not in STAGE_KINDS:
            raise ValueError(f"stage {index} has unsupported kind {kind!r}")
        default = defaults.get(kind)
        if not isinstance(default, Mapping):
            raise ValueError(f"shared defaults are missing for {kind}")
        stages.append(_deep_merge(default, raw))
    return stages


def _validate_graph(config: Mapping[str, Any]) -> None:
    experiment = config.get("experiment", {})
    key = str(experiment.get("key", ""))
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", key):
        raise ValueError("experiment key is not a safe literal identifier")
    if experiment.get("matrix") != "5b_total_pt":
        raise ValueError("this launch wave must belong to the 5B matrix")
    if experiment.get("total_pt_target_tokens") != 5_000_000_000:
        raise ValueError("5B matrix must contain exactly 5,000,000,000 PT targets")

    stages = config.get("stages")
    assert isinstance(stages, list)
    by_id: dict[str, dict[str, Any]] = {}
    pt_total = 0
    sft_total = 0
    rl_total = 0
    for index, stage in enumerate(stages):
        stage_id = str(stage.get("id", ""))
        if not re.fullmatch(r"[a-z][a-z0-9_]*", stage_id) or stage_id in by_id:
            raise ValueError(f"stage {index} has an invalid or duplicate id")
        model_from = stage.get("model_from")
        if model_from != "canonical_seeded_initialization" and model_from not in by_id:
            raise ValueError(
                f"{stage_id} model_from must name an earlier stage or canonical initialization"
            )
        traces_from = stage.get("traces_from")
        if traces_from is not None:
            if traces_from not in by_id or by_id[traces_from]["kind"] != "rl":
                raise ValueError(f"{stage_id} traces_from must name an earlier RL stage")
        kind = stage["kind"]
        if kind in {"pt_sft_mixed", "pt_sft_trace_mixed"}:
            tokens = int(stage.get("pt_target_tokens", 0))
            exposures = int(stage.get("sft_exposures", 0))
            if tokens <= 0 or exposures <= 0:
                raise ValueError(f"{stage_id} must contain both PT and ordinary SFT")
            pt_total += tokens
            sft_total += exposures
            schedule = stage.get("learning_rate", {})
            if (
                schedule.get("schedule") != "linear_warmup_then_cosine"
                or schedule.get("peak") != 1e-3
                or schedule.get("minimum") != 1e-5
            ):
                raise ValueError(f"{stage_id} PT learning-rate schedule drifted")
            if kind == "pt_sft_trace_mixed" and traces_from is None:
                raise ValueError(f"{stage_id} must identify its RL trace source")
        elif kind == "trace_supervised":
            if traces_from is None or stage.get("learning_rate") != 1e-5:
                raise ValueError(f"{stage_id} trace-training contract drifted")
            if stage.get("order") not in {"shuffled", "chronological"}:
                raise ValueError(f"{stage_id} trace order is invalid")
        elif kind == "rl":
            updates = int(stage.get("updates", 0))
            if updates not in {1_500, 3_000} or stage.get("learning_rate") != 1e-5:
                raise ValueError(f"{stage_id} RL update/LR contract drifted")
            rl_total += updates
        by_id[stage_id] = stage

    if pt_total != 5_000_000_000:
        raise ValueError(f"PT target total is {pt_total}, expected 5,000,000,000")
    if sft_total != 77_717 * 3:
        raise ValueError(f"ordinary SFT exposure total is {sft_total}, expected 233,151")
    if rl_total != 3_000:
        raise ValueError(f"RL update total is {rl_total}, expected 3,000")
    if stages[-1]["kind"] != "rl":
        raise ValueError("the final stage must be RL")


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    raw = _load_yaml(config_path)
    if raw.get("schema") != SCHEMA:
        raise ValueError(f"experiment schema drifted: {config_path}")
    extends = raw.get("extends")
    if not isinstance(extends, str) or not extends:
        raise ValueError("experiment must extend one shared YAML contract")
    shared_path = (config_path.parent / extends).resolve()
    shared = _load_yaml(shared_path)
    _validate_shared(shared)
    resolved = _deep_merge(
        {key: value for key, value in shared.items() if key != "stage_defaults"},
        {key: value for key, value in raw.items() if key != "extends"},
    )
    resolved["stages"] = _resolve_stages(shared, raw.get("stages"))
    resolved["config_source"] = {
        "experiment_yaml": f"{config_path.parent.name}/{config_path.name}",
        "experiment_yaml_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "shared_yaml": shared_path.name,
        "shared_yaml_sha256": hashlib.sha256(shared_path.read_bytes()).hexdigest(),
    }
    _validate_graph(resolved)
    resolved["resolved_config_sha256"] = canonical_sha256(resolved)
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="+")
    args = parser.parse_args()
    payload = [load_experiment_config(path) for path in args.configs]
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
