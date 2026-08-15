"""Evaluate four context-2048 FP32-master checkpoints with fixed pass@16 samples.

Evaluation, filtering, and RL launch are deliberately separate actions.  The
evaluation covers the complete balanced RL training parquet subject to the
production post-BOS 512-token prompt cap, generates 16 samples per admitted
prompt, and reports unbiased pass@1 and pass@16.  It uses the same native
context contract intended for RL: at most 512 prompt tokens including exactly
one explicit BOS, at most 1,536 model-generated tokens, and 2,048 total tokens.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal


APP_NAME = "chess-context2048-fp32-master-v13-pass16-eval"
VERSION = "context2048-fp32-master-v13-pass16-native2048-bos-v1-20260814"
RESULTS_VOLUME_NAME = "chess-rl-eval-results-r6"
RESULTS_MOUNT = Path("/results")
RESULTS_ROOT = RESULTS_MOUNT / VERSION
CHECKPOINT_MOUNT = Path("/pretrain-checkpoints")
DATA_MOUNT = Path("/data")
SOURCE_PARQUET = (
    DATA_MOUNT / "chess-rl-data/train_v4_dataset_balanced_multi_turn.parquet"
)
SOURCE_ROWS = 53_225
SOURCE_SHA256 = (
    "bcf131d88cd9916f6ec12a12f8e85901a05e0dabc5d5a1fdebd8ea0b0ae86c30"
)
COMMON_FILTER_FILENAME = (
    "train_v4_ctx2048_fp32masterv13_all4_intersection_"
    "mixed_outcome_1to15of16_multi_turn.parquet"
)

N_SAMPLES = 16
SHARD_COUNT = 4
PROMPT_CAP = 512
RESPONSE_BUDGET = 1_536
MAX_MODEL_LEN = 2_048
CONTEXT_MARGIN = 0
MAX_ENV_CALLS = 6
TEMPERATURE = 1.0
TOP_P = 1.0
RL_UPDATES = 1_500
RL_LR = "1e-5"
RL_PROJECT = "chess-47m-context2048-rl"
RL_GROUP = "all-four-context2048-checkpoints-filtered-lr1e5"
RL_APP_NAME = "chess-interleave-rl"
RL_FUNCTION_NAME = "train_hf"

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1] if modal.is_local() else Path("/opt/context2048_eval")
MILES_PACKAGE_LOCAL = PROJECT_ROOT / "miles/miles"
CHESS_RL_MILES_LOCAL = PROJECT_ROOT / "chess/rl/chess_rl_miles"
ONLINE_REWARD_SHA256 = (
    "1a6065c58f0cf8c775112815c90930a87bde205e484a78572f8a0e54eb2bc5c0"
)
ONLINE_MOVES_SHA256 = (
    "c4680a3e736f9cb7e2abaa5460e5f29f48c68603e924b68a7474fc3cf8c256a0"
)
ONLINE_ROLLOUT_SHA256 = (
    "8414927087039693df11525a827553db5f0b6b4a2660d18ac43e0f338bd376e9"
)


@dataclass(frozen=True)
class CheckpointSpec:
    key: str
    label: str
    subpath: str
    expected_fingerprint: str
    filtered_filename: str
    rl_run_name: str | None


CHECKPOINTS: dict[str, CheckpointSpec] = {
    "vocab81_then_sft3": CheckpointSpec(
        key="vocab81_then_sft3",
        label=(
            "81-token pretraining, deterministic expansion to 85 tokens, "
            "then SFT for 3 epochs"
        ),
        subpath=(
            "context2048_vocab_mixing_fp32_master_v13_20260813/"
            "vocab81_then_sft3/sft/final"
        ),
        expected_fingerprint=(
            "350e1eb7dd87e5fb0107437a3ccdb1dc42efdc034edd4cc0b502738c04de7270"
        ),
        filtered_filename=(
            "train_v4_ctx2048_fp32masterv13_vocab81pt_expand85_sft3_"
            "mixed_outcome_1to15of16_multi_turn.parquet"
        ),
        rl_run_name=(
            "ctx2048-fp32masterv13-vocab81pt-expand85-sft3-filtered-lr1e5-rl1500"
        ),
    ),
    "vocab85_then_sft3": CheckpointSpec(
        key="vocab85_then_sft3",
        label="85-token pretraining, then SFT for 3 epochs",
        subpath=(
            "context2048_vocab_mixing_fp32_master_v13_20260813/"
            "vocab85_then_sft3/sft/final"
        ),
        expected_fingerprint=(
            "0b286a1ad928c1efefb135cdd8d8bf28d867276e28a7dc682ade3684e6ee6c19"
        ),
        filtered_filename=(
            "train_v4_ctx2048_fp32masterv13_vocab85pt_sft3_"
            "mixed_outcome_1to15of16_multi_turn.parquet"
        ),
        rl_run_name="ctx2048-fp32masterv13-vocab85pt-sft3-filtered-lr1e5-rl1500",
    ),
    "mixed_sft1": CheckpointSpec(
        key="mixed_sft1",
        label=(
            "85-token uniformly shuffled mixed pretraining plus one "
            "independently placed SFT copy"
        ),
        subpath=(
            "context2048_vocab_mixing_fp32_master_v13_20260813/"
            "mixed_sft1/mixed/final"
        ),
        expected_fingerprint=(
            "e42a2ed9a5e2b0550c5e5e06ef48e4089ff046d4415d2b4c9c28af0745c0c139"
        ),
        filtered_filename=(
            "train_v4_ctx2048_fp32masterv13_mixed_sft1_"
            "mixed_outcome_1to15of16_multi_turn.parquet"
        ),
        rl_run_name="ctx2048-fp32masterv13-mixed-sft1-filtered-lr1e5-rl1500",
    ),
    "mixed_sft3": CheckpointSpec(
        key="mixed_sft3",
        label=(
            "85-token uniformly shuffled mixed pretraining plus three "
            "independently shuffled SFT copies"
        ),
        subpath=(
            "context2048_vocab_mixing_fp32_master_v13_20260813/"
            "mixed_sft3/mixed/final"
        ),
        expected_fingerprint=(
            "61193269be0afc01e310705fef7ed071ea8b224da83242db52594279edf32075"
        ),
        filtered_filename=(
            "train_v4_ctx2048_fp32masterv13_mixed_sft3_"
            "mixed_outcome_1to15of16_multi_turn.parquet"
        ),
        rl_run_name="ctx2048-fp32masterv13-mixed-sft3-filtered-lr1e5-rl1500",
    ),
}
RL_CHECKPOINTS = tuple(CHECKPOINTS)


image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install("curl", "git")
    .pip_install(
        "vllm==0.8.5",
        "transformers==4.51.3",
        "tokenizers==0.21.4",
        "huggingface_hub==0.36.2",
        "pandas==2.3.3",
        "pyarrow==23.0.1",
        "numpy==2.2.6",
        "chess==1.11.2",
    )
    .env({"PYTHONPATH": "/opt/context2048_eval"})
)
if modal.is_local():
    image = image.add_local_dir(
        str(MILES_PACKAGE_LOCAL),
        remote_path="/opt/context2048_eval/miles",
        copy=True,
    ).add_local_dir(
        str(CHESS_RL_MILES_LOCAL),
        remote_path="/opt/context2048_eval/chess_rl_miles",
        copy=True,
    )

results_volume = modal.Volume.from_name(
    RESULTS_VOLUME_NAME, create_if_missing=True
)
checkpoint_volume = modal.Volume.from_name(
    "rl-reasoning-checkpoints", create_if_missing=False
)
data_volume = modal.Volume.from_name(
    "chess-rl-miles-data", create_if_missing=False
)

app = modal.App(
    APP_NAME,
    image=image,
    secrets=[modal.Secret.from_name("wandb-interleave-pt-rl")],
)


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


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def checkpoint_fingerprint(checkpoint: str | Path) -> str:
    root = Path(checkpoint).resolve(strict=True)
    required = (root / "config.json", root / "interleaved_training_state.json")
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
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 << 20), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def shard_bounds(shard_id: int) -> tuple[int, int]:
    if not 0 <= shard_id < SHARD_COUNT:
        raise ValueError(f"invalid shard_id: {shard_id}")
    return (
        SOURCE_ROWS * shard_id // SHARD_COUNT,
        SOURCE_ROWS * (shard_id + 1) // SHARD_COUNT,
    )


def checkpoint_path(spec: CheckpointSpec) -> Path:
    return CHECKPOINT_MOUNT / spec.subpath


def result_root(key: str, shard_id: int, max_prompts: int = 0) -> Path:
    suffix = f"shard-{shard_id:02d}"
    if max_prompts:
        suffix += f"-first-{max_prompts}"
    return RESULTS_ROOT / key / "n16" / suffix


def _validated_checkpoint(key: str) -> tuple[CheckpointSpec, Path, str]:
    spec = CHECKPOINTS[key]
    path = checkpoint_path(spec)
    observed = checkpoint_fingerprint(path)
    if not spec.expected_fingerprint:
        raise RuntimeError(f"checkpoint fingerprint is not pinned for {key}")
    if observed != spec.expected_fingerprint:
        raise RuntimeError(
            f"checkpoint fingerprint drifted for {key}: "
            f"{observed} != {spec.expected_fingerprint}"
        )
    return spec, path, observed


@app.function(
    cpu=16.0,
    memory=64 * 1024,
    timeout=60 * 60,
    volumes={
        str(CHECKPOINT_MOUNT): checkpoint_volume,
        str(DATA_MOUNT): data_volume,
    },
)
def inspect_inputs() -> dict[str, Any]:
    from transformers import AutoTokenizer

    checkpoint_volume.reload()
    data_volume.reload()
    source_sha256 = sha256_file(SOURCE_PARQUET)
    import pyarrow.parquet as pq

    source_rows = pq.ParquetFile(SOURCE_PARQUET).metadata.num_rows
    checkpoints: dict[str, Any] = {}
    for key, item in CHECKPOINTS.items():
        path = checkpoint_path(item)
        config = json.loads((path / "config.json").read_text())
        state = json.loads(
            (path / "interleaved_training_state.json").read_text()
        )
        tokenizer = AutoTokenizer.from_pretrained(
            str(path), trust_remote_code=True
        )
        checkpoints[key] = {
            "path": str(path),
            "fingerprint": checkpoint_fingerprint(path),
            "max_position_embeddings": config.get("max_position_embeddings"),
            "vocab_size": config.get("vocab_size"),
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "call_env_token_id": tokenizer.convert_tokens_to_ids("<call_env>"),
            "global_step": state.get("global_step"),
        }
    return {
        "version": VERSION,
        "source": {
            "path": str(SOURCE_PARQUET),
            "rows": source_rows,
            "sha256": source_sha256,
            "valid": source_rows == SOURCE_ROWS and source_sha256 == SOURCE_SHA256,
        },
        "checkpoints": checkpoints,
    }


class _State:
    __slots__ = (
        "source_row_index",
        "local_row_index",
        "sample_index",
        "prompt_len",
        "ctx",
        "model_tokens",
        "env_calls",
        "env_replies",
        "finished",
    )

    def __init__(
        self,
        *,
        source_row_index: int,
        local_row_index: int,
        sample_index: int,
        prompt_ids: list[int],
        env_replies: list[str],
    ) -> None:
        self.source_row_index = source_row_index
        self.local_row_index = local_row_index
        self.sample_index = sample_index
        self.prompt_len = len(prompt_ids)
        self.ctx = list(prompt_ids)
        self.model_tokens = 0
        self.env_calls = 0
        self.env_replies = list(env_replies)
        self.finished = False

    @property
    def response_ids(self) -> list[int]:
        return self.ctx[self.prompt_len :]


def frame_prompt_ids(
    unframed_ids: list[int], *, bos_id: int
) -> list[int] | None:
    """Match the production rollout's post-BOS 512-token admission rule."""

    raw = [int(token_id) for token_id in unframed_ids]
    if int(bos_id) in raw:
        raise RuntimeError(
            "source prompt unexpectedly contains BOS while tokenized with "
            "add_special_tokens=False"
        )
    framed = [int(bos_id), *raw]
    if len(framed) > PROMPT_CAP:
        return None
    if framed[0] != int(bos_id) or framed.count(int(bos_id)) != 1:
        raise AssertionError("exactly-one-leading-BOS construction failed")
    return framed


def to_python_value(value: Any) -> Any:
    """Match PyArrow ``to_pylist`` values used by the production data loader."""

    if isinstance(value, dict):
        return {key: to_python_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_python_value(item) for item in value]
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return to_python_value(tolist())
    return value


def _pass_at_k(histogram: dict[int, int], *, n: int, k: int) -> float:
    prompts = sum(histogram.values())
    if prompts <= 0:
        raise ValueError("empty pass@k histogram")
    total = 0.0
    for wins, count in histogram.items():
        if wins <= 0:
            value = 0.0
        elif n - wins < k:
            value = 1.0
        else:
            value = 1.0 - math.comb(n - wins, k) / math.comb(n, k)
        total += value * count
    return total / prompts


def common_mixed_outcome_indices(
    wins_by_checkpoint: dict[str, dict[int, int]],
) -> list[int]:
    """Return the common 1--15/16 cohort for all four checkpoints."""

    if set(wins_by_checkpoint) != set(CHECKPOINTS):
        raise ValueError("common filter must include exactly all checkpoints")
    evaluated_sets = [set(rows) for rows in wins_by_checkpoint.values()]
    if not evaluated_sets or any(rows != evaluated_sets[0] for rows in evaluated_sets[1:]):
        raise ValueError("checkpoint evaluation cohorts differ")
    return sorted(
        source_index
        for source_index in evaluated_sets[0]
        if all(
            0 < wins_by_checkpoint[key][source_index] < N_SAMPLES
            for key in CHECKPOINTS
        )
    )


@app.function(
    gpu="H200",
    cpu=16.0,
    memory=64 * 1024,
    timeout=60 * 60 * 14,
    retries=modal.Retries(initial_delay=10.0, max_retries=2),
    max_containers=16,
    volumes={
        str(RESULTS_MOUNT): results_volume,
        str(CHECKPOINT_MOUNT): checkpoint_volume,
        str(DATA_MOUNT): data_volume,
    },
)
def eval_shard(
    key: str,
    shard_id: int,
    max_prompts: int = 0,
) -> dict[str, Any]:
    from types import SimpleNamespace

    import pandas as pd
    from chess_rl_miles.reward import _score_sample
    from miles.utils.types import Sample
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    checkpoint_volume.reload()
    data_volume.reload()
    results_volume.reload()
    spec, model_path, fingerprint = _validated_checkpoint(key)
    if sha256_file(SOURCE_PARQUET) != SOURCE_SHA256:
        raise RuntimeError("balanced RL source parquet drifted")
    output_root = result_root(key, shard_id, max_prompts)
    marker = output_root / "success.json"
    if marker.is_file():
        return json.loads(marker.read_text())

    start, stop = shard_bounds(shard_id)
    frame = pd.read_parquet(SOURCE_PARQUET).iloc[start:stop].copy()
    if max_prompts:
        frame = frame.iloc[:max_prompts].copy()
    frame.reset_index(drop=True, inplace=True)

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), trust_remote_code=True
    )
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    call_env_id = tokenizer.convert_tokens_to_ids("<call_env>")
    if bos_id is None:
        raise RuntimeError("tokenizer has no bos_token_id")
    if call_env_id is None or call_env_id == tokenizer.unk_token_id:
        raise RuntimeError("<call_env> is missing from the tokenizer")

    config = json.loads((model_path / "config.json").read_text())
    model_context = int(config.get("max_position_embeddings", -1))
    if model_context != MAX_MODEL_LEN:
        raise RuntimeError(
            f"{key} model context {model_context} != {MAX_MODEL_LEN}"
        )
    effective_context_limit = MAX_MODEL_LEN - CONTEXT_MARGIN

    states: list[_State] = []
    skipped_overlong: list[int] = []
    for local_index, row in enumerate(frame.itertuples(index=False)):
        source_index = start + local_index
        unframed = tokenizer(
            row.prompt, add_special_tokens=False
        ).input_ids
        prompt_ids = frame_prompt_ids(unframed, bos_id=int(bos_id))
        if prompt_ids is None:
            skipped_overlong.append(source_index)
            continue
        raw_extra_info = to_python_value(row.extra_info or {})
        raw_replies = raw_extra_info.get("env_replies", [])
        replies = [
            str(reply)
            for reply in (
                raw_replies if raw_replies is not None else []
            )
        ]
        for sample_index in range(N_SAMPLES):
            states.append(
                _State(
                    source_row_index=source_index,
                    local_row_index=local_index,
                    sample_index=sample_index,
                    prompt_ids=prompt_ids,
                    env_replies=replies,
                )
            )
    print(
        f"[context2048-eval] {key} shard={shard_id}: "
        f"{len(frame)} source rows, {len(skipped_overlong)} overlong, "
        f"{len(states)} trajectories",
        flush=True,
    )

    llm = LLM(
        model=str(model_path),
        trust_remote_code=True,
        gpu_memory_utilization=0.60,
        max_model_len=MAX_MODEL_LEN,
        seed=0,
        enforce_eager=False,
        dtype="bfloat16",
    )
    reply_token_cache: dict[str, list[int]] = {}
    generation_round = 0
    while True:
        active = [state for state in states if not state.finished]
        if not active:
            break
        generation_round += 1
        prompts: list[Any] = []
        parameters: list[Any] = []
        for state in active:
            remaining_model = RESPONSE_BUDGET - state.model_tokens
            remaining_context = effective_context_limit - len(state.ctx)
            budget = min(remaining_model, remaining_context)
            if budget <= 0:
                state.finished = True
                continue
            prompts.append(TokensPrompt(prompt_token_ids=state.ctx))
            parameters.append(
                SamplingParams(
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    max_tokens=budget,
                    stop_token_ids=[call_env_id],
                )
            )
        generating = [state for state in active if not state.finished]
        if not generating:
            continue
        outputs = llm.generate(prompts, parameters, use_tqdm=False)
        for state, output in zip(generating, outputs, strict=True):
            completion = output.outputs[0]
            tokens = list(completion.token_ids)
            if (
                completion.finish_reason == "stop"
                and completion.stop_reason == call_env_id
            ):
                tokens.append(call_env_id)
            remaining = RESPONSE_BUDGET - state.model_tokens
            tokens = tokens[:remaining]
            if not tokens:
                state.finished = True
                continue
            if call_env_id in tokens:
                kept = tokens[: tokens.index(call_env_id) + 1]
                state.ctx.extend(kept)
                state.model_tokens += len(kept)
                state.env_calls += 1
                reply = state.env_replies.pop(0) if state.env_replies else None
                if reply:
                    if reply not in reply_token_cache:
                        reply_token_cache[reply] = tokenizer(
                            reply, add_special_tokens=False
                        ).input_ids
                    context_remaining = effective_context_limit - len(state.ctx)
                    reply_ids = reply_token_cache[reply][
                        : max(context_remaining, 0)
                    ]
                    state.ctx.extend(reply_ids)
                    if not reply_ids:
                        state.finished = True
                else:
                    state.finished = True
                if state.env_calls >= MAX_ENV_CALLS:
                    state.finished = True
                if len(state.ctx) >= effective_context_limit:
                    state.finished = True
            else:
                state.ctx.extend(tokens)
                state.model_tokens += len(tokens)
                state.finished = True
        print(
            f"[context2048-eval] {key} shard={shard_id} "
            f"round={generation_round} active={len(active)}",
            flush=True,
        )

    texts = [
        tokenizer.decode(state.response_ids, skip_special_tokens=False)
        for state in states
    ]
    reward_args = SimpleNamespace(
        chess_reward_model_type=None,
        chess_multiturn=None,
        chess_difficulty_threshold=1_500.0,
    )
    scored = []
    for state, text in zip(states, texts, strict=True):
        row = frame.iloc[state.local_row_index]
        scored.append(
            _score_sample(
                reward_args,
                Sample(
                    prompt=str(row["prompt"]),
                    response=text,
                    response_length=len(state.response_ids),
                    label=to_python_value(row["reward_model"]),
                    metadata=to_python_value(row["extra_info"] or {}),
                ),
            )
        )

    output_root.mkdir(parents=True, exist_ok=True)
    generations_path = output_root / "generations.jsonl.gz"
    per_prompt: dict[int, dict[str, int]] = {}
    with gzip.open(generations_path, "wt", encoding="utf-8") as handle:
        for state, text, result in zip(states, texts, scored, strict=True):
            score = float(result["score"])
            if score not in (0.0, 1.0):
                raise ValueError(f"non-binary reward: {score}")
            format_ok = (
                "<call_env>" in text
                and "</T>" in text
                and 0 <= text.find("</T>") < text.find("<call_env>")
            )
            entry = per_prompt.setdefault(
                state.source_row_index,
                {"n": 0, "wins": 0, "format": 0, "slot_mask": 0},
            )
            bit = 1 << state.sample_index
            if entry["slot_mask"] & bit:
                raise ValueError("duplicate sample slot")
            entry["slot_mask"] |= bit
            entry["n"] += 1
            entry["wins"] += int(score)
            entry["format"] += int(format_ok)
            handle.write(
                json.dumps(
                    {
                        "source_row_index": state.source_row_index,
                        "sample_slot": state.sample_index,
                        "score": score,
                        "format_ok": bool(format_ok),
                        "model_tokens": state.model_tokens,
                        "env_calls": state.env_calls,
                        "response": text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    full_mask = (1 << N_SAMPLES) - 1
    if any(
        entry["n"] != N_SAMPLES or entry["slot_mask"] != full_mask
        for entry in per_prompt.values()
    ):
        raise ValueError("incomplete per-prompt sample coverage")
    histogram: dict[int, int] = {}
    format_hits = 0
    for entry in per_prompt.values():
        wins = entry["wins"]
        histogram[wins] = histogram.get(wins, 0) + 1
        format_hits += entry["format"]
    trajectories = len(states)
    summary_core = {
        "schema": "context2048-pass16-shard-summary-v1",
        "version": VERSION,
        "checkpoint": key,
        "checkpoint_label": spec.label,
        "checkpoint_path": str(model_path),
        "checkpoint_fingerprint": fingerprint,
        "shard_id": shard_id,
        "source_row_start": start,
        "source_row_stop": start + len(frame),
        "source_sha256": SOURCE_SHA256,
        "source_rows_seen": len(frame),
        "evaluated_prompts": len(per_prompt),
        "skipped_overlong": skipped_overlong,
        "trajectories": trajectories,
        "n_samples": N_SAMPLES,
        "pass_at_1": _pass_at_k(histogram, n=N_SAMPLES, k=1),
        "pass_at_16": _pass_at_k(histogram, n=N_SAMPLES, k=16),
        "format_rate": format_hits / trajectories,
        "mixed_outcome_prompts": sum(
            count for wins, count in histogram.items() if 0 < wins < N_SAMPLES
        ),
        "wins_histogram": {
            str(wins): count for wins, count in sorted(histogram.items())
        },
        "generation": {
            "engine": "vllm-0.8.5-bare",
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "prompt_cap_including_bos": PROMPT_CAP,
            "dataset_prefilter_cap_excluding_bos": PROMPT_CAP - 1,
            "response_budget": RESPONSE_BUDGET,
            "model_context": MAX_MODEL_LEN,
            "context_margin": CONTEXT_MARGIN,
            "max_env_calls": MAX_ENV_CALLS,
            "bos_prepended_exactly_once_by_evaluator": True,
            "bos_token_id": int(bos_id),
            "eos_token_id": None if eos_id is None else int(eos_id),
            "online_reward_sha256": ONLINE_REWARD_SHA256,
            "online_moves_sha256": ONLINE_MOVES_SHA256,
            "online_rollout_sha256": ONLINE_ROLLOUT_SHA256,
            "scorer": "chess_rl_miles.reward._score_sample",
        },
        "generations": {
            "path": str(generations_path),
            "bytes": generations_path.stat().st_size,
            "sha256": sha256_file(generations_path),
        },
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    summary = {
        **summary_core,
        "summary_sha256": canonical_sha256(summary_core),
    }
    atomic_json(marker, summary)
    results_volume.commit()
    return summary


def _validate_shard_summary(
    summary: dict[str, Any], *, key: str, shard_id: int
) -> None:
    recorded = summary.get("summary_sha256")
    core = {name: value for name, value in summary.items() if name != "summary_sha256"}
    if recorded != canonical_sha256(core):
        raise ValueError("shard summary self hash drifted")
    expected = CHECKPOINTS[key]
    if (
        summary.get("version") != VERSION
        or summary.get("checkpoint") != key
        or summary.get("checkpoint_fingerprint") != expected.expected_fingerprint
        or summary.get("shard_id") != shard_id
        or summary.get("n_samples") != N_SAMPLES
        or summary.get("source_sha256") != SOURCE_SHA256
    ):
        raise ValueError(f"shard summary identity drifted for {key}/{shard_id}")


def _checkpoint_outcomes(
    key: str,
) -> tuple[dict[int, int], set[int], set[int], list[dict[str, Any]]]:
    wins_by_row: dict[int, int] = {}
    evaluated_rows: set[int] = set()
    skipped_rows: set[int] = set()
    shard_summaries: list[dict[str, Any]] = []
    for shard_id in range(SHARD_COUNT):
        root = result_root(key, shard_id)
        summary = json.loads((root / "success.json").read_text())
        _validate_shard_summary(summary, key=key, shard_id=shard_id)
        generations = Path(summary["generations"]["path"])
        if (
            not generations.is_file()
            or generations.stat().st_size != summary["generations"]["bytes"]
            or sha256_file(generations) != summary["generations"]["sha256"]
        ):
            raise ValueError("generation artifact drifted")
        slots: dict[int, set[int]] = {}
        with gzip.open(generations, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                source_index = int(row["source_row_index"])
                slot = int(row["sample_slot"])
                score = float(row["score"])
                if not 0 <= source_index < SOURCE_ROWS:
                    raise ValueError("generation source index is out of range")
                if not 0 <= slot < N_SAMPLES or score not in (0.0, 1.0):
                    raise ValueError("generation slot or score is invalid")
                if slot in slots.setdefault(source_index, set()):
                    raise ValueError("duplicate generation slot")
                slots[source_index].add(slot)
                wins_by_row[source_index] = (
                    wins_by_row.get(source_index, 0) + int(score)
                )
        if any(value != set(range(N_SAMPLES)) for value in slots.values()):
            raise ValueError("generation group lacks exact 0..15 slots")
        evaluated_rows.update(slots)
        skipped_rows.update(map(int, summary["skipped_overlong"]))
        shard_summaries.append(summary)
    if evaluated_rows & skipped_rows:
        raise ValueError("a source row is both evaluated and skipped")
    if evaluated_rows | skipped_rows != set(range(SOURCE_ROWS)):
        raise ValueError("evaluation does not cover the full source parquet")
    if not evaluated_rows:
        raise ValueError("post-BOS prompt-cap cohort is empty")
    return wins_by_row, evaluated_rows, skipped_rows, shard_summaries


@app.function(
    cpu=16.0,
    memory=64 * 1024,
    timeout=2 * 60 * 60,
    max_containers=4,
    volumes={
        str(RESULTS_MOUNT): results_volume,
        str(DATA_MOUNT): data_volume,
    },
)
def build_filtered_dataset(key: str) -> dict[str, Any]:
    import pandas as pd

    results_volume.reload()
    data_volume.reload()
    spec = CHECKPOINTS[key]
    filter_root = RESULTS_ROOT / key / "filter"
    success_path = filter_root / "success.json"
    if success_path.is_file():
        existing = json.loads(success_path.read_text())
        target = Path(existing["filtered_parquet"]["path"])
        if (
            target.is_file()
            and target.stat().st_size == existing["filtered_parquet"]["bytes"]
            and sha256_file(target) == existing["filtered_parquet"]["sha256"]
        ):
            return existing
        raise ValueError("existing filter success marker does not match the parquet")

    wins_by_row: dict[int, int] = {}
    evaluated_rows: set[int] = set()
    skipped_rows: set[int] = set()
    shard_summaries: list[dict[str, Any]] = []
    for shard_id in range(SHARD_COUNT):
        root = result_root(key, shard_id)
        summary = json.loads((root / "success.json").read_text())
        _validate_shard_summary(summary, key=key, shard_id=shard_id)
        generations = Path(summary["generations"]["path"])
        if (
            not generations.is_file()
            or generations.stat().st_size != summary["generations"]["bytes"]
            or sha256_file(generations) != summary["generations"]["sha256"]
        ):
            raise ValueError("generation artifact drifted")
        slots: dict[int, set[int]] = {}
        with gzip.open(generations, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                source_index = int(row["source_row_index"])
                slot = int(row["sample_slot"])
                score = float(row["score"])
                if not 0 <= source_index < SOURCE_ROWS:
                    raise ValueError("generation source index is out of range")
                if not 0 <= slot < N_SAMPLES or score not in (0.0, 1.0):
                    raise ValueError("generation slot or score is invalid")
                if slot in slots.setdefault(source_index, set()):
                    raise ValueError("duplicate generation slot")
                slots[source_index].add(slot)
                wins_by_row[source_index] = wins_by_row.get(source_index, 0) + int(score)
        if any(value != set(range(N_SAMPLES)) for value in slots.values()):
            raise ValueError("generation group lacks exact 0..15 slots")
        evaluated_rows.update(slots)
        skipped_rows.update(map(int, summary["skipped_overlong"]))
        shard_summaries.append(summary)

    if evaluated_rows & skipped_rows:
        raise ValueError("a source row is both evaluated and skipped")
    if evaluated_rows | skipped_rows != set(range(SOURCE_ROWS)):
        raise ValueError("evaluation does not cover the full source parquet")
    if not evaluated_rows:
        raise ValueError("post-BOS prompt-cap cohort is empty")
    selected = sorted(
        index for index, wins in wins_by_row.items() if 0 < wins < N_SAMPLES
    )
    source = pd.read_parquet(SOURCE_PARQUET)
    if len(source) != SOURCE_ROWS or sha256_file(SOURCE_PARQUET) != SOURCE_SHA256:
        raise ValueError("balanced RL source parquet drifted during filtering")
    filtered = source.iloc[selected].copy().reset_index(drop=True)
    output = SOURCE_PARQUET.parent / spec.filtered_filename
    if output.exists():
        raise FileExistsError(
            f"refusing to replace pre-existing filtered parquet: {output}"
        )
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    filtered.to_parquet(temporary, index=False)
    os.replace(temporary, output)
    record = {
        "path": str(output),
        "rows": len(filtered),
        "bytes": output.stat().st_size,
        "sha256": sha256_file(output),
    }
    summary_core = {
        "schema": "context2048-mixed-outcome-filter-v1",
        "version": VERSION,
        "checkpoint": key,
        "checkpoint_fingerprint": spec.expected_fingerprint,
        "source": {
            "path": str(SOURCE_PARQUET),
            "rows": SOURCE_ROWS,
            "sha256": SOURCE_SHA256,
        },
        "rule": "1 <= success_count <= 15 from exactly 16 samples",
        "evaluated_rows": len(evaluated_rows),
        "skipped_overlong_rows": len(skipped_rows),
        "filtered_parquet": record,
        "shard_summary_sha256": [
            summary["summary_sha256"] for summary in shard_summaries
        ],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    summary = {
        **summary_core,
        "filter_sha256": canonical_sha256(summary_core),
    }
    atomic_json(success_path, summary)
    data_volume.commit()
    results_volume.commit()
    return summary


@app.function(
    cpu=16.0,
    memory=64 * 1024,
    timeout=4 * 60 * 60,
    max_containers=1,
    volumes={
        str(RESULTS_MOUNT): results_volume,
        str(DATA_MOUNT): data_volume,
    },
)
def build_common_filtered_dataset() -> dict[str, Any]:
    """Build one identical intersection filter for all four RL comparisons."""

    import pandas as pd

    results_volume.reload()
    data_volume.reload()
    filter_root = RESULTS_ROOT / "common_filter"
    success_path = filter_root / "success.json"
    if success_path.is_file():
        existing = json.loads(success_path.read_text())
        target = Path(existing["filtered_parquet"]["path"])
        if (
            target.is_file()
            and target.stat().st_size == existing["filtered_parquet"]["bytes"]
            and sha256_file(target) == existing["filtered_parquet"]["sha256"]
        ):
            return existing
        raise ValueError(
            "existing common-filter marker does not match its parquet"
        )

    wins_by_checkpoint: dict[str, dict[int, int]] = {}
    cohort: set[int] | None = None
    skipped: set[int] | None = None
    checkpoint_records: dict[str, Any] = {}
    for key in CHECKPOINTS:
        wins, evaluated_rows, skipped_rows, summaries = _checkpoint_outcomes(key)
        if cohort is None:
            cohort = evaluated_rows
            skipped = skipped_rows
        elif evaluated_rows != cohort or skipped_rows != skipped:
            raise ValueError("checkpoint prompt-cap cohorts differ")
        wins_by_checkpoint[key] = wins
        histogram: dict[int, int] = {}
        for value in wins.values():
            histogram[value] = histogram.get(value, 0) + 1
        checkpoint_records[key] = {
            "checkpoint_fingerprint": CHECKPOINTS[key].expected_fingerprint,
            "mixed_outcome_prompts": sum(
                count
                for value, count in histogram.items()
                if 0 < value < N_SAMPLES
            ),
            "wins_histogram": {
                str(value): count
                for value, count in sorted(histogram.items())
            },
            "shard_summary_sha256": [
                summary["summary_sha256"] for summary in summaries
            ],
        }
    assert cohort is not None and skipped is not None
    selected = common_mixed_outcome_indices(wins_by_checkpoint)
    source = pd.read_parquet(SOURCE_PARQUET)
    if len(source) != SOURCE_ROWS or sha256_file(SOURCE_PARQUET) != SOURCE_SHA256:
        raise ValueError("balanced RL source parquet drifted during filtering")
    filtered = source.iloc[selected].copy().reset_index(drop=True)
    output = SOURCE_PARQUET.parent / COMMON_FILTER_FILENAME
    if output.exists():
        raise FileExistsError(
            f"refusing to replace pre-existing common filtered parquet: {output}"
        )
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    filtered.to_parquet(temporary, index=False)
    os.replace(temporary, output)
    record = {
        "path": str(output),
        "rows": len(filtered),
        "bytes": output.stat().st_size,
        "sha256": sha256_file(output),
    }
    summary_core = {
        "schema": "context2048-common-mixed-outcome-filter-v1",
        "version": VERSION,
        "source": {
            "path": str(SOURCE_PARQUET),
            "rows": SOURCE_ROWS,
            "sha256": SOURCE_SHA256,
        },
        "rule": (
            "retain a source prompt iff every one of the four checkpoints "
            "has 1 <= success_count <= 15 from exactly 16 fixed samples"
        ),
        "comparison_contract": (
            "all four RL runs must use this exact parquet and SHA-256"
        ),
        "n_samples_per_checkpoint": N_SAMPLES,
        "evaluated_rows": len(cohort),
        "skipped_overlong_rows": len(skipped),
        "selected_source_row_indices_sha256": canonical_sha256(selected),
        "filtered_parquet": record,
        "checkpoints": checkpoint_records,
        "scorer": {
            "entrypoint": "chess_rl_miles.reward._score_sample",
            "reward_sha256": ONLINE_REWARD_SHA256,
            "moves_sha256": ONLINE_MOVES_SHA256,
            "rollout_sha256": ONLINE_ROLLOUT_SHA256,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    summary = {
        **summary_core,
        "filter_sha256": canonical_sha256(summary_core),
    }
    atomic_json(success_path, summary)
    data_volume.commit()
    results_volume.commit()
    return summary


def rl_launch_kwargs(key: str, filter_summary: dict[str, Any]) -> dict[str, Any]:
    spec = CHECKPOINTS[key]
    if spec.rl_run_name is None:
        raise ValueError(f"{key} has no RL run name")
    filtered = filter_summary["filtered_parquet"]
    return {
        "hf_checkpoint": str(checkpoint_path(spec)),
        "run_name": spec.rl_run_name,
        "num_rollout": RL_UPDATES,
        "dynamic_filter": False,
        "rollout_seed": 42,
        "save_interval": 40,
        "eval_interval": 0,
        "model_id": "context2048_47m_qwen3",
        "resume_if_available": True,
        "wandb_project": RL_PROJECT,
        "wandb_group": RL_GROUP,
        "max_tokens_per_gpu": 131_072,
        "sglang_server_concurrency": 128,
        "deterministic_inference": False,
        "rollout_only": False,
        "canary": False,
        "train_file": filtered["path"],
        "train_file_sha256": filtered["sha256"],
        "lr": RL_LR,
        "kl_loss_type": "low_var_kl",
        "rollout_max_prompt_len": PROMPT_CAP,
        "rollout_max_response_len": RESPONSE_BUDGET,
        "rollout_max_context_len": MAX_MODEL_LEN,
    }


def _merged_metrics(key: str) -> dict[str, Any]:
    histogram: dict[int, int] = {}
    format_hits = trajectories = evaluated = skipped = 0
    summaries = []
    for shard_id in range(SHARD_COUNT):
        summary = json.loads(
            (result_root(key, shard_id) / "success.json").read_text()
        )
        _validate_shard_summary(summary, key=key, shard_id=shard_id)
        summaries.append(summary)
        for wins, count in summary["wins_histogram"].items():
            histogram[int(wins)] = histogram.get(int(wins), 0) + int(count)
        trajectories += int(summary["trajectories"])
        format_hits += round(float(summary["format_rate"]) * summary["trajectories"])
        evaluated += int(summary["evaluated_prompts"])
        skipped += len(summary["skipped_overlong"])
    return {
        "checkpoint": key,
        "label": CHECKPOINTS[key].label,
        "evaluated_prompts": evaluated,
        "skipped_overlong": skipped,
        "trajectories": trajectories,
        "pass_at_1": _pass_at_k(histogram, n=N_SAMPLES, k=1),
        "pass_at_16": _pass_at_k(histogram, n=N_SAMPLES, k=16),
        "format_rate": format_hits / trajectories,
        "mixed_outcome_prompts": sum(
            count for wins, count in histogram.items() if 0 < wins < N_SAMPLES
        ),
        "wins_histogram": {
            str(wins): count for wins, count in sorted(histogram.items())
        },
        "shard_summary_sha256": [
            summary["summary_sha256"] for summary in summaries
        ],
    }


@app.function(
    cpu=4.0,
    memory=8 * 1024,
    timeout=60 * 60 * 30,
    retries=0,
    volumes={str(RESULTS_MOUNT): results_volume},
)
def run_pipeline() -> dict[str, Any]:
    results_volume.reload()
    ledger_path = RESULTS_ROOT / "evaluation.json"
    if ledger_path.is_file():
        ledger = json.loads(ledger_path.read_text())
        if ledger.get("state") in {"evaluating", "complete"}:
            raise FileExistsError(
                "evaluation ledger already exists; refusing a duplicate launch"
            )
    ledger: dict[str, Any] = {
        "schema": "context2048-pass16-evaluation-ledger-v1",
        "version": VERSION,
        "state": "evaluating",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "eval_calls": [],
    }
    atomic_json(ledger_path, ledger)
    results_volume.commit()

    calls: list[tuple[str, int, Any]] = []
    for key in CHECKPOINTS:
        for shard_id in range(SHARD_COUNT):
            call = eval_shard.spawn(key, shard_id, 0)
            calls.append((key, shard_id, call))
            ledger["eval_calls"].append(
                {
                    "checkpoint": key,
                    "shard_id": shard_id,
                    "function_call_id": call.object_id,
                }
            )
    atomic_json(ledger_path, ledger)
    results_volume.commit()
    for key, shard_id, call in calls:
        call.get()
        print(f"[pipeline] evaluation complete: {key}/{shard_id}", flush=True)

    results_volume.reload()
    metrics: dict[str, dict[str, Any]] = {}
    for key in CHECKPOINTS:
        metrics[key] = _merged_metrics(key)
    ledger["state"] = "complete"
    ledger["metrics"] = metrics
    ledger["finished_at"] = datetime.now(timezone.utc).isoformat()
    ledger["ledger_sha256"] = canonical_sha256(ledger)
    atomic_json(ledger_path, ledger)
    results_volume.commit()
    return ledger


@app.function(
    cpu=4.0,
    memory=8 * 1024,
    timeout=4 * 60 * 60,
    retries=0,
    volumes={
        str(RESULTS_MOUNT): results_volume,
        str(DATA_MOUNT): data_volume,
    },
)
def run_filtering() -> dict[str, Any]:
    """Build filters only after every evaluation shard has authenticated."""

    results_volume.reload()
    evaluation_path = RESULTS_ROOT / "evaluation.json"
    if not evaluation_path.is_file():
        raise FileNotFoundError("evaluation ledger is missing")
    evaluation = json.loads(evaluation_path.read_text())
    recorded_hash = evaluation.get("ledger_sha256")
    unhashed = {
        key: value
        for key, value in evaluation.items()
        if key != "ledger_sha256"
    }
    if (
        evaluation.get("schema")
        != "context2048-pass16-evaluation-ledger-v1"
        or evaluation.get("version") != VERSION
        or evaluation.get("state") != "complete"
        or recorded_hash != canonical_sha256(unhashed)
    ):
        raise ValueError("evaluation ledger is not an authenticated completion")
    ledger_path = RESULTS_ROOT / "filtering.json"
    if ledger_path.is_file():
        raise FileExistsError(
            "filtering ledger already exists; refusing a duplicate launch"
        )
    checkpoint_filters: dict[str, dict[str, Any]] = {}
    # Run these serially because every worker commits the shared data and
    # results Volumes.  Concurrent whole-Volume commits could lose another
    # worker's newly written parquet or success marker.
    for key in CHECKPOINTS:
        checkpoint_filters[key] = build_filtered_dataset.remote(key)
    common_filter = build_common_filtered_dataset.remote()
    ledger_core = {
        "schema": "context2048-mixed-outcome-filter-ledger-v1",
        "version": VERSION,
        "state": "complete",
        "evaluation_ledger_sha256": recorded_hash,
        "checkpoint_filters": checkpoint_filters,
        "common_filter": common_filter,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    ledger = {
        **ledger_core,
        "ledger_sha256": canonical_sha256(ledger_core),
    }
    atomic_json(ledger_path, ledger)
    results_volume.commit()
    return ledger


@app.function(
    cpu=2.0,
    memory=4 * 1024,
    timeout=10 * 60,
    volumes={str(RESULTS_MOUNT): results_volume},
)
def remote_status() -> dict[str, Any]:
    results_volume.reload()
    ledger_path = RESULTS_ROOT / "evaluation.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.is_file() else None
    filtering_path = RESULTS_ROOT / "filtering.json"
    filtering = (
        json.loads(filtering_path.read_text())
        if filtering_path.is_file()
        else None
    )
    common_filter_path = RESULTS_ROOT / "common_filter/success.json"
    common_filter = (
        json.loads(common_filter_path.read_text())
        if common_filter_path.is_file()
        else None
    )
    checkpoints: dict[str, Any] = {}
    for key in CHECKPOINTS:
        shards = []
        for shard_id in range(SHARD_COUNT):
            marker = result_root(key, shard_id) / "success.json"
            shards.append(json.loads(marker.read_text()) if marker.is_file() else None)
        filter_marker = RESULTS_ROOT / key / "filter/success.json"
        checkpoints[key] = {
            "shards_complete": sum(item is not None for item in shards),
            "filter": (
                json.loads(filter_marker.read_text())
                if filter_marker.is_file()
                else None
            ),
        }
    return {
        "version": VERSION,
        "evaluation": ledger,
        "filtering": filtering,
        "common_filter": common_filter,
        "checkpoints": checkpoints,
    }


@app.local_entrypoint()
def main(action: str = "inspect") -> None:
    action = action.strip().lower()
    if action == "inspect":
        print(json.dumps(inspect_inputs.remote(), indent=2, sort_keys=True))
        return
    if action == "canary":
        calls = {key: eval_shard.spawn(key, 0, 8) for key in CHECKPOINTS}
        results = {key: call.get() for key, call in calls.items()}
        print(json.dumps(results, indent=2, sort_keys=True))
        return
    if action == "launch":
        call = run_pipeline.spawn()
        print(
            json.dumps(
                {
                    "evaluation_call_id": call.object_id,
                    "version": VERSION,
                    "eval_checkpoints": list(CHECKPOINTS),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if action == "filter":
        call = run_filtering.spawn()
        print(
            json.dumps(
                {
                    "filtering_call_id": call.object_id,
                    "version": VERSION,
                    "checkpoints": list(CHECKPOINTS),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if action == "status":
        print(json.dumps(remote_status.remote(), indent=2, sort_keys=True))
        return
    raise ValueError("action must be inspect, canary, launch, filter, or status")
