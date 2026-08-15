"""Evaluate the five final context-2048 RL checkpoints on held-out B1--B5.

This is the canonical final-checkpoint evaluator for the FP32-master v13 RL
comparison. It converts each authenticated Miles checkpoint to a temporary
FP32 Hugging Face export, runs inference in BF16, and uses the exact online RL
reward implementation. Production launch is fail-closed and exactly once per
version namespace.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import modal

from context2048_eval_core import (
    atomic_json,
    canonical_sha256,
    deterministic_sample_seed,
    frame_prompt_ids,
    sha256_file,
    summarize_histogram,
    to_python_value,
)


APP_NAME = "chess-context2048-final-heldout-test"
VERSION = "context2048-fp32-master-v13-final-b1b5-n16-v2-20260815"
RESULTS_VOLUME_NAME = "chess-rl-eval-results-r6"
RAW_VOLUME_NAME = "chess-rl-miles-checkpoints"
HF_VOLUME_NAME = "rl-reasoning-checkpoints"
SOURCE_DATA_VOLUME_NAME = "chess-rl-miles-data"

RESULTS_MOUNT = Path("/results")
RESULTS_ROOT = RESULTS_MOUNT / VERSION
RAW_MOUNT = Path("/rl-checkpoints")
HF_MOUNT = Path("/pretrain-checkpoints")
SOURCE_DATA_MOUNT = Path("/data")
SOURCE_PARQUET = (
    SOURCE_DATA_MOUNT / "chess-rl-data/train_v4_dataset_balanced_multi_turn.parquet"
)
SOURCE_ROWS = 53_225
SOURCE_SHA256 = (
    "bcf131d88cd9916f6ec12a12f8e85901a05e0dabc5d5a1fdebd8ea0b0ae86c30"
)

N_SAMPLES = 16
PROMPT_CAP = 512
RESPONSE_BUDGET = 1_536
MAX_MODEL_LEN = 2_048
MAX_ENV_CALLS = 6
TEMPERATURE = 1.0
TOP_P = 1.0
BASE_SEED = 0
EXPECTED_RAW_PROMPTS = 1_484
EXPECTED_EVALUATED_PROMPTS = 1_480
EXPECTED_SKIPPED_OVERLONG = 4

ONLINE_REWARD_SHA256 = (
    "1a6065c58f0cf8c775112815c90930a87bde205e484a78572f8a0e54eb2bc5c0"
)
ONLINE_MOVES_SHA256 = (
    "c4680a3e736f9cb7e2abaa5460e5f29f48c68603e924b68a7474fc3cf8c256a0"
)
ONLINE_ROLLOUT_SHA256 = (
    "8414927087039693df11525a827553db5f0b6b4a2660d18ac43e0f338bd376e9"
)

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1] if modal.is_local() else Path("/opt/final_test_eval")
MILES_PACKAGE_LOCAL = PROJECT_ROOT / "miles/miles"
CHESS_RL_MILES_LOCAL = PROJECT_ROOT / "chess/rl/chess_rl_miles"
CONVERTER_LOCAL = PROJECT_ROOT / "miles/tools/convert_fsdp_to_hf.py"
CORE_LOCAL = HERE / "context2048_eval_core.py"
EVALUATOR_LOCAL = HERE / "modal_eval_context2048_final_test.py"
DEFAULT_TEST_DATA_ROOT = PROJECT_ROOT.parent / "Eval/test_data"
TEST_DATA_ROOT = Path(
    os.environ.get("CHESS_HELDOUT_TEST_DATA_ROOT", DEFAULT_TEST_DATA_ROOT)
)

REMOTE_ROOT = Path("/opt/final_test_eval")
REMOTE_TEST_DATA = REMOTE_ROOT / "data"
REMOTE_CONVERTER = REMOTE_ROOT / "convert_fsdp_to_hf.py"
REMOTE_CORE = REMOTE_ROOT / "context2048_eval_core.py"
REMOTE_EVALUATOR = REMOTE_ROOT / "modal_eval_context2048_final_test.py"
REMOTE_REWARD = REMOTE_ROOT / "chess_rl_miles/reward.py"
REMOTE_MOVES = REMOTE_ROOT / "chess_rl_miles/moves.py"
REMOTE_ROLLOUT = REMOTE_ROOT / "chess_rl_miles/rollout.py"


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    filename: str
    rows: int
    sha256: str


DATASETS: dict[str, DatasetSpec] = {
    "B1": DatasetSpec(
        "B1",
        "test_B1_multi_turn.parquet",
        310,
        "3ac5df0af21b395c23f864dd75b6a64335e3fe681c2b774f1485b276c6893c78",
    ),
    "B2": DatasetSpec(
        "B2",
        "test_B2_multi_turn.parquet",
        299,
        "9b315fe82a676b9b817ae77f96f7987be04ab34ec18513e3d42544896a133c3f",
    ),
    "B3": DatasetSpec(
        "B3",
        "test_B3_multi_turn.parquet",
        267,
        "8e41e0cf7c17babf6ae9a17a5b51607eef5674788dd09042e7dbbf90a945a5b9",
    ),
    "B4": DatasetSpec(
        "B4",
        "test_B4_multi_turn.parquet",
        288,
        "9583e4f6621ffee456eefc3e9d9de15800ec24226d20b882ff4805e82c4a985b",
    ),
    "B5": DatasetSpec(
        "B5",
        "test_B5_multi_turn.parquet",
        320,
        "927d62a4994d39e61ffb6719f85961ba14dbd55f365c539477fe6db72288c5cc",
    ),
}


@dataclass(frozen=True)
class CheckpointSpec:
    key: str
    label: str
    run_name: str
    origin_subpath: str
    origin_fingerprint: str
    checkpoint_commit_sha256: str

    @property
    def raw_subpath(self) -> str:
        return (
            "chess-rl-miles-interleave-fp32-master-v3/"
            f"{self.run_name}/iter_0001500"
        )


CHECKPOINTS: dict[str, CheckpointSpec] = {
    "vocab81_expand85_sft3": CheckpointSpec(
        key="vocab81_expand85_sft3",
        label="81-token pretraining, expansion to 85 tokens, then SFT for 3 epochs",
        run_name=(
            "ctx2048-fp32masterv13-vocab81pt-expand85-sft3-filtered-"
            "lr1e5-rl1500-r3"
        ),
        origin_subpath=(
            "context2048_vocab_mixing_fp32_master_v13_20260813/"
            "vocab81_then_sft3/sft/final"
        ),
        origin_fingerprint=(
            "350e1eb7dd87e5fb0107437a3ccdb1dc42efdc034edd4cc0b502738c04de7270"
        ),
        checkpoint_commit_sha256=(
            "76130c6b3e6990264422c8de2d42985466963c7423c88a408f4771c3970fb466"
        ),
    ),
    "vocab85_sft3": CheckpointSpec(
        key="vocab85_sft3",
        label="85-token pretraining, then SFT for 3 epochs",
        run_name=(
            "ctx2048-fp32masterv13-vocab85pt-sft3-filtered-lr1e5-rl1500-r3"
        ),
        origin_subpath=(
            "context2048_vocab_mixing_fp32_master_v13_20260813/"
            "vocab85_then_sft3/sft/final"
        ),
        origin_fingerprint=(
            "0b286a1ad928c1efefb135cdd8d8bf28d867276e28a7dc682ade3684e6ee6c19"
        ),
        checkpoint_commit_sha256=(
            "9967929d610e86b4343d6a434ee1eeac056aa498d827a1a705403f3de18834e3"
        ),
    ),
    "mixed_sft1": CheckpointSpec(
        key="mixed_sft1",
        label="85-token uniformly shuffled mixed pretraining plus one SFT copy",
        run_name=(
            "ctx2048-fp32masterv13-mixed-sft1-filtered-lr1e5-rl1500-r3"
        ),
        origin_subpath=(
            "context2048_vocab_mixing_fp32_master_v13_20260813/"
            "mixed_sft1/mixed/final"
        ),
        origin_fingerprint=(
            "e42a2ed9a5e2b0550c5e5e06ef48e4089ff046d4415d2b4c9c28af0745c0c139"
        ),
        checkpoint_commit_sha256=(
            "cc9b82f2f378cd954a7369ba4458170b245cf6e69e09590515589863f5e2712b"
        ),
    ),
    "mixed_sft3_fresh_adam": CheckpointSpec(
        key="mixed_sft3_fresh_adam",
        label=(
            "85-token uniformly shuffled mixed pretraining plus three SFT copies; "
            "fresh RL Adam"
        ),
        run_name=(
            "ctx2048-fp32masterv13-mixed-sft3-filtered-lr1e5-rl1500-r3"
        ),
        origin_subpath=(
            "context2048_vocab_mixing_fp32_master_v13_20260813/"
            "mixed_sft3/mixed/final"
        ),
        origin_fingerprint=(
            "61193269be0afc01e310705fef7ed071ea8b224da83242db52594279edf32075"
        ),
        checkpoint_commit_sha256=(
            "8e32bacc7a2aa7856a25c678919bc70858716493246573a5cd26517578da40fc"
        ),
    ),
    "mixed_sft3_continued_adam": CheckpointSpec(
        key="mixed_sft3_continued_adam",
        label=(
            "85-token uniformly shuffled mixed pretraining plus three SFT copies; "
            "parent Adam moments and per-parameter step continued from 36,848"
        ),
        run_name=(
            "ctx2048-fp32masterv13-mixed-pt-plus-sft3-continue-adam36848-"
            "filtered-lr1e5-rl1500"
        ),
        origin_subpath=(
            "context2048_vocab_mixing_fp32_master_v13_20260813/"
            "mixed_sft3/mixed/final"
        ),
        origin_fingerprint=(
            "61193269be0afc01e310705fef7ed071ea8b224da83242db52594279edf32075"
        ),
        checkpoint_commit_sha256=(
            "8029879eb82aeb2f07c63a9c3e0e6423fbca8dc58d808752b5ba69abb870dedf"
        ),
    ),
}


PROFILES: dict[str, dict[str, Any]] = {
    "canary": {
        "dataset_keys": ("B1",),
        "max_prompts_per_dataset": 8,
        "n_samples": 2,
    },
    "production": {
        "dataset_keys": tuple(DATASETS),
        "max_prompts_per_dataset": 0,
        "n_samples": N_SAMPLES,
    },
}


image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install("git")
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
    .env({"PYTHONPATH": str(REMOTE_ROOT)})
)
if modal.is_local():
    image = (
        image.add_local_dir(
            str(MILES_PACKAGE_LOCAL),
            remote_path=str(REMOTE_ROOT / "miles"),
            copy=True,
        )
        .add_local_dir(
            str(CHESS_RL_MILES_LOCAL),
            remote_path=str(REMOTE_ROOT / "chess_rl_miles"),
            copy=True,
        )
        .add_local_file(
            str(CONVERTER_LOCAL), remote_path=str(REMOTE_CONVERTER), copy=True
        )
        .add_local_file(str(CORE_LOCAL), remote_path=str(REMOTE_CORE), copy=True)
        .add_local_file(
            str(EVALUATOR_LOCAL), remote_path=str(REMOTE_EVALUATOR), copy=True
        )
    )
    for dataset in DATASETS.values():
        source = TEST_DATA_ROOT / dataset.filename
        image = image.add_local_file(
            str(source),
            remote_path=str(REMOTE_TEST_DATA / dataset.filename),
            copy=True,
        )

results_volume = modal.Volume.from_name(RESULTS_VOLUME_NAME, create_if_missing=True)
raw_volume = modal.Volume.from_name(RAW_VOLUME_NAME, create_if_missing=False)
hf_volume = modal.Volume.from_name(HF_VOLUME_NAME, create_if_missing=False)
source_data_volume = modal.Volume.from_name(
    SOURCE_DATA_VOLUME_NAME, create_if_missing=False
)

app = modal.App(APP_NAME, image=image)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"missing or invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _checkpoint_path(spec: CheckpointSpec) -> Path:
    return RAW_MOUNT / spec.raw_subpath


def _origin_path(spec: CheckpointSpec) -> Path:
    return HF_MOUNT / spec.origin_subpath


def _dataset_path(spec: DatasetSpec) -> Path:
    return REMOTE_TEST_DATA / spec.filename


def _artifact_fingerprint(checkpoint: Path) -> str:
    """Fingerprint the same HF files pinned by the training-stage evaluator."""

    root = checkpoint.resolve(strict=True)
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
    digest = hashlib.sha256()
    for path in sorted(
        set(files), key=lambda item: item.relative_to(root).as_posix()
    ):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 << 20), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _source_code_identity() -> dict[str, str]:
    paths = {
        "evaluator": REMOTE_EVALUATOR,
        "core": REMOTE_CORE,
        "converter": REMOTE_CONVERTER,
        "reward": REMOTE_REWARD,
        "moves": REMOTE_MOVES,
        "rollout": REMOTE_ROLLOUT,
    }
    identity = {name: sha256_file(path) for name, path in paths.items()}
    expected = {
        "reward": ONLINE_REWARD_SHA256,
        "moves": ONLINE_MOVES_SHA256,
        "rollout": ONLINE_ROLLOUT_SHA256,
    }
    for name, digest in expected.items():
        if identity[name] != digest:
            raise RuntimeError(
                f"packaged {name} source drifted: {identity[name]} != {digest}"
            )
    return identity


def evaluation_contract(profile: str, checkpoint_key: str) -> dict[str, Any]:
    if profile not in PROFILES:
        raise ValueError(f"unknown profile: {profile}")
    if checkpoint_key not in CHECKPOINTS:
        raise ValueError(f"unknown checkpoint: {checkpoint_key}")
    profile_spec = PROFILES[profile]
    checkpoint = CHECKPOINTS[checkpoint_key]
    return {
        "schema": "context2048-final-heldout-evaluation-contract-v1",
        "version": VERSION,
        "profile": profile,
        "checkpoint": asdict(checkpoint),
        "datasets": [
            asdict(DATASETS[key]) for key in profile_spec["dataset_keys"]
        ],
        "generation": {
            "base_seed": BASE_SEED,
            "context_tokens": MAX_MODEL_LEN,
            "inference_dtype": "bfloat16",
            "max_env_calls": MAX_ENV_CALLS,
            "n_samples": int(profile_spec["n_samples"]),
            "prompt_cap_including_bos": PROMPT_CAP,
            "response_tokens": RESPONSE_BUDGET,
            "sampling_seed_scope": "dataset,row,sample_slot,generation_round",
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
        },
        "max_prompts_per_dataset": int(
            profile_spec["max_prompts_per_dataset"]
        ),
        "scorer": {
            "entrypoint": "chess_rl_miles.reward._score_sample",
            "reward_sha256": ONLINE_REWARD_SHA256,
            "moves_sha256": ONLINE_MOVES_SHA256,
            "rollout_sha256": ONLINE_ROLLOUT_SHA256,
        },
    }


def _result_root(checkpoint_key: str, profile: str) -> Path:
    return RESULTS_ROOT / profile / checkpoint_key


class _State:
    __slots__ = (
        "dataset_key",
        "row_index",
        "sample_slot",
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
        dataset_key: str,
        row_index: int,
        sample_slot: int,
        prompt_ids: list[int],
        env_replies: list[str],
    ) -> None:
        self.dataset_key = dataset_key
        self.row_index = row_index
        self.sample_slot = sample_slot
        self.prompt_len = len(prompt_ids)
        self.ctx = list(prompt_ids)
        self.model_tokens = 0
        self.env_calls = 0
        self.env_replies = list(env_replies)
        self.finished = False

    @property
    def response_ids(self) -> list[int]:
        return self.ctx[self.prompt_len :]


def _merge_bucket_metrics(
    buckets: Iterable[Mapping[str, Any]], *, n_samples: int
) -> dict[str, Any]:
    histogram = {wins: 0 for wins in range(n_samples + 1)}
    format_hits = 0
    trajectories = 0
    raw_prompts = 0
    skipped_overlong = 0
    for bucket in buckets:
        for wins, count in bucket["wins_histogram"].items():
            histogram[int(wins)] += int(count)
        format_hits += int(bucket["format_hits"])
        trajectories += int(bucket["trajectories"])
        raw_prompts += int(bucket["raw_prompts"])
        skipped_overlong += int(bucket["skipped_overlong_count"])
    summary = summarize_histogram(histogram, n=n_samples)
    return {
        **summary,
        "raw_prompts": raw_prompts,
        "skipped_overlong_count": skipped_overlong,
        "trajectories": trajectories,
        "format_hits": format_hits,
        "format_rate": format_hits / trajectories,
    }


def _validate_success(
    payload: Mapping[str, Any], *, checkpoint_key: str, profile: str
) -> None:
    recorded = payload.get("summary_sha256")
    core = {key: value for key, value in payload.items() if key != "summary_sha256"}
    expected_contract = evaluation_contract(profile, checkpoint_key)
    if (
        recorded != canonical_sha256(core)
        or payload.get("schema") != "context2048-final-heldout-summary-v1"
        or payload.get("version") != VERSION
        or payload.get("profile") != profile
        or payload.get("checkpoint_key") != checkpoint_key
        or payload.get("contract_sha256") != canonical_sha256(expected_contract)
        or payload.get("checkpoint_identity", {}).get("commit_sha256")
        != CHECKPOINTS[checkpoint_key].checkpoint_commit_sha256
    ):
        raise RuntimeError("existing evaluation success marker failed authentication")


def _validate_packaged_dataset(spec: DatasetSpec) -> Path:
    path = _dataset_path(spec)
    observed = sha256_file(path)
    if observed != spec.sha256:
        raise RuntimeError(f"{spec.key} parquet drifted: {observed} != {spec.sha256}")
    return path


def _convert_checkpoint(spec: CheckpointSpec) -> tuple[Path, dict[str, Any]]:
    raw_path = _checkpoint_path(spec)
    origin_path = _origin_path(spec)
    marker = _read_json(raw_path / "COMMITTED.json")
    meta = _read_json(raw_path / "meta.json")
    if (
        marker.get("schema") != "miles-fsdp-checkpoint-commit-v1"
        or marker.get("commit_sha256") != spec.checkpoint_commit_sha256
        or marker.get("iteration") != 1_500
        or meta.get("iteration") != 1_500
        or meta.get("global_step") != 1_500
        or meta.get("rollout_id") != 1_499
    ):
        raise RuntimeError(f"raw checkpoint identity drifted for {spec.key}")
    origin_fingerprint = _artifact_fingerprint(origin_path)
    if origin_fingerprint != spec.origin_fingerprint:
        raise RuntimeError(
            f"conversion origin drifted for {spec.key}: {origin_fingerprint}"
        )

    scratch = Path(f"/tmp/context2048-final-test-{spec.key}")
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True, exist_ok=False)
    model_path = scratch / "hf"
    command = [
        sys.executable,
        str(REMOTE_CONVERTER),
        "--input-dir",
        str(raw_path),
        "--origin-hf-dir",
        str(origin_path),
        "--output-dir",
        str(model_path),
    ]
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if completed.stdout:
        print(completed.stdout[-30_000:], flush=True)
    if completed.returncode:
        raise RuntimeError(
            f"FSDP-to-HF conversion failed with exit {completed.returncode}"
        )
    export_marker = _read_json(model_path / "COMMITTED.json")
    source_identity = export_marker.get("source_checkpoint")
    if (
        export_marker.get("schema") != "miles-hf-export-commit-v1"
        or not isinstance(source_identity, dict)
        or source_identity.get("iteration") != 1_500
        or source_identity.get("commit_sha256")
        != spec.checkpoint_commit_sha256
    ):
        raise RuntimeError(f"converted checkpoint identity drifted for {spec.key}")
    return model_path, export_marker


def _evaluate_dataset(
    *,
    checkpoint_key: str,
    dataset: DatasetSpec,
    llm: Any,
    tokenizer: Any,
    n_samples: int,
    max_prompts: int,
    output_root: Path,
) -> dict[str, Any]:
    from types import SimpleNamespace

    import pandas as pd
    from chess_rl_miles.reward import _score_sample
    from miles.utils.types import Sample
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    source_path = _validate_packaged_dataset(dataset)
    frame = pd.read_parquet(source_path)
    if len(frame) != dataset.rows:
        raise RuntimeError(f"{dataset.key} row count drifted: {len(frame)}")
    if max_prompts:
        frame = frame.iloc[:max_prompts].copy()
    frame.reset_index(drop=True, inplace=True)

    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    call_env_id = tokenizer.convert_tokens_to_ids("<call_env>")
    if bos_id is None:
        raise RuntimeError("tokenizer has no bos_token_id")
    if call_env_id is None or call_env_id == tokenizer.unk_token_id:
        raise RuntimeError("<call_env> is missing from the tokenizer")

    states: list[_State] = []
    skipped_overlong: list[int] = []
    for row_index, row in enumerate(frame.itertuples(index=False)):
        unframed = tokenizer(
            str(row.prompt), add_special_tokens=False
        ).input_ids
        prompt_ids = frame_prompt_ids(
            unframed, bos_id=int(bos_id), prompt_cap=PROMPT_CAP
        )
        if prompt_ids is None:
            skipped_overlong.append(row_index)
            continue
        extra_info = to_python_value(row.extra_info or {})
        raw_replies = extra_info.get("env_replies", [])
        replies = [
            str(reply)
            for reply in (raw_replies if raw_replies is not None else [])
        ]
        for sample_slot in range(n_samples):
            states.append(
                _State(
                    dataset_key=dataset.key,
                    row_index=row_index,
                    sample_slot=sample_slot,
                    prompt_ids=prompt_ids,
                    env_replies=replies,
                )
            )
    print(
        f"[heldout] {checkpoint_key}/{dataset.key}: raw={len(frame)} "
        f"overlong={len(skipped_overlong)} trajectories={len(states)}",
        flush=True,
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
        generating: list[_State] = []
        for state in active:
            budget = min(
                RESPONSE_BUDGET - state.model_tokens,
                MAX_MODEL_LEN - len(state.ctx),
            )
            if budget <= 0:
                state.finished = True
                continue
            generating.append(state)
            prompts.append(TokensPrompt(prompt_token_ids=state.ctx))
            parameters.append(
                SamplingParams(
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    max_tokens=budget,
                    stop_token_ids=[int(call_env_id)],
                    seed=deterministic_sample_seed(
                        base_seed=BASE_SEED,
                        dataset_key=state.dataset_key,
                        row_index=state.row_index,
                        sample_slot=state.sample_slot,
                        generation_round=generation_round,
                    ),
                )
            )
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
                tokens.append(int(call_env_id))
            tokens = tokens[: RESPONSE_BUDGET - state.model_tokens]
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
                    reply_ids = reply_token_cache[reply][
                        : max(MAX_MODEL_LEN - len(state.ctx), 0)
                    ]
                    state.ctx.extend(reply_ids)
                    if not reply_ids:
                        state.finished = True
                else:
                    state.finished = True
                if state.env_calls >= MAX_ENV_CALLS:
                    state.finished = True
                if len(state.ctx) >= MAX_MODEL_LEN:
                    state.finished = True
            else:
                state.ctx.extend(tokens)
                state.model_tokens += len(tokens)
                state.finished = True
        print(
            f"[heldout] {checkpoint_key}/{dataset.key}: "
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
    scored: list[dict[str, Any]] = []
    for state, text in zip(states, texts, strict=True):
        row = frame.iloc[state.row_index]
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

    dataset_root = output_root / dataset.key
    dataset_root.mkdir(parents=True, exist_ok=False)
    generations_path = dataset_root / "generations.jsonl.gz"
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
                state.row_index,
                {"n": 0, "wins": 0, "format": 0, "slot_mask": 0},
            )
            bit = 1 << state.sample_slot
            if entry["slot_mask"] & bit:
                raise ValueError("duplicate sample slot")
            entry["slot_mask"] |= bit
            entry["n"] += 1
            entry["wins"] += int(score)
            entry["format"] += int(format_ok)
            handle.write(
                json.dumps(
                    {
                        "dataset": dataset.key,
                        "row_index": state.row_index,
                        "sample_slot": state.sample_slot,
                        "sampling_seed_base": BASE_SEED,
                        "score": score,
                        "format_ok": bool(format_ok),
                        "model_tokens": state.model_tokens,
                        "env_calls": state.env_calls,
                        "response": text,
                    },
                    allow_nan=False,
                    ensure_ascii=False,
                )
                + "\n"
            )

    full_mask = (1 << n_samples) - 1
    if any(
        entry["n"] != n_samples or entry["slot_mask"] != full_mask
        for entry in per_prompt.values()
    ):
        raise ValueError("incomplete per-prompt sample coverage")
    histogram = {wins: 0 for wins in range(n_samples + 1)}
    format_hits = 0
    for entry in per_prompt.values():
        histogram[entry["wins"]] += 1
        format_hits += entry["format"]
    metrics = summarize_histogram(histogram, n=n_samples)
    trajectories = len(states)
    summary_core = {
        "schema": "context2048-final-heldout-bucket-summary-v1",
        "version": VERSION,
        "checkpoint_key": checkpoint_key,
        "dataset": asdict(dataset),
        "raw_prompts": len(frame),
        "skipped_overlong": skipped_overlong,
        "skipped_overlong_count": len(skipped_overlong),
        "trajectories": trajectories,
        "n_samples": n_samples,
        **metrics,
        "format_hits": format_hits,
        "format_rate": format_hits / trajectories,
        "tokenizer": {
            "bos_token_id": int(bos_id),
            "eos_token_id": None if eos_id is None else int(eos_id),
            "call_env_token_id": int(call_env_id),
        },
        "generations": {
            "path": str(generations_path),
            "bytes": generations_path.stat().st_size,
            "sha256": sha256_file(generations_path),
        },
        "finished_at": _utc_now(),
    }
    summary = {
        **summary_core,
        "summary_sha256": canonical_sha256(summary_core),
    }
    atomic_json(dataset_root / "summary.json", summary)
    return summary


@app.function(
    gpu="H200",
    cpu=16.0,
    memory=128 * 1024,
    timeout=16 * 60 * 60,
    retries=0,
    max_containers=5,
    volumes={
        str(RESULTS_MOUNT): results_volume,
        str(RAW_MOUNT): raw_volume,
        str(HF_MOUNT): hf_volume,
    },
)
def evaluate_checkpoint(checkpoint_key: str, profile: str) -> dict[str, Any]:
    from transformers import AutoTokenizer
    from vllm import LLM

    results_volume.reload()
    raw_volume.reload()
    hf_volume.reload()
    if checkpoint_key not in CHECKPOINTS or profile not in PROFILES:
        raise ValueError(f"invalid checkpoint/profile: {checkpoint_key}/{profile}")
    spec = CHECKPOINTS[checkpoint_key]
    profile_spec = PROFILES[profile]
    contract = evaluation_contract(profile, checkpoint_key)
    contract_sha256 = canonical_sha256(contract)
    output_root = _result_root(checkpoint_key, profile)
    success_path = output_root / "_SUCCESS.json"
    running_path = output_root / "_RUNNING.json"
    failed_path = output_root / "_FAILED.json"
    if success_path.is_file():
        success = _read_json(success_path)
        _validate_success(success, checkpoint_key=checkpoint_key, profile=profile)
        return success
    if running_path.exists() or failed_path.exists():
        raise FileExistsError(
            f"evaluation namespace is already claimed for {checkpoint_key}/{profile}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    started_at = _utc_now()
    started_clock = time.monotonic()
    running = {
        "schema": "context2048-final-heldout-running-v1",
        "version": VERSION,
        "profile": profile,
        "checkpoint_key": checkpoint_key,
        "contract_sha256": contract_sha256,
        "started_at": started_at,
    }
    atomic_json(running_path, running)
    results_volume.commit()

    try:
        source_code = _source_code_identity()
        model_path, export_marker = _convert_checkpoint(spec)
        config = _read_json(model_path / "config.json")
        if int(config.get("max_position_embeddings", -1)) != MAX_MODEL_LEN:
            raise RuntimeError("converted model does not advertise context 2048")
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), trust_remote_code=True, local_files_only=True
        )
        llm = LLM(
            model=str(model_path),
            trust_remote_code=True,
            gpu_memory_utilization=0.60,
            max_model_len=MAX_MODEL_LEN,
            max_num_seqs=128,
            seed=BASE_SEED,
            enforce_eager=False,
            dtype="bfloat16",
        )
        bucket_summaries = []
        for dataset_key in profile_spec["dataset_keys"]:
            bucket_summaries.append(
                _evaluate_dataset(
                    checkpoint_key=checkpoint_key,
                    dataset=DATASETS[dataset_key],
                    llm=llm,
                    tokenizer=tokenizer,
                    n_samples=int(profile_spec["n_samples"]),
                    max_prompts=int(profile_spec["max_prompts_per_dataset"]),
                    output_root=output_root,
                )
            )
        overall = _merge_bucket_metrics(
            bucket_summaries, n_samples=int(profile_spec["n_samples"])
        )
        if profile == "production" and (
            overall["raw_prompts"] != EXPECTED_RAW_PROMPTS
            or overall["evaluated_prompts"] != EXPECTED_EVALUATED_PROMPTS
            or overall["skipped_overlong_count"] != EXPECTED_SKIPPED_OVERLONG
            or overall["trajectories"]
            != EXPECTED_EVALUATED_PROMPTS * N_SAMPLES
        ):
            raise RuntimeError(f"production cohort drifted: {overall}")
        source_identity = export_marker["source_checkpoint"]
        summary_core = {
            "schema": "context2048-final-heldout-summary-v1",
            "version": VERSION,
            "profile": profile,
            "checkpoint_key": checkpoint_key,
            "checkpoint_label": spec.label,
            "run_name": spec.run_name,
            "contract": contract,
            "contract_sha256": contract_sha256,
            "checkpoint_identity": source_identity,
            "hf_export_commit_sha256": export_marker["commit_sha256"],
            "source_code": source_code,
            "buckets": {
                summary["dataset"]["key"]: summary
                for summary in bucket_summaries
            },
            "overall": overall,
            "started_at": started_at,
            "finished_at": _utc_now(),
            "duration_seconds": round(time.monotonic() - started_clock, 3),
        }
        summary = {
            **summary_core,
            "summary_sha256": canonical_sha256(summary_core),
        }
        atomic_json(success_path, summary)
        running_path.unlink()
        results_volume.commit()
        return summary
    except BaseException as exc:
        failure = {
            **running,
            "finished_at": _utc_now(),
            "duration_seconds": round(time.monotonic() - started_clock, 3),
            "error": f"{type(exc).__name__}: {exc}",
        }
        atomic_json(failed_path, failure)
        running_path.unlink(missing_ok=True)
        results_volume.commit()
        raise


def _comparison_payload(results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    table = []
    for key, result in results.items():
        overall = result["overall"]
        table.append(
            {
                "checkpoint_key": key,
                "label": CHECKPOINTS[key].label,
                "pass_at_k": overall["pass_at_k"],
                "format_rate": overall["format_rate"],
                "all_zero_percentage": overall["all_zero_percentage"],
                "all_one_percentage": overall["all_one_percentage"],
                "bucket_pass_at_k": {
                    bucket: summary["pass_at_k"]
                    for bucket, summary in result["buckets"].items()
                },
            }
        )
    fresh = results["mixed_sft3_fresh_adam"]["overall"]
    continued = results["mixed_sft3_continued_adam"]["overall"]
    return {
        "table": table,
        "fresh_vs_continued_adam": {
            "direction": "continued minus fresh",
            "pass_at_k_delta": {
                str(k): continued["pass_at_k"][str(k)]
                - fresh["pass_at_k"][str(k)]
                for k in range(1, N_SAMPLES + 1)
            },
            "format_rate_delta": continued["format_rate"]
            - fresh["format_rate"],
            "all_zero_percentage_delta": continued["all_zero_percentage"]
            - fresh["all_zero_percentage"],
        },
    }


@app.function(
    cpu=4.0,
    memory=8 * 1024,
    timeout=30 * 60 * 60,
    retries=0,
    volumes={str(RESULTS_MOUNT): results_volume},
)
def run_pipeline() -> dict[str, Any]:
    """Claim the version once, launch exactly five workers, then merge."""

    results_volume.reload()
    ledger_path = RESULTS_ROOT / "evaluation.json"
    if ledger_path.exists():
        raise FileExistsError(
            f"evaluation ledger already exists; refusing duplicate launch: {ledger_path}"
        )
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    ledger: dict[str, Any] = {
        "schema": "context2048-final-heldout-ledger-v1",
        "version": VERSION,
        "state": "running",
        "created_at": _utc_now(),
        "expected_checkpoints": list(CHECKPOINTS),
        "workers": [],
    }
    atomic_json(ledger_path, ledger)
    results_volume.commit()

    calls: list[tuple[str, Any]] = []
    for key in CHECKPOINTS:
        call = evaluate_checkpoint.spawn(key, "production")
        calls.append((key, call))
        ledger["workers"].append(
            {"checkpoint_key": key, "function_call_id": call.object_id}
        )
        atomic_json(ledger_path, ledger)
        results_volume.commit()

    results: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}
    for key, call in calls:
        try:
            results[key] = call.get()
        except BaseException as exc:
            errors[key] = f"{type(exc).__name__}: {exc}"
    results_volume.reload()
    if errors:
        ledger["state"] = "failed"
        ledger["errors"] = errors
        ledger["finished_at"] = _utc_now()
        atomic_json(ledger_path, ledger)
        results_volume.commit()
        raise RuntimeError(f"one or more evaluation workers failed: {errors}")

    comparison = _comparison_payload(results)
    ledger_core = {
        **ledger,
        "state": "complete",
        "results": {
            key: {
                "summary_sha256": result["summary_sha256"],
                "checkpoint_identity": result["checkpoint_identity"],
                "overall": result["overall"],
            }
            for key, result in results.items()
        },
        "comparison": comparison,
        "finished_at": _utc_now(),
    }
    complete = {
        **ledger_core,
        "ledger_sha256": canonical_sha256(ledger_core),
    }
    atomic_json(ledger_path, complete)
    results_volume.commit()
    return complete


@app.function(
    cpu=8.0,
    memory=32 * 1024,
    timeout=2 * 60 * 60,
    volumes={
        str(RAW_MOUNT): raw_volume,
        str(HF_MOUNT): hf_volume,
        str(SOURCE_DATA_MOUNT): source_data_volume,
    },
)
def inspect_inputs() -> dict[str, Any]:
    """Authenticate datasets, prompt separation, origins, and final markers."""

    import pandas as pd

    raw_volume.reload()
    hf_volume.reload()
    source_data_volume.reload()
    source_hash = sha256_file(SOURCE_PARQUET)
    source_frame = pd.read_parquet(SOURCE_PARQUET, columns=["prompt"])
    if len(source_frame) != SOURCE_ROWS or source_hash != SOURCE_SHA256:
        raise RuntimeError("RL source parquet identity drifted")
    source_prompts = set(source_frame["prompt"].astype(str))

    dataset_records: dict[str, Any] = {}
    total_rows = 0
    for key, spec in DATASETS.items():
        path = _validate_packaged_dataset(spec)
        frame = pd.read_parquet(path, columns=["prompt"])
        overlap = len(set(frame["prompt"].astype(str)) & source_prompts)
        if len(frame) != spec.rows or overlap != 0:
            raise RuntimeError(
                f"held-out dataset check failed for {key}: rows={len(frame)} "
                f"overlap={overlap}"
            )
        total_rows += len(frame)
        dataset_records[key] = {
            **asdict(spec),
            "exact_prompt_overlap_with_rl_source": overlap,
        }
    if total_rows != EXPECTED_RAW_PROMPTS:
        raise RuntimeError(f"held-out raw prompt count drifted: {total_rows}")

    origins: dict[str, str] = {}
    checkpoint_records: dict[str, Any] = {}
    for key, spec in CHECKPOINTS.items():
        if spec.origin_subpath not in origins:
            origins[spec.origin_subpath] = _artifact_fingerprint(_origin_path(spec))
        if origins[spec.origin_subpath] != spec.origin_fingerprint:
            raise RuntimeError(f"origin checkpoint drifted for {key}")
        marker = _read_json(_checkpoint_path(spec) / "COMMITTED.json")
        meta = _read_json(_checkpoint_path(spec) / "meta.json")
        if (
            marker.get("commit_sha256") != spec.checkpoint_commit_sha256
            or marker.get("iteration") != 1_500
            or meta.get("global_step") != 1_500
            or meta.get("rollout_id") != 1_499
        ):
            raise RuntimeError(f"final checkpoint drifted for {key}")
        checkpoint_records[key] = {
            "run_name": spec.run_name,
            "raw_path": str(_checkpoint_path(spec)),
            "checkpoint_commit_sha256": marker["commit_sha256"],
            "global_step": meta["global_step"],
            "rollout_id": meta["rollout_id"],
            "origin_fingerprint": origins[spec.origin_subpath],
        }
    return {
        "version": VERSION,
        "source": {
            "rows": len(source_frame),
            "sha256": source_hash,
        },
        "datasets": dataset_records,
        "total_raw_prompts": total_rows,
        "checkpoints": checkpoint_records,
        "production_contract_sha256": canonical_sha256(
            {
                key: evaluation_contract("production", key)
                for key in CHECKPOINTS
            }
        ),
        "checked_at": _utc_now(),
    }


@app.function(
    cpu=2.0,
    memory=4 * 1024,
    timeout=10 * 60,
    volumes={str(RESULTS_MOUNT): results_volume},
)
def remote_status() -> dict[str, Any]:
    results_volume.reload()
    ledger_path = RESULTS_ROOT / "evaluation.json"
    ledger = _read_json(ledger_path) if ledger_path.is_file() else None
    checkpoints: dict[str, Any] = {}
    for key in CHECKPOINTS:
        root = _result_root(key, "production")
        state = "not_started"
        payload: dict[str, Any] | None = None
        for marker, candidate in (
            ("complete", root / "_SUCCESS.json"),
            ("failed", root / "_FAILED.json"),
            ("running", root / "_RUNNING.json"),
        ):
            if candidate.is_file():
                state = marker
                payload = _read_json(candidate)
                break
        checkpoints[key] = {
            "state": state,
            "function_call_id": next(
                (
                    worker["function_call_id"]
                    for worker in (ledger or {}).get("workers", [])
                    if worker["checkpoint_key"] == key
                ),
                None,
            ),
            "overall": payload.get("overall") if payload else None,
            "error": payload.get("error") if payload else None,
        }
    return {"version": VERSION, "ledger": ledger, "checkpoints": checkpoints}


@app.local_entrypoint()
def main(action: str = "inspect") -> None:
    action = action.strip().lower()
    if action == "inspect":
        print(json.dumps(inspect_inputs.remote(), indent=2, sort_keys=True))
        return
    if action == "canary":
        key = "mixed_sft3_continued_adam"
        call = evaluate_checkpoint.spawn(key, "canary")
        print(
            json.dumps(
                {
                    "checkpoint_key": key,
                    "function_call_id": call.object_id,
                    "result": call.get(),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if action == "launch":
        call = run_pipeline.spawn()
        print(
            json.dumps(
                {
                    "pipeline_function_call_id": call.object_id,
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
    raise ValueError("action must be inspect, canary, launch, or status")
