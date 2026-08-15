"""pass@16 sweep for the SFT-injection ablation checkpoints.

Evaluates each of the 8 final checkpoints (A1-A4, B1-B4) on ALL 53,225 rows of
the balanced RL parquet with n=16 samples/prompt, temp 1.0, via the same
verl_eval.sh harness and settings the official B1-B5 evaluator uses.
No RL training anywhere.

  modal run Eval/modal_eval_ablation_pass16.py --action canary          # A1, n=1
  modal run --detach Eval/modal_eval_ablation_pass16.py --action launch # all 8, n=16
  modal run Eval/modal_eval_ablation_pass16.py --action status
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal

HERE = Path(__file__).resolve().parent
VERL_ROOT = HERE / "pre2post-chess" / "rl"
EVAL_DATA_ROOT = HERE / "test_data"
REMOTE_VERL_ROOT = "/root/verl"
REMOTE_EVAL_DATA = "/eval-data"

EXPERIMENT_VERSION = "sft_injection_ablation_v1_20260801"
CKPT_MOUNT = "/pretrain-checkpoints"
CKPT_ROOT = (
    f"{CKPT_MOUNT}/interleave_50m/pretrain/{EXPERIMENT_VERSION}"
)
ARMS = ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4"]
ARM_SLUGS = {a: a.lower() for a in ARMS}

DATASET = "eval_train_v4_balanced_multi_turn"
DATASET_SHA256 = (
    "bcf131d88cd9916f6ec12a12f8e85901a05e0dabc5d5a1fdebd8ea0b0ae86c30"
)
EXPECTED_PROMPTS = 53_225

RESULTS_VOLUME_NAME = "chess-rl-eval-results-r6"
RESULTS_ROOT = "/results"
NAMESPACE = "ablation_pass16_v1"

SETTINGS = {
    "response_length": 2560,
    "temperature": 1,
    "n_samples": 16,
    "rollout": "vllm",
    "model_impl": "vllm",
    "multi_turn": True,
    "thinking": True,
    "tts": False,
    "seed": 0,
    "max_num_seqs": 2048,
    "max_num_batched_tokens": 131072,
    "gpu_memory": 0.45,
    "enforce_eager": False,
    "free_cache_engine": True,
}

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install("curl", "git", "vim", "htop")
    .pip_install(
        "wheel==0.46.3", "packaging==24.1", "ninja==1.13.0",
        "setuptools==71.1.0",
    )
    .pip_install(
        "torch==2.6.0", "torchvision==0.21.0", "torchaudio==2.6.0",
        "torchdata==0.11.0", "triton==3.2.0", "accelerate==1.13.0",
        "transformers==4.51.3", "tokenizers==0.21.4", "datasets==4.8.4",
        "huggingface_hub==0.36.2", "safetensors==0.7.0",
        "sentencepiece==0.2.1", "pyarrow==23.0.1", "pandas==2.3.3",
        "numpy==2.2.6", "scipy==1.17.1", "hydra-core==1.3.2",
        "omegaconf==2.3.0", "wandb==0.26.0", "mlflow==3.11.1",
        "tensordict==0.6.2", "ray[default]==2.47.1", "vllm==0.8.5",
        "xformers==0.0.29.post2", "peft==0.19.1", "dill==0.4.1",
        "codetiming==1.4.0", "pylatexenc==2.10", "pybind11==3.0.4",
        "chess==1.11.2", "pydantic==2.13.3", "openai==2.32.0",
        "compressed-tensors==0.9.3", "xgrammar==0.1.18",
        "outlines==0.1.11", "lm-format-enforcer==0.10.12",
    )
    .pip_install("flash-attn==2.7.4.post1", extra_options="--no-build-isolation")
    .add_local_dir(str(VERL_ROOT), remote_path=REMOTE_VERL_ROOT)
    .add_local_dir(str(EVAL_DATA_ROOT), remote_path=REMOTE_EVAL_DATA)
)

results_volume = modal.Volume.from_name(RESULTS_VOLUME_NAME, create_if_missing=True)
checkpoint_volume = modal.Volume.from_name(
    "rl-reasoning-checkpoints", create_if_missing=False
)

app = modal.App("chess-ablation-pass16-eval", image=image)


def _sha256_file(path: Path, chunk: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def _summarize(output_dir: Path, arm: str, n_samples: int) -> dict[str, Any]:
    """Group generations by prompt; compute pass@k, protocol and variance rates."""
    import numpy as np
    import pandas as pd

    rows: list[dict[str, Any]] = []
    for jsonl in sorted(output_dir.rglob("*.jsonl")):
        with jsonl.open() as handle:
            for line in handle:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    row["_src"] = jsonl.stem  # sub-dataset indexes restart at 0
                    rows.append(row)
    if not rows:
        raise RuntimeError("No generation rows found")

    def score_of(row: dict[str, Any]) -> float:
        for key in ("score", "reward", "acc", "correct"):
            if key in row and row[key] is not None:
                return float(row[key])
        raise KeyError(f"No score field in row keys={list(row)[:12]}")

    def prompt_key(row: dict[str, Any]) -> str:
        for key in ("index", "prompt_index", "sample_index", "idx"):
            if key in row and row[key] is not None:
                return f"i{int(row[key]) // max(1, n_samples) if key == 'sample_index' else row[key]}"
        for key in ("prompt", "input", "question"):
            if key in row and row[key]:
                return hashlib.sha256(str(row[key]).encode()).hexdigest()[:24]
        raise KeyError(f"No prompt key in row keys={list(row)[:12]}")

    def protocol_ok(row: dict[str, Any]) -> bool:
        text = str(
            row.get("response") or row.get("output") or row.get("responses") or ""
        )
        t_end = text.find("</T>")
        call = text.find("<call_env>")
        return 0 <= t_end < call

    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(f"{row.get('_src','')}:{prompt_key(row)}", []).append(row)

    per_prompt = []
    for key, group in groups.items():
        wins = sum(1 for r in group if score_of(r) > 0)
        proto = sum(1 for r in group if protocol_ok(r))
        per_prompt.append({"key": key, "n": len(group), "wins": wins, "proto": proto})
    frame = pd.DataFrame(per_prompt)
    n = frame["n"].to_numpy()
    wins = frame["wins"].to_numpy()

    def pass_at_k(k: int) -> float:
        # unbiased estimator: 1 - C(n-wins, k)/C(n, k), averaged over prompts
        from math import comb
        vals = []
        for total, won in zip(n, wins):
            kk = min(k, int(total))
            if won <= 0:
                vals.append(0.0)
            elif total - won < kk:
                vals.append(1.0)
            else:
                vals.append(1.0 - comb(int(total - won), kk) / comb(int(total), kk))
        return float(np.mean(vals))

    summary = {
        "schema": "ablation-pass16-summary-v1",
        "arm": arm,
        "prompts": int(len(frame)),
        "rows": int(frame["n"].sum()),
        "pass_at_1": pass_at_k(1),
        "pass_at_8": pass_at_k(8),
        "pass_at_16": pass_at_k(16),
        "avg_reward": float(wins.sum() / max(1, frame["n"].sum())),
        "format_rate": float(frame["proto"].sum() / max(1, frame["n"].sum())),
        "variance_rate": float(
            ((wins > 0) & (wins < n)).sum() / max(1, len(frame))
        ),
        "solved_prompts": int((wins > 0).sum()),
        "wins_histogram": {
            str(int(w)): int((wins == w).sum()) for w in sorted(set(wins))
        },
        "format_hits": int(frame["proto"].sum()),
    }
    return summary


@app.function(
    gpu="H200",
    cpu=16.0,
    memory=96 * 1024,
    timeout=60 * 60 * 20,
    retries=0,
    volumes={RESULTS_ROOT: results_volume, CKPT_MOUNT: checkpoint_volume},
)
def eval_arm(arm: str, n_samples: int = 16, dataset: str = DATASET) -> str:
    checkpoint_volume.reload()
    results_volume.reload()
    model_path = Path(CKPT_ROOT) / ARM_SLUGS[arm] / "final"
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(model_path)
    data_file = Path(REMOTE_EVAL_DATA) / f"{dataset}.parquet"
    if dataset == DATASET and _sha256_file(data_file) != DATASET_SHA256:
        raise RuntimeError("Eval dataset drifted from pinned balanced parquet")

    # The proven evaluator config handles ~1,480 prompts per dataset; larger
    # datasets OOM the worker. Slice this dataset into <=1,400-prompt
    # sub-datasets and let verl_eval.sh run them sequentially (B1-B5 style).
    import pandas as pd
    frame = pd.read_parquet(data_file)
    local_data = Path("/tmp/eval_data_local")
    if local_data.exists():
        shutil.rmtree(local_data)
    local_data.mkdir(parents=True)
    chunk = 1400
    sub_names = []
    for i in range(0, len(frame), chunk):
        name = f"{dataset}_part{i // chunk:03d}"
        frame.iloc[i : i + chunk].to_parquet(
            local_data / f"{name}.parquet", index=False
        )
        sub_names.append(name)
    print(f"[ablation-eval] {arm}: {len(frame)} prompts -> {len(sub_names)} sub-datasets")

    result_root = Path(RESULTS_ROOT) / NAMESPACE / arm / f"n{n_samples}" / dataset
    marker = result_root / "success.json"
    if marker.is_file():
        return json.dumps(json.loads(marker.read_text()))

    output_dir = Path("/tmp/ablation_eval") / arm
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    env = os.environ.copy()
    env.update({
        "PYTHONPATH": f"{REMOTE_VERL_ROOT}:{env.get('PYTHONPATH', '')}",
        "PYTHONUNBUFFERED": "1",
        "CUDA_VISIBLE_DEVICES": "0",
        "GPUS": "0",
        "N_GPUS": "1",
        "MODEL_PATH": str(model_path),
        "OUTPUT_DIR": str(output_dir),
        "EXPERIMENT_NAME": "eval",
        "EVAL_DATA_DIR": str(local_data),
        "EVAL_DATASETS": ",".join(sub_names),
        "RES_LENGTH": str(SETTINGS["response_length"]),
        "TEMPERATURE": str(SETTINGS["temperature"]),
        "N_SAMPLES": str(n_samples),
        "ROLLOUT_NAME": SETTINGS["rollout"],
        "VLLM_MODEL_IMPL": SETTINGS["model_impl"],
        "MULTI_TURN": str(SETTINGS["multi_turn"]),
        "THINKING": str(SETTINGS["thinking"]),
        "TTS": str(SETTINGS["tts"]),
        "SEED": str(SETTINGS["seed"]),
        "MAX_NUM_SEQS": str(SETTINGS["max_num_seqs"]),
        "MAX_NUM_BATCHED_TOKENS": str(SETTINGS["max_num_batched_tokens"]),
        "GPU_MEMORY": str(SETTINGS["gpu_memory"]),
        "MICRO_BATCH_SIZE": "32",
        "ENFORCE_EAGER": str(SETTINGS["enforce_eager"]),
        "FREE_CACHE_ENGINE": str(SETTINGS["free_cache_engine"]),
        "DEBUG": "False",
        "TOKENIZERS_PARALLELISM": "true",
        "NCCL_DEBUG": "WARN",
        "VLLM_LOGGING_LEVEL": "WARN",
    })
    command = ["bash", f"{REMOTE_VERL_ROOT}/verl/eval_bash/verl_eval.sh"]
    print(f"[ablation-eval] {arm} n={n_samples} dataset={dataset}")
    process = subprocess.Popen(
        command, cwd=REMOTE_VERL_ROOT, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        sys.stdout.write(line)
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"{arm} eval exited {return_code}")

    summary = _summarize(output_dir, arm, n_samples)
    summary.update({
        "dataset": dataset,
        "dataset_sha256": DATASET_SHA256 if dataset == DATASET else None,
        "model_path": str(model_path),
        "n_samples": n_samples,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "settings": SETTINGS,
    })

    result_root.mkdir(parents=True, exist_ok=True)
    # persist raw generations compressed alongside the summary
    import gzip
    for jsonl in sorted(output_dir.rglob("*.jsonl")):
        target = result_root / (jsonl.name + ".gz")
        with jsonl.open("rb") as src, gzip.open(target, "wb") as dst:
            shutil.copyfileobj(src, dst)
    for metrics in sorted(output_dir.rglob("*metrics*")):
        if metrics.is_file():
            shutil.copyfile(metrics, result_root / metrics.name)
    marker.write_text(json.dumps(summary, indent=2, sort_keys=True))
    results_volume.commit()
    print(json.dumps(summary, indent=2))
    return json.dumps(summary)


@app.local_entrypoint()
def main(action: str = "status", arm: str = "", n_samples: int = 16) -> None:
    if action == "canary":
        # one arm, n=1, full dataset: validates model load, env loop, scoring
        target = arm or "A1"
        print(eval_arm.remote(target, 1))
    elif action == "one":
        if not arm:
            raise ValueError("--arm required")
        handle = eval_arm.spawn(arm, n_samples)
        print(json.dumps({"arm": arm, "call": handle.object_id}))
    elif action == "launch":
        calls = {a: eval_arm.spawn(a, n_samples) for a in ARMS}
        print(json.dumps({a: c.object_id for a, c in calls.items()}, indent=2))
    elif action == "launch-sharded":
        shards = [f"eval_train_v4_balanced_shard{i}" for i in range(4)]
        calls = {}
        for a in ARMS:
            for shard in shards:
                calls[f"{a}/{shard[-6:]}"] = eval_arm.spawn(a, n_samples, shard)
        print(json.dumps({k: c.object_id for k, c in calls.items()}, indent=2))
    elif action == "merge":
        import subprocess as sp, tempfile, math as m
        matrix = {}
        with tempfile.TemporaryDirectory() as tmp:
            sp.run(["modal", "volume", "get", RESULTS_VOLUME_NAME, NAMESPACE, tmp,
                    "--force"], capture_output=True, text=True)
            root = Path(tmp) / NAMESPACE
            for a in ARMS:
                hist: dict[int, int] = {}
                fmt_hits = rows = 0
                shard_dirs = sorted((root / a).glob(f"n{n_samples}/*shard*")) if (root / a).exists() else []
                complete = 0
                for sd in shard_dirs:
                    marker = sd / "success.json"
                    if not marker.is_file():
                        continue
                    complete += 1
                    d = json.loads(marker.read_text())
                    for w, c in d.get("wins_histogram", {}).items():
                        hist[int(w)] = hist.get(int(w), 0) + int(c)
                    fmt_hits += d.get("format_hits", 0)
                    rows += d.get("rows", 0)
                if not hist:
                    matrix[a] = {"shards_done": complete, "status": "pending"}
                    continue
                prompts = sum(hist.values())
                def pak(k: int) -> float:
                    total = 0.0
                    for w, c in hist.items():
                        if w <= 0: v = 0.0
                        elif 16 - w < k: v = 1.0
                        else: v = 1.0 - m.comb(16 - w, k) / m.comb(16, k)
                        total += v * c
                    return total / prompts
                matrix[a] = {
                    "shards_done": complete,
                    "prompts": prompts,
                    "pass@1": round(pak(1), 4),
                    "pass@8": round(pak(8), 4),
                    "pass@16": round(pak(16), 4),
                    "format_rate": round(fmt_hits / max(1, rows), 4),
                    "variance_rate": round(
                        sum(c for w, c in hist.items() if 0 < w < 16) / prompts, 4
                    ),
                }
        print(json.dumps(matrix, indent=2))
    elif action == "status":
        import subprocess as sp
        out = sp.run(
            ["modal", "volume", "ls", RESULTS_VOLUME_NAME, NAMESPACE],
            capture_output=True, text=True,
        )
        print(out.stdout or out.stderr)
    else:
        raise ValueError(f"Unknown action: {action}")
