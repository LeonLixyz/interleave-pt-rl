"""Modal SFT for the 1B pre-to-post-olmo checkpoints via LLaMA-Factory.

Usage:
    # Smoke test — 20 steps on step95368 anneal to check per_device batch size:
    modal run sft.py::smoke \
        --base-model pre-to-post-olmo/math-1b-anneal-from-step95368 \
        --per-device-batch 32 --grad-accum 1

    # Single production run:
    modal run --detach sft.py::sft_single \
        --anchor 95368

    # 10-way sweep (all anneal ckpts, parallel via Function.spawn):
    modal run sft.py::sft_sweep
"""

from __future__ import annotations

import modal

from common import (
    CACHE_MOUNT,
    CACHE_VOLUME_NAME,
    CHECKPOINT_MOUNT,
    CHECKPOINT_VOLUME_NAME,
    hf_image_base,
)

LOCAL_SFT_REPO = "/Users/leonli66/Desktop/Research/RL/Chess RL/pre2post-LM-SFT"
REMOTE_SFT_REPO = "/root/pre2post-LM-SFT"
GPUS_PER_NODE = 8
HF_ORG = "pre-to-post-olmo"


def _img() -> modal.Image:
    return (
        modal.Image.from_registry(
            "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11"
        )
        .apt_install("build-essential", "git", "curl")
        .pip_install("wheel", "packaging", "ninja", "setuptools")
        .pip_install("torch==2.6.0")
        .pip_install("flash-attn==2.7.4.post1", extra_options="--no-build-isolation")
        .pip_install(
            "transformers>=4.46,<4.48",  # LLaMA-Factory + deepspeed compat window
            "accelerate>=1.0",
            "deepspeed>=0.15,<=0.16.9",  # transformers version check upper bound
            "liger-kernel>=0.5",  # chunked CE loss for 100k vocab
            "datasets>=3.0",
            "peft>=0.13",
            "trl>=0.11",
            "safetensors>=0.4",
            "sentencepiece>=0.2",
            "tiktoken>=0.7",
            "einops",
            "matplotlib",
            "wandb>=0.18",
            "huggingface_hub>=0.26",
            "mlflow>=2.16",
        )
        .add_local_dir(
            LOCAL_SFT_REPO,
            remote_path=REMOTE_SFT_REPO,
            copy=True,
            ignore=[".git", ".git/**", ".venv/**", "__pycache__/**", "saves/**", "*.jsonl"],
        )
        .run_commands(f"cd {REMOTE_SFT_REPO} && pip install -e '.[torch,metrics]'")
        .add_local_python_source("common")
    )


app = modal.App("math-1b-sft", image=_img())

cache_volume = modal.Volume.from_name(CACHE_VOLUME_NAME, create_if_missing=True)
checkpoint_volume = modal.Volume.from_name(
    CHECKPOINT_VOLUME_NAME, create_if_missing=True, version=2
)
hf_secret = modal.Secret.from_name("huggingface-secret")
wandb_secret = modal.Secret.from_name("wandb-secret")


def _train(
    *,
    base_model: str,
    output_repo: str | None,
    per_device_batch: int,
    grad_accum: int,
    learning_rate: float,
    epochs: float,
    warmup_ratio: float,
    seq_len: int,
    save_strategy: str,
    benchmark_steps: int | None,
    run_name: str,
    yaml_path: str = "examples/train_full/olmo_sft_1b.yaml",
) -> dict:
    """Actual worker — runs `llamafactory-cli train` with overrides."""
    import json
    import os
    import shutil
    import subprocess
    from pathlib import Path

    cache_volume.reload()
    checkpoint_volume.reload()

    # Env
    os.environ["HF_HOME"] = f"{CACHE_MOUNT}/hf"
    os.environ["HF_DATASETS_CACHE"] = f"{CACHE_MOUNT}/hf/datasets"
    os.environ["TRANSFORMERS_CACHE"] = f"{CACHE_MOUNT}/hf/transformers"
    os.environ["WANDB_PROJECT"] = os.environ.get("WANDB_PROJECT", "math-pretraining-sft")
    os.environ["WANDB_RUN_ID"] = run_name

    output_dir = f"{CHECKPOINT_MOUNT}/sft/{run_name}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for d in (
        f"{CACHE_MOUNT}/hf",
        f"{CACHE_MOUNT}/hf/datasets",
        f"{CACHE_MOUNT}/hf/transformers",
    ):
        Path(d).mkdir(parents=True, exist_ok=True)

    # Build the CLI overrides — LLaMA-Factory supports key=value tail args
    cmd = [
        "llamafactory-cli", "train",
        yaml_path,
        f"model_name_or_path={base_model}",
        f"output_dir={output_dir}",
        f"per_device_train_batch_size={per_device_batch}",
        f"gradient_accumulation_steps={grad_accum}",
        f"learning_rate={learning_rate}",
        f"num_train_epochs={epochs}",
        f"warmup_ratio={warmup_ratio}",
        f"cutoff_len={seq_len}",
        f"run_name={run_name}",
    ]
    # save_strategy override: only pass if not the default; YAML/CLI "no" gets
    # parsed as Python False, so use enum-safe strings.
    if save_strategy and save_strategy not in ("no", "false"):
        cmd += [f"save_strategy={save_strategy}"]
    if benchmark_steps is not None:
        cmd += [f"max_steps={benchmark_steps}"]

    print("[sft] cmd:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=REMOTE_SFT_REPO, check=False)
    if result.returncode != 0:
        return {
            "error": f"llamafactory-cli returned {result.returncode}",
            "run_name": run_name,
            "output_dir": output_dir,
        }
    checkpoint_volume.commit()

    out = {
        "run_name": run_name,
        "output_dir": output_dir,
        "base_model": base_model,
        "per_device_batch": per_device_batch,
        "grad_accum": grad_accum,
        "effective_batch": per_device_batch * grad_accum * GPUS_PER_NODE,
        "returncode": 0,
    }

    # Upload to HF if requested
    if output_repo and benchmark_steps is None:
        from huggingface_hub import HfApi, create_repo
        token = os.environ["HF_TOKEN"]
        api = HfApi(token=token)
        repo_id = output_repo if "/" in output_repo else f"{HF_ORG}/{output_repo}"
        create_repo(repo_id=repo_id, token=token, exist_ok=True, repo_type="model")
        print(f"[sft] uploading {output_dir} -> {repo_id}")
        api.upload_folder(
            folder_path=output_dir,
            repo_id=repo_id,
            commit_message=f"SFT: {base_model} -> {run_name}",
            ignore_patterns=[
                "checkpoint-*", "checkpoint-*/**",
                "*.pt", "*optimizer*", "global_step*",
            ],
        )
        out["hf_repo"] = repo_id
    return out


@app.function(
    gpu=f"H200:{GPUS_PER_NODE}",
    timeout=60 * 60 * 12,
    volumes={
        CACHE_MOUNT: cache_volume,
        CHECKPOINT_MOUNT: checkpoint_volume,
    },
    secrets=[hf_secret, wandb_secret],
)
def sft_train(
    base_model: str,
    output_repo: str | None = None,
    per_device_batch: int = 16,
    grad_accum: int = 2,
    learning_rate: float = 1.0e-5,
    epochs: float = 3.0,
    warmup_ratio: float = 0.1,
    seq_len: int = 8192,
    save_strategy: str = "no",
    benchmark_steps: int | None = None,
    run_name: str = "sft-olmo-1b",
    yaml_path: str = "examples/train_full/olmo_sft_1b.yaml",
) -> dict:
    return _train(
        base_model=base_model,
        output_repo=output_repo,
        yaml_path=yaml_path,
        per_device_batch=per_device_batch,
        grad_accum=grad_accum,
        learning_rate=learning_rate,
        epochs=epochs,
        warmup_ratio=warmup_ratio,
        seq_len=seq_len,
        save_strategy=save_strategy,
        benchmark_steps=benchmark_steps,
        run_name=run_name,
    )


ANCHORS = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 95368]
ANCHORS_5K = [5000, 15000, 25000, 35000, 45000, 55000, 65000, 75000, 85000, 95000]


@app.local_entrypoint()
def smoke(
    base_model: str = "pre-to-post-olmo/math-1b-anneal-from-step95368",
    per_device_batch: int = 32,
    grad_accum: int = 1,
    benchmark_steps: int = 20,
) -> None:
    """Fire a short training run to test if per_device_batch fits."""
    r = sft_train.remote(
        base_model=base_model,
        per_device_batch=per_device_batch,
        grad_accum=grad_accum,
        benchmark_steps=benchmark_steps,
        run_name=f"smoke-pd{per_device_batch}-ga{grad_accum}",
    )
    print(r)


@app.local_entrypoint()
def sft_single(anchor: int = 95368,
                per_device_batch: int = 16,
                grad_accum: int = 2) -> None:
    base = f"pre-to-post-olmo/math-1b-anneal-from-step{anchor}"
    out_repo = f"math-1b-sft-openthoughts-from-step{anchor}"
    r = sft_train.remote(
        base_model=base,
        output_repo=out_repo,
        per_device_batch=per_device_batch,
        grad_accum=grad_accum,
        run_name=out_repo,
    )
    print(r)


@app.local_entrypoint()
def sft_numinamath(
    anchor: int = 95368,
    base_model: str | None = None,
    output_repo: str | None = None,
    per_device_batch: int = 8,
    grad_accum: int = 4,
) -> None:
    """SFT on AI-MO/NuminaMath-CoT (unpacked, 1 epoch)."""
    base = base_model or f"pre-to-post-olmo/math-1b-anneal-from-step{anchor}"
    out_repo = output_repo or f"math-1b-sft-numinamath-from-step{anchor}"
    r = sft_train.remote(
        base_model=base,
        output_repo=out_repo,
        per_device_batch=per_device_batch,
        grad_accum=grad_accum,
        learning_rate=1.0e-5,
        epochs=1.0,
        warmup_ratio=0.03,
        run_name=out_repo,
        yaml_path="examples/train_full/olmo_sft_1b_numinamath.yaml",
    )
    print(r)


@app.local_entrypoint()
def sft_sweep(per_device_batch: int = 16, grad_accum: int = 2) -> None:
    """Fire 10 SFT jobs in parallel (one per 4096-anneal anchor)."""
    print(f"firing {len(ANCHORS)} SFT jobs in parallel (pd={per_device_batch}, ga={grad_accum})")
    futures = []
    for a in ANCHORS:
        base = f"pre-to-post-olmo/math-1b-anneal-from-step{a}"
        out_repo = f"math-1b-sft-openthoughts-from-step{a}"
        f = sft_train.spawn(
            base_model=base,
            output_repo=out_repo,
            per_device_batch=per_device_batch,
            grad_accum=grad_accum,
            run_name=out_repo,
        )
        futures.append((a, f))
    for a, f in futures:
        try:
            r = f.get()
            print(f"  ✓ from step{a}: {r.get('hf_repo')}")
        except Exception as e:
            print(f"  ✗ from step{a}: {e}")


@app.local_entrypoint()
def sft_sweep_5k(per_device_batch: int = 8, grad_accum: int = 4) -> None:
    """Fire 10 NuminaMath SFT jobs on the every-5k-stride anneal anchors.

    Base repos: pre-to-post-olmo/math-1b-anneal-from-step{5000,15000,...,95000}
    (must be uploaded to HF first — see upload_to_hf.py::upload_anneal_5k).
    Output repos: math-1b-sft-numinamath-bs512-from-step{step} (same schema as 10k stride).
    """
    print(f"firing {len(ANCHORS_5K)} NuminaMath SFT jobs (5k stride) in parallel (pd={per_device_batch}, ga={grad_accum})")
    futures = []
    for a in ANCHORS_5K:
        base = f"pre-to-post-olmo/math-1b-anneal-from-step{a}"
        out_repo = f"math-1b-sft-numinamath-bs512-from-step{a}"
        f = sft_train.spawn(
            base_model=base,
            output_repo=out_repo,
            per_device_batch=per_device_batch,
            grad_accum=grad_accum,
            learning_rate=1.0e-5,
            epochs=1.0,
            warmup_ratio=0.03,
            run_name=out_repo,
            yaml_path="examples/train_full/olmo_sft_1b_numinamath.yaml",
        )
        futures.append((a, f))
    for a, f in futures:
        try:
            r = f.get()
            print(f"  ✓ from step{a}: {r.get('hf_repo')}")
        except Exception as e:
            print(f"  ✗ from step{a}: {e}")


@app.local_entrypoint()
def sft_sweep_8192(per_device_batch: int = 8, grad_accum: int = 4) -> None:
    """Fire 10 SFT jobs on the 8192-anneal models. Smaller pd because seq_len
    matches SFT at 8192 → activations are the same but base model context extends
    further; keeps within memory budget on 1B/8k pipeline."""
    print(f"firing {len(ANCHORS)} SFT jobs (8192 anneal) in parallel (pd={per_device_batch}, ga={grad_accum})")
    futures = []
    for a in ANCHORS:
        base = f"pre-to-post-olmo/math-1b-anneal8192-from-step{a}"
        out_repo = f"math-1b-sft-openthoughts-8192-from-step{a}"
        f = sft_train.spawn(
            base_model=base,
            output_repo=out_repo,
            per_device_batch=per_device_batch,
            grad_accum=grad_accum,
            run_name=out_repo,
        )
        futures.append((a, f))
    for a, f in futures:
        try:
            r = f.get()
            print(f"  ✓ from step{a}: {r.get('hf_repo')}")
        except Exception as e:
            print(f"  ✗ from step{a}: {e}")
