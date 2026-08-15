"""
Modal launcher: relaunch 200m/410m/680m_C_6p5e18_alpha0.100 with lr=1e-3
and eta_min=1e-4 overrides (matches 6p5e18_small/ + 6p5e19_leon/ convention).

Original yamls have lr=2e-3, eta_min=1e-5. Old finished runs were renamed to
*_lr2e-3/ on the volume so these relaunches take the original names.

8x H200, bf16. Modal app name: "pretrain".

Usage:
  modal run modal_scripts/launch_200_410_680m_alpha0p1_relaunch_lr1e-3.py --dry-run
  modal run --detach modal_scripts/launch_200_410_680m_alpha0p1_relaunch_lr1e-3.py
"""
import os
from pathlib import Path

import modal

GPUS_PER_NODE = 8
GPU_TYPE = "H200"
TIMEOUT_HOURS = 47
MAX_CHECKPOINTS = 3
MIXED_PRECISION = "bf16"

DATA_DIR = "/data/pretrain_v1_20b"
OUTPUT_DIR = "/checkpoints/6p5e18"

cuda_version = "12.4.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

repo_dir = Path(__file__).parent.parent

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .apt_install("curl", "git", "vim", "htop")
    .pip_install(
        "torch>=2.6.0",
        "accelerate>=1.10.0",
        "transformers>=4.50.0",
        "datasets>=3.0.0",
        "pyarrow>=17.0.0",
        "pandas>=2.0.0",
        "pyyaml>=6.0",
        "omegaconf>=2.3.0",
        "wandb>=0.19.0",
        "einops>=0.7.0",
        "tokenizers>=0.19.0",
        "tqdm>=4.66.0",
        "chess>=1.11.0",
        "numpy>=2.0.0",
        "safetensors>=0.5.0",
        "sentencepiece>=0.2.0",
        "huggingface-hub>=0.28.0",
    )
    .run_commands(
        'python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained(\'Qwen/Qwen3-0.6B\')"'
    )
    .add_local_dir(str(repo_dir / "scripts"), remote_path="/root/chess/scripts")
    .add_local_dir(str(repo_dir / "training"), remote_path="/root/chess/training")
    .add_local_dir(str(repo_dir / "config"), remote_path="/root/chess/config")
    .add_local_dir(str(repo_dir / "llm_tokens"), remote_path="/root/chess/llm_tokens")
    .add_local_dir(str(repo_dir / "evaluation"), remote_path="/root/chess/evaluation")
)

env_file = repo_dir / ".env"
if env_file.exists():
    image = image.add_local_file(str(env_file), remote_path="/root/chess/.env")

data_volume = modal.Volume.from_name("rl-reasoning-training-data", create_if_missing=True)
ckpt_volume = modal.Volume.from_name("rl-reasoning-checkpoints", create_if_missing=True)

app = modal.App(
    "pretrain",
    image=image,
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("huggingface-secret"),
    ],
    volumes={
        "/data": data_volume,
        "/checkpoints": ckpt_volume,
    },
)


def _run_training(config: str, num_gpus: int, overrides: list[str]):
    import shutil
    import subprocess
    import sys as _sys
    import yaml

    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

    # Force fresh volume snapshot so we see prior renames/deletions.
    ckpt_volume.reload()

    config_path = Path("/root/chess") / config
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    exp_name = cfg.get("training", {}).get("experiment_name")
    if exp_name:
        run_save_dir = Path(OUTPUT_DIR) / exp_name
        if run_save_dir.exists():
            print(f"[launch] Wiping pre-existing run dir: {run_save_dir}")
            shutil.rmtree(run_save_dir)
            ckpt_volume.commit()
            ckpt_volume.reload()

    print(f"[launch] Starting: {config} on {num_gpus}x {GPU_TYPE}")
    print(f"[launch] OVERRIDES: {overrides}")

    cmd = [
        "accelerate", "launch",
        "--multi_gpu",
        "--num_processes", str(num_gpus),
        "--mixed_precision", MIXED_PRECISION,
        "scripts/train/train_hf.py",
        "--config", config,
        "--data_dir", DATA_DIR,
        "--output_dir", OUTPUT_DIR,
        "--test_data_dir", "/data/test",
        "--max_checkpoints", str(MAX_CHECKPOINTS),
    ]
    if overrides:
        cmd.append("--override")
        cmd.extend(overrides)

    print(f"  cmd: {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(cmd, cwd="/root/chess", stdout=_sys.stdout, stderr=_sys.stderr)
    rc = proc.wait()

    ckpt_volume.commit()
    if rc != 0:
        raise RuntimeError(f"Training failed (exit {rc}): {config}")
    print(f"[launch] Done: {config}")


@app.function(
    gpu=f"{GPU_TYPE}:{GPUS_PER_NODE}",
    timeout=60 * 60 * TIMEOUT_HOURS,
    retries=modal.Retries(initial_delay=0.0, max_retries=5),
)
def train_h200(config: str, overrides: list[str]):
    _run_training(config, GPUS_PER_NODE, overrides)


@app.local_entrypoint()
def main(dry_run: bool = False):
    configs = [
        "config/configs/6p5e18/200m_alpha0.100.yaml",
        "config/configs/6p5e18/410m_alpha0.100.yaml",
        "config/configs/6p5e18/680m_alpha0.100.yaml",
    ]
    overrides = [
        "training.gpu_peak_tflops=989",
        "training.cache_size=0",
        "training.mixed_precision=bf16",
        "training.optimizer.lr=1e-3",       # OVERRIDE: was 2e-3 in yaml
        "training.scheduler.eta_min=1e-4",  # OVERRIDE: was 1e-5 in yaml
    ]
    print(f"Relaunch 200m/410m/680m_alpha0.100 (6p5e18) at lr=1e-3, eta_min=1e-4")
    for c in configs:
        print(f"  config: {c}")
    print(f"  GPU: {GPUS_PER_NODE}x {GPU_TYPE}")
    print(f"  overrides: {overrides}")

    if dry_run:
        print("(dry-run -- nothing launched)")
        return

    for cfg in configs:
        handle = train_h200.spawn(config=cfg, overrides=overrides)
        print(f"SPAWNED: {cfg} (handle={handle.object_id})")
