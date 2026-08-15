"""
Modal launcher: 5 alpha=0.030 configs at C_total=6.5e18.

Jobs (5 total):
  - 5m_alpha0.030  (6.324B tokens)
  - 10m_alpha0.030 (2.748B tokens)
  - 20m_alpha0.030 (1.585B tokens)
  - 32m_alpha0.030 (1.032B tokens)
  - 50m_alpha0.030 (690M tokens)

Each job: 8x H200, bf16, lr=1e-3 cosine to eta_min=1e-4.

Usage:
  modal run modal_scripts/launch_6p5e18_small_alpha0p03.py --dry-run
  modal run --detach modal_scripts/launch_6p5e18_small_alpha0p03.py
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
    "train",
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


def _run_training(config: str, num_gpus: int, overrides: list[str] = None):
    import subprocess
    import sys as _sys
    import yaml

    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

    config_path = Path("/root/chess") / config
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    exp_name = cfg.get("training", {}).get("experiment_name")
    if overrides:
        for override in overrides:
            if override.startswith("training.experiment_name="):
                exp_name = override.split("=", 1)[1]
                break
    if exp_name:
        run_save_dir = Path(OUTPUT_DIR) / exp_name
        final_dir = run_save_dir / "final"
        if final_dir.is_dir():
            print(f"[launch] Skipping {config}: final exists at {final_dir}")
            return

    print(f"[launch] Starting: {config} on {num_gpus}x {GPU_TYPE}")

    cmd = [
        "accelerate", "launch",
        "--multi_gpu",
        "--num_processes", str(num_gpus),
        "--mixed_precision", MIXED_PRECISION,
        "scripts/train/train_hf.py",
        "--config", config,
        "--auto_resume",
        "--data_dir", DATA_DIR,
        "--output_dir", OUTPUT_DIR,
        "--test_data_dir", "/data/test",
        "--max_checkpoints", str(MAX_CHECKPOINTS),
    ]
    if overrides:
        for override in overrides:
            cmd.extend(["--override", override])

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
    max_containers=8,
)
def train_h200(config: str, overrides: list[str] = None):
    _run_training(config, GPUS_PER_NODE, overrides)


JOBS = [
    "config/configs/6p5e18_small/5m_alpha0.030.yaml",   # 6.324B
    "config/configs/6p5e18_small/10m_alpha0.030.yaml",  # 2.748B
    "config/configs/6p5e18_small/20m_alpha0.030.yaml",  # 1.585B
    "config/configs/6p5e18_small/32m_alpha0.030.yaml",  # 1.032B
    "config/configs/6p5e18_small/50m_alpha0.030.yaml",  # 690M
]


def _overrides() -> list[str]:
    return [
        "training.gpu_peak_tflops=989",
        "training.cache_size=0",
        "training.mixed_precision=bf16",
    ]


@app.local_entrypoint()
def main(dry_run: bool = False):
    print(f"alpha=0.030 6p5e18_small launcher: {len(JOBS)} jobs")
    print(f"GPU: {GPUS_PER_NODE}x {GPU_TYPE} per job")
    print(f"Data: {DATA_DIR}")
    print(f"Checkpoints: {OUTPUT_DIR}")
    print("=" * 70)
    for cfg in JOBS:
        print(f"  {Path(cfg).stem}")
    print("=" * 70)

    if dry_run:
        print("(dry-run -- nothing launched)")
        return

    handles = []
    for cfg in JOBS:
        handle = train_h200.spawn(config=cfg, overrides=_overrides())
        handles.append((Path(cfg).stem, handle))
        print(f"  SPAWNED: {Path(cfg).stem}")

    print(f"\n{len(handles)} jobs spawned. Waiting for completion...")
    failed, succeeded = [], []
    for name, handle in handles:
        try:
            handle.get()
            succeeded.append(name)
            print(f"  OK: {name}")
        except Exception as e:
            print(f"  FAILED: {name}: {e}")
            failed.append(name)

    print(f"\nDone: {len(succeeded)}/{len(JOBS)} succeeded")
    if failed:
        print(f"Failed: {', '.join(failed)}")
