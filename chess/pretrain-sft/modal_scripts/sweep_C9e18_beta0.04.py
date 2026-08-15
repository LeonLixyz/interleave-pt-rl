"""
Modal sweep launcher: C_total = 9e18, beta = 0.04

Spawns all 14 pretraining jobs (200M, 300M, 500M x alpha allocations)
as concurrent Modal functions. Each job gets its own checkpoint directory
on the shared volume and auto-resumes on retry.

Usage:
  # Launch all 14 jobs
  modal run modal_scripts/sweep_C9e18_beta0.04.py

  # Launch only 500M jobs
  modal run modal_scripts/sweep_C9e18_beta0.04.py --model-filter 500m

  # Dry run — just print what would launch
  modal run modal_scripts/sweep_C9e18_beta0.04.py --dry-run
"""
import os, sys, time
from pathlib import Path

import modal

# --------------------------------------------------------------------------- #
#  GPU config per model size                                                   #
# --------------------------------------------------------------------------- #

GPU_CONFIG = {
    "200m": ("H100", 4),
    "300m": ("H100", 4),
    "500m": ("H100", 8),
}
DEFAULT_GPU = ("H100", 8)

TIMEOUT_HOURS = 48
MAX_CHECKPOINTS = 3
MIXED_PRECISION = "bf16"

# --------------------------------------------------------------------------- #
#  Modal image (same as train.py)                                              #
# --------------------------------------------------------------------------- #

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

# --------------------------------------------------------------------------- #
#  Modal resources                                                             #
# --------------------------------------------------------------------------- #

data_volume = modal.Volume.from_name("rl-reasoning-training-data", create_if_missing=True)
ckpt_volume = modal.Volume.from_name("rl-reasoning-checkpoints", create_if_missing=True)

app = modal.App(
    "chess-sweep-C9e18",
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


# --------------------------------------------------------------------------- #
#  Training functions — one per GPU config                                     #
# --------------------------------------------------------------------------- #

def _run_training(config: str, num_gpus: int):
    """Run a single training job inside a Modal container."""
    import subprocess, sys as _sys

    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

    print(f"[sweep] Starting: {config} on {num_gpus} GPUs")

    cmd = [
        "accelerate", "launch",
        "--multi_gpu",
        "--num_processes", str(num_gpus),
        "--mixed_precision", MIXED_PRECISION,
        "scripts/train/train_hf.py",
        "--config", config,
        "--auto_resume",
        "--data_dir", "/data/tokenized",
        "--output_dir", f"/checkpoints/C9e18_beta0.04",
        "--test_data_dir", "/data/test",
        "--max_checkpoints", str(MAX_CHECKPOINTS),
    ]

    print(f"  cmd: {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(cmd, cwd="/root/chess", stdout=_sys.stdout, stderr=_sys.stderr)
    rc = proc.wait()

    ckpt_volume.commit()
    if rc != 0:
        raise RuntimeError(f"Training failed (exit {rc}): {config}")
    print(f"[sweep] Done: {config}")


@app.function(
    gpu="H100:4",
    timeout=60 * 60 * TIMEOUT_HOURS,
    retries=modal.Retries(initial_delay=0.0, max_retries=5),
)
def train_4gpu(config: str):
    _run_training(config, 4)


@app.function(
    gpu="H100:8",
    timeout=60 * 60 * TIMEOUT_HOURS,
    retries=modal.Retries(initial_delay=0.0, max_retries=5),
)
def train_8gpu(config: str):
    _run_training(config, 8)


# --------------------------------------------------------------------------- #
#  Sweep configs                                                               #
# --------------------------------------------------------------------------- #

SWEEP_CONFIGS = [
    ("200m", "config/configs/C9e18_beta0.04/200m_alpha0.048.yaml"),
    ("200m", "config/configs/C9e18_beta0.04/200m_alpha0.276.yaml"),
    ("200m", "config/configs/C9e18_beta0.04/200m_alpha0.504.yaml"),
    ("200m", "config/configs/C9e18_beta0.04/200m_alpha0.732.yaml"),
    ("300m", "config/configs/C9e18_beta0.04/300m_alpha0.048.yaml"),
    ("300m", "config/configs/C9e18_beta0.04/300m_alpha0.276.yaml"),
    ("300m", "config/configs/C9e18_beta0.04/300m_alpha0.504.yaml"),
    ("300m", "config/configs/C9e18_beta0.04/300m_alpha0.732.yaml"),
    ("300m", "config/configs/C9e18_beta0.04/300m_alpha0.960.yaml"),
    ("500m", "config/configs/C9e18_beta0.04/500m_alpha0.048.yaml"),
    ("500m", "config/configs/C9e18_beta0.04/500m_alpha0.276.yaml"),
    ("500m", "config/configs/C9e18_beta0.04/500m_alpha0.504.yaml"),
    ("500m", "config/configs/C9e18_beta0.04/500m_alpha0.732.yaml"),
    ("500m", "config/configs/C9e18_beta0.04/500m_alpha0.960.yaml"),
]


# --------------------------------------------------------------------------- #
#  Entrypoint                                                                  #
# --------------------------------------------------------------------------- #

@app.local_entrypoint()
def main(model_filter: str = "", dry_run: bool = False):
    jobs = []
    for model_size, config_path in SWEEP_CONFIGS:
        if model_filter and model_size != model_filter:
            continue
        gpu_type, num_gpus = GPU_CONFIG.get(model_size, DEFAULT_GPU)
        jobs.append((model_size, config_path, num_gpus))

    print(f"C9e18 beta=0.04 sweep: {len(jobs)} jobs")
    print("=" * 60)
    for model_size, config_path, num_gpus in jobs:
        name = Path(config_path).stem
        print(f"  {name:30s}  {num_gpus}x H100")
    print("=" * 60)

    if dry_run:
        print("(dry-run — nothing launched)")
        return

    # Spawn all jobs concurrently
    handles = []
    for model_size, config_path, num_gpus in jobs:
        fn = train_4gpu if num_gpus <= 4 else train_8gpu
        handle = fn.spawn(config_path)
        handles.append((Path(config_path).stem, handle))
        print(f"  spawned: {Path(config_path).stem}")

    print(f"\n{len(handles)} jobs spawned. Waiting for completion...")

    # Wait for all and report results
    failed = []
    for name, handle in handles:
        try:
            handle.get()
            print(f"  OK: {name}")
        except Exception as e:
            print(f"  FAILED: {name}: {e}")
            failed.append(name)

    print(f"\nDone: {len(handles) - len(failed)}/{len(handles)} succeeded")
    if failed:
        print(f"Failed: {', '.join(failed)}")
