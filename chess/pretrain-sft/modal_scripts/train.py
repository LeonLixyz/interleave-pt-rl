"""
Modal cloud training launcher for chess reasoning pretraining.

Features:
  1. Single or multi-GPU distributed training via Accelerate
  2. Auto-resume from latest checkpoint on Modal Volume
  3. Fault tolerance via Modal retries + checkpoint resume
  4. Data download from HuggingFace to Modal Volume

Configuration:
  Edit GPUS_PER_NODE, GPU_TYPE at the top of this file.

Usage:
  # Download data to Modal volume (first time only)
  modal run modal_scripts/train.py::download_data --max-shards 64

  # Download test data (for evaluation during training)
  modal run modal_scripts/train.py::download_test_data

  # Launch training (uses CONFIG_FILE by default)
  modal run modal_scripts/train.py

  # Launch with a specific config and shard count
  modal run modal_scripts/train.py --config config/configs/pretrain_sl/qwen3_100m.yaml --num-shards 2000
"""
import os
from pathlib import Path

import modal

# --------------------------------------------------------------------------- #
#  Cluster configuration — edit these before launching                         #
# --------------------------------------------------------------------------- #

GPUS_PER_NODE = 8          # GPUs per container
GPU_TYPE = "H100"          # H100, H200, A100, etc.
TIMEOUT_HOURS = 24         # max wall-clock time
MAX_CHECKPOINTS = 3        # checkpoints to keep on volume (0 = unlimited)
MIXED_PRECISION = "bf16"   # bf16, fp16, or no

CONFIG_FILE = "config/configs/pretrain_sl/qwen3_10m.yaml"  # default training config

# --------------------------------------------------------------------------- #
#  Modal image                                                                 #
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
    # Pre-cache the Qwen tokenizer
    .run_commands(
        'python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained(\'Qwen/Qwen3-0.6B\')"'
    )
    # Add project source code
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
    "rl-reasoning-training",
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
#  Training function                                                           #
# --------------------------------------------------------------------------- #

@app.function(
    gpu=f"{GPU_TYPE}:{GPUS_PER_NODE}",
    timeout=60 * 60 * TIMEOUT_HOURS,
    retries=modal.Retries(initial_delay=0.0, max_retries=5),
)
def train(config: str = CONFIG_FILE, num_shards: int = None):
    import subprocess, sys

    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

    print(f"[Modal] Starting chess reasoning training")
    print(f"  Config       : {config}")
    print(f"  GPUs         : {GPUS_PER_NODE}")
    print(f"  Data         : /data/tokenized")
    print(f"  Checkpoints  : /checkpoints")

    # Build accelerate launch command
    cmd = [
        "accelerate", "launch",
        "--multi_gpu",
        "--num_processes", str(GPUS_PER_NODE),
        "--mixed_precision", MIXED_PRECISION,
        "scripts/train/train_hf.py",
        "--config", config,
        "--auto_resume",
        "--data_dir", "/data/tokenized",
        "--output_dir", "/checkpoints",
        "--max_checkpoints", str(MAX_CHECKPOINTS),
        "--test_data_dir", "/data/test",
    ]
    if num_shards is not None:
        cmd.extend(["--override", f"data.num_shards={num_shards}"])

    print(f"  Command      : {' '.join(cmd)}")
    print("=" * 60, flush=True)

    # Stream output in real time so Modal captures logs
    proc = subprocess.Popen(
        cmd,
        cwd="/root/chess",
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    returncode = proc.wait()

    # Commit checkpoints to persistent volume regardless of exit code
    ckpt_volume.commit()

    if returncode != 0:
        raise RuntimeError(f"Training failed with exit code {returncode}")

    print("[Modal] Training complete, checkpoint volume committed.")


# --------------------------------------------------------------------------- #
#  Data download helper                                                        #
# --------------------------------------------------------------------------- #

@app.function(
    timeout=60 * 60 * 4,  # 4 hours for large downloads
    volumes={"/data": data_volume},
)
def download_data(max_shards: int = None, workers: int = 16):
    """Download tokenized training data from HuggingFace to Modal volume."""
    import sys
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from huggingface_hub import HfApi, hf_hub_download
    from tqdm import tqdm

    repo_id = "Evangelinejy/chess-train-data-balanced-tokenized"
    out_dir = Path("/data/tokenized")
    out_dir.mkdir(parents=True, exist_ok=True)

    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("ERROR: HF_TOKEN not set. Add huggingface-secret to Modal.")

    api = HfApi(token=token)
    all_files = sorted(
        f for f in api.list_repo_files(repo_id, repo_type="dataset")
        if f.endswith((".npy", ".txt", ".parquet"))
    )

    if max_shards is not None:
        all_files = all_files[:max_shards]

    already = sum(1 for f in all_files if (out_dir / f).exists())
    todo = [f for f in all_files if not (out_dir / f).exists()]
    print(f"Shards: {len(all_files)} total, {already} present, {len(todo)} to fetch")

    if not todo:
        print("All shards already present.")
        return

    def dl(filename):
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            token=token,
            local_dir=str(out_dir),
        )
        return filename

    errors = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(dl, f): f for f in todo}
        with tqdm(total=len(todo), unit="shard") as bar:
            for fut in as_completed(futures):
                fname = futures[fut]
                try:
                    fut.result()
                except Exception as e:
                    errors.append((fname, str(e)))
                    tqdm.write(f"ERROR {fname}: {e}")
                bar.update(1)

    data_volume.commit()
    print(f"Done. {len(todo) - len(errors)} downloaded, {len(errors)} failed.")
    if errors:
        for f, e in errors:
            print(f"  FAILED: {f}: {e}")


# --------------------------------------------------------------------------- #
#  Test data download                                                          #
# --------------------------------------------------------------------------- #

@app.function(
    timeout=60 * 60 * 2,
    volumes={"/data": data_volume},
)
def download_test_data():
    """Download test datasets from HuggingFace."""
    import sys
    from huggingface_hub import snapshot_download

    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("ERROR: HF_TOKEN not set. Add huggingface-secret to Modal.")

    out_dir = Path("/data/test")
    out_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id="Evangelinejy/chess-test-data",
        repo_type="dataset",
        token=token,
        local_dir=str(out_dir),
    )

    data_volume.commit()
    print(f"Test data downloaded to {out_dir}")


# --------------------------------------------------------------------------- #
#  Local entrypoint                                                            #
# --------------------------------------------------------------------------- #

@app.local_entrypoint()
def main(config: str = CONFIG_FILE, num_shards: int = None):
    train.remote(config=config, num_shards=num_shards)
