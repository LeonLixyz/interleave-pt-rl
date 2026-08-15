"""
Modal sweep launcher: C_total = 9e18, beta = 0.03

Runs all 16 pretraining jobs (100M, 200M, 400M, 800M x alpha allocations)
on 8x H200 GPUs each, with max 8 concurrent jobs.

Features:
  - Job queuing: max 8 concurrent, remaining jobs wait in queue
  - Batch size tuning: per-model-size overrides for H200
  - New dataset: chess-pre-to-post/pretraining_dataset_v1_tokenized
  - Auto HuggingFace upload: checkpoints pushed to chess-pre-to-post/<experiment_name>
  - Full shard loading: cache_size=0 (auto-detect) to use all tokens per shard

Usage:
  # Download new dataset to Modal volume (first time)
  modal run modal_scripts/sweep_C9e18_beta003.py::download_pretraining_v1

  # Launch all 16 jobs (max 8 concurrent)
  modal run modal_scripts/sweep_C9e18_beta003.py

  # Launch only 800m jobs
  modal run modal_scripts/sweep_C9e18_beta003.py --model-filter 800m

  # Dry run — print what would launch
  modal run modal_scripts/sweep_C9e18_beta003.py --dry-run
"""
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import modal

# --------------------------------------------------------------------------- #
#  Configuration                                                               #
# --------------------------------------------------------------------------- #

GPUS_PER_NODE = 8
GPU_TYPE = "H200"
TIMEOUT_HOURS = 47  # Modal max is 48h
MAX_CHECKPOINTS = 3
MIXED_PRECISION = "bf16"

# Per-model batch size config for H200 (update after running benchmark)
# Target: maximize per-GPU batch, set grad_accum=1 where possible
# All models: bs=32, ga=1, effective batch = 32 * 8 GPUs = 256
BATCH_CONFIG = {
    "100m": {"batch_size": 32, "gradient_accumulation_steps": 1},
    "200m": {"batch_size": 32, "gradient_accumulation_steps": 1},
    "400m": {"batch_size": 32, "gradient_accumulation_steps": 1},
    "800m": {"batch_size": 32, "gradient_accumulation_steps": 1},
}

DATA_DIR = "/data/pretrain_v1_20b"
OUTPUT_DIR = "/checkpoints/C9e18"
HF_ORG = "chess-pre-to-post"

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
    "rl-reasoning-sweep-C9e18",
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

def _run_training(config: str, num_gpus: int, overrides: list[str] = None):
    """Run a single training job inside a Modal container."""
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
            print(f"[sweep] Skipping {config}: final checkpoint already exists at {final_dir}")
            return

    print(f"[sweep] Starting: {config} on {num_gpus}x {GPU_TYPE}")

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

    # Add config overrides
    if overrides:
        for override in overrides:
            cmd.extend(["--override", override])

    print(f"  cmd: {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(cmd, cwd="/root/chess", stdout=_sys.stdout, stderr=_sys.stderr)
    rc = proc.wait()

    ckpt_volume.commit()
    if rc != 0:
        raise RuntimeError(f"Training failed (exit {rc}): {config}")
    print(f"[sweep] Done: {config}")


@app.function(
    gpu=f"{GPU_TYPE}:{GPUS_PER_NODE}",
    timeout=60 * 60 * TIMEOUT_HOURS,
    retries=modal.Retries(initial_delay=0.0, max_retries=5),
)
def train_h200(config: str, overrides: list[str] = None):
    _run_training(config, GPUS_PER_NODE, overrides)


# --------------------------------------------------------------------------- #
#  Data download                                                               #
# --------------------------------------------------------------------------- #

@app.function(
    timeout=60 * 60 * 6,
    volumes={"/data": data_volume},
)
def download_pretraining_v1():
    """Download pretrain_v1_20b dataset, flatten shard_xxx folders into one dir."""
    import shutil
    from huggingface_hub import snapshot_download

    out_dir = Path(DATA_DIR)

    # Check if already downloaded
    existing_npy = sorted(out_dir.glob("*.npy")) if out_dir.exists() else []
    if len(existing_npy) > 0:
        print(f"Dataset already present at {out_dir} ({len(existing_npy)} .npy shards). Skipping.")
        import numpy as np
        sample = np.load(existing_npy[0])
        print(f"  Sample shard: {existing_npy[0].name}, {len(sample)} tokens")
        return

    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("ERROR: HF_TOKEN not set. Add huggingface-secret to Modal.")

    # Download to temp dir first
    tmp_dir = Path("/data/_pretrain_v1_20b_tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading chess-pre-to-post/pretrain_v1_20b to {tmp_dir}")
    snapshot_download(
        repo_id="chess-pre-to-post/pretrain_v1_20b",
        repo_type="dataset",
        token=token,
        local_dir=str(tmp_dir),
    )

    # Flatten: move all .npy files from shard_xxx/ subdirs into out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    moved = 0
    for shard_dir in sorted(tmp_dir.glob("shard_*")):
        if shard_dir.is_dir():
            for npy_file in sorted(shard_dir.glob("*.npy")):
                dest = out_dir / npy_file.name
                shutil.move(str(npy_file), str(dest))
                moved += 1
    # Also move any .npy files in the root
    for npy_file in sorted(tmp_dir.glob("*.npy")):
        dest = out_dir / npy_file.name
        shutil.move(str(npy_file), str(dest))
        moved += 1

    print(f"Moved {moved} .npy files to {out_dir}")

    # Cleanup tmp
    shutil.rmtree(str(tmp_dir), ignore_errors=True)

    data_volume.commit()
    print(f"Dataset ready at {out_dir}")

    # Print info
    npy_files = sorted(out_dir.glob("*.npy"))
    print(f"  Total .npy shards: {len(npy_files)}")
    if npy_files:
        import numpy as np
        sample = np.load(npy_files[0])
        print(f"  Sample shard: {npy_files[0].name}, {len(sample)} tokens")


@app.function(
    timeout=60 * 60 * 2,
    volumes={"/data": data_volume},
)
def download_test_data():
    """Download test datasets from HuggingFace."""
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
#  Sweep configs                                                               #
# --------------------------------------------------------------------------- #

SWEEP_CONFIGS = [
    ("50m",   "config/configs/C9e18/50m_alpha0.050.yaml"),
    ("50m",   "config/configs/C9e18/50m_alpha0.100.yaml"),
    ("50m",   "config/configs/C9e18/50m_alpha0.200.yaml"),
    ("50m",   "config/configs/C9e18/50m_alpha0.400.yaml"),
    ("50m",   "config/configs/C9e18/50m_alpha0.700.yaml"),
    ("110m",  "config/configs/C9e18/110m_alpha0.050.yaml"),
    ("110m",  "config/configs/C9e18/110m_alpha0.100.yaml"),
    ("110m",  "config/configs/C9e18/110m_alpha0.200.yaml"),
    ("110m",  "config/configs/C9e18/110m_alpha0.400.yaml"),
    ("110m",  "config/configs/C9e18/110m_alpha0.700.yaml"),
    ("110m",  "config/configs/C9e18/110m_alpha0.950.yaml"),
    ("200m",  "config/configs/C9e18/200m_alpha0.050.yaml"),
    ("200m",  "config/configs/C9e18/200m_alpha0.100.yaml"),
    ("200m",  "config/configs/C9e18/200m_alpha0.200.yaml"),
    ("200m",  "config/configs/C9e18/200m_alpha0.400.yaml"),
    ("200m",  "config/configs/C9e18/200m_alpha0.700.yaml"),
    ("200m",  "config/configs/C9e18/200m_alpha0.950.yaml"),
    ("410m",  "config/configs/C9e18/410m_alpha0.050.yaml"),
    ("410m",  "config/configs/C9e18/410m_alpha0.100.yaml"),
    ("410m",  "config/configs/C9e18/410m_alpha0.200.yaml"),
    ("410m",  "config/configs/C9e18/410m_alpha0.400.yaml"),
    ("410m",  "config/configs/C9e18/410m_alpha0.700.yaml"),
    ("410m",  "config/configs/C9e18/410m_alpha0.950.yaml"),
    ("670m",  "config/configs/C9e18/670m_alpha0.050.yaml"),
    ("670m",  "config/configs/C9e18/670m_alpha0.100.yaml"),
    ("670m",  "config/configs/C9e18/670m_alpha0.200.yaml"),
    ("670m",  "config/configs/C9e18/670m_alpha0.400.yaml"),
    ("670m",  "config/configs/C9e18/670m_alpha0.700.yaml"),
    ("670m",  "config/configs/C9e18/670m_alpha0.950.yaml"),
    ("1000m", "config/configs/C9e18/1000m_alpha0.050.yaml"),
    ("1000m", "config/configs/C9e18/1000m_alpha0.100.yaml"),
    ("1000m", "config/configs/C9e18/1000m_alpha0.200.yaml"),
    ("1000m", "config/configs/C9e18/1000m_alpha0.400.yaml"),
    ("1000m", "config/configs/C9e18/1000m_alpha0.700.yaml"),
    ("1000m", "config/configs/C9e18/1000m_alpha0.950.yaml"),
]


def _build_overrides(model_size: str, config_path: str) -> list[str]:
    """Build config overrides for a job."""
    config_path_obj = Path(config_path)
    budget = config_path_obj.parent.name  # e.g. "C9e18"
    stem = config_path_obj.stem           # e.g. "110m_alpha0.050"
    canonical_name = f"{budget}_{stem}"   # e.g. "C9e18_110m_alpha0.050"

    return [
        "training.gpu_peak_tflops=989",
        "training.cache_size=0",
        "training.mixed_precision=bf16",
        f"training.experiment_name={canonical_name}",
        f"training.run_name={canonical_name}",
        f"training.hf_upload_repo={HF_ORG}/{canonical_name}",
    ]


# --------------------------------------------------------------------------- #
#  Entrypoint                                                                  #
# --------------------------------------------------------------------------- #

@app.local_entrypoint()
def main(model_filter: str = "", dry_run: bool = False, max_concurrent: int = 8):
    jobs = []
    for model_size, config_path in SWEEP_CONFIGS:
        if model_filter and model_size != model_filter:
            continue
        overrides = _build_overrides(model_size, config_path)
        name = Path(config_path).stem
        jobs.append((name, config_path, overrides))

    print(f"C9e18 beta=0.03 sweep: {len(jobs)} jobs, max {max_concurrent} concurrent")
    print(f"GPU: {GPUS_PER_NODE}x {GPU_TYPE} per job")
    print(f"Data: {DATA_DIR}")
    print(f"Checkpoints: {OUTPUT_DIR}")
    print("=" * 70)
    for name, config_path, overrides in jobs:
        batch_info = [o for o in overrides if "batch_size" in o or "gradient_acc" in o]
        print(f"  {name:45s}  {' '.join(batch_info)}")
    print("=" * 70)

    if dry_run:
        print("(dry-run -- nothing launched)")
        return

    # Use ThreadPoolExecutor for concurrency control.
    # Each thread calls train_h200.remote() which blocks until the Modal function completes.
    # max_workers=max_concurrent limits to 8 concurrent jobs.
    failed = []
    succeeded = []

    def run_job(name, config, overrides):
        print(f"  STARTING: {name}")
        start_time = time.time()
        train_h200.remote(config=config, overrides=overrides)
        elapsed = time.time() - start_time
        print(f"  FINISHED: {name} ({elapsed/3600:.1f}h)")
        return name

    with ThreadPoolExecutor(max_workers=max_concurrent) as pool:
        futures = {
            pool.submit(run_job, name, cfg, ovr): name
            for name, cfg, ovr in jobs
        }
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                fut.result()
                succeeded.append(name)
                print(f"  OK: {name}")
            except Exception as e:
                print(f"  FAILED: {name}: {e}")
                failed.append(name)

    print(f"\nDone: {len(succeeded)}/{len(jobs)} succeeded")
    if failed:
        print(f"Failed: {', '.join(failed)}")
