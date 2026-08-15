"""Test a single 680m job to debug errors."""
import os
from pathlib import Path
import modal

cuda_version = "12.4.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"
repo_dir = Path(__file__).parent.parent

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .apt_install("curl", "git", "vim")
    .pip_install(
        "torch>=2.6.0", "accelerate>=1.10.0", "transformers>=4.50.0",
        "datasets>=3.0.0", "pyarrow>=17.0.0", "pandas>=2.0.0",
        "pyyaml>=6.0", "omegaconf>=2.3.0", "wandb>=0.19.0",
        "einops>=0.7.0", "tokenizers>=0.19.0", "tqdm>=4.66.0",
        "chess>=1.11.0", "numpy>=2.0.0", "safetensors>=0.5.0",
        "sentencepiece>=0.2.0", "huggingface-hub>=0.28.0",
    )
    .add_local_dir(str(repo_dir / "scripts"), remote_path="/root/chess/scripts")
    .add_local_dir(str(repo_dir / "training"), remote_path="/root/chess/training")
    .add_local_dir(str(repo_dir / "config"), remote_path="/root/chess/config")
    .add_local_dir(str(repo_dir / "llm_tokens"), remote_path="/root/chess/llm_tokens")
    .add_local_dir(str(repo_dir / "evaluation"), remote_path="/root/chess/evaluation")
)

data_volume = modal.Volume.from_name("rl-reasoning-training-data", create_if_missing=True)
ckpt_volume = modal.Volume.from_name("rl-reasoning-checkpoints", create_if_missing=True)

app = modal.App(
    "test-680m",
    image=image,
    secrets=[modal.Secret.from_name("wandb-secret"), modal.Secret.from_name("huggingface-secret")],
    volumes={"/data": data_volume, "/checkpoints": ckpt_volume},
)

@app.function(gpu="H200:8", timeout=60*60*2)
def test():
    import subprocess, sys
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["WANDB_MODE"] = "disabled"
    cmd = [
        "accelerate", "launch", "--multi_gpu", "--num_processes", "8", "--mixed_precision", "bf16",
        "scripts/train/train_hf.py",
        "--config", "config/configs/6p5e18/680m_alpha0.050.yaml",
        "--data_dir", "/data/pretrain_v1_20b",
        "--output_dir", "/checkpoints/test_680m",
        "--override", "training.cache_size=0", "training.mixed_precision=bf16",
    ]
    proc = subprocess.Popen(cmd, cwd="/root/chess", stdout=sys.stdout, stderr=sys.stderr)
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"exit code {rc}")

@app.local_entrypoint()
def main():
    test.remote()
