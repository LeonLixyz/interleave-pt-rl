"""
End-to-end test on 8x H200. Verifies LR schedule, warmup, checkpointing, data loading.

Usage:
  modal run modal_scripts/test_e2e.py
"""
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
    .run_commands(
        'python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained(\'Qwen/Qwen3-0.6B\')"'
    )
    .add_local_dir(str(repo_dir / "scripts"), remote_path="/root/chess/scripts")
    .add_local_dir(str(repo_dir / "training"), remote_path="/root/chess/training")
    .add_local_dir(str(repo_dir / "config"), remote_path="/root/chess/config")
    .add_local_dir(str(repo_dir / "llm_tokens"), remote_path="/root/chess/llm_tokens")
    .add_local_dir(str(repo_dir / "evaluation"), remote_path="/root/chess/evaluation")
    .add_local_dir(str(repo_dir / "modal_scripts"), remote_path="/root/chess/modal_scripts")
)

app = modal.App("chess-test-e2e", image=image)


@app.function(gpu="H200:8", timeout=60 * 60 * 1)
def test_all():
    import subprocess, sys

    env = os.environ.copy()
    env["CUDA_LAUNCH_BLOCKING"] = "1"

    cmd = [
        "accelerate", "launch",
        "--multi_gpu",
        "--num_processes", "8",
        "--mixed_precision", "bf16",
        "/root/chess/modal_scripts/_test_e2e_inner.py",
    ]
    print(f"Running: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, cwd="/root/chess", stdout=sys.stdout, stderr=sys.stderr, env=env)
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"Test failed with exit code {rc}")


@app.local_entrypoint()
def main():
    test_all.remote()
