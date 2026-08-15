"""
One-off Modal upload: push a checkpoint dir from the rl-reasoning-checkpoints
volume to a HuggingFace repo. Used to recover from failed auto-uploads.

Usage:
  modal run modal_scripts/upload_to_hf.py \
      --src 6p5e18/680m_C_6p5e18_alpha0.100 \
      --repo chess-pre-to-post/C6p5e18_680m_alpha0.100
"""
import os
from pathlib import Path

import modal

CKPT_DIR = "/checkpoints"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface-hub>=0.28.0")
)

ckpt_volume = modal.Volume.from_name("rl-reasoning-checkpoints", create_if_missing=False)

app = modal.App(
    "train",
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={CKPT_DIR: ckpt_volume},
)


@app.function(timeout=60 * 60 * 2)
def upload(src: str, repo: str, private: bool = False):
    from huggingface_hub import HfApi

    src_path = Path(CKPT_DIR) / src
    if not src_path.is_dir():
        raise FileNotFoundError(f"Source not found on volume: {src_path}")

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN not set (huggingface-secret)")

    api = HfApi(token=token)
    api.create_repo(repo_id=repo, repo_type="model", private=private, exist_ok=True)

    print(f"[upload] {src_path} -> {repo}")
    print(f"[upload] files in {src_path}:")
    for p in sorted(src_path.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(src_path)}  ({p.stat().st_size:,} B)")

    api.upload_folder(
        folder_path=str(src_path),
        repo_id=repo,
        repo_type="model",
    )
    print(f"[upload] done -> https://huggingface.co/{repo}")


@app.local_entrypoint()
def main(src: str, repo: str, private: bool = False):
    upload.remote(src=src, repo=repo, private=private)
