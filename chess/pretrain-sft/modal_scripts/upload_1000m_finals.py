"""
One-shot: upload missing /final/ for the 3 1000m runs in 6p5e19_leon
to their HF repos under chess-pre-to-post/.

Usage:
  modal run modal_scripts/upload_1000m_finals.py
"""
import os
from pathlib import Path

import modal

ckpt_volume = modal.Volume.from_name("rl-reasoning-checkpoints", create_if_missing=False)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface-hub>=0.28.0")
)

app = modal.App(
    "upload-1000m-finals",
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/checkpoints": ckpt_volume},
)

# (local checkpoint subdir name, target HF repo)
JOBS = [
    ("1000m_C_6p5e19_alpha0.400", "chess-pre-to-post/6p5e19_leon_1000m_alpha0.400"),
    ("1000m_C_6p5e19_alpha0.750", "chess-pre-to-post/6p5e19_leon_1000m_alpha0.750"),
    ("1000m_C_6p5e19_alpha1.000", "chess-pre-to-post/6p5e19_leon_1000m_alpha1.000"),
]


@app.function(timeout=60 * 60 * 2)
def upload_finals():
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN not set (need huggingface-secret).")

    api = HfApi(token=token)
    for local_subdir, hf_repo in JOBS:
        local_final = Path("/checkpoints/6p5e19_leon") / local_subdir / "final"
        if not local_final.is_dir():
            print(f"[skip] {local_final} not found")
            continue
        files = sorted(p.name for p in local_final.iterdir())
        print(f"[upload] {local_final} -> {hf_repo}/final  ({len(files)} files)")
        api.create_repo(hf_repo, exist_ok=True, repo_type="model")
        api.upload_folder(
            folder_path=str(local_final),
            repo_id=hf_repo,
            path_in_repo="final",
            commit_message="Upload final/ from Modal volume (post-hoc rescue)",
        )
        print(f"[done]   {hf_repo}/final")


@app.local_entrypoint()
def main():
    upload_finals.remote()
