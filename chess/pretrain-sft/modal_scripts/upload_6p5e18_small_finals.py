"""
Backfill: upload final/ for the 15 6p5e18_small runs to HF.

Uploads /checkpoints/6p5e18/<exp_name>/final  ->  chess-pre-to-post/<exp_name>/final

Usage:
  modal run modal_scripts/upload_6p5e18_small_finals.py
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
    "upload-6p5e18-small-finals",
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/checkpoints": ckpt_volume},
)

EXP_NAMES = [
    "5m_C_6p5e18_alpha0.010",
    "5m_C_6p5e18_alpha0.050",
    "5m_C_6p5e18_alpha0.100",
    "5m_C_6p5e18_alpha0.200",
    "10m_C_6p5e18_alpha0.010",
    "10m_C_6p5e18_alpha0.050",
    "10m_C_6p5e18_alpha0.100",
    "10m_C_6p5e18_alpha0.200",
    "10m_C_6p5e18_alpha0.400",
    "20m_C_6p5e18_alpha0.010",
    "20m_C_6p5e18_alpha0.050",
    "20m_C_6p5e18_alpha0.100",
    "20m_C_6p5e18_alpha0.200",
    "20m_C_6p5e18_alpha0.400",
    "20m_C_6p5e18_alpha0.750",
]

HF_ORG = "chess-pre-to-post"


@app.function(timeout=60 * 60 * 4)
def upload_finals():
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN not set (need huggingface-secret).")

    api = HfApi(token=token)
    for exp_name in EXP_NAMES:
        local_final = Path("/checkpoints/6p5e18") / exp_name / "final"
        hf_repo = f"{HF_ORG}/{exp_name}"
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
            commit_message="Backfill: upload final/ from Modal volume",
        )
        print(f"[done]   {hf_repo}/final")


@app.local_entrypoint()
def main():
    upload_finals.remote()
