"""
Generic backfill: upload `final/` for a pretrain run on the rl-reasoning-checkpoints
volume to chess-pre-to-post/<exp_name>/final.

Usage:
  modal run modal_scripts/upload_pretrain_final.py \
      --exp-name "50m_C_6p5e19_alpha0.200" --budget "6p5e19"

  modal run modal_scripts/upload_pretrain_final.py \
      --exp-name "100m_C_6p5e18_alpha0.100"  # budget defaults to 6p5e18
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
    "train",
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/checkpoints": ckpt_volume},
)

HF_ORG = "chess-pre-to-post"


@app.function(timeout=60 * 60 * 2)
def upload_one(exp_name: str, budget: str, hf_org: str = HF_ORG):
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN not set (need huggingface-secret).")

    local_final = Path("/checkpoints") / budget / exp_name / "final"
    if not local_final.is_dir():
        raise FileNotFoundError(f"final/ dir not on volume: {local_final}")

    files = sorted(p.name for p in local_final.iterdir())
    repo_id = f"{hf_org}/{exp_name}"
    print(f"[upload] {local_final} -> {repo_id}/final  ({len(files)} files: {files})")

    api = HfApi(token=token)
    api.create_repo(repo_id, exist_ok=True, repo_type="model")
    api.upload_folder(
        folder_path=str(local_final),
        repo_id=repo_id,
        path_in_repo="final",
        commit_message=f"Backfill: upload final/ from Modal volume",
    )
    print(f"[done] https://huggingface.co/{repo_id}/tree/main/final")
    return repo_id


@app.local_entrypoint()
def main(
    exp_name: str = "",
    budget: str = "6p5e18",
    batch: str = "",
    hf_org: str = HF_ORG,
):
    """Single: --exp-name X --budget Y.
    Batch:  --batch "<budget>:<exp_name>,<budget>:<exp_name>,..."
    """
    if batch:
        pairs = []
        for entry in batch.split(","):
            entry = entry.strip()
            if not entry:
                continue
            b, name = entry.split(":", 1)
            pairs.append((name.strip(), b.strip()))
        print(f"Batch upload: {len(pairs)} run(s)")
        for name, b in pairs:
            print(f"  {b}/{name}")
        handles = [upload_one.spawn(name, b, hf_org) for name, b in pairs]
        for (name, b), h in zip(pairs, handles):
            try:
                h.get()
                print(f"  OK  {b}/{name}")
            except Exception as e:
                print(f"  FAIL {b}/{name}: {e}")
        return

    if not exp_name:
        raise SystemExit("provide --exp-name or --batch")
    upload_one.remote(exp_name, budget, hf_org)
