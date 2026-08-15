"""Upload final checkpoints and metrics.jsonl from Modal volume to HuggingFace.

Usage:
  modal run scripts/upload_finals_to_hf.py
  modal run scripts/upload_finals_to_hf.py --dry-run
"""
import os
from pathlib import Path
import modal

image = modal.Image.debian_slim(python_version="3.11").pip_install("huggingface-hub>=0.28.0")

data_volume = modal.Volume.from_name("rl-reasoning-training-data", create_if_missing=True)
ckpt_volume = modal.Volume.from_name("rl-reasoning-checkpoints", create_if_missing=True)

app = modal.App(
    "rl-reasoning-upload-hf",
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/checkpoints": ckpt_volume},
)

HF_ORG = "chess-pre-to-post"


@app.function(timeout=60 * 60 * 4)
def upload_all(dry_run: bool = False):
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)

    # Only upload the last missing one
    experiments = [
        Path("/checkpoints/6p5e18/5m_C_6p5e18_alpha0.030"),
        Path("/checkpoints/6p5e18/10m_C_6p5e18_alpha0.030"),
        Path("/checkpoints/6p5e18/20m_C_6p5e18_alpha0.030"),
        Path("/checkpoints/6p5e18/32m_C_6p5e18_alpha0.030"),
        Path("/checkpoints/6p5e18/50m_C_6p5e18_alpha0.030"),
    ]

    print(f"Found {len(experiments)} experiments")
    uploaded = 0
    skipped = 0
    failed = 0

    for exp_dir in experiments:
        name = exp_dir.name  # e.g. "50m_C_6p5e18_alpha0.050"
        final_dir = exp_dir / "final"
        metrics_file = exp_dir / "metrics.jsonl"

        has_final = final_dir.exists() and any(final_dir.glob("*.safetensors"))
        has_metrics = metrics_file.exists()

        if not has_final:
            print(f"  SKIP (no final): {name}")
            skipped += 1
            continue

        repo_id = f"{HF_ORG}/{name}"
        print(f"  Uploading: {name} -> {repo_id}")

        if dry_run:
            print(f"    [dry-run] final/: {list(final_dir.iterdir())}")
            if has_metrics:
                print(f"    [dry-run] metrics.jsonl: {metrics_file.stat().st_size} bytes")
            continue

        try:
            api.create_repo(repo_id, exist_ok=True, repo_type="model")

            # Upload final checkpoint
            api.upload_folder(
                folder_path=str(final_dir),
                repo_id=repo_id,
                path_in_repo="final",
                commit_message="Upload final checkpoint",
            )

            # Upload metrics.jsonl if exists
            if has_metrics:
                api.upload_file(
                    path_or_fileobj=str(metrics_file),
                    path_in_repo="metrics.jsonl",
                    repo_id=repo_id,
                    commit_message="Upload metrics",
                )

            print(f"    OK: {repo_id}")
            uploaded += 1
        except Exception as e:
            print(f"    FAILED: {repo_id}: {e}")
            failed += 1

    print(f"\nDone: {uploaded} uploaded, {skipped} skipped, {failed} failed")


@app.local_entrypoint()
def main(dry_run: bool = False):
    upload_all.remote(dry_run=dry_run)
