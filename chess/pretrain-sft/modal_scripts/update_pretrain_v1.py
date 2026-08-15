"""Incremental updater for the chess-pre-to-post/pretrain_v1_20b dataset.

The HF dataset is organized as `shard_xxx/<files>.npy`. Our training scripts
expect a FLAT layout at /data/pretrain_v1_20b/<files>.npy so they can do
`glob("*.npy")`. This script:

  1. snapshot_download(...) into a staging dir (HF skips files it already has).
  2. Walks all shard_xxx/ subdirs and moves .npy files into the flat output dir.
     If a destination file already exists, it's overwritten (in case the HF
     repo updated a previously-published shard).
  3. Reports how many files were new vs. already present.

Usage:
    modal run --detach chess_reasoning/modal_scripts/update_pretrain_v1.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import modal

HF_REPO = "chess-pre-to-post/pretrain_v1_20b"
DATA_DIR = "/data/pretrain_v1_20b"
STAGING_DIR = "/data/_pretrain_v1_20b_staging"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "huggingface_hub==0.36.2",
        "numpy==2.2.6",
    )
)

data_volume = modal.Volume.from_name("rl-reasoning-training-data", create_if_missing=False)

app = modal.App(
    "update-pretrain-v1",
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)


@app.function(
    timeout=60 * 60 * 6,
    cpu=8.0,
    memory=32 * 1024,
    volumes={"/data": data_volume},
)
def update():
    import shutil
    from huggingface_hub import snapshot_download

    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("ERROR: HF_TOKEN not set (need huggingface-secret).")

    out_dir = Path(DATA_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    staging = Path(STAGING_DIR)
    staging.mkdir(parents=True, exist_ok=True)

    existing_before = {p.name for p in out_dir.glob("*.npy")}
    print(f"[update] flat dir before: {len(existing_before)} .npy files")

    # snapshot_download is idempotent: only fetches files HF marks as new/changed.
    print(f"[update] snapshot_download {HF_REPO} -> {staging}")
    snapshot_download(
        repo_id=HF_REPO,
        repo_type="dataset",
        token=token,
        local_dir=str(staging),
    )

    # Collect all .npy files from both shard_xxx/ subdirs and the staging root.
    candidates: list[Path] = []
    for shard_dir in sorted(staging.glob("shard_*")):
        if shard_dir.is_dir():
            candidates.extend(sorted(shard_dir.glob("*.npy")))
    candidates.extend(sorted(staging.glob("*.npy")))

    new_count = 0
    overwrite_count = 0
    for src in candidates:
        dest = out_dir / src.name
        if dest.exists():
            # Overwrite — HF may have re-published an updated shard with the same name.
            overwrite_count += 1
        else:
            new_count += 1
        shutil.move(str(src), str(dest))

    # Clean up the staging dir.
    shutil.rmtree(str(staging), ignore_errors=True)

    existing_after = {p.name for p in out_dir.glob("*.npy")}
    print(f"[update] flat dir after:  {len(existing_after)} .npy files")
    print(f"[update] new files:       {new_count}")
    print(f"[update] overwritten:     {overwrite_count}")
    print(f"[update] added since:     {sorted(existing_after - existing_before)[:5]}{' ...' if len(existing_after - existing_before) > 5 else ''}")

    data_volume.commit()
    print("[update] data_volume committed")


@app.local_entrypoint()
def main():
    update.remote()
