"""
Sync pretrain_v1_20b dataset from HF to Modal volume, flattened.

Downloads chess-pre-to-post/pretrain_v1_20b from HF, walks the tree,
and moves every *.npy file to /pretrain_v1_20b/<filename> on the
rl-reasoning-training-data volume (root, no nested subfolders).

Existing files with the same name on the volume are overwritten
(idempotent). New files are added. No files are deleted from the volume.

Usage:
  modal run --detach modal_scripts/sync_pretrain_data.py
"""
from pathlib import Path
import shutil

import modal

app = modal.App("pretrain")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface-hub>=0.28.0", "tqdm>=4.66.0")
)

data_volume = modal.Volume.from_name("rl-reasoning-training-data", create_if_missing=True)


@app.function(
    image=image,
    volumes={"/data": data_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    cpu=4,
    memory=16384,
    timeout=60 * 60 * 6,  # 6h to be safe for very large datasets
    ephemeral_disk=2 * 1024 * 1024,  # 2 TB scratch in MB
)
def sync():
    import os
    from huggingface_hub import snapshot_download

    HF_REPO = "chess-pre-to-post/pretrain_v1_20b"
    SCRATCH = Path("/tmp/hf_download")
    TARGET = Path("/data/pretrain_v1_20b")
    TARGET.mkdir(parents=True, exist_ok=True)

    # Inventory before sync
    before = sorted(p.name for p in TARGET.iterdir() if p.is_file())
    print(f"[sync] volume before: {len(before)} files at {TARGET}", flush=True)

    print(f"[sync] downloading {HF_REPO} -> {SCRATCH}", flush=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    local_dir = snapshot_download(
        repo_id=HF_REPO,
        repo_type="dataset",
        local_dir=str(SCRATCH),
        max_workers=8,
    )
    print(f"[sync] download complete: {local_dir}", flush=True)

    # Walk and move every .npy file to TARGET root
    npy_files = list(Path(local_dir).rglob("*.npy"))
    print(f"[sync] found {len(npy_files)} .npy files; flattening into {TARGET}", flush=True)

    if npy_files:
        # Show a sample of the source paths so we can see how nested they were
        sample = [p.relative_to(local_dir) for p in npy_files[:5]]
        print(f"[sync] sample source rel paths: {sample}", flush=True)

    moved = 0
    overwrote = 0
    for src in npy_files:
        dst = TARGET / src.name
        if dst.exists():
            overwrote += 1
        # Use shutil.move to handle cross-device renames (scratch may be tmpfs)
        shutil.move(str(src), str(dst))
        moved += 1
        if moved % 1000 == 0:
            print(f"[sync] moved {moved}/{len(npy_files)}", flush=True)

    print(f"[sync] moved {moved} files ({overwrote} overwrote existing)", flush=True)

    # Cleanup scratch
    shutil.rmtree(SCRATCH, ignore_errors=True)

    after = sorted(p.name for p in TARGET.iterdir() if p.is_file())
    new_files = sorted(set(after) - set(before))
    print(f"[sync] volume after:  {len(after)} files at {TARGET}", flush=True)
    print(f"[sync] net new files: {len(new_files)}", flush=True)
    if new_files[:5]:
        print(f"[sync] sample new: {new_files[:5]}", flush=True)
    if len(new_files) > 5:
        print(f"[sync] (... and {len(new_files) - 5} more)", flush=True)

    data_volume.commit()
    print("[sync] volume committed. done.", flush=True)


@app.local_entrypoint()
def main():
    sync.remote()
