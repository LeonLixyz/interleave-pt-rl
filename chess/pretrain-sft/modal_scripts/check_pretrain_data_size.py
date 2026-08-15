"""Quick token count of /data/pretrain_v1_20b to verify actual size."""
import os
from pathlib import Path

import modal

data_volume = modal.Volume.from_name("rl-reasoning-training-data")

image = modal.Image.debian_slim().pip_install("numpy>=2.0.0")
app = modal.App("data-size-check", image=image)


@app.function(volumes={"/data": data_volume}, timeout=600)
def check():
    import numpy as np

    base = Path("/data/pretrain_v1_20b")
    files = sorted(base.glob("*.npy"))
    print(f"Found {len(files)} .npy files in {base}")
    if not files:
        return

    # Sample a few to estimate avg tokens per shard
    sample = files[:5] + files[-5:] + files[len(files) // 2 : len(files) // 2 + 5]
    sizes = []
    for f in sample:
        try:
            a = np.load(f, mmap_mode="r")
            sizes.append(int(a.shape[0]))
        except Exception as e:
            print(f"  err {f.name}: {e}")
    if sizes:
        avg = sum(sizes) / len(sizes)
        est_total = avg * len(files)
        print(f"  sampled {len(sizes)} shards, sizes: min={min(sizes)} max={max(sizes)} avg={int(avg)}")
        print(f"  estimated total tokens: {est_total:,.0f} ({est_total / 1e9:.2f}B)")


@app.local_entrypoint()
def main():
    check.remote()
