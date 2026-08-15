"""Download nvidia/Nemotron-CC-Math-v1 parquet shards to a Modal Volume.

Usage:
    modal run --detach download.py                    # download Math-3+ subset (default, 57 shards)
    modal run --detach download.py --subset 4plus     # 4plus only (46 shards)
    modal run --detach download.py --subset 4plus_MIND
    modal run --detach download.py --subset all       # everything (193 parquet)

The Math-3+ subset alone is the 133B-token corpus we plan to train on.
"""

from __future__ import annotations

import time
from typing import Iterable

import modal

from common import (
    DATA_MOUNT,
    HF_DATASET_ID,
    data_volume,
    hf_image,
)

app = modal.App("nemotron-cc-math-download", image=hf_image())

hf_secret = modal.Secret.from_name("huggingface-secret")

SUBSET_PREFIXES = {
    "3": ["3/"],
    "4plus": ["4plus/"],
    "4plus_MIND": ["4plus_MIND/"],
    "all": ["3/", "4plus/", "4plus_MIND/"],
}


def _list_parquet(prefixes: Iterable[str]) -> list[str]:
    from huggingface_hub import HfApi

    api = HfApi()
    all_files = api.list_repo_files(repo_id=HF_DATASET_ID, repo_type="dataset")
    selected = [
        f
        for f in all_files
        if f.endswith(".parquet") and any(f.startswith(p) for p in prefixes)
    ]
    selected.sort()
    return selected


@app.function(
    secrets=[hf_secret],
    volumes={DATA_MOUNT: data_volume},
    timeout=60 * 60 * 6,
    cpu=4.0,
    memory=8 * 1024,
)
def download_shard(file_path: str) -> dict:
    """Download a single parquet file from HF to the Modal volume."""
    import os
    from pathlib import Path

    from huggingface_hub import hf_hub_download

    dest_dir = Path(DATA_MOUNT) / Path(file_path).parent
    dest_path = Path(DATA_MOUNT) / file_path
    dest_dir.mkdir(parents=True, exist_ok=True)

    if dest_path.exists():
        size = dest_path.stat().st_size
        return {"file": file_path, "skipped": True, "bytes": size}

    start = time.time()
    local = hf_hub_download(
        repo_id=HF_DATASET_ID,
        filename=file_path,
        repo_type="dataset",
        local_dir=DATA_MOUNT,
        token=os.environ["HF_TOKEN"],
    )
    elapsed = time.time() - start
    size = Path(local).stat().st_size
    return {
        "file": file_path,
        "skipped": False,
        "bytes": size,
        "seconds": elapsed,
        "MB_per_s": (size / 1e6) / max(elapsed, 1e-6),
    }


@app.function(
    secrets=[hf_secret],
    volumes={DATA_MOUNT: data_volume},
    timeout=60 * 60 * 24,
)
def orchestrate(subset: str = "3") -> dict:
    if subset not in SUBSET_PREFIXES:
        raise ValueError(f"subset must be one of {list(SUBSET_PREFIXES)}, got {subset}")
    prefixes = SUBSET_PREFIXES[subset]
    files = _list_parquet(prefixes)
    print(f"[orchestrate] subset={subset} prefixes={prefixes} -> {len(files)} files")
    for f in files[:5]:
        print("  ", f)
    if len(files) > 5:
        print(f"  ... and {len(files) - 5} more")

    results = []
    total_bytes = 0
    skipped = 0
    failed = 0
    for r in download_shard.map(files, return_exceptions=True):
        if isinstance(r, Exception):
            print(f"[FAIL] {r!r}")
            failed += 1
            continue
        results.append(r)
        total_bytes += r["bytes"]
        if r.get("skipped"):
            skipped += 1
        print(
            f"[ok] {r['file']} "
            + ("[cached] " if r.get('skipped') else f"{r.get('MB_per_s', 0):.0f} MB/s ")
            + f"{r['bytes'] / 1e9:.2f} GB"
        )

    print()
    print(f"[orchestrate] downloaded {len(results)} files ({skipped} cached, {failed} failed)")
    print(f"[orchestrate] total size: {total_bytes / 1e9:.2f} GB")
    data_volume.commit()
    return {
        "subset": subset,
        "files": len(results),
        "skipped": skipped,
        "failed": failed,
        "total_bytes": total_bytes,
    }


@app.local_entrypoint()
def main(subset: str = "3") -> None:
    summary = orchestrate.remote(subset=subset)
    print()
    print("=== SUMMARY ===")
    print(summary)
