"""Download allenai/dolma3_dolmino_mix-100B-1125 to a Modal Volume.

This is OLMo3's stage-2 anneal midtraining pool (~212 GB compressed across
99,676 .jsonl.zst files), curated into two complete 100B-token "ingredients"
(Ingredient 1 and Ingredient 2) for OLMo3 model merging research.

Ingredient 1 (~106 GB) contains the synthetic-math + reasoning + code +
high-quality-crawl mix that complements Nemotron-CC-Math for our use case.

Top-level folders under data/ all start with `ingredient1-*` or `ingredient2-*`.

Usage:
    # Default: Ingredient 1 only (~106 GB, ~10 min)
    modal run --detach download_dolma3.py

    # Both ingredients (~212 GB)
    modal run --detach download_dolma3.py --ingredients 1,2

    # Specific subsets (e.g., math + reasoning only):
    modal run --detach download_dolma3.py --prefixes ingredient1-cranemath,ingredient1-dolmino-math,ingredient1-general_reasoning_mix
"""

from __future__ import annotations

import time
from typing import Iterable

import modal

from common import DOLMINO_MOUNT, dolmino_volume, hf_image

HF_DATASET_ID = "allenai/dolma3_dolmino_mix-100B-1125"

app = modal.App("dolma3-dolmino-mix-100B-1125-download", image=hf_image())
hf_secret = modal.Secret.from_name("huggingface-secret")


def _list_files(ingredients: list[int] | None, prefixes: list[str] | None) -> list[str]:
    from huggingface_hub import HfApi

    api = HfApi()
    all_files = api.list_repo_files(repo_id=HF_DATASET_ID, repo_type="dataset")
    candidates = [f for f in all_files if f.startswith("data/") and f.endswith(".jsonl.zst")]

    if prefixes:
        wanted = tuple(f"data/{p.strip()}/" for p in prefixes if p.strip())
        selected = [f for f in candidates if f.startswith(wanted)]
    elif ingredients:
        # Match e.g. ingredient1-* or ingredient2-*
        wanted = tuple(f"data/ingredient{i}-" for i in ingredients)
        selected = [f for f in candidates if f.startswith(wanted)]
    else:
        selected = candidates

    selected.sort()
    return selected


@app.function(
    secrets=[hf_secret],
    volumes={DOLMINO_MOUNT: dolmino_volume},
    timeout=60 * 60 * 6,
    cpu=4.0,
    memory=8 * 1024,
)
def download_file(file_path: str) -> dict:
    import os
    from pathlib import Path

    from huggingface_hub import hf_hub_download

    dest_path = Path(DOLMINO_MOUNT) / file_path
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists():
        size = dest_path.stat().st_size
        return {"file": file_path, "skipped": True, "bytes": size}

    start = time.time()
    hf_hub_download(
        repo_id=HF_DATASET_ID,
        filename=file_path,
        repo_type="dataset",
        local_dir=DOLMINO_MOUNT,
        token=os.environ["HF_TOKEN"],
    )
    elapsed = time.time() - start
    size = dest_path.stat().st_size
    return {
        "file": file_path,
        "skipped": False,
        "bytes": size,
        "seconds": elapsed,
        "MB_per_s": (size / 1e6) / max(elapsed, 1e-6),
    }


@app.function(
    secrets=[hf_secret],
    volumes={DOLMINO_MOUNT: dolmino_volume},
    timeout=60 * 60 * 24,
)
def orchestrate(ingredients: str = "1", prefixes: str = "") -> dict:
    ing_list: list[int] | None = None
    pref_list: list[str] | None = None
    if prefixes.strip():
        pref_list = [p.strip() for p in prefixes.split(",") if p.strip()]
    else:
        ing_list = [int(i.strip()) for i in ingredients.split(",") if i.strip()]

    files = _list_files(ing_list, pref_list)
    print(f"[orchestrate] selected {len(files)} files (ingredients={ing_list} prefixes={pref_list})")

    results = []
    total_bytes = 0
    by_bucket_bytes: dict[str, int] = {}
    by_bucket_files: dict[str, int] = {}
    skipped = 0
    failed = 0
    for r in download_file.map(files, return_exceptions=True):
        if isinstance(r, Exception):
            print(f"[FAIL] {r!r}")
            failed += 1
            continue
        results.append(r)
        total_bytes += r["bytes"]
        bucket = r["file"].split("/")[1]
        by_bucket_bytes[bucket] = by_bucket_bytes.get(bucket, 0) + r["bytes"]
        by_bucket_files[bucket] = by_bucket_files.get(bucket, 0) + 1
        if r.get("skipped"):
            skipped += 1

    print()
    print(f"[orchestrate] downloaded {len(results)} files ({skipped} cached, {failed} failed)")
    print(f"[orchestrate] total size: {total_bytes / 1e9:.2f} GB")
    print(f"[orchestrate] top buckets by size:")
    for b, sz in sorted(by_bucket_bytes.items(), key=lambda kv: -kv[1])[:15]:
        print(f"  {b:60s} {by_bucket_files[b]:>6} files  {sz / 1e9:>7.2f} GB")
    dolmino_volume.commit()
    return {
        "ingredients": ing_list,
        "prefixes": pref_list,
        "files": len(results),
        "skipped": skipped,
        "failed": failed,
        "total_bytes": total_bytes,
    }


@app.local_entrypoint()
def main(ingredients: str = "1", prefixes: str = "") -> None:
    summary = orchestrate.remote(ingredients=ingredients, prefixes=prefixes)
    print()
    print("=== SUMMARY ===")
    print(summary)
