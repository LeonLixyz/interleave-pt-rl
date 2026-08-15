"""Download allenai/dolmino-mix-1124 to a Modal Volume.

Dolmino-100B is the OLMo2 midtraining corpus. Files are `.json.zst` (zstd-compressed
JSON Lines, one document per line). Top-level subsets under `data/`:

    math: 5129 files (Stage-2 Math: TinyGSM-MIND, MathCoder2, TuluMath, etc.)
    dclm: 1970 files (filtered Common Crawl baseline)
    flan: 209 files (instruction-format data)
    pes2o: 26 files (academic papers)
    stackexchange: 16 files
    wiki: 2 files

Usage:
    # Download everything (~250 GB raw .json.zst, ~30-45 min wall):
    modal run --detach download_dolmino.py

    # Only specific subsets:
    modal run --detach download_dolmino.py --subsets math,dclm,flan
"""

from __future__ import annotations

import time
from typing import Iterable

import modal

from common import (
    DOLMINO_MOUNT,
    dolmino_volume,
    hf_image,
)

HF_DATASET_ID = "allenai/dolmino-mix-1124"
ALL_SUBSETS = ["math", "dclm", "flan", "pes2o", "stackexchange", "wiki"]

app = modal.App("dolmino-mix-1124-download", image=hf_image())

hf_secret = modal.Secret.from_name("huggingface-secret")


# Dolmino sub-folders use a mix of formats. Per-subset extensions observed:
#   dclm:          .json.zst
#   flan:          .json.gz
#   math:          .jsonl.gz, .jsonl, .jsonl.zst, and quirky variants like .NNNNN.jsonl.gz
#   pes2o, etc.:   .jsonl, .jsonl.gz, .jsonl.zst, or .json.gz
# We accept any JSON-Lines-ish file under the chosen subsets.
_DATA_EXTS = (".json.zst", ".json.gz", ".jsonl.zst", ".jsonl.gz", ".jsonl")


def _list_files(subsets: Iterable[str]) -> list[str]:
    from huggingface_hub import HfApi

    api = HfApi()
    all_files = api.list_repo_files(repo_id=HF_DATASET_ID, repo_type="dataset")
    wanted_prefixes = tuple(f"data/{s}/" for s in subsets)
    selected = [
        f
        for f in all_files
        if f.startswith(wanted_prefixes)
        and any(f.endswith(ext) for ext in _DATA_EXTS)
    ]
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
    """Download a single file from HF to the Modal volume."""
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
def orchestrate(subsets: str = ",".join(ALL_SUBSETS)) -> dict:
    subset_list = [s.strip() for s in subsets.split(",") if s.strip()]
    bad = [s for s in subset_list if s not in ALL_SUBSETS]
    if bad:
        raise ValueError(f"unknown subsets {bad}; valid: {ALL_SUBSETS}")
    files = _list_files(subset_list)
    print(f"[orchestrate] subsets={subset_list} -> {len(files)} files")

    results = []
    total_bytes = 0
    by_subset_bytes: dict[str, int] = {s: 0 for s in subset_list}
    by_subset_files: dict[str, int] = {s: 0 for s in subset_list}
    skipped = 0
    failed = 0
    for r in download_file.map(files, return_exceptions=True):
        if isinstance(r, Exception):
            print(f"[FAIL] {r!r}")
            failed += 1
            continue
        results.append(r)
        total_bytes += r["bytes"]
        # data/<subset>/...
        subset = r["file"].split("/")[1]
        by_subset_bytes[subset] = by_subset_bytes.get(subset, 0) + r["bytes"]
        by_subset_files[subset] = by_subset_files.get(subset, 0) + 1
        if r.get("skipped"):
            skipped += 1

    print()
    print(f"[orchestrate] downloaded {len(results)} files ({skipped} cached, {failed} failed)")
    print(f"[orchestrate] total size: {total_bytes / 1e9:.2f} GB")
    print("[orchestrate] per-subset:")
    for s in subset_list:
        print(
            f"  {s:>15}: {by_subset_files.get(s, 0):>5} files, "
            f"{by_subset_bytes.get(s, 0) / 1e9:.2f} GB"
        )
    dolmino_volume.commit()
    return {
        "subsets": subset_list,
        "files": len(results),
        "skipped": skipped,
        "failed": failed,
        "total_bytes": total_bytes,
        "by_subset_bytes": by_subset_bytes,
    }


@app.local_entrypoint()
def main(subsets: str = ",".join(ALL_SUBSETS)) -> None:
    summary = orchestrate.remote(subsets=subsets)
    print()
    print("=== SUMMARY ===")
    print(summary)
