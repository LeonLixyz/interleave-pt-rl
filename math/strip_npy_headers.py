"""Strip NPY headers from all tokenized .npy files, converting them to raw uint32 bytes.

Background
----------
Our `tokenize_*.py` scripts use `np.save(path, arr)` which writes files in the
.npy format (a small ~80-byte header + raw bytes). But OLMo-core's data loaders
(both `NumpyFSLDataset` and the composable `NumpyDocumentSource`) treat every
`.npy` file as RAW uint32 binary (`file_size // 4 == num_tokens`). The header
bytes get interpreted as tokens, producing IDs outside the [0, vocab_size) range.

This script walks the tokenized volume and rewrites each `.npy` to its raw
contents (drop the npy header) in-place. ~5-10 min wall on Modal.

Usage:
    modal run --detach strip_npy_headers.py --root /tokenized/3
    modal run --detach strip_npy_headers.py --root /tokenized/4plus
    modal run --detach strip_npy_headers.py --root /tokenized/4plus_MIND
    modal run --detach strip_npy_headers.py --root /tokenized/dolma3
"""

from __future__ import annotations

import time
from pathlib import Path

import modal

from common import TOKENIZED_MOUNT, hf_image_base, tokenized_volume


def _strip_image() -> modal.Image:
    return hf_image_base().add_local_python_source("common")


app = modal.App("strip-npy-headers", image=_strip_image())


@app.function(
    volumes={TOKENIZED_MOUNT: tokenized_volume},
    timeout=60 * 30,
    cpu=4.0,
    memory=8 * 1024,
)
def strip_one(path: str) -> dict:
    """Read a .npy file with np.load (which honors the header) and rewrite it
    as raw uint32 bytes."""
    import numpy as np

    p = Path(path)
    if not p.is_file():
        return {"path": path, "status": "missing"}
    raw_size_before = p.stat().st_size
    # Detect if the file already has no NPY header by sniffing the magic.
    with p.open("rb") as fh:
        head = fh.read(6)
    if head != b"\x93NUMPY":
        # Already raw — nothing to do.
        return {"path": path, "status": "already_raw", "bytes": raw_size_before}

    # Load via np.load to get the data array, then rewrite raw.
    arr = np.load(p, allow_pickle=False)
    if arr.dtype != np.uint32:
        return {"path": path, "status": "wrong_dtype", "dtype": str(arr.dtype)}

    tmp = p.with_name(p.stem + ".tmp_raw")
    arr.tofile(tmp)
    # Atomic replace
    tmp.replace(p)
    raw_size_after = p.stat().st_size
    return {
        "path": path,
        "status": "stripped",
        "tokens": int(arr.shape[0]),
        "bytes_before": raw_size_before,
        "bytes_after": raw_size_after,
        "header_size": raw_size_before - raw_size_after,
    }


@app.function(
    volumes={TOKENIZED_MOUNT: tokenized_volume},
    timeout=60 * 60 * 4,
    cpu=4.0,
    memory=8 * 1024,
)
def orchestrate(root: str = "/tokenized") -> dict:
    tokenized_volume.reload()
    root_path = Path(root)
    if not root_path.is_dir():
        raise FileNotFoundError(f"{root_path} not found on tokenized volume")
    files = sorted(str(p) for p in root_path.rglob("*.npy"))
    print(f"[orchestrate] root={root}  files={len(files)}")

    stripped = 0
    already_raw = 0
    missing = 0
    wrong_dtype = 0
    total_tokens = 0
    failed = 0
    start = time.time()
    for r in strip_one.map(files, return_exceptions=True):
        if isinstance(r, Exception):
            print(f"[FAIL] {r!r}")
            failed += 1
            continue
        if r["status"] == "stripped":
            stripped += 1
            total_tokens += r["tokens"]
        elif r["status"] == "already_raw":
            already_raw += 1
        elif r["status"] == "missing":
            missing += 1
        elif r["status"] == "wrong_dtype":
            wrong_dtype += 1
            print(f"[wrong_dtype] {r['path']}: {r['dtype']}")
    elapsed = time.time() - start
    print(
        f"[orchestrate] DONE in {elapsed:.0f}s: "
        f"stripped={stripped} already_raw={already_raw} "
        f"missing={missing} wrong_dtype={wrong_dtype} failed={failed}"
    )
    print(f"[orchestrate] total tokens (from stripped files): {total_tokens:,}")
    tokenized_volume.commit()
    return {
        "root": root,
        "files": len(files),
        "stripped": stripped,
        "already_raw": already_raw,
        "missing": missing,
        "wrong_dtype": wrong_dtype,
        "failed": failed,
        "total_tokens": total_tokens,
        "elapsed_seconds": elapsed,
    }


@app.local_entrypoint()
def main(root: str = "/tokenized") -> None:
    summary = orchestrate.remote(root=root)
    print()
    print("=== SUMMARY ===")
    print(summary)
