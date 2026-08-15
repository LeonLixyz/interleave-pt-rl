"""Tokenize Nemotron-CC-Math-v1 parquet shards into OLMo-core-compatible `.npy` files.

Output format (matches `olmo_core.data.NumpyFSLDataset` expectations):
- Plain `numpy.memmap` `.npy` (memmap binary, no header beyond the npy header).
- dtype: `uint32` (dolma2 vocab is 100,278 — too large for uint16).
- Documents concatenated with EOS (token id 100257) as separator.

Usage:
    # Inspect one shard first to make sure column name + tokenizer work:
    modal run tokenize_corpus.py::inspect --shard 3/part_000000.parquet

    # Tokenize one shard (smoke test):
    modal run tokenize_corpus.py::tokenize_one --shard 3/part_000000.parquet

    # Fan out across all Math-3+ shards (detached):
    modal run --detach tokenize_corpus.py --subset 3
"""

from __future__ import annotations

import time
from pathlib import Path

import modal

from common import (
    DATA_MOUNT,
    TOKENIZED_MOUNT,
    data_volume,
    hf_image,
    tokenized_volume,
)

# dolma2 tokenizer (matches `olmo_core.data.TokenizerConfig.dolma2()`)
TOKENIZER_ID = "allenai/dolma2-tokenizer"
EOS_TOKEN_ID = 100257
VOCAB_SIZE = 100278


def _tokenize_image() -> modal.Image:
    # hf_image already has tokenizers, pyarrow, numpy
    return hf_image()


app = modal.App("nemotron-cc-math-tokenize", image=_tokenize_image())

hf_secret = modal.Secret.from_name("huggingface-secret")


def _list_shards(subset: str) -> list[str]:
    """List parquet shards under <subset>/ that are already downloaded on the data volume."""
    root = Path(DATA_MOUNT) / subset
    if not root.is_dir():
        raise FileNotFoundError(f"{root} not found on data volume; download first")
    shards = sorted(str(p.relative_to(DATA_MOUNT)) for p in root.glob("part_*.parquet"))
    return shards


def _text_column(table) -> str:
    """Auto-detect the text column in the parquet table."""
    for cand in ("text", "content", "raw_content"):
        if cand in table.column_names:
            return cand
    raise RuntimeError(f"no text column found; columns are {table.column_names}")


@app.function(
    secrets=[hf_secret],
    volumes={DATA_MOUNT: data_volume},
    timeout=600,
    cpu=2.0,
    memory=8 * 1024,
)
def inspect(shard: str) -> dict:
    """Read a parquet shard, print schema + sample, return basic stats."""
    import pyarrow.parquet as pq

    data_volume.reload()
    path = Path(DATA_MOUNT) / shard
    print(f"[inspect] file: {path}  size={path.stat().st_size / 1e9:.2f} GB")
    pf = pq.ParquetFile(str(path))
    print(f"[inspect] num_row_groups={pf.num_row_groups}  num_rows={pf.metadata.num_rows}")
    print(f"[inspect] schema:\n{pf.schema}")
    # Read first row group only for a sample (pure pyarrow, no pandas).
    rg = pf.read_row_group(0)
    text_col = _text_column(rg)
    print(f"[inspect] columns: {rg.column_names}")
    first_text = rg.column(text_col)[0].as_py()
    print(f"[inspect] first row text head (500 chars):")
    print(repr(first_text[:500]))
    # Also dump first 3 doc lengths so we know the size distribution.
    sample_texts = rg.column(text_col).slice(0, 5).to_pylist()
    print("[inspect] first 5 doc lengths (chars):", [len(t) for t in sample_texts])
    return {
        "file": shard,
        "rows": pf.metadata.num_rows,
        "row_groups": pf.num_row_groups,
        "text_col": text_col,
        "size_gb": path.stat().st_size / 1e9,
    }


@app.function(
    secrets=[hf_secret],
    volumes={DATA_MOUNT: data_volume, TOKENIZED_MOUNT: tokenized_volume},
    timeout=60 * 60 * 4,
    cpu=16.0,
    memory=32 * 1024,
)
def tokenize_one(shard: str, batch_size: int = 1024, overwrite: bool = False) -> dict:
    """Tokenize a single parquet shard → `<TOKENIZED_MOUNT>/<shard>.npy`."""
    import os

    import numpy as np
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer

    out_rel = Path(shard).with_suffix(".npy")
    out_path = Path(TOKENIZED_MOUNT) / out_rel
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not overwrite:
        size = out_path.stat().st_size
        # 4 bytes/uint32 token
        n_tok = size // 4
        print(f"[tokenize] {shard} already tokenized ({n_tok:,} tokens) — skip")
        return {"shard": shard, "skipped": True, "tokens": n_tok}

    data_volume.reload()

    # Load tokenizer
    print(f"[tokenize] loading tokenizer {TOKENIZER_ID}")
    tok = Tokenizer.from_pretrained(TOKENIZER_ID)
    n_threads = os.cpu_count() or 16
    print(f"[tokenize] using {n_threads} threads")

    # Stream the parquet shard row group by row group
    in_path = Path(DATA_MOUNT) / shard
    pf = pq.ParquetFile(str(in_path))
    total_rows = pf.metadata.num_rows
    print(f"[tokenize] {shard}: {total_rows:,} rows across {pf.num_row_groups} row groups")

    # Write to a temp file ending in .npy (so np.save doesn't append another extension),
    # then atomic-rename to the final path.
    tmp_path = out_path.with_name(out_path.stem + ".tmp.npy")
    chunks: list[np.ndarray] = []
    total_tokens = 0
    start = time.time()

    for rg_idx in range(pf.num_row_groups):
        rg = pf.read_row_group(rg_idx)
        text_col = _text_column(rg)
        texts = rg.column(text_col).to_pylist()

        # Batch encode for speed
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encodings = tok.encode_batch(batch)
            for enc in encodings:
                ids = enc.ids
                # Concatenate as uint32, append EOS
                arr = np.empty(len(ids) + 1, dtype=np.uint32)
                arr[:-1] = ids
                arr[-1] = EOS_TOKEN_ID
                chunks.append(arr)
                total_tokens += len(arr)

        elapsed = time.time() - start
        rate = total_tokens / max(elapsed, 1e-6)
        print(
            f"[tokenize] {shard} rg {rg_idx + 1}/{pf.num_row_groups} | "
            f"{total_tokens / 1e9:.3f}B tokens | {rate / 1e6:.1f} Mtok/s"
        )

    # Concatenate all chunks once, write as .npy
    print(f"[tokenize] {shard} concatenating {len(chunks)} doc arrays ({total_tokens:,} tokens)")
    full = np.concatenate(chunks)
    assert full.dtype == np.uint32, f"unexpected dtype: {full.dtype}"
    assert int(full.max()) < VOCAB_SIZE, f"token id {full.max()} >= vocab size {VOCAB_SIZE}"

    # Write RAW uint32 bytes (no NPY header). OLMo-core's data loaders read
    # `.npy` files via `file_size // 4 == num_tokens`, so the npy header would
    # be interpreted as garbage token IDs.
    full.tofile(tmp_path)
    os.replace(tmp_path, out_path)
    # Free memory before we move on
    del full, chunks
    tokenized_volume.commit()

    elapsed = time.time() - start
    print(
        f"[tokenize] DONE {shard} -> {out_path.relative_to(TOKENIZED_MOUNT)} "
        f"{total_tokens:,} tokens in {elapsed:.0f}s"
    )
    return {
        "shard": shard,
        "skipped": False,
        "tokens": int(total_tokens),
        "rows": int(total_rows),
        "elapsed_s": elapsed,
    }


@app.function(
    volumes={DATA_MOUNT: data_volume, TOKENIZED_MOUNT: tokenized_volume},
    timeout=60 * 60 * 24,
)
def orchestrate(subset: str = "3") -> dict:
    data_volume.reload()
    shards = _list_shards(subset)
    print(f"[orchestrate] subset={subset} -> {len(shards)} shards to tokenize")
    for s in shards[:3]:
        print("  ", s)
    if len(shards) > 3:
        print(f"  ... and {len(shards) - 3} more")

    results = []
    total_tokens = 0
    failed = 0
    for r in tokenize_one.map(shards, return_exceptions=True):
        if isinstance(r, Exception):
            print(f"[FAIL] {r!r}")
            failed += 1
            continue
        results.append(r)
        total_tokens += r["tokens"]
        print(
            f"[ok] {r['shard']} "
            + ("[cached] " if r.get('skipped') else "")
            + f"{r['tokens'] / 1e9:.3f}B tokens"
        )

    print()
    print(f"[orchestrate] tokenized {len(results)} shards, {failed} failed")
    print(f"[orchestrate] TOTAL TOKENS: {total_tokens:,} ({total_tokens / 1e9:.2f}B)")
    tokenized_volume.commit()
    return {
        "subset": subset,
        "shards": len(results),
        "failed": failed,
        "total_tokens": total_tokens,
    }


@app.function(
    volumes={TOKENIZED_MOUNT: tokenized_volume},
    timeout=60 * 60,
    cpu=4.0,
    memory=8 * 1024,
)
def count_tokens(subset: str = "3") -> dict:
    """Walk the tokenized volume and return the total number of uint32 tokens.

    Searches recursively for *.npy files under the subset root, so both flat
    (Nemotron /3/, /4plus/) and nested (dolma3/ingredient1-*) layouts work.
    """
    import numpy as np

    tokenized_volume.reload()
    root = Path(TOKENIZED_MOUNT) / subset
    if not root.is_dir():
        return {"subset": subset, "files": 0, "tokens": 0, "bytes": 0}
    total_tokens = 0
    total_bytes = 0
    files = sorted(root.rglob("*.npy"))
    for p in files:
        size = p.stat().st_size
        total_bytes += size
        # np.load with mmap_mode is fast — just reads the header to get shape.
        arr = np.load(p, mmap_mode="r")
        assert arr.dtype == np.uint32, f"{p}: unexpected dtype {arr.dtype}"
        total_tokens += int(arr.shape[0])
    out = {
        "subset": subset,
        "files": len(files),
        "tokens": total_tokens,
        "bytes": total_bytes,
        "tokens_billions": total_tokens / 1e9,
        "bytes_gib": total_bytes / 1024**3,
    }
    print(f"\n=== TOKEN COUNT (subset={subset}) ===")
    print(f"  files:  {out['files']}")
    print(f"  tokens: {out['tokens']:,}  ({out['tokens_billions']:.3f}B)")
    print(f"  bytes:  {out['bytes']:,}  ({out['bytes_gib']:.2f} GiB)")
    return out


@app.local_entrypoint()
def main(subset: str = "3") -> None:
    summary = orchestrate.remote(subset=subset)
    print()
    print("=== SUMMARY ===")
    print(summary)
