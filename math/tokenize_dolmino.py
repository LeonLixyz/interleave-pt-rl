"""Tokenize Dolmino's .json.zst shards into OLMo-core-compatible .npy files.

Dolmino files: zstd-compressed JSON Lines, each line a doc dict with a 'text' field.
Output: uint32 .npy with docs concatenated and EOS (100257) separated, mirroring
input directory structure. One .npy per input .json.zst.

Output layout on the math-pretraining-tokenized volume:
    /tokenized/dolmino/<subset>/<batch>/<basename>.npy

Usage:
    modal run tokenize_dolmino.py::inspect --file data/math/tinyGSM-MIND/0000.json.zst
    modal run tokenize_dolmino.py::tokenize_one --file data/dclm/0000/dclm-0000.json.zst
    modal run --detach tokenize_dolmino.py --subsets math,dclm,flan,pes2o,stackexchange,wiki
"""

from __future__ import annotations

import time
from pathlib import Path

import modal

from common import (
    DOLMINO_MOUNT,
    TOKENIZED_MOUNT,
    dolmino_volume,
    hf_image_base,
    tokenized_volume,
)

TOKENIZER_ID = "allenai/dolma2-tokenizer"
EOS_TOKEN_ID = 100257
VOCAB_SIZE = 100278


def _dolmino_tokenize_image() -> modal.Image:
    # Pip-install extras BEFORE adding local Python sources (Modal rule).
    return (
        hf_image_base()
        .pip_install("zstandard>=0.22", "orjson>=3.10")
        .add_local_python_source("common")
    )


app = modal.App("dolmino-tokenize", image=_dolmino_tokenize_image())

hf_secret = modal.Secret.from_name("huggingface-secret")


_DATA_EXTS = (".json.zst", ".json.gz", ".jsonl.zst", ".jsonl.gz", ".jsonl")


def _list_files(subsets: list[str]) -> list[str]:
    """List all JSON-Lines-ish files under each subset on the volume.

    Each "subset" may be either an exact top-level folder name (e.g.
    "ingredient1-cranecode") OR a glob-like prefix (e.g. "ingredient1-" to
    match all of Ingredient 1's sub-buckets).

    The special value "*" or "all" matches every top-level folder under data/.
    """
    data_root = Path(DOLMINO_MOUNT) / "data"
    if not data_root.is_dir():
        print(f"[list] WARN: {data_root} not found on volume")
        return []

    # Build set of matching top-level folder names.
    top_dirs = sorted(p.name for p in data_root.iterdir() if p.is_dir())
    matched: list[str] = []
    for sub in subsets:
        s = sub.strip()
        if s in ("*", "all"):
            matched = top_dirs
            break
        if s in top_dirs:
            matched.append(s)
        else:
            # Treat as prefix.
            for d in top_dirs:
                if d.startswith(s):
                    matched.append(d)
    # De-dup while preserving order.
    seen: set[str] = set()
    uniq_matched = [m for m in matched if not (m in seen or seen.add(m))]

    files: list[str] = []
    for d in uniq_matched:
        root = data_root / d
        for p in root.rglob("*"):
            if p.is_file() and any(p.name.endswith(ext) for ext in _DATA_EXTS):
                files.append(str(p.relative_to(DOLMINO_MOUNT)))
    files.sort()
    return files


def _open_lines(path: Path):
    """Open a Dolmino file (.json.zst | .json.gz | .jsonl.zst | .jsonl.gz | .jsonl)
    and yield decoded JSON-Lines as text lines."""
    import gzip
    import io

    import zstandard as zstd

    name = path.name
    if name.endswith(".zst"):
        decompressor = zstd.ZstdDecompressor()
        fh = path.open("rb")
        stream = decompressor.stream_reader(fh)
        text = io.TextIOWrapper(stream, encoding="utf-8")
        for line in text:
            yield line
        text.close()
    elif name.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                yield line
    else:
        # plain .jsonl
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                yield line


def _output_path_for(input_rel: str) -> Path:
    """Map data/<subset>/<batch>/<name>.<ext> -> dolma3/<subset>/<batch>/<name>.npy

    Strips any of .json.zst, .json.gz, .jsonl.zst, .jsonl.gz, .jsonl, and any
    quirky variants by repeatedly peeling known suffixes.
    """
    p = Path(input_rel)
    parts = p.parts
    if parts[0] == "data":
        rest = parts[1:]
    else:
        rest = parts
    name = rest[-1]
    for ext in (".json.zst", ".json.gz", ".jsonl.zst", ".jsonl.gz", ".jsonl", ".zst", ".gz"):
        if name.endswith(ext):
            name = name[: -len(ext)]
            break
    out_rel = Path("dolma3") / Path(*rest[:-1]) / f"{name}.npy"
    return Path(TOKENIZED_MOUNT) / out_rel


@app.function(
    secrets=[hf_secret],
    volumes={DOLMINO_MOUNT: dolmino_volume},
    timeout=300,
    cpu=2.0,
    memory=4 * 1024,
)
def inspect(file: str) -> dict:
    """Read a few lines from a .json.zst file and show keys + a text sample."""
    import orjson

    dolmino_volume.reload()
    path = Path(DOLMINO_MOUNT) / file
    print(f"[inspect] file: {path}  size={path.stat().st_size / 1e6:.1f} MB")
    keys_seen: set[str] = set()
    samples: list[dict] = []
    for i, line in enumerate(_open_lines(path)):
        if i >= 5:
            break
        line = line.strip()
        if not line:
            continue
        try:
            doc = orjson.loads(line)
            keys_seen.update(doc.keys())
            samples.append(doc)
        except Exception as e:
            samples.append({"_parse_error": str(e), "_line_head": line[:200]})
    print(f"[inspect] keys observed: {sorted(keys_seen)}")
    for i, s in enumerate(samples[:3]):
        # Prefer 'text', fall back to first string-valued field
        text = s.get("text") or next((v for v in s.values() if isinstance(v, str)), "")
        print(f"[inspect] sample {i} keys={list(s.keys())} text_head={text[:300]!r}")
    return {"file": file, "keys": sorted(keys_seen), "samples_read": len(samples)}


@app.function(
    secrets=[hf_secret],
    volumes={DOLMINO_MOUNT: dolmino_volume, TOKENIZED_MOUNT: tokenized_volume},
    timeout=60 * 60 * 4,
    cpu=16.0,
    memory=32 * 1024,
)
def tokenize_one(file: str, batch_size: int = 1024, overwrite: bool = False) -> dict:
    """Tokenize a single .json.zst file → .npy."""
    import os

    import numpy as np
    import orjson
    from tokenizers import Tokenizer

    out_path = _output_path_for(file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not overwrite:
        n_tok = out_path.stat().st_size // 4
        print(f"[tokenize] {file} already tokenized ({n_tok:,} tokens) — skip")
        return {"file": file, "skipped": True, "tokens": n_tok}

    dolmino_volume.reload()

    tok = Tokenizer.from_pretrained(TOKENIZER_ID)
    n_threads = os.cpu_count() or 16
    print(f"[tokenize] loading {file} with {n_threads} tokenizer threads")

    in_path = Path(DOLMINO_MOUNT) / file

    chunks: list[np.ndarray] = []
    total_tokens = 0
    doc_count = 0
    text_buf: list[str] = []
    start = time.time()

    def flush_batch(texts: list[str]) -> None:
        nonlocal total_tokens
        if not texts:
            return
        encodings = tok.encode_batch(texts)
        for enc in encodings:
            ids = enc.ids
            arr = np.empty(len(ids) + 1, dtype=np.uint32)
            arr[:-1] = ids
            arr[-1] = EOS_TOKEN_ID
            chunks.append(arr)
            total_tokens += len(arr)

    for line in _open_lines(in_path):
        line = line.strip()
        if not line:
            continue
        try:
            doc = orjson.loads(line)
        except Exception:
            continue
        text = doc.get("text") or doc.get("content") or doc.get("raw_content")
        if not text:
            continue
        text_buf.append(text)
        doc_count += 1
        if len(text_buf) >= batch_size:
            flush_batch(text_buf)
            text_buf = []
    if text_buf:
        flush_batch(text_buf)

    if total_tokens == 0:
        print(f"[tokenize] {file}: 0 tokens; writing empty placeholder")
        empty = np.zeros(0, dtype=np.uint32)
        tmp = out_path.with_name(out_path.stem + ".tmp_raw")
        empty.tofile(tmp)
        os.replace(tmp, out_path)
        tokenized_volume.commit()
        return {"file": file, "skipped": False, "tokens": 0, "docs": doc_count}

    elapsed = time.time() - start
    print(
        f"[tokenize] {file}: {doc_count:,} docs -> {total_tokens:,} tokens "
        f"in {elapsed:.1f}s ({total_tokens / max(elapsed, 1e-6) / 1e6:.1f} Mtok/s)"
    )

    full = np.concatenate(chunks)
    assert full.dtype == np.uint32
    assert int(full.max()) < VOCAB_SIZE, f"token {full.max()} >= vocab {VOCAB_SIZE}"

    # RAW uint32 bytes (no NPY header) — OLMo-core data loaders read raw bytes.
    tmp_path = out_path.with_name(out_path.stem + ".tmp_raw")
    full.tofile(tmp_path)
    os.replace(tmp_path, out_path)
    tokenized_volume.commit()

    del full, chunks
    return {
        "file": file,
        "skipped": False,
        "tokens": int(total_tokens),
        "docs": doc_count,
        "elapsed_s": elapsed,
    }


@app.function(
    volumes={DOLMINO_MOUNT: dolmino_volume, TOKENIZED_MOUNT: tokenized_volume},
    timeout=60 * 60 * 24,
)
def orchestrate(
    subsets: str = "ingredient1-",
) -> dict:
    dolmino_volume.reload()
    subset_list = [s.strip() for s in subsets.split(",") if s.strip()]
    files = _list_files(subset_list)
    print(f"[orchestrate] subsets={subset_list} -> {len(files)} files to tokenize")
    for f in files[:3]:
        print("  ", f)
    if len(files) > 3:
        print(f"  ... and {len(files) - 3} more")

    results = []
    total_tokens = 0
    by_subset_tokens: dict[str, int] = {}
    by_subset_files: dict[str, int] = {}
    failed = 0
    for r in tokenize_one.map(files, return_exceptions=True):
        if isinstance(r, Exception):
            print(f"[FAIL] {r!r}")
            failed += 1
            continue
        results.append(r)
        total_tokens += r["tokens"]
        sub = r["file"].split("/")[1]
        by_subset_tokens[sub] = by_subset_tokens.get(sub, 0) + r["tokens"]
        by_subset_files[sub] = by_subset_files.get(sub, 0) + 1

    print()
    print(f"[orchestrate] tokenized {len(results)} files, {failed} failed")
    print(f"[orchestrate] TOTAL TOKENS: {total_tokens:,} ({total_tokens / 1e9:.2f}B)")
    print("[orchestrate] per-subset:")
    for sub in subset_list:
        print(
            f"  {sub:>15}: {by_subset_files.get(sub, 0):>5} files, "
            f"{by_subset_tokens.get(sub, 0) / 1e9:.2f}B tokens"
        )
    tokenized_volume.commit()
    return {
        "subsets": subset_list,
        "files": len(results),
        "failed": failed,
        "total_tokens": total_tokens,
        "by_subset_tokens": by_subset_tokens,
    }


@app.local_entrypoint()
def main(subsets: str = "ingredient1-") -> None:
    summary = orchestrate.remote(subsets=subsets)
    print()
    print("=== SUMMARY ===")
    print(summary)
