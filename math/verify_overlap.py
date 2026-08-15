"""Verify WARC_ID / DOC_ID overlap across the three Nemotron-CC-Math-v1 subsets.

Reads samples from the parquet shards already staged on the
"nemotron-cc-math-v1" Modal volume and computes pairwise intersection sizes
between the (3, 4plus, 4plus_MIND) subsets for both the top-level `id`
column and the `metadata.warc_id` column.

Usage:
    modal run verify_overlap.py
"""

from __future__ import annotations

import modal

from common import (
    DATA_MOUNT,
    TOKENIZED_MOUNT,
    data_volume,
    hf_image,
    tokenized_volume,
)

app = modal.App("nemotron-overlap-verify", image=hf_image())

SUBSETS = ["3", "4plus", "4plus_MIND"]
SHARDS_PER_SUBSET = 4
ROWS_PER_SHARD = 12500


@app.function(
    volumes={DATA_MOUNT: data_volume},
    timeout=60 * 30,
    cpu=4.0,
    memory=16 * 1024,
)
def check_id_overlap() -> dict:
    """Sample ~50k docs from each subset and compute pairwise ID overlap."""
    from pathlib import Path

    import pyarrow.parquet as pq

    root = Path(DATA_MOUNT)

    # ---- 1) Sample IDs + WARC IDs from each subset ----------------------
    samples: dict[str, dict[str, set]] = {}
    per_subset_info: dict[str, dict] = {}

    for subset in SUBSETS:
        subset_dir = root / subset
        all_shards = sorted(subset_dir.glob("*.parquet"))
        if not all_shards:
            raise RuntimeError(f"No parquet files found under {subset_dir}")

        # Pick shards evenly spaced across the subset so we sample widely.
        if len(all_shards) <= SHARDS_PER_SUBSET:
            chosen = all_shards
        else:
            step = max(1, len(all_shards) // SHARDS_PER_SUBSET)
            chosen = [all_shards[i * step] for i in range(SHARDS_PER_SUBSET)]

        ids: set[str] = set()
        warc_ids: set[str] = set()
        rows_read = 0

        print(f"[{subset}] {len(all_shards)} total shards; sampling {len(chosen)}:")
        for shard in chosen:
            pf = pq.ParquetFile(shard)
            if pf.num_row_groups == 0:
                print(f"  - {shard.name}: 0 row groups, skipping")
                continue
            # Read the first row group, only the columns we need.
            tbl = pf.read_row_group(0, columns=["id", "metadata"])
            n = min(ROWS_PER_SHARD, tbl.num_rows)
            tbl = tbl.slice(0, n)

            id_col = tbl.column("id").to_pylist()
            meta_col = tbl.column("metadata").to_pylist()

            for v in id_col:
                if v is not None:
                    ids.add(v)
            for m in meta_col:
                if isinstance(m, dict):
                    w = m.get("warc_id")
                    if w is not None:
                        warc_ids.add(w)
            rows_read += n
            print(
                f"  - {shard.name}: read {n} rows "
                f"(cum ids={len(ids)}, cum warc_ids={len(warc_ids)})"
            )

        samples[subset] = {"id": ids, "warc_id": warc_ids}
        per_subset_info[subset] = {
            "shards_sampled": len(chosen),
            "shards_total": len(all_shards),
            "rows_read": rows_read,
            "unique_ids": len(ids),
            "unique_warc_ids": len(warc_ids),
        }
        print(
            f"[{subset}] done: rows={rows_read} "
            f"unique_ids={len(ids)} unique_warc_ids={len(warc_ids)}"
        )

    # ---- 2) Pairwise intersections --------------------------------------
    pairs = [("3", "4plus"), ("4plus", "4plus_MIND"), ("3", "4plus_MIND")]
    results: dict[str, dict] = {}

    print()
    print("=" * 78)
    print("PAIRWISE OVERLAP")
    print("=" * 78)

    for a, b in pairs:
        pair_key = f"{a} vs {b}"
        pair_stats: dict = {}
        for col in ("id", "warc_id"):
            sa = samples[a][col]
            sb = samples[b][col]
            inter = sa & sb
            sym_diff = sa ^ sb
            n_inter = len(inter)
            n_sa, n_sb = len(sa), len(sb)
            min_size = min(n_sa, n_sb) or 1
            rate = n_inter / min_size

            pair_stats[col] = {
                "size_a": n_sa,
                "size_b": n_sb,
                "intersection": n_inter,
                "symmetric_difference": len(sym_diff),
                "intersection_rate_over_min": rate,
            }

            print(
                f"[{pair_key}] col={col:8s} "
                f"|A|={n_sa:6d}  |B|={n_sb:6d}  "
                f"|A∩B|={n_inter:6d}  rate={rate:.4f}  "
                f"|A△B|={len(sym_diff):6d}"
            )
        results[pair_key] = pair_stats

    print()
    print("=" * 78)
    print("PER-SUBSET INFO")
    print("=" * 78)
    for s, info in per_subset_info.items():
        print(f"  {s}: {info}")

    # ---- 3) Headline verdict --------------------------------------------
    print()
    print("=" * 78)
    print("HEADLINE")
    print("=" * 78)
    for a, b in pairs:
        key = f"{a} vs {b}"
        wi = results[key]["warc_id"]["intersection"]
        ii = results[key]["id"]["intersection"]
        print(f"  {key}:  id_overlap={ii}   warc_id_overlap={wi}")

    return {
        "per_subset": per_subset_info,
        "pairs": results,
    }


# ---------------------------------------------------------------------------
# TOKEN-LEVEL N-GRAM OVERLAP CHECK
# ---------------------------------------------------------------------------
# Verifies whether the three Nemotron-CC-Math-v1 subsets ("3", "4plus",
# "4plus_MIND") share content at the actual training-data layer (uint32
# tokens stored on the `math-pretraining-tokenized` Modal volume).

TOKEN_SUBSETS = ("3", "4plus", "4plus_MIND")
# Distinct sampling seeds per subset so the positions we draw N-grams from
# are NOT correlated across subsets.
SUBSET_SEEDS = {"3": 11, "4plus": 22, "4plus_MIND": 33}


@app.function(
    volumes={TOKENIZED_MOUNT: tokenized_volume},
    timeout=60 * 60,
    cpu=4.0,
    memory=32 * 1024,
)
def check_token_ngram_overlap(
    n_values: str = "32,128",
    target_samples: int = 1_000_000,
    files_per_subset: int = 3,
) -> dict:
    """For each subset, sample ~target_samples random N-grams of length N from
    a handful of `.npy` shards on the tokenized volume, hash them, and compute
    pairwise intersection sizes between the three subsets.

    A "natural baseline" rate (~0.01-0.1%) is expected from common N-grams
    like "the equation is" or LaTeX boilerplate. Significantly higher rates,
    especially for the (3, 4plus) pair (which is supposed to be totally
    disjoint), indicate effective duplication at the training-data layer.
    """
    import random
    from hashlib import blake2b
    from pathlib import Path

    import numpy as np

    tokenized_volume.reload()

    n_list = tuple(int(x.strip()) for x in str(n_values).split(",") if x.strip())

    def _pick_random_shards(subset: str, k: int, rng: random.Random) -> list[Path]:
        root = Path(TOKENIZED_MOUNT) / subset
        if not root.is_dir():
            raise FileNotFoundError(f"{root} not found on tokenized volume")
        shards = sorted(root.glob("part_*.npy"))
        if not shards:
            raise FileNotFoundError(f"no part_*.npy files under {root}")
        if len(shards) <= k:
            return shards
        return rng.sample(shards, k)

    def _sample_ngram_hashes(subset: str, n: int) -> tuple[set, int]:
        seed = SUBSET_SEEDS[subset]
        rng = random.Random(seed)
        np_rng = np.random.default_rng(seed)

        shards = _pick_random_shards(subset, files_per_subset, rng)
        print(f"[{subset}] using {len(shards)} shards:")
        for p in shards:
            sz = p.stat().st_size
            print(f"    {p.name}  ({sz / 1e9:.2f} GB)")

        per_file = max(1, target_samples // len(shards))
        hashes: set = set()
        total_sampled = 0

        for p in shards:
            arr = np.load(p, mmap_mode="r")
            assert arr.dtype == np.uint32, f"{p}: dtype={arr.dtype}"
            length = arr.shape[0]
            if length <= n:
                print(f"[{subset}] WARN: {p.name} only has {length} tokens, skipping")
                continue
            max_start = length - n
            starts = np_rng.integers(0, max_start, size=per_file, dtype=np.int64)
            # Sort to keep mmap reads roughly sequential (faster on cold cache).
            starts.sort()
            for start in starts:
                ng = arr[int(start) : int(start) + n]
                h = blake2b(ng.tobytes(), digest_size=16).digest()
                hashes.add(h)
                total_sampled += 1
            del arr

        print(
            f"[{subset}] N={n}: sampled {total_sampled:,} N-grams, "
            f"{len(hashes):,} unique "
            f"({len(hashes) / max(total_sampled, 1) * 100:.2f}%)"
        )
        return hashes, total_sampled

    results: dict = {"by_n": {}}

    for n in n_list:
        print()
        print("=" * 78)
        print(f"  N-GRAM OVERLAP CHECK  N={n}  target_samples={target_samples:,}")
        print("=" * 78)

        sets_by_subset: dict[str, set] = {}
        sizes_by_subset: dict[str, int] = {}
        for subset in TOKEN_SUBSETS:
            hashes, drawn = _sample_ngram_hashes(subset, n=n)
            sets_by_subset[subset] = hashes
            sizes_by_subset[subset] = drawn

        print()
        print(f"--- Pairwise intersections at N={n} ---")
        pairs = [("3", "4plus"), ("3", "4plus_MIND"), ("4plus", "4plus_MIND")]
        pair_results: dict[str, dict] = {}
        for a, b in pairs:
            sa, sb = sets_by_subset[a], sets_by_subset[b]
            inter = sa & sb
            denom = min(len(sa), len(sb))
            rate = len(inter) / max(denom, 1)
            print(
                f"  {a:>11s} <-> {b:<11s}  |a|={len(sa):,}  |b|={len(sb):,}  "
                f"|a&b|={len(inter):,}  rate={rate * 100:.5f}%"
            )
            pair_results[f"{a}__{b}"] = {
                "a_size": len(sa),
                "b_size": len(sb),
                "intersection": len(inter),
                "rate_pct": rate * 100,
            }

        results["by_n"][n] = {
            "subset_sample_sizes": sizes_by_subset,
            "subset_unique_sizes": {k: len(v) for k, v in sets_by_subset.items()},
            "pairs": pair_results,
        }

        del sets_by_subset

    print()
    print("=" * 78)
    print("  TOKEN N-GRAM FINAL SUMMARY")
    print("=" * 78)
    for n, rec in results["by_n"].items():
        print(f"  N={n}:")
        for pair_name, pair in rec["pairs"].items():
            print(
                f"    {pair_name:>24s}  rate={pair['rate_pct']:.5f}%  "
                f"|a&b|={pair['intersection']:,}"
            )
    return results


# ---------------------------------------------------------------------------
# DOLMINO vs NEMOTRON OVERLAP CHECK
# ---------------------------------------------------------------------------
# Most likely contamination risk: Dolmino's dclm subset is Common-Crawl-derived
# and Nemotron-CC-Math-v1 is ALSO Common-Crawl-derived. If the same WARC docs
# were filtered into both, we'd have effective duplication in our training mix.
# This probe samples warc_ids from Nemotron parquet AND from Dolmino .json.zst/
# .gz/.jsonl and reports pairwise intersection.

from common import (
    DOLMINO_MOUNT,
    dolmino_volume,
)


def _dolmino_open_lines(path):
    """Yield decoded lines from a Dolmino file (mirrors tokenize_dolmino logic)."""
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
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                yield line


@app.function(
    image=hf_image().pip_install("zstandard>=0.22", "orjson>=3.10"),
    volumes={DATA_MOUNT: data_volume, DOLMINO_MOUNT: dolmino_volume},
    timeout=60 * 30,
    cpu=4.0,
    memory=16 * 1024,
)
def check_dolmino_nemotron_overlap(
    nemotron_subsets: str = "3,4plus,4plus_MIND",
    dolmino_subsets: str = "dclm,math",
    nemotron_rows_per_subset: int = 30_000,
    dolmino_docs_per_subset: int = 30_000,
) -> dict:
    """Probe cross-corpus warc_id / id overlap between Nemotron-CC-Math and Dolmino.

    Returns counts and overlap rate. We expect near-zero overlap; significantly
    above the boilerplate baseline (~0.01-0.1%) would indicate dedup is needed.
    """
    from pathlib import Path

    import orjson
    import pyarrow.parquet as pq

    data_volume.reload()
    dolmino_volume.reload()

    n_subs = [s.strip() for s in nemotron_subsets.split(",") if s.strip()]
    d_subs = [s.strip() for s in dolmino_subsets.split(",") if s.strip()]

    # ---- Sample Nemotron warc_ids + ids -----------------------------------
    nemotron_ids: set[str] = set()
    nemotron_warc: set[str] = set()
    for sub in n_subs:
        sub_dir = Path(DATA_MOUNT) / sub
        shards = sorted(sub_dir.glob("*.parquet"))
        if not shards:
            print(f"[nemotron/{sub}] no shards found, skipping")
            continue
        # Sample evenly across shards
        step = max(1, len(shards) // 5)
        chosen = shards[::step][:5]
        rows_per = nemotron_rows_per_subset // max(len(chosen), 1)
        print(f"[nemotron/{sub}] sampling {rows_per} rows from {len(chosen)} shards")
        for shard in chosen:
            pf = pq.ParquetFile(shard)
            if pf.num_row_groups == 0:
                continue
            tbl = pf.read_row_group(0, columns=["id", "metadata"])
            n = min(rows_per, tbl.num_rows)
            tbl = tbl.slice(0, n)
            for v in tbl.column("id").to_pylist():
                if v is not None:
                    nemotron_ids.add(v)
            for m in tbl.column("metadata").to_pylist():
                if isinstance(m, dict) and m.get("warc_id"):
                    nemotron_warc.add(m["warc_id"])
    print(
        f"[nemotron] cumulative: unique_ids={len(nemotron_ids):,} "
        f"unique_warc_ids={len(nemotron_warc):,}"
    )

    # ---- Sample Dolmino IDs (try a few common field names) ----------------
    # Dolmino docs typically have an 'id' field; some sub-subsets may carry
    # 'warc_id' or 'metadata.warc_id' too. Collect any string identifiers
    # plus any explicit warc_ids we find.
    dolmino_ids: set[str] = set()
    dolmino_warc: set[str] = set()
    dolmino_field_keys: dict[str, int] = {}

    for sub in d_subs:
        sub_dir = Path(DOLMINO_MOUNT) / "data" / sub
        if not sub_dir.is_dir():
            print(f"[dolmino/{sub}] {sub_dir} not on volume, skipping")
            continue
        # Sample some files spread across the subset
        all_files = []
        for ext in (".json.zst", ".json.gz", ".jsonl.zst", ".jsonl.gz", ".jsonl"):
            all_files.extend(sub_dir.rglob(f"*{ext}"))
        all_files = sorted(set(all_files))
        if not all_files:
            print(f"[dolmino/{sub}] no files in {sub_dir}, skipping")
            continue
        step = max(1, len(all_files) // 3)
        chosen = all_files[::step][:3]
        docs_per = dolmino_docs_per_subset // max(len(chosen), 1)
        print(f"[dolmino/{sub}] sampling {docs_per} docs from {len(chosen)} files")
        for path in chosen:
            count = 0
            try:
                for line in _dolmino_open_lines(path):
                    if count >= docs_per:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        doc = orjson.loads(line)
                    except Exception:
                        continue
                    for k in doc.keys():
                        dolmino_field_keys[k] = dolmino_field_keys.get(k, 0) + 1
                    # Collect any string-valued 'id'-ish identifier
                    if isinstance(doc.get("id"), str):
                        dolmino_ids.add(doc["id"])
                    # warc_id might live at top-level or under 'metadata'
                    if isinstance(doc.get("warc_id"), str):
                        dolmino_warc.add(doc["warc_id"])
                    meta = doc.get("metadata")
                    if isinstance(meta, dict) and isinstance(meta.get("warc_id"), str):
                        dolmino_warc.add(meta["warc_id"])
                    count += 1
            except Exception as e:
                print(f"[dolmino/{sub}] WARN reading {path.name}: {e}")
    print(
        f"[dolmino] cumulative: unique_ids={len(dolmino_ids):,} "
        f"unique_warc_ids={len(dolmino_warc):,}"
    )
    print(f"[dolmino] top field keys observed: {sorted(dolmino_field_keys.items(), key=lambda x: -x[1])[:10]}")

    # ---- Overlap ---------------------------------------------------------
    print()
    print("=" * 78)
    print("DOLMINO vs NEMOTRON OVERLAP")
    print("=" * 78)
    overlap_id = nemotron_ids & dolmino_ids
    overlap_warc = nemotron_warc & dolmino_warc
    denom_id = min(len(nemotron_ids), len(dolmino_ids))
    denom_warc = min(len(nemotron_warc), len(dolmino_warc))
    rate_id = (len(overlap_id) / denom_id) if denom_id else 0.0
    rate_warc = (len(overlap_warc) / denom_warc) if denom_warc else 0.0

    print(
        f"  id_overlap:   nemotron|={len(nemotron_ids):,} dolmino|={len(dolmino_ids):,} "
        f"int={len(overlap_id):,} rate={rate_id * 100:.4f}%"
    )
    print(
        f"  warc_overlap: nemotron|={len(nemotron_warc):,} dolmino|={len(dolmino_warc):,} "
        f"int={len(overlap_warc):,} rate={rate_warc * 100:.4f}%"
    )

    return {
        "nemotron_ids": len(nemotron_ids),
        "nemotron_warc": len(nemotron_warc),
        "dolmino_ids": len(dolmino_ids),
        "dolmino_warc": len(dolmino_warc),
        "dolmino_field_keys": dolmino_field_keys,
        "id_intersection": len(overlap_id),
        "warc_intersection": len(overlap_warc),
        "id_rate_pct": rate_id * 100,
        "warc_rate_pct": rate_warc * 100,
    }


@app.local_entrypoint()
def main() -> None:
    out = check_id_overlap.remote()
    print()
    print("=== RETURNED ===")
    print(out)
