"""Empirically verify FineMath score partitioning across Nemotron-CC-Math-v1 subsets.

For each subset (3, 4plus, 4plus_MIND), samples ~20k docs from 2-3 random parquet
shards, extracts metadata.finemath_int_scores, and prints the score histogram.

Usage:
    modal run verify_scores.py::check_score_distribution
"""

from __future__ import annotations

import modal

from common import DATA_MOUNT, data_volume, hf_image

app = modal.App("nemotron-cc-math-verify-scores", image=hf_image())

SUBSETS = ["3", "4plus", "4plus_MIND"]
DOCS_PER_SUBSET = 20_000
SHARDS_PER_SUBSET = 3


@app.function(
    volumes={DATA_MOUNT: data_volume},
    timeout=60 * 30,
    cpu=4.0,
    memory=16 * 1024,
)
def check_score_distribution() -> dict:
    """Report the FineMath score histogram for each subset."""
    import random
    from collections import Counter
    from pathlib import Path

    import pyarrow.parquet as pq

    random.seed(0)
    results: dict[str, dict] = {}

    for subset in SUBSETS:
        subset_dir = Path(DATA_MOUNT) / subset
        if not subset_dir.exists():
            print(f"[{subset}] directory missing at {subset_dir} — skipping")
            results[subset] = {"error": "missing"}
            continue

        shards = sorted(subset_dir.glob("*.parquet"))
        if not shards:
            print(f"[{subset}] no parquet shards found in {subset_dir} — skipping")
            results[subset] = {"error": "no_shards"}
            continue

        n_pick = min(SHARDS_PER_SUBSET, len(shards))
        picked = random.sample(shards, n_pick)
        print(f"[{subset}] total shards: {len(shards)}; sampling {n_pick}:")
        for p in picked:
            print(f"    - {p.name}")

        score_counts: Counter = Counter()
        total_docs = 0
        per_shard_target = max(1, DOCS_PER_SUBSET // n_pick)
        score_field_found = None
        sample_metadata_keys: list[str] = []

        for shard in picked:
            pf = pq.ParquetFile(shard)
            schema_names = pf.schema_arrow.names
            # Some Nemotron parquet files store the score at top level; others
            # nest it inside a `metadata` struct.  Try the nested path first.
            top_level_candidates = [
                c for c in schema_names
                if "finemath" in c.lower() and "score" in c.lower()
            ]

            collected = 0
            for batch in pf.iter_batches(batch_size=4096):
                tbl = batch
                # Resolve where the score lives.
                if score_field_found is None:
                    if "metadata" in schema_names:
                        meta = tbl.column("metadata")
                        # struct
                        if hasattr(meta.type, "fields"):
                            field_names = [f.name for f in meta.type]
                            sample_metadata_keys = field_names
                            for fname in field_names:
                                if "finemath" in fname.lower() and "score" in fname.lower():
                                    score_field_found = ("metadata_struct", fname)
                                    break
                        # Could be a JSON-encoded string — handle below
                        if score_field_found is None and not sample_metadata_keys:
                            # try parsing first row as JSON to see keys
                            import json as _json
                            first = meta[0].as_py()
                            if isinstance(first, str):
                                try:
                                    parsed = _json.loads(first)
                                    if isinstance(parsed, dict):
                                        sample_metadata_keys = list(parsed.keys())
                                        for k in parsed:
                                            if (
                                                "finemath" in k.lower()
                                                and "score" in k.lower()
                                            ):
                                                score_field_found = ("metadata_json", k)
                                                break
                                except Exception:
                                    pass
                    if score_field_found is None and top_level_candidates:
                        score_field_found = ("top_level", top_level_candidates[0])

                    if score_field_found is None:
                        print(
                            f"[{subset}] could not find finemath score field. "
                            f"schema={schema_names}; metadata sub-keys="
                            f"{sample_metadata_keys}"
                        )
                        break

                kind, fname = score_field_found
                if kind == "top_level":
                    values = tbl.column(fname).to_pylist()
                elif kind == "metadata_struct":
                    meta = tbl.column("metadata")
                    values = meta.field(fname).to_pylist()
                elif kind == "metadata_json":
                    import json as _json
                    meta_strs = tbl.column("metadata").to_pylist()
                    values = []
                    for s in meta_strs:
                        try:
                            values.append(_json.loads(s).get(fname))
                        except Exception:
                            values.append(None)
                else:
                    values = []

                for v in values:
                    if v is None:
                        score_counts["null"] += 1
                    else:
                        score_counts[int(v)] += 1
                collected += len(values)
                total_docs += len(values)

                if collected >= per_shard_target:
                    break

            if score_field_found is None:
                # nothing more we can do for this subset
                break

        # Tidy printable histogram
        hist = {k: score_counts[k] for k in sorted(score_counts.keys(), key=lambda x: (isinstance(x, str), x))}
        print(
            f"subset {subset}: scores = {hist} "
            f"(total sampled: {total_docs}; field: {score_field_found})"
        )
        results[subset] = {
            "total_sampled": total_docs,
            "histogram": hist,
            "score_field": score_field_found,
        }

    print()
    print("=== VERDICT ===")
    for subset, info in results.items():
        if "histogram" not in info:
            print(f"  {subset}: {info}")
            continue
        keys = [k for k in info["histogram"] if isinstance(k, int)]
        if subset == "3":
            if keys == [3]:
                print(f"  /3/: score==3 ONLY (confirms research)")
            else:
                print(f"  /3/: scores observed = {sorted(keys)} (contradicts research)")
        elif subset == "4plus":
            print(f"  /4plus/: scores observed = {sorted(keys)}")
        elif subset == "4plus_MIND":
            print(f"  /4plus_MIND/: scores observed = {sorted(keys)}")

    return results


@app.local_entrypoint()
def main() -> None:
    check_score_distribution.remote()
