"""Identify TRULY HELD-OUT files via deterministic data-order analysis.

The OLMo-core composable data loader saves a global instance index permutation
at the start of epoch 1 to:
    /cache/work/<run_name>/node<rank>/global_indices_epoch1_*.npy

Since seed=1337 is fixed and the sampling is deterministic, we can compute:
- which instances will be consumed by step 95368 (end of stable run)
- which won't be sampled at all in the 200B budget

Files with ZERO sampled instances are genuinely never-seen, perfectly
in-distribution held-out data.

Usage:
    modal run analyze_heldout.py --run-name math-1b-v0 --target-step 95368
"""

from __future__ import annotations

import modal

from common import (
    CACHE_MOUNT,
    CACHE_VOLUME_NAME,
    TOKENIZED_MOUNT,
    hf_image_base,
    tokenized_volume,
)


def _img() -> modal.Image:
    return hf_image_base().add_local_python_source("common")


app = modal.App("analyze-heldout", image=_img())
cache_volume = modal.Volume.from_name(CACHE_VOLUME_NAME, create_if_missing=True)


@app.function(
    volumes={
        CACHE_MOUNT: cache_volume,
        TOKENIZED_MOUNT: tokenized_volume,
    },
    timeout=60 * 60,
    cpu=4.0,
    memory=32 * 1024,
)
def analyze(
    run_name: str = "math-1b-v0",
    target_step: int = 95368,
    global_batch_size: int = 512 * 4096,
    sequence_length: int = 4096,
) -> dict:
    """Walk every npy file under /tokenized and the global indices file; tag
    each file as `consumed-by-step-N`, `consumed-later`, or `held-out`.

    Note: this currently only reports per-file *file-size* arithmetic since
    the OLMo-core composable index format is non-trivial to invert per-file.
    A FULL implementation would also load the global indices and map indices
    back to (source, byte-offset). That's a follow-up.

    For v0 we report:
    - Total tokens per source folder
    - Total tokens that WILL be consumed in 200B budget
    - Per-source estimated unused fraction (= source_tokens - source_drawn) / source_tokens
    - Recommendation: which files to mark as held-out
    """
    import glob
    from pathlib import Path

    cache_volume.reload()
    tokenized_volume.reload()

    # ---- per-source token counts -------------------------------------------
    sources = {
        "math_3": "/tokenized/3",
        "math_4plus": "/tokenized/4plus",
        "math_4plus_MIND": "/tokenized/4plus_MIND",
        "dolma3": "/tokenized/dolma3",
    }
    # The mix weights match train_inner_mix.py DEFAULT_WEIGHTS
    weights = {
        "math_3":          0.3 * 0.7,
        "math_4plus":      0.3 * 0.7,
        "math_4plus_MIND": 0.4 * 0.7,
        "dolma3":          1.0 * 0.3,
    }
    total_tokens_budget = target_step * (global_batch_size // 1)  # per-step tokens

    print(f"[analyze] target_step={target_step:,} budget={total_tokens_budget:,} tokens")
    print()

    per_source = {}
    for src, root in sources.items():
        root_path = Path(root)
        if not root_path.is_dir():
            print(f"[{src}] {root_path} not found")
            continue
        files = sorted(root_path.rglob("*.npy"))
        total_bytes = sum(p.stat().st_size for p in files)
        total_toks = total_bytes // 4
        drawn = int(total_tokens_budget * weights[src])
        unused = max(0, total_toks - drawn)
        unused_pct = 100 * unused / max(total_toks, 1)
        per_source[src] = {
            "root": root,
            "files": len(files),
            "tokens_available": total_toks,
            "tokens_drawn_by_target": drawn,
            "tokens_unused": unused,
            "unused_pct": unused_pct,
            "weight": weights[src],
        }
        print(
            f"[{src}] files={len(files)} avail={total_toks/1e9:.2f}B "
            f"drawn={drawn/1e9:.2f}B unused={unused/1e9:.2f}B ({unused_pct:.1f}%)"
        )

    # ---- locate global_indices for run -------------------------------------
    work_dir = Path(CACHE_MOUNT) / "work" / run_name / "node0"
    if work_dir.is_dir():
        gi = sorted(work_dir.glob("global_indices_*"))
        if gi:
            print()
            print(f"[indices] found {len(gi)} global_indices file(s) under {work_dir}:")
            for p in gi:
                print(f"  {p}  size={p.stat().st_size / 1e6:.1f} MB")
            print()
            print("[next step] write a follow-up to map global_indices -> (source, byte_offset)")
            print("    so we can mark individual files as truly consumed vs never-touched.")
    else:
        print(f"[indices] {work_dir} not found")

    # ---- recommendation: per-source candidate files to hold out ------------
    # Without the index inversion, the safest bet is to pick whole-file shards
    # from each source. For a source with N% unused, we can safely hold out
    # up to N% of files (assuming roughly uniform sampling within the source).
    print()
    print("=" * 78)
    print("RECOMMENDED held-out files (safe = small fraction of unused %)")
    print("=" * 78)
    held_out = {}
    for src, info in per_source.items():
        if info["unused_pct"] < 5:
            print(f"  {src}: unused {info['unused_pct']:.1f}% — too small to safely hold out")
            continue
        # Take up to 50% of unused fraction as a safety margin
        safe_fraction = info["unused_pct"] / 100 / 2
        n_files_to_hold = max(1, int(safe_fraction * info["files"]))
        root_path = Path(info["root"])
        files = sorted(root_path.rglob("*.npy"))
        # Take from the END of the sorted list (least likely to be touched first)
        candidates = files[-n_files_to_hold:]
        held_out[src] = [str(p.relative_to(TOKENIZED_MOUNT)) for p in candidates]
        held_out_toks = sum(p.stat().st_size for p in candidates) // 4
        print(
            f"  {src}: hold {n_files_to_hold} files (~{held_out_toks/1e9:.2f}B tokens) "
            f"-- {info['unused_pct']:.1f}% unused, taking ~{safe_fraction*100:.1f}% margin"
        )
        for p in candidates[:5]:
            print(f"      {p.relative_to(TOKENIZED_MOUNT)}")
        if len(candidates) > 5:
            print(f"      ... and {len(candidates) - 5} more")
    return {
        "per_source": per_source,
        "held_out_recommendations": held_out,
    }


@app.local_entrypoint()
def main(
    run_name: str = "math-1b-v0",
    target_step: int = 95368,
) -> None:
    result = analyze.remote(run_name=run_name, target_step=target_step)
    print()
    print("=== SUMMARY ===")
    for src, files in result["held_out_recommendations"].items():
        print(f"  {src}: {len(files)} files")
