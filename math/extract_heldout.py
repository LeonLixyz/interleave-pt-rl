"""Extract the TRULY HELD-OUT files — zero contamination.

How it works:
- OLMo-core's SamplingInstanceSource writes a `<fingerprint>-<source.fingerprint>-indices.npy`
  file for each underlying source it samples from. The contents are the explicit list of
  instance indices selected from that source for the 200B-token budget.
- We rebuild the same MixingInstanceSource tree locally (same code as train_inner_mix.py)
  to get the source tree + fingerprints.
- For each leaf ConcatAndChunkInstanceSource (one per math_3, math_4plus, math_4plus_MIND,
  dolma3), we:
    1. Find the indices file that matches its fingerprint
    2. Compute set(indices) — the unique instance indices that get sampled
    3. Map each instance index to (file_path, instance_idx_within_file) via cumulative
       instance counts
    4. Per file: count how many of its instances appear in set(indices)
    5. Files with count == 0 = TRULY HELD-OUT (zero contamination)
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
    # Include OLMo-core so we can build the same mix tree.
    return (
        hf_image_base()
        .pip_install("torch==2.6.0")
        .add_local_dir(
            "/Users/leonli66/Desktop/Research/RL/Chess RL/OLMo-core",
            remote_path="/root/OLMo-core",
            copy=True,
            ignore=[".git", ".git/**", ".venv/**", "__pycache__/**", "build/**"],
        )
        .run_commands("cd /root/OLMo-core && pip install -e .")
        .add_local_python_source("common")
    )


app = modal.App("extract-heldout", image=_img())
cache_volume = modal.Volume.from_name(CACHE_VOLUME_NAME, create_if_missing=True)


@app.function(
    volumes={CACHE_MOUNT: cache_volume, TOKENIZED_MOUNT: tokenized_volume},
    timeout=60 * 60,
    cpu=8.0,
    memory=32 * 1024,
)
def extract(
    run_name: str = "math-1b-v0",
    total_tokens: int = 200_000_000_000,
    sequence_length: int = 4096,
) -> dict:
    """Walk the SamplingInstanceSource indices files and identify per-file held-out status."""
    import json
    from pathlib import Path

    import numpy as np
    from olmo_core.data import TokenizerConfig
    from olmo_core.data.composable import (
        ConcatAndChunkInstanceSource,
        MixingInstanceSource,
        NumpyDocumentSource,
    )

    cache_volume.reload()
    tokenized_volume.reload()

    # ---- 1) Rebuild the same mix tree (matches train_inner_mix.py) -----------
    DEFAULT_PATHS = {
        "math_3": ["/tokenized/3/part_*.npy"],
        "math_4plus": ["/tokenized/4plus/part_*.npy"],
        "math_4plus_MIND": ["/tokenized/4plus_MIND/part_*.npy"],
        "dolma3": ["/tokenized/dolma3/**/*.npy"],
    }
    DEFAULT_WEIGHTS = {
        "math_3":          0.3 * 0.7,
        "math_4plus":      0.3 * 0.7,
        "math_4plus_MIND": 0.4 * 0.7,
        "dolma3":          1.0 * 0.3,
    }

    tokenizer = TokenizerConfig.dolma2()
    token_sources = NumpyDocumentSource.Config.from_source_groups(
        {label: DEFAULT_PATHS[label] for label in DEFAULT_PATHS},
        tokenizer=tokenizer,
    )

    def chunk(label):
        return ConcatAndChunkInstanceSource.Config(
            sources=[token_sources[label]],
            label=label,
            sequence_length=sequence_length,
        )

    math_total_w = DEFAULT_WEIGHTS["math_3"] + DEFAULT_WEIGHTS["math_4plus"] + DEFAULT_WEIGHTS["math_4plus_MIND"]
    math_sub_cfg = MixingInstanceSource.Config(
        source_specs=[
            MixingInstanceSource.Spec.Config(source=chunk("math_3"), ratio=DEFAULT_WEIGHTS["math_3"] / math_total_w, label="math_3"),
            MixingInstanceSource.Spec.Config(source=chunk("math_4plus"), ratio=DEFAULT_WEIGHTS["math_4plus"] / math_total_w, label="math_4plus"),
            MixingInstanceSource.Spec.Config(source=chunk("math_4plus_MIND"), ratio=DEFAULT_WEIGHTS["math_4plus_MIND"] / math_total_w, label="math_4plus_MIND"),
        ],
    )
    mix_cfg = MixingInstanceSource.Config(
        source_specs=[
            MixingInstanceSource.Spec.Config(source=math_sub_cfg, ratio=math_total_w, label="math"),
            MixingInstanceSource.Spec.Config(source=chunk("dolma3"), ratio=DEFAULT_WEIGHTS["dolma3"], label="dolma3"),
        ],
        num_tokens=total_tokens,
    )

    work_dir = f"{CACHE_MOUNT}/work/{run_name}/node0"
    print(f"[mix] building mix at {work_dir} ...")
    mix = mix_cfg.build(work_dir)
    print("[mix] built. Visualizing:")
    mix.visualize()

    # ---- 2) Walk the source tree, find leaves (ConcatAndChunkInstanceSource) ---
    # Each leaf is wrapped by a SamplingInstanceSource whose indices file is at
    # source._source_sample_paths.
    results = {}

    def walk(node, path=""):
        cls_name = type(node).__name__
        label = getattr(node, "label", None) or ""
        if cls_name == "SamplingInstanceSource":
            sources = node._sources
            sample_paths = node._source_sample_paths
            for src, idx_path in zip(sources, sample_paths):
                sub_label = getattr(src, "label", None) or ""
                inner_cls = type(src).__name__
                if inner_cls == "ConcatAndChunkInstanceSource":
                    process_leaf(src, idx_path, label=sub_label or label)
                elif inner_cls == "MixingInstanceSource":
                    walk(src, path + "/" + (sub_label or inner_cls))
                else:
                    print(f"  [walk] unknown inner type at {path}: {inner_cls}, label={sub_label}")
        elif cls_name == "MixingInstanceSource":
            # .sampled_sources returns Tuple[SamplingInstanceSource, ...]
            for s in node.sampled_sources:
                walk(s, path + "/" + (getattr(s, "label", None) or type(s).__name__))
        elif cls_name == "ConcatenatedInstanceSource":
            for s in node.sources:
                walk(s, path + "/" + (getattr(s, "label", None) or type(s).__name__))
        else:
            print(f"  [walk] unhandled type at {path}: {cls_name}, label={label}")

    def process_leaf(leaf, idx_path, label):
        print(f"\n=== leaf: {label} (ConcatAndChunkInstanceSource) ===")
        # The leaf wraps a NumpyDocumentSource (or list). Get the underlying file paths
        # and cumulative instance counts.
        underlying = leaf._sources  # list of token sources
        per_file = []  # list of (path, num_instances, cum_instances_after)
        cum = 0
        for src in underlying:
            # NumpyDocumentSource has source_paths + source_sizes
            paths = src.source_paths
            sizes = src.source_sizes  # in tokens, not instances
            for p, sz in zip(paths, sizes):
                n_inst = sz // sequence_length
                cum += n_inst
                per_file.append({"path": p, "n_instances": n_inst, "cum_end": cum})
        total_instances = cum

        print(f"  total files: {len(per_file)}")
        print(f"  total instances: {total_instances:,}")

        # Load sampled indices
        idx_arr = np.memmap(idx_path, dtype=np.uint32, mode="r")
        print(f"  sampled indices: {len(idx_arr):,} (uniq estimate via set sample)")
        unique_idx = np.unique(idx_arr)
        n_unique = len(unique_idx)
        print(f"  unique sampled indices: {n_unique:,} of {total_instances:,} ({100*n_unique/total_instances:.2f}%)")
        print(f"  truly UNSAMPLED instances: {total_instances - n_unique:,} ({100*(total_instances - n_unique)/total_instances:.2f}%)")

        # Per-file: count how many of its instances appear in unique_idx
        # We need to map each unique index → which file it belongs to.
        # Build cum_end array and use searchsorted.
        cum_ends = np.array([f["cum_end"] for f in per_file])  # sorted ascending
        # File index = searchsorted(cum_ends, idx, side='right') — gives index of first cum_end > idx
        file_idx_for_each_unique = np.searchsorted(cum_ends, unique_idx, side='right')
        # Per-file consumed count
        per_file_consumed = np.zeros(len(per_file), dtype=np.int64)
        np.add.at(per_file_consumed, file_idx_for_each_unique, 1)

        # Identify held-out files
        held_out = []
        partial = []
        full = []
        for i, f in enumerate(per_file):
            c = int(per_file_consumed[i])
            if c == 0:
                held_out.append(f["path"])
            elif c >= f["n_instances"]:
                full.append((f["path"], c, f["n_instances"]))
            else:
                partial.append((f["path"], c, f["n_instances"]))
        print(f"  -> HELD OUT (0 instances sampled): {len(held_out)} files")
        print(f"  -> PARTIAL (some sampled): {len(partial)} files")
        print(f"  -> FULL (all instances sampled): {len(full)} files")
        if held_out[:3]:
            print(f"  Sample held-out paths:")
            for h in held_out[:3]:
                print(f"    {h}")

        results[label] = {
            "total_files": len(per_file),
            "total_instances": int(total_instances),
            "unique_sampled": int(n_unique),
            "held_out_files": held_out,
            "n_partial": len(partial),
            "n_full": len(full),
        }

    walk(mix)

    # ---- 3) Save manifest ---------------------------------------------------
    manifest_path = Path(CACHE_MOUNT) / "work" / "heldout_manifest.json"
    manifest = {
        "run_name": run_name,
        "total_tokens": total_tokens,
        "sequence_length": sequence_length,
        "per_source": {
            label: {
                "total_files": info["total_files"],
                "total_instances": info["total_instances"],
                "unique_sampled": info["unique_sampled"],
                "n_held_out": len(info["held_out_files"]),
                "n_partial": info["n_partial"],
                "n_full": info["n_full"],
                "held_out_files": info["held_out_files"],
            }
            for label, info in results.items()
        },
    }
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
    cache_volume.commit()

    # Summary
    print("\n" + "=" * 78)
    print("HELD-OUT SUMMARY (zero contamination)")
    print("=" * 78)
    total_held = 0
    total_held_tokens = 0
    for label, info in results.items():
        n_held = len(info["held_out_files"])
        # Conservative token estimate: avg instance size * count
        held_tokens = n_held * (info["total_instances"] // info["total_files"]) * sequence_length if info["total_files"] else 0
        total_held += n_held
        total_held_tokens += held_tokens
        print(f"  {label:20s}: {n_held:>6} held-out files (~{held_tokens/1e9:.2f}B tokens)")
    print(f"  {'TOTAL':20s}: {total_held:>6} files (~{total_held_tokens/1e9:.2f}B tokens)")
    print(f"\nManifest saved to: {manifest_path}")

    return {"manifest": str(manifest_path), "per_source_summary": {
        label: len(info["held_out_files"]) for label, info in results.items()
    }}


@app.local_entrypoint()
def main(run_name: str = "math-1b-v0") -> None:
    r = extract.remote(run_name=run_name)
    print()
    print("=== RESULT ===")
    print(r)
