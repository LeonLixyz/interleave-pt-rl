"""Extract the TRULY UNTRAINED token instances from each source.

This goes beyond extract_heldout.py (which only finds whole-file held-outs).
For each leaf source (math_3 / math_4plus / math_4plus_MIND / dolma3):

  1. Load the SamplingInstanceSource indices file (uint32 list of sampled
     instance indices written by stable's data loader).
  2. Compute sampled_set = unique(indices).
  3. unsampled_set = range(total_instances) - sampled_set.
  4. For each unsampled instance index, map → (source_file, position_within_file)
     using the leaf's cumulative file boundaries.
  5. Read the exact 4096-token chunk from the source .npy and append to a new
     output file: /untrained/<label>/part_NNNN.npy (raw uint32).

Output corpus has zero overlap with stable training, by construction. Suitable
for anneal training data AND held-out eval, partitioned by the consumer.

Usage:
    modal run extract_untrained.py
"""

from __future__ import annotations

import json

import modal

from common import (
    CACHE_MOUNT,
    CACHE_VOLUME_NAME,
    TOKENIZED_MOUNT,
    UNTRAINED_MOUNT,
    UNTRAINED_VOLUME_NAME,
    hf_image_base,
    tokenized_volume,
    untrained_volume,
)


def _img() -> modal.Image:
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


app = modal.App("extract-untrained", image=_img())
cache_volume = modal.Volume.from_name(CACHE_VOLUME_NAME, create_if_missing=True)


@app.function(
    volumes={
        CACHE_MOUNT: cache_volume,
        TOKENIZED_MOUNT: tokenized_volume,
        UNTRAINED_MOUNT: untrained_volume,
    },
    timeout=60 * 60 * 6,
    cpu=16.0,
    memory=128 * 1024,
)
def extract(
    run_name: str = "math-1b-v0",
    total_tokens: int = 200_000_000_000,
    sequence_length: int = 4096,
    chunk_instances: int = 250_000,  # ~4 GB per output file
) -> dict:
    """For each leaf source, write a new .npy corpus containing only untrained instances."""
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
    untrained_volume.reload()

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

    def chunk_cfg(label: str) -> ConcatAndChunkInstanceSource.Config:
        return ConcatAndChunkInstanceSource.Config(
            sources=[token_sources[label]],
            label=label,
            sequence_length=sequence_length,
        )

    math_total_w = (
        DEFAULT_WEIGHTS["math_3"]
        + DEFAULT_WEIGHTS["math_4plus"]
        + DEFAULT_WEIGHTS["math_4plus_MIND"]
    )
    math_sub_cfg = MixingInstanceSource.Config(
        source_specs=[
            MixingInstanceSource.Spec.Config(
                source=chunk_cfg("math_3"),
                ratio=DEFAULT_WEIGHTS["math_3"] / math_total_w,
                label="math_3",
            ),
            MixingInstanceSource.Spec.Config(
                source=chunk_cfg("math_4plus"),
                ratio=DEFAULT_WEIGHTS["math_4plus"] / math_total_w,
                label="math_4plus",
            ),
            MixingInstanceSource.Spec.Config(
                source=chunk_cfg("math_4plus_MIND"),
                ratio=DEFAULT_WEIGHTS["math_4plus_MIND"] / math_total_w,
                label="math_4plus_MIND",
            ),
        ],
    )
    mix_cfg = MixingInstanceSource.Config(
        source_specs=[
            MixingInstanceSource.Spec.Config(source=math_sub_cfg, ratio=math_total_w, label="math"),
            MixingInstanceSource.Spec.Config(
                source=chunk_cfg("dolma3"),
                ratio=DEFAULT_WEIGHTS["dolma3"],
                label="dolma3",
            ),
        ],
        num_tokens=total_tokens,
    )

    work_dir = f"{CACHE_MOUNT}/work/{run_name}/node0"
    print(f"[mix] building mix at {work_dir} ...")
    mix = mix_cfg.build(work_dir)

    summary: dict = {}

    def process_leaf(leaf, idx_path: str, label: str) -> None:
        print(f"\n=== {label} ===")
        underlying = leaf._sources
        per_file = []  # ordered list aligned to ConcatAndChunkInstanceSource's stream
        cum = 0
        for src in underlying:
            paths = src.source_paths
            sizes = src.source_sizes  # in tokens
            for p, sz in zip(paths, sizes):
                n_inst = sz // sequence_length
                per_file.append({"path": p, "n_instances": n_inst, "cum_start": cum})
                cum += n_inst
        total_instances = cum
        if total_instances == 0:
            print(f"  {label}: no instances. skipping.")
            return
        cum_ends = np.array(
            [f["cum_start"] + f["n_instances"] for f in per_file], dtype=np.int64
        )

        idx_arr = np.memmap(idx_path, dtype=np.uint32, mode="r")
        unique_sampled = np.unique(idx_arr)
        all_idx = np.arange(total_instances, dtype=np.uint32)
        unsampled = np.setdiff1d(all_idx, unique_sampled, assume_unique=False)
        print(
            f"  total={total_instances:,} sampled={len(unique_sampled):,} "
            f"untrained={len(unsampled):,} ({100*len(unsampled)/total_instances:.2f}%)"
        )

        if len(unsampled) == 0:
            summary[label] = {
                "total_instances": int(total_instances),
                "untrained_instances": 0,
                "untrained_tokens": 0,
                "files": [],
            }
            return

        out_dir = Path(UNTRAINED_MOUNT) / label
        out_dir.mkdir(parents=True, exist_ok=True)
        for old in out_dir.glob("part_*.npy"):
            old.unlink()

        # Map each unsampled idx to its containing file. unsampled is sorted ascending.
        file_idx_for_each = np.searchsorted(cum_ends, unsampled, side="right")

        chunk_id = 0
        buf = np.empty(chunk_instances * sequence_length, dtype=np.uint32)
        buf_count = 0
        total_written = 0
        out_paths = []

        current_file_idx = -1
        current_mm = None

        # Iterate in stream order so we keep memmaps for whole stretches of one file.
        for i in range(len(unsampled)):
            ui = int(unsampled[i])
            fi = int(file_idx_for_each[i])
            if fi != current_file_idx:
                current_file_idx = fi
                file_info = per_file[fi]
                current_mm = np.memmap(file_info["path"], dtype=np.uint32, mode="r")
            pos = ui - per_file[fi]["cum_start"]
            start = pos * sequence_length
            end = start + sequence_length
            buf[buf_count * sequence_length : (buf_count + 1) * sequence_length] = current_mm[
                start:end
            ]
            buf_count += 1

            if buf_count >= chunk_instances:
                out_path = out_dir / f"part_{chunk_id:04d}.npy"
                buf[: buf_count * sequence_length].tofile(str(out_path))
                print(
                    f"  wrote {out_path.name} "
                    f"({buf_count:,} inst, {buf_count*sequence_length/1e6:.1f}M tokens)"
                )
                out_paths.append(str(out_path))
                total_written += buf_count
                buf_count = 0
                chunk_id += 1

        if buf_count > 0:
            out_path = out_dir / f"part_{chunk_id:04d}.npy"
            buf[: buf_count * sequence_length].tofile(str(out_path))
            print(
                f"  wrote {out_path.name} "
                f"({buf_count:,} inst, {buf_count*sequence_length/1e6:.1f}M tokens)"
            )
            out_paths.append(str(out_path))
            total_written += buf_count

        summary[label] = {
            "total_instances": int(total_instances),
            "untrained_instances": int(total_written),
            "untrained_tokens": int(total_written * sequence_length),
            "files": out_paths,
        }
        untrained_volume.commit()  # commit incrementally per leaf

    def walk(node, path: str = "") -> None:
        cls_name = type(node).__name__
        if cls_name == "SamplingInstanceSource":
            sources = node._sources
            sample_paths = node._source_sample_paths
            for src, idx_path in zip(sources, sample_paths):
                sub_label = getattr(src, "label", None) or ""
                inner_cls = type(src).__name__
                if inner_cls == "ConcatAndChunkInstanceSource":
                    process_leaf(src, idx_path, label=sub_label)
                elif inner_cls == "MixingInstanceSource":
                    walk(src, path + "/" + (sub_label or inner_cls))
        elif cls_name == "MixingInstanceSource":
            for s in node.sampled_sources:
                walk(s, path + "/" + (getattr(s, "label", None) or type(s).__name__))

    walk(mix)
    untrained_volume.commit()

    # Save manifest
    manifest = {
        "run_name": run_name,
        "total_tokens_budget": total_tokens,
        "sequence_length": sequence_length,
        "per_source": summary,
    }
    manifest_path = Path(UNTRAINED_MOUNT) / "untrained_manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
    untrained_volume.commit()

    print("\n" + "=" * 78)
    print("UNTRAINED CORPUS EXTRACTED")
    print("=" * 78)
    total_tok = 0
    for label, info in summary.items():
        tok = info["untrained_tokens"]
        total_tok += tok
        print(
            f"  {label:20s}: {info['untrained_instances']:>12,} instances, "
            f"{tok/1e9:>6.2f}B tokens, {len(info['files'])} files"
        )
    print(f"  {'TOTAL':20s}: {'':>12s}            {total_tok/1e9:>6.2f}B tokens")
    print(f"\nmanifest: {manifest_path}")
    return manifest


@app.local_entrypoint()
def main(run_name: str = "math-1b-v0") -> None:
    r = extract.remote(run_name=run_name)
    print()
    print("=== RESULT ===")
    print(json.dumps(r, indent=2) if isinstance(r, dict) else r)
