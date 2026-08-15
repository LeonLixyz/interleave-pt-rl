"""Download pre-to-post-olmo/rl-math-skyeasy25k-omi2 and save train/val
as parquet on the Modal Volume (verl expects parquet input).
"""

from __future__ import annotations

import modal
from common import CACHE_MOUNT, CACHE_VOLUME_NAME, CHECKPOINT_MOUNT, CHECKPOINT_VOLUME_NAME, hf_image_base


def _img() -> modal.Image:
    return hf_image_base().pip_install(
        "datasets==4.0.0", "pyarrow==17.0.0", "pandas==2.2.3",
    ).add_local_python_source("common")


app = modal.App("rl-preprocess", image=_img())
cache_volume = modal.Volume.from_name(CACHE_VOLUME_NAME, create_if_missing=True)
checkpoint_volume = modal.Volume.from_name(CHECKPOINT_VOLUME_NAME, create_if_missing=True, version=2)
hf_secret = modal.Secret.from_name("huggingface-secret")


@app.function(
    volumes={CACHE_MOUNT: cache_volume, CHECKPOINT_MOUNT: checkpoint_volume},
    secrets=[hf_secret],
    timeout=600,
    cpu=4,
)
def preprocess(
    dataset: str = "pre-to-post-olmo/rl-math-skyeasy25k-omi2",
    out_dir: str = f"{CHECKPOINT_MOUNT}/rl_data/skyeasy25k_omi2",
) -> dict:
    import os
    from pathlib import Path
    from datasets import load_dataset

    os.environ["HF_HOME"] = f"{CACHE_MOUNT}/hf"
    cache_volume.reload()
    checkpoint_volume.reload()
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    ds = load_dataset(dataset, download_mode="reuse_cache_if_exists")
    print(f"[ds] splits: {list(ds.keys())}")

    out = {}
    for split_name, split_ds in ds.items():
        # Save as parquet
        outfile = f"{out_dir}/{split_name}.parquet"
        split_ds.to_parquet(outfile)
        n = len(split_ds)
        srcs = {}
        for row in split_ds:
            src = row.get("data_source", "?")
            srcs[src] = srcs.get(src, 0) + 1
        out[split_name] = {
            "n_rows": n,
            "parquet": outfile,
            "data_sources": dict(sorted(srcs.items(), key=lambda x: -x[1])),
        }
        print(f"[{split_name}] n={n} -> {outfile}")
        print(f"  data_sources: {out[split_name]['data_sources']}")

    checkpoint_volume.commit()
    return out


@app.local_entrypoint()
def main() -> None:
    import json
    print(json.dumps(preprocess.remote(), indent=2))
