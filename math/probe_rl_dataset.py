"""Measure prompt length distribution of pre-to-post-olmo/rl-math-skyeasy25k-omi2.

Uses our SFT model's tokenizer (dolma2, 100k vocab) to get accurate token counts.
"""
from __future__ import annotations

import modal
from common import CACHE_MOUNT, CACHE_VOLUME_NAME, hf_image_base


def _img() -> modal.Image:
    return (
        hf_image_base()
        .pip_install(
            "transformers==4.52.4",
            "datasets>=3.0",
            "numpy>=1.26",
        )
        .add_local_python_source("common")
    )


app = modal.App("probe-rl-dataset", image=_img())
cache_volume = modal.Volume.from_name(CACHE_VOLUME_NAME, create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")


@app.function(volumes={CACHE_MOUNT: cache_volume}, secrets=[hf_secret], timeout=600, cpu=4)
def probe(
    dataset: str = "pre-to-post-olmo/rl-math-skyeasy25k-omi2",
    tokenizer_repo: str = "pre-to-post-olmo/math-1b-sft-numinamath-bs512-from-step95368",
) -> dict:
    import os
    import numpy as np
    from datasets import load_dataset
    from transformers import AutoTokenizer
    os.environ["HF_HOME"] = f"{CACHE_MOUNT}/hf"
    cache_volume.reload()

    tok = AutoTokenizer.from_pretrained(tokenizer_repo, use_fast=True, trust_remote_code=True)
    print(f"[tok] {type(tok).__name__}")

    ds = load_dataset(dataset)
    print(f"[ds] splits: {list(ds.keys())}")
    for split in ds:
        n = len(ds[split])
        print(f"  {split}: {n} rows, cols={list(ds[split].column_names)}")

    def _prompt_to_str(p):
        if isinstance(p, list):
            return " ".join(str(m.get("content", "")) for m in p if isinstance(m, dict))
        return str(p)

    out = {}
    for split in ds:
        d = ds[split]
        # Value counts by data_source
        ds_counter = {}
        for row in d:
            src = row.get("data_source", "?")
            ds_counter[src] = ds_counter.get(src, 0) + 1

        # Tokenize a random sample for length distribution
        sample_size = min(2000, len(d))
        idxs = list(range(len(d)))
        if sample_size < len(d):
            # deterministic stride
            step = max(1, len(d) // sample_size)
            idxs = list(range(0, len(d), step))[:sample_size]
        prompts = []
        for i in idxs:
            row = d[int(i)]
            p = _prompt_to_str(row.get("prompt"))
            # If chat template application changes length, apply it. Since verl uses
            # apply_chat_template(user_message=prompt), simulate that.
            messages = [{"role": "user", "content": p}]
            try:
                s = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                s = p
            prompts.append(s)
        toks = tok(prompts, add_special_tokens=False)["input_ids"]
        lens = np.array([len(t) for t in toks])
        stats = {
            "n_sample": len(lens),
            "min": int(lens.min()),
            "max": int(lens.max()),
            "mean": float(lens.mean()),
            "median": int(np.median(lens)),
            "p90": int(np.percentile(lens, 90)),
            "p95": int(np.percentile(lens, 95)),
            "p99": int(np.percentile(lens, 99)),
            "p999": int(np.percentile(lens, 99.9)),
            "gte_512": int((lens >= 512).sum()),
            "gte_1024": int((lens >= 1024).sum()),
            "gte_2048": int((lens >= 2048).sum()),
        }
        out[split] = {
            "n_rows": len(d),
            "data_source_counts": dict(sorted(ds_counter.items(), key=lambda x: -x[1])),
            "prompt_len_tokens": stats,
        }
        print(f"\n[{split}]")
        print(f"  data_sources: {out[split]['data_source_counts']}")
        print(f"  prompt lens (n={sample_size}): {stats}")

    return out


@app.local_entrypoint()
def main() -> None:
    import json
    print(json.dumps(probe.remote(), indent=2))
