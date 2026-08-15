"""Smoke-test and upload the consolidated export_bundle tree to
`pre-to-post-olmo/Math-Models`.

Usage:
    modal run upload_bundle.py::smoke_test          # load 3 leaves + generate
    modal run --detach upload_bundle.py::upload      # upload the whole tree
"""

from __future__ import annotations

import modal

from common import CHECKPOINT_MOUNT, CHECKPOINT_VOLUME_NAME

BUNDLE = f"{CHECKPOINT_MOUNT}/export_bundle"
REPO_ID = "pre-to-post-olmo/Math-Models"
LOCAL_ASSETS = "/Users/leonli66/Desktop/Research/RL/Chess RL/math-pretraining/bundle_assets"
REMOTE_ASSETS = "/root/bundle_assets"


def _img() -> modal.Image:
    return (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install("torch==2.6.0", "transformers>=4.46", "safetensors>=0.4",
                     "accelerate>=1.0", "huggingface_hub>=0.26", "hf_transfer>=0.1.8")
        .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
        .add_local_dir(LOCAL_ASSETS, remote_path=REMOTE_ASSETS, copy=True)
        .add_local_python_source("common")
    )


app = modal.App("upload-bundle", image=_img())
checkpoint_volume = modal.Volume.from_name(CHECKPOINT_VOLUME_NAME, create_if_missing=True, version=2)
hf_secret = modal.Secret.from_name("huggingface-secret")


@app.function(volumes={CHECKPOINT_MOUNT: checkpoint_volume}, timeout=60 * 20,
              cpu=8.0, memory=64 * 1024)
def smoke_test() -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    checkpoint_volume.reload()
    leaves = ["step95368/rl/step3000", "step20000/sft", "step5000/pretrain"]
    results = {}
    for leaf in leaves:
        path = f"{BUNDLE}/{leaf}"
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16)
        model.eval()
        prompt = "What is 12 * 13?"
        if tok.chat_template and ("sft" in leaf or "rl" in leaf):
            text = tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True)
        else:
            text = prompt
        ins = tok(text, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**ins, max_new_tokens=64, do_sample=False)
        gen = tok.decode(out[0][ins["input_ids"].shape[1]:], skip_special_tokens=True)
        results[leaf] = {
            "params_B": round(sum(p.numel() for p in model.parameters()) / 1e9, 3),
            "dtype": str(next(model.parameters()).dtype),
            "sample": gen[:200],
        }
        print(f"[smoke] {leaf}: {results[leaf]}", flush=True)
        del model
    return results


@app.function(volumes={CHECKPOINT_MOUNT: checkpoint_volume}, secrets=[hf_secret],
              timeout=60 * 60 * 12, cpu=8.0, memory=64 * 1024)
def upload(private: bool = True) -> dict:
    import os
    import shutil
    from pathlib import Path

    from huggingface_hub import HfApi, create_repo

    checkpoint_volume.reload()
    token = os.environ["HF_TOKEN"]
    create_repo(REPO_ID, token=token, exist_ok=True, repo_type="model", private=private)

    # place README + generate.py at the tree root
    for fn in ("README.md", "generate.py"):
        shutil.copy(f"{REMOTE_ASSETS}/{fn}", f"{BUNDLE}/{fn}")

    n = sum(1 for _ in Path(BUNDLE).rglob("*") if _.is_file())
    gb = sum(p.stat().st_size for p in Path(BUNDLE).rglob("*") if p.is_file()) / 1e9
    print(f"[upload] {n} files, {gb:.1f} GB -> {REPO_ID} (private={private})", flush=True)

    api = HfApi(token=token)
    api.upload_large_folder(repo_id=REPO_ID, folder_path=BUNDLE, repo_type="model")
    return {"repo": REPO_ID, "files": n, "gb": round(gb, 1), "private": private}
