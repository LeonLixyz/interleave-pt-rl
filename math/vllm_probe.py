"""Probe: try vLLM 0.9.2 + transformers 4.52 (verl-compatible). No patches."""

from __future__ import annotations

import modal
from common import CACHE_MOUNT, CACHE_VOLUME_NAME, hf_image_base


def _img() -> modal.Image:
    return (
        modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
        .apt_install("build-essential", "git")
        .pip_install("wheel", "packaging", "ninja")
        .pip_install("torch==2.7.0")
        .pip_install("vllm==0.9.2", "transformers==4.52.4", "huggingface_hub>=0.26")
        .add_local_python_source("common")
    )


app = modal.App("vllm-probe-v2", image=_img())
cache_volume = modal.Volume.from_name(CACHE_VOLUME_NAME, create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")


@app.function(gpu="H200:1", volumes={CACHE_MOUNT: cache_volume}, secrets=[hf_secret], timeout=1800)
def probe(repo: str) -> dict:
    import os
    os.environ["HF_HOME"] = f"{CACHE_MOUNT}/hf"
    cache_volume.reload()
    from vllm import LLM, SamplingParams

    try:
        llm = LLM(model=repo, dtype="bfloat16", max_model_len=1024,
                  gpu_memory_utilization=0.5, trust_remote_code=True)
        out = llm.generate(["What is 2+2?"], SamplingParams(max_tokens=20, temperature=0))
        return {"repo": repo, "ok": True, "sample": out[0].outputs[0].text[:120]}
    except Exception as e:
        return {"repo": repo, "ok": False, "error": f"{type(e).__name__}: {str(e)[:400]}"}


@app.local_entrypoint()
def main() -> None:
    import json
    repos = [
        "pre-to-post-olmo/math-1b-sft-openthoughts-from-step95368",
        "pre-to-post-olmo/math-1b-anneal-from-step95368",
    ]
    for r in repos:
        try:
            print(json.dumps(probe.remote(repo=r), indent=2))
        except Exception as e:
            print(f"[{r}] outer error: {e}")
