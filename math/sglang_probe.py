"""Probe SGLang: same eval methodology as HF/vLLM, on step95368 SFT.

Full GSM8K (1319) greedy pass@1. Compare accuracy against HF (~20.6%) and
vLLM (~4.5%). If SGLang matches HF → confirms vLLM has a quirk in its
inference path. If SGLang matches vLLM → the "20%" HF number was somehow
lucky-truncation-artifact.
"""

from __future__ import annotations

import modal
from common import CACHE_MOUNT, CACHE_VOLUME_NAME, hf_image_base


def _img() -> modal.Image:
    return (
        modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
        .apt_install("build-essential", "git")
        .pip_install("wheel", "packaging", "ninja")
        .pip_install("torch==2.7.0")
        .pip_install(
            "sglang[all]==0.4.9",
            "transformers==4.52.4",
            "datasets>=3.0",
            "math-verify>=0.5",
            "huggingface_hub>=0.26",
        )
        .add_local_python_source("common")
    )


app = modal.App("sglang-probe", image=_img())
cache_volume = modal.Volume.from_name(CACHE_VOLUME_NAME, create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")


SYSTEM_PROMPT = (
    "You are a helpful assistant. When answering math problems, first think "
    "step by step inside <think>...</think> tags, then give your final "
    "answer in \\boxed{...}."
)


@app.function(
    gpu="H200:1", volumes={CACHE_MOUNT: cache_volume}, secrets=[hf_secret],
    timeout=60 * 60 * 3,
)
def sglang_eval(
    hf_repo: str = "pre-to-post-olmo/math-1b-sft-openthoughts-from-step95368",
    max_new_tokens: int = 8192,
    gsm8k_limit: int = 1319,
) -> dict:
    import os
    import re
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from math_verify import parse, verify, ExprExtractionConfig, LatexExtractionConfig
    import sglang as sgl

    os.environ["HF_HOME"] = f"{CACHE_MOUNT}/hf"
    cache_volume.reload()

    tok = AutoTokenizer.from_pretrained(hf_repo, use_fast=True, trust_remote_code=True)
    print(f"[tok] class={type(tok).__name__}")

    # SGLang embedded runtime.
    engine = sgl.Engine(
        model_path=hf_repo,
        dtype="bfloat16",
        mem_fraction_static=0.85,
        context_length=8192,
        trust_remote_code=True,
    )

    def _prompt(q):
        return tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": q}],
            tokenize=False, add_generation_prompt=True,
        )

    def _grade(pred, gold):
        after = pred.split("</think>")[-1] if "</think>" in pred else pred
        boxed = re.findall(r"\\boxed\{([^}]*)\}", after)
        ans = boxed[-1] if boxed else (after.strip().splitlines()[-1] if after.strip() else "")
        try:
            return bool(verify(
                parse(str(gold), extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()]),
                parse(f"\\boxed{{{ans}}}", extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()]),
            ))
        except Exception:
            return False

    gsm = load_dataset("openai/gsm8k", "main", split="test").select(range(gsm8k_limit))
    prompts = [_prompt(row["question"]) for row in gsm]
    golds = [row["answer"].split("####")[-1].strip() for row in gsm]

    sampling_params = {"temperature": 0.0, "max_new_tokens": max_new_tokens}
    print(f"[sglang] running {len(prompts)} prompts, max_new={max_new_tokens}")
    outputs = engine.generate(prompts, sampling_params)
    texts = [o["text"] for o in outputs]

    correct = sum(1 for t, g in zip(texts, golds) if _grade(t, g))
    acc = correct / len(prompts)
    print(f"[sglang] {correct}/{len(prompts)} = {acc:.4f}")

    # Sample output
    print(f"\n[sample] first output first 400 chars:\n{texts[0][:400]}")

    return {"hf_repo": hf_repo, "max_new_tokens": max_new_tokens, "n": len(prompts),
            "correct": correct, "accuracy": acc}


@app.local_entrypoint()
def main(max_new: int = 8192, gsm_limit: int = 1319) -> None:
    r = sglang_eval.remote(max_new_tokens=max_new, gsm8k_limit=gsm_limit)
    import json
    print(json.dumps(r, indent=2))
