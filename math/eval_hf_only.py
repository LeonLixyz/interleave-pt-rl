"""HF-transformers-only eval (no vLLM). One model, one config. Used to fill
the max_new=8192 HF-transformers cell in the length-vs-framework diagnostic.
"""

from __future__ import annotations

import modal
from common import CACHE_MOUNT, CACHE_VOLUME_NAME, CHECKPOINT_MOUNT, CHECKPOINT_VOLUME_NAME, hf_image_base


def _img() -> modal.Image:
    return (
        modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
        .apt_install("build-essential", "git")
        .pip_install("wheel", "packaging", "ninja")
        .pip_install("torch==2.6.0")
        .pip_install(
            "transformers>=4.46,<4.48",
            "accelerate>=1.0",
            "datasets>=3.0",
            "math-verify>=0.5",
            "huggingface_hub>=0.26",
        )
        .add_local_python_source("common")
    )


app = modal.App("eval-hf-only", image=_img())
cache_volume = modal.Volume.from_name(CACHE_VOLUME_NAME, create_if_missing=True)
checkpoint_volume = modal.Volume.from_name(
    CHECKPOINT_VOLUME_NAME, create_if_missing=True, version=2
)
hf_secret = modal.Secret.from_name("huggingface-secret")


@app.function(
    gpu="H200:1",
    timeout=60 * 60 * 6,
    volumes={CACHE_MOUNT: cache_volume, CHECKPOINT_MOUNT: checkpoint_volume},
    secrets=[hf_secret],
)
def run(
    hf_repo: str,
    max_new_tokens: int,
    gsm8k_limit: int,
) -> dict:
    import os
    import re
    import torch
    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from math_verify import parse, verify, ExprExtractionConfig, LatexExtractionConfig

    os.environ["HF_HOME"] = f"{CACHE_MOUNT}/hf"
    cache_volume.reload()

    model_dir = snapshot_download(
        repo_id=hf_repo,
        force_download=False,
        allow_patterns=["*.json", "*.txt", "chat_template.jinja", "*.safetensors", "*.bin"],
    )
    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True, attn_implementation="sdpa",
    ).eval()

    SYSTEM_PROMPT = (
        "You are a helpful assistant. When answering math problems, first think "
        "step by step inside <think>...</think> tags, then give your final "
        "answer in \\boxed{...}."
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

    BATCH = 8
    correct = 0
    import time
    t0 = time.time()
    with torch.inference_mode():
        for i in range(0, len(prompts), BATCH):
            chunk = prompts[i : i + BATCH]
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=1024).to("cuda")
            gen = model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
            )
            in_len = enc["input_ids"].shape[1]
            for j in range(gen.shape[0]):
                text = tok.decode(gen[j, in_len:], skip_special_tokens=True)
                if _grade(text, golds[i + j]):
                    correct += 1
            if (i // BATCH) % 10 == 0:
                print(f"[{i + len(chunk)}/{len(prompts)}] correct={correct} elapsed={time.time()-t0:.0f}s")

    acc = correct / len(prompts)
    return {"hf_repo": hf_repo, "max_new_tokens": max_new_tokens, "n": len(prompts), "correct": correct, "accuracy": acc}


@app.local_entrypoint()
def main(
    hf_repo: str = "pre-to-post-olmo/math-1b-sft-openthoughts-from-step95368",
    max_new_tokens: int = 8192,
    gsm8k_limit: int = 1319,
) -> None:
    r = run.remote(hf_repo=hf_repo, max_new_tokens=max_new_tokens, gsm8k_limit=gsm8k_limit)
    import json
    print(json.dumps(r, indent=2))
