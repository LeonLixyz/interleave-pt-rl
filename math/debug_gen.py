"""Debug: are we truncating? For 20 GSM8K prompts, print:
  - prompt length
  - generated length
  - whether generation ended at EOS or max_tokens
  - whether `</think>` and `\\boxed{...}` appear
  - the parsed answer + gold + grade
"""

from __future__ import annotations

import modal

from common import CACHE_MOUNT, CACHE_VOLUME_NAME, hf_image_base


def _img() -> modal.Image:
    return (
        modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
        .apt_install("build-essential", "git")
        .pip_install("wheel", "packaging", "ninja", "setuptools")
        .pip_install("torch==2.6.0")
        .pip_install(
            "transformers>=4.53",
            "tokenizers>=0.20.3",
            "accelerate>=1.0",
            "datasets>=3.0",
            "math-verify>=0.5",
            "huggingface_hub>=0.26",
        )
        .add_local_python_source("common")
    )


app = modal.App("debug-gen", image=_img())
cache_volume = modal.Volume.from_name(CACHE_VOLUME_NAME, create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")


@app.function(
    gpu="H200:1",
    volumes={CACHE_MOUNT: cache_volume},
    secrets=[hf_secret],
    timeout=1800,
)
def debug(
    hf_repo: str = "pre-to-post-olmo/math-1b-sft-openthoughts-from-step95368",
    n_samples: int = 20,
    max_new_tokens: int = 4096,
) -> None:
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
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True, attn_implementation="sdpa",
    ).eval()

    SYSTEM_PROMPT = (
        "You are a helpful assistant. When answering math problems, first think "
        "step by step inside <think>...</think> tags, then give your final "
        "answer in \\boxed{...}."
    )

    def make_prompt(q):
        return tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": q}],
            tokenize=False, add_generation_prompt=True,
        )

    gsm = load_dataset("openai/gsm8k", "main", split="test").select(range(n_samples))

    stats = {"has_end_think": 0, "has_boxed": 0, "hit_max": 0, "ended_eos": 0, "correct": 0}
    for i, row in enumerate(gsm):
        q = row["question"]
        gold_raw = row["answer"]
        gold = gold_raw.split("####")[-1].strip()

        p = make_prompt(q)
        enc = tokenizer(p, return_tensors="pt").to("cuda")
        prompt_len = enc["input_ids"].shape[1]
        with torch.inference_mode():
            gen = model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            )
        new_tokens = gen[0, prompt_len:]
        n_new = new_tokens.shape[0]
        text = tokenizer.decode(new_tokens, skip_special_tokens=False)
        text_clean = tokenizer.decode(new_tokens, skip_special_tokens=True)

        hit_max = n_new >= max_new_tokens
        # Check if last token is EOS
        eos_id = tokenizer.eos_token_id
        ended_eos = int(new_tokens[-1].item()) == eos_id if n_new > 0 else False
        has_end_think = "</think>" in text_clean
        boxed = re.findall(r"\\boxed\{([^}]*)\}", text_clean)
        has_boxed = bool(boxed)
        after_think = text_clean.split("</think>")[-1] if has_end_think else text_clean
        pred_ans = boxed[-1] if boxed else after_think.strip().splitlines()[-1] if after_think.strip() else ""
        try:
            correct = bool(verify(
                parse(str(gold), extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()]),
                parse(f"\\boxed{{{pred_ans}}}", extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()]),
            ))
        except Exception:
            correct = False

        stats["has_end_think"] += int(has_end_think)
        stats["has_boxed"] += int(has_boxed)
        stats["hit_max"] += int(hit_max)
        stats["ended_eos"] += int(ended_eos)
        stats["correct"] += int(correct)

        # Print detailed for first 5 + all failures
        show = (i < 5) or not correct
        if show:
            print(f"\n{'='*70}")
            print(f"[{i}] q={q[:100]}...")
            print(f"    prompt_len={prompt_len} gen_len={n_new} hit_max={hit_max} ended_eos={ended_eos}")
            print(f"    has_</think>={has_end_think} has_boxed={has_boxed} n_boxed={len(boxed)}")
            print(f"    gold={gold!r} pred_ans={pred_ans!r} correct={correct}")
            print(f"    --- last 400 chars of gen ---")
            print(text_clean[-400:])

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for k, v in stats.items():
        print(f"  {k}: {v}/{n_samples} ({100*v/n_samples:.1f}%)")


@app.local_entrypoint()
def main(
    hf_repo: str = "pre-to-post-olmo/math-1b-sft-openthoughts-from-step95368",
    n_samples: int = 20,
) -> None:
    debug.remote(hf_repo=hf_repo, n_samples=n_samples)
