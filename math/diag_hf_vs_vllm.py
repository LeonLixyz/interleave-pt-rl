"""Diagnostic: HF vs vLLM on the SAME prompt, same weights, greedy.

Steps:
  1. Tokenize identical messages via HF and vLLM. Compare input token IDs.
  2. Run HF greedy generation for 32 tokens. Print output ids + text.
  3. Run vLLM greedy generation for 32 tokens. Print output ids + text.
  4. Diff — find the first token that differs (if any).

If input IDs differ  → chat template application differs.
If input IDs match but outputs diverge at token 0 → attention math differs.
If they match for a while and diverge later → cumulative sampling difference.
"""

from __future__ import annotations

import modal
from common import CACHE_MOUNT, CACHE_VOLUME_NAME, hf_image_base


def _img() -> modal.Image:
    return (
        modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
        .apt_install("build-essential", "git")
        .pip_install("wheel", "packaging", "ninja", "setuptools")
        .pip_install("torch==2.7.0")
        .pip_install("flash-attn==2.7.4.post1", extra_options="--no-build-isolation")
        .pip_install(
            "vllm==0.9.2",
            "transformers==4.52.4",
            "accelerate>=1.0",
            "huggingface_hub>=0.26",
        )
        .env({"VLLM_ATTENTION_BACKEND": "FLASH_ATTN"})
        .add_local_python_source("common")
    )


app = modal.App("diag-hf-vs-vllm", image=_img())
cache_volume = modal.Volume.from_name(CACHE_VOLUME_NAME, create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")


@app.function(gpu="H200:1", volumes={CACHE_MOUNT: cache_volume}, secrets=[hf_secret], timeout=1800)
def diag(
    hf_repo: str = "pre-to-post-olmo/math-1b-sft-openthoughts-from-step95368",
    max_new: int = 8192,
) -> dict:
    import os
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from vllm import LLM, SamplingParams

    os.environ["HF_HOME"] = f"{CACHE_MOUNT}/hf"
    cache_volume.reload()

    SYSTEM_PROMPT = (
        "You are a helpful assistant. When answering math problems, first think "
        "step by step inside <think>...</think> tags, then give your final "
        "answer in \\boxed{...}."
    )
    QUESTION = "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast every morning and bakes muffins for her friends every day with 4. She sells the remainder at the farmers' market daily for $2 per egg. How much in dollars does she make every day at the farmers' market?"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": QUESTION},
    ]

    tok = AutoTokenizer.from_pretrained(hf_repo, use_fast=True, trust_remote_code=True)
    print(f"[tok] class={type(tok).__name__}")

    prompt_str = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print(f"[prompt] len={len(prompt_str)} chars, first 300:\n{prompt_str[:300]}\n---")

    # ---- Reference input IDs via HF ----
    hf_input_ids = tok(prompt_str, return_tensors="pt", add_special_tokens=False).input_ids
    hf_ids = hf_input_ids[0].tolist()
    print(f"[hf tokenize] {len(hf_ids)} tokens; first 20 = {hf_ids[:20]}")

    # ---- HF greedy generation ----
    model = AutoModelForCausalLM.from_pretrained(
        hf_repo, torch_dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True, attn_implementation="flash_attention_2",
    ).eval()
    with torch.inference_mode():
        gen = model.generate(
            hf_input_ids.to("cuda"),
            max_new_tokens=max_new, do_sample=False,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
    hf_gen_ids = gen[0, len(hf_ids):].tolist()
    hf_gen_text = tok.decode(hf_gen_ids, skip_special_tokens=False)
    print(f"[hf gen] {len(hf_gen_ids)} tokens; ids = {hf_gen_ids}")
    print(f"[hf gen] text = {hf_gen_text!r}")

    del model
    torch.cuda.empty_cache()

    # ---- vLLM greedy generation via chat API ----
    llm = LLM(
        model=hf_repo, dtype="bfloat16", max_model_len=9216,
        gpu_memory_utilization=0.8, trust_remote_code=True,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=max_new)
    # via llm.chat
    out_chat = llm.chat([messages], sp, add_generation_prompt=True)
    vllm_chat_ids = list(out_chat[0].outputs[0].token_ids)
    vllm_chat_text = tok.decode(vllm_chat_ids, skip_special_tokens=False)
    vllm_chat_input_ids = list(out_chat[0].prompt_token_ids)
    print(f"[vllm chat input] {len(vllm_chat_input_ids)} tokens; first 20 = {vllm_chat_input_ids[:20]}")
    print(f"[vllm chat gen] {len(vllm_chat_ids)} tokens; ids = {vllm_chat_ids}")
    print(f"[vllm chat gen] text = {vllm_chat_text!r}")

    # via llm.generate(prompt_str) — same prompt string HF used
    out_gen = llm.generate([prompt_str], sp)
    vllm_gen_ids = list(out_gen[0].outputs[0].token_ids)
    vllm_gen_text = tok.decode(vllm_gen_ids, skip_special_tokens=False)
    vllm_gen_input_ids = list(out_gen[0].prompt_token_ids)
    print(f"[vllm gen input] {len(vllm_gen_input_ids)} tokens; first 20 = {vllm_gen_input_ids[:20]}")
    print(f"[vllm gen out] {len(vllm_gen_ids)} tokens; ids = {vllm_gen_ids}")
    print(f"[vllm gen out] text = {vllm_gen_text!r}")

    # ---- Diff ----
    print("\n=== DIFF ===")
    print(f"HF input ids == vLLM chat input ids? {hf_ids == vllm_chat_input_ids}")
    print(f"HF input ids == vLLM gen input ids? {hf_ids == vllm_gen_input_ids}")
    if hf_ids != vllm_chat_input_ids:
        # find first diff
        for i, (a, b) in enumerate(zip(hf_ids, vllm_chat_input_ids)):
            if a != b:
                print(f"  chat input diverges at pos {i}: hf={a} ({tok.decode([a])!r}) vllm={b} ({tok.decode([b])!r})")
                break
        if len(hf_ids) != len(vllm_chat_input_ids):
            print(f"  chat input length differs: hf={len(hf_ids)} vllm={len(vllm_chat_input_ids)}")
    print()
    print(f"HF gen == vLLM chat gen? {hf_gen_ids == vllm_chat_ids}")
    print(f"HF gen == vLLM gen gen? {hf_gen_ids == vllm_gen_ids}")
    if hf_gen_ids != vllm_gen_ids:
        for i, (a, b) in enumerate(zip(hf_gen_ids, vllm_gen_ids)):
            if a != b:
                print(f"  gen output diverges at pos {i}: hf={a} ({tok.decode([a])!r}) vllm={b} ({tok.decode([b])!r})")
                break

    return {
        "hf_input_ids_first20": hf_ids[:20],
        "vllm_chat_input_ids_first20": vllm_chat_input_ids[:20],
        "vllm_gen_input_ids_first20": vllm_gen_input_ids[:20],
        "hf_gen_ids": hf_gen_ids,
        "vllm_chat_gen_ids": vllm_chat_ids,
        "vllm_gen_gen_ids": vllm_gen_ids,
        "hf_gen_text": hf_gen_text,
        "vllm_chat_gen_text": vllm_chat_text,
        "vllm_gen_gen_text": vllm_gen_text,
    }


@app.local_entrypoint()
def main(max_new: int = 512) -> None:
    import json
    print(json.dumps(diag.remote(max_new=max_new), indent=2)[:3000])
