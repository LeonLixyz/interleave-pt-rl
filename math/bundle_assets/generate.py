#!/usr/bin/env python
"""Generate a rollout from any model in this repo.

Each model lives in a subfolder: step{anchor}/{pretrain,anneal,sft,rl/step{K}}.

Usage:
    python generate.py --model step20000/sft --prompt "What is 12 * 13?"
    python generate.py --model step95368/rl/step3000 --prompt "..." --max-new-tokens 1024

The models use the standard OLMo-2 architecture, so no `trust_remote_code` is
needed. If you haven't cloned the repo, snapshot just the subfolder you want:

    from huggingface_hub import snapshot_download
    snapshot_download("pre-to-post-olmo/Math-Models",
                      allow_patterns="step20000/sft/*", local_dir="Math-Models")
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="subfolder path, e.g. step20000/rl/step3000")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    # SFT/RL models are chat-tuned; use the chat template when present.
    if tok.chat_template:
        text = tok.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            tokenize=False, add_generation_prompt=True)
    else:
        text = args.prompt

    inputs = tok(text, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.temperature > 0,
        temperature=args.temperature if args.temperature > 0 else None,
    )
    print(tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))


if __name__ == "__main__":
    main()
