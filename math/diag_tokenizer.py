"""Diagnostic: what does AutoTokenizer.from_pretrained() actually return?

Tests our SFT repo, our anneal repo, and the official allenai/OLMo-2-1124-1B.
"""

from __future__ import annotations

import modal

from common import hf_image_base


def _img() -> modal.Image:
    return (
        hf_image_base()
        .pip_install(
            "torch==2.6.0",
            "transformers>=4.53",
            "tokenizers>=0.20.3",
            "huggingface_hub>=0.26",
        )
        .add_local_python_source("common")
    )


app = modal.App("diag-tokenizer", image=_img())
hf_secret = modal.Secret.from_name("huggingface-secret")


@app.function(secrets=[hf_secret], timeout=600, cpu=4)
def diag() -> dict:
    import json
    import os
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer, GPT2TokenizerFast

    out = {}
    repos = [
        "pre-to-post-olmo/math-1b-sft-openthoughts-from-step95368",
        "pre-to-post-olmo/math-1b-anneal-from-step95368",
        "allenai/OLMo-2-1124-1B",
    ]
    for r in repos:
        entry: dict = {"repo": r}
        try:
            path = snapshot_download(
                repo_id=r,
                token=os.environ["HF_TOKEN"],
                force_download=True,
                allow_patterns=["*.json", "*.txt", "chat_template.jinja"],
            )
            entry["snapshot"] = path
            # tokenizer_config content
            tcfg = json.loads(open(f"{path}/tokenizer_config.json").read())
            entry["tokenizer_class_from_config"] = tcfg.get("tokenizer_class")

            # Try AutoTokenizer default
            try:
                tok = AutoTokenizer.from_pretrained(path, use_fast=True, trust_remote_code=True)
                entry["auto_use_fast_true"] = type(tok).__name__
            except Exception as e:
                entry["auto_use_fast_true_error"] = f"{type(e).__name__}: {e}"

            # Try explicit GPT2TokenizerFast
            try:
                tok2 = GPT2TokenizerFast.from_pretrained(path)
                entry["gpt2_fast_direct"] = type(tok2).__name__
            except Exception as e:
                entry["gpt2_fast_direct_error"] = f"{type(e).__name__}: {e}"

            # Check files present
            import os as _os
            entry["files"] = sorted(f for f in _os.listdir(path) if not f.startswith("."))
        except Exception as e:
            entry["outer_error"] = f"{type(e).__name__}: {e}"
        out[r] = entry
    return out


@app.local_entrypoint()
def main() -> None:
    r = diag.remote()
    import json
    print(json.dumps(r, indent=2))
