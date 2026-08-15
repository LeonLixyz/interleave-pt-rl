"""Patch tokenizer_class from GPT2Tokenizer -> GPT2TokenizerFast in all 8 SFT repos.

The SFT'd HF repos have tokenizer_class=GPT2Tokenizer which forces slow-tokenizer
loading in vLLM. Fast tokenizer files (tokenizer.json) DO exist. Rewriting the
class name lets vLLM/transformers load the fast tokenizer.
"""

from __future__ import annotations

import modal
from common import hf_image_base


def _img() -> modal.Image:
    return (
        hf_image_base()
        .pip_install("huggingface_hub>=0.26")
        .add_local_python_source("common")
    )


app = modal.App("fix-tokenizer-class", image=_img())
hf_secret = modal.Secret.from_name("huggingface-secret")

REPOS = [
    "pre-to-post-olmo/math-1b-sft-openthoughts-from-step10000",
    "pre-to-post-olmo/math-1b-sft-openthoughts-from-step20000",
    "pre-to-post-olmo/math-1b-sft-openthoughts-from-step30000",
    "pre-to-post-olmo/math-1b-sft-openthoughts-from-step40000",
    "pre-to-post-olmo/math-1b-sft-openthoughts-from-step50000",
    "pre-to-post-olmo/math-1b-sft-openthoughts-from-step60000",
    "pre-to-post-olmo/math-1b-sft-openthoughts-from-step70000",
    "pre-to-post-olmo/math-1b-sft-openthoughts-from-step80000",
    "pre-to-post-olmo/math-1b-sft-openthoughts-from-step90000",
    "pre-to-post-olmo/math-1b-sft-openthoughts-from-step95368",
    "pre-to-post-olmo/math-1b-anneal-from-step10000",
    "pre-to-post-olmo/math-1b-anneal-from-step20000",
    "pre-to-post-olmo/math-1b-anneal-from-step30000",
    "pre-to-post-olmo/math-1b-anneal-from-step40000",
    "pre-to-post-olmo/math-1b-anneal-from-step50000",
    "pre-to-post-olmo/math-1b-anneal-from-step60000",
    "pre-to-post-olmo/math-1b-anneal-from-step70000",
    "pre-to-post-olmo/math-1b-anneal-from-step80000",
    "pre-to-post-olmo/math-1b-anneal-from-step90000",
    "pre-to-post-olmo/math-1b-anneal-from-step95368",
    "pre-to-post-olmo/math-1b-stable-step10000",
    "pre-to-post-olmo/math-1b-stable-step20000",
    "pre-to-post-olmo/math-1b-stable-step30000",
    "pre-to-post-olmo/math-1b-stable-step40000",
    "pre-to-post-olmo/math-1b-stable-step50000",
    "pre-to-post-olmo/math-1b-stable-step60000",
    "pre-to-post-olmo/math-1b-stable-step70000",
    "pre-to-post-olmo/math-1b-stable-step80000",
    "pre-to-post-olmo/math-1b-stable-step90000",
    "pre-to-post-olmo/math-1b-stable-step95368",
]


@app.function(secrets=[hf_secret], timeout=1200, cpu=2)
def patch_one(repo_id: str) -> dict:
    import json
    import os
    import tempfile
    from pathlib import Path
    from huggingface_hub import HfApi, hf_hub_download

    token = os.environ["HF_TOKEN"]
    api = HfApi(token=token)
    tmp = tempfile.mkdtemp()
    local = hf_hub_download(repo_id=repo_id, filename="tokenizer_config.json", local_dir=tmp)
    cfg = json.loads(Path(local).read_text())
    before = cfg.get("tokenizer_class")
    if before == "GPT2TokenizerFast":
        return {"repo": repo_id, "no_change": True}
    cfg["tokenizer_class"] = "GPT2TokenizerFast"
    Path(local).write_text(json.dumps(cfg, indent=2))
    api.upload_file(
        path_or_fileobj=local,
        path_in_repo="tokenizer_config.json",
        repo_id=repo_id,
        commit_message=f"fix: tokenizer_class {before} -> GPT2TokenizerFast (vLLM compat)",
    )
    return {"repo": repo_id, "before": before, "after": "GPT2TokenizerFast"}


@app.local_entrypoint()
def main() -> None:
    print(f"patching {len(REPOS)} repos in parallel")
    futures = [patch_one.spawn(repo_id=r) for r in REPOS]
    for f in futures:
        try:
            print(f.get())
        except Exception as e:
            print(f"err: {e}")
