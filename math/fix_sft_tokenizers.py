"""Overwrite each SFT repo's tokenizer files with the anneal repo's tokenizer.

The anneal HF repos (produced by olmo_core.nn.hf.convert_checkpoint_to_hf) have
a proper vLLM-compatible OLMo-2 tokenizer. LLaMA-Factory's SFT save wrote a
different (broken w.r.t. vLLM's fast-tokenizer path) tokenizer_config.json.
This script downloads the anneal tokenizer files once and pushes copies into
each SFT repo (overwriting the broken files) — so vLLM loads them cleanly.
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


app = modal.App("fix-sft-tokenizers", image=_img())
hf_secret = modal.Secret.from_name("huggingface-secret")

REFERENCE_REPO = "pre-to-post-olmo/math-1b-anneal-from-step95368"
SFT_REPOS = [
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
]

# Files that comprise the tokenizer (safe to overwrite from a compatible model).
TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
]


@app.function(secrets=[hf_secret], timeout=600, cpu=2)
def fetch_reference_files() -> dict:
    """Download tokenizer files from REFERENCE_REPO on Modal side (uses Modal's HF secret)."""
    import os
    from huggingface_hub import hf_hub_download

    token = os.environ["HF_TOKEN"]
    files = {}
    for name in TOKENIZER_FILES:
        try:
            path = hf_hub_download(repo_id=REFERENCE_REPO, filename=name, token=token)
            files[name] = open(path, "rb").read()
            print(f"  {name}: {len(files[name])} bytes")
        except Exception as e:
            print(f"  {name}: MISSING ({e}) — skipping")
    return files


@app.function(secrets=[hf_secret], timeout=1200, cpu=2)
def surgery(sft_repo: str, reference_files: dict) -> dict:
    """Upload each tokenizer file from `reference_files` into `sft_repo`."""
    import os
    from io import BytesIO
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HF_TOKEN"])
    out = {"repo": sft_repo, "uploaded": []}
    for name, blob in reference_files.items():
        if blob is None:
            continue
        try:
            api.upload_file(
                path_or_fileobj=BytesIO(blob),
                path_in_repo=name,
                repo_id=sft_repo,
                commit_message=f"tokenizer surgery: overwrite {name} with anneal-model tokenizer (vLLM compat)",
            )
            out["uploaded"].append(name)
        except Exception as e:
            out[f"err_{name}"] = f"{type(e).__name__}: {e}"
    return out


@app.local_entrypoint()
def main() -> None:
    """Fetch reference tokenizer files once, then push into each SFT repo in parallel."""
    print(f"[ref] downloading tokenizer files from {REFERENCE_REPO} (via Modal)")
    ref_files = fetch_reference_files.remote()
    for k, v in ref_files.items():
        print(f"  {k}: {len(v) if v else '(missing)'} bytes")

    print(f"[surgery] pushing to {len(SFT_REPOS)} SFT repos in parallel")
    futures = [surgery.spawn(sft_repo=r, reference_files=ref_files) for r in SFT_REPOS]
    for f in futures:
        try:
            print(f.get())
        except Exception as e:
            print(f"err: {e}")
