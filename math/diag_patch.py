"""Diagnose whether the transformers __init__.py patch is present."""
from __future__ import annotations

import modal
from common import hf_image_base


def _img() -> modal.Image:
    return (
        hf_image_base()
        .pip_install("transformers>=4.53")
        .run_commands(
            "SITE=$(python -c 'import site; print(site.getsitepackages()[0])') && "
            "printf '\\n# --- p2p patch ---\\n"
            "from transformers.tokenization_utils_base import PreTrainedTokenizerBase as _PTB\\n"
            "if not hasattr(_PTB, \"_p2p_patched\"):\\n"
            "    _PTB.all_special_tokens_extended = property(lambda self: list(getattr(self, \"all_special_tokens\", []) or []))\\n"
            "    _PTB._p2p_patched = True\\n' >> $SITE/transformers/__init__.py"
        )
        .add_local_python_source("common")
    )


app = modal.App("diag-patch", image=_img())


@app.function(timeout=120, cpu=1)
def check() -> dict:
    import subprocess
    import site
    site_pkg = site.getsitepackages()[0]
    init_path = f"{site_pkg}/transformers/__init__.py"
    tail = subprocess.check_output(["tail", "-20", init_path]).decode()

    # Try import + check
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    has_p2p = hasattr(PreTrainedTokenizerBase, "_p2p_patched")
    has_attr = hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended")

    # Load a slow-style tokenizer and check
    from transformers import GPT2Tokenizer
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    try:
        val = tok.all_special_tokens_extended
        access_result = f"OK: {type(val).__name__}"
    except Exception as e:
        access_result = f"FAILED: {type(e).__name__}: {e}"

    return {
        "init_tail": tail,
        "has_p2p_flag": has_p2p,
        "has_attr_on_class": has_attr,
        "runtime_access_on_slow_tok": access_result,
    }


@app.local_entrypoint()
def main() -> None:
    r = check.remote()
    import json
    print(json.dumps(r, indent=2))
