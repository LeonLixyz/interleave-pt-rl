"""Patch multiple layers so vLLM tolerates OLMo-2's slow-tokenizer fallback:

1. Patch `PreTrainedTokenizerBase.__getattr__` to answer for
   `all_special_tokens_extended` before its buggy `vars(cls)[key]` path.
2. Patch `vLLM/tokenizer.py` to route through a safe helper.
3. Verify at end: import transformers, load a slow GPT2Tokenizer, access the
   attribute — if any step fails, print the error.
"""

import re
import site
import sys
from pathlib import Path

site_pkg = Path(site.getsitepackages()[0])

# ------ 1. Patch transformers: only add the class-level property. Do NOT modify
#     __getattr__ (that touched some hot init path in transformers 4.53+ and
#     broke AutoTokenizer.from_pretrained).
tf_base = site_pkg / "transformers/tokenization_utils_base.py"
src = tf_base.read_text()
orig = src

CLASS_LINE_RE = r"(class PreTrainedTokenizerBase[^\n]*:\n)"
CLASS_ATTR = (
    '    all_special_tokens_extended = property(\n'
    '        lambda self: list(getattr(self, "all_special_tokens", []) or [])\n'
    '    )\n'
)
if "all_special_tokens_extended = property" not in src:
    src = re.sub(CLASS_LINE_RE, r"\1" + CLASS_ATTR, src, count=1)

if src != orig:
    tf_base.write_text(src)
    print(f"[patch] transformers: {tf_base}")

# ------ 2. Patch vLLM tokenizer.py ------
vllm_tok = site_pkg / "vllm/transformers_utils/tokenizer.py"
if vllm_tok.exists():
    src = vllm_tok.read_text()
    orig = src
    src = src.replace(
        "tokenizer.all_special_tokens_extended",
        '_p2p_safe_extended(tokenizer)',
    )
    helper = (
        "def _p2p_safe_extended(t):\n"
        "    try:\n"
        "        return list(t.all_special_tokens_extended)\n"
        "    except (AttributeError, TypeError):\n"
        "        try:\n"
        "            return list(t.all_special_tokens)\n"
        "        except AttributeError:\n"
        "            return []\n\n"
    )
    if "_p2p_safe_extended" not in src:
        marker = "def get_cached_tokenizer"
        if marker in src:
            src = src.replace(marker, helper + marker, 1)
    if src != orig:
        vllm_tok.write_text(src)
        print(f"[patch] vLLM: {vllm_tok}")

# ------ 3. Verify ------
print("[verify] loading transformers …")
from transformers import AutoTokenizer, GPT2Tokenizer

print(f"[verify] hasattr(GPT2Tokenizer, 'all_special_tokens_extended'): "
      f"{hasattr(GPT2Tokenizer, 'all_special_tokens_extended')}")

try:
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    val = tok.all_special_tokens_extended
    print(f"[verify] slow tokenizer .all_special_tokens_extended = {val!r} OK")
except Exception as e:
    print(f"[verify] slow tokenizer access FAILED: {type(e).__name__}: {e}")
    # Don't raise — patches are what matter for runtime.
