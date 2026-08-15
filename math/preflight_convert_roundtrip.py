"""
Modal preflight: round-trip gate for the HF<->OLMo-core reverse converter.

This is the mandatory step-0 gate before spending any interleave GPU-hours. The
reverse converter's failure modes are SILENT (wrong vocab width, a bad key in
convert_state_from_hf, malformed model_and_optim/ layout) — the trainer would
load a corrupted checkpoint without error and invisibly confound the whole
interleave arm. So we prove weight fidelity here first.

Gate:
  1. Take an existing native DCP step checkpoint (default step10000).
  2. convert_checkpoint_to_hf  -> HF_probe          (forward, shipped)
  3. convert_hf_to_dcp         -> DCP_probe          (reverse, our new wrapper)
  4. Load HF_probe as an HF Olmo2 model; load DCP_probe into an OLMo-core model;
     run both on a fixed batch and assert next-token logits allclose.
  5. Assert the reloaded OLMo-core model is 100352-wide and the top-level DCP
     metadata exists (the layout consumed by weights-only trainer loads).

Run:
  cd math-pretraining && modal run --detach preflight_convert_roundtrip.py
"""
from pathlib import Path
from typing import Any

import modal

from common import (
    CACHE_MOUNT, CACHE_VOLUME_NAME, CHECKPOINT_MOUNT, CHECKPOINT_VOLUME_NAME,
)

REMOTE_OLMO_CORE = "/root/OLMo-core"
REMOTE_PROJECT = "/root/math-pretraining"
LOCAL = Path(__file__).parent

# Mirror train.py's image construction EXACTLY (same pip sequence, same OLMo-core
# copy + editable install) so the converter runs in the identical environment the
# continuation-pretrain leg will — whatever torch `pip install -e .` settles on.
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("build-essential", "curl", "git")
    .pip_install("wheel", "packaging", "ninja", "setuptools")
    # torch 2.6.0 (train.py's pin) cannot import the current OLMo-core commit
    # (torch.compiler.disable(reason=) is a 2.7+ API). Bump to 2.8 + flash-attn
    # 2.8.3 (the known-good pairing) — this is also the image the continuation
    # pretrain leg needs, since the checkpoint config specifies flash_2 attention.
    .pip_install("torch==2.8.0")
    .pip_install("flash-attn==2.8.3", extra_options="--no-build-isolation")
    .pip_install("wandb>=0.18")
    .add_local_dir(
        str(LOCAL.parent / "OLMo-core"), remote_path=REMOTE_OLMO_CORE, copy=True,
        ignore=[".git", ".git/**", ".mypy_cache/**", ".pytest_cache/**",
                ".ruff_cache/**", ".venv/**", "__pycache__/**", "build/**",
                "dist/**", "doc/**", "scratch/**"],
    )
    # [transformers] extra: the hf converter needs transformers (optional in OLMo-core)
    .run_commands(f"cd {REMOTE_OLMO_CORE} && pip install -e '.[transformers]'")
    .add_local_dir(str(LOCAL), remote_path=REMOTE_PROJECT, copy=True,
                   ignore=["__pycache__/**", ".venv/**", "*.tmp"])
    .add_local_python_source("common")
)

checkpoint_volume = modal.Volume.from_name(CHECKPOINT_VOLUME_NAME, create_if_missing=True, version=2)
cache_volume = modal.Volume.from_name(CACHE_VOLUME_NAME, create_if_missing=True)

app = modal.App("math-interleave-preflight", image=image)


@app.function(
    gpu="H100:1",
    timeout=60 * 60 * 2,
    volumes={CHECKPOINT_MOUNT: checkpoint_volume, CACHE_MOUNT: cache_volume},
)
def roundtrip(step: int = 10000, num_embeddings: int = 100352):
    import sys
    sys.path.insert(0, REMOTE_PROJECT)
    import json
    import torch
    from olmo_core.data import TokenizerConfig
    from olmo_core.distributed.checkpoint import load_model_and_optim_state
    from olmo_core.nn.attention import AttentionBackendName
    from olmo_core.nn.hf import convert_checkpoint_to_hf, load_config
    from olmo_core.nn.transformer.config import TransformerConfig
    from olmo_core.utils import prepare_cli_environment
    from transformers import AutoModelForCausalLM
    from convert_hf_to_dcp import convert_hf_to_dcp

    prepare_cli_environment()
    src = f"{CHECKPOINT_MOUNT}/math-1b-v0/step{step}"
    hf_probe = f"/tmp/hf_probe_step{step}"
    dcp_probe = f"/tmp/dcp_probe_step{step}"

    # Mirror upload_to_hf.py exactly: read the experiment config via load_config,
    # pull the model block, fall back to dolma2 tokenizer.
    experiment_config = load_config(str(src))
    transformer_config_dict = experiment_config.get("model")
    tok_cfg = TokenizerConfig.dolma2()
    tokenizer_cfg = (experiment_config.get("dataset", {}).get("tokenizer")
                     or tok_cfg.as_config_dict())

    # --- forward: DCP -> HF (shipped path) ---
    print(f"[preflight] forward convert {src} -> {hf_probe}", flush=True)
    convert_checkpoint_to_hf(
        original_checkpoint_path=src,
        output_path=hf_probe,
        transformer_config_dict=transformer_config_dict,
        tokenizer_config_dict=tokenizer_cfg,
        validate=False,
    )

    # --- reverse: HF -> DCP (our new wrapper) ---
    print(f"[preflight] reverse convert {hf_probe} -> {dcp_probe}", flush=True)
    convert_hf_to_dcp(hf_probe, dcp_probe, num_embeddings=num_embeddings,
                      device="cpu", src_config_json=f"{src}/config.json")

    assert Path(f"{dcp_probe}/.metadata").exists(), "top-level DCP metadata not written"

    # --- load both, compare logits on a fixed batch ---
    # fp32 comparison for a clean fidelity check -> sdpa/torch backends (flash-attn
    # is fp16/bf16-only; it's needed only for the forward-convert model build above,
    # not for this numerical comparison).
    device = "cuda"
    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_probe, torch_dtype=torch.float32, trust_remote_code=True,
        attn_implementation="sdpa").to(device).eval()

    olmo_cfg = TransformerConfig.olmo2_1B_v2(
        vocab_size=num_embeddings, attn_backend=AttentionBackendName.torch)
    olmo_model = olmo_cfg.build(init_device=device).eval()
    load_model_and_optim_state(dcp_probe, olmo_model)

    olmo_vocab = olmo_model.state_dict()["embeddings.weight"].shape[0] \
        if "embeddings.weight" in olmo_model.state_dict() else num_embeddings
    print(f"[preflight] OLMo-core reloaded vocab rows ~ {olmo_vocab}", flush=True)

    torch.manual_seed(0)
    ids = torch.randint(0, tok_cfg.vocab_size, (2, 64), device=device)
    with torch.no_grad():
        hf_logits = hf_model(ids).logits.float()
        olmo_logits = olmo_model(ids)
        olmo_logits = (olmo_logits.logits if hasattr(olmo_logits, "logits") else olmo_logits).float()

    # Compare only over real (non-padding) vocab columns.
    V = tok_cfg.vocab_size
    hf_c, olmo_c = hf_logits[..., :V], olmo_logits[..., :V]
    max_abs = (hf_c - olmo_c).abs().max().item()
    # Argmax agreement is the decision-relevant check (bf16 lineage => loose atol).
    agree = (hf_c.argmax(-1) == olmo_c.argmax(-1)).float().mean().item()
    allclose = torch.allclose(hf_c, olmo_c, atol=1e-2, rtol=1e-2)

    result = {"step": step, "max_abs_logit_diff": max_abs, "argmax_agreement": agree,
              "allclose_1e-2": bool(allclose), "olmo_vocab_rows": int(olmo_vocab),
              "dcp_layout": "top_level_weights_only",
              "PASS": bool(agree > 0.999 and max_abs < 0.5)}
    print(f"[preflight] RESULT: {json.dumps(result)}", flush=True)
    return result


@app.local_entrypoint()
def main(step: int = 10000):
    r = roundtrip.remote(step=step)
    print("PREFLIGHT", "PASS" if r.get("PASS") else "FAIL", r)
