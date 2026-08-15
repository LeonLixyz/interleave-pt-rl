"""
Reverse checkpoint converter: HF Olmo2ForCausalLM  ->  OLMo-core native DCP.

The forward direction (OLMo-core DCP -> HF) ships as
olmo_core.nn.hf.convert_checkpoint_to_hf. The reverse has no CLI, but the hard
part (HF<->OLMo-core key remapping + embedding resize) is already implemented by
olmo_core.nn.hf.load_hf_model, which populates an OLMo-core model state dict
in-place from an HF directory. This wrapper builds the OLMo2-1B model at the
pretraining vocab (padded 100352), loads HF weights into it, and writes a
weights-only DCP checkpoint the trainer can fork-resume from.

Why weights-only (no optimizer): an RL'd model carries a verl/FSDP optimizer
state that does not map to OLMo-core AdamW. The interleave fork resumes with
load_trainer_state=False anyway, so a fresh optimizer is the intended behavior
(and the control arm resets its optimizer identically).

Vocab note: the merged RL/HF config reports vocab_size=100278 (dolma2 real
tokens); the pretraining model is built at padded_vocab_size()=100352. The 74
extra rows are pure padding (never indexed, token ids <= 100277), so passing
num_embeddings=100352 is a shape-pad, not a semantic change.

Run single-process (1 GPU or CPU); OLMo-core DCP is reshard-aware, so the
multi-GPU trainer reshards on load.
"""
import argparse
import json
import logging
import shutil
from pathlib import Path

import torch

from olmo_core.data import TokenizerConfig
from olmo_core.distributed.checkpoint import save_model_and_optim_state
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.hf import load_hf_model
from olmo_core.nn.transformer.config import TransformerConfig

log = logging.getLogger("convert_hf_to_dcp")


def build_olmo2_1b_config(vocab_size: int, attn_backend: AttentionBackendName):
    # MUST match the config used in training (train_inner_mix.py:407-410) so the
    # written DCP is shape/key-compatible with the trainer's model.
    return TransformerConfig.olmo2_1B_v2(vocab_size=vocab_size, attn_backend=attn_backend)


def convert_hf_to_dcp(
    hf_path: str,
    dcp_out: str,
    *,
    num_embeddings: int = 100352,
    device: str = "cpu",
    src_config_json: str | None = None,
):
    dcp_out = Path(dcp_out)
    dcp_out.mkdir(parents=True, exist_ok=True)

    # CPU works and avoids GPU flash-attn deps for a pure weight copy; the model
    # is only used as a state-dict container, never run here.
    attn_backend = (
        AttentionBackendName.flash_2 if device == "cuda" else AttentionBackendName.torch
    )
    model_config = build_olmo2_1b_config(num_embeddings, attn_backend)

    log.info("Building OLMo-core OLMo2-1B model (vocab=%d) on %s", num_embeddings, device)
    model = model_config.build(init_device=device)

    model_state_dict = model.state_dict()
    log.info("Loading HF weights from %s into OLMo-core state dict", hf_path)
    load_hf_model(
        hf_path,
        model_state_dict,
        num_embeddings=num_embeddings,
        work_dir=str(dcp_out / "_hf_tmp"),
    )
    # load_hf_model mutates model_state_dict in place; push it back into the model.
    model.load_state_dict(model_state_dict)

    # Write the DCP DIRECTLY to dcp_out (not a model_and_optim/ subdir): this
    # produces {dcp_out}/.metadata + {dcp_out}/__*.distcp, which the trainer's
    # Checkpointer.dir_is_checkpoint() recognizes as a model-only checkpoint
    # (top-level .metadata branch) and load(load_trainer_state=False) reads via
    # its "base directory" fallback. Nesting under model_and_optim/ fails the gate.
    log.info("Writing weights-only DCP to %s", dcp_out)
    save_model_and_optim_state(str(dcp_out), model, save_overwrite=True)

    # Preserve the source step's config.json alongside so the trainer's loader and
    # any downstream tooling see the same model metadata the run expects.
    if src_config_json and Path(src_config_json).exists():
        shutil.copy(src_config_json, dcp_out / "config.json")

    # Record provenance for auditing which HF model seeded this DCP.
    (dcp_out / "conversion_meta.json").write_text(
        json.dumps({"source_hf": str(hf_path), "num_embeddings": num_embeddings,
                    "optim_state": "fresh (weights-only)"}, indent=2)
    )
    log.info("Done: %s", dcp_out)
    return str(dcp_out)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--hf-path", required=True, help="Merged HF Olmo2 dir (from legacy_model_merger)")
    p.add_argument("--dcp-out", required=True, help="Output OLMo-core DCP checkpoint dir")
    p.add_argument("--num-embeddings", type=int, default=100352,
                   help="Padded vocab of the pretraining model (dolma2 padded_vocab_size)")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--src-config-json", default=None,
                   help="Optional config.json from the source step ckpt to copy alongside")
    args = p.parse_args()

    # Sanity: dolma2 padded vocab should equal the default; warn if overridden oddly.
    expected = TokenizerConfig.dolma2().padded_vocab_size()
    if args.num_embeddings != expected:
        log.warning("num_embeddings=%d != dolma2 padded_vocab_size=%d",
                    args.num_embeddings, expected)

    convert_hf_to_dcp(args.hf_path, args.dcp_out, num_embeddings=args.num_embeddings,
                      device=args.device, src_config_json=args.src_config_json)


if __name__ == "__main__":
    main()
