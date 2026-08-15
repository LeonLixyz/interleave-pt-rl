"""Stage pretrain/anneal/sft leaves into the export_bundle tree on the volume,
weights-only, alongside the already-merged rl/ leaves. After this runs, the full
`/checkpoints/export_bundle/step{anchor}/{pretrain,anneal,sft,rl}/...` tree is
ready to upload to `pre-to-post-olmo/Math-Models`.

Sources:
  anneal -> HF pre-to-post-olmo/math-1b-anneal-from-step{N}   (drop raw/)
  sft    -> HF pre-to-post-olmo/math-1b-sft-numinamath-bs512-from-step{N} (drop training junk)
  pretrain, 10k-stride anchors -> HF pre-to-post-olmo/math-1b-stable-step{N} (drop raw/)
  pretrain, 5k-stride anchors  -> convert volume DCP math-1b-v0/step{N} -> HF (dolma2 tok, bf16)

Usage:
    modal run stage_bundle.py::stage_one --anchor 20000 --stage anneal
    modal run --detach stage_bundle.py::stage_all
"""

from __future__ import annotations

import modal

from common import CHECKPOINT_MOUNT, CHECKPOINT_VOLUME_NAME, hf_image_base

LOCAL_OLMO_CORE = "/Users/leonli66/Desktop/Research/RL/Chess RL/OLMo-core"
REMOTE_OLMO_CORE = "/root/OLMo-core"
HF_ORG = "pre-to-post-olmo"

ANCHORS = [5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000,
           50000, 60000, 70000, 80000, 90000, 95368]
# pretrain snapshots that already exist as HF stable repos (10k stride); the rest
# come from the volume DCP.
STABLE_ON_HF = {10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 95368}

# weights-only file sets (everything else — raw/, trainer logs, pngs — is dropped)
_BASE_KEEP = ["config.json", "model.safetensors", "generation_config.json",
              "chat_template.jinja", "tokenizer.json", "tokenizer_config.json"]
_SFT_KEEP = _BASE_KEEP + ["vocab.json", "merges.txt", "special_tokens_map.json"]


def _img() -> modal.Image:
    return (
        hf_image_base()
        .pip_install("torch==2.6.0", "transformers>=4.46", "safetensors>=0.4",
                     "accelerate>=1.0", "huggingface_hub>=0.26")
        .add_local_dir(LOCAL_OLMO_CORE, remote_path=REMOTE_OLMO_CORE, copy=True,
                       ignore=[".git", ".git/**", ".venv/**", "__pycache__/**", "build/**"])
        .run_commands(f"cd {REMOTE_OLMO_CORE} && pip install -e .")
        .add_local_python_source("common")
    )


app = modal.App("stage-bundle", image=_img())
checkpoint_volume = modal.Volume.from_name(CHECKPOINT_VOLUME_NAME, create_if_missing=True, version=2)
hf_secret = modal.Secret.from_name("huggingface-secret")


def _repo_for(anchor: int, stage: str) -> str | None:
    if stage == "anneal":
        return f"{HF_ORG}/math-1b-anneal-from-step{anchor}"
    if stage == "sft":
        return f"{HF_ORG}/math-1b-sft-numinamath-bs512-from-step{anchor}"
    if stage == "pretrain" and anchor in STABLE_ON_HF:
        return f"{HF_ORG}/math-1b-stable-step{anchor}"
    return None  # pretrain from DCP


@app.function(volumes={CHECKPOINT_MOUNT: checkpoint_volume}, secrets=[hf_secret],
              timeout=60 * 30, cpu=8.0, memory=64 * 1024)
def stage_one(anchor: int, stage: str) -> dict:
    import os
    import shutil
    from pathlib import Path

    checkpoint_volume.reload()
    out_dir = Path(f"{CHECKPOINT_MOUNT}/export_bundle/step{anchor}/{stage}")
    if (out_dir / "model.safetensors").exists():
        return {"anchor": anchor, "stage": stage, "status": "already_done"}
    out_dir.mkdir(parents=True, exist_ok=True)

    repo = _repo_for(anchor, stage)

    if repo is not None:
        # ---- pull weights-only from an existing HF repo ----
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError
        keep = _SFT_KEEP if stage == "sft" else _BASE_KEEP
        got = []
        for fn in keep:
            try:
                p = hf_hub_download(repo_id=repo, filename=fn,
                                    token=os.environ["HF_TOKEN"])
                shutil.copy(p, out_dir / fn)
                got.append(fn)
            except EntryNotFoundError:
                pass  # optional file (e.g. generation_config) absent
        status = "ok_hf" if (out_dir / "model.safetensors").exists() else "no_weights"
        checkpoint_volume.commit()
        return {"anchor": anchor, "stage": stage, "status": status,
                "source": repo, "files": got}

    # ---- pretrain from DCP: convert math-1b-v0/step{N} -> HF ----
    import inspect
    import tempfile
    import torch
    _orig = torch.compiler.disable
    if "reason" not in inspect.signature(_orig).parameters:
        def _shim(fn=None, recursive=True, reason=None):
            del reason
            return _orig(fn=fn, recursive=recursive)
        torch.compiler.disable = _shim

    src = f"{CHECKPOINT_MOUNT}/math-1b-v0/step{anchor}"
    if not Path(src).exists():
        return {"anchor": anchor, "stage": stage, "status": "missing_dcp", "src": src}
    try:
        from olmo_core.config import DType
        from olmo_core.data import TokenizerConfig
        from olmo_core.nn.hf import convert_checkpoint_to_hf as _do_convert
        from olmo_core.nn.hf import load_config
        from olmo_core.utils import prepare_cli_environment
        prepare_cli_environment()
        cfg = load_config(src)
        tconf = cfg.get("model")
        tok = (cfg.get("dataset", {}).get("tokenizer")
               or TokenizerConfig.dolma2().as_config_dict())
        with tempfile.TemporaryDirectory(prefix="hfconv_") as tmp:
            _do_convert(original_checkpoint_path=src, output_path=tmp,
                        transformer_config_dict=tconf, tokenizer_config_dict=tok,
                        dtype=DType.bfloat16, validate=False)
            for f in Path(tmp).glob("*"):
                if f.is_file():
                    shutil.copy(f, out_dir / f.name)
    except Exception as e:
        import traceback
        return {"anchor": anchor, "stage": stage, "status": "convert_failed",
                "error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-1500:]}

    status = "ok_dcp" if (out_dir / "model.safetensors").exists() else "no_weights"
    os.sync()
    checkpoint_volume.commit()
    return {"anchor": anchor, "stage": stage, "status": status, "source": src,
            "files": sorted(p.name for p in out_dir.glob("*"))}


@app.function(timeout=60 * 60 * 6)
def stage_all() -> dict:
    jobs = [(a, s) for a in ANCHORS for s in ("pretrain", "anneal", "sft")]
    res = list(stage_one.starmap(jobs, return_exceptions=True))
    ok = [r for r in res if isinstance(r, dict) and r.get("status", "").startswith(("ok", "already"))]
    bad = [r for r in res if r not in ok]
    return {"total": len(jobs), "ok": len(ok), "bad": bad}
