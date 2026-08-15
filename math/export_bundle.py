"""Merge verl FSDP RL checkpoints -> weights-only HF safetensors, staged on the
volume for later upload into the consolidated `pre-to-post-olmo/Math-Models` repo.

This is the same merge `rl_eval.py` runs before eval (legacy_model_merger.py,
fsdp backend), except we WRITE it durably to the volume instead of /tmp + rmtree.

Layout written:
    /checkpoints/export_bundle/step{anchor}/rl/step{rl_step}/
        config.json  model.safetensors  generation_config.json
        tokenizer.json tokenizer_config.json vocab.json merges.txt
        special_tokens_map.json chat_template.jinja

Usage:
    # validate one checkpoint (also prints the real safetensors size)
    modal run export_bundle.py::merge_one --anchor step95368 --rl-step 3000
    # fan out every 100 steps (<=3000) across all 15 anchors
    modal run --detach export_bundle.py::merge_all
"""

from __future__ import annotations

import modal

from common import CACHE_MOUNT, CACHE_VOLUME_NAME, CHECKPOINT_MOUNT, CHECKPOINT_VOLUME_NAME

LOCAL_VERL_DIR = "/Users/leonli66/Desktop/Research/RL/Chess RL/pretrain-rl-scaling/verl-olmo3"
REMOTE_VERL_DIR = "/root/verl-olmo3"

ANCHORS = [
    "step5000", "step10000", "step15000", "step20000", "step25000",
    "step30000", "step35000", "step40000", "step45000", "step50000",
    "step60000", "step70000", "step80000", "step90000", "step95368",
]
RL_STEPS = list(range(100, 3001, 100))  # 100,200,...,3000  (30 per anchor)


def _img() -> modal.Image:
    # Same env rl_eval uses for the merger (verl + torch 2.8).
    return (
        modal.Image.from_registry(
            "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11"
        )
        .apt_install("build-essential", "git", "curl", "libibverbs-dev", "libibverbs1")
        .pip_install("wheel", "packaging", "ninja", "setuptools")
        .pip_install("torch==2.8.0")
        .pip_install("huggingface_hub==0.34.4", "hf_xet==1.1.5", "safetensors>=0.4")
        .pip_install("transformers==4.57.1")
        .pip_install(
            "ray[default]==2.43.0", "tensordict==0.10.0", "pyarrow==17.0.0",
            "pandas==2.2.3",
        )
        .pip_install("omegaconf==2.4.0.dev3", "hydra-core==1.4.0.dev1", extra_options="--no-deps")
        .pip_install("importlib-resources", "packaging", "codetiming==1.4.0", "dill==0.3.8",
                     "accelerate==1.2.1")
        .add_local_dir(
            LOCAL_VERL_DIR, remote_path=REMOTE_VERL_DIR, copy=True,
            ignore=[".git", ".git/**", "__pycache__/**", "docs/**", "tests/**"],
        )
        .run_commands(f"cd {REMOTE_VERL_DIR} && pip install -e . --no-deps")
        .add_local_python_source("common")
    )


app = modal.App("export-bundle", image=_img())

checkpoint_volume = modal.Volume.from_name(CHECKPOINT_VOLUME_NAME, create_if_missing=True, version=2)
cache_volume = modal.Volume.from_name(CACHE_VOLUME_NAME, create_if_missing=True)

_KEEP = {
    "config.json", "generation_config.json", "merges.txt",
    "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json",
    "vocab.json", "chat_template.jinja",
}


@app.function(
    volumes={CHECKPOINT_MOUNT: checkpoint_volume, CACHE_MOUNT: cache_volume},
    timeout=60 * 30, cpu=8.0, memory=64 * 1024,
)
def merge_one(anchor: str, rl_step: int) -> dict:
    import os
    import shutil
    import subprocess
    from pathlib import Path

    checkpoint_volume.reload()
    run = f"math-1b-rl-deepscaler-from-{anchor}"
    ckpt_dir = f"{CHECKPOINT_MOUNT}/rl/{run}/checkpoints/global_step_{rl_step}/actor"
    out_dir = f"{CHECKPOINT_MOUNT}/export_bundle/{anchor}/rl/step{rl_step}"
    done_marker = Path(out_dir) / "model.safetensors"

    if not Path(ckpt_dir).exists():
        return {"anchor": anchor, "rl_step": rl_step, "status": "missing_src"}
    if done_marker.exists():
        return {"anchor": anchor, "rl_step": rl_step, "status": "already_done"}

    tmp = f"/tmp/merged_{anchor}_{rl_step}"
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"[merge] {ckpt_dir} -> {tmp}", flush=True)
    r = subprocess.run(
        ["python", f"{REMOTE_VERL_DIR}/scripts/legacy_model_merger.py", "merge",
         "--backend", "fsdp", "--local_dir", ckpt_dir, "--target_dir", tmp],
        capture_output=True, text=True, timeout=1500,
    )
    print(f"[merge] rc={r.returncode}", flush=True)
    print("[merge stdout tail]\n" + (r.stdout or "")[-2000:], flush=True)
    print("[merge stderr tail]\n" + (r.stderr or "")[-3000:], flush=True)
    if r.returncode != 0:
        return {"anchor": anchor, "rl_step": rl_step, "status": "merge_failed",
                "stderr": r.stderr[-1500:]}

    # bring in tokenizer/config from the actor's huggingface/ folder
    hf_src = Path(ckpt_dir) / "huggingface"
    for f in hf_src.glob("*"):
        if f.is_file() and not (Path(tmp) / f.name).exists():
            shutil.copy(f, Path(tmp) / f.name)

    # keep weights-only: drop anything that isn't a weight/config/tokenizer
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for f in Path(tmp).glob("*"):
        if f.is_file() and (f.name in _KEEP or f.suffix == ".safetensors"):
            shutil.copy(f, Path(out_dir) / f.name)

    size = sum(p.stat().st_size for p in Path(out_dir).glob("*") if p.is_file())
    st = next(Path(out_dir).glob("*.safetensors"), None)
    st_size = st.stat().st_size if st else 0
    shutil.rmtree(tmp, ignore_errors=True)
    checkpoint_volume.commit()

    os.sync()
    return {
        "anchor": anchor, "rl_step": rl_step, "status": "ok",
        "out_dir": out_dir,
        "safetensors_gb": round(st_size / 1e9, 3),
        "total_gb": round(size / 1e9, 3),
        "files": sorted(p.name for p in Path(out_dir).glob("*")),
    }


@app.function(timeout=60 * 60 * 12)
def merge_all() -> dict:
    jobs = [(a, s) for a in ANCHORS for s in RL_STEPS]
    results = list(merge_one.starmap(jobs, return_exceptions=True))
    ok = sum(1 for r in results if isinstance(r, dict) and r.get("status") in ("ok", "already_done"))
    gb = sum(r.get("safetensors_gb", 0) for r in results if isinstance(r, dict))
    bad = [r for r in results if not (isinstance(r, dict) and r.get("status") in ("ok", "already_done"))]
    return {"total_jobs": len(jobs), "ok": ok, "approx_rl_gb": round(gb, 1),
            "failures": bad[:20]}
