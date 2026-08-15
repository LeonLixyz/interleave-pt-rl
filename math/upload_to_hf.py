"""Upload OLMo-core checkpoints (raw + HF-converted) to HuggingFace.

Target repo: `pre-to-post-olmo` (user's own namespace inferred from HF token).

For each (checkpoint_path, subpath) pair:
  1. Upload raw DCP folder (weights + optimizer state) to <subpath>/raw/
  2. Run convert_checkpoint_to_hf -> HF format
  3. Upload HF folder to <subpath>/hf/

All uploads run in parallel via Function.spawn — one container per checkpoint.

Usage:
    # Phase 1: stable checkpoints (already done training)
    modal run upload_to_hf.py::upload_stable

    # Phase 2: anneal final checkpoints (run AFTER anneals finish)
    modal run upload_to_hf.py::upload_anneal

    # Or a single ckpt:
    modal run upload_to_hf.py::upload_single \
        --checkpoint-path /checkpoints/math-1b-v0/step95368 \
        --subpath stable/step95368
"""

from __future__ import annotations

import modal

from common import (
    CHECKPOINT_MOUNT,
    CHECKPOINT_VOLUME_NAME,
    hf_image_base,
)

HF_ORG = "pre-to-post-olmo"  # HuggingFace organization; one repo per checkpoint
LOCAL_OLMO_CORE = "/Users/leonli66/Desktop/Research/RL/Chess RL/OLMo-core"
REMOTE_OLMO_CORE = "/root/OLMo-core"


def _img() -> modal.Image:
    return (
        hf_image_base()
        .pip_install(
            "torch==2.6.0",
            "transformers>=4.46",
            "safetensors>=0.4",
            "accelerate>=1.0",
            "huggingface_hub>=0.26",
        )
        .add_local_dir(
            LOCAL_OLMO_CORE,
            remote_path=REMOTE_OLMO_CORE,
            copy=True,
            ignore=[".git", ".git/**", ".venv/**", "__pycache__/**", "build/**"],
        )
        .run_commands(f"cd {REMOTE_OLMO_CORE} && pip install -e .")
        .add_local_python_source("common")
    )


app = modal.App("upload-to-hf", image=_img())

checkpoint_volume = modal.Volume.from_name(
    CHECKPOINT_VOLUME_NAME, create_if_missing=True, version=2
)
hf_secret = modal.Secret.from_name("huggingface-secret")


def _resolve_repo(repo_name: str) -> str:
    """Return the full HF repo id `<org>/<repo_name>`."""
    return f"{HF_ORG}/{repo_name}"


@app.function(
    volumes={CHECKPOINT_MOUNT: checkpoint_volume},
    secrets=[hf_secret],
    timeout=60 * 60 * 6,
    cpu=8.0,
    memory=64 * 1024,
)
def upload_one(
    checkpoint_path: str,
    repo_name: str,
    do_raw: bool = True,
    do_convert: bool = True,
    skip_validation: bool = True,
    flat_hf: bool = True,
    delete_old_hf_subfolder: bool = True,
) -> dict:
    """Upload one checkpoint to its OWN HF repo (under the pre-to-post-olmo org).

    Repo layout when flat_hf=True (default, plug-and-play with AutoModelForCausalLM):
        <repo>/
        ├── config.json                (from HF conversion)
        ├── model.safetensors
        ├── tokenizer.*
        ├── ...
        └── raw/                       (OLMo-core DCP + optim states)
    When flat_hf=False the HF files live at <repo>/hf/ instead.
    """
    import os
    import subprocess
    import tempfile
    from pathlib import Path

    from huggingface_hub import HfApi, create_repo

    checkpoint_volume.reload()
    src = Path(checkpoint_path)
    if not src.exists():
        return {"error": f"checkpoint not found: {checkpoint_path}", "repo_name": repo_name}

    token = os.environ["HF_TOKEN"]
    api = HfApi(token=token)
    repo_id = _resolve_repo(repo_name)
    create_repo(repo_id=repo_id, token=token, exist_ok=True, repo_type="model")
    print(f"[repo] {repo_id}")

    out = {
        "checkpoint": str(src),
        "repo_name": repo_name,
        "repo_id": repo_id,
    }

    # 1) Upload raw DCP folder
    if do_raw:
        n_files = sum(1 for _ in src.rglob("*") if _.is_file())
        total_bytes = sum(p.stat().st_size for p in src.rglob("*") if p.is_file())
        print(
            f"[raw] uploading {n_files} files, {total_bytes / 1e9:.2f} GB "
            f"from {src} -> {repo_id}/raw"
        )
        api.upload_folder(
            folder_path=str(src),
            repo_id=repo_id,
            path_in_repo="raw",
            commit_message=f"raw ckpt {repo_name}",
            ignore_patterns=["*.tmp"],
        )
        out["raw_uploaded"] = True
        out["raw_files"] = n_files
        out["raw_bytes"] = total_bytes

    # 2) Convert to HF format, then upload
    if do_convert:
        # Apply torch.compiler.disable(reason=...) compat shim BEFORE importing
        # OLMo-core (its TEAttentionBackend uses the kwarg). Same shim as
        # train_inner_mix.py.
        import inspect
        import torch
        _orig_disable = torch.compiler.disable
        if "reason" not in inspect.signature(_orig_disable).parameters:
            def _disable_compat(fn=None, recursive=True, reason=None):
                del reason
                return _orig_disable(fn=fn, recursive=recursive)
            torch.compiler.disable = _disable_compat

        with tempfile.TemporaryDirectory(prefix="hfconv_") as tmp:
            hf_dir = Path(tmp) / "hf_out"
            print(f"[convert] {src} -> {hf_dir}")
            try:
                from olmo_core.config import DType
                from olmo_core.data import TokenizerConfig
                from olmo_core.nn.hf import convert_checkpoint_to_hf as _do_convert
                from olmo_core.nn.hf import load_config
                from olmo_core.utils import prepare_cli_environment

                prepare_cli_environment()

                experiment_config = load_config(str(src))
                if experiment_config is None:
                    raise RuntimeError(f"experiment config missing in {src}")
                transformer_config_dict = experiment_config.get("model")
                if transformer_config_dict is None:
                    raise RuntimeError(f"model config missing in {src}/config.json")

                # Our ConfigSaverCallback didn't write a `dataset.tokenizer` block.
                # Fall back to the known training tokenizer (dolma2) — same as the
                # one we trained with.
                tokenizer_cfg = (
                    experiment_config.get("dataset", {}).get("tokenizer")
                    or TokenizerConfig.dolma2().as_config_dict()
                )

                _do_convert(
                    original_checkpoint_path=str(src),
                    output_path=str(hf_dir),
                    transformer_config_dict=transformer_config_dict,
                    tokenizer_config_dict=tokenizer_cfg,
                    dtype=DType.bfloat16,
                    validate=not skip_validation,
                )
            except Exception as e:
                import traceback
                out["convert_error"] = f"{type(e).__name__}: {e}"
                out["convert_traceback"] = traceback.format_exc()
                return out

            n_files = sum(1 for _ in hf_dir.rglob("*") if _.is_file())
            total_bytes = sum(p.stat().st_size for p in hf_dir.rglob("*") if p.is_file())
            path_in_repo = "" if flat_hf else "hf"
            dest = f"{repo_id}/{'(root)' if flat_hf else 'hf'}"
            print(f"[hf] uploading {n_files} files, {total_bytes / 1e9:.2f} GB -> {dest}")
            api.upload_folder(
                folder_path=str(hf_dir),
                repo_id=repo_id,
                path_in_repo=path_in_repo,
                commit_message=f"hf ckpt {repo_name} (flat={flat_hf})",
            )
            out["hf_uploaded"] = True
            out["hf_flat"] = flat_hf
            out["hf_files"] = n_files
            out["hf_bytes"] = total_bytes

            if flat_hf and delete_old_hf_subfolder:
                try:
                    api.delete_folder(
                        path_in_repo="hf",
                        repo_id=repo_id,
                        commit_message=f"remove obsolete hf/ subfolder (flat re-upload)",
                    )
                    out["deleted_hf_subfolder"] = True
                except Exception as e:
                    out["delete_hf_subfolder_error"] = f"{type(e).__name__}: {e}"

    return out


STABLE_STEPS = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 95368]
ANNEAL_FORKS = [
    # (stable_anchor_step, anneal_run_name_suffix)
    (10000, "math-1b-v0-anneal-step10000-1node"),
    (20000, "math-1b-v0-anneal-step20000-1node"),
    (30000, "math-1b-v0-anneal-step30000-1node"),
    (40000, "math-1b-v0-anneal-step40000-1node"),
    (50000, "math-1b-v0-anneal-step50000-1node"),
    (60000, "math-1b-v0-anneal-step60000-1node"),
    (70000, "math-1b-v0-anneal-step70000-1node"),
    (80000, "math-1b-v0-anneal-step80000-1node"),
    (90000, "math-1b-v0-anneal-step90000-1node"),
    (95368, "math-1b-v0-anneal-step95368"),  # the original 4-node anneal
]
# Anneal saves one permanent at duration end. Empirically: step2385 for the 4-node
# anneal (it overshoots by 1). The 1-node anneals will save at step2384 (5B / 2.1M).
ANNEAL_FINAL_STEPS = {
    "math-1b-v0-anneal-step95368": 2385,  # original 4-node anneal
}
DEFAULT_ANNEAL_FINAL_STEP = 2385  # all 1-node anneals ended at step2385


def _gather(targets, do_raw=True, do_convert=True, flat_hf=True):
    """Fire spawn() for each target; collect results."""
    print(f"firing {len(targets)} upload jobs in parallel (one repo per ckpt, flat_hf={flat_hf})")
    futures = []
    for t in targets:
        f = upload_one.spawn(
            checkpoint_path=t["checkpoint_path"],
            repo_name=t["repo_name"],
            do_raw=do_raw,
            do_convert=do_convert,
            flat_hf=flat_hf,
        )
        futures.append((t, f))

    results = []
    for t, f in futures:
        try:
            r = f.get()
        except Exception as e:
            r = {"error": str(e), "repo_name": t["repo_name"]}
        results.append(r)
        print(f"  ✓ {t['repo_name']}: raw={r.get('raw_uploaded')} hf={r.get('hf_uploaded')}")
    return results


@app.local_entrypoint()
def upload_stable(do_raw: bool = True, do_convert: bool = True, flat_hf: bool = True) -> None:
    """Upload all 10 stable checkpoints, each to its OWN repo:
        pre-to-post-olmo/math-1b-stable-step<N>
    """
    targets = [
        {
            "checkpoint_path": f"{CHECKPOINT_MOUNT}/math-1b-v0/step{s}",
            "repo_name": f"math-1b-stable-step{s}",
        }
        for s in STABLE_STEPS
    ]
    results = _gather(targets, do_raw=do_raw, do_convert=do_convert, flat_hf=flat_hf)
    print(f"\nFINAL: {sum(1 for r in results if 'error' not in r)}/{len(results)} succeeded")


@app.local_entrypoint()
def upload_anneal(do_raw: bool = True, do_convert: bool = True, flat_hf: bool = True) -> None:
    """Upload the FINAL checkpoint of each anneal fork, each to its OWN repo:
        pre-to-post-olmo/math-1b-anneal-from-step<N>
    """
    targets = []
    for anchor, run_name in ANNEAL_FORKS:
        final = ANNEAL_FINAL_STEPS.get(run_name, DEFAULT_ANNEAL_FINAL_STEP)
        targets.append({
            "checkpoint_path": f"{CHECKPOINT_MOUNT}/{run_name}/step{final}",
            "repo_name": f"math-1b-anneal-from-step{anchor}",
        })
    results = _gather(targets, do_raw=do_raw, do_convert=do_convert, flat_hf=flat_hf)
    print(f"\nFINAL: {sum(1 for r in results if 'error' not in r)}/{len(results)} succeeded")


ANNEAL_FORKS_5K = [
    (a, f"math-1b-v0-anneal-step{a}-1node") for a in
    (5000, 15000, 25000, 35000, 45000, 55000, 65000, 75000, 85000, 95000)
]


@app.local_entrypoint()
def upload_anneal_5k(do_raw: bool = False, do_convert: bool = True, flat_hf: bool = True) -> None:
    """Upload the FINAL checkpoint of each every-5k anneal fork:
        pre-to-post-olmo/math-1b-anneal-from-step{5000,15000,...,95000}
    Skips raw by default (10 × ~4GB DCP = heavy, hf-only is enough for downstream).
    """
    targets = []
    for anchor, run_name in ANNEAL_FORKS_5K:
        final = ANNEAL_FINAL_STEPS.get(run_name, DEFAULT_ANNEAL_FINAL_STEP)
        targets.append({
            "checkpoint_path": f"{CHECKPOINT_MOUNT}/{run_name}/step{final}",
            "repo_name": f"math-1b-anneal-from-step{anchor}",
        })
    results = _gather(targets, do_raw=do_raw, do_convert=do_convert, flat_hf=flat_hf)
    print(f"\nFINAL: {sum(1 for r in results if 'error' not in r)}/{len(results)} succeeded")


@app.local_entrypoint()
def upload_single(
    checkpoint_path: str,
    repo_name: str,
    do_raw: bool = True,
    do_convert: bool = True,
) -> None:
    """Upload one ckpt to a single repo (pre-to-post-olmo/<repo_name>)."""
    r = upload_one.remote(
        checkpoint_path=checkpoint_path,
        repo_name=repo_name,
        do_raw=do_raw,
        do_convert=do_convert,
    )
    print(r)
