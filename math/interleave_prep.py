"""
Arm-B leg-2 init prep for the math interleaving experiment.

Takes the EXISTING anchor-sweep RL checkpoint (math-1b-rl-deepscaler-from-step{N}
at global_step {rl_step}) — i.e. "pretrain->N, anneal, SFT, RL" — and stages it
to re-enter pretraining:
  1. [verl image]  merge verl FSDP shards -> HF Olmo2 safetensors
  2. [olmo image]  convert HF -> OLMo-core native DCP (validated converter)

Outputs on the checkpoint volume:
  /checkpoints/interleave/{tag}/rl_leg1_hf    (HF, also a deliverable eval point)
  /checkpoints/interleave/{tag}/rl_leg1_dcp   (DCP, --load-path for continue-pretrain)

Usage:
  cd math-pretraining && modal run --detach interleave_prep.py \
      --anchor step10000 --rl-step 1500 --tag armB_small
"""
from pathlib import Path

import modal

# Hardcoded from common.py (avoids sibling-module import fragility under `modal run`).
CHECKPOINT_VOLUME_NAME = "olmo-core-checkpoints-v2"
CACHE_VOLUME_NAME = "olmo-core-cache"
CHECKPOINT_MOUNT = "/checkpoints"
CACHE_MOUNT = "/cache"

LOCAL = Path(__file__).parent
REMOTE_OLMO_CORE = "/root/OLMo-core"
REMOTE_PROJECT = "/root/math-pretraining"
REMOTE_VERL_DIR = "/root/verl-olmo3"
LOCAL_VERL_DIR = str(LOCAL.parent / "pretrain-rl-scaling" / "verl-olmo3")


def verl_img() -> modal.Image:
    # Inlined copy of rl_eval._img() (sibling-module import fails under `modal run`).
    return (
        modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
        .apt_install("build-essential", "git", "curl", "libibverbs-dev", "libibverbs1")
        .pip_install("wheel", "packaging", "ninja", "setuptools")
        .pip_install("torch==2.8.0")
        .pip_install("huggingface_hub==0.34.4", "hf_xet==1.1.5")
        .pip_install("vllm==0.11.0", "transformers==4.57.1", "flash-attn==2.8.3",
                     extra_options="--no-build-isolation")
        .pip_install("antlr4-python3-runtime==4.13.2")
        .pip_install("ray[default]==2.43.0", "tensordict==0.10.0", "datasets==4.0.0",
                     "pyarrow==17.0.0", "pandas==2.2.3", "wandb==0.19.11", "mlflow==3.0.0")
        .pip_install("omegaconf==2.4.0.dev3", "hydra-core==1.4.0.dev1", extra_options="--no-deps")
        .pip_install("importlib-resources", "packaging")
        .pip_install("codetiming==1.4.0", "accelerate==1.2.1", "peft==0.14.0",
                     "liger-kernel==0.5.4", "pybind11", "pylatexenc", "dill==0.3.8",
                     "torchdata==0.10.0", "tensorboard", "uvicorn", "fastapi")
        .pip_install("math-verify==0.5.2")
        .add_local_dir(LOCAL_VERL_DIR, remote_path=REMOTE_VERL_DIR, copy=True,
                       ignore=[".git", ".git/**", "__pycache__/**", "docs/**", "tests/**"])
        .run_commands(f"cd {REMOTE_VERL_DIR} && pip install -e . --no-deps")
        .env({"VLLM_ATTENTION_BACKEND": "FLASH_ATTN"})
    )

# olmo image: identical to the validated preflight converter image.
olmo_img = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("build-essential", "curl", "git")
    .pip_install("wheel", "packaging", "ninja", "setuptools")
    .pip_install("torch==2.8.0")
    .pip_install("flash-attn==2.8.3", extra_options="--no-build-isolation")
    .pip_install("wandb>=0.18")
    .add_local_dir(str(LOCAL.parent / "OLMo-core"), remote_path=REMOTE_OLMO_CORE, copy=True,
                   ignore=[".git", ".git/**", ".venv/**", "__pycache__/**", "build/**", "dist/**"])
    .run_commands(f"cd {REMOTE_OLMO_CORE} && pip install -e '.[transformers]'")
    .add_local_dir(str(LOCAL), remote_path=REMOTE_PROJECT, copy=True, ignore=["__pycache__/**"])
)

checkpoint_volume = modal.Volume.from_name(CHECKPOINT_VOLUME_NAME, create_if_missing=True, version=2)
cache_volume = modal.Volume.from_name(CACHE_VOLUME_NAME, create_if_missing=True)

app = modal.App("math-interleave-prep")


@app.function(image=verl_img(), gpu="H200:1", timeout=60 * 60 * 2,
              volumes={CHECKPOINT_MOUNT: checkpoint_volume, CACHE_MOUNT: cache_volume})
def merge_rl_to_hf(anchor: str, rl_step: int, tag: str) -> str:
    import shutil, subprocess
    from pathlib import Path

    checkpoint_volume.reload()
    run = f"math-1b-rl-deepscaler-from-{anchor}"
    ckpt_dir = f"{CHECKPOINT_MOUNT}/rl/{run}/checkpoints/global_step_{rl_step}/actor"
    out_hf = f"{CHECKPOINT_MOUNT}/interleave/{tag}/rl_leg1_hf"
    assert Path(ckpt_dir).exists(), f"missing {ckpt_dir}"

    print(f"[merge] {ckpt_dir} -> {out_hf}", flush=True)
    r = subprocess.run(
        ["python", f"{REMOTE_VERL_DIR}/scripts/legacy_model_merger.py", "merge",
         "--backend", "fsdp", "--local_dir", ckpt_dir, "--target_dir", out_hf],
        text=True, timeout=1800,
    )
    if r.returncode != 0:
        raise RuntimeError(f"merge failed rc={r.returncode}")
    # ensure tokenizer/config from the actor/huggingface/ subdir are present
    hf_src = Path(ckpt_dir) / "huggingface"
    for f in hf_src.glob("*"):
        if f.is_file() and not (Path(out_hf) / f.name).exists():
            shutil.copy(f, Path(out_hf) / f.name)
    checkpoint_volume.commit()
    print(f"[merge] done -> {out_hf}", flush=True)
    return out_hf


@app.function(image=olmo_img, gpu="H100:1", timeout=60 * 60 * 2,
              volumes={CHECKPOINT_MOUNT: checkpoint_volume, CACHE_MOUNT: cache_volume})
def hf_to_dcp(tag: str, src_config_step: str, num_embeddings: int = 100352) -> str:
    import sys
    sys.path.insert(0, REMOTE_PROJECT)
    from convert_hf_to_dcp import convert_hf_to_dcp

    import shutil
    checkpoint_volume.reload()
    hf_in = f"{CHECKPOINT_MOUNT}/interleave/{tag}/rl_leg1_hf"
    dcp_out = f"{CHECKPOINT_MOUNT}/interleave/{tag}/rl_leg1_dcp"
    src_cfg = f"{CHECKPOINT_MOUNT}/math-1b-v0/{src_config_step}/config.json"

    shutil.rmtree(dcp_out, ignore_errors=True)  # clear any stale (nested) layout
    convert_hf_to_dcp(hf_in, dcp_out, num_embeddings=num_embeddings, device="cpu",
                      src_config_json=src_cfg)
    checkpoint_volume.commit()
    print(f"[convert] done -> {dcp_out}", flush=True)
    return dcp_out


@app.function(image=olmo_img, gpu="H100:1", timeout=60 * 60 * 2,
              volumes={CHECKPOINT_MOUNT: checkpoint_volume, CACHE_MOUNT: cache_volume})
def volume_hf_to_dcp(
    src_hf: str,
    out_dcp: str,
    src_config_step: str = "step10000",
    num_embeddings: int = 100352,
    overwrite: bool = False,
) -> str:
    """Convert an existing HF directory on the checkpoint volume to DCP.

    Unlike ``hf_to_dcp()``, this entrypoint does not assume the source came from
    a verl merge. It is used by the matched W0 control, whose source is the
    pre-RL SFT policy already stored under ``/checkpoints/sft``.
    """
    import shutil
    import sys

    sys.path.insert(0, REMOTE_PROJECT)
    from convert_hf_to_dcp import convert_hf_to_dcp

    def checkpoint_path(value: str) -> Path:
        path = Path(value)
        if not path.is_absolute():
            path = Path(CHECKPOINT_MOUNT) / path
        if path != Path(CHECKPOINT_MOUNT) and Path(CHECKPOINT_MOUNT) not in path.parents:
            raise ValueError(f"path must be inside {CHECKPOINT_MOUNT}: {value}")
        return path

    checkpoint_volume.reload()
    hf_in = checkpoint_path(src_hf)
    dcp_out = checkpoint_path(out_dcp)
    src_cfg = Path(CHECKPOINT_MOUNT) / "math-1b-v0" / src_config_step / "config.json"

    for required in ("config.json", "model.safetensors"):
        if not (hf_in / required).is_file():
            raise FileNotFoundError(f"missing HF input file: {hf_in / required}")
    if not src_cfg.is_file():
        raise FileNotFoundError(f"missing source training config: {src_cfg}")
    if dcp_out.exists():
        if not overwrite:
            raise FileExistsError(f"refusing to replace existing output: {dcp_out}")
        shutil.rmtree(dcp_out)

    print(f"[convert-volume-hf] {hf_in} -> {dcp_out}", flush=True)
    convert_hf_to_dcp(
        str(hf_in),
        str(dcp_out),
        num_embeddings=num_embeddings,
        device="cpu",
        src_config_json=str(src_cfg),
    )
    checkpoint_volume.commit()
    print(f"[convert-volume-hf] done -> {dcp_out}", flush=True)
    return str(dcp_out)


@app.function(image=olmo_img, gpu="H100:1", timeout=60 * 60 * 2,
              volumes={CHECKPOINT_MOUNT: checkpoint_volume, CACHE_MOUNT: cache_volume})
def dcp_to_hf(src_dcp: str, out_hf: str) -> str:
    """Forward convert an OLMo-core DCP checkpoint -> HF Olmo2 dir on the volume,
    so the (torch-2.6) SFT image can consume it as a local --base-model."""
    import sys
    sys.path.insert(0, REMOTE_PROJECT)
    from olmo_core.data import TokenizerConfig
    from olmo_core.nn.hf import convert_checkpoint_to_hf, load_config
    from olmo_core.utils import prepare_cli_environment

    prepare_cli_environment()
    checkpoint_volume.reload()
    src = f"{CHECKPOINT_MOUNT}/{src_dcp}"
    outp = f"{CHECKPOINT_MOUNT}/{out_hf}"
    cfg = load_config(src)
    convert_checkpoint_to_hf(
        original_checkpoint_path=src,
        output_path=outp,
        transformer_config_dict=cfg.get("model"),
        tokenizer_config_dict=(cfg.get("dataset", {}).get("tokenizer")
                               or TokenizerConfig.dolma2().as_config_dict()),
        validate=False,
    )
    checkpoint_volume.commit()
    print(f"[dcp_to_hf] {src} -> {outp}", flush=True)
    return outp


@app.local_entrypoint()
def convert_dcp(src_dcp: str, out_hf: str):
    """e.g. --src-dcp interleave/armB_small/leg2_anneal/step2385
            --out-hf interleave/armB_small/leg2_anneal_hf"""
    print(f"[prep] DCP->HF: {src_dcp} -> {out_hf}")
    print(dcp_to_hf.remote(src_dcp=src_dcp, out_hf=out_hf))


@app.local_entrypoint()
def stage_hf(
    src_hf: str,
    out_dcp: str,
    src_config_step: str = "step10000",
    num_embeddings: int = 100352,
    overwrite: bool = False,
):
    """Stage a pre-existing HF checkpoint as a fresh-optimizer OLMo DCP."""
    print(f"[prep] volume HF->DCP: {src_hf} -> {out_dcp}")
    print(volume_hf_to_dcp.remote(
        src_hf=src_hf,
        out_dcp=out_dcp,
        src_config_step=src_config_step,
        num_embeddings=num_embeddings,
        overwrite=overwrite,
    ))


@app.local_entrypoint()
def main(anchor: str = "step10000", rl_step: int = 1500, tag: str = "armB_small",
         skip_merge: bool = False):
    if not skip_merge:
        print(f"[prep] merge {anchor}@rl{rl_step} then HF->DCP, tag={tag}")
        merge_rl_to_hf.remote(anchor=anchor, rl_step=rl_step, tag=tag)
    else:
        print(f"[prep] skip_merge: reuse staged HF, convert only, tag={tag}")
    dcp = hf_to_dcp.remote(tag=tag, src_config_step=anchor)
    print(f"[prep] Arm-B leg-2 init DCP staged at: {dcp}")
    print(f"[prep] next: continue-pretrain with --load-path {dcp}")
