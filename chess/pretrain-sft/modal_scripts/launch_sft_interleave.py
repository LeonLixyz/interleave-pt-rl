"""
Modal launcher: SFT for the interleave experiment (replica of the sweep recipe
that produced chess-pre-to-post/sft_trajectory_no_labels, which originally ran
on NYU Greene SLURM — no Modal SFT launcher existed before this one).

Recipe (verified against the uploaded C6p5e18_20m_alpha1.000_beta0.008 ckpt):
  2 GPUs, fp32, lr 3e-4 cosine->1e-5 warmup 50, 3 epochs, eff. batch 256,
  block 3072 (+yarn 3.0 on ctx-1024 pretrains), dataset sft_v1_200m_90k,
  first 134 sorted shards (132 train + 2 eval holdout).

Model-id naming reuses the pretrain_spec scheme with a free-form alpha token,
so downstream miles RL works unchanged via --spec, e.g.:
  alpha token "0.600i"  -> C6p5e18_20m_alpha0.600i_beta0.008
  miles spec            -> '6p5e18|20m|0.600i|0.008'

Usage:
  # Arm B leg-1 SFT (on the hot alpha1.000@step_60480 snapshot):
  modal run --detach modal_scripts/launch_sft_interleave.py \
    --pretrained-model /checkpoints/6p5e18/20m_C_6p5e18_alpha1.000/step_60480 \
    --alpha-token 0.600i
"""
from pathlib import Path

import modal

GPU_TYPE = "H200"
N_GPUS = 2
TIMEOUT_HOURS = 8

SFT_DATASET = "chess-pre-to-post/sft_v1_200m_90k"
SFT_HF_REPO = "chess-pre-to-post/sft_trajectory_no_labels"
CONFIG = "config/configs/qwen_multiturn_sft/sft_interleave_3072.yaml"

cuda_tag = "12.4.0-devel-ubuntu22.04"
repo_dir = Path(__file__).parent.parent

image = (
    modal.Image.from_registry(f"nvidia/cuda:{cuda_tag}", add_python="3.11")
    .apt_install("curl", "git")
    .pip_install(
        # exact mirror of the proven pretrain image (launch_interleave_pretrains.py)
        "torch>=2.6.0",
        "accelerate>=1.10.0",
        "transformers>=4.50.0",
        "datasets>=3.0.0",
        "pyarrow>=17.0.0",
        "pandas>=2.0.0",
        "pyyaml>=6.0",
        "omegaconf>=2.3.0",
        "wandb>=0.19.0",
        "einops>=0.7.0",
        "tokenizers>=0.19.0",
        "tqdm>=4.66.0",
        "chess>=1.11.0",
        "numpy>=2.0.0",
        "safetensors>=0.5.0",
        "sentencepiece>=0.2.0",
        "huggingface-hub>=0.28.0",
    )
    .add_local_dir(str(repo_dir / "scripts"), remote_path="/root/chess/scripts")
    .add_local_dir(str(repo_dir / "training"), remote_path="/root/chess/training")
    .add_local_dir(str(repo_dir / "config"), remote_path="/root/chess/config")
    .add_local_dir(str(repo_dir / "llm_tokens"), remote_path="/root/chess/llm_tokens")
    .add_local_dir(str(repo_dir / "evaluation"), remote_path="/root/chess/evaluation")
)

ckpt_volume = modal.Volume.from_name("rl-reasoning-checkpoints", create_if_missing=True)

app = modal.App(
    "sft-interleave",
    image=image,
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("huggingface-secret"),
    ],
    volumes={"/checkpoints": ckpt_volume},
)


@app.function(gpu=f"{GPU_TYPE}:{N_GPUS}", timeout=60 * 60 * TIMEOUT_HOURS, retries=0)
def sft_one(pretrained_model: str, alpha_token: str, beta_token: str = "0.008",
            upload: bool = True):
    import os
    import subprocess
    import sys

    os.environ["PYTHONUNBUFFERED"] = "1"

    model_id = f"C6p5e18_20m_alpha{alpha_token}_beta{beta_token}"
    final_dir = Path(f"/checkpoints/sft_interleave/trajectory_sep_no_labels/{model_id}/final")
    if final_dir.is_dir():
        print(f"[sft] Skipping: final exists at {final_dir}")
        return str(final_dir)

    if not Path(pretrained_model, "model.safetensors").exists():
        raise FileNotFoundError(f"pretrained model not staged: {pretrained_model}")

    from huggingface_hub import snapshot_download
    print(f"[sft] Downloading SFT dataset {SFT_DATASET} ...")
    snapshot_download(SFT_DATASET, repo_type="dataset", local_dir="/root/sft_data")

    cmd = [
        "accelerate", "launch", "--num_processes", str(N_GPUS),
        # default rendezvous port 29500 was already bound in the Modal container
        "--main_process_port", "29671",
        "scripts/train/run_sft.py",
        "--config", CONFIG,
        "--lr", "3e-4",
        "--block-size", "3072",
        "--pretrained-model", pretrained_model,
        "--naming-scheme", "pretrain_spec",
        "--total-compute", "6p5e18",
        "--modelsize", "20m",
        "--alpha", alpha_token,
        "--beta", beta_token,
        "--cot-field", "cot_by_method.trajectory_sep.cot_format_no_labels",
        "--cot-type", "trajectory_sep_no_labels",
        "--train-files", "/root/sft_data",
        "--data-name", "200m_generated",
        "--max-train-files", "134",
    ]
    print(f"[sft] cmd: {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(cmd, cwd="/root/chess", stdout=sys.stdout, stderr=sys.stderr)
    rc = proc.wait()
    ckpt_volume.commit()
    if rc != 0:
        raise RuntimeError(f"SFT failed (exit {rc})")

    if upload:
        from huggingface_hub import HfApi
        print(f"[sft] Uploading {final_dir} -> {SFT_HF_REPO}/{model_id}")
        HfApi().upload_folder(
            folder_path=str(final_dir),
            repo_id=SFT_HF_REPO,
            path_in_repo=model_id,
            commit_message=f"Upload SFT checkpoint: {model_id} (interleave experiment)",
        )
    print(f"[sft] Done: {model_id}")
    return str(final_dir)


@app.local_entrypoint()
def main(pretrained_model: str, alpha_token: str, beta_token: str = "0.008",
         dry_run: bool = False):
    model_id = f"C6p5e18_20m_alpha{alpha_token}_beta{beta_token}"
    print(f"SFT interleave launcher: {pretrained_model} -> {model_id}")
    if dry_run:
        print("(dry-run -- nothing launched)")
        return
    handle = sft_one.spawn(pretrained_model=pretrained_model, alpha_token=alpha_token,
                           beta_token=beta_token)
    print(f"SPAWNED (function call id: {handle.object_id})")
