"""Launch the two fresh 680M chess pretraining jobs.

These are random-initialized Qwen3 models trained on the tokenized chess
pretraining corpus. They are not Miles or veRL reinforcement-learning jobs.

Production:
  modal run --detach modal_scripts/launch_fresh_680m_pretrains.py \
    --targets 6p5e18
  modal run --detach modal_scripts/launch_fresh_680m_pretrains.py \
    --targets 6p5e19

One-step smoke (request two H100s through environment variables):
  CHESS_PRETRAIN_GPU_TYPE=H100 CHESS_PRETRAIN_GPUS=2 \
    modal run --detach modal_scripts/launch_fresh_680m_pretrains.py \
    --targets 6p5e18 --smoke
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import modal

GPU_TYPE = os.environ.get("CHESS_PRETRAIN_GPU_TYPE", "H200").strip()
GPUS_PER_NODE = int(os.environ.get("CHESS_PRETRAIN_GPUS", "8"))
TIMEOUT_HOURS = 47
MAX_CHECKPOINTS = 3
MIXED_PRECISION = "bf16"

DATA_DIR = "/data/pretrain_v1_20b"
FRESH_TAG = os.environ.get(
    "CHESS_PRETRAIN_FRESH_TAG", "fresh_pretrain_20260724_r2"
).strip()
OUTPUT_ROOT = f"/checkpoints/{FRESH_TAG}"
SMOKE_OUTPUT_ROOT = f"/checkpoints/{FRESH_TAG}_smoke"

JOBS = {
    "6p5e18": {
        "config": "config/configs/6p5e18_jingyan/680m_alpha1.000.yaml",
        "canonical_name": "680m_C_6p5e18_alpha1.000",
        "historical_hf_repo": "chess-pre-to-post/6p5e18_680m_alpha1.000",
        "output_subdir": "6p5e18_jingyan",
    },
    "6p5e19": {
        "config": "config/configs/6p5e19_leon/680m_alpha0.750.yaml",
        "canonical_name": "680m_C_6p5e19_alpha0.750",
        "historical_hf_repo": (
            "chess-pre-to-post/680m_C_6p5e19_alpha0.750"
        ),
        "output_subdir": "6p5e19_leon",
    },
}

if not re.fullmatch(r"[A-Za-z0-9_.-]+", FRESH_TAG):
    raise ValueError("CHESS_PRETRAIN_FRESH_TAG must be path-safe")
if GPUS_PER_NODE < 1:
    raise ValueError("CHESS_PRETRAIN_GPUS must be positive")

cuda_version = "12.4.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"
repo_dir = Path(__file__).parent.parent


def _load_local_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.removeprefix("export ").strip()
        if key not in {"WANDB_API_KEY"}:
            continue
        value = value.strip().strip('"').strip("'")
        if value:
            values[key] = value
    return values


_local_secret_env = _load_local_env(repo_dir / ".env")
_wandb_api_key = os.environ.get("WANDB_API_KEY") or _local_secret_env.get(
    "WANDB_API_KEY"
)
if _wandb_api_key:
    wandb_secret = modal.Secret.from_dict(
        {
            "WANDB_API_KEY": _wandb_api_key,
            "WANDB_ENTITY": "jingyanshen-new-york-university",
        }
    )
else:
    # This fallback is useful for importing the module in environments without
    # the workspace .env, but production launch validation below fails closed
    # unless a team-authorized key was resolved locally.
    wandb_secret = modal.Secret.from_name("wandb-secret")

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .apt_install("curl", "git", "vim", "htop")
    .pip_install(
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
    .run_commands(
        "python -c \"from transformers import AutoTokenizer; "
        "AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B')\""
    )
    # Modal imports this module inside the worker. Preserve the resource values
    # resolved by the local launcher so worker logs and runtime globals match
    # the GPU request (especially for the reduced two-H100 smoke).
    .env(
        {
            "CHESS_PRETRAIN_GPU_TYPE": GPU_TYPE,
            "CHESS_PRETRAIN_GPUS": str(GPUS_PER_NODE),
            "CHESS_PRETRAIN_FRESH_TAG": FRESH_TAG,
            # Accelerate's current WandB tracker does not reliably forward the
            # entity from our config, so make the intended team explicit.
            "WANDB_ENTITY": "jingyanshen-new-york-university",
        }
    )
    .add_local_dir(str(repo_dir / "scripts"), remote_path="/root/chess/scripts")
    .add_local_dir(str(repo_dir / "training"), remote_path="/root/chess/training")
    .add_local_dir(str(repo_dir / "config"), remote_path="/root/chess/config")
    .add_local_dir(str(repo_dir / "llm_tokens"), remote_path="/root/chess/llm_tokens")
    .add_local_dir(str(repo_dir / "evaluation"), remote_path="/root/chess/evaluation")
)

data_volume = modal.Volume.from_name(
    "rl-reasoning-training-data", create_if_missing=True
)
ckpt_volume = modal.Volume.from_name(
    "rl-reasoning-checkpoints", create_if_missing=True
)

app = modal.App(
    "chess-pretrain-680m-fresh",
    image=image,
    secrets=[
        wandb_secret,
        modal.Secret.from_name("huggingface-secret"),
    ],
    volumes={
        "/data": data_volume,
        "/checkpoints": ckpt_volume,
    },
)


def _experiment_name(target: str, *, smoke: bool) -> str:
    suffix = f"{FRESH_TAG}_smoke1" if smoke else FRESH_TAG
    return f"{JOBS[target]['canonical_name']}_{suffix}"


def _output_dir(target: str, *, smoke: bool) -> str:
    root = SMOKE_OUTPUT_ROOT if smoke else OUTPUT_ROOT
    return f"{root}/{JOBS[target]['output_subdir']}"


def _build_overrides(target: str, *, smoke: bool, num_gpus: int) -> list[str]:
    experiment_name = _experiment_name(target, smoke=smoke)
    historical_hf_repo = JOBS[target]["historical_hf_repo"]
    # Never let a fresh run sync into the historical model repository. Smoke
    # runs do not need a Hub upload; production gets a fresh-tagged repository.
    hf_upload_repo = (
        "null" if smoke else f"{historical_hf_repo}_{FRESH_TAG}"
    )
    overrides = [
        "training.gpu_peak_tflops=989",
        "training.cache_size=0",
        "training.mixed_precision=bf16",
        "training.seed=42",
        f"training.experiment_name={experiment_name}",
        f"training.run_name={experiment_name}",
        f"training.hf_upload_repo={hf_upload_repo}",
    ]
    if smoke:
        # Exactly one optimizer step: bs=1, seq=1024, GA=1, and num_gpus.
        overrides.extend(
            [
                "training.batch_size=1",
                "training.gradient_accumulation_steps=1",
                "training.num_workers=2",
                "training.log_interval=1",
                "training.eval_max_steps=1",
                f"data.pretrain_tokens={1024 * num_gpus}",
            ]
        )
    return overrides


def _overridden_value(
    overrides: list[str], key: str, default: str | None = None
) -> str | None:
    prefix = f"{key}="
    for override in overrides:
        if override.startswith(prefix):
            return override.split("=", 1)[1]
    return default


def _run_training(
    *,
    config: str,
    output_dir: str,
    overrides: list[str],
    num_gpus: int,
) -> None:
    import subprocess
    import sys

    import yaml
    from omegaconf import OmegaConf

    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

    config_path = Path("/root/chess") / config
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    model_cfg = cfg.get("model", {})
    training_cfg = cfg.get("training", {})
    if model_cfg.get("pretrained_model") or training_cfg.get("pretrained_weights"):
        raise RuntimeError(
            f"{config} is not a from-scratch pretraining config; refusing to launch"
        )

    data_files = list(Path(DATA_DIR).glob("*.npy"))
    if not data_files:
        raise FileNotFoundError(f"No pretraining shards found under {DATA_DIR}")

    experiment_name = _overridden_value(
        overrides,
        "training.experiment_name",
        training_cfg.get("experiment_name"),
    )
    if not experiment_name:
        raise RuntimeError(f"No experiment name resolved for {config}")

    run_dir = Path(output_dir) / experiment_name
    final_dir = run_dir / "final"
    latest_dir = run_dir / "latest"
    if final_dir.is_dir():
        print(f"[pretrain] Final already exists; skipping: {final_dir}", flush=True)
        return
    if latest_dir.is_dir():
        print(f"[pretrain] Retrying from fresh-run checkpoint: {latest_dir}", flush=True)
    else:
        print(
            f"[pretrain] Fresh random initialization: {experiment_name}",
            flush=True,
        )

    # HFTrainer snapshots the source YAML, which does not include CLI
    # overrides. Preserve the fully resolved launch config alongside it so the
    # effective batch, token budget, seed, and unique output identity remain
    # auditable.
    effective_cfg = OmegaConf.merge(
        OmegaConf.create(cfg),
        OmegaConf.from_dotlist(overrides),
    )
    historical_hf_repo = training_cfg.get("hf_upload_repo") or cfg.get(
        "logging", {}
    ).get("hf_upload_repo")
    effective_hf_repo = effective_cfg.training.get("hf_upload_repo")
    if effective_hf_repo and effective_hf_repo == historical_hf_repo:
        raise RuntimeError(
            "Fresh pretraining must not upload into the historical HF repo: "
            f"{historical_hf_repo}"
        )
    if effective_hf_repo and FRESH_TAG not in effective_hf_repo:
        raise RuntimeError(
            "Fresh pretraining HF repo is not isolated by the fresh tag: "
            f"{effective_hf_repo}"
        )
    effective_cfg.training.auto_resume = True
    effective_cfg.training.save_dir = output_dir
    effective_cfg.training.max_checkpoints = MAX_CHECKPOINTS
    effective_cfg.data.txt_path = DATA_DIR
    effective_cfg.data.test_data_dir = "/data/test"
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(effective_cfg, run_dir / "effective_config.yaml")

    print(
        f"[pretrain] config={config} GPUs={num_gpus}x{GPU_TYPE} "
        f"shards={len(data_files)} output={run_dir} "
        f"hf_upload_repo={effective_hf_repo or 'disabled'}",
        flush=True,
    )

    cmd = [
        "accelerate",
        "launch",
        "--multi_gpu",
        "--num_processes",
        str(num_gpus),
        "--mixed_precision",
        MIXED_PRECISION,
        "scripts/train/train_hf.py",
        "--config",
        config,
        "--auto_resume",
        "--data_dir",
        DATA_DIR,
        "--output_dir",
        output_dir,
        "--test_data_dir",
        "/data/test",
        "--max_checkpoints",
        str(MAX_CHECKPOINTS),
    ]
    # train_hf.py declares --override with nargs="*". Supplying the option
    # repeatedly makes argparse retain only the final occurrence, so pass the
    # full dot-list after one flag.
    if overrides:
        cmd.append("--override")
        cmd.extend(overrides)

    print("[pretrain] " + " ".join(cmd), flush=True)
    proc = subprocess.run(
        cmd,
        cwd="/root/chess",
        stdout=sys.stdout,
        stderr=sys.stderr,
        check=False,
    )
    ckpt_volume.commit()
    if proc.returncode != 0:
        raise RuntimeError(f"Pretraining failed (exit {proc.returncode}): {config}")
    print(f"[pretrain] Completed: {experiment_name}", flush=True)


@app.function(
    gpu=f"{GPU_TYPE}:{GPUS_PER_NODE}",
    timeout=60 * 60 * TIMEOUT_HOURS,
    retries=modal.Retries(initial_delay=0.0, max_retries=5),
    max_containers=2,
)
def train_production(
    config: str,
    output_dir: str,
    overrides: list[str],
    num_gpus: int,
) -> None:
    _run_training(
        config=config,
        output_dir=output_dir,
        overrides=overrides,
        num_gpus=num_gpus,
    )


@app.function(
    gpu=f"{GPU_TYPE}:{GPUS_PER_NODE}",
    timeout=60 * 60,
    max_containers=1,
)
def train_smoke(
    config: str,
    output_dir: str,
    overrides: list[str],
    num_gpus: int,
) -> None:
    _run_training(
        config=config,
        output_dir=output_dir,
        overrides=overrides,
        num_gpus=num_gpus,
    )


def _choose_targets(targets: str) -> list[str]:
    chosen = [target.strip() for target in targets.split(",") if target.strip()]
    unknown = sorted(set(chosen) - set(JOBS))
    if unknown:
        raise ValueError(f"Unknown target(s): {', '.join(unknown)}")
    if not chosen:
        raise ValueError("At least one target is required")
    return chosen


@app.local_entrypoint()
def main(
    targets: str = "6p5e18,6p5e19",
    smoke: bool = False,
    dry_run: bool = False,
) -> None:
    chosen = _choose_targets(targets)
    if not _wandb_api_key:
        raise RuntimeError(
            "A team-authorized WANDB_API_KEY is required in the environment "
            "or chess_reasoning/.env"
        )
    if smoke and (GPU_TYPE != "H100" or GPUS_PER_NODE != 2):
        raise ValueError(
            "--smoke requires CHESS_PRETRAIN_GPU_TYPE=H100 and "
            "CHESS_PRETRAIN_GPUS=2"
        )
    if not smoke and (GPU_TYPE != "H200" or GPUS_PER_NODE != 8):
        raise ValueError(
            "production requires the default 8xH200 topology; unset "
            "CHESS_PRETRAIN_GPU_TYPE and CHESS_PRETRAIN_GPUS"
        )
    print(
        f"Fresh 680M pretraining: targets={','.join(chosen)} "
        f"GPU={GPUS_PER_NODE}x{GPU_TYPE} smoke={smoke}",
        flush=True,
    )

    handles = []
    for target in chosen:
        job = JOBS[target]
        output_dir = _output_dir(target, smoke=smoke)
        overrides = _build_overrides(
            target,
            smoke=smoke,
            num_gpus=GPUS_PER_NODE,
        )
        print(
            f"  {target}: {job['config']} -> "
            f"{output_dir}/{_experiment_name(target, smoke=smoke)} "
            f"(hf={_overridden_value(overrides, 'training.hf_upload_repo')})",
            flush=True,
        )
        if dry_run:
            continue
        launch_function = train_smoke if smoke else train_production
        handle = launch_function.spawn(
            config=job["config"],
            output_dir=output_dir,
            overrides=overrides,
            num_gpus=GPUS_PER_NODE,
        )
        handles.append((target, handle))
        print(f"  SPAWNED {target}: {handle.object_id}", flush=True)

    if dry_run:
        print("(dry-run -- nothing launched)", flush=True)
    else:
        print(f"{len(handles)} pretraining job(s) spawned.", flush=True)
