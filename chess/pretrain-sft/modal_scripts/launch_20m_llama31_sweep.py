"""Launch the controlled ~20M Llama-3.1 five-alpha pretraining sweep.

The jobs use official Hugging Face ``LlamaConfig`` semantics with a scaled
Llama-3.1-8B core. They are isolated from the existing Qwen sweep by config
directory, experiment name, checkpoint root, W&B project, and Hub repo.

Full sweep (run each alpha as an independent detached 8xH200 app):

  for alpha in 0.050 0.100 0.200 0.400 0.750; do
    modal run --detach modal_scripts/launch_20m_llama31_sweep.py \
      --alphas "$alpha"
  done

Select a subset:

  modal run --detach modal_scripts/launch_20m_llama31_sweep.py \
    --alphas 0.050,0.200

One-step smoke on two H100s (Hub upload disabled):

  CHESS_LLAMA31_GPU_TYPE=H100 CHESS_LLAMA31_GPUS=2 \
    modal run --detach modal_scripts/launch_20m_llama31_sweep.py \
    --alphas 0.050 --smoke

Local launch-plan validation only:

  modal run modal_scripts/launch_20m_llama31_sweep.py --dry-run
"""

from __future__ import annotations

import os
import json
import time
from pathlib import Path

import modal

GPU_TYPE = os.environ.get("CHESS_LLAMA31_GPU_TYPE", "H200").strip()
GPUS_PER_NODE = int(os.environ.get("CHESS_LLAMA31_GPUS", "8"))
TIMEOUT_HOURS = 47
MAX_CHECKPOINTS = 3
MIXED_PRECISION = "bf16"

DATA_DIR = "/data/pretrain_v1_20b"
OUTPUT_ROOT = "/checkpoints/6p5e18_llama31"
SMOKE_OUTPUT_ROOT = "/checkpoints/6p5e18_llama31_smoke"
WANDB_PROJECT = "chess-scaling-C_6p5e18-llama31"
SMOKE_WANDB_PROJECT = "chess-scaling-C_6p5e18-llama31-smoke"
WANDB_ENTITY = "jingyanshen-new-york-university"
HF_PREFIX = "Pre-to-Post-2/pretrain_20m_llama31_C_6p5e18_alpha"

ALPHAS = ("0.050", "0.100", "0.200", "0.400", "0.750")
TOKEN_BUDGETS = {
    "0.050": 2_641_541_000,
    "0.100": 5_284_229_000,
    "0.200": 10_569_605_000,
    "0.400": 21_139_210_000,
    "0.750": 39_636_879_000,
}
JOBS = {
    alpha: {
        "config": f"config/configs/6p5e18_llama31/20m_alpha{alpha}.yaml",
        "experiment_name": f"20m_llama31_C_6p5e18_alpha{alpha}",
        "hf_upload_repo": f"{HF_PREFIX}{alpha}",
        "pretrain_tokens": TOKEN_BUDGETS[alpha],
    }
    for alpha in ALPHAS
}

if GPUS_PER_NODE < 1:
    raise ValueError("CHESS_LLAMA31_GPUS must be positive")

cuda_version = "12.4.0"
tag = f"{cuda_version}-devel-ubuntu22.04"
repo_dir = Path(__file__).parent.parent


def _load_local_wandb_key(path: Path) -> str | None:
    """Read only WANDB_API_KEY without copying the workspace .env into Modal."""
    if not path.is_file():
        return None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.removeprefix("export ").strip()
        if key == "WANDB_API_KEY":
            value = value.strip().strip('"').strip("'")
            return value or None
    return None


_wandb_api_key = os.environ.get("WANDB_API_KEY") or _load_local_wandb_key(
    repo_dir / ".env"
)
if _wandb_api_key:
    wandb_secret = modal.Secret.from_dict(
        {
            "WANDB_API_KEY": _wandb_api_key,
            "WANDB_ENTITY": WANDB_ENTITY,
        }
    )
else:
    wandb_secret = modal.Secret.from_name("wandb-secret")

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .apt_install("curl", "git", "vim", "htop")
    .pip_install(
        # Torch 2.6.0's Linux wheel carries the CUDA 12.4 runtime used by the
        # base image. A floating lower bound currently resolves to CUDA 13.
        "torch==2.6.0",
        "accelerate==1.13.0",
        # Pin the tested LlamaConfig/RoPE contract instead of floating across
        # incompatible Transformers releases during a long sweep.
        "transformers==5.3.0",
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
        "python -c \"from transformers import LlamaConfig, "
        "LlamaForCausalLM; LlamaForCausalLM(LlamaConfig("
        "vocab_size=81, hidden_size=64, intermediate_size=128, "
        "num_hidden_layers=1, num_attention_heads=1))\""
    )
    .env(
        {
            "CHESS_LLAMA31_GPU_TYPE": GPU_TYPE,
            "CHESS_LLAMA31_GPUS": str(GPUS_PER_NODE),
            "WANDB_ENTITY": WANDB_ENTITY,
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
    "chess-pretrain-20m-llama31",
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


def _choose_alphas(alphas: str) -> list[str]:
    if alphas.strip().lower() == "all":
        return list(ALPHAS)
    chosen = [alpha.strip() for alpha in alphas.split(",") if alpha.strip()]
    unknown = sorted(set(chosen) - set(ALPHAS))
    if unknown:
        raise ValueError(
            f"Unknown alpha(s): {', '.join(unknown)}; choose from {', '.join(ALPHAS)}"
        )
    if not chosen:
        raise ValueError("At least one alpha is required")
    if len(set(chosen)) != len(chosen):
        raise ValueError("Duplicate alphas are not allowed")
    return chosen


def _experiment_name(alpha: str, *, smoke: bool) -> str:
    base = JOBS[alpha]["experiment_name"]
    return f"{base}_smoke1" if smoke else base


def _output_root(*, smoke: bool) -> str:
    return SMOKE_OUTPUT_ROOT if smoke else OUTPUT_ROOT


def _build_overrides(alpha: str, *, smoke: bool, num_gpus: int) -> list[str]:
    experiment_name = _experiment_name(alpha, smoke=smoke)
    project = SMOKE_WANDB_PROJECT if smoke else WANDB_PROJECT
    hf_upload_repo = "null" if smoke else JOBS[alpha]["hf_upload_repo"]
    overrides = [
        "training.gpu_peak_tflops=989",
        "training.cache_size=0",
        "training.mixed_precision=bf16",
        "training.seed=42",
        f"training.experiment_name={experiment_name}",
        f"training.run_name={experiment_name}",
        f"training.hf_upload_repo={hf_upload_repo}",
        f"logging.project={project}",
        f"logging.entity={WANDB_ENTITY}",
    ]
    if smoke:
        # One global optimizer step on the requested topology.
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


def _build_training_command(
    *,
    config: str,
    output_root: str,
    overrides: list[str],
    num_gpus: int,
) -> list[str]:
    command = [
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
        output_root,
        "--test_data_dir",
        "/data/test",
        "--max_checkpoints",
        str(MAX_CHECKPOINTS),
    ]
    # argparse declares --override with nargs="*": it must appear exactly once.
    if overrides:
        command.append("--override")
        command.extend(overrides)
    return command


def _validate_legacy_rope_aliases(
    config_dict: dict,
    *,
    source: str,
) -> None:
    """Fail closed unless a Llama config has equivalent v5 and v4 RoPE fields."""
    if config_dict.get("model_type") != "llama":
        raise ValueError(f"{source}: expected model_type='llama'")
    rope_parameters = config_dict.get("rope_parameters")
    if not isinstance(rope_parameters, dict):
        raise ValueError(f"{source}: missing Transformers-5 rope_parameters")
    if "rope_type" not in rope_parameters or "rope_theta" not in rope_parameters:
        raise ValueError(
            f"{source}: rope_parameters must include rope_type and rope_theta"
        )

    expected_theta = float(rope_parameters["rope_theta"])
    if config_dict.get("rope_theta") != expected_theta:
        raise ValueError(
            f"{source}: legacy rope_theta does not match rope_parameters"
        )
    expected_scaling = {
        key: value
        for key, value in rope_parameters.items()
        if key != "rope_theta"
    }
    if config_dict.get("rope_scaling") != expected_scaling:
        raise ValueError(
            f"{source}: legacy rope_scaling does not match rope_parameters"
        )


def _patch_rope_config_file(config_path: Path) -> bool:
    """Atomically add Transformers-4 RoPE aliases to one saved Llama config."""
    import sys

    remote_repo = Path("/root/chess")
    if remote_repo.is_dir() and str(remote_repo) not in sys.path:
        sys.path.insert(0, str(remote_repo))
    from training.trainer_hf import _add_legacy_llama_rope_aliases

    with config_path.open("r", encoding="utf-8") as handle:
        config_dict = json.load(handle)
    original = json.dumps(config_dict, sort_keys=True, separators=(",", ":"))
    _add_legacy_llama_rope_aliases(config_dict)
    _validate_legacy_rope_aliases(config_dict, source=str(config_path))
    updated = json.dumps(config_dict, sort_keys=True, separators=(",", ":"))
    if updated == original:
        return False

    temporary_path = config_path.with_name(
        f".{config_path.name}.rope-alias-repair-{os.getpid()}"
    )
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(config_dict, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, config_path)
    return True


def _patch_run_rope_configs(run_dir: Path) -> tuple[list[Path], int]:
    """Patch every saved HF model config in a completed Llama run."""
    config_paths = sorted(run_dir.rglob("config.json"))
    final_config = run_dir / "final" / "config.json"
    if final_config not in config_paths:
        raise FileNotFoundError(f"Final config is missing from {run_dir}")

    changed = 0
    for config_path in config_paths:
        changed += int(_patch_rope_config_file(config_path))
    return config_paths, changed


def _wait_for_completed_run(
    *,
    alpha: str,
    deadline: float,
    poll_seconds: int,
) -> Path:
    """Wait until a trainer has committed its complete final artifact."""
    run_dir = Path(OUTPUT_ROOT) / _experiment_name(alpha, smoke=False)
    required_final_files = (
        run_dir / "final" / "config.json",
        run_dir / "final" / "model.safetensors",
        run_dir / "final" / "generation_config.json",
    )
    while True:
        ckpt_volume.reload()
        missing = [path.name for path in required_final_files if not path.is_file()]
        if not missing:
            return run_dir
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Timed out waiting for alpha={alpha}; missing final files: "
                f"{', '.join(missing)}"
            )
        print(
            f"[rope-repair] alpha={alpha} still training; "
            f"waiting for committed final ({', '.join(missing)})",
            flush=True,
        )
        time.sleep(poll_seconds)


def _upload_and_verify_rope_configs(
    *,
    alpha: str,
    run_dir: Path,
    config_paths: list[Path],
) -> None:
    """Upload repaired configs, recovering a missing final artifact if necessary."""
    import tempfile

    from huggingface_hub import HfApi, hf_hub_download

    repo_id = JOBS[alpha]["hf_upload_repo"]
    api = HfApi()
    api.create_repo(repo_id, exist_ok=True, repo_type="model")
    remote_files = set(api.list_repo_files(repo_id=repo_id, repo_type="model"))
    required_remote_files = {
        "final/config.json",
        "final/model.safetensors",
        "final/generation_config.json",
    }
    if not required_remote_files.issubset(remote_files):
        print(
            f"[rope-repair] alpha={alpha} remote final is incomplete; "
            "recovering the full committed run",
            flush=True,
        )
        api.upload_folder(
            folder_path=str(run_dir),
            repo_id=repo_id,
            repo_type="model",
            path_in_repo="",
            commit_message=(
                f"Recover completed Llama-3.1 alpha={alpha} run and RoPE aliases"
            ),
        )

    relative_configs = [path.relative_to(run_dir).as_posix() for path in config_paths]
    api.upload_folder(
        folder_path=str(run_dir),
        repo_id=repo_id,
        repo_type="model",
        path_in_repo="",
        allow_patterns=relative_configs,
        commit_message=(
            f"Add Transformers-4-compatible RoPE aliases for alpha={alpha}"
        ),
    )

    remote_files = set(api.list_repo_files(repo_id=repo_id, repo_type="model"))
    expected_files = required_remote_files | set(relative_configs)
    missing_remote = sorted(expected_files - remote_files)
    if missing_remote:
        raise RuntimeError(
            f"{repo_id}: upload verification found missing files: "
            f"{', '.join(missing_remote)}"
        )

    with tempfile.TemporaryDirectory(prefix=f"rope-repair-{alpha}-") as cache_dir:
        for relative_path in relative_configs:
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                repo_type="model",
                filename=relative_path,
                cache_dir=cache_dir,
                force_download=True,
            )
            with open(downloaded_path, "r", encoding="utf-8") as handle:
                remote_config = json.load(handle)
            _validate_legacy_rope_aliases(
                remote_config,
                source=f"{repo_id}/{relative_path}",
            )

    print(
        f"[rope-repair] VERIFIED alpha={alpha}: "
        f"{len(relative_configs)} config(s), including final/config.json -> "
        f"https://huggingface.co/{repo_id}",
        flush=True,
    )


def _run_training(
    *,
    alpha: str,
    config: str,
    output_root: str,
    overrides: list[str],
    num_gpus: int,
    smoke: bool,
) -> None:
    import subprocess
    import sys

    import yaml
    from omegaconf import OmegaConf

    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

    config_path = Path("/root/chess") / config
    with config_path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle)

    model_config = raw_config.get("model", {})
    training_config = raw_config.get("training", {})
    if str(model_config.get("model_family", "")).lower() != "llama":
        raise RuntimeError(f"{config} is not an explicit Llama-family config")
    if model_config.get("pretrained_model") or training_config.get(
        "pretrained_weights"
    ):
        raise RuntimeError(f"{config} is not a random-init pretraining config")

    data_files = list(Path(DATA_DIR).glob("*.npy"))
    if not data_files:
        raise FileNotFoundError(f"No pretraining shards found under {DATA_DIR}")

    effective_config = OmegaConf.merge(
        OmegaConf.create(raw_config),
        OmegaConf.from_dotlist(overrides),
    )
    experiment_name = str(effective_config.training.experiment_name)
    expected_name = _experiment_name(alpha, smoke=smoke)
    if experiment_name != expected_name:
        raise RuntimeError(
            f"Resolved experiment name {experiment_name!r} != {expected_name!r}"
        )

    hf_upload_repo = effective_config.training.get("hf_upload_repo", None)
    if smoke:
        if hf_upload_repo is not None:
            raise RuntimeError("Smoke runs must have Hugging Face upload disabled")
    elif hf_upload_repo != JOBS[alpha]["hf_upload_repo"]:
        raise RuntimeError(
            "Full run resolved an unexpected Hugging Face repo: "
            f"{hf_upload_repo!r}"
        )

    run_dir = Path(output_root) / experiment_name
    final_dir = run_dir / "final"
    if final_dir.is_dir():
        if not smoke:
            # A previous attempt may have saved final/ and then failed during
            # the synchronous Hub upload. Modal retries must repair the remote
            # artifact instead of treating the local directory as completion.
            from huggingface_hub import HfApi

            api = HfApi()
            api.create_repo(hf_upload_repo, exist_ok=True, repo_type="model")
            api.upload_folder(
                folder_path=str(run_dir),
                repo_id=hf_upload_repo,
                path_in_repo="",
                commit_message="Retry sync of completed Llama-3.1 run",
            )
            print(
                f"[llama31] Re-synced completed run -> {hf_upload_repo}",
                flush=True,
            )
        print(f"[llama31] Final already exists; skipping: {final_dir}", flush=True)
        return

    effective_config.training.auto_resume = True
    effective_config.training.save_dir = output_root
    effective_config.training.max_checkpoints = MAX_CHECKPOINTS
    effective_config.data.txt_path = DATA_DIR
    effective_config.data.test_data_dir = "/data/test"
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(effective_config, run_dir / "effective_config.yaml")

    print(
        f"[llama31] alpha={alpha} config={config} GPUs={num_gpus}x{GPU_TYPE} "
        f"shards={len(data_files)} output={run_dir} "
        f"hf_upload_repo={hf_upload_repo or 'disabled'}",
        flush=True,
    )
    command = _build_training_command(
        config=config,
        output_root=output_root,
        overrides=overrides,
        num_gpus=num_gpus,
    )
    print("[llama31] " + " ".join(command), flush=True)
    process = subprocess.run(
        command,
        cwd="/root/chess",
        stdout=sys.stdout,
        stderr=sys.stderr,
        check=False,
    )
    ckpt_volume.commit()
    if process.returncode != 0:
        raise RuntimeError(
            f"Llama-3.1 pretraining failed (exit {process.returncode}): {config}"
        )
    print(f"[llama31] Completed: {experiment_name}", flush=True)


@app.function(
    gpu=f"{GPU_TYPE}:{GPUS_PER_NODE}",
    timeout=60 * 60 * TIMEOUT_HOURS,
    retries=modal.Retries(initial_delay=0.0, max_retries=5),
    max_containers=len(ALPHAS),
)
def train_production(
    alpha: str,
    config: str,
    output_root: str,
    overrides: list[str],
    num_gpus: int,
) -> None:
    _run_training(
        alpha=alpha,
        config=config,
        output_root=output_root,
        overrides=overrides,
        num_gpus=num_gpus,
        smoke=False,
    )


@app.function(
    gpu=f"{GPU_TYPE}:{GPUS_PER_NODE}",
    timeout=60 * 60,
    max_containers=len(ALPHAS),
)
def train_smoke(
    alpha: str,
    config: str,
    output_root: str,
    overrides: list[str],
    num_gpus: int,
) -> None:
    _run_training(
        alpha=alpha,
        config=config,
        output_root=output_root,
        overrides=overrides,
        num_gpus=num_gpus,
        smoke=True,
    )


@app.function(
    cpu=2,
    memory=4096,
    timeout=60 * 60 * 12,
    retries=modal.Retries(initial_delay=5.0, backoff_coefficient=2.0, max_retries=3),
)
def repair_legacy_rope_configs(
    alphas: list[str],
    poll_seconds: int,
    max_wait_seconds: int,
) -> None:
    """Repair completed runs now and remain alive for still-running sweeps."""
    if not alphas:
        raise ValueError("At least one alpha is required")
    if poll_seconds < 5:
        raise ValueError("poll_seconds must be at least 5")
    if max_wait_seconds <= 0:
        raise ValueError("max_wait_seconds must be positive")

    deadline = time.monotonic() + max_wait_seconds
    for alpha in alphas:
        if alpha not in JOBS:
            raise ValueError(f"Unknown alpha: {alpha}")
        run_dir = _wait_for_completed_run(
            alpha=alpha,
            deadline=deadline,
            poll_seconds=poll_seconds,
        )
        config_paths, changed = _patch_run_rope_configs(run_dir)
        ckpt_volume.commit()
        print(
            f"[rope-repair] alpha={alpha} committed {changed} changed "
            f"config(s) out of {len(config_paths)}",
            flush=True,
        )
        _upload_and_verify_rope_configs(
            alpha=alpha,
            run_dir=run_dir,
            config_paths=config_paths,
        )
    print("[rope-repair] all requested Llama runs are verified", flush=True)


@app.local_entrypoint()
def main(
    alphas: str = "all",
    smoke: bool = False,
    dry_run: bool = False,
    repair_only: bool = False,
    repair_poll_seconds: int = 30,
    repair_timeout_hours: int = 10,
) -> None:
    chosen = _choose_alphas(alphas)
    if repair_only:
        if smoke:
            raise ValueError("--repair-only cannot be combined with --smoke")
        if repair_poll_seconds < 5:
            raise ValueError("--repair-poll-seconds must be at least 5")
        if repair_timeout_hours <= 0:
            raise ValueError("--repair-timeout-hours must be positive")
        print(
            "Llama RoPE compatibility watcher: "
            f"alphas={','.join(chosen)} poll={repair_poll_seconds}s "
            f"timeout={repair_timeout_hours}h",
            flush=True,
        )
        if dry_run:
            print("(dry-run -- watcher not launched)", flush=True)
            return
        handle = repair_legacy_rope_configs.spawn(
            alphas=chosen,
            poll_seconds=repair_poll_seconds,
            max_wait_seconds=repair_timeout_hours * 60 * 60,
        )
        print(f"  SPAWNED rope-repair watcher: {handle.object_id}", flush=True)
        return

    if smoke and (GPU_TYPE != "H100" or GPUS_PER_NODE != 2):
        raise ValueError(
            "--smoke requires CHESS_LLAMA31_GPU_TYPE=H100 and "
            "CHESS_LLAMA31_GPUS=2"
        )
    if not smoke and (GPU_TYPE != "H200" or GPUS_PER_NODE != 8):
        raise ValueError(
            "Full runs require 8xH200; unset CHESS_LLAMA31_GPU_TYPE and "
            "CHESS_LLAMA31_GPUS"
        )

    print(
        f"Controlled Llama-3.1 pretraining: alphas={','.join(chosen)} "
        f"GPU={GPUS_PER_NODE}x{GPU_TYPE} smoke={smoke}",
        flush=True,
    )
    output_root = _output_root(smoke=smoke)
    plans = []
    for alpha in chosen:
        job = JOBS[alpha]
        overrides = _build_overrides(
            alpha,
            smoke=smoke,
            num_gpus=GPUS_PER_NODE,
        )
        command = _build_training_command(
            config=job["config"],
            output_root=output_root,
            overrides=overrides,
            num_gpus=GPUS_PER_NODE,
        )
        print(
            f"  alpha={alpha} tokens={job['pretrain_tokens']:,} "
            f"run={_experiment_name(alpha, smoke=smoke)} "
            f"hf={'disabled' if smoke else job['hf_upload_repo']}",
            flush=True,
        )
        print("    " + " ".join(command), flush=True)
        plans.append((alpha, job, overrides))

    if dry_run:
        print("(dry-run -- nothing launched)", flush=True)
        return

    launch_function = train_smoke if smoke else train_production
    for alpha, job, overrides in plans:
        handle = launch_function.spawn(
            alpha=alpha,
            config=job["config"],
            output_root=output_root,
            overrides=overrides,
            num_gpus=GPUS_PER_NODE,
        )
        print(f"  SPAWNED alpha={alpha}: {handle.object_id}", flush=True)
