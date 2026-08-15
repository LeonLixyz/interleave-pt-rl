from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import modal

PROJECT_DIR = "/root/chess-rl-miles"
# The pinned runtime image contains its own Miles checkout at /root/miles.
# Modal local-directory mounts overlay that tree instead of deleting stale
# paths, which can leave baked packages shadowing files from our checkout
# (notably miles/ray/rollout/ shadowing miles/ray/rollout.py). Mount the local
# checkout at an image-empty path so imports are an exact view of this source.
MILES_DIR = "/opt/chess-rl-local/miles"
SFT_DIR = "/sft"
DATA_ROOT = "/data"
DATA_DIR = f"{DATA_ROOT}/chess-rl-data"
CKPT_DIR = "/checkpoints"
# The pinned Miles image already contains files under /root/.cache/huggingface,
# and Modal volumes may only be mounted on an empty image path.
HF_CACHE_DIR = "/hf-cache"


def _find_project_dir() -> Path:
    current = Path(__file__).resolve()
    for parent in (current.parent, *current.parents):
        if (parent / "chess_rl_miles").is_dir() and (parent / "pyproject.toml").exists():
            return parent
    remote = Path(PROJECT_DIR)
    if remote.exists():
        return remote
    return current.parent


PROJECT_LOCAL = _find_project_dir()
WORKSPACE_LOCAL = PROJECT_LOCAL.parent


def _find_miles_dir() -> Path:
    """Find Miles in either the legacy sibling or clean monorepo layout."""

    candidates = [WORKSPACE_LOCAL / "miles"]
    candidates.extend(parent / "miles" for parent in PROJECT_LOCAL.parents)
    for candidate in candidates:
        if (candidate / "miles").is_dir() and (
            candidate / "pyproject.toml"
        ).is_file():
            return candidate
    remote = Path(MILES_DIR)
    if remote.is_dir():
        return remote
    raise FileNotFoundError(
        "Could not locate the Miles source tree in the legacy sibling or "
        "clean monorepo layout."
    )


MILES_LOCAL = _find_miles_dir()
if str(PROJECT_LOCAL) not in sys.path:
    sys.path.insert(0, str(PROJECT_LOCAL))


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export ") :].strip()
        if key not in {"WANDB_API_KEY", "WANDB_ENTITY"}:
            continue
        value = value.strip().strip('"').strip("'")
        if value and not os.environ.get(key):
            os.environ[key] = value


_load_env_file(WORKSPACE_LOCAL.parent / ".env")

from chess_rl_miles.data import (
    COT_TYPE,
    DEFAULT_TRAIN_FILE,
    DEFAULT_TRAIN_FILE_SHA256,
    ensure_chess_data,
    ensure_sft_model,
    model_id_from_spec,
)
from chess_rl_miles.scripts.run_chess_miles import SPECS
from chess_rl_miles.provenance import source_path_is_excluded

DEFAULT_MODAL_TRAIN_FILE = f"{DATA_DIR}/{DEFAULT_TRAIN_FILE}"

# Keep H200 as the production default while allowing a compatible H100 smoke
# request when tightly constrained 8-GPU H200 capacity is unavailable.
GPU_TYPE = os.environ.get("CHESS_RL_MILES_GPU_TYPE", "H200").strip()
GPUS_PER_NODE = int(os.environ.get("CHESS_RL_MILES_GPUS_PER_NODE", "8"))
# Prior 680M runs peaked around 140 GB of host memory and used well below this
# CPU allocation. Keeping headroom without requesting an unusually large host
# materially widens Modal's eligible 8xH200 worker pool.
CPU_COUNT = float(os.environ.get("CHESS_RL_MILES_CPU_COUNT", "48"))
MEMORY_GB = int(os.environ.get("CHESS_RL_MILES_MEMORY_GB", "384"))
MEMORY_MB = MEMORY_GB * 1024
TIMEOUT_HOURS = 24
# SGLang/rollout crashes can kill long runs. For new paths, --resume-if-available
# starts from SFT once and lets later Modal retries use the newest checkpoint.
RETRIES = 10

DEFAULT_RUN_SUFFIX_BY_SPEC = {
    "6p5e18|680m|1.000|0.296": "miles_cispo_minimax_jingyan_from_verl_step60",
    "6p5e19|680m|0.750|0.030": "miles_cispo_minimax_jingyan_from_verl_step280",
}

# Pin the exact image used by the 2026-07-22 fast-rollout canary. In particular,
# the token-only heterogeneous batch API is tied to this SGLang build. Override
# CHESS_RL_MILES_IMAGE deliberately when validating a newer image.
MILES_IMAGE = os.environ.get(
    "CHESS_RL_MILES_IMAGE",
    "radixark/miles@sha256:5b41bff2ecd42f1e71b5d8658e777541a821ef96556ae06b48333d521e0ca25e",
)
RAY_ADDRESS = "127.0.0.1:6379"
RAY_AGENT_LISTEN_PORT = 52365
RAY_AGENT_GRPC_PORT = 52366
RAY_RUNTIME_ENV_AGENT_PORT = 52367
RAY_METRICS_EXPORT_PORT = 52368

runtime_secrets = [modal.Secret.from_name("huggingface-secret")]
_secret_env = {k: v for k in ("WANDB_API_KEY", "WANDB_ENTITY") if (v := os.environ.get(k))}
if _secret_env:
    runtime_secrets.append(modal.Secret.from_dict(_secret_env))


def _ignore_modal_source_path(path: Path) -> bool:
    """Use exactly the same inventory policy as source provenance."""

    return source_path_is_excluded(Path(path))

image = (
    modal.Image.from_registry(MILES_IMAGE)
    .apt_install("git", "curl", "rsync", "htop")
    .pip_install("chess==1.11.2")
    # Modal imports this module again inside the container. Persist the values
    # resolved by the local launcher so Ray advertises the resources that the
    # function actually requested (especially for reduced-resource smokes).
    .env(
        {
            "CHESS_RL_MILES_GPU_TYPE": GPU_TYPE,
            "CHESS_RL_MILES_GPUS_PER_NODE": str(GPUS_PER_NODE),
            "CHESS_RL_MILES_CPU_COUNT": str(CPU_COUNT),
            "CHESS_RL_MILES_MEMORY_GB": str(MEMORY_GB),
        }
    )
    .add_local_dir(
        str(MILES_LOCAL),
        remote_path=MILES_DIR,
        ignore=_ignore_modal_source_path,
    )
    .add_local_dir(
        str(PROJECT_LOCAL),
        remote_path=PROJECT_DIR,
        ignore=_ignore_modal_source_path,
    )
)

sft_vol = modal.Volume.from_name("chess-rl-miles-sft", create_if_missing=True)
data_vol = modal.Volume.from_name("chess-rl-miles-data", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("chess-rl-miles-checkpoints", create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

app = modal.App("train rl", image=image, secrets=runtime_secrets)


def _choose_specs(specs: str = "") -> list[str]:
    chosen = [s.strip() for s in specs.split(",") if s.strip()] if specs else list(SPECS)
    unknown = sorted(set(chosen) - set(SPECS))
    if unknown:
        raise ValueError(f"Unknown spec(s): {', '.join(unknown)}")
    return chosen


def _tag_float(value: float | str) -> str:
    text = f"{value:g}" if isinstance(value, float) else str(value)
    return text.replace("e-0", "e-").replace("e+0", "e+")


def _default_hparam_tag(
    *,
    cispo: bool,
    hparam_tag_suffix: str,
    optim_tag: str,
    global_batch_size: int,
    lr: float = 1e-5,
    kl_loss_coef: float = 0.001,
    rollout_max_response_len: int = 2560,
    advantage_estimator: str = "grpo",
) -> str:
    suffixes = [optim_tag, "cispo" if cispo else advantage_estimator, "miles"]
    if hparam_tag_suffix:
        suffixes.append(hparam_tag_suffix.strip("_"))
    return (
        f"multi_turn_lr{_tag_float(lr)}"
        f"_bs{global_batch_size}"
        f"_kl{_tag_float(kl_loss_coef)}"
        f"_res{rollout_max_response_len}"
        f"_{'_'.join(s for s in suffixes if s)}"
    )


def _validate_resume_modes(
    *,
    resume_from_save: bool,
    resume_if_available: bool,
    resume_ckpt_step: int,
) -> None:
    if resume_from_save and resume_if_available:
        raise ValueError("--resume-from-save and --resume-if-available are mutually exclusive")
    if resume_ckpt_step < 0:
        raise ValueError("--resume-ckpt-step must be non-negative")
    if resume_ckpt_step > 0 and not (resume_from_save or resume_if_available):
        raise ValueError(
            "--resume-ckpt-step requires --resume-from-save or --resume-if-available"
        )


def _resolve_resume_checkpoint(
    run_save_path: str | Path,
    *,
    resume_from_save: bool,
    resume_if_available: bool,
    resume_ckpt_step: int = 0,
) -> tuple[Path | None, int | None]:
    """Resolve a validated checkpoint, or select an intentional fresh start."""
    _validate_resume_modes(
        resume_from_save=resume_from_save,
        resume_if_available=resume_if_available,
        resume_ckpt_step=resume_ckpt_step,
    )
    if not (resume_from_save or resume_if_available):
        return None, None

    save_path = Path(run_save_path)
    tracker_path = save_path / "latest_checkpointed_iteration.txt"
    if not tracker_path.is_file():
        if resume_from_save or resume_ckpt_step > 0:
            raise FileNotFoundError(
                "Resume requested but the checkpoint tracker is missing: "
                f"{tracker_path}. Use --resume-if-available only for a unique "
                "fresh-run path, or restore the checkpoint before launching."
            )
        return None, None

    try:
        latest_step = int(tracker_path.read_text().strip())
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Invalid checkpoint tracker: {tracker_path}") from exc
    if latest_step <= 0:
        raise RuntimeError(
            f"Invalid checkpoint step {latest_step} in tracker: {tracker_path}"
        )

    selected_step = resume_ckpt_step or latest_step
    checkpoint_path = save_path / f"iter_{selected_step:07d}"
    model_path = checkpoint_path / "model"
    if not model_path.is_dir():
        raise FileNotFoundError(
            f"Checkpoint step {selected_step} is incomplete under {save_path}: "
            f"missing model directory {model_path}"
        )
    return save_path, selected_step


def _cleanup_runtime() -> None:
    subprocess.run(
        [
            "bash",
            "-lc",
            (
                "pkill -9 sglang || true; "
                "ray stop --force || true; "
                "pkill -9 ray || true; "
                "pkill -9 miles || true; "
                "pkill -9 redis || true"
            ),
        ],
        check=False,
    )


def _print_command(cmd: list[str], *, env: dict[str, str] | None = None) -> int:
    print("[diag] " + " ".join(cmd), flush=True)
    return subprocess.call(cmd, env=env)


def _tail_file(path: Path, max_bytes: int = 32768) -> None:
    try:
        data = path.read_bytes()
    except OSError as exc:
        print(f"[diag] cannot read {path}: {exc}", flush=True)
        return
    text = data[-max_bytes:].decode("utf-8", errors="replace")
    print(f"\n===== {path} =====\n{text}\n===== end {path} =====", flush=True)


def _dump_ray_logs() -> None:
    roots = [Path("/tmp/ray/session_latest/logs"), Path("/tmp/ray")]
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            print(f"[diag] missing {root}", flush=True)
            continue
        for pattern in (
            "gcs_server.*",
            "raylet.*",
            "dashboard.*",
            "dashboard_agent.*",
            "runtime_env_agent.*",
            "monitor.*",
            "log_monitor.*",
        ):
            for path in sorted(root.glob(pattern)):
                if path.is_file() and path not in seen:
                    seen.add(path)
                    _tail_file(path)


def _start_ray_head(env: dict[str, str], *, cpu_threads: int, num_gpus: int = GPUS_PER_NODE) -> None:
    cmd = [
        "ray",
        "start",
        "--head",
        "--node-ip-address",
        "127.0.0.1",
        "--port",
        RAY_ADDRESS.rsplit(":", 1)[1],
        "--num-cpus",
        str(cpu_threads),
        "--num-gpus",
        str(num_gpus),
        "--disable-usage-stats",
        "--include-dashboard=false",
        "--dashboard-agent-listen-port",
        str(RAY_AGENT_LISTEN_PORT),
        "--dashboard-agent-grpc-port",
        str(RAY_AGENT_GRPC_PORT),
        "--runtime-env-agent-port",
        str(RAY_RUNTIME_ENV_AGENT_PORT),
        "--metrics-export-port",
        str(RAY_METRICS_EXPORT_PORT),
    ]
    rc = _print_command(cmd, env=env)
    if rc != 0:
        time.sleep(2)
        _dump_ray_logs()
        raise RuntimeError(f"ray start failed: {rc}")


@app.function(
    gpu=f"{GPU_TYPE}:{GPUS_PER_NODE}",
    cpu=CPU_COUNT,
    memory=MEMORY_MB,
    timeout=60 * 20,
    volumes={HF_CACHE_DIR: hf_cache},
)
def diagnose_ray() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{PROJECT_DIR}:{MILES_DIR}:{env.get('PYTHONPATH', '')}"
    env["PYTHONUNBUFFERED"] = "1"
    env["RAY_DEDUP_LOGS"] = "0"
    print(f"[diag] os.cpu_count={os.cpu_count()} CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    _print_command(["bash", "-lc", "nvidia-smi -L || true"], env=env)
    _print_command(["bash", "-lc", "df -h /tmp /dev/shm || true"], env=env)
    _print_command(["bash", "-lc", "ulimit -a"], env=env)
    _cleanup_runtime()

    _start_ray_head(env, cpu_threads=64)
    env["RAY_ADDRESS"] = RAY_ADDRESS
    rc = _print_command(
        [
            sys.executable,
            "-c",
            (
                "import os, ray; "
                "print('ray', ray.__version__, 'pid', os.getpid(), flush=True); "
                "ray.init(address=os.environ['RAY_ADDRESS'], ignore_reinit_error=True); "
                "print('resources', ray.cluster_resources(), flush=True); "
                "ray.shutdown()"
            ),
        ],
        env=env,
    )
    print(f"[diag] ray.init rc={rc}", flush=True)
    time.sleep(2)
    _dump_ray_logs()
    if rc != 0:
        raise RuntimeError(f"ray diagnostic failed: {rc}")


@app.function(
    cpu=8.0,
    memory=64 * 1024,
    timeout=60 * 10,
)
def diagnose_ray_cpu() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{PROJECT_DIR}:{MILES_DIR}:{env.get('PYTHONPATH', '')}"
    env["PYTHONUNBUFFERED"] = "1"
    env["RAY_DEDUP_LOGS"] = "0"
    print(f"[diag-cpu] os.cpu_count={os.cpu_count()}", flush=True)
    _cleanup_runtime()

    _start_ray_head(env, cpu_threads=8, num_gpus=0)
    env["RAY_ADDRESS"] = RAY_ADDRESS
    rc = _print_command(
        [
            sys.executable,
            "-c",
            (
                "import os, ray; "
                "print('ray', ray.__version__, 'pid', os.getpid(), flush=True); "
                "ray.init(address=os.environ['RAY_ADDRESS'], ignore_reinit_error=True); "
                "print('resources', ray.cluster_resources(), flush=True); "
                "ray.shutdown()"
            ),
        ],
        env=env,
    )
    print(f"[diag-cpu] ray.init rc={rc}", flush=True)
    _dump_ray_logs()
    if rc != 0:
        raise RuntimeError(f"cpu ray diagnostic failed: {rc}")


@app.function(
    cpu=2.0,
    memory=8 * 1024,
    timeout=60 * 10,
)
def diagnose_sglang_args() -> None:
    """Print the pinned SGLang parser surface and exercise Miles validation."""
    import argparse
    import dataclasses
    import importlib.metadata

    if MILES_DIR not in sys.path:
        sys.path.insert(0, MILES_DIR)
    import miles.backends.sglang_utils.arguments as sglang_arguments
    from miles.backends.sglang_utils.arguments import add_sglang_arguments
    from miles.backends.sglang_utils.arguments import validate_args as validate_sglang_args
    from sglang.srt.server_args import ServerArgs

    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-num-gpus-per-engine", type=int, default=1)
    parser.add_argument("--true-on-policy-mode", action="store_true")
    parser.add_argument("--recompute-logprobs-via-prefill", action="store_true")
    add_sglang_arguments(parser)
    args = parser.parse_args(
        ["--sglang-cuda-graph-backend-prefill", "disabled"]
    )

    try:
        version = importlib.metadata.version("sglang")
    except importlib.metadata.PackageNotFoundError:
        version = "<unknown>"
    parsed = sorted(name for name in vars(args) if name.startswith("sglang_"))
    server_fields = sorted(field.name for field in dataclasses.fields(ServerArgs))
    print(f"[sglang-args] version={version}", flush=True)
    print(f"[sglang-args] miles_module={sglang_arguments.__file__}", flush=True)
    print(f"[sglang-args] parsed={parsed}", flush=True)
    print(f"[sglang-args] ServerArgs.fields={server_fields}", flush=True)
    validate_sglang_args(args)
    normalized = {
        name: getattr(args, name)
        for name in (
            "sglang_tp_size",
            "sglang_dp_size",
            "sglang_pp_size",
            "sglang_ep_size",
            "sglang_attn_cp_size",
        )
        if hasattr(args, name)
    }
    print(f"[sglang-args] normalized={normalized}", flush=True)


@app.function(
    timeout=60 * 60 * 4,
    volumes={
        SFT_DIR: sft_vol,
        DATA_ROOT: data_vol,
        HF_CACHE_DIR: hf_cache,
    },
)
def prepare_assets(specs: str = "") -> list[str]:
    chosen = _choose_specs(specs)
    ensure_chess_data(DATA_DIR)
    for spec in chosen:
        ensure_sft_model(model_id_from_spec(spec), SFT_DIR, cot_type=COT_TYPE)
    data_vol.commit()
    sft_vol.commit()
    return chosen


@app.function(
    gpu=f"{GPU_TYPE}:{GPUS_PER_NODE}",
    cpu=CPU_COUNT,
    memory=MEMORY_MB,
    timeout=60 * 60 * TIMEOUT_HOURS,
    retries=modal.Retries(initial_delay=0.0, max_retries=RETRIES),
    volumes={
        SFT_DIR: sft_vol,
        DATA_ROOT: data_vol,
        CKPT_DIR: ckpt_vol,
        HF_CACHE_DIR: hf_cache,
    },
)
def train_one(
    spec: str,
    num_rollout: int = 500,
    save_interval: int = 20,
    eval_interval: int = 0,
    rollout_batch_size: int = 256,
    n_samples_per_prompt: int = 8,
    over_sampling_batch_size: int = 256,
    global_batch_size: int = 2048,
    dynamic_filter: bool = False,
    run_name_suffix: str = "",
    hparam_tag_suffix: str = "",
    extra_args: str = "",
    wandb_project: str = "chess_rl_6p5e18",
    wandb_group: str = "multi_turn_miles",
    wandb_team: str = "jingyanshen-new-york-university",
    io_layout: str = "chess-rl",
    resume_from_save: bool = False,
    resume_if_available: bool = False,
    cispo: bool = True,
    resume_ckpt_step: int = 0,
    optim_tag: str = "minimax",
    adam_beta2: float = 0.95,
    adam_eps: float = 1e-15,
    weight_decay: float = 0.0,
    batched_rollout: bool = True,
    sglang_token_id_only: bool = True,
    sglang_server_concurrency: int = 128,
    train_file: str = DEFAULT_MODAL_TRAIN_FILE,
    train_file_sha256: str = "",
    rollout_seed: int = 42,
) -> str:
    model_id = model_id_from_spec(spec)
    if run_name_suffix == "auto":
        run_name_suffix = DEFAULT_RUN_SUFFIX_BY_SPEC.get(spec, "")
    run_name = f"{model_id}_{run_name_suffix}" if run_name_suffix else model_id
    effective_hparam_tag_suffix = hparam_tag_suffix or (run_name_suffix if io_layout == "flat" else "")
    save_root = f"{CKPT_DIR}/chess-rl-miles"
    if io_layout == "flat":
        run_save_path = f"{save_root}/{run_name}"
    else:
        hparam_tag = _default_hparam_tag(
            cispo=cispo,
            hparam_tag_suffix=effective_hparam_tag_suffix,
            optim_tag=optim_tag,
            global_batch_size=global_batch_size,
        )
        run_save_path = f"{save_root}/{COT_TYPE}/{hparam_tag}/{model_id}/checkpoints"
    cpu_threads = int(CPU_COUNT)

    _validate_resume_modes(
        resume_from_save=resume_from_save,
        resume_if_available=resume_if_available,
        resume_ckpt_step=resume_ckpt_step,
    )
    if resume_from_save or resume_if_available:
        # A retry may reuse a warm container, so refresh the mounted view before
        # deciding whether a checkpoint from the previous attempt is available.
        ckpt_vol.reload()
    resume_path, selected_resume_step = _resolve_resume_checkpoint(
        run_save_path,
        resume_from_save=resume_from_save,
        resume_if_available=resume_if_available,
        resume_ckpt_step=resume_ckpt_step,
    )

    cmd = [
        sys.executable,
        "-m",
        "chess_rl_miles.scripts.run_chess_miles",
        "--miles-dir",
        MILES_DIR,
        "--project-dir",
        PROJECT_DIR,
        "--spec",
        spec,
        "--run-name",
        run_name,
        "--io-layout",
        io_layout,
        "--hparam-tag-suffix",
        effective_hparam_tag_suffix,
        "--prepare-data",
        "--prepare-sft",
        "--sft-root",
        SFT_DIR,
        "--data-dir",
        DATA_DIR,
        "--train-file",
        train_file,
        "--save-dir",
        save_root,
        "--rollout-seed",
        str(rollout_seed),
        "--num-rollout",
        str(num_rollout),
        "--save-interval",
        str(save_interval),
        "--rollout-batch-size",
        str(rollout_batch_size),
        "--n-samples-per-prompt",
        str(n_samples_per_prompt),
        "--over-sampling-batch-size",
        str(over_sampling_batch_size),
        "--global-batch-size",
        str(global_batch_size),
        "--sglang-server-concurrency",
        str(sglang_server_concurrency),
        "--optim-tag",
        optim_tag,
        "--adam-beta2",
        str(adam_beta2),
        "--adam-eps",
        str(adam_eps),
        "--weight-decay",
        str(weight_decay),
        "--actor-num-gpus-per-node",
        str(GPUS_PER_NODE),
        "--cpu-threads",
        str(cpu_threads),
        "--wandb-project",
        wandb_project,
        "--wandb-group",
        wandb_group,
        "--wandb-team",
        wandb_team,
    ]
    if train_file_sha256:
        cmd.extend(["--train-file-sha256", train_file_sha256])
    cmd.append("--cispo" if cispo else "--no-cispo")
    cmd.append("--batched-rollout" if batched_rollout else "--no-batched-rollout")
    cmd.append("--sglang-token-id-only" if sglang_token_id_only else "--no-sglang-token-id-only")
    passthrough_args: list[str] = []
    if resume_path is not None:
        cmd.extend(["--load", str(resume_path)])
        if resume_ckpt_step > 0:
            passthrough_args.extend(
                [
                    "--ckpt-step",
                    str(selected_resume_step),
                    "--start-rollout-id",
                    str(selected_resume_step),
                ]
            )
    if eval_interval > 0:
        cmd.extend(["--eval-interval", str(eval_interval)])
    if dynamic_filter:
        cmd.append("--dynamic-filter")
    if extra_args:
        passthrough_args.extend(shlex.split(extra_args))
    if passthrough_args:
        cmd.append("--")
        cmd.extend(passthrough_args)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{PROJECT_DIR}:{MILES_DIR}:{env.get('PYTHONPATH', '')}"
    env["PYTHONUNBUFFERED"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "true"
    env["OMP_NUM_THREADS"] = str(cpu_threads)
    env["RAYON_NUM_THREADS"] = str(cpu_threads)
    env["SGLANG_CPU_THREAD_POOL_SIZE"] = str(cpu_threads)
    env["MILES_DISABLE_TQDM"] = "1"
    env["TQDM_DISABLE"] = "1"
    env["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    env["RAY_DEDUP_LOGS"] = "0"
    if batched_rollout:
        # The rollout refactor is selected while Ray actors import Miles, so
        # this must be present when the Ray head starts, not only in the driver.
        env["MILES_EXPERIMENTAL_ROLLOUT_REFACTOR"] = "1"
    env.setdefault("HF_HOME", HF_CACHE_DIR)
    # Ray is started before the Miles driver, so rollout actors inherit the
    # environment below from the Ray head rather than from build_command().
    # Set the artifact root here or the custom JSONL log hooks silently no-op.
    artifact_root = run_save_path if io_layout == "flat" else str(Path(run_save_path).parent)
    env["CHESS_RL_MILES_ARTIFACT_ROOT"] = artifact_root

    redacted = list(cmd)
    for idx, item in enumerate(redacted[:-1]):
        if item == "--wandb-key":
            redacted[idx + 1] = "<redacted>"
    print("[train] " + " ".join(redacted), flush=True)
    expected_train_sha256 = train_file_sha256 or (
        DEFAULT_TRAIN_FILE_SHA256 if train_file == DEFAULT_MODAL_TRAIN_FILE else ""
    )
    print(
        f"[train] train_file={train_file} "
        f"expected_sha256={expected_train_sha256 or '<not-enforced>'} "
        f"rollout_seed={rollout_seed}",
        flush=True,
    )
    if resume_path is not None:
        print(
            f"[train] resume_checkpoint={resume_path} step={selected_resume_step}",
            flush=True,
        )
    elif resume_if_available:
        print(
            f"[train] no checkpoint tracker under {run_save_path}; "
            "starting fresh from SFT (future retries will resume)",
            flush=True,
        )

    _cleanup_runtime()
    _start_ray_head(env, cpu_threads=cpu_threads)
    env["RAY_ADDRESS"] = RAY_ADDRESS
    rc = subprocess.call(cmd, env=env, cwd=PROJECT_DIR)
    ckpt_vol.commit()
    sft_vol.commit()
    data_vol.commit()
    if rc != 0:
        raise RuntimeError(f"training failed for {model_id}: exit {rc}")
    return model_id


@app.function(
    cpu=2.0,
    memory=8 * 1024,
    timeout=60 * 10,
    volumes={CKPT_DIR: ckpt_vol},
)
def wipe_model_artifacts(spec: str) -> list[str]:
    ckpt_vol.reload()
    model_id = model_id_from_spec(spec)
    root = Path(CKPT_DIR) / "chess-rl-miles"
    if not root.exists():
        print(f"[wipe] missing {root}", flush=True)
        return []

    candidates: list[Path] = []
    for path in root.rglob("*"):
        if path.is_dir() and (path.name == model_id or model_id in path.name):
            candidates.append(path)

    unique: list[Path] = []
    for path in sorted(set(candidates), key=lambda p: (len(p.parts), str(p))):
        if any(path == parent or parent in path.parents for parent in unique):
            continue
        unique.append(path)

    removed: list[str] = []
    for path in unique:
        print(f"[wipe] DELETE {path}", flush=True)
        shutil.rmtree(path)
        removed.append(str(path))

    ckpt_vol.commit()
    print(f"[wipe] removed {len(removed)} artifact dir(s) for {model_id}", flush=True)
    return removed


@app.local_entrypoint()
def main(
    specs: str = "",
    dry_run: bool = False,
    wait: bool = False,
    prepare_only: bool = False,
    ray_diag: bool = False,
    ray_diag_cpu: bool = False,
    sglang_args_diag: bool = False,
    num_rollout: int = 500,
    save_interval: int = 20,
    eval_interval: int = 0,
    rollout_batch_size: int = 256,
    n_samples_per_prompt: int = 8,
    over_sampling_batch_size: int = 256,
    global_batch_size: int = 2048,
    dynamic_filter: bool = False,
    run_name_suffix: str = "",
    hparam_tag_suffix: str = "",
    extra_args: str = "",
    wandb_project: str = "chess_rl_6p5e18",
    wandb_group: str = "multi_turn_miles",
    wandb_team: str = "jingyanshen-new-york-university",
    io_layout: str = "chess-rl",
    resume_from_save: bool = False,
    resume_if_available: bool = False,
    cispo: bool = True,
    resume_ckpt_step: int = 0,
    optim_tag: str = "minimax",
    adam_beta2: float = 0.95,
    adam_eps: float = 1e-15,
    weight_decay: float = 0.0,
    wipe_first: bool = False,
    batched_rollout: bool = True,
    sglang_token_id_only: bool = True,
    sglang_server_concurrency: int = 128,
    train_file: str = DEFAULT_MODAL_TRAIN_FILE,
    train_file_sha256: str = "",
    rollout_seed: int = 42,
) -> None:
    _validate_resume_modes(
        resume_from_save=resume_from_save,
        resume_if_available=resume_if_available,
        resume_ckpt_step=resume_ckpt_step,
    )
    chosen = _choose_specs(specs)
    print(f"Chess-RL Miles sweep: {len(chosen)} job(s) on {GPUS_PER_NODE}x{GPU_TYPE}")
    expected_train_sha256 = train_file_sha256 or (
        DEFAULT_TRAIN_FILE_SHA256 if train_file == DEFAULT_MODAL_TRAIN_FILE else ""
    )
    print(
        f"RL data: {train_file} "
        f"(sha256={expected_train_sha256 or 'logged-only'}, rollout_seed={rollout_seed})"
    )
    for spec in chosen:
        print(f"  {model_id_from_spec(spec)} ({spec})")

    if dry_run:
        print("(dry-run)")
        return

    if ray_diag:
        diagnose_ray.remote()
        return

    if ray_diag_cpu:
        diagnose_ray_cpu.remote()
        return

    if sglang_args_diag:
        diagnose_sglang_args.remote()
        return

    if prepare_only:
        handle = prepare_assets.spawn(specs)
        if wait:
            handle.get()
        print("Spawned asset preparation job.")
        return

    if wipe_first:
        for spec in chosen:
            wipe_model_artifacts.remote(spec)

    handles = [
        (
            spec,
            train_one.spawn(
                spec,
                num_rollout,
                save_interval,
                eval_interval,
                rollout_batch_size,
                n_samples_per_prompt,
                over_sampling_batch_size,
                global_batch_size,
                dynamic_filter,
                run_name_suffix,
                hparam_tag_suffix,
                extra_args,
                wandb_project,
                wandb_group,
                wandb_team,
                io_layout,
                resume_from_save,
                resume_if_available,
                cispo,
                resume_ckpt_step,
                optim_tag,
                adam_beta2,
                adam_eps,
                weight_decay,
                batched_rollout,
                sglang_token_id_only,
                sglang_server_concurrency,
                train_file,
                train_file_sha256,
                rollout_seed,
            ),
        )
        for spec in chosen
    ]
    for spec, handle in handles:
        print(f"SPAWNED {model_id_from_spec(spec)}: {handle.object_id}")
    print(f"Spawned {len(handles)} training job(s). Monitor in Modal app chess-rl-miles.")

    if not wait:
        return

    failed = []
    for spec, handle in handles:
        try:
            handle.get()
        except Exception as exc:
            print(f"FAILED {model_id_from_spec(spec)}: {exc}")
            failed.append(spec)
    if failed:
        raise SystemExit(1)
