"""Launch corrected, from-scratch 6p5e18 pretraining reruns.

The historical configs still contain the old lr=2e-3 / eta_min=1e-5 values.
This launcher keeps those configs as provenance and applies the corrected
runtime settings in an isolated checkpoint root:

  - peak lr: 1e-3
  - cosine eta_min: 1e-4
  - warmup: 5%
  - effective global batch: 512 sequences (8 GPUs x bs32 x GA2)
  - seed: 42

Production:
  modal run --detach modal_scripts/launch_corrected_6p5e18_pretrains.py \
    --targets 50m_a0p400
  modal run --detach modal_scripts/launch_corrected_6p5e18_pretrains.py \
    --targets 200m_a0p200
  modal run --detach modal_scripts/launch_corrected_6p5e18_pretrains.py \
    --targets 200m_a0p750

One-step architecture/data smoke:
  CHESS_PRETRAIN_GPU_TYPE=H100 CHESS_PRETRAIN_GPUS=2 \
    modal run --detach modal_scripts/launch_corrected_6p5e18_pretrains.py \
    --targets 50m_a0p400,200m_a0p200 --smoke
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path

import modal

GPU_TYPE = os.environ.get("CHESS_PRETRAIN_GPU_TYPE", "H200").strip()
GPUS_PER_NODE = int(os.environ.get("CHESS_PRETRAIN_GPUS", "8"))
TIMEOUT_HOURS = 47
MAX_CHECKPOINTS = 3
MIXED_PRECISION = "bf16"

DATA_DIR = "/data/pretrain_v1_20b"
LINEAGE_DATA_DIR = "/tmp/pretrain_v1_20b_c0cf90d"
DATASET_REPO = "chess-pre-to-post/pretrain_v1_20b"
DATASET_REVISION = "c0cf90d274339f6b48ab12b98377b7e49e787b1c"
DATASET_FIRST_SHARD = 0
DATASET_LAST_SHARD = 24_774
EVAL_SHARD_IDS = (24_773, 24_774)
FRESH_TAG = os.environ.get(
    "CHESS_PRETRAIN_FRESH_TAG",
    "corrected_lr1e-3_20260726_r4",
).strip()
OUTPUT_ROOT = f"/checkpoints/6p5e18_{FRESH_TAG}"
SMOKE_OUTPUT_ROOT = f"/checkpoints/6p5e18_{FRESH_TAG}_smoke"
WANDB_ENTITY = "jingyanshen-new-york-university"
WANDB_PROJECT = "chess-scaling-C_6p5e18-corrected-lr1e-3"
CORRECT_LINEAGE_COMMIT = "9db8b50be20375708bdc7de78e8554eb3dfcd738"
CORRECT_LINEAGE_PATCH = "april_2026_numeric_shard_order.patch"

JOBS = {
    "50m_a0p400": {
        "config": "config/configs/6p5e18/50m_alpha0.400.yaml",
        "canonical_name": "50m_C_6p5e18_alpha0.400",
        "hf_stem": "C6p5e18_50m_alpha0.400",
        "pretrain_tokens": 9_181_735_000,
    },
    "200m_a0p200": {
        "config": "config/configs/6p5e18/200m_alpha0.200.yaml",
        "canonical_name": "200m_C_6p5e18_alpha0.200",
        "hf_stem": "C6p5e18_200m_alpha0.200",
        "pretrain_tokens": 1_069_004_000,
    },
    "200m_a0p750": {
        "config": "config/configs/6p5e18/200m_alpha0.750.yaml",
        "canonical_name": "200m_C_6p5e18_alpha0.750",
        "hf_stem": "C6p5e18_200m_alpha0.750",
        "pretrain_tokens": 4_008_765_000,
    },
}

if not re.fullmatch(r"[A-Za-z0-9_.-]+", FRESH_TAG):
    raise ValueError("CHESS_PRETRAIN_FRESH_TAG must be path-safe")
if GPUS_PER_NODE < 1:
    raise ValueError("CHESS_PRETRAIN_GPUS must be positive")

cuda_version = "12.8.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
cuda_tag = f"{cuda_version}-{flavor}-{operating_sys}"
repo_dir = Path(__file__).parent.parent


def _materialize_source_snapshot() -> tuple[tempfile.TemporaryDirectory, Path]:
    """Create a clean local build context for the recorded Git commit."""
    temp_dir = tempfile.TemporaryDirectory(prefix="chess-pretrain-source-")
    snapshot_dir = Path(temp_dir.name) / "chess"
    snapshot_dir.mkdir()

    archive = subprocess.Popen(
        ["git", "archive", "--format=tar", CORRECT_LINEAGE_COMMIT],
        cwd=repo_dir,
        stdout=subprocess.PIPE,
    )
    if archive.stdout is None:
        raise RuntimeError("Could not open git archive output")
    extract = subprocess.run(
        ["tar", "-x", "-C", str(snapshot_dir)],
        stdin=archive.stdout,
        check=False,
    )
    archive.stdout.close()
    archive_rc = archive.wait()
    if archive_rc != 0 or extract.returncode != 0:
        temp_dir.cleanup()
        raise RuntimeError(
            f"Could not materialize source commit {CORRECT_LINEAGE_COMMIT}: "
            f"git={archive_rc}, tar={extract.returncode}"
        )
    patch_path = (
        repo_dir / "modal_scripts" / "patches" / CORRECT_LINEAGE_PATCH
    )
    apply_patch = subprocess.run(
        [
            "patch",
            "-p1",
            "-d",
            str(snapshot_dir),
            "-i",
            str(patch_path),
        ],
        check=False,
    )
    if apply_patch.returncode != 0:
        temp_dir.cleanup()
        raise RuntimeError(
            f"Could not apply inferred April lineage patch: {patch_path}"
        )
    (snapshot_dir / ".source_commit").write_text(
        CORRECT_LINEAGE_COMMIT + "\n",
        encoding="utf-8",
    )
    (snapshot_dir / ".source_patch").write_text(
        CORRECT_LINEAGE_PATCH + "\n",
        encoding="utf-8",
    )
    return temp_dir, snapshot_dir


if modal.is_local():
    _source_snapshot_temp, source_snapshot_dir = _materialize_source_snapshot()
else:
    _source_snapshot_temp = None
    source_snapshot_dir = Path("/root/chess")


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
        if key != "WANDB_API_KEY":
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
            "WANDB_ENTITY": WANDB_ENTITY,
        }
    )
else:
    wandb_secret = modal.Secret.from_name("wandb-secret")

image = (
    modal.Image.from_registry(f"nvidia/cuda:{cuda_tag}", add_python="3.11")
    .apt_install("curl", "git", "vim", "htop")
    .pip_install(
        # Match the surviving W&B runtime metadata from the deleted corrected
        # lineage instead of resolving floating dependency versions.
        "torch==2.9.0",
        "accelerate==1.10.1",
        "transformers==4.57.0",
        "datasets==4.2.0",
        "pyarrow>=17.0.0",
        "pandas>=2.0.0",
        "pyyaml>=6.0",
        "omegaconf>=2.3.0",
        "wandb>=0.19.0",
        "einops>=0.7.0",
        "tokenizers==0.22.1",
        "tqdm>=4.66.0",
        "chess>=1.11.0",
        "numpy==2.2.6",
        "safetensors==0.6.2",
        "sentencepiece>=0.2.0",
        "huggingface-hub==0.35.3",
    )
    .run_commands(
        "python -c \"from transformers import AutoTokenizer; "
        "AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B')\""
    )
    .env(
        {
            "CHESS_PRETRAIN_GPU_TYPE": GPU_TYPE,
            "CHESS_PRETRAIN_GPUS": str(GPUS_PER_NODE),
            "CHESS_PRETRAIN_FRESH_TAG": FRESH_TAG,
            "WANDB_ENTITY": WANDB_ENTITY,
        }
    )
)
if modal.is_local():
    image = image.add_local_dir(
        str(source_snapshot_dir),
        remote_path="/root/chess",
    )

data_volume = modal.Volume.from_name(
    "rl-reasoning-training-data",
    create_if_missing=True,
)
ckpt_volume = modal.Volume.from_name(
    "rl-reasoning-checkpoints",
    create_if_missing=True,
)

app = modal.App(
    "chess-pretrain-6p5e18-corrected",
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


def _output_root(*, smoke: bool) -> str:
    return SMOKE_OUTPUT_ROOT if smoke else OUTPUT_ROOT


def _hf_repo(target: str, *, smoke: bool) -> str:
    if smoke:
        return "null"
    return (
        f"Pre-to-Post-2/pretrain_{JOBS[target]['hf_stem']}_{FRESH_TAG}"
    )


def _shard_name(shard_id: int) -> str:
    return f"raw.{shard_id:04d}.npy"


def _materialize_dataset_snapshot() -> tuple[Path, list[Path], list[Path]]:
    """Expose exactly the immutable April shard set from the shared volume.

    The shared directory has since grown from 24,775 to 47,090 shards. The
    original shards are unchanged and still present, so an ephemeral symlink
    view avoids copying roughly 114 GB per worker while preventing newer
    shards from entering the seeded shuffle.
    """
    source_dir = Path(DATA_DIR)
    lineage_dir = Path(LINEAGE_DATA_DIR)
    lineage_dir.mkdir(parents=True, exist_ok=True)

    expected_names = [
        _shard_name(shard_id)
        for shard_id in range(DATASET_FIRST_SHARD, DATASET_LAST_SHARD + 1)
    ]
    source_files: list[Path] = []
    missing: list[str] = []
    for name in expected_names:
        source = source_dir / name
        if not source.is_file():
            missing.append(name)
            continue
        source_files.append(source)
        link = lineage_dir / name
        if link.is_symlink():
            if link.resolve() != source.resolve():
                raise RuntimeError(
                    f"Snapshot link points at the wrong shard: {link}"
                )
        elif link.exists():
            raise RuntimeError(
                f"Snapshot view contains a non-symlink entry: {link}"
            )
        else:
            link.symlink_to(source)

    if missing:
        sample = ", ".join(missing[:5])
        raise FileNotFoundError(
            f"Dataset revision {DATASET_REVISION} is incomplete on Modal: "
            f"{len(missing)} missing shard(s), first: {sample}"
        )

    view_files = list(lineage_dir.glob("*.npy"))
    unexpected = sorted({path.name for path in view_files} - set(expected_names))
    if len(view_files) != len(expected_names) or unexpected:
        raise RuntimeError(
            f"Invalid snapshot view: expected={len(expected_names)} "
            f"actual={len(view_files)} unexpected={unexpected[:5]}"
        )

    eval_files = [
        lineage_dir / _shard_name(shard_id) for shard_id in EVAL_SHARD_IDS
    ]
    train_files = [
        path for path in view_files if path.name not in {p.name for p in eval_files}
    ]
    if len(train_files) != 24_773 or len(eval_files) != 2:
        raise RuntimeError(
            f"Invalid train/eval split: train={len(train_files)} "
            f"eval={len(eval_files)}"
        )
    return lineage_dir, train_files, eval_files


def _build_overrides(
    target: str,
    *,
    smoke: bool,
    num_gpus: int,
) -> list[str]:
    experiment_name = _experiment_name(target, smoke=smoke)
    overrides = [
        "training.gpu_peak_tflops=989",
        "training.cache_size=0",
        "training.mixed_precision=bf16",
        "training.seed=42",
        "training.batch_size=32",
        "training.gradient_accumulation_steps=2",
        "training.optimizer.lr=1e-3",
        "training.scheduler.eta_min=1e-4",
        "training.scheduler.warmup_ratio=0.05",
        f"training.experiment_name={experiment_name}",
        f"training.run_name={experiment_name}",
        f"training.hf_upload_repo={_hf_repo(target, smoke=smoke)}",
        f"data.txt_path={LINEAGE_DATA_DIR}",
        "data.eval_holdout=2",
        f"logging.entity={WANDB_ENTITY}",
        f"logging.project={WANDB_PROJECT}",
    ]
    if smoke:
        overrides.extend(
            [
                # Match the deleted 2xH100 lineage exactly for the smoke:
                # 32 sequences/GPU x GA8 x 2 GPUs = 512 sequences/update.
                "training.batch_size=32",
                "training.gradient_accumulation_steps=8",
                "training.num_workers=2",
                "training.log_interval=1",
                "training.eval_max_steps=1",
                f"data.pretrain_tokens={1024 * 32 * 8 * num_gpus}",
            ]
        )
    return overrides


def _overridden_value(
    overrides: list[str],
    key: str,
    default: str | None = None,
) -> str | None:
    prefix = f"{key}="
    for override in reversed(overrides):
        if override.startswith(prefix):
            return override.split("=", 1)[1]
    return default


def _expected_steps(
    *,
    pretrain_tokens: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    num_gpus: int,
) -> int:
    tokens_per_step = (
        batch_size * 1024 * gradient_accumulation_steps * num_gpus
    )
    return pretrain_tokens // tokens_per_step


def _run_training(
    *,
    target: str,
    smoke: bool,
    num_gpus: int,
) -> None:
    import subprocess
    import sys

    import yaml
    from omegaconf import OmegaConf

    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

    source_commit_file = Path("/root/chess/.source_commit")
    actual_source_commit = source_commit_file.read_text(encoding="utf-8").strip()
    if actual_source_commit != CORRECT_LINEAGE_COMMIT:
        raise RuntimeError(
            f"Source lineage mismatch: {actual_source_commit} != "
            f"{CORRECT_LINEAGE_COMMIT}"
        )
    source_patch_file = Path("/root/chess/.source_patch")
    actual_source_patch = source_patch_file.read_text(encoding="utf-8").strip()
    if actual_source_patch != CORRECT_LINEAGE_PATCH:
        raise RuntimeError(
            f"Source patch mismatch: {actual_source_patch} != "
            f"{CORRECT_LINEAGE_PATCH}"
        )

    job = JOBS[target]
    config = str(job["config"])
    config_path = Path("/root/chess") / config
    with config_path.open("r", encoding="utf-8") as handle:
        source_cfg = yaml.safe_load(handle)

    model_cfg = source_cfg.get("model", {})
    training_cfg = source_cfg.get("training", {})
    if model_cfg.get("pretrained_model") or training_cfg.get("pretrained_weights"):
        raise RuntimeError(
            f"{config} is not a from-scratch pretraining config; refusing launch"
        )
    if int(source_cfg["data"]["pretrain_tokens"]) != int(job["pretrain_tokens"]):
        raise RuntimeError(
            f"Token-budget mismatch for {target}: "
            f"{source_cfg['data']['pretrain_tokens']} != {job['pretrain_tokens']}"
        )

    lineage_data_dir, train_files, eval_files = _materialize_dataset_snapshot()

    overrides = _build_overrides(
        target,
        smoke=smoke,
        num_gpus=num_gpus,
    )
    effective_cfg = OmegaConf.merge(
        OmegaConf.create(source_cfg),
        OmegaConf.from_dotlist(overrides),
    )
    effective_cfg.launch_provenance = OmegaConf.create(
        {
            "fresh_tag": FRESH_TAG,
            "correct_lineage_base_commit": CORRECT_LINEAGE_COMMIT,
            "correct_lineage_patch": CORRECT_LINEAGE_PATCH,
            "lineage_patch_basis": (
                "minimal inferred reconstruction from the April W&B "
                "dataloader length and observed eval shards"
            ),
            "dataset_repo": DATASET_REPO,
            "dataset_revision": DATASET_REVISION,
            "dataset_shards": len(train_files) + len(eval_files),
            "dataset_train_shards": len(train_files),
            "dataset_eval_shards": [path.name for path in eval_files],
            "python": "3.11",
            "cuda": cuda_version,
            "torch": "2.9.0",
            "transformers": "4.57.0",
            "accelerate": "1.10.1",
            "tokenizers": "0.22.1",
            "numpy": "2.2.6",
            "safetensors": "0.6.2",
            "datasets": "4.2.0",
            "huggingface_hub": "0.35.3",
        }
    )
    effective_tokens = int(effective_cfg.data.pretrain_tokens)
    batch_size = int(effective_cfg.training.batch_size)
    grad_accum = int(effective_cfg.training.gradient_accumulation_steps)
    expected_steps = _expected_steps(
        pretrain_tokens=effective_tokens,
        batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_gpus=num_gpus,
    )

    if not smoke and batch_size * grad_accum * num_gpus != 512:
        raise RuntimeError(
            "Production effective batch must be exactly 512 sequences: "
            f"{batch_size} x {grad_accum} x {num_gpus}"
        )
    if float(effective_cfg.training.optimizer.lr) != 1e-3:
        raise RuntimeError("Corrected peak LR must be exactly 1e-3")
    if float(effective_cfg.training.scheduler.eta_min) != 1e-4:
        raise RuntimeError("Corrected eta_min must be exactly 1e-4")
    if float(effective_cfg.training.scheduler.warmup_ratio) != 0.05:
        raise RuntimeError("Corrected warmup_ratio must be exactly 0.05")

    output_root = _output_root(smoke=smoke)
    experiment_name = _experiment_name(target, smoke=smoke)
    if FRESH_TAG not in output_root or FRESH_TAG not in experiment_name:
        raise RuntimeError("Fresh run is not isolated by FRESH_TAG")
    run_dir = Path(output_root) / experiment_name

    ckpt_volume.reload()
    final_dir = run_dir / "final"
    latest_dir = run_dir / "latest"
    if final_dir.is_dir():
        print(f"[pretrain] Final already exists; skipping: {final_dir}", flush=True)
        return
    if latest_dir.is_dir():
        print(f"[pretrain] Retrying corrected run from: {latest_dir}", flush=True)
    else:
        print(
            f"[pretrain] Fresh random initialization: {experiment_name}",
            flush=True,
        )

    effective_hf_repo = effective_cfg.training.get("hf_upload_repo")
    if not smoke and (
        not effective_hf_repo or FRESH_TAG not in str(effective_hf_repo)
    ):
        raise RuntimeError(
            f"Production HF repo is not fresh-tagged: {effective_hf_repo}"
        )
    if smoke and effective_hf_repo:
        raise RuntimeError("Smoke run must not upload to Hugging Face")

    effective_cfg.training.auto_resume = True
    effective_cfg.training.save_dir = output_root
    effective_cfg.training.max_checkpoints = MAX_CHECKPOINTS
    effective_cfg.data.txt_path = str(lineage_data_dir)
    effective_cfg.data.eval_holdout = 2
    if effective_cfg.data.get("eval_txt_files"):
        raise RuntimeError(
            "April lineage uses numeric shard ordering plus eval_holdout=2; "
            "explicit eval_txt_files must be unset"
        )
    effective_cfg.data.test_data_dir = "/data/test"
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(effective_cfg, run_dir / "effective_config.yaml")
    ckpt_volume.commit()

    print(
        f"[pretrain] target={target} config={config} "
        f"GPUs={num_gpus}x{GPU_TYPE} "
        f"dataset={DATASET_REPO}@{DATASET_REVISION} "
        f"shards={len(train_files)}+{len(eval_files)} "
        f"eval={','.join(path.name for path in eval_files)} "
        f"tokens={effective_tokens} batch={batch_size}x{grad_accum}x{num_gpus} "
        f"max_steps={expected_steps} lr=1e-3 warmup=0.05 eta_min=1e-4 "
        f"output={run_dir} hf={effective_hf_repo or 'disabled'}",
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
        str(lineage_data_dir),
        "--output_dir",
        output_root,
        "--test_data_dir",
        "/data/test",
        "--max_checkpoints",
        str(MAX_CHECKPOINTS),
        "--override",
        *overrides,
    ]
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
        raise RuntimeError(f"Pretraining failed (exit {proc.returncode}): {target}")
    print(f"[pretrain] Completed: {experiment_name}", flush=True)


@app.function(
    gpu=f"{GPU_TYPE}:{GPUS_PER_NODE}",
    timeout=60 * 60 * TIMEOUT_HOURS,
    retries=modal.Retries(initial_delay=0.0, max_retries=5),
    max_containers=3,
)
def train_production(target: str, num_gpus: int) -> None:
    _run_training(target=target, smoke=False, num_gpus=num_gpus)


@app.function(
    gpu=f"{GPU_TYPE}:{GPUS_PER_NODE}",
    timeout=60 * 60,
    max_containers=2,
)
def train_smoke(target: str, num_gpus: int) -> None:
    _run_training(target=target, smoke=True, num_gpus=num_gpus)


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
    targets: str = "50m_a0p400,200m_a0p200,200m_a0p750",
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
            "Production requires the default 8xH200 topology; unset "
            "CHESS_PRETRAIN_GPU_TYPE and CHESS_PRETRAIN_GPUS"
        )

    print(
        f"Corrected 6p5e18 pretraining: targets={','.join(chosen)} "
        f"GPU={GPUS_PER_NODE}x{GPU_TYPE} smoke={smoke} tag={FRESH_TAG}",
        flush=True,
    )
    for target in chosen:
        job = JOBS[target]
        if smoke:
            tokens = 1024 * 32 * 8 * GPUS_PER_NODE
            batch_size = 32
            grad_accum = 8
        else:
            tokens = int(job["pretrain_tokens"])
            batch_size = 32
            grad_accum = 2
        steps = _expected_steps(
            pretrain_tokens=tokens,
            batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            num_gpus=GPUS_PER_NODE,
        )
        print(
            f"  {target}: {job['config']} -> "
            f"{_output_root(smoke=smoke)}/{_experiment_name(target, smoke=smoke)} "
            f"steps={steps} hf={_hf_repo(target, smoke=smoke)}",
            flush=True,
        )

    if dry_run:
        print("(dry-run -- nothing launched)", flush=True)
        return

    launch_function = train_smoke if smoke else train_production
    handles = []
    for target in chosen:
        handle = launch_function.spawn(
            target=target,
            num_gpus=GPUS_PER_NODE,
        )
        handles.append((target, handle))
        print(f"  SPAWNED {target}: {handle.object_id}", flush=True)
    print(f"{len(handles)} pretraining job(s) spawned.", flush=True)
    for target, handle in handles:
        run_kind = "smoke" if smoke else "production"
        print(f"  WAITING for {run_kind} {target}", flush=True)
        handle.get()
        print(f"  COMPLETED {run_kind} {target}", flush=True)
