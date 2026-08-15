"""Future-only FA2 comparison for the frozen interleaved-50M production v1.

This is intentionally a separate Modal app and image.  Production v1 is
already frozen to explicit SDPA without torch.compile; results from this app
must not mutate that experiment contract after P1/Exp2 launch.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import modal


APP_NAME = "chess-50m-interleaved-fa2-benchmark"
FLASH_ATTENTION_VERSION = "2.8.3"
EXPECTED_P1_MANIFEST_SHA256 = (
    "6e2bbd62283df234e8ea10c1d27d871e4ce59991785c58b13de3d6b721feaf5a"
)
REFERENCE_SDPA_FIRST_LOSS = 4.4960103034973145
LOSS_PARITY_ABS_TOLERANCE = 5e-3
BENCHMARK_STEPS = 30
BENCHMARK_WARMUP_STEPS = 10
REPO_DIR = Path(__file__).resolve().parent.parent
REMOTE_REPO = Path("/root/chess")
OUTPUT_ROOT = Path("/tmp/chess-interleave-fa2-future")
BASE_CONFIG = "config/configs/interleaved_50m/base_3072.yaml"
TRAIN_CLI = "scripts/train/train_interleaved_hf.py"
SOURCE_ROOT = "/data/pretrain_v1_20b"
SOURCE_MANIFEST = (
    "/data/50m_interleaved_mix10b_sft90k_v1/source_manifest.json"
)
SELECTION_MANIFEST = (
    "/data/50m_interleaved_mix10b_sft90k_v1/pretrain_selection.json"
)
SFT_CACHE = "/data/50m_interleaved_mix10b_sft90k_v1/sft_cache"
P1_MANIFEST = (
    "/data/50m_interleaved_mix10b_sft90k_v1/legs/p1/metadata.json"
)


def _source_digest(root: Path) -> str:
    digest = hashlib.sha256()
    files: list[Path] = []
    for relative in ("config", "llm_tokens", "scripts", "training"):
        files.extend(
            path
            for path in (root / relative).rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix not in {".pyc", ".pyo"}
        )
    files.append(Path(__file__))
    for path in sorted(
        set(files),
        key=lambda item: (
            str(item.relative_to(root))
            if item.is_relative_to(root)
            else item.name
        ),
    ):
        relative = (
            str(path.relative_to(root))
            if path.is_relative_to(root)
            else path.name
        )
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


_computed_source_tree_sha256 = _source_digest(REPO_DIR)
SOURCE_TREE_SHA256 = os.environ.get(
    "CHESS_FA2_BENCHMARK_SOURCE_TREE_SHA256",
    _computed_source_tree_sha256,
).strip()
if not re.fullmatch(r"[0-9a-f]{64}", SOURCE_TREE_SHA256):
    raise RuntimeError("Invalid FA2 benchmark source-tree SHA-256")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("build-essential", "curl", "git")
    .pip_install(
        "ninja==1.13.0",
        "packaging==25.0",
        "setuptools==80.9.0",
        "wheel==0.45.1",
        "torch==2.9.0",
        "accelerate==1.10.1",
        "transformers==4.57.0",
        "datasets==4.2.0",
        "huggingface-hub==0.35.3",
        "numpy==2.2.6",
        "safetensors==0.6.2",
        "pyarrow>=17.0.0",
        "pandas>=2.0.0",
        "pyyaml>=6.0",
        "omegaconf>=2.3.0",
        "wandb>=0.19.0",
        "einops>=0.7.0",
        "tokenizers==0.22.1",
        "tqdm>=4.66.0",
        "chess>=1.11.0",
        "sentencepiece>=0.2.0",
    )
    .env(
        {
            "FLASH_ATTENTION_FORCE_BUILD": "TRUE",
            "FLASH_ATTN_CUDA_ARCHS": "90",
            "MAX_JOBS": "8",
        }
    )
    .pip_install(
        f"flash-attn=={FLASH_ATTENTION_VERSION}",
        extra_options="--no-build-isolation",
    )
    .env(
        {
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": str(REMOTE_REPO),
            "CHESS_FA2_BENCHMARK_SOURCE_TREE_SHA256": SOURCE_TREE_SHA256,
        }
    )
    .add_local_dir(
        str(REPO_DIR / "scripts"),
        remote_path=str(REMOTE_REPO / "scripts"),
    )
    .add_local_dir(
        str(REPO_DIR / "training"),
        remote_path=str(REMOTE_REPO / "training"),
    )
    .add_local_dir(
        str(REPO_DIR / "config"),
        remote_path=str(REMOTE_REPO / "config"),
    )
    .add_local_dir(
        str(REPO_DIR / "llm_tokens"),
        remote_path=str(REMOTE_REPO / "llm_tokens"),
    )
)

data_volume = modal.Volume.from_name(
    "rl-reasoning-training-data",
    create_if_missing=False,
)
app = modal.App(
    APP_NAME,
    image=image,
    volumes={"/data": data_volume},
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compile_mode(value: str) -> str:
    mode = value.strip().lower()
    if mode not in {"none", "default"}:
        raise ValueError("compile_mode must be 'none' or 'default'")
    return mode


def _command(compile_mode: str) -> tuple[list[str], Path]:
    mode = _compile_mode(compile_mode)
    output = OUTPUT_ROOT / f"flash_attention_2-{mode}"
    overrides = [
        f"training.output_dir={output}",
        f"training.run_name=50m-fa2-future-{mode}",
        "training.seed=42",
        "training.local_batch_size=21",
        "training.gradient_accumulation_steps=1",
        "training.total_steps=9920",
        "training.arc_steps=[9920]",
        "model.attn_implementation=flash_attention_2",
        f"model.flash_attention_version={FLASH_ATTENTION_VERSION}",
        f"training.torch_compile={mode}",
        "training.reset_optimizer_between_arcs=true",
        "training.scheduler.warmup_ratio=0.05",
        "training.scheduler.eta_min=1e-5",
        "training.optimizer.lr=1e-3",
        "training.optimizer.weight_decay=0.1",
        "training.optimizer.betas=[0.9,0.95]",
        f"data.source_root={SOURCE_ROOT}",
        f"data.source_manifest_path={SOURCE_MANIFEST}",
        f"data.selection_manifest_path={SELECTION_MANIFEST}",
        f"data.sft_cache_dir={SFT_CACHE}",
        f"data.leg_manifest_path={P1_MANIFEST}",
        f"data.expected_manifest_hash={EXPECTED_P1_MANIFEST_SHA256}",
        "data.num_workers=8",
        f"training.max_steps={BENCHMARK_STEPS}",
        "training.benchmark_only=true",
        f"training.benchmark_warmup_steps={BENCHMARK_WARMUP_STEPS}",
        "training.persistent_workers=true",
        "training.save_interval=0",
        "training.export_interval=0",
        "training.log_interval=1",
        "logging.backend=none",
        "provenance.attention_backend=flash_attention_2",
        f"provenance.flash_attention_version={FLASH_ATTENTION_VERSION}",
        f"provenance.torch_compile_mode={mode}",
        "provenance.data_num_workers=8",
        f"provenance.source_tree_sha256={SOURCE_TREE_SHA256}",
    ]
    command = [
        "accelerate",
        "launch",
        "--multi_gpu",
        "--num_processes",
        "8",
        "--mixed_precision",
        "bf16",
        "--main_process_port",
        "29671",
        TRAIN_CLI,
        "--config",
        BASE_CONFIG,
        "--override",
        *overrides,
    ]
    return command, output


@app.function(
    gpu="H200:8",
    cpu=32.0,
    memory=128 * 1024,
    timeout=60 * 60 * 2,
    retries=0,
    max_containers=2,
)
def benchmark(compile_mode: str = "none") -> dict[str, object]:
    mode = _compile_mode(compile_mode)
    data_volume.reload()
    manifest_path = Path(P1_MANIFEST)
    actual_manifest_hash = _sha256_file(manifest_path)
    if actual_manifest_hash != EXPECTED_P1_MANIFEST_SHA256:
        raise RuntimeError(
            f"P1 manifest drift: {actual_manifest_hash} != "
            f"{EXPECTED_P1_MANIFEST_SHA256}"
        )
    command, output = _command(mode)
    print("[fa2-future] " + " ".join(command), flush=True)
    result = subprocess.run(
        command,
        cwd=REMOTE_REPO,
        stdout=sys.stdout,
        stderr=sys.stderr,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"FA2 benchmark failed with exit {result.returncode}")
    result_path = output / "benchmark_result.json"
    with result_path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    first_loss = float(value["loss_trace"][0]["loss"])
    loss_delta = abs(first_loss - REFERENCE_SDPA_FIRST_LOSS)
    value["sdpa_reference_first_loss"] = REFERENCE_SDPA_FIRST_LOSS
    value["first_loss_absolute_delta"] = loss_delta
    value["loss_parity_abs_tolerance"] = LOSS_PARITY_ABS_TOLERANCE
    value["loss_parity_passed"] = loss_delta <= LOSS_PARITY_ABS_TOLERANCE
    if not value["loss_parity_passed"]:
        raise RuntimeError(
            f"FA2 loss parity failed: delta={loss_delta} > "
            f"{LOSS_PARITY_ABS_TOLERANCE}"
        )
    print("[fa2-future-result] " + json.dumps(value, sort_keys=True))
    return value


@app.local_entrypoint()
def main(compile_mode: str = "none", dry_run: bool = False) -> None:
    mode = _compile_mode(compile_mode)
    if dry_run:
        command, output = _command(mode)
        print(
            json.dumps(
                {
                    "compile_mode": mode,
                    "command": command,
                    "output": str(output),
                    "source_tree_sha256": SOURCE_TREE_SHA256,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    handle = benchmark.spawn(compile_mode=mode)
    print(f"SPAWNED fa2-future-{mode} (function call id: {handle.object_id})")
