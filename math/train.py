"""Modal training app for math LLM pretraining on Nemotron-CC-Math-v1.

Two modes (--mode flag passed through to the inner script):

    stable   WSD warmup + constant LR; produces checkpoints at every save_interval.
             These checkpoints are the fork points for downstream anneals.

    anneal   Loads a stable-phase checkpoint; linear LR decay to zero over --tokens.
             Produces the "final" math model at the chosen token count.

Examples:
    # Single-node H100 smoke test (20 optimizer steps, no checkpoints):
    modal run train.py --gpu-type H100 --tokens 1_000_000_000 --benchmark-steps 20

    # Single-node H200 stable phase, 130B tokens, detached:
    modal run --detach train.py --gpu-type H200 --mode stable --tokens 130_000_000_000

    # 4-node H200 multinode stable phase:
    modal run --detach train.py --gpu-type H200 --nodes 4 --mode stable --tokens 130_000_000_000

    # Anneal from a saved checkpoint, 5B tokens:
    modal run --detach train.py --gpu-type H200 --mode anneal \\
        --load-path /checkpoints/stable-run/step-50000 --tokens 5_000_000_000
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import modal
import modal.experimental

from common import (
    CACHE_MOUNT,
    CACHE_VOLUME_NAME,
    CHECKPOINT_MOUNT,
    CHECKPOINT_VOLUME_NAME,
    TOKENIZED_MOUNT,
    TOKENIZED_VOLUME_NAME,
    UNTRAINED_MOUNT,
    UNTRAINED_VOLUME_NAME,
)

GPUS_PER_NODE = 8
DEFAULT_SEED = 1337

# Repo paths
_this_file = Path(__file__).resolve()
LOCAL_PROJECT_DIR = _this_file.parent
LOCAL_OLMO_CORE_DIR = LOCAL_PROJECT_DIR.parent / "OLMo-core"
REMOTE_OLMO_CORE = "/root/OLMo-core"
REMOTE_PROJECT = "/root/math-pretraining"

# Image: cuda + flash-attn + OLMo-core + our scripts. Adapted from
# OLMo-core/src/scripts/modal/olmo3_7b_anneal_modal.py.
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install(
        "build-essential",
        "curl",
        "git",
        "libhwloc15",
        "libnl-route-3-200",
        "libibverbs-dev",
        "libibverbs1",
        "ibverbs-utils",
        "librdmacm-dev",
        "libibumad-dev",
        "libpci-dev",
        "iproute2",
    )
    .pip_install("wheel", "packaging", "ninja", "setuptools")
    # torch 2.6.0 cannot import the current OLMo-core checkout
    # (torch.compiler.disable(reason=) is a 2.7+ API); 2.8.0 + flash-attn 2.8.3 is
    # the validated pairing (see preflight_convert_roundtrip.py).
    .pip_install("torch==2.8.0")
    .pip_install("flash-attn==2.8.3", extra_options="--no-build-isolation")
    .pip_install("wandb>=0.18")
    .add_local_dir(
        str(LOCAL_OLMO_CORE_DIR),
        remote_path=REMOTE_OLMO_CORE,
        copy=True,
        ignore=[
            ".git",
            ".git/**",
            ".mypy_cache/**",
            ".pytest_cache/**",
            ".ruff_cache/**",
            ".venv/**",
            "__pycache__/**",
            "build/**",
            "dist/**",
            "doc/**",
            "scratch/**",
        ],
    )
    .run_commands(f"cd {REMOTE_OLMO_CORE} && pip install -e .")
    .add_local_dir(
        str(LOCAL_PROJECT_DIR),
        remote_path=REMOTE_PROJECT,
        copy=True,
        ignore=["__pycache__/**", ".venv/**", "*.tmp"],
    )
    # train.py imports `common` directly; add it as a Python module on the worker.
    .add_local_python_source("common")
)

tokenized_volume = modal.Volume.from_name(TOKENIZED_VOLUME_NAME, create_if_missing=True)
untrained_volume = modal.Volume.from_name(UNTRAINED_VOLUME_NAME, create_if_missing=True)
checkpoint_volume = modal.Volume.from_name(
    CHECKPOINT_VOLUME_NAME, create_if_missing=True, version=2
)
cache_volume = modal.Volume.from_name(CACHE_VOLUME_NAME, create_if_missing=True)

wandb_secret = modal.Secret.from_name("wandb-secret")

app = modal.App("math-pretraining-train", image=image)


def _run(
    *,
    gpu_type: str,
    num_nodes: int,
    node_rank: int,
    master_addr: str | None,
    master_port: int,
    mode: str,
    data_glob: str,
    tokens: int,
    warmup_tokens: int,
    lr: float,
    seed: int = DEFAULT_SEED,
    load_path: str | None,
    load_optim: bool,
    save_folder: str,
    run_name: str,
    rank_microbatch_size_tokens: int,
    compile_model: bool,
    num_workers: int,
    save_interval: int,
    ephemeral_save_interval: int,
    save_async: bool,
    benchmark_steps: int | None,
    shard_degree: int | None,
    use_mix: bool = False,
    sequence_length: int = 4096,
) -> dict[str, Any]:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "NCCL_DEBUG": "WARN",
            "NCCL_NVLS_ENABLE": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "HF_HOME": f"{CACHE_MOUNT}/hf",
            "XDG_CACHE_HOME": f"{CACHE_MOUNT}/xdg",
            "CACHED_PATH_CACHE_ROOT": f"{CACHE_MOUNT}/cached-path",
            "OLMO_CORE_FS_CACHE_DIR": f"{CACHE_MOUNT}/olmo-fs-cache",
            "TORCHINDUCTOR_CACHE_DIR": f"{CACHE_MOUNT}/torchinductor",
            "OLMO_ALLOW_LOCAL_NON_SHARED_CHECKPOINT": "1",
        }
    )
    checkpoint_volume.reload()
    tokenized_volume.reload()
    if Path(UNTRAINED_MOUNT).is_dir():
        untrained_volume.reload()

    work_dir = f"{CACHE_MOUNT}/work/{run_name}/node{node_rank}"
    for d in (
        f"{CACHE_MOUNT}/hf",
        f"{CACHE_MOUNT}/xdg",
        f"{CACHE_MOUNT}/cached-path",
        f"{CACHE_MOUNT}/olmo-fs-cache",
        f"{CACHE_MOUNT}/torchinductor",
        save_folder,
        work_dir,
    ):
        Path(d).mkdir(parents=True, exist_ok=True)

    world_size = num_nodes * GPUS_PER_NODE
    if (512 * 4096) % (world_size * rank_microbatch_size_tokens) != 0:
        # GLOBAL_BATCH_SIZE in inner script is 512 * 4096
        raise ValueError(
            "global batch must be divisible by world size * rank microbatch size: "
            f"world_size={world_size} rank_microbatch_size={rank_microbatch_size_tokens}"
        )

    print(
        json.dumps(
            {
                "event": "math_pretrain_start",
                "gpu_type": gpu_type,
                "num_nodes": num_nodes,
                "node_rank": node_rank,
                "master_addr": master_addr,
                "master_port": master_port,
                "mode": mode,
                "tokens": tokens,
                "warmup_tokens": warmup_tokens,
                "lr": lr,
                "seed": seed,
                "load_path": load_path,
                "load_optim": load_optim,
                "save_folder": save_folder,
                "data_glob": data_glob,
                "benchmark_steps": benchmark_steps,
                "rank_microbatch_size_tokens": rank_microbatch_size_tokens,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    torchrun_args = [
        "torchrun",
        f"--nnodes={num_nodes}",
        "--nproc-per-node",
        str(GPUS_PER_NODE),
    ]
    if num_nodes == 1:
        torchrun_args.append("--standalone")
    else:
        if not master_addr:
            raise ValueError("master_addr is required for multi-node torchrun")
        torchrun_args.extend(
            [
                f"--node-rank={node_rank}",
                f"--master-addr={master_addr}",
                f"--master-port={master_port}",
            ]
        )

    inner_script = "train_inner_mix.py" if use_mix else "train_inner.py"
    cmd = [
        *torchrun_args,
        f"{REMOTE_PROJECT}/{inner_script}",
        "--name",
        run_name,
        "--save-folder",
        save_folder,
        "--work-dir",
        work_dir,
        "--mode",
        mode,
        "--tokens",
        str(tokens),
        "--warmup-tokens",
        str(warmup_tokens),
        "--lr",
        str(lr),
        "--rank-microbatch-size-tokens",
        str(rank_microbatch_size_tokens),
        "--sequence-length",
        str(sequence_length),
        "--num-workers",
        str(num_workers),
        "--save-interval",
        str(save_interval),
        "--ephemeral-save-interval",
        str(ephemeral_save_interval),
        "--load-optim" if load_optim else "--no-load-optim",
        "--compile-model" if compile_model else "--no-compile-model",
        "--save-async" if save_async else "--no-save-async",
    ]
    # train_inner.py (single-source) takes --data-glob; train_inner_mix.py uses
    # a hard-coded composable mix over the tokenized volume.
    if use_mix:
        cmd.extend(["--seed", str(seed)])
    else:
        cmd.extend(["--data-glob", data_glob])
    if shard_degree is not None:
        cmd.extend(["--shard-degree", str(shard_degree)])
    if load_path:
        cmd.extend(["--load-path", load_path])
    if benchmark_steps is not None:
        cmd.extend(["--benchmark-steps", str(benchmark_steps)])

    print("[modal] running:", " ".join(cmd), flush=True)
    start = time.time()
    process = subprocess.Popen(
        cmd,
        cwd=REMOTE_OLMO_CORE,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        lines.append(line.rstrip("\n"))

    return_code = process.wait()
    elapsed = time.time() - start
    if not save_folder.startswith(("gs://", "s3://", "r2://", "weka://")):
        checkpoint_volume.commit()

    summary = {
        "return_code": return_code,
        "gpu_type": gpu_type,
        "num_nodes": num_nodes,
        "node_rank": node_rank,
        "mode": mode,
        "tokens": tokens,
        "save_folder": save_folder,
        "elapsed_seconds_including_startup": elapsed,
    }
    print("[modal] summary:", json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)
    return summary


def _cluster_ips(cluster_info: Any) -> list[str]:
    return list(
        getattr(cluster_info, "container_ips", None)
        or getattr(cluster_info, "container_ipv4_ips", None)
        or []
    )


@app.function(
    gpu="H100:8",
    timeout=60 * 60 * 24,
    retries=modal.Retries(max_retries=10, initial_delay=60, backoff_coefficient=1.0),
    volumes={
        CHECKPOINT_MOUNT: checkpoint_volume,
        CACHE_MOUNT: cache_volume,
        TOKENIZED_MOUNT: tokenized_volume,
        UNTRAINED_MOUNT: untrained_volume,
    },
    secrets=[wandb_secret],
)
def run_h100(**kwargs: Any) -> dict[str, Any]:
    return _run(
        gpu_type="H100",
        num_nodes=1,
        node_rank=0,
        master_addr=None,
        master_port=29500,
        **kwargs,
    )


@app.function(
    gpu="H200:8",
    timeout=60 * 60 * 24,
    retries=modal.Retries(max_retries=10, initial_delay=60, backoff_coefficient=1.0),
    volumes={
        CHECKPOINT_MOUNT: checkpoint_volume,
        CACHE_MOUNT: cache_volume,
        TOKENIZED_MOUNT: tokenized_volume,
        UNTRAINED_MOUNT: untrained_volume,
    },
    secrets=[wandb_secret],
)
def run_h200(**kwargs: Any) -> dict[str, Any]:
    return _run(
        gpu_type="H200",
        num_nodes=1,
        node_rank=0,
        master_addr=None,
        master_port=29500,
        **kwargs,
    )


@app.function(
    gpu="H200:8",
    experimental_options={"efa_enabled": True},
    timeout=60 * 60 * 24,
    retries=modal.Retries(max_retries=10, initial_delay=60, backoff_coefficient=1.0),
    volumes={
        CHECKPOINT_MOUNT: checkpoint_volume,
        CACHE_MOUNT: cache_volume,
        TOKENIZED_MOUNT: tokenized_volume,
        UNTRAINED_MOUNT: untrained_volume,
    },
    secrets=[wandb_secret],
)
@modal.experimental.clustered(size=4, rdma=True)
def run_h200_4node(**kwargs: Any) -> dict[str, Any]:
    cluster_info = modal.experimental.get_cluster_info()
    node_rank = int(cluster_info.rank)
    ips = _cluster_ips(cluster_info)
    if not ips:
        raise RuntimeError(f"Modal cluster info did not include container IPs: {cluster_info}")
    return _run(
        gpu_type="H200",
        num_nodes=4,
        node_rank=node_rank,
        master_addr=ips[0],
        master_port=29500,
        **kwargs,
    )


@app.local_entrypoint()
def main(
    gpu_type: str = "H200",
    nodes: int = 1,
    mode: str = "stable",
    tokens: int = 200_000_000_000,
    warmup_tokens: int = 2_000_000_000,
    lr: float = 4e-4,
    seed: int = DEFAULT_SEED,
    load_path: str | None = None,
    load_optim: bool = False,
    save_folder: str | None = None,
    run_name: str = "math-1b-v0",
    use_mix: bool = True,
    data_glob: str = f"{TOKENIZED_MOUNT}/3/part_*.npy",
    rank_microbatch_size_tokens: int = 8 * 4096,
    compile_model: bool = True,
    num_workers: int = 4,
    save_interval: int = 2500,
    ephemeral_save_interval: int = 250,
    save_async: bool = False,
    benchmark_steps: int = 0,
    shard_degree: int = 0,
    sequence_length: int = 4096,
) -> None:
    gpu_type_upper = gpu_type.upper()
    resolved_benchmark_steps = benchmark_steps if benchmark_steps > 0 else None
    resolved_save_folder = save_folder or f"{CHECKPOINT_MOUNT}/{run_name}"
    resolved_shard_degree = shard_degree if shard_degree > 0 else None

    kwargs = {
        "mode": mode,
        "data_glob": data_glob,
        "tokens": tokens,
        "warmup_tokens": warmup_tokens,
        "lr": lr,
        "seed": seed,
        "load_path": load_path,
        "load_optim": load_optim,
        "save_folder": resolved_save_folder,
        "run_name": run_name,
        "rank_microbatch_size_tokens": rank_microbatch_size_tokens,
        "compile_model": compile_model,
        "num_workers": num_workers,
        "save_interval": save_interval,
        "ephemeral_save_interval": ephemeral_save_interval,
        "save_async": save_async,
        "benchmark_steps": resolved_benchmark_steps,
        "shard_degree": resolved_shard_degree,
        "use_mix": use_mix,
        "sequence_length": sequence_length,
    }

    if nodes == 1 and gpu_type_upper == "H100":
        summary = run_h100.remote(**kwargs)
    elif nodes == 1 and gpu_type_upper == "H200":
        summary = run_h200.remote(**kwargs)
    elif nodes == 4 and gpu_type_upper == "H200":
        summary = run_h200_4node.remote(**kwargs)
    else:
        raise ValueError("supported combos: H100/H200 with nodes=1, or H200 with nodes=4")

    json.dump(summary, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
