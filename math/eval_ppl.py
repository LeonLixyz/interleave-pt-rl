"""Modal app for held-out perplexity evaluation on the EXTRACTED UNTRAINED corpus.

Loads a single OLMo-core checkpoint (e.g. /checkpoints/math-1b-v0/step89000) and
computes cross-entropy / perplexity on the held-out eval slice carved from the
instance-level untrained corpus (see extract_untrained.py).

Two top-level eval modes (both enabled by default):
    dolma3   100% dolma3 from the untrained corpus eval slice.
    mixture  Renormalized 70/30 mixture across (math_3, math_4plus_MIND, dolma3)
             — matches the anneal/training distribution.

Per-source PPL is reported by running an extra eval pass per leaf source (cheap
since each eval is ~1B tokens). The eval slice is determined by a SEEDED SPLIT
of the untrained corpus (see ANNEAL_EVAL_SPLIT_SEED in train_inner_mix.py); we
take the LAST `eval_ratio` fraction so the eval set is disjoint from anneal.

Results are written to /checkpoints/evals/<run_name>/<checkpoint_stepNNN>.json
on the olmo-core-checkpoints-v2 volume.

Examples:
    # Evaluate the step89000 checkpoint on both dolma3 + mixture (default ~1B tokens):
    modal run eval_ppl.py --checkpoint /checkpoints/math-1b-v0/step89000

    # Cheap smoke test, dolma3 only, 100M tokens:
    modal run eval_ppl.py --checkpoint /checkpoints/math-1b-v0/step89000 \\
        --tokens 100_000_000 --eval-mode dolma3
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import modal

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

# Repo paths (mirror train.py exactly so the image build is shared/cached).
_this_file = Path(__file__).resolve()
LOCAL_PROJECT_DIR = _this_file.parent
LOCAL_OLMO_CORE_DIR = LOCAL_PROJECT_DIR.parent / "OLMo-core"
REMOTE_OLMO_CORE = "/root/OLMo-core"
REMOTE_PROJECT = "/root/math-pretraining"

# Image: same recipe as train.py (cuda 12.4 + torch 2.6 + flash-attn + OLMo-core
# editable install + math-pretraining repo + `common` python source).
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
    .pip_install("torch==2.6.0")
    .pip_install("flash-attn==2.7.4.post1", extra_options="--no-build-isolation")
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
    .add_local_python_source("common")
)

tokenized_volume = modal.Volume.from_name(TOKENIZED_VOLUME_NAME, create_if_missing=True)
untrained_volume = modal.Volume.from_name(UNTRAINED_VOLUME_NAME, create_if_missing=True)
checkpoint_volume = modal.Volume.from_name(
    CHECKPOINT_VOLUME_NAME, create_if_missing=True, version=2
)
cache_volume = modal.Volume.from_name(CACHE_VOLUME_NAME, create_if_missing=True)

app = modal.App("math-pretraining-eval-ppl", image=image)


# ---------------------------------------------------------------------------
# Worker: runs under torchrun in a single H200:8 container.
# ---------------------------------------------------------------------------


def _worker_main() -> None:
    """Forward-only PPL eval driver. Runs once per torchrun rank."""
    import argparse
    import inspect
    import logging

    import torch

    if "LOCAL_RANK" in os.environ:
        os.environ.setdefault("FS_LOCAL_RANK", os.environ["LOCAL_RANK"])

    # Same torch.compiler.disable shim as train_inner_mix.py
    _torch_compiler_disable = torch.compiler.disable
    if "reason" not in inspect.signature(_torch_compiler_disable).parameters:

        def _torch_compiler_disable_compat(fn=None, recursive=True, reason=None):
            del reason
            return _torch_compiler_disable(fn=fn, recursive=recursive)

        torch.compiler.disable = _torch_compiler_disable_compat

    from olmo_core.config import DType
    from olmo_core.data import TokenizerConfig
    from olmo_core.data.composable import (
        ComposableDataLoaderConfig,
        MixingInstanceSource,
    )
    from olmo_core.data.utils import get_labels
    from olmo_core.distributed.parallel import DataParallelType
    from olmo_core.float8 import Float8Config
    from olmo_core.nn.attention import AttentionBackendName
    from olmo_core.nn.transformer import TransformerConfig
    from olmo_core.nn.transformer.config import TransformerActivationCheckpointingMode
    from olmo_core.optim import (
        ConstantWithWarmup,
        OptimGroupOverride,
        SkipStepAdamWConfig,
    )
    from olmo_core.train import (
        Duration,
        Trainer,
        TrainerConfig,
        prepare_training_environment,
        teardown_training_environment,
    )
    from olmo_core.train.train_module import (
        TransformerDataParallelConfig,
        TransformerDataParallelWrappingStrategy,
        TransformerTrainModuleConfig,
    )
    from olmo_core.train.train_module.transformer.config import (
        TransformerActivationCheckpointingConfig,
    )
    from olmo_core.utils import get_default_device, seed_all

    # Re-use mix builders from the training script so the eval slice definition
    # stays perfectly in sync with anneal.
    sys.path.insert(0, REMOTE_PROJECT)
    from train_inner_mix import (
        ANNEAL_EVAL_SPLIT_SEED,
        DEFAULT_ANNEAL_EVAL_RATIO,
        UNTRAINED_PATHS,
        UNTRAINED_WEIGHTS,
        _untrained_chunk,
        build_eval_mix_dolma3,
        build_eval_mix_mixture,
    )

    log = logging.getLogger("eval_ppl")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")

    SEQUENCE_LENGTH = 4096
    SEED = 1337
    # GLOBAL_BATCH_SIZE for eval is set later, AFTER world_size is known, to
    # rank_microbatch_size_tokens * world_size — no gradient accumulation, so each
    # loader iteration yields exactly one microbatch per rank. This avoids the
    # 98GiB logits OOM that happened when we used the training-time 512*4096 GBS
    # (eval_batch processes the full rank batch in one shot, not microbatched).

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokens", type=int, default=1_000_000_000)
    parser.add_argument(
        "--eval-mode", choices=["dolma3", "mixture", "both"], default="both"
    )
    parser.add_argument("--eval-ratio", type=float, default=DEFAULT_ANNEAL_EVAL_RATIO)
    parser.add_argument(
        "--rank-microbatch-size-tokens", type=int, default=4 * SEQUENCE_LENGTH
    )
    parser.add_argument("--run-name", default="eval-math-1b-v0")
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--save-folder", required=True)
    parser.add_argument(
        "--output-json", required=True, help="Where rank 0 writes the result JSON."
    )
    parser.add_argument(
        "--per-source",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run an extra eval pass per leaf source for per-source PPL.",
    )
    opts = parser.parse_args()

    if opts.rank_microbatch_size_tokens % SEQUENCE_LENGTH != 0:
        raise ValueError(
            "rank microbatch must be divisible by sequence length "
            f"({opts.rank_microbatch_size_tokens=} {SEQUENCE_LENGTH=})"
        )

    backend = "cpu:gloo,cuda:nccl" if torch.cuda.is_available() else None
    prepare_training_environment(shared_filesystem=False, backend=backend)
    seed_all(SEED)
    device = get_default_device()

    is_rank_zero = int(os.environ.get("RANK", "0")) == 0
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    GLOBAL_BATCH_SIZE = opts.rank_microbatch_size_tokens * world_size
    if is_rank_zero:
        log.info(
            "[eval] world_size=%d rank_microbatch_tokens=%d -> global_batch_tokens=%d",
            world_size, opts.rank_microbatch_size_tokens, GLOBAL_BATCH_SIZE,
        )

    tokenizer = TokenizerConfig.dolma2()

    # Match training model config exactly.
    model_config = TransformerConfig.olmo2_1B_v2(
        vocab_size=tokenizer.padded_vocab_size(),
        attn_backend=AttentionBackendName.flash_2,
    )

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=opts.rank_microbatch_size_tokens,
        max_sequence_length=SEQUENCE_LENGTH,
        # The optimizer / scheduler are unused (no .step() is taken) but must be
        # configured so the train_module builds.
        optim=SkipStepAdamWConfig(
            lr=4e-4,
            weight_decay=0.033,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
        ),
        scheduler=ConstantWithWarmup(warmup=1, warmup_min_lr=0.0),
        compile_model=False,  # forward-only — skip the compile cost
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.blocks,
        ),
        ac_config=TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.selected_modules,
            modules=["blocks.*.feed_forward"],
        ),
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
    )

    model = model_config.build(init_device="meta")
    train_module = train_module_config.build(model)

    data_loader_config = ComposableDataLoaderConfig(
        tokenizer=tokenizer,
        global_batch_size=GLOBAL_BATCH_SIZE,
        seed=SEED,
        work_dir=opts.work_dir,
        num_workers=0,  # eval is short-lived; avoid the cost of worker spinup
    )

    # ------- Build the eval passes we'll run -------
    # Each entry: (name, mix_config_builder_fn). Each builder takes total_tokens
    # and returns a MixingInstanceSource.Config sized to that token budget.
    passes: list[tuple[str, Any]] = []
    if opts.eval_mode in ("dolma3", "both"):
        passes.append(
            (
                "dolma3",
                lambda n: build_eval_mix_dolma3(
                    sequence_length=SEQUENCE_LENGTH,
                    total_tokens=n,
                    tokenizer=tokenizer,
                    eval_ratio=opts.eval_ratio,
                ),
            )
        )
    if opts.eval_mode in ("mixture", "both"):
        passes.append(
            (
                "mixture",
                lambda n: build_eval_mix_mixture(
                    sequence_length=SEQUENCE_LENGTH,
                    total_tokens=n,
                    tokenizer=tokenizer,
                    eval_ratio=opts.eval_ratio,
                ),
            )
        )

    # Per-source passes: one MixingInstanceSource with a single spec (weight=1.0)
    # per leaf label, drawing from that source's eval slice. Smaller token budget
    # per source so the wall clock stays reasonable.
    per_source_token_budget = max(SEQUENCE_LENGTH * 1024, opts.tokens // 4)
    if opts.per_source:
        for label in UNTRAINED_PATHS:
            def _make(lbl: str):
                def builder(n: int):
                    chunk = _untrained_chunk(lbl, SEQUENCE_LENGTH, tokenizer)
                    _, eval_cfg = chunk.split(
                        ratio=1.0 - opts.eval_ratio, seed=ANNEAL_EVAL_SPLIT_SEED
                    )
                    return MixingInstanceSource.Config(
                        source_specs=[
                            MixingInstanceSource.Spec.Config(
                                source=eval_cfg, ratio=1.0, label=lbl
                            ),
                        ],
                        num_tokens=n,
                    )
                return builder

            passes.append((f"per_source:{label}", _make(label)))

    # Build the trainer ONCE with a placeholder data loader, load the checkpoint,
    # then swap the data loader per eval pass. This keeps the model load and FSDP
    # setup cost paid exactly once.
    first_mix_config = passes[0][1](opts.tokens)
    first_mix = first_mix_config.build(opts.work_dir)
    if is_rank_zero:
        first_mix.visualize()
    data_loader = data_loader_config.build(
        first_mix,
        dp_process_group=train_module.dp_process_group,
        work_dir=opts.work_dir,
        tokenizer=tokenizer,
    )
    trainer_config = TrainerConfig(
        save_folder=opts.save_folder,
        save_overwrite=True,
        max_duration=Duration.tokens(opts.tokens),
        work_dir=opts.work_dir,
        metrics_collect_interval=10,
        no_evals=True,
    )
    trainer: Trainer = trainer_config.build(train_module, data_loader)

    log.info("Loading checkpoint %s (weights only)...", opts.checkpoint)
    t0 = time.time()
    trainer.load_checkpoint(opts.checkpoint, load_trainer_state=False)
    log.info("Checkpoint loaded in %.1fs", time.time() - t0)

    results: dict[str, dict[str, Any]] = {}

    def _run_one_pass(name: str, builder, total_tokens: int) -> dict[str, Any]:
        log.info("[eval:%s] building mix for %.2fM tokens", name, total_tokens / 1e6)
        mix_cfg = builder(total_tokens)
        built = mix_cfg.build(opts.work_dir)
        if is_rank_zero:
            built.visualize()
        loader = data_loader_config.build(
            built,
            dp_process_group=train_module.dp_process_group,
            work_dir=opts.work_dir,
            tokenizer=tokenizer,
        )
        loader.reshuffle(epoch=1)

        total_loss = torch.zeros((), dtype=torch.float64, device=device)
        total_tokens_acc = torch.zeros((), dtype=torch.float64, device=device)
        batches_seen = 0

        t_start = time.time()
        with torch.inference_mode():
            for batch in loader:
                # Move batch tensors to device.
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(device, non_blocking=True)

                labels = get_labels(batch, label_ignore_index=-100)

                output = train_module.eval_batch(batch, labels=labels)
                # `output` is LMOutputWithLoss(logits, loss, ce_loss, z_loss).
                # ce_loss shape can be (B, S) or (B, S-1) depending on OLMo-core
                # version — align valid_mask to whatever it is.
                ce_loss = output.ce_loss
                sl = ce_loss.shape[-1]
                if sl == labels.shape[-1]:
                    valid_mask = labels != -100
                    lm_aligned = batch["label_mask"] if "label_mask" in batch else None
                elif sl == labels.shape[-1] - 1:
                    valid_mask = labels[..., 1:] != -100
                    lm_aligned = (
                        batch["label_mask"][..., 1:] if "label_mask" in batch else None
                    )
                else:
                    raise RuntimeError(
                        f"unexpected ce_loss shape {ce_loss.shape} vs labels {labels.shape}"
                    )
                if lm_aligned is not None:
                    valid_mask = valid_mask & lm_aligned.to(torch.bool)

                ce_loss = ce_loss.to(torch.float64)
                masked = torch.where(valid_mask, ce_loss, torch.zeros_like(ce_loss))
                total_loss += masked.sum()
                total_tokens_acc += valid_mask.to(torch.float64).sum()
                batches_seen += 1

        # All-reduce across ranks.
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(total_loss, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(total_tokens_acc, op=torch.distributed.ReduceOp.SUM)

        n_tokens = int(total_tokens_acc.item())
        loss = float(total_loss.item() / max(1, n_tokens))
        ppl = math.exp(min(loss, 50.0))  # clamp to avoid math.exp overflow
        elapsed = time.time() - t_start
        log.info(
            "[eval:%s] loss=%.4f ppl=%.3f n_tokens=%d batches=%d (%.1fs)",
            name, loss, ppl, n_tokens, batches_seen, elapsed,
        )
        return {"loss": loss, "ppl": ppl, "n_tokens": n_tokens, "elapsed_s": elapsed}

    for name, builder in passes:
        results[name] = _run_one_pass(
            name,
            builder,
            opts.tokens if not name.startswith("per_source:") else per_source_token_budget,
        )

    # Split out top-level vs per_source for cleaner JSON.
    top: dict[str, Any] = {}
    per_source: dict[str, Any] = {}
    for k, v in results.items():
        if k.startswith("per_source:"):
            per_source[k.split(":", 1)[1]] = v
        else:
            top[k] = v

    # Take the LAST step-N match, not the first — paths like
    # /checkpoints/math-1b-v0-anneal-step95368/step500 must resolve to step500
    # (the leaf), not step95368 (the run-name prefix).
    step_matches = re.findall(r"step(\d+)", opts.checkpoint)
    step = int(step_matches[-1]) if step_matches else None
    total_eval_tokens = sum(v["n_tokens"] for v in results.values())

    payload = {
        "checkpoint": opts.checkpoint,
        "step": step,
        "tokens_evaluated": total_eval_tokens,
        "eval_ratio": opts.eval_ratio,
        "eval_split_seed": ANNEAL_EVAL_SPLIT_SEED,
        "untrained_weights": UNTRAINED_WEIGHTS,
        "results": {**top, "per_source": per_source},
    }

    if is_rank_zero:
        out = Path(opts.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True))
        log.info("Wrote results to %s", out)
        print("=" * 78)
        print(f"EVAL RESULTS: {opts.checkpoint} (step {step})")
        print("=" * 78)
        print(f"{'pass':24s} {'loss':>10s} {'ppl':>10s} {'n_tokens':>14s}")
        print("-" * 78)
        for name, r in top.items():
            print(f"{name:24s} {r['loss']:>10.4f} {r['ppl']:>10.3f} {r['n_tokens']:>14,}")
        for name, r in per_source.items():
            print(
                f"  per_source:{name:12s} {r['loss']:>10.4f} {r['ppl']:>10.3f} "
                f"{r['n_tokens']:>14,}"
            )
        print("=" * 78)
        print(f"Wrote: {out}")
        sys.stdout.flush()

    teardown_training_environment()


# ---------------------------------------------------------------------------
# Modal launcher: torchrun on a single H200:8 node.
# ---------------------------------------------------------------------------


def _launch(
    *,
    checkpoint: str,
    tokens: int,
    eval_mode: str,
    eval_ratio: float,
    rank_microbatch_size_tokens: int,
    run_name: str,
    per_source: bool,
) -> dict[str, Any]:
    import subprocess

    checkpoint_volume.reload()
    if Path(UNTRAINED_MOUNT).is_dir():
        untrained_volume.reload()

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

    # See _worker_main: use the LAST step-N (leaf), not the first (run-name prefix).
    step_matches = re.findall(r"step(\d+)", checkpoint)
    step_str = f"step{step_matches[-1]}" if step_matches else "unknown"

    work_dir = f"{CACHE_MOUNT}/work/{run_name}/eval-{step_str}"
    eval_dir = f"{CHECKPOINT_MOUNT}/evals/{run_name}"
    output_json = f"{eval_dir}/{step_str}.json"
    save_folder = f"{eval_dir}/_trainer_workdir/{step_str}"

    for d in (
        f"{CACHE_MOUNT}/hf",
        f"{CACHE_MOUNT}/xdg",
        f"{CACHE_MOUNT}/cached-path",
        f"{CACHE_MOUNT}/olmo-fs-cache",
        f"{CACHE_MOUNT}/torchinductor",
        work_dir,
        eval_dir,
        save_folder,
    ):
        Path(d).mkdir(parents=True, exist_ok=True)

    print(
        json.dumps(
            {
                "event": "eval_ppl_start",
                "checkpoint": checkpoint,
                "step": step_str,
                "tokens": tokens,
                "eval_mode": eval_mode,
                "eval_ratio": eval_ratio,
                "output_json": output_json,
                "rank_microbatch_size_tokens": rank_microbatch_size_tokens,
                "per_source": per_source,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    cmd = [
        "torchrun",
        "--nnodes=1",
        f"--nproc-per-node={GPUS_PER_NODE}",
        "--standalone",
        f"{REMOTE_PROJECT}/eval_ppl.py",
        "--worker",
        "--checkpoint",
        checkpoint,
        "--tokens",
        str(tokens),
        "--eval-mode",
        eval_mode,
        "--eval-ratio",
        str(eval_ratio),
        "--rank-microbatch-size-tokens",
        str(rank_microbatch_size_tokens),
        "--run-name",
        run_name,
        "--work-dir",
        work_dir,
        "--save-folder",
        save_folder,
        "--output-json",
        output_json,
        "--per-source" if per_source else "--no-per-source",
    ]

    print("[modal] running:", " ".join(cmd), flush=True)
    t0 = time.time()
    process = subprocess.Popen(
        cmd,
        cwd=REMOTE_OLMO_CORE,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
    return_code = process.wait()
    elapsed = time.time() - t0

    checkpoint_volume.commit()

    summary = {
        "return_code": return_code,
        "checkpoint": checkpoint,
        "step": step_str,
        "tokens": tokens,
        "eval_mode": eval_mode,
        "output_json": output_json,
        "elapsed_seconds": elapsed,
    }
    print("[modal] summary:", json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)

    # Attach the parsed results so the caller can see them in the function return.
    try:
        summary["results"] = json.loads(Path(output_json).read_text())
    except Exception as e:  # pragma: no cover
        summary["results_error"] = repr(e)
    return summary


@app.function(
    gpu="H200:8",
    timeout=60 * 60 * 6,
    retries=modal.Retries(max_retries=3, initial_delay=60, backoff_coefficient=1.0),
    volumes={
        CHECKPOINT_MOUNT: checkpoint_volume,
        CACHE_MOUNT: cache_volume,
        TOKENIZED_MOUNT: tokenized_volume,
        UNTRAINED_MOUNT: untrained_volume,
    },
)
def run_eval(**kwargs: Any) -> dict[str, Any]:
    return _launch(**kwargs)


@app.local_entrypoint()
def main(
    checkpoint: str,
    tokens: int = 1_000_000_000,
    eval_mode: str = "both",
    eval_ratio: float = 0.15,
    rank_microbatch_size_tokens: int = 4 * 4096,
    run_name: str = "eval-math-1b-v0",
    per_source: bool = True,
) -> None:
    if eval_mode not in ("dolma3", "mixture", "both"):
        raise ValueError(f"--eval-mode must be one of dolma3/mixture/both, got {eval_mode!r}")

    summary = run_eval.remote(
        checkpoint=checkpoint,
        tokens=tokens,
        eval_mode=eval_mode,
        eval_ratio=eval_ratio,
        rank_microbatch_size_tokens=rank_microbatch_size_tokens,
        run_name=run_name,
        per_source=per_source,
    )
    json.dump(summary, sys.stdout, indent=2, sort_keys=True, default=str)
    sys.stdout.write("\n")
    print(f"\nResults JSON: {summary.get('output_json')}", flush=True)


# When invoked under torchrun (--worker flag) this file dispatches into the
# distributed worker. The local entrypoint is bypassed in that path because
# torchrun starts the script via `python eval_ppl.py --worker ...`.
if __name__ == "__main__" and "--worker" in sys.argv:
    # Drop the --worker flag before argparse runs in the worker.
    sys.argv.remove("--worker")
    _worker_main()
