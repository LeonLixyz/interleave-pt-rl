"""Inner training script — runs under `torchrun` inside the Modal training container.

Two modes (selected via --mode):
    stable   Warmup-Stable phase: `WSD(warmup=..., decay=0)`. Constant LR after
             warmup; periodic checkpoints define the fork points.
    anneal   Fork-and-anneal: load a stable-phase checkpoint, decay LR linearly
             to zero over --tokens. Used to produce the "final" model at a
             given token milestone.

The model config is `TransformerConfig.olmo2_1B_v2()`. Tokenizer is dolma2.
Data is read from a list of pre-tokenized `.npy` paths produced by
`tokenize_corpus.py`.
"""

from __future__ import annotations

import argparse
import glob
import inspect
import os
from pathlib import Path
from typing import List

import torch

# Match the OLMo3 anneal script's LOCAL_RANK shim
if "LOCAL_RANK" in os.environ:
    os.environ.setdefault("FS_LOCAL_RANK", os.environ["LOCAL_RANK"])

_torch_compiler_disable = torch.compiler.disable
if "reason" not in inspect.signature(_torch_compiler_disable).parameters:

    def _torch_compiler_disable_compat(fn=None, recursive=True, reason=None):
        del reason
        return _torch_compiler_disable(fn=fn, recursive=recursive)

    torch.compiler.disable = _torch_compiler_disable_compat

from olmo_core.config import DType
from olmo_core.data import (
    NumpyDataLoaderConfig,
    NumpyFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.data.types import NumpyDatasetDType
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import Float8Config
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.nn.transformer.config import TransformerActivationCheckpointingMode
from olmo_core.optim import (
    ConstantWithWarmup,
    LinearWithWarmup,
    OptimGroupOverride,
    SkipStepAdamWConfig,
)
from olmo_core.script_utils import ExperimentConfig, get_cli_parser, main
from olmo_core.train import Duration, TrainerConfig
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    CheckpointRemovalStrategy,
    ConfigSaverCallback,
)
from olmo_core.train.common import LoadStrategy
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)
from olmo_core.train.train_module.transformer.config import (
    TransformerActivationCheckpointingConfig,
)

DEFAULT_SEQUENCE_LENGTH = 4096
GLOBAL_BATCH_SIZE = 512 * DEFAULT_SEQUENCE_LENGTH  # ~2M tokens, matches OLMo2-1B
DEFAULT_RANK_MICROBATCH_SIZE = 8 * DEFAULT_SEQUENCE_LENGTH
DEFAULT_LR = 4e-4  # OLMo2-1B uses 4e-4; we keep the same for v0
SEED = 1337


def get_parser() -> argparse.ArgumentParser:
    parser = get_cli_parser()
    parser.add_argument(
        "--mode",
        choices=["stable", "anneal"],
        default="stable",
        help="`stable` = WSD warmup+constant (with periodic checkpoints); "
        "`anneal` = load checkpoint and linearly decay LR to zero.",
    )
    parser.add_argument(
        "--data-glob",
        required=True,
        help="Glob pattern selecting pre-tokenized .npy files (e.g. /tokenized/3/part_*.npy).",
    )
    parser.add_argument(
        "--load-path",
        default=None,
        help="Checkpoint to load (required for --mode anneal).",
    )
    parser.add_argument(
        "--load-optim",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Load optimizer state from --load-path. Defaults to False for anneal "
        "(fresh optimizer is the common choice).",
    )
    parser.add_argument(
        "--tokens",
        type=int,
        required=True,
        help="Target token budget for this run.",
    )
    parser.add_argument(
        "--warmup-tokens",
        type=int,
        default=2_000_000_000,
        help="Warmup length in tokens (stable mode only). Ignored in anneal mode.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LR,
        help="Peak learning rate.",
    )
    parser.add_argument(
        "--rank-microbatch-size-tokens",
        type=int,
        default=DEFAULT_RANK_MICROBATCH_SIZE,
    )
    parser.add_argument(
        "--shard-degree",
        type=int,
        default=None,
        help="HSDP shard degree; defaults to OLMo-core's one-shard-group-per-node behavior.",
    )
    parser.add_argument(
        "--compile-model",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=2500,
        help="Permanent checkpoint interval (optimizer steps). Set <=0 to disable.",
    )
    parser.add_argument(
        "--ephemeral-save-interval",
        type=int,
        default=250,
        help="Rolling ephemeral checkpoint interval (optimizer steps). Set <=0 to disable.",
    )
    parser.add_argument(
        "--save-async",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--benchmark-steps",
        type=int,
        default=None,
        help="Hard-stop after this many optimizer steps (smoke-test/benchmark only).",
    )
    return parser


def _optional_interval(value: int) -> int | None:
    return value if value > 0 else None


def _resolve_paths(pattern: str) -> List[str]:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no files matched data glob: {pattern}")
    return paths


def build_config(opts: argparse.Namespace, overrides: List[str]) -> ExperimentConfig:
    sequence_length = opts.sequence_length or DEFAULT_SEQUENCE_LENGTH
    rank_microbatch_size = opts.rank_microbatch_size_tokens
    if rank_microbatch_size % sequence_length != 0:
        raise ValueError(
            f"rank microbatch size {rank_microbatch_size} must be divisible by sequence length "
            f"{sequence_length}"
        )
    if GLOBAL_BATCH_SIZE % rank_microbatch_size != 0:
        raise ValueError(
            f"global batch size {GLOBAL_BATCH_SIZE} must be divisible by rank microbatch size "
            f"{rank_microbatch_size}"
        )

    if opts.mode == "anneal" and not opts.load_path:
        raise ValueError("--load-path is required for --mode anneal")

    tokenizer_config = TokenizerConfig.dolma2()

    model_config = TransformerConfig.olmo2_1B_v2(
        vocab_size=tokenizer_config.padded_vocab_size(),
        attn_backend=AttentionBackendName.flash_2,
    )

    data_paths = _resolve_paths(opts.data_glob)
    print(f"[train_inner] data: {len(data_paths)} files matched {opts.data_glob}")
    for p in data_paths[:3]:
        print("  ", p)
    if len(data_paths) > 3:
        print(f"  ... and {len(data_paths) - 3} more")

    dataset_config = NumpyFSLDatasetConfig(
        paths=data_paths,
        tokenizer=tokenizer_config,
        sequence_length=sequence_length,
        max_target_sequence_length=max(DEFAULT_SEQUENCE_LENGTH, sequence_length),
        dtype=NumpyDatasetDType.uint32,
        work_dir=opts.work_dir,
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=GLOBAL_BATCH_SIZE,
        seed=SEED,
        num_workers=opts.num_workers,
        ignore_fingerprint_mismatch=True,
    )

    if opts.mode == "stable":
        # Warmup, then constant LR. Anneal happens later, in a separate "anneal" run
        # forked from one of the checkpoints produced here (this gives us the
        # OLMo2 / MiniCPM WSD-style fork-and-anneal recipe).
        scheduler = ConstantWithWarmup(
            warmup=opts.warmup_tokens,
            warmup_min_lr=0.0,
        )
    else:
        # Anneal: linear decay from peak LR to 0 over the full budget, no warmup.
        scheduler = LinearWithWarmup(warmup=0, alpha_f=0.0)

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=rank_microbatch_size,
        max_sequence_length=sequence_length,
        optim=SkipStepAdamWConfig(
            lr=opts.lr,
            weight_decay=0.033,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
        ),
        scheduler=scheduler,
        compile_model=opts.compile_model,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            shard_degree=opts.shard_degree,
            # Wrap per-block instead of as one big unit so FSDP can actually free
            # activations between blocks (the default whole-model wrap causes OOM
            # for 1B+ models on H100).
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

    # In stable mode we may want to load an earlier stable-phase checkpoint to resume training.
    # In anneal mode we load weights only.
    load_strategy = LoadStrategy.never if opts.load_path else LoadStrategy.if_available

    trainer_config = (
        TrainerConfig(
            save_folder=opts.save_folder,
            save_overwrite=True,
            load_path=None,  # Loading is handled via ExperimentConfig.load_path below
            load_strategy=load_strategy,
            load_trainer_state=(opts.mode == "stable" and opts.load_path is not None),
            load_optim_state=opts.load_optim,
            max_duration=Duration.tokens(opts.tokens),
            hard_stop=None
            if opts.benchmark_steps is None
            else Duration.steps(opts.benchmark_steps),
            work_dir=opts.work_dir,
            metrics_collect_interval=10,
            no_evals=True,
        )
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=_optional_interval(opts.save_interval),
                ephemeral_save_interval=_optional_interval(opts.ephemeral_save_interval),
                save_async=opts.save_async,
                remove=CheckpointRemovalStrategy.ephemeral_only,
                enabled=opts.benchmark_steps is None,
            ),
        )
    )

    return ExperimentConfig(
        model=model_config,
        dataset=dataset_config,
        data_loader=data_loader_config,
        train_module=train_module_config,
        trainer=trainer_config,
        init_seed=SEED,
        load_path=opts.load_path,
    ).merge(overrides)


if __name__ == "__main__":
    main(build_config, parser=get_parser())
