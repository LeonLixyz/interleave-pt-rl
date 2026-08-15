"""Multi-source pretraining via OLMo-core's composable data API.

Mixes Nemotron-CC-Math (3, 4plus, 4plus_MIND) and Dolmino-100B at the configured
ratios (default: 70% math / 30% dolmino; math sub-mix 30/30/40 across the three
Nemotron subsets). Single pass, WSD-style schedule. Anneal mode loads a stable
checkpoint and linearly decays LR to zero on the same mix.

Driver replaces `script_utils.main()` (which is hardcoded to NumpyDatasetConfig)
with a minimal hand-wired path that wires the composable instance source mix
into the existing Trainer / TransformerTrainModule plumbing.
"""

from __future__ import annotations

import argparse
import inspect
import logging
import os
import sys
from typing import cast

import torch

if "LOCAL_RANK" in os.environ:
    os.environ.setdefault("FS_LOCAL_RANK", os.environ["LOCAL_RANK"])

# torch.compiler.disable shim — same as olmo3 anneal script
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
    ConcatAndChunkInstanceSource,
    MixingInstanceSource,
    NumpyDocumentSource,
)
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
from olmo_core.train import (
    Duration,
    Trainer,
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    CheckpointRemovalStrategy,
    ConfigSaverCallback,
    WandBCallback,
)
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)
from olmo_core.train.train_module.transformer.config import (
    TransformerActivationCheckpointingConfig,
)
from olmo_core.utils import seed_all

log = logging.getLogger(__name__)

DEFAULT_SEQUENCE_LENGTH = 4096
GLOBAL_BATCH_SIZE = 512 * DEFAULT_SEQUENCE_LENGTH  # 2.1M tokens
DEFAULT_RANK_MICROBATCH_SIZE = 2 * DEFAULT_SEQUENCE_LENGTH
DEFAULT_LR = 4e-4
DEFAULT_SEED = 1337

# Path on the worker image to the held-out manifest (baked in via add_local_dir).
HELDOUT_MANIFEST_PATH = "/root/math-pretraining/heldout_manifest.json"

# Default mix paths assume our pre-tokenized layout on the `math-pretraining-tokenized` volume.
DEFAULT_PATHS = {
    "math_3": ["/tokenized/3/part_*.npy"],
    "math_4plus": ["/tokenized/4plus/part_*.npy"],
    "math_4plus_MIND": ["/tokenized/4plus_MIND/part_*.npy"],
    "dolma3": ["/tokenized/dolma3/**/*.npy"],
}

# Default mix weights: 70% math (30/30/40 sub-weights) + 30% dolma3
DEFAULT_WEIGHTS = {
    "math_3": 0.3 * 0.7,         # 21% of total
    "math_4plus": 0.3 * 0.7,     # 21% of total
    "math_4plus_MIND": 0.4 * 0.7, # 28% of total
    "dolma3": 1.0 * 0.3,         # 30% of total
}

# Mode `anneal` and held-out eval read from the EXTRACTED UNTRAINED CORPUS — the
# instance-level subset that stable training provably never touched
# (see extract_untrained.py for how this is produced).
#
# math_4plus is excluded: stable trained on 100% of it, so there's no untrained
# slice. We renormalize math sub-weights (math_3=21, MIND=28) and keep dolma3=30,
# rescaling so the three sum to 1.0. Net result: anneal/eval mixture still ≈ 70/30
# math/dolma3 in spirit but without the math_4plus contribution.
UNTRAINED_PATHS = {
    "math_3": ["/untrained/math_3/part_*.npy"],
    "math_4plus_MIND": ["/untrained/math_4plus_MIND/part_*.npy"],
    "dolma3": ["/untrained/dolma3/part_*.npy"],
}

_UNTRAINED_RAW = {"math_3": 0.21, "math_4plus_MIND": 0.28, "dolma3": 0.30}
_UNTRAINED_DENOM = sum(_UNTRAINED_RAW.values())  # 0.79
UNTRAINED_WEIGHTS = {k: v / _UNTRAINED_DENOM for k, v in _UNTRAINED_RAW.items()}
# Concretely: math_3 ≈ 0.266, math_4plus_MIND ≈ 0.354, dolma3 ≈ 0.380.

# Eval set is carved out of the untrained corpus via a SEEDED SPLIT — the eval
# fraction is held out from anneal training. Anneal sees first (1 - EVAL_RATIO);
# eval sees the last EVAL_RATIO. Same seed → reproducible disjoint partition.
ANNEAL_EVAL_SPLIT_SEED = 20260626
DEFAULT_ANNEAL_EVAL_RATIO = 0.15


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="Run name.")
    parser.add_argument("--save-folder", required=True)
    parser.add_argument("--work-dir", required=True, help="Cache dir for instance source builds.")
    parser.add_argument(
        "--mode",
        choices=["stable", "anneal"],
        default="stable",
    )
    parser.add_argument("--tokens", type=int, required=True, help="Token budget for this run.")
    parser.add_argument("--warmup-tokens", type=int, default=2_000_000_000)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--rank-microbatch-size-tokens", type=int, default=DEFAULT_RANK_MICROBATCH_SIZE
    )
    parser.add_argument("--sequence-length", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    parser.add_argument("--shard-degree", type=int, default=None)
    parser.add_argument(
        "--compile-model", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-interval", type=int, default=2500)
    parser.add_argument("--ephemeral-save-interval", type=int, default=250)
    parser.add_argument(
        "--save-async", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--load-path", default=None, help="Checkpoint for anneal mode.")
    parser.add_argument(
        "--load-optim", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--benchmark-steps",
        type=int,
        default=None,
        help="Hard-stop after this many optimizer steps (smoke test).",
    )
    parser.add_argument(
        "--anneal-eval-ratio",
        type=float,
        default=DEFAULT_ANNEAL_EVAL_RATIO,
        help=(
            "Fraction of the untrained corpus reserved for held-out PPL eval. "
            "Anneal trains on the remaining (1 - ratio). Same seed across runs "
            "so eval slice is reproducible."
        ),
    )
    return parser


def _optional_interval(value: int) -> int | None:
    return value if value > 0 else None


def build_mix(
    *,
    sequence_length: int,
    total_tokens: int,
    tokenizer: TokenizerConfig,
    paths: dict[str, list[str]] = DEFAULT_PATHS,
    weights: dict[str, float] = DEFAULT_WEIGHTS,
) -> MixingInstanceSource.Config:
    """Build the stable-phase mix: 70% math (30/30/40 sub-weights) + 30% dolma3."""
    token_sources = NumpyDocumentSource.Config.from_source_groups(
        {label: paths[label] for label in paths},
        tokenizer=tokenizer,
    )

    def chunk(label: str) -> ConcatAndChunkInstanceSource.Config:
        return ConcatAndChunkInstanceSource.Config(
            sources=[token_sources[label]],
            label=label,
            sequence_length=sequence_length,
        )

    math_total_w = weights["math_3"] + weights["math_4plus"] + weights["math_4plus_MIND"]
    math_sub = MixingInstanceSource.Config(
        source_specs=[
            MixingInstanceSource.Spec.Config(
                source=chunk("math_3"),
                ratio=weights["math_3"] / math_total_w,
                label="math_3",
            ),
            MixingInstanceSource.Spec.Config(
                source=chunk("math_4plus"),
                ratio=weights["math_4plus"] / math_total_w,
                label="math_4plus",
            ),
            MixingInstanceSource.Spec.Config(
                source=chunk("math_4plus_MIND"),
                ratio=weights["math_4plus_MIND"] / math_total_w,
                label="math_4plus_MIND",
            ),
        ],
    )

    return MixingInstanceSource.Config(
        source_specs=[
            MixingInstanceSource.Spec.Config(
                source=math_sub,
                ratio=math_total_w,
                label="math",
            ),
            MixingInstanceSource.Spec.Config(
                source=chunk("dolma3"),
                ratio=weights["dolma3"],
                label="dolma3",
            ),
        ],
        num_tokens=total_tokens,
    )


def _untrained_chunk(label: str, sequence_length: int, tokenizer: TokenizerConfig):
    """One ConcatAndChunkInstanceSource.Config per untrained source label."""
    token_sources = NumpyDocumentSource.Config.from_source_groups(
        {label: UNTRAINED_PATHS[label]},
        tokenizer=tokenizer,
    )
    return ConcatAndChunkInstanceSource.Config(
        sources=[token_sources[label]],
        label=label,
        sequence_length=sequence_length,
    )


def _split_pair(label: str, sequence_length: int, tokenizer: TokenizerConfig, eval_ratio: float):
    """Return (anneal_cfg, eval_cfg) for one source. Anneal = first 1-eval_ratio,
    eval = last eval_ratio. Same seed across calls so the partition is reproducible.
    """
    chunk = _untrained_chunk(label, sequence_length, tokenizer)
    return chunk.split(ratio=1.0 - eval_ratio, seed=ANNEAL_EVAL_SPLIT_SEED)


def build_anneal_mix(
    *,
    sequence_length: int,
    total_tokens: int,
    tokenizer: TokenizerConfig,
    eval_ratio: float = DEFAULT_ANNEAL_EVAL_RATIO,
) -> MixingInstanceSource.Config:
    """Anneal training: 100% dolma3 from the untrained corpus.

    Uses the first (1 - eval_ratio) slice of /untrained/dolma3 — never seen by
    stable, and disjoint from the eval slice. (math_3 and MIND aren't touched by
    anneal at all, leaving their full untrained corpus available for eval and
    future experiments.)
    """
    assert 0.0 < eval_ratio < 1.0
    anneal_cfg, _ = _split_pair("dolma3", sequence_length, tokenizer, eval_ratio)
    return MixingInstanceSource.Config(
        source_specs=[
            MixingInstanceSource.Spec.Config(
                source=anneal_cfg,
                ratio=1.0,
                label="dolma3",
            ),
        ],
        num_tokens=total_tokens,
    )


def build_eval_mix_mixture(
    *,
    sequence_length: int,
    total_tokens: int,
    tokenizer: TokenizerConfig,
    eval_ratio: float = DEFAULT_ANNEAL_EVAL_RATIO,
    weights: dict[str, float] = UNTRAINED_WEIGHTS,
) -> MixingInstanceSource.Config:
    """Mixture eval: renormalized 70/30 across (math_3, MIND, dolma3).

    Every source drawn from its last eval_ratio slice. Reflects the stable-phase
    training distribution (minus math_4plus, which was 100% consumed). Use this
    PPL to track in-distribution quality across the *whole* mix.
    """
    assert 0.0 < eval_ratio < 1.0
    eval_slices = {
        label: _split_pair(label, sequence_length, tokenizer, eval_ratio)[1]
        for label in UNTRAINED_PATHS
    }
    return MixingInstanceSource.Config(
        source_specs=[
            MixingInstanceSource.Spec.Config(
                source=eval_slices[label],
                ratio=weights[label],
                label=label,
            )
            for label in UNTRAINED_PATHS
        ],
        num_tokens=total_tokens,
    )


def build_eval_mix_dolma3(
    *,
    sequence_length: int,
    total_tokens: int,
    tokenizer: TokenizerConfig,
    eval_ratio: float = DEFAULT_ANNEAL_EVAL_RATIO,
) -> MixingInstanceSource.Config:
    """Dolma3-only eval: 100% from /untrained/dolma3 last eval_ratio slice.

    Disjoint from anneal training (which takes the first 1-eval_ratio of the
    same dolma3 source) and from stable. Use this PPL to track anneal-distribution
    quality directly — should drop sharply during anneal.
    """
    assert 0.0 < eval_ratio < 1.0
    _, eval_cfg = _split_pair("dolma3", sequence_length, tokenizer, eval_ratio)
    return MixingInstanceSource.Config(
        source_specs=[
            MixingInstanceSource.Spec.Config(
                source=eval_cfg,
                ratio=1.0,
                label="dolma3",
            ),
        ],
        num_tokens=total_tokens,
    )


def main() -> None:
    opts = get_parser().parse_args()
    sequence_length = opts.sequence_length
    rank_microbatch_size = opts.rank_microbatch_size_tokens

    if rank_microbatch_size % sequence_length != 0:
        raise ValueError(
            f"rank microbatch ({rank_microbatch_size}) must be divisible by seq len "
            f"({sequence_length})"
        )
    if GLOBAL_BATCH_SIZE % rank_microbatch_size != 0:
        raise ValueError(
            f"global batch ({GLOBAL_BATCH_SIZE}) must be divisible by rank microbatch "
            f"({rank_microbatch_size})"
        )
    if opts.mode == "anneal" and not opts.load_path:
        raise ValueError("--load-path is required for --mode anneal")

    # Distributed env
    backend = "cpu:gloo,cuda:nccl" if torch.cuda.is_available() else None
    prepare_training_environment(shared_filesystem=False, backend=backend)
    seed_all(opts.seed)

    # Tokenizer (dolma2 — same as our tokenization output)
    tokenizer = TokenizerConfig.dolma2()

    # Build the composable mix
    log.info(
        "[mix] building composable instance source for %sB tokens (mode=%s)",
        opts.tokens // 1_000_000_000,
        opts.mode,
    )
    if opts.mode == "anneal":
        log.info(
            "[mix] anneal mode: 100%% dolma3 from /untrained/dolma3 "
            "first %.0f%% slice (eval_ratio=%.2f, seed=%d)",
            (1.0 - opts.anneal_eval_ratio) * 100,
            opts.anneal_eval_ratio,
            ANNEAL_EVAL_SPLIT_SEED,
        )
        mix_config = build_anneal_mix(
            sequence_length=sequence_length,
            total_tokens=opts.tokens,
            tokenizer=tokenizer,
            eval_ratio=opts.anneal_eval_ratio,
        )
    else:
        mix_config = build_mix(
            sequence_length=sequence_length,
            total_tokens=opts.tokens,
            tokenizer=tokenizer,
        )

    # Build the model
    model_config = TransformerConfig.olmo2_1B_v2(
        vocab_size=tokenizer.padded_vocab_size(),
        attn_backend=AttentionBackendName.flash_2,
    )

    # Scheduler: stable = warmup + constant; anneal = linear-to-zero, no warmup.
    # IMPORTANT: OLMo-core's Scheduler.get_lr receives `current` and `t_max` in
    # OPTIMIZER STEPS, not tokens. We must convert warmup_tokens -> warmup_steps
    # via the global batch size.
    if opts.mode == "stable":
        warmup_steps = max(1, opts.warmup_tokens // GLOBAL_BATCH_SIZE)
        print(
            f"[scheduler] warmup_tokens={opts.warmup_tokens:,} "
            f"-> warmup_steps={warmup_steps:,} (GBS={GLOBAL_BATCH_SIZE:,})"
        )
        scheduler = ConstantWithWarmup(
            warmup=warmup_steps,
            warmup_min_lr=0.0,
        )
    else:
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

    # Data loader (composable)
    data_loader_config = ComposableDataLoaderConfig(
        tokenizer=tokenizer,
        global_batch_size=GLOBAL_BATCH_SIZE,
        seed=opts.seed,
        work_dir=opts.work_dir,
        num_workers=opts.num_workers,
    )

    trainer_config = (
        TrainerConfig(
            save_folder=opts.save_folder,
            save_overwrite=True,
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
        .with_callback(
            "wandb",
            WandBCallback(
                name=opts.name,
                project=os.environ.get("WANDB_PROJECT", "math-pretraining"),
                entity=os.environ.get("WANDB_ENTITY"),
                # Enable iff WANDB_API_KEY is in env AND we're not in benchmark mode.
                enabled=bool(os.environ.get("WANDB_API_KEY"))
                and opts.benchmark_steps is None,
                cancel_check_interval=50,
            ),
        )
    )

    # Build everything: model -> train_module -> mix -> data loader -> trainer
    model = model_config.build(init_device="meta")
    train_module = train_module_config.build(model)

    log.info("[mix] building instance source (this resolves globs and builds caches)")
    built_mix = mix_config.build(opts.work_dir)
    built_mix.visualize()

    data_loader = data_loader_config.build(
        built_mix,
        dp_process_group=train_module.dp_process_group,
        work_dir=opts.work_dir,
        tokenizer=tokenizer,
    )

    trainer: Trainer = trainer_config.build(train_module, data_loader)

    # Save the config to the config_saver callback
    for cb in trainer.callbacks.values():
        if isinstance(cb, ConfigSaverCallback):
            cb.config = {
                "model": model_config.as_config_dict(),
                "train_module": train_module_config.as_config_dict(),
                "trainer": trainer_config.as_config_dict(),
                "data_loader": data_loader_config.as_config_dict(),
                "mix_weights": (
                    UNTRAINED_WEIGHTS if opts.mode == "anneal" else DEFAULT_WEIGHTS
                ),
                "mix_paths": (
                    UNTRAINED_PATHS if opts.mode == "anneal" else DEFAULT_PATHS
                ),
                "mode": opts.mode,
                "tokens": opts.tokens,
                "warmup_tokens": opts.warmup_tokens,
                "lr": opts.lr,
                "seed": opts.seed,
                "anneal_eval_ratio": (
                    opts.anneal_eval_ratio if opts.mode == "anneal" else None
                ),
                "anneal_eval_split_seed": (
                    ANNEAL_EVAL_SPLIT_SEED if opts.mode == "anneal" else None
                ),
            }
            break

    # If a load path was provided (anneal mode), load weights only.
    if (
        not trainer.no_checkpoints
        and not trainer.maybe_load_checkpoint()
        and opts.load_path
    ):
        log.info("Loading checkpoint from %s (weights only)", opts.load_path)
        trainer.load_checkpoint(opts.load_path, load_trainer_state=False)

    trainer.fit()
    teardown_training_environment()


if __name__ == "__main__":
    main()
