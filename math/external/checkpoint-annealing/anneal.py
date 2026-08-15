"""Real (LR-decay) annealing of an OLMo-3 7B native checkpoint, run locally.

A no-Beaker port of OLMo's own ``OLMo3-7B-anneal.py``. Given a native OLMo-core
checkpoint (model + optimizer + data-loader state), it:

  1. reads the checkpoint's current learning rate from the optimizer state,
  2. decays that LR linearly to 0 with a WSD scheduler over a fixed token budget,
  3. continues training on the ORIGINAL pretraining mix (OLMo-mix-0625, streamed
     from gs://ai2-llm) from where the data loader left off,
  4. writes the annealed checkpoint locally.

This is exactly the procedure OLMo used to produce clean-loss intermediate
checkpoints for the 7B (the tech report's "anneal LR to zero before evaluation").
The model/data/optimizer builders are reused verbatim from ``OLMo3-7B.py``; only the
LR schedule changes.

Launch with torchrun (single node, N GPUs):

    torchrun --nproc_per_node=8 anneal.py CHECKPOINT --save-folder OUT [--anneal-tokens 10e9]

``--smoke-steps N`` hard-stops after N optimizer steps for a quick "does it run" test.
"""

import argparse
import importlib.util
import json
import logging
import os
from copy import deepcopy
from math import ceil
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from olmo_core.config import DType
from olmo_core.data import TokenizerConfig
from olmo_core.distributed.checkpoint import load_state_dict
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.internal.experiment import (
    CommonComponents,
    ExperimentConfig,
    SubCmd,
    _build_required_callbacks,
)
from olmo_core.io import join_path, resource_path
from olmo_core.nn.transformer import TransformerActivationCheckpointingMode, TransformerConfig
from olmo_core.optim import SchedulerUnits
from olmo_core.optim.scheduler import WSD
from olmo_core.train import Duration, LoadStrategy, TrainerConfig
from olmo_core.train.callbacks import CheckpointerCallback
from olmo_core.train.train_module import (
    TransformerActivationCheckpointingConfig,
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
)

log = logging.getLogger(__name__)

# OLMo-core checkout that ships alongside this folder (../OLMo-core).
# Override with --olmo-core-root or $OLMO_CORE_ROOT.
DEFAULT_OLMO_CORE_ROOT = Path(
    os.environ.get("OLMO_CORE_ROOT", Path(__file__).resolve().parents[1] / "OLMo-core")
)


def load_olmo3_7b_module(olmo_core_root: Path):
    """Import OLMo's official ``OLMo3-7B.py`` so we reuse its exact builders."""
    script = olmo_core_root / "src/scripts/train/OLMo3/OLMo3-7B.py"
    if not script.exists():
        raise FileNotFoundError(f"OLMo3-7B.py not found at {script}; set --olmo-core-root")
    spec = importlib.util.spec_from_file_location("OLMo3_7B", script)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_model_config_from_ckpt(model_dict: Dict[str, Any]) -> TransformerConfig:
    """Rebuild the exact architecture from the checkpoint's serialized config.

    Newer OLMo-core ``main`` has drifted from the version that trained the released
    OLMo-3 7B (its ``olmo3_7B()`` factory now defaults to MHA + a new sliding-window
    schema), so building from the factory yields the WRONG architecture. We instead
    deserialize the checkpoint's own model config, translating only the one field whose
    schema changed (sliding window). NOTE: pinning OLMo-core to the OLMo-3-era commit
    makes this shim unnecessary and is recommended for the final sweep.
    """
    m = deepcopy(model_dict)
    att = m.get("block", {}).get("attention", {})
    sw = att.get("sliding_window")
    if isinstance(sw, dict) and "window_size" in sw:
        win = sw["window_size"]
        old_pattern = sw.get("pattern", [False, False, False, True])
        # old: pattern=[bool] (True == full attention) + window_size
        # new: pattern=[int]  (window size, or -1 for full attention)
        att["sliding_window"] = {
            "pattern": [-1 if bool(is_full) else win for is_full in old_pattern],
            "force_full_attention_on_first_layer": bool(sw.get("force_first", False)),
            "force_full_attention_on_last_layer": bool(sw.get("force_last", True)),
            "_CLASS_": sw.get("_CLASS_", "olmo_core.nn.attention.SlidingWindowAttentionConfig"),
        }
    return TransformerConfig.from_dict(m)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "checkpoint",
        help="Native OLMo-core checkpoint dir (local or gs://); must contain "
        "model_and_optim/, train/, config.json",
    )
    p.add_argument("--save-folder", required=True, help="Local dir to write the annealed checkpoint.")
    p.add_argument(
        "--anneal-tokens",
        type=lambda s: int(float(s)),
        default=int(10e9),
        help="Token budget over which the LR decays linearly to 0 (default 10e9).",
    )
    p.add_argument(
        "--root-dir",
        default="gs://ai2-llm",
        help="Base dir for the pretraining data mix (default gs://ai2-llm).",
    )
    p.add_argument(
        "--work-dir",
        required=True,
        help="Local cache dir for the dataset index (needs a few GB; reusable across runs).",
    )
    p.add_argument(
        "--olmo-core-root",
        type=Path,
        default=DEFAULT_OLMO_CORE_ROOT,
        help=f"Path to the OLMo-core checkout (default {DEFAULT_OLMO_CORE_ROOT}).",
    )
    p.add_argument(
        "--smoke-steps",
        type=int,
        default=None,
        help="If set, hard-stop after this many optimizer steps (quick test only).",
    )
    p.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch.compile (uses full activation checkpointing; for debugging / "
        "envs where compile fails). Production matches OLMo with compile ON.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    olmo = load_olmo3_7b_module(args.olmo_core_root)
    batch_size = olmo.GLOBAL_BATCH_SIZE  # tokens per optimizer step (4,194,304)

    # --- read original run state + starting LR from the checkpoint ---
    trainer_state = torch.load(
        resource_path(join_path(args.checkpoint, "train"), "rank0.pt"), weights_only=False
    )
    with open(resource_path(args.checkpoint, "config.json"), "rb") as f:
        ckpt_config = json.load(f)

    global_step = trainer_state["global_step"]
    tokens_processed = trainer_state["data_loader"].get("tokens_processed")
    sequence_length = trainer_state["data_loader"]["sequence_length"]
    length_in_steps = ceil(args.anneal_tokens / batch_size)

    lr_key = "optim.param_groups.embeddings.weight.lr"
    lr_state: Dict[str, Optional[float]] = {lr_key: None}
    load_state_dict(join_path(args.checkpoint, "model_and_optim"), lr_state)
    assert lr_state[lr_key] is not None, "could not read starting LR from optimizer state"
    lr = float(lr_state[lr_key])  # type: ignore[arg-type]

    run_name = f"{ckpt_config.get('run_name', 'OLMo3-7B')}-anneal-from{global_step}"
    base_tokens_b = (tokens_processed or global_step * batch_size) / 1e9
    log.info(
        "Annealing %s: base step %d (~%.1fB tokens consumed), LR %.3e -> 0 over %.1fB tokens "
        "(%d steps). Effective tokens ~%.1fB.",
        run_name, global_step, base_tokens_b, lr, args.anneal_tokens / 1e9,
        length_in_steps, base_tokens_b + args.anneal_tokens / 1e9,
    )

    common = CommonComponents(
        run_name=run_name,
        root_dir=args.root_dir,
        work_dir=args.work_dir,
        save_folder=args.save_folder,
        launch=None,  # no Beaker
        tokenizer=TokenizerConfig.dolma2(),
        max_sequence_length=sequence_length,
        global_batch_size=batch_size,
    )

    # Model rebuilt from the checkpoint's own config (version-faithful GQA);
    # data + optimizer from OLMo's official builders.
    model_config = build_model_config_from_ckpt(ckpt_config["model"])
    data = olmo.build_data_components(common)

    tm = olmo.build_train_module_config(common)
    tm.optim.lr = lr
    tm.scheduler = WSD(
        units=SchedulerUnits.steps,
        warmup=global_step,  # already past warmup -> no re-warmup
        warmup_fraction=None,
        decay=length_in_steps,  # linear decay LR -> 0 over the budget
        decay_fraction=None,
    )
    # Single-node-friendly full-shard FSDP (HSDP needs >1 node). Global batch is fixed,
    # so this does not change the optimization vs OLMo's multi-node run.
    tm.dp_config = TransformerDataParallelConfig(
        name=DataParallelType.fsdp,
        param_dtype=DType.bfloat16,
        reduce_dtype=DType.float32,
        wrapping_strategy=TransformerDataParallelWrappingStrategy.blocks,
    )
    if args.no_compile:
        tm.compile_model = False
        tm.ac_config = TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.full  # 'budget' mode requires compile
        )
    else:
        tm.ac_config = TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.budget, activation_memory_budget=0.85
        )

    trainer = TrainerConfig(
        save_folder=args.save_folder,
        save_overwrite=True,
        metrics_collect_interval=10,
        cancel_check_interval=10,
        max_duration=Duration.steps(global_step + length_in_steps),
    ).with_callback(
        "checkpointer",
        # A final checkpoint is always written at end-of-training (Checkpointer.post_train),
        # so save_interval only controls optional intermediate saves.
        CheckpointerCallback(save_interval=10000, ephemeral_save_interval=None, save_async=False),
    )
    trainer.load_path = args.checkpoint
    trainer.load_strategy = LoadStrategy.always  # resume model + optimizer + data-loader state
    if args.smoke_steps is not None:
        trainer.hard_stop = Duration.steps(global_step + args.smoke_steps)

    for name, cb in _build_required_callbacks(common).items():
        if name not in trainer.callbacks:
            trainer.add_callback(name, cb)

    config = ExperimentConfig(
        run_name=run_name,
        launch=None,
        model=model_config,
        dataset=data.dataset,
        data_loader=data.data_loader,
        train_module=tm,
        trainer=trainer,
    )

    cmd = SubCmd.train
    cmd.prepare_environment(config)
    cmd.run(config)


if __name__ == "__main__":
    main()
