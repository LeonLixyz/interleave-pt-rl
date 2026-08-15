"""CLI for the strict 50M unified pretraining/SFT trainer."""
from __future__ import annotations

import argparse
import pathlib
import sys


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from accelerate.utils import set_seed
from omegaconf import OmegaConf

from config import load_config
from training.interleaved_hf_trainer import InterleavedHFTrainer


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the 47.245M Qwen on one shuffled PT+SFT manifest"
    )
    parser.add_argument("--config", required=True, help="YAML experiment config")
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="OmegaConf dot-list overrides, for example training.max_steps=1",
    )
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument(
        "--resume",
        help="Full in-leg resume directory containing trainer_state.json",
    )
    initialization.add_argument(
        "--weights-only",
        help="Clean HF directory or unsharded state dict for a fresh stage",
    )
    parser.add_argument(
        "--output-dir",
        help="Override training.output_dir",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Stop early without shortening the configured LR arc (canaries)",
    )
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace):
    cfg = load_config(args.config)
    if args.override:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.override))
    if args.resume:
        cfg.training.resume = args.resume
    if args.weights_only:
        cfg.training.weights_only = args.weights_only
    if args.output_dir:
        cfg.training.output_dir = args.output_dir
    if args.max_steps is not None:
        cfg.training.max_steps = args.max_steps
    return cfg


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = build_config(args)
    set_seed(int(cfg.training.get("seed", 42)))
    trainer = InterleavedHFTrainer(cfg, run_config_path=args.config)
    trainer.train()


if __name__ == "__main__":
    main()
