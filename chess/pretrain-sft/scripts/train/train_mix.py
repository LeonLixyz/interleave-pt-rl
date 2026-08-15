"""CLI entrypoint for mixed pretraining + SFT training.

Loads a config file and launches :class:`training.mixed_trainer.MixedTrainer`.
"""
import argparse
import pathlib
import sys

repo_root = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from accelerate.utils import set_seed
from config import load_config
from training.mixed_trainer import MixedTrainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML/JSON")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.training.get("seed", 42)))

    trainer = MixedTrainer(cfg, run_config_path=args.config)
    trainer.train()


if __name__ == "__main__":
    main()
