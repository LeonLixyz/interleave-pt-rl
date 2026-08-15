# scripts/train/train_hf.py
import sys, pathlib, argparse
repo_root = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from omegaconf import OmegaConf
from accelerate.utils import set_seed
from config import load_config
from training.trainer_hf import HFTrainer

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--override", nargs="*", default=[],
                   help="Config overrides in dot-list format, e.g. data.num_shards=100")
    p.add_argument("--auto_resume", action="store_true",
                   help="Auto-resume from latest checkpoint in save_dir")
    p.add_argument("--data_dir", type=str, default=None,
                   help="Override data.txt_path")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Override training.save_dir")
    p.add_argument("--max_checkpoints", type=int, default=None,
                   help="Max checkpoints to keep (0 = unlimited)")
    p.add_argument("--test_data_dir", type=str, default=None,
                   help="Override data.test_data_dir (base path for eval test files)")
    return p.parse_args()

def main():
    from omegaconf import OmegaConf
    args = parse_args()
    cfg = load_config(args.config)
    if args.override:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.override))
    if args.auto_resume:
        cfg.training.auto_resume = True
    if args.data_dir:
        cfg.data.txt_path = args.data_dir
    if args.output_dir:
        cfg.training.save_dir = args.output_dir
    if args.max_checkpoints is not None:
        cfg.training.max_checkpoints = args.max_checkpoints
    if args.test_data_dir:
        cfg.data.test_data_dir = args.test_data_dir
    set_seed(int(cfg.training.get("seed", 42)))

    trainer = HFTrainer(cfg, run_config_path=args.config)
    trainer.train()

if __name__ == "__main__":
    main()
