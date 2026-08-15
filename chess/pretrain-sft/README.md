# Chess Reasoning LLM Pretraining

Distributed LLM pretraining framework for chess reasoning, supporting scaling law experiments across multiple model sizes (5M-300M parameters) with Qwen3 architectures.

## Installation

**Recommended (uv):**
```bash
uv sync
```

**Conda (alternative):**
```bash
conda env create -f environment.yml
# Edit prefix in environment.yml first
```

## Dataset

Non-bullet chess games spanning Elo 800-3000, sampled uniformly from February and March 2022.

- **Training data**: 10.5B tokens across 9,052 shards (~1.15M tokens/shard)
  - [chess-pre-to-post/pretraining_dataset_v1_tokenized]([https://huggingface.co/datasets/Evangelinejy/chess-train-data-balanced-tokenized](https://huggingface.co/chess-pre-to-post/pretraining_dataset_v1_tokenized))
- **Test data**: [Evangelinejy/chess-test-data](https://huggingface.co/datasets/Evangelinejy/chess-test-data)

## Training Configs

Configs live in `config/configs/pretrain_sl/` for scaling law experiments:

| Config | Params | Layers | Hidden | Heads |
|--------|--------|--------|--------|-------|
| `qwen3_5m.yaml` | ~5M | 6 | 256 | 8 |
| `qwen3_10m.yaml` | ~10M | 8 | 320 | 8 |
| `qwen3_25m.yaml` | ~25M | 12 | 448 | 8 |
| `qwen3_50m.yaml` | ~50M | 16 | 512 | 8 |
| `qwen3_100m.yaml` | ~100M | 20 | 640 | 10 |
| `qwen3_200m.yaml` | ~200M | 24 | 768 | 12 |
| `qwen3_300m.yaml` | ~300M | 24 | 1024 | 16 |

Edit `data.txt_path` in your config to point to your local data directory.

## Quick Start

### Single-GPU training
```bash
uv run python scripts/train/train_hf.py --config config/configs/pretrain_sl/qwen3_10m.yaml
```

### Multi-GPU training (e.g. 4 GPUs)
```bash
uv run accelerate launch --num_processes 4 scripts/train/train_hf.py \
  --config config/configs/pretrain_sl/qwen3_10m.yaml
```

### 8-GPU training
```bash
uv run accelerate launch --num_processes 8 scripts/train/train_hf.py \
  --config config/configs/pretrain_sl/qwen3_100m.yaml \
  --override training.gradient_accumulation_steps=4
```

### CLI flags
```
--config CONFIG         Path to YAML config (required)
--override KEY=VAL ...  Config overrides in dot-list format
--auto_resume           Auto-resume from latest checkpoint in save_dir
--data_dir DIR          Override data.txt_path
--output_dir DIR        Override training.save_dir
--max_checkpoints N     Keep only N most recent checkpoints (0 = unlimited)
```

## Auto-Resume

Training can automatically resume from the latest checkpoint:

```bash
# Via CLI flag
uv run accelerate launch --num_processes 4 scripts/train/train_hf.py \
  --config config/configs/pretrain_sl/qwen3_10m.yaml --auto_resume

# Via config
# Add to your YAML:
#   training:
#     auto_resume: true
```

When `--auto_resume` is set, the trainer scans `save_dir` for the latest `step*` checkpoint directory and resumes from it (optimizer state, scheduler, training step all restored).

Use `--max_checkpoints N` to automatically clean up old checkpoints and save disk space.

## SLURM

```bash
# Scaling law sweep (submits jobs for all model sizes x FLOP budgets)
cd slurm_scripts
bash isoflop_sweep.sh

# Dry run (see what would be submitted)
bash isoflop_sweep.sh --dry-run

# Single budget
bash isoflop_sweep.sh --budget 1e17

# 8-GPU training
bash train_8gpu.sh config/configs/pretrain_sl/qwen3_100m.yaml 2000
```

## Modal (Cloud Training)

Train on Modal GPUs with auto-resume and fault tolerance:

```bash
# 1. Download training data to Modal volume (first time only)
modal run modal_train.py::download_data --max-shards 100

# 2. Download test data
modal run modal_train.py::download_test_data

# 3. Launch training (default config)
modal run modal_train.py

# 4. Launch with specific config / shard count
modal run modal_train.py --config config/configs/pretrain_sl/qwen3_100m.yaml --num-shards 2000
```

Configure GPU type, count, and node count at the top of `modal_train.py`:
```python
N_NODES = 1
GPUS_PER_NODE = 8
GPU_TYPE = "H100"
```

## Project Structure

```
config/                  # OmegaConf YAML configs
  configs/pretrain_sl/   # Scaling law experiment configs
  configs/qwen_sft/      # SFT configs
training/                # Core training code
  trainer.py             # Custom GPT2 trainer
  trainer_hf.py          # HuggingFace model trainer (Qwen, etc)
  sft_trainer.py         # Supervised fine-tuning trainer
  data_utils.py          # Sharded dataset loading
  optim_sched.py         # Optimizer & scheduler builders
scripts/train/           # Training entry points
llm_tokens/chess/        # Chess LAN tokenizer
evaluation/              # Chess-specific evaluation (legal moves, puzzles)
slurm_scripts/           # SLURM job submission scripts
modal_train.py           # Modal cloud training launcher
```
