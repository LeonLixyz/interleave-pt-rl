# Isoflop Scaling Law Sweep

## Overview

This experiment finds the **compute-optimal** allocation between model size (N) and training tokens (D) for chess move prediction, following the Chinchilla/isoflop methodology.

**Core idea:** Fix a compute budget C, then sweep model sizes. For each model size N, train on D = C / (6N) tokens. The model size that achieves the lowest loss at each budget is compute-optimal for that budget. Plotting across budgets reveals the scaling law.

```
C = 6 * N * D    (forward + backward FLOPs)
```

## Architecture

All models use **Qwen3** (initialized from scratch, not pretrained):
- SwiGLU MLP + RoPE + GQA (Grouped Query Attention)
- Template: `Qwen/Qwen3-0.6B` with custom dimension overrides
- Tokenizer: `LanTokenizer` (chess-specific, PGN to LAN notation, vocab ~81 tokens)
- Context length: 1024

### Model Size Grid

| Config | Layers | d_model | Heads | KV Heads | Intermediate | ~Params |
|--------|--------|---------|-------|----------|-------------|---------|
| `qwen3_5m.yaml` | 6 | 256 | 4 | 2 | 896 | 5M |
| `qwen3_10m.yaml` | 8 | 320 | 8 | 2 | 1120 | 10M |
| `qwen3_25m.yaml` | 12 | 448 | 8 | 2 | 1568 | 25M |
| `qwen3_50m.yaml` | 16 | 576 | 8 | 2 | 2048 | 50M |
| `qwen3_100m.yaml` | 20 | 720 | 12 | 4 | 2560 | 100M |
| `qwen3_200m.yaml` | 24 | 896 | 16 | 4 | 3136 | 200M |
| `qwen3_300m.yaml` | 24 | 1024 | 16 | 4 | 3584 | 300M |

Config files: `config/configs/pretrain_sl/qwen3_*.yaml`

## FLOP Budgets and Sweep Grid

5 FLOP budgets x 7 model sizes = **32 valid jobs** (3 skipped due to data caps or too few shards).

| Budget C | 5M | 10M | 25M | 50M | 100M | 200M | 300M |
|----------|-----|------|------|------|-------|-------|-------|
| 3e15 | 88 shards / 752 steps | 44 / 367 | 18 / 140 | 9 / 61 | 5 / 26 | skip | skip |
| 1e16 | 291 / 2529 | 146 / 1260 | 59 / 498 | 30 / 245 | 15 / 113 | 8 / 52 | 5 / 26 |
| 3e16 | 872 / 7613 | 436 / 3797 | 175 / 1513 | 88 / 752 | 44 / 367 | 22 / 175 | 15 / 113 |
| 1e17 | 2907 / 25421 | 1454 / 12706 | 582 / 5075 | 291 / 2529 | 146 / 1260 | 73 / 621 | 49 / 411 |
| 3e17 | skip (data-capped) | 4360 / 38136 | 1744 / 15244 | 872 / 7613 | 436 / 3797 | 218 / 1890 | 146 / 1260 |

Each cell shows: shards / gradient-accumulation steps.

## Data

- **Source:** `Evangelinejy/chess-train-data-balanced-tokenized` on HuggingFace
- **Total:** 5860 `.npy` shards, ~6.7B tokens
- **Tokens per shard:** ~1,147,000
- **Eval holdout:** 2 shards
- **Default path:** `/scratch/js15262/LLM-Pretraining/data/chess/train/tokenized`

## Training Details

- **Optimizer:** AdamW (lr=1e-3, weight_decay=0.1, betas=[0.9, 0.95])
- **Scheduler:** Cosine decay to eta_min=1e-5, with adaptive warmup
- **Batch size:** 16 per GPU, gradient accumulation 8, context 1024
- **Effective batch:** 131,072 tokens/step (per GPU)
- **Max grad norm:** 1.0
- **Epochs:** 1 (single pass)
- **Precision:** Default (fp32). Set `training.mixed_precision: bf16` for faster runs.

## Logging

- **WandB project:** `chess-isoflop-scaling`
- Each run is tagged with model name and FLOP budget in WandB notes

## SLURM Configuration

- **GPUs:** 2x H100 per job
- **Account:** `torch_pr_114_tandon_advanced`
- **Time limit:** 48 hours
- **Memory:** 300G
- **Logs:** `slurm_scripts/isoflop_slurm_logs/`

## How to Launch

```bash
# Preview all jobs without submitting
bash slurm_scripts/isoflop_sweep.sh --dry-run

# Submit all 32 jobs
bash slurm_scripts/isoflop_sweep.sh

# Submit jobs for a single FLOP budget (e.g., 7 jobs)
bash slurm_scripts/isoflop_sweep.sh --budget 1e16
```

### Local debug test (no SLURM)

```bash
cd scripts/train
WANDB_MODE=disabled accelerate launch --num_processes 1 --gpu_ids 0 train_hf.py \
  --config ../../config/configs/pretrain_sl/qwen3_5m.yaml \
  --override \
    data.txt_path=/path/to/tokenized \
    data.num_shards=10 \
    training.save_dir=/tmp/isoflop_test \
    logging.mode=disabled
```

## Key Files

| File | Purpose |
|------|---------|
| `slurm_scripts/isoflop_sweep.sh` | Main sweep launcher (SLURM sbatch) |
| `scripts/train/train_hf.py` | Training entry point (supports `--override`) |
| `training/trainer_hf.py` | HFTrainer class (Accelerate-based) |
| `config/configs/pretrain_sl/qwen3_*.yaml` | Per-model-size configs |
| `llm_tokens/chess/lan_tokenizer.py` | Chess tokenizer |
| `training/data_utils.py` | ShardedPackedTextDataset |

## Expected Output

After the sweep, for each FLOP budget plot final eval loss vs model size. The minimum of each curve gives the compute-optimal model size at that budget. Fitting a power law N*(C) ~ C^a gives the scaling exponent.
