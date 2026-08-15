#!/bin/bash
# ============================================================================
# Isoflop Scaling Law Sweep
# ============================================================================
# For each FLOP budget C, trains multiple model sizes N, each for D = C/(6N)
# tokens, to find the compute-optimal (N, D) allocation.
#
# C ≈ 6 * N * D   (forward + backward FLOPs)
#
# Usage:
#   bash isoflop_sweep.sh                  # submit all jobs
#   bash isoflop_sweep.sh --dry-run        # print what would be submitted
#   bash isoflop_sweep.sh --budget 1e19    # run only one FLOP budget
# ============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_DIR="$REPO_ROOT/config/configs/pretrain_sl"
TRAIN_SCRIPT="$REPO_ROOT/scripts/train/train_hf.py"

# ---- Knobs ----
TOKENS_PER_SHARD=1147000
BATCH_SIZE=16
SEQ_LEN=1024
GRAD_ACCUM=8
TOKENS_PER_STEP=$((BATCH_SIZE * SEQ_LEN * GRAD_ACCUM))  # 131,072
MIN_SHARDS=5         # skip runs with fewer shards (need eval_holdout + training)
MAX_SHARDS=5861      # total available in dataset
EVAL_HOLDOUT=2
NUM_GPUS=2
GPU_CONSTRAINT="h100"
SLURM_ACCOUNT="torch_pr_114_tandon_advanced"
TIME_LIMIT="48:00:00"
MEM="300G"

# ---- Model definitions: name:approx_params:config_file ----
# Architecture: Qwen3 (SwiGLU + RoPE + GQA), initialized from scratch
MODELS=(
  "5m:5000000:qwen3_5m.yaml"
  "10m:10000000:qwen3_10m.yaml"
  "25m:25000000:qwen3_25m.yaml"
  "50m:50000000:qwen3_50m.yaml"
  "100m:100000000:qwen3_100m.yaml"
  "200m:200000000:qwen3_200m.yaml"
  "300m:300000000:qwen3_300m.yaml"
)

# ---- FLOP budgets (C = 6 * N * D) ----
# With 6.7B tokens available and models 5M–300M params:
#   - Model N is data-capped when C > 6*N*6.7e9
#   - E.g., 5M model caps at C ≈ 2e17; 300M at C ≈ 1.2e19
# These budgets give 5–7 non-capped models per curve:
FLOP_BUDGETS=("3e15" "1e16" "3e16" "1e17" "3e17")

# ---- Parse CLI args ----
DRY_RUN=false
BUDGET_FILTER=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=true; shift ;;
    --budget)  BUDGET_FILTER="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

mkdir -p "$SCRIPT_DIR/isoflop_slurm_logs"

TOTAL_JOBS=0
SKIPPED_JOBS=0

for budget in "${FLOP_BUDGETS[@]}"; do
  # Filter to single budget if requested
  if [[ -n "$BUDGET_FILTER" && "$budget" != "$BUDGET_FILTER" ]]; then
    continue
  fi

  echo "=========================================="
  echo "FLOP budget: C = $budget"
  echo "=========================================="

  for model_entry in "${MODELS[@]}"; do
    IFS=':' read -r model_name model_params config_file <<< "$model_entry"
    config_path="$CONFIG_DIR/$config_file"

    # Compute tokens D = C / (6 * N)
    D=$(python3 -c "print(int(float('$budget') / (6 * $model_params)))")

    # Convert to shards
    num_shards=$(python3 -c "import math; print(max(1, math.ceil($D / $TOKENS_PER_SHARD)))")

    # Skip if data-capped (can't achieve target FLOPs — not a valid isoflop point)
    if [[ $num_shards -gt $MAX_SHARDS ]]; then
      echo "  SKIP: $model_name @ C=$budget  (needs ${num_shards} shards > ${MAX_SHARDS} available, data-capped)"
      SKIPPED_JOBS=$((SKIPPED_JOBS + 1))
      continue
    fi

    # Skip if too few shards for meaningful training
    if [[ $num_shards -lt $MIN_SHARDS ]]; then
      echo "  SKIP: $model_name @ C=$budget  (${num_shards} shards < ${MIN_SHARDS} min)"
      SKIPPED_JOBS=$((SKIPPED_JOBS + 1))
      continue
    fi

    # Actual tokens (accounting for shard granularity and eval holdout)
    actual_tokens=$(python3 -c "print($num_shards * $TOKENS_PER_SHARD)")
    train_shards=$((num_shards - EVAL_HOLDOUT))
    total_steps=$(python3 -c "print($train_shards * $TOKENS_PER_SHARD // $TOKENS_PER_STEP)")

    # Adaptive intervals based on run length
    eval_interval=$(python3 -c "print(max(10, int($total_steps) // 10))")
    eval_rollouts_interval=$(python3 -c "print(max(50, int($total_steps) // 5))")
    save_interval=$(python3 -c "print(max(50, int($total_steps) // 5))")
    warmup_steps=$(python3 -c "print(min(200, max(10, int($total_steps) // 20)))")

    # Unique save directory per (model, budget) pair
    save_dir="$REPO_ROOT/checkpoints/isoflop/${model_name}_C${budget}"

    # Job name for SLURM
    job_name="iso_${model_name}_C${budget}"

    echo "  $model_name: N=${model_params}, D=${actual_tokens} tok (${num_shards} shards), ~${total_steps} steps"

    if $DRY_RUN; then
      TOTAL_JOBS=$((TOTAL_JOBS + 1))
      continue
    fi

    sbatch <<EOF
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:${NUM_GPUS}
#SBATCH --constraint=${GPU_CONSTRAINT}
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --mem=${MEM}
#SBATCH --job-name=${job_name}
#SBATCH --output=${SCRIPT_DIR}/isoflop_slurm_logs/${job_name}_%j.log

echo "=== Isoflop run: model=${model_name} C=${budget} D=${actual_tokens} ==="
echo "Job \$SLURM_JOB_ID on \${SLURM_NODELIST:-local}"
echo "Config: ${config_path}"
echo "Shards: ${num_shards}, Steps: ~${total_steps}"

source ~/.bashrc
export PYTHONWARNINGS="ignore"

cd "$REPO_ROOT"

uv run accelerate launch --num_processes ${NUM_GPUS} scripts/train/train_hf.py \\
  --config ${config_path} --auto_resume \\
  --override \\
    data.num_shards=${num_shards} \\
    training.save_dir=${save_dir} \\
    training.save_interval=${save_interval} \\
    training.eval_interval=${eval_interval} \\
    training.eval_rollouts_interval=${eval_rollouts_interval} \\
    training.scheduler.warmup_steps=${warmup_steps}
EOF

    TOTAL_JOBS=$((TOTAL_JOBS + 1))
    echo "    -> submitted"
  done
done

echo ""
echo "=========================================="
echo "Summary: ${TOTAL_JOBS} jobs submitted, ${SKIPPED_JOBS} skipped"
if $DRY_RUN; then
  echo "(dry-run mode — no jobs were actually submitted)"
fi
echo "=========================================="
