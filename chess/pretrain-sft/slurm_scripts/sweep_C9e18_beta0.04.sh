#!/bin/bash
# ============================================================================
# Pretrain sweep: C_total = 9e18, beta = 0.04
# ============================================================================
# Submits all 14 pretraining jobs (200M, 300M, 500M) x alpha allocations.
#
# Usage:
#   bash sweep_C9e18_beta0.04.sh              # submit all jobs
#   bash sweep_C9e18_beta0.04.sh --dry-run    # print what would be submitted
#   bash sweep_C9e18_beta0.04.sh --model 500m # only submit 500M jobs
# ============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_DIR="$REPO_ROOT/config/configs/C9e18_beta0.04"

# ---- GPU config per model size ----
# 500M needs more GPUs; 200M/300M can run on fewer
declare -A GPU_MAP
GPU_MAP[200m]=4
GPU_MAP[300m]=4
GPU_MAP[500m]=8

NUM_GPUS_DEFAULT=4
GPU_CONSTRAINT="h100"
SLURM_ACCOUNT="torch_pr_114_tandon_advanced"
TIME_LIMIT="48:00:00"
MEM="300G"

# ---- Parse CLI args ----
DRY_RUN=false
MODEL_FILTER=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)  DRY_RUN=true; shift ;;
    --model)    MODEL_FILTER="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

mkdir -p "$SCRIPT_DIR/C9e18_slurm_logs"

TOTAL_JOBS=0
SKIPPED_JOBS=0

for config_path in "$CONFIG_DIR"/*.yaml; do
  cfg_base="$(basename "${config_path%.yaml}")"  # e.g. 200m_alpha0.048
  model_size="${cfg_base%%_*}"                     # e.g. 200m

  # Filter by model if requested
  if [[ -n "$MODEL_FILTER" && "$model_size" != "$MODEL_FILTER" ]]; then
    SKIPPED_JOBS=$((SKIPPED_JOBS + 1))
    continue
  fi

  NUM_GPUS=${GPU_MAP[$model_size]:-$NUM_GPUS_DEFAULT}
  CPUS=$((NUM_GPUS * 4))
  if [[ $NUM_GPUS -ge 8 ]]; then
    MEM="600G"
  else
    MEM="300G"
  fi

  job_name="C9e18_${cfg_base}"

  echo "  $cfg_base: ${NUM_GPUS} GPUs"

  if $DRY_RUN; then
    TOTAL_JOBS=$((TOTAL_JOBS + 1))
    continue
  fi

  sbatch <<EOF
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --gres=gpu:${NUM_GPUS}
#SBATCH --constraint=${GPU_CONSTRAINT}
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --mem=${MEM}
#SBATCH --job-name=${job_name}
#SBATCH --output=${SCRIPT_DIR}/C9e18_slurm_logs/${job_name}_%j.log

echo "=== C9e18 pretrain: ${cfg_base} ==="
echo "Job \$SLURM_JOB_ID on \${SLURM_NODELIST:-local}"
echo "Config: ${config_path}"
echo "GPUs: ${NUM_GPUS}"

source ~/.bashrc
export PYTHONWARNINGS="ignore"

cd "$REPO_ROOT"

uv run accelerate launch --num_processes ${NUM_GPUS} scripts/train/train_hf.py \\
  --config ${config_path} --auto_resume
EOF

  TOTAL_JOBS=$((TOTAL_JOBS + 1))
  echo "    -> submitted"
done

echo ""
echo "=========================================="
echo "Summary: ${TOTAL_JOBS} jobs submitted, ${SKIPPED_JOBS} skipped"
if $DRY_RUN; then
  echo "(dry-run mode -- no jobs were actually submitted)"
fi
echo "=========================================="
