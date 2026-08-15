#!/bin/bash
# Usage: bash train_8gpu.sh [CONFIG] [NUM_SHARDS]
# Example: bash train_8gpu.sh ../../config/configs/pretrain_sl/qwen3_100m.yaml 2000
#
# Launches training on 8 GPUs (1 node) with auto-resume enabled.

CONFIG=${1:-"../../config/configs/pretrain_sl/qwen3_100m.yaml"}
NUM_SHARDS=${2:-""}
cfg_base="$(basename "${CONFIG%.yaml}")"

mkdir -p slurm_logs

sbatch <<EOF
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8
#SBATCH --constraint=h100
#SBATCH --account=torch_pr_114_tandon_advanced
#SBATCH --time=48:00:00
#SBATCH --mem=600G
#SBATCH --job-name=train_8gpu_${cfg_base}
#SBATCH --output=slurm_logs/train_8gpu_${cfg_base}_%j.log

echo "Starting job \$SLURM_JOB_NAME (\$SLURM_JOB_ID) on \${SLURM_NODELIST:-local}"
echo "Config: ${CONFIG}"
echo "GPUs: 8"

source ~/.bashrc
cd /scratch/js15262/LLM-Pretraining
export PYTHONWARNINGS="ignore"

uv run accelerate launch --num_processes 8 scripts/train/train_hf.py \\
  --config ${CONFIG} --auto_resume \\
  ${NUM_SHARDS:+--override data.num_shards=${NUM_SHARDS} training.gradient_accumulation_steps=4}
EOF

echo "Submitted 8-GPU training job: ${cfg_base}"
