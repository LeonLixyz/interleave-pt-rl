#!/bin/bash

tokenizers=("LanTokenizer")  # "BpeTokenizer" "LibPGNTokenizer" "SanTokenizer"
configs=(
  # "../../config/configs/chess_gpt2_lr3e-4.yaml"
  # "../../config/configs/chess_gpt2_lr3e-5.yaml"
  # "../../config/configs/chess_gpt2_lr1e-3.yaml"
  # "../../config/configs/chess_gpt2_lr3e-4_10B.yaml"
  # "../../config/configs/chess_gpt2_lr1e-3_10B.yaml"
  # "../../config/configs/eval/chess_gpt2_lr1e-3_8B_pretrained.yaml"
  # "../../config/configs/chess_gpt2_lr1e-3_8B.yaml"
  # "../../config/configs/eval/chess_gpt2_lr1e-3_8B_pretrained_qwen.yaml"
  # "../../config/configs/eval/chess_gpt2_lr1e-3_8B_pretrained_qwen_1024.yaml"
  # "../../config/configs/eval/chess_gpt2_lr1e-3_8B_pretrained_qwen_512.yaml"
  # "../../config/configs/chess_gpt2_lr1e-3_8B_from_ckp_qwen.yaml"
  # "../../config/configs/chess_gpt2_lr1e-3_8B_from_ckp.yaml"
)
## SBATCH --account=torch_pr_114_tandom_advanced
mkdir -p slurm_logs

for tok in "${tokenizers[@]}"; do
  for config in "${configs[@]}"; do
    cfg_base="$(basename "${config%.yaml}")"
    sbatch <<EOF
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:l40s:1
#SBATCH --account=torch_pr_114_general
#SBATCH --time=48:00:00
#SBATCH --mem=100G
#SBATCH --job-name=eval_${tok}_${cfg_base}
#SBATCH --output=eval_slurm_logs/eval_${tok}_${cfg_base}_%j.log


echo "Starting job \$SLURM_JOB_NAME (\$SLURM_JOB_ID) on \${SLURM_NODELIST:-local}"

source ~/.bashrc
conda activate pretraining || true
cd /scratch/js15262/LLM-Pretraining/scripts/train
export PYTHONWARNINGS="ignore"

accelerate launch --num_processes 1 train_hf.py --config ${config} --eval-only --eval-pass-at-k
EOF
  done
done
