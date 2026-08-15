#!/bin/bash

tokenizers=("LanTokenizer")  # "BpeTokenizer" "LibPGNTokenizer" "SanTokenizer"
configs=(
  # "../../config/configs/chess_gpt2_lr3e-4.yaml"
  # "../../config/configs/chess_gpt2_lr3e-5.yaml"
  # "../../config/configs/chess_gpt2_lr1e-3.yaml"
  # "../../config/configs/chess_gpt2_lr3e-4_10B.yaml"
  # "../../config/configs/chess_gpt2_lr1e-3_10B.yaml"
  # "../../config/configs/chess_gpt2_lr1e-3_8B_pretrained.yaml"
  # "../../config/configs/chess_gpt2_lr1e-3_8B.yaml"
  # "../../config/configs/chess_gpt2_lr1e-3_8B_pretrained_qwen.yaml"
  # "../../config/configs/chess_gpt2_lr1e-3_8B_from_ckp_qwen.yaml"
  # "../../config/configs/chess_gpt2_lr1e-3_8B_from_ckp.yaml"
  "/scratch/js15262/LLM-Pretraining/config/configs/pretrain_sl/chess_gpt2_lr1e-3_8B_pretrained_qwen_100m.yaml"
  "/scratch/js15262/LLM-Pretraining/config/configs/pretrain_sl/chess_gpt2_lr1e-3_8B_pretrained_qwen_200m.yaml"
  "/scratch/js15262/LLM-Pretraining/config/configs/pretrain_sl/chess_gpt2_lr1e-3_8B_pretrained_qwen_400m.yaml"
  "/scratch/js15262/LLM-Pretraining/config/configs/pretrain_sl/chess_gpt2_lr1e-3_8B_pretrained_qwen.yaml"
)
## SBATCH --account=torch_pr_114_tandom_advanced
mkdir -p pretrain_slurm_logs

for tok in "${tokenizers[@]}"; do
  for config in "${configs[@]}"; do
    cfg_base="$(basename "${config%.yaml}")"
    sbatch <<EOF
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --constraint=h200
#SBATCH --account=torch_pr_114_tandon_advanced
#SBATCH --time=48:00:00
#SBATCH --mem=300G
#SBATCH --job-name=train_${tok}_${cfg_base}
#SBATCH --output=pretrain_slurm_logs/train_${tok}_${cfg_base}_%j.log


echo "Starting job \$SLURM_JOB_NAME (\$SLURM_JOB_ID) on \${SLURM_NODELIST:-local}"

source ~/.bashrc
cd /scratch/js15262/LLM-Pretraining
export PYTHONWARNINGS="ignore"

accelerate launch --num_processes 2 train_hf.py --config ${config} 
EOF
  done
done
