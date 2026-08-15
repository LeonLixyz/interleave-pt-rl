#!/bin/bash

set -euo pipefail

BASE="/scratch/js15262"

configs=(
  "../../config/configs/qwen_multiturn_sft/sft_config_multi_path_2048.yaml"
)
lrs=("3e-4")

# New pretrain checkpoint root (no pretrain_checkpoints/ subdir; model dirs use
# C_{total_compute}_alpha{alpha}_beta{beta} format without modelsize).
PRETRAIN_ROOT="${BASE}/LLM-Pretraining/checkpoints/C_3p3e19"
MAP_CONFIG_FILE="${BASE}/LLM-Pretraining/slurm_scripts/sft_max_train_files_map.json"

# Each spec entry: total_compute|modelsize|alpha|beta
pretrain_specs=(
  # "6p5e18|50m|0.400|0.045"
  # "6p5e18|50m|0.750|0.045"
  # "6p5e18|100m|0.750|0.045"
  # "6p5e18|100m|0.050|0.045"
  # "6p5e18|100m|0.200|0.045"
  # "6p5e18|100m|0.400|0.045"
  # "6p5e18|100m|0.950|0.045"
  # "6p5e18|200m|0.050|0.045"
  # "6p5e18|200m|0.200|0.045"
  # "6p5e18|200m|0.400|0.045"
  # "6p5e18|200m|0.750|0.045"
  # "6p5e18|200m|0.950|0.045"
  # "6p5e18|410m|0.050|0.045"
  # "6p5e18|410m|0.200|0.045"
  # "6p5e18|410m|0.400|0.045"
  # "6p5e18|410m|0.750|0.045"
  # "6p5e18|410m|0.950|0.045"
  "3p3e19|200m|0.400|0.009"
  "3p3e19|200m|0.750|0.009"

  # "6p5e18|400m|0.049|0.03"
  # "6p5e18|400m|0.279|0.03"
  # "6p5e18|400m|0.509|0.03"
  # "6p5e18|400m|0.740|0.03"
  # "6p5e18|400m|0.970|0.03"
  # "6p5e18|1000m|0.049|0.03"
  # "6p5e18|1000m|0.279|0.03"
  # "6p5e18|1000m|0.509|0.03"
  # "6p5e18|1000m|0.970|0.03"
  # "6p5e18|1000m|0.740|0.03"
)

# Default train data dir/name used when a model has no explicit entry below.
DEFAULT_TRAIN_DATA_DIR="${BASE}/LLM-Pretraining/data/sft/cot_data_puzzle_seq_qwen/qwen06b_final_step26528/hf_model_seed0_evalminimax_candpolicy_nc8_nt10_nta8_d20_dmrandom_wt300_pm8_inj"
DEFAULT_DATA_NAME="200m_generated"

# Optional per-model metadata keyed by canonical model id.
declare -A TRAIN_DATA_DIR_BY_MODEL_ID=(
  ["C9e18_200m_alpha0.740_beta0.03"]="${BASE}/LLM-Pretraining/data/sft/cot_data_puzzle_seq_qwen_stockfish/stockfish_seed0_evalminimax_candpolicy_nc8_nt10_nta20_d20_dmfixed_wt300_pm8_inj"
)
declare -A DATA_NAME_BY_MODEL_ID=(
  ["C9e18_200m_alpha0.740_beta0.03"]="stockfish_generated" #"200m_generated"
)
# Keep this for rare non-standard layouts; default is "final".
declare -A PRETRAIN_SUBDIR_OVERRIDE=(
  # ["C9e18_200m_alpha0.740_beta0.03"]="final"
)

cot_fields=(
  "cot_by_method.trajectory_sep.cot_format_no_labels"
  # "cot_by_method.trajectory_sep.cot_format"
  # "cot_by_method.monotonic_improvement.cot_format"
)
cot_types=(
  "trajectory_sep_no_labels"
  # "trajectory_sep"
  # "monotonic_improvement"
)
block_sizes=(
  "4096"
  # "4096"
  # "4096"
)

wandb_entity="jingyanshen-new-york-university"

if [[ ${#cot_fields[@]} -ne ${#cot_types[@]} || ${#cot_fields[@]} -ne ${#block_sizes[@]} ]]; then
  echo "[ERROR] cot_fields/cot_types/block_sizes length mismatch:"
  echo "        cot_fields=${#cot_fields[@]}, cot_types=${#cot_types[@]}, block_sizes=${#block_sizes[@]}"
  exit 1
fi

# SFT run-name id (unchanged): includes modelsize for W&B / checkpoint naming.
model_id_from_spec() {
  local total_compute="$1"
  local modelsize="$2"
  local alpha="$3"
  local beta="$4"
  printf "C%s_%s_alpha%s_beta%s" "$total_compute" "$modelsize" "$alpha" "$beta"
}

# Pretrain checkpoint dir name: {total_compute}_{modelsize}_alpha{alpha}.
pretrain_dir_from_spec() {
  local total_compute="$1"
  local modelsize="$2"
  local alpha="$3"
  printf "%s_%s_alpha%s" "$total_compute" "$modelsize" "$alpha"
}

max_train_files_for_spec() {
  local total_compute="$1"
  local modelsize="$2"
  local alpha="$3"
  local beta="$4"
  python - "$MAP_CONFIG_FILE" "$total_compute" "$modelsize" "$alpha" "$beta" <<'PY'
import json
import pathlib
import sys

cfg_path = pathlib.Path(sys.argv[1])
total_compute = str(sys.argv[2])
modelsize = str(sys.argv[3])
alpha = str(sys.argv[4])
beta = str(sys.argv[5])

if not cfg_path.exists():
    raise FileNotFoundError(f"Mapping file not found: {cfg_path}")

cfg = json.loads(cfg_path.read_text())
default_value = cfg.get("default_max_train_files")
entries = cfg.get("entries_by_spec", [])

matched_value = None
for item in entries:
    if not isinstance(item, dict):
        continue
    if (
        str(item.get("total_compute")) == total_compute
        and str(item.get("modelsize")) == modelsize
        and str(item.get("alpha")) == alpha
        and str(item.get("beta")) == beta
    ):
        matched_value = item.get("max_train_files")
        break

if matched_value is not None:
    print(f"{int(matched_value)}\tentry")
elif default_value is not None:
    print(f"{int(default_value)}\tdefault")
else:
    raise ValueError(
        f"No max_train_files mapping for spec "
        f"(C={total_compute}, modelsize={modelsize}, alpha={alpha}, beta={beta}), "
        f"and default_max_train_files is unset in {cfg_path}"
    )
PY
}

mkdir -p slurm_logs

for config in "${configs[@]}"; do
  for lr in "${lrs[@]}"; do
    for spec in "${pretrain_specs[@]}"; do
      IFS="|" read -r total_compute modelsize alpha beta <<<"${spec}"
      # SFT run-name id (includes modelsize, unchanged).
      model_id="$(model_id_from_spec "${total_compute}" "${modelsize}" "${alpha}" "${beta}")"

      # Pretrain checkpoint path (new layout).
      pretrain_ckpt_name="$(pretrain_dir_from_spec "${total_compute}" "${modelsize}" "${alpha}")"
      pretrain_subdir="${PRETRAIN_SUBDIR_OVERRIDE[$model_id]:-final}"
      pretrain_dir="${PRETRAIN_ROOT}/${pretrain_ckpt_name}/${pretrain_subdir}"

      pretrained_model=""
      if [[ -f "${pretrain_dir}/config.json" ]]; then
        pretrained_model="${pretrain_dir}"
      elif [[ -f "${pretrain_dir}/hf_model/config.json" ]]; then
        pretrained_model="${pretrain_dir}/hf_model"
      else
        echo "[WARN] HF model files not found at: ${pretrain_dir} — trainer will resolve from spec"
        # Do NOT continue: the trainer's _resolve_pretrain_hf_model fallback will handle this.
        pretrained_model="${pretrain_dir}"  # pass the expected path; trainer resolves if missing
      fi

      pretrained_weights=""
      for w in \
        "${pretrain_dir}/model.safetensors" \
        "${pretrain_dir}/pytorch_model.bin" \
        "${pretrained_model}/model.safetensors" \
        "${pretrained_model}/pytorch_model.bin"; do
        if [[ -f "${w}" ]]; then
          pretrained_weights="${w}"
          break
        fi
      done
      if [[ -z "${pretrained_weights}" ]]; then
        echo "[WARN] No standalone weights file found for ${model_id}; will rely on --pretrained-model only"
      fi

      if ! max_lookup="$(max_train_files_for_spec "${total_compute}" "${modelsize}" "${alpha}" "${beta}")"; then
        echo "[ERROR] Unable to resolve max_train_files for ${model_id}"
        continue
      fi
      IFS=$'\t' read -r max_train_files max_source <<<"${max_lookup}"
      if [[ "${max_source}" != "entry" ]]; then
        echo "[WARN] No exact spec mapping for ${model_id}; using default_max_train_files=${max_train_files}"
      fi

      train_data_dir="${TRAIN_DATA_DIR_BY_MODEL_ID[$model_id]:-${DEFAULT_TRAIN_DATA_DIR}}"
      data_name="${DATA_NAME_BY_MODEL_ID[$model_id]:-${DEFAULT_DATA_NAME}}"

      for j in "${!cot_fields[@]}"; do
        cot_field="${cot_fields[$j]}"
        cot_type="${cot_types[$j]}"
        block_size="${block_sizes[$j]}"
        cot_short="${cot_field##*.}"

        extra_args=""
        extra_args+=" --pretrained-model ${pretrained_model}"
        [[ -n "${pretrained_weights}" ]] && extra_args+=" --pretrained-weights ${pretrained_weights}"
        extra_args+=" --naming-scheme pretrain_spec"
        extra_args+=" --total-compute ${total_compute}"
        extra_args+=" --modelsize ${modelsize}"
        extra_args+=" --alpha ${alpha}"
        extra_args+=" --beta ${beta}"
        extra_args+=" --cot-field ${cot_field}"
        extra_args+=" --cot-type ${cot_type}"
        [[ -n "${block_size}" ]] && extra_args+=" --block-size ${block_size}"
        [[ -n "${train_data_dir}" ]] && extra_args+=" --train-files ${train_data_dir}"
        [[ -n "${data_name}" ]] && extra_args+=" --data-name ${data_name}"
        [[ -n "${wandb_entity}" ]] && extra_args+=" --wandb-entity ${wandb_entity}"
        extra_args+=" --max-train-files ${max_train_files}"

        job_tag="${model_id}"
        sbatch <<EOF
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --account=torch_pr_114_tandon_advanced
#SBATCH --time=6:00:00
#SBATCH --mem=300G
#SBATCH --constraint=h100|h200
#SBATCH --job-name=sft_${job_tag}_${cot_short}
#SBATCH --output=slurm_logs/sft_${job_tag}_${cot_short}_%j.log

echo "Starting job \$SLURM_JOB_NAME (\$SLURM_JOB_ID) on \${SLURM_NODELIST:-local}"
echo "[SFT] model_id=${model_id} pretrain_dir=${pretrain_dir} max_train_files=${max_train_files}"

source ~/.bashrc
conda activate pretraining || true
cd ${BASE}/LLM-Pretraining/scripts/train
export PYTHONWARNINGS="ignore"
export NCCL_TIMEOUT=1800000
export TORCH_NCCL_BLOCKING_WAIT=1

accelerate launch --num_processes 2 run_sft.py --config ${config} --lr ${lr}${extra_args}
EOF
      done
    done
  done
done
