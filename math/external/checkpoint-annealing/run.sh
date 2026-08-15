#!/usr/bin/env bash
# Anneal ONE OLMo-3 7B checkpoint end-to-end:
#   download (gs -> local) -> anneal (LR -> 0 over the budget) -> convert to HF.
#
# Usage:
#   OUT_DIR=<dir> bash run.sh <STEP> [NUM_GPUS] [--smoke]
#
# Examples:
#   OUT_DIR=/data/anneal bash run.sh 6000              # 25B base point, all GPUs on the node
#   OUT_DIR=/data/anneal bash run.sh 6000 2 --smoke    # quick 2-GPU test (a few steps, then stop)
#
# STEP -> base token count (run OLMo3-7B-swafix; the step number is the HF token label):
#   25B=6000  50B=12000  100B=24000  250B=60000  400B=95000  600B=143000  1T=238000  2.5T=596000
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

STEP="${1:?usage: OUT_DIR=<dir> bash run.sh STEP [NUM_GPUS] [--smoke]}"
NUM_GPUS="${2:-$(nvidia-smi -L | wc -l)}"
SMOKE=""
[[ "${3:-}" == "--smoke" ]] && SMOKE="--smoke-steps 5"

# OUT_DIR holds all artifacts. For multi-node it must be on storage every node can reach.
OUT_DIR="${OUT_DIR:?set OUT_DIR to an output directory (shared fast storage on multi-node)}"
ANNEAL_TOKENS="${ANNEAL_TOKENS:-10e9}"
GS_RUN="gs://ai2-llm/checkpoints/OLMo3-7B-swafix"

NATIVE="${OUT_DIR}/native/swafix-step${STEP}"
ANNEALED="${OUT_DIR}/annealed/swafix-step${STEP}"
HF="${OUT_DIR}/hf/swafix-step${STEP}"
WORK="${OUT_DIR}/dataset-cache"

echo "[1/3] download native checkpoint step${STEP}"
python "${HERE}/download_checkpoint.py" --gs "${GS_RUN}/step${STEP}" --out "${NATIVE}"

echo "[2/3] anneal LR->0 over ${ANNEAL_TOKENS} tokens on ${NUM_GPUS} GPU(s)"
torchrun --nproc_per_node="${NUM_GPUS}" "${HERE}/anneal.py" "${NATIVE}" \
  --save-folder "${ANNEALED}" \
  --anneal-tokens "${ANNEAL_TOKENS}" \
  --work-dir "${WORK}" ${SMOKE}

echo "[3/3] convert annealed checkpoint -> HF"
FINAL="$(ls -d "${ANNEALED}"/step* | sort -V | tail -1)"
echo "      final annealed checkpoint: ${FINAL}"
python "${HERE}/convert_to_hf.py" -i "${FINAL}" -o "${HF}"

echo "DONE. HF checkpoint ready for SFT/RL at: ${HF}"
