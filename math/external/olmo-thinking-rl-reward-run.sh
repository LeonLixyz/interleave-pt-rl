#!/bin/bash
# VERL GRPO RL on OLMo-3 7B annealed checkpoints (pretraining->RL scaling study)
# Reward: pure correctness (math_verify). Train: Skywork-OR1 45000 | Val: held-out 750 iid
#
# Usage:
#   DEBUG=True  bash pretraining-rl/rl-run/anneal-checkpoint/olmo-thinking-format-reward-run.sh   # 1 GPU smoke test
#   DEBUG=False bash pretraining-rl/rl-run/anneal-checkpoint/olmo-thinking-format-reward-run.sh   # 4 GPUs, full run

# =============================================================================
#  EASY CONFIGURATION (MODIFY THESE FOR DIFFERENT EXPERIMENTS)
# =============================================================================
export DEBUG=${DEBUG:-False}  # Can be overridden: DEBUG=True bash script/rlvr_8k.sh
export REWARD_MODEL_TYPE=RULE_BASED  # pure correctness (math_verify), no format gate — train==val reward, avoids format-strength confound
export BASE_MODEL=/local2/salman/pretrain-rl/sft/anneal/7b_olmo_25b_open_thought_43k_sft # OLMo-3 7B annealed @25B-token SFT checkpoint

# Set save directory based on debug mode
if [ "$DEBUG" = "True" ]; then
export SAVE_DIR="/local2/salman/debug_save" # Debug save directory
else
export SAVE_DIR="/local2/salman/pretrain-rl/rl-run-merge-anneal" # Production save directory
fi

export EXPERIMENT_NAME=25b # pretraining token size of this checkpoint (sweep variable)

# Absolute repo root: paths must be cwd-independent because we cd to a neutral dir
# before launching (so `python -m verl...` resolves to the INSTALLED verl 0.9 in the
# olmo3-rl env, not the local ./verl 0.4.1 source tree).
REPO="/home/salman/reward-signal-analysis"

TOTAL_EPOCHS=6 # compute axis (recipe A11): passes over the fixed prompt set; FREEZE across all 13
SAVE_FREQ=100
TEST_FREQ=100   # dense val for the fittable curve (recipe B2: ~every 100 steps)

# Validation generation config
VAL_N_SAMPLES=8          # 8 samples/problem -> pass@1/4/8 + mean@8 for the fittable curve (recipe B2)
VAL_TEMPERATURE=1        # Temperature for validation (0=greedy/deterministic, >0=sampling)
VAL_DO_SAMPLE=True       # False=greedy decoding (deterministic), True=sampling (with temperature)

# =============================================================================
#  TRAINING CONFIGURATION
# =============================================================================

if [ "$DEBUG" = "True" ]; then
unset VLLM_ATTENTION_BACKEND
export CUDA_VISIBLE_DEVICES="0"
N_GPUS_PER_NODE=1
else
export VLLM_ATTENTION_BACKEND=FLASH_ATTN   # OLMo3 interleaved sliding-window needs flash-attn backend (XFORMERS mishandles per-layer sliding window)
# Force set GPUs (don't use fallback syntax to avoid empty string issues)
export CUDA_VISIBLE_DEVICES="4,5,6,7" # CHANGE THIS if using different GPUs
N_GPUS_PER_NODE=4
fi

if [ "$DEBUG" = "True" ]; then
TRAIN_DATA_PATH="$REPO/data/math/train_think/llama-3b_think/llama_sky_math_8_upsample.parquet"
EVAL_DATA_PATH_1="$REPO/data/math/eval_data_think/scp_test_medium_2_8.parquet"
EVAL_DATA_PATH_2="$REPO/data/math/eval_data_think/aime2024.parquet"
else
TRAIN_DATA_PATH="$REPO/pretraining-rl/rl-data/train-val-data/train_45000.parquet"
EVAL_DATA_PATH_1="$REPO/pretraining-rl/rl-data/train-val-data/val_750.parquet"   # held-out iid fitting set (recipe B1)
# Benchmarks (coarse credibility, D6): same thinking system prompt as train/val, prompts <=1036 tok (fit in 1536)
EVAL_DATA_PATH_2="$REPO/data/math/eval_data_think/aime2024.parquet"
EVAL_DATA_PATH_3="$REPO/data/math/eval_data_think/aime2025.parquet"
EVAL_DATA_PATH_4="$REPO/data/math/eval_data_think/math500.parquet"
EVAL_DATA_PATH_5="$REPO/data/math/eval_data_think/amc_test.parquet"
EVAL_DATA_PATH_6="$REPO/data/math/eval_data_think/scp_test_difficult_1.parquet"
EVAL_DATA_PATH_7="$REPO/data/math/eval_data_think/scp_test_very_difficult_0.parquet"
EVAL_DATA_PATH_8="$REPO/data/math/eval_data_think/scp_test_medium_2_8.parquet"
fi

CUSTOM_REWARD_PATH="$REPO/reward_function.py"
CHECKPOINT_DIR="$SAVE_DIR/$EXPERIMENT_NAME/checkpoints"
LOG_FILE="$SAVE_DIR/$EXPERIMENT_NAME/logs/log_$EXPERIMENT_NAME.log"
MLFLOW_DIR="$SAVE_DIR/$EXPERIMENT_NAME/mlflow"
ROLLOUT_DIR="$SAVE_DIR/$EXPERIMENT_NAME/rollouts/training"
VALIDATION_DIR="$SAVE_DIR/$EXPERIMENT_NAME/rollouts/validation"
export MLFLOW_TRACKING_URI=file://$MLFLOW_DIR
export MLFLOW_ALLOW_FILE_STORE=true   # mlflow>=3.x: keep file-store backend (else it errors out and disables tracking)

if [ "$DEBUG" = "True" ]; then
TRAIN_BATCH_SIZE=4
PPO_MINI_BATCH=4
MAX_PROMPT_LENGTH=1024
RES_LENGTH=1024
GROUP_SIZE=4
else
TRAIN_BATCH_SIZE=64    # A9: 64 prompts x G=8 = 512 rollouts/step. FREEZE across all 13. Probe up on the WEAKEST/rambliest ckpt if it fits.
PPO_MINI_BATCH=64      # == train_batch -> one on-policy update per step
MAX_PROMPT_LENGTH=1536   # prompts are short (sys+math: mean 316, p99.9 969, max 2273 tok) -> 1536 drops ~9/45000 train, ~0/750 val
RES_LENGTH=6656   # prompt + response must be <= 8192 (OLMo3 8K trained window; this ckpt rope_scaling=None, NOT YaRN-extended). 1536+6656=8192
GROUP_SIZE=8
fi

ACTOR_LR=1e-6
LR_WARMUP_RATIO=0.01
ENTROPY_COEFF=0.0

# KL term: False = no reference model (recipe default). Set True to add KL back (beta=0.001).
USE_KL=False
KL_LOSS_COEF=0.001
KL_LOSS_TYPE=low_var_kl

if [ "$USE_KL" = "True" ]; then
  KL_ARGS="actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF actor_rollout_ref.actor.kl_loss_type=$KL_LOSS_TYPE"
else
  KL_ARGS="actor_rollout_ref.actor.use_kl_loss=False"
fi

# Asymmetric clipping + prompt-average aggregation (ScaleRL DAPO baseline): stability guard in place of KL.
USE_DAPO_CLIP=True
CLIP_RATIO_LOW=0.2
CLIP_RATIO_HIGH=0.26
CLIP_RATIO_C=10.0
LOSS_AGG_MODE=seq-mean-token-mean

if [ "$USE_DAPO_CLIP" = "True" ]; then
  CLIP_ARGS="actor_rollout_ref.actor.clip_ratio_low=$CLIP_RATIO_LOW actor_rollout_ref.actor.clip_ratio_high=$CLIP_RATIO_HIGH actor_rollout_ref.actor.clip_ratio_c=$CLIP_RATIO_C actor_rollout_ref.actor.loss_agg_mode=$LOSS_AGG_MODE"
else
  CLIP_ARGS=""
fi

# Rollout sampling temperature (fixed across all checkpoints).
ROLLOUT_TEMPERATURE=1.0

if [ "$DEBUG" = "True" ]; then
PPO_MICRO_BATCH_SIZE_PER_GPU=1
LOG_PROB_MICRO_BATCH_SIZE=1
TENSOR_MODEL_PARALLEL_SIZE=1
GPU_MEMORY_UTILIZATION=0.6
MAX_NUM_BATCHED_TOKENS=12288
else
PPO_MICRO_BATCH_SIZE_PER_GPU=2
LOG_PROB_MICRO_BATCH_SIZE=2
TENSOR_MODEL_PARALLEL_SIZE=1
GPU_MEMORY_UTILIZATION=0.7   # more vLLM KV cache -> faster generation; watch for actor OOM with FP32 logits (drop back to 0.6 if it OOMs)
MAX_NUM_BATCHED_TOKENS=12288
fi


PARAM_OFFLOAD=False
OPTIMIZER_OFFLOAD=False
REF_PARAM_OFFLOAD=True

USE_REMOVE_PADDING=True
ENABLE_GRADIENT_CHECKPOINTING=True

# =============================================================================
#  SETUP
# =============================================================================
train_files="$TRAIN_DATA_PATH"
if [ "$DEBUG" = "True" ]; then
# test_files="['$EVAL_DATA_PATH_1']"
test_files="['$EVAL_DATA_PATH_1','$EVAL_DATA_PATH_2']"
else
test_files="['$EVAL_DATA_PATH_1','$EVAL_DATA_PATH_2','$EVAL_DATA_PATH_3','$EVAL_DATA_PATH_4','$EVAL_DATA_PATH_5','$EVAL_DATA_PATH_6','$EVAL_DATA_PATH_7','$EVAL_DATA_PATH_8']"
fi

mkdir -p $CHECKPOINT_DIR
mkdir -p "$(dirname $LOG_FILE)"
mkdir -p $MLFLOW_DIR
mkdir -p $ROLLOUT_DIR
mkdir -p $VALIDATION_DIR

# cd to a neutral dir (no local ./verl) so `python -m verl...` uses the installed verl 0.9.
cd "$SAVE_DIR/$EXPERIMENT_NAME"

python3 -m verl.trainer.main_ppo \
algorithm.adv_estimator=grpo \
data.train_files=$train_files \
data.val_files=$test_files \
data.train_batch_size=$TRAIN_BATCH_SIZE \
data.max_prompt_length=$MAX_PROMPT_LENGTH \
data.max_response_length=$RES_LENGTH \
data.filter_overlong_prompts=True \
data.truncation='error' \
actor_rollout_ref.model.path=$BASE_MODEL \
actor_rollout_ref.actor.optim.lr=$ACTOR_LR \
actor_rollout_ref.model.use_remove_padding=$USE_REMOVE_PADDING \
actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH \
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU \
$KL_ARGS \
$CLIP_ARGS \
actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEFF \
actor_rollout_ref.model.enable_gradient_checkpointing=$ENABLE_GRADIENT_CHECKPOINTING \
actor_rollout_ref.actor.fsdp_config.param_offload=$PARAM_OFFLOAD \
actor_rollout_ref.actor.fsdp_config.optimizer_offload=$OPTIMIZER_OFFLOAD \
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE \
actor_rollout_ref.rollout.tensor_model_parallel_size=$TENSOR_MODEL_PARALLEL_SIZE \
actor_rollout_ref.rollout.name=vllm \
actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION \
actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS \
actor_rollout_ref.rollout.temperature=$ROLLOUT_TEMPERATURE \
actor_rollout_ref.rollout.n=$GROUP_SIZE \
actor_rollout_ref.rollout.val_kwargs.n=$VAL_N_SAMPLES \
actor_rollout_ref.rollout.val_kwargs.temperature=$VAL_TEMPERATURE \
actor_rollout_ref.rollout.val_kwargs.do_sample=$VAL_DO_SAMPLE \
actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE \
actor_rollout_ref.ref.fsdp_config.param_offload=$REF_PARAM_OFFLOAD \
algorithm.use_kl_in_reward=False \
reward_model.reward_manager=batch \
custom_reward_function.path=$CUSTOM_REWARD_PATH \
custom_reward_function.name=compute_score_batch \
trainer.critic_warmup=0 \
trainer.default_hdfs_dir=null \
trainer.default_local_dir=$CHECKPOINT_DIR \
trainer.resume_mode=auto \
trainer.logger='["console","mlflow"]' \
trainer.project_name=$EXPERIMENT_NAME \
trainer.experiment_name=$EXPERIMENT_NAME \
trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
trainer.nnodes=1 \
trainer.save_freq=$SAVE_FREQ \
trainer.test_freq=$TEST_FREQ \
trainer.total_epochs=$TOTAL_EPOCHS \
trainer.rollout_data_dir=$ROLLOUT_DIR \
trainer.validation_data_dir=$VALIDATION_DIR 2>&1 | tee $LOG_FILE









