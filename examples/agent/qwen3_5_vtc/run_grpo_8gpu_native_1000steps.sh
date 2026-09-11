#!/usr/bin/env bash
set -euo pipefail

# Qwen3.5-9B random-grounding SFT checkpoint VTC GRPO run on one 8-GPU node.
#
# This is a thin wrapper around the maintained 8-GPU baseline.  It selects the
# random-grounding SFT checkpoint and a fixed, proportionally stratified 10K
# training subset; agent/tool/reward settings stay in the baseline script.
# Batch defaults are exported explicitly so they reach the baseline process.
# With 10K rows and batch=8, one epoch has 1250 full batches, enough for the
# default 100 steps or an override of 1000 steps. PPO and log-prob micro-batch
# sizes are independent; both default to 2 in this recipe.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${VTC_GRPO_DATA_ROOT:-/vepfs-mlp2/c20250405/400042/data/VTC/GRPO}"

export VTC_TRAIN_BATCH="${VTC_TRAIN_BATCH:-8}"
export VTC_ROLLOUT_N="${VTC_ROLLOUT_N:-4}"
export VTC_PPO_MINI="${VTC_PPO_MINI:-8}"
export VTC_PPO_MICRO="${VTC_PPO_MICRO:-2}"
export VTC_ROLLOUT_LOGPROB_MICRO="${VTC_ROLLOUT_LOGPROB_MICRO:-2}"

export VTC_GRPO_TRAIN="${VTC_GRPO_TRAIN:-${DATA_ROOT}/train_10k_uniform.parquet}"
export VTC_GRPO_VAL="${VTC_GRPO_VAL:-${DATA_ROOT}/val_500_uniform.parquet}"
export VTC_VAL_BATCH="${VTC_VAL_BATCH:-16}"
export VTC_VAL_NUM_WORKERS="${VTC_VAL_NUM_WORKERS:-0}"
export VTC_GRPO_MODEL="${VTC_GRPO_MODEL:-/vepfs-mlp2/c20250405/400042/VTC/train/SFT/output/qwen3_5_9b_vtc_250k_random_grounding_freeze_visual/checkpoint-7000_merged}"
export VTC_TOTAL_STEPS="${VTC_TOTAL_STEPS:-100}"
export VTC_TOTAL_EPOCHS="${VTC_TOTAL_EPOCHS:-1}"
export VTC_SAVE_FREQ="${VTC_SAVE_FREQ:-20}"
export VTC_TEST_FREQ="${VTC_TEST_FREQ:-20}"
export VTC_GRPO_OUTPUT="${VTC_GRPO_OUTPUT:-${DATA_ROOT}/checkpoints/qwen35_vtc_random_grounding_step7000_train10k_8gpu}"
# export VTC_GRPO_MM_PROCESSOR_CACHE_GB="${VTC_GRPO_MM_PROCESSOR_CACHE_GB:-8}"
export WANDB_PROJECT="${WANDB_PROJECT:-qwen35_vtc_grpo}"
export WANDB_NAME="${WANDB_NAME:-qwen35_vtc_random_grounding_step7000_train10k_8gpu}"

exec "${SCRIPT_DIR}/run_grpo_8gpu_baseline.sh"
