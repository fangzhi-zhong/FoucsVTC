#!/usr/bin/env bash
set -euo pipefail

# Two-GPU, 200-step Qwen3.5-VL VTC GRPO run.
#
# This reuses the memory-tested dual-GPU recipe while switching to the
# constructed 50K training shard.  The underlying smoke script now exposes
# total steps/epochs as environment overrides, so this launcher remains easy
# to adjust without duplicating the full Hydra command.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${VTC_GRPO_ROOT:-/vepfs-mlp2/c20250405/400042/VTC/train/GRPO}"
DATA_ROOT="${VTC_GRPO_DATA_ROOT:-/vepfs-mlp2/c20250405/400042/data/VTC/GRPO}"

export VTC_GRPO_TOTAL_STEPS="${VTC_GRPO_TOTAL_STEPS:-200}"
export VTC_GRPO_TOTAL_EPOCHS="${VTC_GRPO_TOTAL_EPOCHS:-1}"
export VTC_GRPO_MM_PROCESSOR_CACHE_GB="${VTC_GRPO_MM_PROCESSOR_CACHE_GB:-8}"
export VTC_GRPO_TRAIN="${VTC_GRPO_TRAIN:-${DATA_ROOT}/train.parquet}"
export VTC_GRPO_VAL="${VTC_GRPO_VAL:-${DATA_ROOT}/val.parquet}"
export VTC_GRPO_OUTPUT="${VTC_GRPO_OUTPUT:-${DATA_ROOT}/checkpoints/qwen35_vtc_2gpu_200steps}"
export WANDB_PROJECT="${WANDB_PROJECT:-qwen35_vtc_grpo}"
export WANDB_NAME="${WANDB_NAME:-qwen35_vtc_2gpu_200steps}"

exec "${SCRIPT_DIR}/run_grpo_2gpu_smoke.sh"
