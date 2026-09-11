#!/usr/bin/env bash
set -euo pipefail

# Qwen3.5-VL-9B VTC GRPO baseline for one 8-GPU node.
#
# The default batch/n values give 16 trajectories per optimizer data shard:
#   train_batch_size (32) * rollout.n (4) / 8 GPUs = 16 per rank.
# ppo_mini_batch_size is normalized by verl across ranks, while the explicit
# *_per_gpu micro-batches are already per rank.  Override any value with an
# environment variable when doing a memory sweep.

ROOT="${VTC_GRPO_ROOT:-/vepfs-mlp2/c20250405/400042/VTC/train/GRPO}"
PY="${VTC_GRPO_PY:-/vepfs-mlp2/c20250405/400042/miniconda3/envs/vtc-grpo/bin/python}"
MODEL="${VTC_GRPO_MODEL:-/vepfs-mlp2/c20250405/400042/VTC/train/SFT/output/qwen3_5_9b_vtc_250k_random_grounding_freeze_visual/checkpoint-7000_merged}"
DATA_ROOT="${VTC_GRPO_DATA_ROOT:-/vepfs-mlp2/c20250405/400042/data/VTC/GRPO}"
TRAIN="${VTC_GRPO_TRAIN:-${DATA_ROOT}/train.parquet}"
VAL="${VTC_GRPO_VAL:-${DATA_ROOT}/val_500_uniform.parquet}"
TOOLS="${VTC_GRPO_TOOLS:-${ROOT}/examples/agent/qwen3_vl_vtc_tool/zoom_region_tools.json}"
OUTPUT="${VTC_GRPO_OUTPUT:-${ROOT}/output/qwen35_vtc_50k_bs32_rn8_mini32_micro2}"

# WandB uses the same credentials as the SFT jobs.  On this machine the
# credential is stored in the user's ~/.netrc, so it is picked up by the
# WandB SDK without copying the API key into this repository.  Set
# WANDB_MODE=offline explicitly if an offline run is desired.
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-qwen35_vtc_grpo}"
export WANDB_NAME="${WANDB_NAME:-qwen35_vtc_50k_bs32_rn8_mini32_micro2}"

if [[ "${WANDB_MODE}" == "online" ]]; then
  WANDB_NETRC="${NETRC:-${HOME}/.netrc}"
  _wandb_netrc_ok=false
  if [[ -n "${WANDB_API_KEY:-}" ]]; then
    _wandb_netrc_ok=true
  elif [[ -f "${WANDB_NETRC}" ]]; then
    if command -v rg >/dev/null 2>&1; then
      rg -q 'machine[[:space:]]+(api\.)?wandb\.ai' "${WANDB_NETRC}" && _wandb_netrc_ok=true
    else
      grep -Eq 'machine[[:space:]]+(api\.)?wandb\.ai' "${WANDB_NETRC}" && _wandb_netrc_ok=true
    fi
  fi
  if [[ "${_wandb_netrc_ok}" != true ]]; then
    echo "W&B online mode requested, but no API key or W&B netrc entry was found." >&2
    echo "Run 'wandb login' in the vtc-grpo environment, or set WANDB_MODE=offline." >&2
    exit 2
  fi
fi

TRAIN_BATCH="${VTC_TRAIN_BATCH:-64}"
VAL_BATCH="${VTC_VAL_BATCH:-8}"
VAL_NUM_WORKERS="${VTC_VAL_NUM_WORKERS:-0}"
ROLLOUT_N="${VTC_ROLLOUT_N:-8}"
PPO_MINI="${VTC_PPO_MINI:-64}"
PPO_MICRO="${VTC_PPO_MICRO:-1}"
ROLLOUT_LOGPROB_MICRO="${VTC_ROLLOUT_LOGPROB_MICRO:-1}"
ENTROPY_CHUNK_SIZE="${VTC_ENTROPY_CHUNK_SIZE:-4096}"
PROMPT_LENGTH="${VTC_MAX_PROMPT_LENGTH:-32768}"
PROMPT_FILTER_METHOD="${VTC_PROMPT_FILTER_METHOD:-image_size}"
PROMPT_FILTER_WORKERS="${VTC_PROMPT_FILTER_WORKERS:-8}"
RESPONSE_LENGTH="${VTC_MAX_RESPONSE_LENGTH:-16384}"
MAX_MODEL_LEN=$((PROMPT_LENGTH + RESPONSE_LENGTH))
MAX_NUM_BATCHED_TOKENS="${VTC_MAX_NUM_BATCHED_TOKENS:-50000}"
MAX_NUM_SEQS="${VTC_MAX_NUM_SEQS:-32}"
ENFORCE_EAGER="${VTC_ENFORCE_EAGER:-false}"
GPU_MEMORY_UTIL="${VTC_GPU_MEMORY_UTILIZATION:-0.80}"
MM_PROCESSOR_CACHE_GB="${VTC_GRPO_MM_PROCESSOR_CACHE_GB:-8}"
TOTAL_STEPS="${VTC_TOTAL_STEPS:-100}"
TOTAL_EPOCHS="${VTC_TOTAL_EPOCHS:-1}"
SAVE_FREQ="${VTC_SAVE_FREQ:-25}"
TEST_FREQ="${VTC_TEST_FREQ:-0}"

if [[ ! -x "${PY}" ]]; then
  echo "Python executable not found: ${PY}" >&2
  exit 2
fi
for required in "${MODEL}/config.json" "${TRAIN}" "${VAL}" "${TOOLS}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Required path not found: ${required}" >&2
    exit 2
  fi
done

export PYTHONUNBUFFERED=1
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
# Required by the vLLM external-launcher/SPMD backend in this checkout.
export VLLM_ENABLE_V1_MULTIPROCESSING=0

cd "${ROOT}"
echo "Starting 8-GPU Qwen3.5-VL GRPO baseline"
echo "  model=${MODEL}"
echo "  train=${TRAIN}"
echo "  val=${VAL}"
echo "  val_batch=${VAL_BATCH}, val_num_workers=${VAL_NUM_WORKERS}, test_freq=${TEST_FREQ}"
echo "  output=${OUTPUT}"
echo "  batch=${TRAIN_BATCH}, rollout.n=${ROLLOUT_N}, ppo_mini=${PPO_MINI}, ppo_micro_per_gpu=${PPO_MICRO}"
echo "  logprob_micro_per_gpu=${ROLLOUT_LOGPROB_MICRO}, entropy_chunk_size=${ENTROPY_CHUNK_SIZE}"
echo "  prompt_length=${PROMPT_LENGTH}, response_length=${RESPONSE_LENGTH}, max_model_len=${MAX_MODEL_LEN}"
echo "  prompt_filter_method=${PROMPT_FILTER_METHOD}, filter_workers=${PROMPT_FILTER_WORKERS}, skip_overlong_prompts=true"
echo "  max_num_seqs=${MAX_NUM_SEQS}, max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS}, enforce_eager=${ENFORCE_EAGER}"
echo "  mm_processor_cache_gb=${MM_PROCESSOR_CACHE_GB}"
echo "  total_epochs=${TOTAL_EPOCHS}, total_training_steps=${TOTAL_STEPS}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<launcher default>}"

# A zero TOTAL_STEPS lets verl run the configured number of epochs over the
# 50K shard.  Set VTC_TOTAL_STEPS for an exact step limit; TOTAL_EPOCHS must be
# large enough to provide that many dataloader batches.
TOTAL_STEPS_OVERRIDE=()
if [[ "${TOTAL_STEPS}" != "0" ]]; then
  TOTAL_STEPS_OVERRIDE=("trainer.total_training_steps=${TOTAL_STEPS}")
fi

exec "${PY}" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=false \
  data.train_files="['${TRAIN}']" \
  data.val_files="['${VAL}']" \
  data.train_batch_size="${TRAIN_BATCH}" \
  data.val_batch_size="${VAL_BATCH}" \
  data.val_num_workers="${VAL_NUM_WORKERS}" \
  data.max_prompt_length="${PROMPT_LENGTH}" \
  data.max_response_length="${RESPONSE_LENGTH}" \
  data.filter_overlong_prompts=true \
  data.filter_overlong_prompts_method="${PROMPT_FILTER_METHOD}" \
  data.filter_overlong_prompts_workers="${PROMPT_FILTER_WORKERS}" \
  data.skip_overlong_prompts=true \
  data.truncation=error \
  data.image_key=images \
  data.high_res_image_key=high_res_images \
  data.tools_schema_path="${TOOLS}" \
  data.tools_key=tools \
  data.tools_enabled_key=enable_tools \
  actor_rollout_ref.model.path="${MODEL}" \
  actor_rollout_ref.model.attn_implementation=flash_attention_2 \
  actor_rollout_ref.model.use_remove_padding=false \
  actor_rollout_ref.model.use_liger=true \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO}" \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${MAX_MODEL_LEN}" \
  actor_rollout_ref.actor.use_dynamic_bsz=false \
  actor_rollout_ref.actor.use_kl_loss=false \
  actor_rollout_ref.actor.use_torch_compile=false \
  actor_rollout_ref.actor.entropy_chunk_size="${ENTROPY_CHUNK_SIZE}" \
  actor_rollout_ref.actor.fsdp_config.param_offload=true \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.prompt_length="${PROMPT_LENGTH}" \
  actor_rollout_ref.rollout.response_length="${RESPONSE_LENGTH}" \
  actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}" \
  actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS}" \
  actor_rollout_ref.rollout.max_num_seqs="${MAX_NUM_SEQS}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTIL}" \
  actor_rollout_ref.rollout.enable_chunked_prefill=true \
  actor_rollout_ref.rollout.enforce_eager="${ENFORCE_EAGER}" \
  actor_rollout_ref.rollout.free_cache_engine=false \
  +actor_rollout_ref.rollout.mm_processor_cache_gb="${MM_PROCESSOR_CACHE_GB}" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${ROLLOUT_LOGPROB_MICRO}" \
  actor_rollout_ref.rollout.agent.activate_agent=true \
  actor_rollout_ref.rollout.agent.tool_name_key=env_name \
  actor_rollout_ref.rollout.agent.tool_meta_key=null \
  actor_rollout_ref.rollout.agent.vl_model_path="${MODEL}" \
  actor_rollout_ref.rollout.agent.single_response_max_tokens=2048 \
  actor_rollout_ref.rollout.agent.max_turns=9 \
  actor_rollout_ref.rollout.agent.max_tool_calls=8 \
  actor_rollout_ref.rollout.agent.concurrent_workers=1 \
  actor_rollout_ref.rollout.agent.show_tqdm=false \
  reward_model.enable=false \
  custom_reward_function.path="${ROOT}/examples/reward_function/qwen35_vtc_reward.py" \
  custom_reward_function.name=compute_score \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=4 \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  trainer.resume_mode=disable \
  trainer.val_before_train=false \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.default_local_dir="${OUTPUT}" \
  trainer.logger="['console','wandb']" \
  trainer.project_name="${WANDB_PROJECT}" \
  trainer.experiment_name="${WANDB_NAME}" \
  "${TOTAL_STEPS_OVERRIDE[@]}"
