#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
ROOT="${SCRIPT_DIR}"
PY="${VTC_GRPO_PY:-$(command -v python)}"
MODEL="${VTC_GRPO_MODEL:-${REPO_ROOT}/models/focusvtc-sft}"
DATA_ROOT="${VTC_GRPO_DATA_ROOT:-${REPO_ROOT}/datasets/grpo}"
TRAIN="${VTC_GRPO_TRAIN:-${DATA_ROOT}/train.parquet}"
VAL="${VTC_GRPO_VAL:-${DATA_ROOT}/val.parquet}"
TOOLS="${ROOT}/examples/agent/qwen3_vl_vtc_tool/zoom_region_tools.json"
OUTPUT="${VTC_GRPO_OUTPUT:-${REPO_ROOT}/outputs/grpo}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_PROJECT="${WANDB_PROJECT:-FocusVTC}"
export WANDB_NAME="${WANDB_NAME:-focusvtc-grpo}"

TRAIN_BATCH="${VTC_TRAIN_BATCH:-8}"
VAL_BATCH="${VTC_VAL_BATCH:-8}"
VAL_NUM_WORKERS="${VTC_VAL_NUM_WORKERS:-0}"
ROLLOUT_N="${VTC_ROLLOUT_N:-4}"
PPO_MINI="${VTC_PPO_MINI:-8}"
PPO_MICRO="${VTC_PPO_MICRO:-1}"
ROLLOUT_LOGPROB_MICRO="${VTC_ROLLOUT_LOGPROB_MICRO:-1}"
ENTROPY_CHUNK_SIZE="${VTC_ENTROPY_CHUNK_SIZE:-1024}"
PROMPT_LENGTH="${VTC_MAX_PROMPT_LENGTH:-8192}"
PROMPT_FILTER_METHOD="${VTC_PROMPT_FILTER_METHOD:-image_size}"
PROMPT_FILTER_WORKERS="${VTC_PROMPT_FILTER_WORKERS:-8}"
RESPONSE_LENGTH="${VTC_MAX_RESPONSE_LENGTH:-10240}"
MAX_MODEL_LEN=$((PROMPT_LENGTH + RESPONSE_LENGTH))
MAX_NUM_BATCHED_TOKENS="${VTC_MAX_NUM_BATCHED_TOKENS:-32768}"
MAX_NUM_SEQS="${VTC_MAX_NUM_SEQS:-32}"
ENFORCE_EAGER="${VTC_ENFORCE_EAGER:-false}"
GPU_MEMORY_UTIL="${VTC_GPU_MEMORY_UTILIZATION:-0.80}"
MM_PROCESSOR_CACHE_GB="${VTC_GRPO_MM_PROCESSOR_CACHE_GB:-8}"
TOTAL_STEPS="${VTC_TOTAL_STEPS:-0}"
TOTAL_EPOCHS="${VTC_TOTAL_EPOCHS:-1}"
SAVE_FREQ="${VTC_SAVE_FREQ:-50}"
TEST_FREQ="${VTC_TEST_FREQ:-50}"

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

cd "${REPO_ROOT}"
echo "Starting FocusVTC GRPO on ${NGPUS:-8} GPUs"
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
# training shard.  Set VTC_TOTAL_STEPS for an exact step limit; TOTAL_EPOCHS must be
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
  trainer.n_gpus_per_node="${NGPUS:-8}" \
  trainer.nnodes=1 \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  trainer.resume_mode=disable \
  trainer.val_before_train=false \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.default_local_dir="${OUTPUT}" \
  trainer.logger="${VTC_LOGGERS:-['console']}" \
  trainer.project_name="${WANDB_PROJECT}" \
  trainer.experiment_name="${WANDB_NAME}" \
  "${TOTAL_STEPS_OVERRIDE[@]}" \
  "$@"
