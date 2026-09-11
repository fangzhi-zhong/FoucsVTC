#!/usr/bin/env bash
set -euo pipefail

# This is the first, intentionally small, Qwen3.5-VL VTC GRPO run.  It uses
# four sampled trajectories per prompt and at most three zoom turns.  Increase
# the data shard/steps only after this run has produced valid tool calls.

ROOT="${VTC_GRPO_ROOT:-/vepfs-mlp2/c20250405/400042/VTC/train/GRPO}"
PY="${VTC_GRPO_PY:-/vepfs-mlp2/c20250405/400042/miniconda3/envs/vtc-grpo/bin/python}"
MODEL="${VTC_GRPO_MODEL:-/vepfs-mlp2/c20250405/400042/VTC/train/SFT/output/qwen3_5_9b_vtc_250k_freeze_visual_lr_1e-6_1000steps_merged}"
TRAIN="${VTC_GRPO_TRAIN:-${ROOT}/examples/data/qwen35_vtc/train_8_deepeyes_prompt.parquet}"
VAL="${VTC_GRPO_VAL:-${ROOT}/examples/data/qwen35_vtc/val_4_deepeyes_prompt.parquet}"
TOOLS="${VTC_GRPO_TOOLS:-${ROOT}/examples/agent/qwen3_vl_vtc_tool/zoom_region_tools.json}"
MM_PROCESSOR_CACHE_GB="${VTC_GRPO_MM_PROCESSOR_CACHE_GB:-8}"

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
# Required by the vLLM 0.19.1 external-launcher SPMD backend.  The Python
# entrypoint also propagates this variable into Ray workers.
export VLLM_ENABLE_V1_MULTIPROCESSING=0

cd "${ROOT}"
# response_length is the *whole* agent trajectory budget (actions + tool
# observations), while single_response_max_tokens below caps each turn.
# 10240 leaves room for three zoom/retry turns after the 16k visual prompt.
# Keep the actor/Adam states on CPU between FSDP and vLLM phases; this is
# important for a 9B model plus a 24K multimodal sequence on 80GB cards.
exec "${PY}" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=false \
  data.train_files="['${TRAIN}']" \
  data.val_files="['${VAL}']" \
  data.train_batch_size=128 \
  data.max_prompt_length=16384 \
  data.max_response_length=10240 \
  data.filter_overlong_prompts=true \
  data.truncation=error \
  data.image_key=images \
  data.high_res_image_key=high_res_images \
  data.tools_schema_path="${TOOLS}" \
  data.tools_key=tools \
  data.tools_enabled_key=enable_tools \
  actor_rollout_ref.model.path="${MODEL}" \
  actor_rollout_ref.model.attn_implementation=flash_attention_2 \
  actor_rollout_ref.model.use_remove_padding=true \
  actor_rollout_ref.model.use_liger=true \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=128 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=26624 \
  actor_rollout_ref.actor.use_dynamic_bsz=false \
  actor_rollout_ref.actor.use_kl_loss=false \
  actor_rollout_ref.actor.use_torch_compile=false \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.prompt_length=16384 \
  actor_rollout_ref.rollout.response_length=10240 \
  actor_rollout_ref.rollout.max_model_len=26624 \
  actor_rollout_ref.rollout.max_num_batched_tokens=28672 \
  actor_rollout_ref.rollout.max_num_seqs=32 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
  actor_rollout_ref.rollout.enable_chunked_prefill=true \
  actor_rollout_ref.rollout.enforce_eager=false \
  actor_rollout_ref.rollout.free_cache_engine=false \
  +actor_rollout_ref.rollout.mm_processor_cache_gb="${MM_PROCESSOR_CACHE_GB}" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.rollout.agent.activate_agent=true \
  actor_rollout_ref.rollout.agent.tool_name_key=env_name \
  actor_rollout_ref.rollout.agent.tool_meta_key=null \
  actor_rollout_ref.rollout.agent.vl_model_path="${MODEL}" \
  actor_rollout_ref.rollout.agent.single_response_max_tokens=1024 \
  actor_rollout_ref.rollout.agent.max_turns=5 \
  actor_rollout_ref.rollout.agent.max_tool_calls=3 \
  actor_rollout_ref.rollout.agent.concurrent_workers=1 \
  actor_rollout_ref.rollout.agent.show_tqdm=false \
  reward_model.enable=false \
  custom_reward_function.path="${ROOT}/examples/reward_function/qwen35_vtc_reward.py" \
  custom_reward_function.name=compute_score \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1 \
  trainer.total_epochs=1 \
  trainer.total_training_steps=1 \
  trainer.resume_mode=disable \
  trainer.val_before_train=false \
  trainer.test_freq=-1 \
  trainer.save_freq=-1 \
  trainer.logger="['console']" \
  trainer.project_name=qwen35_vtc_grpo \
  trainer.experiment_name=debug_8rows
