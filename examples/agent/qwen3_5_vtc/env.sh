#!/usr/bin/env bash
# Source this file before running the Qwen3.5-VL GRPO recipe.
export VTC_GRPO_ROOT="${VTC_GRPO_ROOT:-/vepfs-mlp2/c20250405/400042/VTC/train/GRPO}"
export VTC_GRPO_PY="${VTC_GRPO_PY:-/vepfs-mlp2/c20250405/400042/miniconda3/envs/vtc-grpo/bin/python}"
export PATH="$(dirname "${VTC_GRPO_PY}"):${PATH}"
export PYTHONPATH="${VTC_GRPO_ROOT}:${PYTHONPATH:-}"
# vLLM 0.19.1 external-launcher/SPMD requirement.
export VLLM_ENABLE_V1_MULTIPROCESSING=0

echo "VTC_GRPO_ROOT=${VTC_GRPO_ROOT}"
echo "VTC_GRPO_PY=${VTC_GRPO_PY}"
"${VTC_GRPO_PY}" - <<'PY'
import torch, transformers, vllm
print(f"torch={torch.__version__}")
print(f"transformers={transformers.__version__}")
print(f"vllm={vllm.__version__}")
PY
