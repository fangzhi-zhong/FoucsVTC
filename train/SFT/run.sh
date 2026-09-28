#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${SCRIPT_DIR}/lmms-engine/src:${PYTHONPATH:-}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
exec torchrun --nproc_per_node="${NGPUS:-8}" \
  --nnodes="${NNODES:-1}" --node_rank="${NODE_RANK:-0}" \
  --master_addr="${MASTER_ADDR:-127.0.0.1}" --master_port="${MASTER_PORT:-29500}" \
  -m lmms_engine.launch.cli \
  config_yaml="${CONFIG:-${SCRIPT_DIR}/configs/qwen3_5.yaml}" \
  hydra.run.dir=. hydra.output_subdir=null hydra.job.chdir=false
