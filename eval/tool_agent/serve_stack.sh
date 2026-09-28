#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:-}"
if [[ -z "${CONFIG}" ]]; then
  echo "usage: $0 configs/<benchmark>.json" >&2
  exit 2
fi
CONFIG="$(realpath "${CONFIG}")"
CONFIG_PYTHON="${CONFIG_PYTHON:-python}"

readarray -t values < <("${CONFIG_PYTHON}" - "${ROOT}" "${CONFIG}" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
from config import load_config
c = load_config(sys.argv[2]); b = c["backend"]
for value in (c["model"], c["served_model_name"], b["python"], b["port"],
              b["tensor_parallel_size"], b["max_model_len"], b["max_images"],
              b["max_num_seqs"], b["gpu_memory_utilization"], b["allowed_local_media_path"],
              c["gateway_port"], c["chat_template_path"], int(b["enable_prefix_caching"]),
              int(b["enable_chunked_prefill"]), b["mamba_cache_mode"]):
    print(value)
PY
)
MODEL="${values[0]}"
SERVED_NAME="${values[1]}"
BACKEND_PYTHON="${BACKEND_PYTHON:-${values[2]}}"
BACKEND_PORT="${values[3]}"
TP="${values[4]}"
MAX_MODEL_LEN="${values[5]}"
MAX_IMAGES="${values[6]}"
MAX_NUM_SEQS="${values[7]}"
GPU_MEM="${values[8]}"
MEDIA_ROOT="${values[9]}"
GATEWAY_PORT="${values[10]}"
CHAT_TEMPLATE="${values[11]}"
PREFIX_CACHING="${values[12]}"
CHUNKED_PREFILL="${values[13]}"
MAMBA_CACHE_MODE="${values[14]}"

cache_args=(--no-enable-prefix-caching)
if [[ "${PREFIX_CACHING}" == "1" ]]; then
  cache_args=(--enable-prefix-caching --mamba-cache-mode "${MAMBA_CACHE_MODE}")
fi
prefill_args=(--no-enable-chunked-prefill)
if [[ "${CHUNKED_PREFILL}" == "1" ]]; then
  prefill_args=(--enable-chunked-prefill)
fi

if [[ ! -f "${MODEL}/config.json" ]]; then
  echo "Set FOCUSVTC_MODEL to a merged Hugging Face checkpoint: ${MODEL}" >&2
  exit 2
fi

LOG_DIR="${FOCUSVTC_OUTPUT_ROOT:-${ROOT}/../../outputs}/logs"
mkdir -p "${LOG_DIR}"
backend_pid=""
gateway_pid=""
cleanup() {
  if [[ -n "${gateway_pid}" ]] && kill -0 "${gateway_pid}" 2>/dev/null; then
    kill "${gateway_pid}" 2>/dev/null || true
    wait "${gateway_pid}" 2>/dev/null || true
  fi
  if [[ -n "${backend_pid}" ]] && kill -0 "${backend_pid}" 2>/dev/null; then
    kill "${backend_pid}" 2>/dev/null || true
    wait "${backend_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

"${BACKEND_PYTHON}" -m vllm.entrypoints.openai.api_server \
  --model "${MODEL}" --served-model-name "${SERVED_NAME}" \
  --host 127.0.0.1 --port "${BACKEND_PORT}" \
  --tensor-parallel-size "${TP}" --max-model-len "${MAX_MODEL_LEN}" \
  --limit-mm-per-prompt "{\"image\":${MAX_IMAGES}}" \
  --max-num-seqs "${MAX_NUM_SEQS}" --gpu-memory-utilization "${GPU_MEM}" \
  --allowed-local-media-path "${MEDIA_ROOT}" --reasoning-parser qwen3 \
  --chat-template "${CHAT_TEMPLATE}" \
  "${cache_args[@]}" "${prefill_args[@]}" \
  >"${LOG_DIR}/backend.$(basename "${CONFIG}" .json).log" 2>&1 &
backend_pid=$!

deadline=$((SECONDS + ${STARTUP_TIMEOUT:-2400}))
until curl --noproxy '*' -sf "http://127.0.0.1:${BACKEND_PORT}/health" >/dev/null 2>&1; do
  if ! kill -0 "${backend_pid}" 2>/dev/null; then
    tail -80 "${LOG_DIR}/backend.$(basename "${CONFIG}" .json).log" >&2
    exit 1
  fi
  if (( SECONDS > deadline )); then
    echo "backend startup timed out" >&2
    exit 1
  fi
  sleep 5
done

echo "Backend ready on :${BACKEND_PORT}; tool gateway starting on :${GATEWAY_PORT}"
"${CONFIG_PYTHON}" "${ROOT}/gateway.py" --config "${CONFIG}" &
gateway_pid=$!
wait "${gateway_pid}"
