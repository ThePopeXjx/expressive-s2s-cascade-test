#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CACHE_DIR="/mnt/data1/jiaxingxu/.cache/huggingface/datasets"
MODEL_CACHE_DIR="/mnt/data1/jiaxingxu/.cache/huggingface"
OUTPUT_DIR="${PROJECT_ROOT}/outputs"
LOGS_DIR="${PROJECT_ROOT}/logs"

DATASET_NAME="ylacombe/expresso"
DATASET_CONFIG=""
SPLIT="train"

MODEL_PATH="Qwen/Qwen3-Omni-30B-A3B-Instruct"
TRANSCRIBE_PROMPT="Translate the speech into Chinese text."
MAX_NEW_TOKENS=1024
USE_FLASH_ATTN2=flash

SAMPLE_START=0
SAMPLE_END=""
MAX_SAMPLES=""

IO_WORKERS=4
LOG_LEVEL="INFO"
LOG_FILE_PREFIX="run_transcribe"
RESUME=true
OVERWRITE_AUDIO=false
CONTINUE_ON_ERROR=true

ARGS=(
  --cache-dir "${CACHE_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --logs-dir "${LOGS_DIR}"
  --dataset-name "${DATASET_NAME}"
  --split "${SPLIT}"
  --model-path "${MODEL_PATH}"
  --model-cache-dir "${MODEL_CACHE_DIR}"
  --transcribe-prompt "${TRANSCRIBE_PROMPT}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --sample-start "${SAMPLE_START}"
  --io-workers "${IO_WORKERS}"
  --log-level "${LOG_LEVEL}"
  --log-file-prefix "${LOG_FILE_PREFIX}"
)

if [[ -n "${DATASET_CONFIG}" ]]; then
  ARGS+=(--dataset-config "${DATASET_CONFIG}")
fi

if [[ -n "${SAMPLE_END}" ]]; then
  ARGS+=(--sample-end "${SAMPLE_END}")
fi

if [[ -n "${MAX_SAMPLES}" ]]; then
  ARGS+=(--max-samples "${MAX_SAMPLES}")
fi

if [[ "${USE_FLASH_ATTN2}" == "true" ]]; then
  ARGS+=(--use-flash-attn2)
fi

if [[ "${RESUME}" == "true" ]]; then
  ARGS+=(--resume)
else
  ARGS+=(--no-resume)
fi

if [[ "${OVERWRITE_AUDIO}" == "true" ]]; then
  ARGS+=(--overwrite-audio)
fi

if [[ "${CONTINUE_ON_ERROR}" == "true" ]]; then
  ARGS+=(--continue-on-error)
else
  ARGS+=(--stop-on-error)
fi

# Two A6000 GPUs for local Qwen-Omni inference.
export CUDA_VISIBLE_DEVICES="0,1"

python "${PROJECT_ROOT}/codes/transcribe.py" \
  "${ARGS[@]}" \
  "$@"
