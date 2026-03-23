#!/usr/bin/env bash
set -euo pipefail

INDEXTTS2_PYTHON="/home/jiaxingxu/index-tts/.venv/bin/python"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

INDEXTTS2_ROOT="/home/jiaxingxu/index-tts"
INDEXTTS2_MODEL_DIR="/mnt/data1/jiaxingxu/pretrained_models/IndexTTS-2"
HF_CACHE_DIR="/mnt/data1/jiaxingxu/.cache/huggingface"
INDEXTTS2_CFG_PATH=""
USE_FP16=false
USE_CUDA_KERNEL=false
USE_DEEPSPEED=false

OUTPUT_DIR="${PROJECT_ROOT}/outputs"
TRANSCRIPT_DIR="${OUTPUT_DIR}/transcript"
PROMPT_AUDIO_DIR="${OUTPUT_DIR}/audio"
SPEECH_INDEXTTS2_DIR="${OUTPUT_DIR}/speech_indextts2"
LOGS_DIR="${PROJECT_ROOT}/logs"

SAMPLE_START=0
SAMPLE_END=""
MAX_SAMPLES=""
MAX_SAMPLES_PER_STYLE=25

IO_WORKERS=4
LOG_LEVEL="INFO"
LOG_FILE_PREFIX="run_tts_indextts2"
RESUME=true
CONTINUE_ON_ERROR=true

ARGS=(
  --indextts2-root "${INDEXTTS2_ROOT}"
  --indextts2-model-dir "${INDEXTTS2_MODEL_DIR}"
  --hf-cache-dir "${HF_CACHE_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --transcript-dir "${TRANSCRIPT_DIR}"
  --prompt-audio-dir "${PROMPT_AUDIO_DIR}"
  --speech-indextts2-dir "${SPEECH_INDEXTTS2_DIR}"
  --sample-start "${SAMPLE_START}"
  --io-workers "${IO_WORKERS}"
  --logs-dir "${LOGS_DIR}"
  --log-level "${LOG_LEVEL}"
  --log-file-prefix "${LOG_FILE_PREFIX}"
)

if [[ -n "${INDEXTTS2_CFG_PATH}" ]]; then
  ARGS+=(--indextts2-cfg-path "${INDEXTTS2_CFG_PATH}")
fi

if [[ -n "${SAMPLE_END}" ]]; then
  ARGS+=(--sample-end "${SAMPLE_END}")
fi

if [[ -n "${MAX_SAMPLES}" ]]; then
  ARGS+=(--max-samples "${MAX_SAMPLES}")
fi

if [[ -n "${MAX_SAMPLES_PER_STYLE}" ]]; then
  ARGS+=(--max-samples-per-style "${MAX_SAMPLES_PER_STYLE}")
fi

if [[ "${USE_FP16}" == "true" ]]; then
  ARGS+=(--use-fp16)
fi

if [[ "${USE_CUDA_KERNEL}" == "true" ]]; then
  ARGS+=(--use-cuda-kernel)
fi

if [[ "${USE_DEEPSPEED}" == "true" ]]; then
  ARGS+=(--use-deepspeed)
fi

if [[ "${RESUME}" == "true" ]]; then
  ARGS+=(--resume)
else
  ARGS+=(--no-resume)
fi

if [[ "${CONTINUE_ON_ERROR}" == "true" ]]; then
  ARGS+=(--continue-on-error)
else
  ARGS+=(--stop-on-error)
fi

"${INDEXTTS2_PYTHON}" "${PROJECT_ROOT}/codes/tts_indextts2.py" \
  "${ARGS[@]}" \
  "$@"
