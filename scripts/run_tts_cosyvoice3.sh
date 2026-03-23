#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

COSYVOICE3_ROOT="/home/jiaxingxu/CosyVoice"
COSYVOICE3_MODEL_DIR="/mnt/data1/jiaxingxu/pretrained_models/Fun-CosyVoice3-0.5B"

OUTPUT_DIR="${PROJECT_ROOT}/outputs"
TRANSCRIPT_DIR="${OUTPUT_DIR}/transcript"
PROMPT_AUDIO_DIR="${OUTPUT_DIR}/audio"
SPEECH_COSYVOICE3_DIR="${OUTPUT_DIR}/speech_cosyvoice3"
LOGS_DIR="${PROJECT_ROOT}/logs"

SYSTEM_PROMPT="You are a helpful assistant.<|endofprompt|>"
USE_SYSTEM_PROMPT=true
STREAM=false

SAMPLE_START=0
SAMPLE_END=""
MAX_SAMPLES=""
MAX_SAMPLES_PER_STYLE=25

IO_WORKERS=4
LOG_LEVEL="INFO"
LOG_FILE_PREFIX="run_tts_cosyvoice3"
RESUME=true
CONTINUE_ON_ERROR=true

ARGS=(
  --cosyvoice3-root "${COSYVOICE3_ROOT}"
  --cosyvoice3-model-dir "${COSYVOICE3_MODEL_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --transcript-dir "${TRANSCRIPT_DIR}"
  --prompt-audio-dir "${PROMPT_AUDIO_DIR}"
  --speech-cosyvoice3-dir "${SPEECH_COSYVOICE3_DIR}"
  --system-prompt "${SYSTEM_PROMPT}"
  --sample-start "${SAMPLE_START}"
  --io-workers "${IO_WORKERS}"
  --logs-dir "${LOGS_DIR}"
  --log-level "${LOG_LEVEL}"
  --log-file-prefix "${LOG_FILE_PREFIX}"
)

if [[ -n "${SAMPLE_END}" ]]; then
  ARGS+=(--sample-end "${SAMPLE_END}")
fi

if [[ -n "${MAX_SAMPLES}" ]]; then
  ARGS+=(--max-samples "${MAX_SAMPLES}")
fi

if [[ -n "${MAX_SAMPLES_PER_STYLE}" ]]; then
  ARGS+=(--max-samples-per-style "${MAX_SAMPLES_PER_STYLE}")
fi

if [[ "${USE_SYSTEM_PROMPT}" == "true" ]]; then
  ARGS+=(--use-system-prompt)
else
  ARGS+=(--no-system-prompt)
fi

if [[ "${STREAM}" == "true" ]]; then
  ARGS+=(--stream)
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

python "${PROJECT_ROOT}/codes/tts_cosyvoice3.py" \
  "${ARGS[@]}" \
  "$@"
