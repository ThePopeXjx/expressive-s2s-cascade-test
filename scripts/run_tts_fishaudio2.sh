#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

FISHAUDIO2_ROOT="/home/jiaxingxu/fish-speech"
LLAMA_CHECKPOINT_PATH="/mnt/data1/jiaxingxu/pretrained_models/FishAudio-S2"
DECODER_CHECKPOINT_PATH="/mnt/data1/jiaxingxu/pretrained_models/FishAudio-S2/codec.pth"
DEVICE="cuda"
HALF=false
COMPILE=false
WARMUP=true

OUTPUT_DIR="${PROJECT_ROOT}/outputs"
TRANSCRIPT_DIR="${OUTPUT_DIR}/transcript-ja"
METADATA_DIR="${OUTPUT_DIR}/metadata"
PROMPT_AUDIO_DIR="${OUTPUT_DIR}/audio"
SPEECH_FISHAUDIO2_DIR="${OUTPUT_DIR}/speech_fishaudio2"
LOGS_DIR="${PROJECT_ROOT}/logs"

SAMPLE_START=0
SAMPLE_END=""
MAX_SAMPLES=""
MAX_SAMPLES_PER_STYLE=25

MAX_NEW_TOKENS=1024
CHUNK_LENGTH=300
TOP_P=0.8
TOP_K=30
REPETITION_PENALTY=1.1
TEMPERATURE=0.8
SEED=""
ITERATIVE_PROMPT=true

IO_WORKERS=4
LOG_LEVEL="INFO"
LOG_FILE_PREFIX="run_tts_fishaudio2"
RESUME=true
CONTINUE_ON_ERROR=true

ARGS=(
  --fishaudio2-root "${FISHAUDIO2_ROOT}"
  --llama-checkpoint-path "${LLAMA_CHECKPOINT_PATH}"
  --decoder-checkpoint-path "${DECODER_CHECKPOINT_PATH}"
  --device "${DEVICE}"
  --output-dir "${OUTPUT_DIR}"
  --transcript-dir "${TRANSCRIPT_DIR}"
  --metadata-dir "${METADATA_DIR}"
  --prompt-audio-dir "${PROMPT_AUDIO_DIR}"
  --speech-fishaudio2-dir "${SPEECH_FISHAUDIO2_DIR}"
  --sample-start "${SAMPLE_START}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --chunk-length "${CHUNK_LENGTH}"
  --top-p "${TOP_P}"
  --top-k "${TOP_K}"
  --repetition-penalty "${REPETITION_PENALTY}"
  --temperature "${TEMPERATURE}"
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

if [[ -n "${SEED}" ]]; then
  ARGS+=(--seed "${SEED}")
fi

if [[ "${ITERATIVE_PROMPT}" == "true" ]]; then
  ARGS+=(--iterative-prompt)
else
  ARGS+=(--no-iterative-prompt)
fi

if [[ "${HALF}" == "true" ]]; then
  ARGS+=(--half)
fi

if [[ "${COMPILE}" == "true" ]]; then
  ARGS+=(--compile)
fi

if [[ "${WARMUP}" == "true" ]]; then
  ARGS+=(--warmup)
else
  ARGS+=(--no-warmup)
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

python "${PROJECT_ROOT}/codes/tts_fishaudio2.py" \
  "${ARGS[@]}" \
  "$@"
