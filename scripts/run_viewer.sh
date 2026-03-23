#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

HOST="0.0.0.0"
PORT="8501"
OUTPUTS_DIR="${PROJECT_ROOT}/outputs"
SPEECH_COSYVOICE3_SUBDIR="speech_cosyvoice3"
APP_TITLE="Cascade S2S Sample Viewer"
LOGS_DIR="${PROJECT_ROOT}/logs"
LOG_FILE_PREFIX="run_viewer"

mkdir -p "${LOGS_DIR}"
TIMESTAMP="$(TZ=America/New_York date +%m%d%H%M)"
LOG_PATH="${LOGS_DIR}/${LOG_FILE_PREFIX}_${TIMESTAMP}.log"
exec > >(tee -a "${LOG_PATH}") 2>&1
echo "[$(date '+%F %T')] log file: ${LOG_PATH}"

APP_ARGS=(
  --outputs-dir "${OUTPUTS_DIR}"
  --speech-cosyvoice3-subdir "${SPEECH_COSYVOICE3_SUBDIR}"
  --title "${APP_TITLE}"
)

streamlit run "${PROJECT_ROOT}/codes/view_samples.py" \
  --server.address "${HOST}" \
  --server.port "${PORT}" \
  -- \
  "${APP_ARGS[@]}" \
  "$@"
