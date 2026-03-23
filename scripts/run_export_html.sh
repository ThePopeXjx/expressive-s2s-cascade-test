#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

OUTPUTS_DIR="${PROJECT_ROOT}/outputs"
EXPORT_DIR="${OUTPUTS_DIR}/static_viewer_indextts2"
TARGET_SPEECH_SUBDIR="speech_indextts2"
MAX_SAMPLES=""
OVERWRITE=true
ASSET_MODE="hardlink"

ARGS=(
  --outputs-dir "${OUTPUTS_DIR}"
  --export-dir "${EXPORT_DIR}"
  --target-speech-subdir "${TARGET_SPEECH_SUBDIR}"
  --asset-mode "${ASSET_MODE}"
)

if [[ -n "${MAX_SAMPLES}" ]]; then
  ARGS+=(--max-samples "${MAX_SAMPLES}")
fi

if [[ "${OVERWRITE}" == "true" ]]; then
  ARGS+=(--overwrite)
fi

python "${PROJECT_ROOT}/codes/export_static_html.py" \
  "${ARGS[@]}" \
  "$@"
