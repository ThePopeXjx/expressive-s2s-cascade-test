#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

OUTPUTS_DIR="${PROJECT_ROOT}/outputs"
EXPORT_DIR="${OUTPUTS_DIR}/static_viewer_cosyvoice3"
MAX_SAMPLES=""
OVERWRITE=true
ASSET_MODE="hardlink"

ARGS=(
  --outputs-dir "${OUTPUTS_DIR}"
  --export-dir "${EXPORT_DIR}"
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
