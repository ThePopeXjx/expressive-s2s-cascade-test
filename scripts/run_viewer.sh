#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

HOST="0.0.0.0"
PORT="8501"

streamlit run "${PROJECT_ROOT}/codes/view_samples.py" \
  --server.address "${HOST}" \
  --server.port "${PORT}" \
  "$@"
