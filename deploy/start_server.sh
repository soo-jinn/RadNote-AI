#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export RADNOTE_ENABLE_OCR="${RADNOTE_ENABLE_OCR:-1}"
export RADNOTE_ALLOW_MODEL_DOWNLOAD="${RADNOTE_ALLOW_MODEL_DOWNLOAD:-0}"
exec .venv/bin/python -m uvicorn api.main:app --host "${RADNOTE_BIND:-127.0.0.1}" --port "${RADNOTE_PORT:-8000}"
