#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

PFP_PYTHON="${PUFFIN_PYTHON:-python}"

if [ ! -x "$PFP_PYTHON" ]; then
  echo "PUFFIN Python not found: $PFP_PYTHON" >&2
  echo "Activate the PUFFIN environment or set PUFFIN_PYTHON explicitly." >&2
  exit 1
fi

exec "$PFP_PYTHON" src/gradio_app.py
