#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export SERVE_HOST="${SERVE_HOST:-0.0.0.0}"
export SERVE_PORT="${SERVE_PORT:-8000}"
export RAY_ADDRESS="${RAY_ADDRESS:-}"
export RAY_WORKING_DIR="${RAY_WORKING_DIR:-$ROOT_DIR}"
export SERVE_MAX_ONGOING_REQUESTS="${SERVE_MAX_ONGOING_REQUESTS:-100}"

echo "Starting Ray Serve (Router+Manager) on ${SERVE_HOST}:${SERVE_PORT}..."
python -m src.serve.service
