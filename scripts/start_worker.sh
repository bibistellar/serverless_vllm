#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export WORKER_ID="${WORKER_ID:-worker-1}"
export WORKER_HOST="${WORKER_HOST:-0.0.0.0}"
export WORKER_PORT="${WORKER_PORT:-7000}"
export MANAGER_URL="${MANAGER_URL:-http://127.0.0.1:8000}"
export PUBLIC_URL="${PUBLIC_URL:-}"

echo "Starting Worker Service: ${WORKER_ID} on ${WORKER_HOST}:${WORKER_PORT}..."
python -m src.worker.service
