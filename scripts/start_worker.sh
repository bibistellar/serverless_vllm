#!/bin/bash

export WORKER_ID="${WORKER_ID:-worker-1}"
export WORKER_HOST="${WORKER_HOST:-0.0.0.0}"
export WORKER_PORT="${WORKER_PORT:-9003}"
export MANAGER_URL="${MANAGER_URL:-http://localhost:18000}"
export PUBLIC_URL="${PUBLIC_URL:-}"

echo "Starting Worker Service: $WORKER_ID"
if [ -n "$PUBLIC_URL" ]; then
    echo "Public URL: $PUBLIC_URL"
fi

python -m src.worker.service
