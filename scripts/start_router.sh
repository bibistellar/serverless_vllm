#!/bin/bash

export ROUTER_HOST="${ROUTER_HOST:-0.0.0.0}"
export ROUTER_PORT="${ROUTER_PORT:-8000}"
export MANAGER_URL="${MANAGER_URL:-http://localhost:9000}"

echo "Starting Router Service..."
python -m src.router.service
