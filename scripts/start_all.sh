#!/bin/bash

# 启动 Manager 服务
echo "Starting Manager Service..."
python -m src.manager.service &
MANAGER_PID=$!
echo "Manager started with PID: $MANAGER_PID"

# 等待 Manager 启动
sleep 5

# 启动 Worker 服务
echo "Starting Worker Service..."
export WORKER_ID="${WORKER_ID:-worker-1}"
export WORKER_HOST="${WORKER_HOST:-0.0.0.0}"
export WORKER_PORT="${WORKER_PORT:-7000}"
export MANAGER_URL="${MANAGER_URL:-http://localhost:9000}"
export PUBLIC_URL="${PUBLIC_URL:-}"

if [ -n "$PUBLIC_URL" ]; then
    echo "Worker Public URL: $PUBLIC_URL"
fi

python -m src.worker.service &
WORKER_PID=$!
echo "Worker started with PID: $WORKER_PID"

# 等待 Worker 启动
sleep 5

# 启动 Router 服务
echo "Starting Router Service..."
export ROUTER_HOST="${ROUTER_HOST:-0.0.0.0}"
export ROUTER_PORT="${ROUTER_PORT:-8000}"
export MANAGER_URL="${MANAGER_URL:-http://localhost:9000}"

python -m src.router.service &
ROUTER_PID=$!
echo "Router started with PID: $ROUTER_PID"

echo ""
echo "All services started!"
echo "Manager: http://localhost:9000"
echo "Worker: http://localhost:7000"
echo "Router (API): http://localhost:8000"
echo ""
echo "To stop all services, run: kill $MANAGER_PID $WORKER_PID $ROUTER_PID"

# 等待所有进程
wait
