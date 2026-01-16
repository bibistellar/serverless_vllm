# Serverless vLLM - 混合架构

## 新架构说明（Ray + HTTP 混合）

本项目采用混合架构，结合了 Ray 和 HTTP 的优势：

```
┌─────────────────── Ray Cluster ───────────────────┐
│                                                    │
│  ┌─────────────┐        Ray remote 调用          │
│  │   Router    │◄────────────────────────────────┐│
│  │ (Ray Serve) │                                 ││
│  └──────┬──────┘                                 ││
│         │                                        ││
│         │ Ray remote                             ││
│         ▼                                        ││
│  ┌─────────────┐                                ││
│  │   Manager   │                                ││
│  │ (Ray Actor) │                                ││
│  └──────┬──────┘                                ││
│         │                                        ││
└─────────┼────────────────────────────────────────┘│
          │ HTTP                                    │
          ▼                                         │
  ┌───────────────┐  ┌───────────────┐            │
  │  Worker 1     │  │  Worker 2     │            │
  │ (HTTP Service)│  │ (HTTP Service)│            │
  └───────┬───────┘  └───────┬───────┘            │
          │                  │                     │
          │ HTTP             │ HTTP (心跳/注册)    │
          │                  └─────────────────────┘
          ▼
  vLLM Server (随机端口)
```

### 组件职责

#### 1. **Router** (Ray Serve Deployment)
- **部署位置**: Ray Cluster
- **对外端口**: 8000
- **职责**:
  - 提供 OpenAI 兼容的 HTTP API
  - 通过 Ray remote 调用 Manager
  - 将推理请求转发到 Worker
- **优势**: 可在 Ray cluster 内多地域部署

#### 2. **Manager** (Ray Actor)
- **部署位置**: Ray Cluster (detached actor)
- **通信方式**: 
  - 与 Router: Ray remote 调用
  - 与 Worker: HTTP
- **职责**:
  - 维护 Worker 信息和健康状态
  - 管理模型路由表
  - 决策模型部署位置
  - 接收 Worker 注册和心跳

#### 3. **Worker** (独立 HTTP 服务)
- **部署位置**: 各个 GPU 机器
- **通信方式**: HTTP
- **职责**:
  - 检测 GPU 环境
  - 启动和管理 vLLM server
  - 转发推理请求
  - 定期向 Manager 发送心跳

## 部署指南

### 方案一：单机快速部署

**1. 启动 Ray Services (Manager + Router)**
```bash
python start_ray_services.py
```

这将启动：
- Ray Cluster (如果不存在)
- Manager Actor (名称: "manager")
- Router Serve Deployment (端口: 8000)

**2. 启动 Worker**
```bash
# Worker 1
python -m src.worker.service \
  --worker-id worker-1 \
  --manager-url http://localhost:8000 \
  --port 7000

# Worker 2  
python -m src.worker.service \
  --worker-id worker-2 \
  --manager-url http://localhost:8000 \
  --port 7001
```

### 方案二：分布式部署

**1. 中央节点：启动 Ray Head + Manager + Router**
```bash
# 启动 Ray head node
ray start --head --port=6379 --dashboard-host=0.0.0.0

# 启动 Manager + Router
python start_ray_services.py
```

**2. 其他节点：加入 Ray Cluster (可选)**
```bash
# 如果想让 Router 也部署在边缘节点
ray start --address='<head-node-ip>:6379'
```

**3. GPU 节点：启动 Worker**
```bash
# 在每台 GPU 机器上
python -m src.worker.service \
  --worker-id worker-gpu-1 \
  --manager-url http://<router-ip>:8000 \
  --port 7000
```

## API 使用

### 1. 注册模型
```bash
curl -X POST http://localhost:8000/v1/models/register \
  -H "Content-Type: application/json" \
  -d '{
    "alias": "qwen-vl-2b",
    "model_name": "Qwen/Qwen3-VL-2B-Instruct",
    "max_model_len": 8192,
    "gpu_memory_utilization": 0.9
  }'
```

### 2. 聊天补全
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-vl-2b",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ],
    "max_tokens": 100
  }'
```

### 3. 查看状态
```bash
# 健康检查
curl http://localhost:8000/health

# 列出所有模型
curl http://localhost:8000/v1/models

# 列出所有 Worker
curl http://localhost:8000/workers
```

## 架构优势

### ✅ Ray 的优势（Manager + Router）
1. **统一管理**: Manager 和 Router 在同一个 Ray cluster
2. **高效通信**: Router 和 Manager 之间通过 Ray remote 调用（内存共享）
3. **容错机制**: Ray 提供 Actor 重启和监控
4. **灵活扩展**: Router 可以多地域部署
5. **代码简洁**: 不需要 HTTP 序列化/反序列化

### ✅ HTTP 的优势（Worker）
1. **独立部署**: Worker 可以在任何机器上启动
2. **跨机房**: 不需要 Worker 加入 Ray cluster
3. **技术栈解耦**: Worker 可以用任何语言实现
4. **简单调试**: HTTP 请求易于追踪和调试

## 监控和管理

### Ray Dashboard
访问 Ray Dashboard 查看 Actor 和 Serve 状态：
```
http://localhost:8265
```

### 查看 Manager Actor
```python
import ray
ray.init(address="auto")
manager = ray.get_actor("manager")
result = ray.get(manager.health.remote())
print(result)
```

### 查看 Router Deployment
```bash
serve status
```

## 配置说明

### Manager 配置
在 `start_ray_services.py` 中：
```python
manager = ManagerService.options(
    name="manager",
    lifetime="detached",
    max_concurrency=100,
    # 可添加更多 Ray actor options
).remote(heartbeat_timeout=60)
```

### Router 配置
```python
serve.start(
    http_options={
        "host": "0.0.0.0",
        "port": 8000,
        "location": "EveryNode"  # 在每个节点启动
    }
)
```

### Worker 配置
Worker 保持原有的 HTTP 服务架构，通过环境变量或命令行参数配置。

**基本配置:**
- `WORKER_ID`: Worker 唯一标识
- `WORKER_HOST`: 监听地址（默认 0.0.0.0）
- `WORKER_PORT`: 监听端口（默认 7000）
- `MANAGER_URL`: Manager 地址（通过 Router，如 http://localhost:8000）

**端口转发配置（云平台部署）:**
- `PUBLIC_URL`: Worker 的公网访问地址（可选）

示例：
```bash
# 基本配置
export WORKER_ID="worker-1"
export WORKER_PORT="7000"
export MANAGER_URL="http://localhost:8000"
python -m src.worker.service

# 配置端口转发（云平台）
export WORKER_ID="worker-1"
export WORKER_PORT="6006"
export MANAGER_URL="http://localhost:8000"
export PUBLIC_URL="https://u557149-9507-6992150f.bjb2.seetacloud.com:8443"
python -m src.worker.service
```

详细的端口转发配置请参考：[docs/PORT_FORWARDING.md](docs/PORT_FORWARDING.md)

## 故障恢复

### Manager 重启
Manager 设置为 `detached`，即使启动脚本退出也会继续运行。
如果 Manager 崩溃，Ray 会自动重启（需配置）。

### Router 重启  
Router 是无状态的，可以随时重启：
```bash
serve delete router
# 然后重新运行 start_ray_services.py
```

### Worker 重启
Worker 重启后会自动重新注册到 Manager。

## 依赖项

```bash
pip install ray[serve] fastapi uvicorn httpx torch vllm transformers pyyaml pynvml
```

## 迁移说明

从纯 HTTP 架构迁移到混合架构：

**保持不变**:
- Worker 代码和部署方式
- API 接口和使用方式

**改变**:
- Manager 从 HTTP 服务变为 Ray Actor
- Router 从独立服务变为 Ray Serve Deployment
- Router 与 Manager 通信从 HTTP 改为 Ray remote

## License
MIT
