# 端口转发配置指南

## 功能说明

当Worker服务部署在需要端口转发的环境中（如云平台、容器等），可以配置公网访问URL，使Manager能够通过公网URL访问Worker服务。

## 场景示例

Worker监听在本地端口：
```
http://127.0.0.1:6006
```

但从公网访问需要使用：
```
https://u557149-9507-6992150f.bjb2.seetacloud.com:8443
```

## 配置方法

### 方式1：环境变量

启动Worker时设置`PUBLIC_URL`环境变量：

```bash
export WORKER_ID="worker-1"
export WORKER_HOST="0.0.0.0"
export WORKER_PORT="6006"
export MANAGER_URL="http://localhost:18000"
export PUBLIC_URL="https://u557149-9507-6992150f.bjb2.seetacloud.com:8443"

python -m src.worker.service
```

### 方式2：代码配置

```python
from src.worker.service import WorkerService

service = WorkerService(
    worker_id="worker-1",
    listen_host="0.0.0.0",
    listen_port=6006,
    manager_url="http://localhost:18000",
    public_url="https://u557149-9507-6992150f.bjb2.seetacloud.com:8443"
)

service.run()
```

## 工作原理

1. **Worker注册**：Worker向Manager注册时，会同时发送内网URL和公网URL
   ```json
   {
     "worker_id": "worker-1",
     "worker_url": "http://127.0.0.1:6006",
     "public_worker_url": "https://u557149-9507-6992150f.bjb2.seetacloud.com:8443",
     "gpu_info": {...}
   }
   ```

2. **Manager保存**：Manager保存Worker的两个URL
   - `worker_url`: 内网URL，用于Manager直接管理Worker（启动/停止实例等）
   - `public_worker_url`: 公网URL，用于模型路由

3. **模型路由**：当注册模型时，Manager创建路由记录时会优先使用`public_worker_url`
   ```json
   {
     "alias": "qwen-vl-2b",
     "model_name": "Qwen/Qwen3-VL-2B-Instruct",
     "worker_id": "worker-1",
     "worker_url": "https://u557149-9507-6992150f.bjb2.seetacloud.com:8443",
     "vllm_port": 0
   }
   ```

4. **请求转发**：Router从Manager获取路由信息后，使用公网URL转发请求
   ```
   Client → Router → Manager (获取路由) → Router → Worker (公网URL) → vLLM Engine
   ```

## 完整示例

### 启动Manager和Router（Ray服务）

```bash
# 启动Ray
ray start --head

# 启动Manager和Router
python start_ray_services.py
```

### 启动Worker（配置端口转发）

```bash
export WORKER_ID="worker-gpu-1"
export WORKER_HOST="0.0.0.0"
export WORKER_PORT="6006"
export MANAGER_URL="http://localhost:18000"
export PUBLIC_URL="https://u557149-9507-6992150f.bjb2.seetacloud.com:8443"

python -m src.worker.service
```

### 注册模型

```bash
curl -X POST http://localhost:18000/v1/models/register \
  -H "Content-Type: application/json" \
  -d '{
    "alias": "qwen-vl-2b",
    "model_name": "Qwen/Qwen3-VL-2B-Instruct",
    "max_model_len": 8192,
    "gpu_memory_utilization": 0.5
  }'
```

### 使用模型

```bash
curl -X POST http://localhost:18000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-vl-2b",
    "messages": [
      {"role": "user", "content": "你好"}
    ]
  }'
```

## 注意事项

1. **内网管理**：如果Manager和Worker在同一内网，可以不配置`PUBLIC_URL`，系统会自动使用`worker_url`

2. **URL格式**：
   - 内网URL: `http://127.0.0.1:6006` 或 `http://0.0.0.0:6006`
   - 公网URL: `https://domain:port` 或 `http://domain:port`

3. **路由路径**：Worker的路由路径保持不变
   - Chat: `{worker_url}/proxy/{alias}/v1/chat/completions`
   - Completions: `{worker_url}/proxy/{alias}/v1/completions`

4. **安全性**：如果使用HTTPS公网URL，确保证书配置正确

## 数据模型变更

### WorkerInfo
```python
@dataclass
class WorkerInfo:
    worker_id: str
    worker_url: str                      # 内网URL
    public_worker_url: Optional[str]     # 公网URL（新增）
    gpu_info: GPUInfo
    last_heartbeat: float
    status: WorkerStatus
    instances: Dict[str, VLLMInstanceInfo]
```

### ModelRouting
```python
@dataclass
class ModelRouting:
    alias: str
    model_name: str
    worker_id: str
    worker_url: str      # 优先使用 public_worker_url（如果有）
    vllm_port: int
    created_at: float
```
