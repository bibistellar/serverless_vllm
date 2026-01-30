# model_pool_serve README

Ray Serve 控制面 + Worker 执行面的 OpenAI 兼容推理服务。

本 README 为唯一权威文档，旧文档已移除。

## 组件与职责

### Serve（Ray Serve 控制面）
- 对外提供 OpenAI 兼容 API 与管理 API（单入口）。
- 维护 Worker 与模型实例映射。
- 依据负载自动扩/缩容（多实例/多副本）。
- 负载策略集中在 `src/serve/autoscaler.py`。

### Worker（执行面）
- 常驻服务，管理本机 vLLM 实例生命周期。
- 支持 sleep/wake/unload。
- 直接在进程内使用 `AsyncLLMEngine`（无独立 vLLM server）。
- `/instances/{alias}/status` 提供负载与延迟指标（TTFT/E2E）。

### 关键特性
- OpenAI 兼容 API（支持流式输出）。
- 单 alias 多实例（alias-r2、alias-r3…）。
- 负载自动扩容/缩容。
- 假模型（fake）用于调度压测。

## 快速开始

### 1) 启动 Serve
```bash
bash scripts/start_serve.sh
```

可选环境变量：
- `SERVE_HOST` / `SERVE_PORT`
- `RAY_ADDRESS` / `RAY_WORKING_DIR`
- `SERVE_MAX_ONGOING_REQUESTS`（默认 100）
- `MODEL_LOAD_HIGH` / `MODEL_LOAD_LOW`
- `MODEL_INSTANCE_CAPACITY`
- `MODEL_MIN_REPLICAS` / `MODEL_MAX_REPLICAS`
- `MODEL_SCALE_INTERVAL`

### 2) 启动 Worker
```bash
export MANAGER_URL=http://127.0.0.1:8000
bash scripts/start_worker.sh
```

### 3) 注册模型
```bash
curl -X POST http://localhost:8000/admin/models/register \
  -H "Content-Type: application/json" \
  -d '{
    "alias": "qwen3-vl-2b",
    "model_name": "Qwen/Qwen3-VL-2B-Instruct",
    "gpu_memory_utilization": 0.6,
    "max_model_len": 2048
  }'
```

### 4) 推理请求（OpenAI 兼容）
```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-vl-2b",
    "messages": [{"role": "user", "content": "你好"}]
  }'
```

## OpenAI SDK 用法
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000", api_key="dummy")
resp = client.chat.completions.create(
    model="qwen3-vl-2b",
    messages=[{"role": "user", "content": "你好"}],
)
print(resp.choices[0].message.content)
```

## 自动扩缩策略

策略集中在 `src/serve/autoscaler.py`：
- 平均负载 > 0.7：优先唤醒休眠实例，否则启动新实例。
- 平均负载 < 0.3：按 ACTIVE → SLEEP_1 → SLEEP_2 → UNLOADED 逐步休眠。
- 负载 = `inflight_requests / capacity`（capacity 可配置或从假模型返回）。

## 假模型（调度压测）

注册 fake 模型：
```bash
curl -X POST http://localhost:8000/admin/models/register \
  -H "Content-Type: application/json" \
  -d '{
    "alias": "fake-model",
    "model_name": "__fake__",
    "fake": true,
    "fake_response": "FAKE_OK",
    "fake_delay_ms": 3000,
    "fake_capacity": 1
  }'
```

该模型不加载权重，按延迟/并发模拟负载。

## API 说明（简版）

### Serve（对外）
- `GET /health`
- `POST /v1/chat/completions`（支持 `stream`）
- `POST /v1/completions`
- `GET /v1/models`
- `POST /admin/models/register`
- `DELETE /admin/models/{alias}`
- `GET /admin/models`
- `GET /admin/workers`
- `GET /admin/status`
- `POST /workers/register`
- `DELETE /workers/{worker_id}/unregister`

### Worker（内部）
- `GET /health`
- `GET /info`
- `POST /instances/start`
- `POST /instances/{alias}/stop`
- `GET /instances`
- `GET /instances/{alias}`
- `GET /instances/{alias}/status`
- `POST /instances/{alias}/sleep`
- `POST /instances/{alias}/wake`
- `POST /proxy/{alias}/v1/chat/completions`
- `POST /proxy/{alias}/v1/completions`

### `/instances/{alias}/status` 返回字段
- `sleep_level_value`
- `inflight_requests`
- `ttft_last` / `ttft_avg`
- `e2e_last` / `e2e_avg`
- `gpu_info`（含 `available_memory_gb`、`utilization`）

## 测试

基础测试：
```bash
python tests/test_system.py http://localhost:8000 http://localhost:7000
```

完整测试（包含假模型扩容）：
```bash
RUN_FULL_TESTS=1 python tests/test_system.py http://localhost:8000 http://localhost:7000
```

真模型扩容测试：
```bash
RUN_FULL_TESTS=1 RUN_REAL_AUTOSCALE_TEST=1 python tests/test_system.py http://localhost:8000 http://localhost:7000
```
POST /workers/register
Content-Type: application/json

{
  "worker_id": "worker-1",
  "worker_url": "http://worker1:7000",
  "gpu_info": {...}
}
```

**Worker 心跳**
```bash
POST /workers/{worker_id}/heartbeat
```

## 项目结构

```
serverless_vllm/
├── src/
│   ├── common/          # 公共模块
│   │   ├── models.py    # 数据模型
│   │   └── utils.py     # 工具函数
│   ├── manager/         # Manager 服务
│   │   └── service.py
│   ├── worker/          # Worker 服务
│   │   ├── vllm_manager.py  # vLLM 实例管理
│   │   └── service.py
│   └── router/          # Router 服务
│       └── service.py
├── config/              # 配置文件
│   ├── models.yaml
│   └── docker-compose.yaml
├── scripts/             # 启动脚本
│   ├── start_all.sh
│   ├── start_manager.sh
│   ├── start_worker.sh
│   └── start_router.sh
├── docs/                # 文档
├── tests/               # 测试
└── requirements.txt
```

## 部署建议

### 单机多 GPU
- 1 个 Manager
- 1 个 Worker（管理所有 GPU）
- 1 个 Router

### 多机分布式
- 1 个 Manager（可独立机器或与 Router 同机）
- 每台 GPU 机器 1 个 Worker
- 1 个 Router（统一入口）

### 高可用
- Manager: 使用负载均衡 + 共享存储
- Worker: 自动重启 + 健康检查
- Router: 多实例 + 负载均衡

## 环境变量

### Manager
- `MANAGER_HOST`: 监听地址，默认 0.0.0.0
- `MANAGER_PORT`: 监听端口，默认 9000

### Worker
- `WORKER_ID`: Worker 唯一标识
- `WORKER_HOST`: 监听地址，默认 0.0.0.0
- `WORKER_PORT`: 监听端口，默认 7000
- `MANAGER_URL`: Manager 地址

### Router
- `ROUTER_HOST`: 监听地址，默认 0.0.0.0
- `ROUTER_PORT`: 监听端口，默认 8000
- `MANAGER_URL`: Manager 地址

## 故障排查

### Worker 无法注册
1. 检查 MANAGER_URL 是否正确
2. 检查网络连通性
3. 查看 Manager 日志

### 模型启动失败
1. 检查 GPU 显存是否充足
2. 检查模型路径是否正确
3. 查看 Worker 日志

### 推理请求超时
1. 检查模型状态（是否 RUNNING）
2. 检查 Worker 健康状态
3. 增加超时时间

## 开发计划

- [ ] 实现模型自动休眠唤醒
- [ ] 支持多副本负载均衡
- [ ] 添加请求队列和限流
- [ ] 实现模型预热
- [ ] 添加监控和指标收集
- [ ] 支持更多模型格式

## License

MIT
