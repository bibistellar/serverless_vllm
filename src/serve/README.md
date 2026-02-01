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
    "gpu_memory_gb": 12.0,
    "max_model_len": 4096
  }'
```

说明：
- `gpu_memory_gb`：按显存绝对值分配，服务端会根据 worker 总显存换算为 utilization（非 fake 模型必填）

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

## Serve 对模型可用性的判断（当前逻辑）

本节描述当前讨论后的 Serve 行为，用于后续实现细化。

### 1) 状态获取与缓存
- Serve **定时探测** Worker `/instances/{alias}/status` 并缓存实例状态。
- 新请求到来后，先根据缓存选择候选实例，再对候选实例做 **一次即时探测** 以确认可用性。

### 2) 可用性判定
- **可用**：`status == RUNNING` 且 `sleep_level_value == 0`（ACTIVE）。
- **不可用**：`status == STARTING`，或 `sleep_level_value > 0`（SLEEP_1/2/UNLOADED），或实例不可达。

### 3) /v1/chat 的处理流程
- **缓存存在可用实例**：选中候选实例后即时探测。
  - 探测可用：正常转发。
  - 探测不可用：转入“休眠处理逻辑”（不一定立即返回失败）。
- **缓存显示全部休眠**：直接转入“休眠处理逻辑”。

### 4) 休眠处理逻辑（分级）
- **SLEEP_1 / SLEEP_2**：认为可快速唤醒，由 Serve 负责完成唤醒。
  - Serve 触发唤醒，并进行 **短窗口轮询**（例如 1~3 秒）等待 RUNNING/ACTIVE。
  - 短窗口内恢复：正常转发（表现为本次调用就绪时间偏长）。
  - 未恢复：返回“服务准备中，请稍后重试”（HTTP 503）。
- **UNLOADED / STARTING**：认为冷启动耗时较长。
  - Serve 触发唤醒/加载（如有必要），**直接返回**“服务准备中，请稍后重试”（HTTP 503）。

### 5) 错误返回建议
当服务尚不可用时，返回结构示例：
```json
{
  "error": {
    "message": "Model qwen3-vl-2b is starting or sleeping, please retry later.",
    "type": "model_loading",
    "code": "model_not_ready",
    "status": "starting",
    "retry_after": 5
  }
}
```

## 过载表现与弹性调度指标（当前结论）

**过载表现（日常使用）**
- 不一定出现 `model_not_ready` 或错误码。
- 主要表现为 **请求排队、延迟上升**（p50/p95/p99 增大）。

**弹性调度应关注的核心指标**
- **排队/等待**：TTFT、E2E 延迟分位数（p50/p95/p99）。
- **在途请求**：`inflight_requests`。
- **实例容量占用**：`inflight_requests / capacity`（负载）。

**调度判断建议**
- 当延迟分位数持续上升且负载高于阈值时触发扩容。
- 当负载长期低于阈值且延迟稳定时触发缩容/休眠。

## 自动调度原则（当前讨论结论）

**总体原则**
- 请求路径只在 **ACTIVE 列表** 中选择实例转发（ACTIVE = RUNNING + sleep_level=ACTIVE）。
- ACTIVE 列表由 autoscaler 维护，Serve 请求路径不负责唤醒或启动新实例。

**路由策略（当前约定）**
- 先采用 **平均分发**（后续如需优化再讨论）。

**扩容逻辑（以时延为主）**
- autoscaler 定期采样实例 E2E 平均延迟（或滑动平均），只统计 **近一段时间内有请求** 的样本（`SCALE_LATENCY_WINDOW_S`）。
- 每个模型首次就绪后自动发起一次 **基线请求**，测得延迟后将 **扩容阈值** 设置为 `baseline * SCALE_BASELINE_MULTIPLIER`（默认 `2.0`）。
- 若基线请求失败，则使用固定阈值 `SCALE_UP_LATENCY_THRESHOLD` 作为兜底。
- 当平均延迟 **高于阈值** 时触发扩容；若没有近期样本则跳过扩容。
- 每次扩容 **只增加一个实例**，并在扩容后进入冷却时间。
- 扩容顺序：先唤醒 SLEEP_1 → SLEEP_2 → UNLOADED，若仍不足再启动新实例并加入 ACTIVE 列表。
- 支持模型级别的 `min_replicas`，默认使用全局 `min_replicas=1`，可在注册模型时覆盖。

**缩容逻辑（以时延为主）**
- 不看单实例负载，先看整体平均延迟。
- 当平均延迟 **低于阈值**（暂定 `SCALE_DOWN_LATENCY_THRESHOLD = 2s`）时触发缩容。
- 每次缩容 **只移除一个实例**，并在缩容后进入冷却时间。
- autoscaler 从 ACTIVE 列表移除实例（不强制 sleep，由 Worker 自然降级）。

**说明**
- 扩容阈值支持基线自适应，缩容阈值仍为固定值，后续可再结合负载指标。
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
