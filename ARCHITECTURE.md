# model_pool_serve 架构说明（现状）

本文描述当前模型池（model_pool_serve）的实际架构与关键流程，作为后续改进讨论的基础。内容以代码现状为准，不包含改进方案。

## 1. 组件与职责

### 1.1 Serve（Router + Manager，Ray Serve）
代码入口：`model_pool_serve/src/serve/service.py`

- 通过 Ray Serve + FastAPI 对外提供 **OpenAI 兼容 API** 和 **管理 API**。
- 维护 Worker 注册表、模型配置、实例路由表（内存态）。
- 选择实例并将请求转发到 Worker（HTTP 代理）。
- 内置健康检查循环：定期访问 Worker `/health`，失败达到阈值后自动注销。
- 内置负载路由策略与自动扩缩容逻辑（见 `LoadBasedAutoscaler`）。

### 1.2 Worker（GPU 执行面）
代码入口：`model_pool_serve/src/worker/service.py`

- 在 GPU 机器上常驻运行。
- 管理 vLLM 实例生命周期（启动、停止、睡眠、唤醒）。
- 通过 `/proxy/{alias}/v1/...` 转发 OpenAI 请求到本地 vLLM 引擎。
- 启动时向 Serve 注册；不再保留心跳机制。

### 1.3 vLLM Manager（进程内引擎管理）
代码入口：`model_pool_serve/src/worker/vllm_manager.py`

- 进程内管理 `AsyncLLMEngine`（不再启动独立 vLLM server 进程）。
- 支持多级睡眠：ACTIVE / SLEEP_1 / SLEEP_2 / UNLOADED。
- 维护请求数、TTFT/E2E 延迟统计。
- 支持 Fake 模型，用于调度/负载测试。

## 2. 逻辑视图与数据流

### 2.1 推理请求路径（OpenAI 兼容）

```
Client (OpenAI SDK / HTTP)
  -> Serve (Ray Serve / FastAPI)
      -> Worker /proxy/{instance_alias}/v1/chat/completions
          -> VLLMManager (AsyncLLMEngine.generate)
```

要点：
- `model` 字段是 **模型 alias**（如 `qwen3-vl-2b`）。
- Serve 选择一个 **实例 alias**（如 `qwen3-vl-2b-r2`）后转发。
- Worker 在转发前会确保实例被唤醒（包括 UNLOADED）。

### 2.2 管理路径（模型/Worker）

```
Worker startup
  -> POST /workers/register (Serve)

Admin register model
  -> POST /admin/models/register (Serve)
      -> POST /instances/start (Worker)
      -> 更新模型路由表 (Serve 内存态)
```

要点：
- 模型注册会触发 Worker 启动实例。
- Serve 将实例别名写入路由表；不做持久化。

### 2.3 自动扩缩容与睡眠

```
Serve Autoscaler
  -> 周期性拉取 /instances/{alias}/status
  -> load = inflight / capacity
  -> 按阈值执行：
     - scale up: wake / start new instance
     - scale down: sleep_1 -> sleep_2 -> unloaded

Worker Sleep Monitor
  -> 实例空闲时间触发降级
```

要点：
- Worker 内部有 **自动睡眠监控**（可配置超时）。
- Serve 内部有 **负载扩缩容逻辑**，但当前版本未在启动时显式调用 `LoadBasedAutoscaler.start(...)`，仅在路由时使用其选择策略（负载刷新与扩缩容循环不会自动运行，除非手动启动）。

## 3. 关键概念与路由

### 3.1 alias 与 instance_alias
- `alias`：对外模型名（请求中的 `model` 字段）。
- `instance_alias`：具体实例名，多个副本时为 `alias-r2`、`alias-r3` 等。

### 3.2 worker_url vs public_worker_url
- `worker_url`：控制面使用的内网地址（用于启动/睡眠/状态检查）。
- `public_worker_url`：对外转发使用的公网地址（用于请求代理）。

### 3.3 负载与容量
- 负载定义：`inflight_requests / capacity`。
- `capacity` 来源：
  - Fake 模型：由 `fake_capacity` 指定。
  - 真实模型：缺省使用全局 `MODEL_INSTANCE_CAPACITY`。

## 4. 主要接口分层

### 4.1 Serve（对外）
- `POST /v1/chat/completions`
- `POST /v1/completions`
- `GET /v1/models`

### 4.2 Serve（管理）
- `POST /admin/models/register`
- `DELETE /admin/models/{alias}`
- `GET /admin/models`
- `GET /admin/workers`
- `GET /admin/status`

### 4.3 Worker（控制面）
- `POST /instances/start`
- `POST /instances/{alias}/stop`
- `GET /instances/{alias}/status`
- `POST /instances/{alias}/sleep`
- `POST /instances/{alias}/wake`

### 4.4 Worker（代理转发）
- `POST /proxy/{alias}/v1/chat/completions`
- `POST /proxy/{alias}/v1/completions`

## 5. 状态与边界（当前实现）

- **状态在内存中**：Serve 的模型/实例/Worker 信息不持久化，服务重启后需重新注册。
- **健康检查替代心跳**：Serve 通过 `/health` 探测 Worker 存活；Worker 不再推心跳。
- **实例负载来自 Worker 状态接口**：`/instances/{alias}/status` 提供 inflight、sleep level、延迟指标。
- **单控制面**：Serve 部署为单一 Ray Serve actor（单点）。
- **显存匹配**：模型注册时按 `gpu_memory_gb` 与 Worker 可用显存匹配设备。
- **Fake 模型**：不加载权重，用于压测调度与扩缩容逻辑。

## 6. 代码索引（关键文件）

- Serve 控制面：`model_pool_serve/src/serve/service.py`
- 自动扩缩容：`model_pool_serve/src/serve/autoscaler.py`
- Worker 服务：`model_pool_serve/src/worker/service.py`
- vLLM 管理器：`model_pool_serve/src/worker/vllm_manager.py`
- 数据模型：`model_pool_serve/src/common/models.py`

