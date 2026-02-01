# Worker 架构说明（设计稿）

本文用于记录当前讨论的 Worker 架构目标与接口草案，作为后续实现与改进的依据。
不改变“Worker 通过 HTTP 与 Serve 连接”的前提。

## 1. 目标与边界

**目标**
- 管理本机 GPU 上的 vLLM 实例生命周期（start/stop/sleep/wake）。
- 按配置执行睡眠策略（空闲降级，必要时唤醒）。
- 对外提供推理代理与控制面 API。
- 启动时向 Serve 注册本机资源（GPU 信息、访问地址）。

**边界**
- Worker 不负责路由选择与全局扩缩容（由 Serve 负责）。
- Worker 不改变与 Serve 的 HTTP 通信方式。
- Worker 不负责跨机调度或模型池全局状态持久化。
- 模型注册为异步触发：注册请求仅启动动作，调用方需轮询确认可用性。

---

## 2. 逻辑分层

```
HTTP API Layer
  - /health /info
  - /instances/*
  - /proxy/*

Instance Controller
  - 生命周期状态机
  - 并发锁与状态保护
  - ensure_active() 入口

VLLM Manager
  - AsyncLLMEngine 生命周期
  - sleep/wake/unload
  - 统计指标 (TTFT/E2E)

Sleep Monitor
  - idle_time 触发睡眠级别降级
```

---

## 3. 数据结构（Worker 内存态）

- `instances[alias]`：实例元信息（model_name、status、created_at、last_used）
- `engines[alias]`：AsyncLLMEngine 实例
- `sleep_levels[alias]`：当前睡眠等级
- `_inflight_requests[alias]`：当前并发数
- `_latency_metrics[alias]`：TTFT/E2E 指标
- `engine_args[alias]`：重载引擎所需参数

---

## 4. Worker 主要组件（文件列表）

- `model_pool_serve/src/worker/service.py`：Worker HTTP 服务入口（注册、实例控制、代理转发）
- `model_pool_serve/src/worker/vllm_instance.py`：vLLM 实例管理（启动、停止、睡眠、唤醒）
- `model_pool_serve/src/worker/openai_protocol.py`：OpenAI 请求/响应适配与格式转换
- `model_pool_serve/src/common/models.py`：数据模型定义
- `model_pool_serve/src/common/utils.py`：GPU 信息采集与通用工具

---

## 5. 接口总表（唯一维护点）

此处为 **唯一维护点**：后续所有 API 与函数接口更新都在本节同步。

### 5.1 健康与资源

```
GET /health
  -> { status, worker_id, gpu_info, instances_count }

GET /info
  -> { worker_id, gpu_info, instances: { alias: instance_info } }
```

### 5.2 实例控制

```
POST /instances/start
  req: { alias, model_name, model_path?, gpu_memory_utilization?, max_model_len?, tensor_parallel_size?, fake?, fake_response?, fake_delay_ms?, fake_capacity? }
  resp: { status, instance }
  语义: 仅触发启动动作（异步），实例是否可用需轮询 /instances/{alias}/status 确认

POST /instances/{alias}/stop
  resp: { status, message }

POST /instances/{alias}/sleep
  req: { level | level_name }
  resp: { status, level, level_name }

POST /instances/{alias}/wake
  resp: { status, message }

GET /instances/{alias}/status
  resp: { alias, status, sleep_level, inflight_requests, ttft_last, ttft_avg, e2e_last, e2e_avg, capacity, is_fake }
```

### 5.3 推理代理

```
POST /proxy/{alias}/v1/chat/completions
```

**代理流程**
1) 校验实例存在
2) ensure_active(alias)
3) 生成请求（流式或非流式）
4) 更新统计指标

### 5.4 睡眠管理状态机（描述）

- **状态**：ACTIVE -> SLEEP_1 -> SLEEP_2 -> UNLOADED
- **触发**：以实例空闲时间为主触发降级；推理请求触发唤醒
- **约束**：进入睡眠前需确保 `inflight_requests == 0`
- **唤醒**：所有推理入口需先执行唤醒（确保 ACTIVE）


---

## 6. vllm_instance 说明

- vllm_instance 对应 **一个独立的 vLLM 实例进程**（单实例/单进程语义），由 Worker 直接管理。
- Worker 与 vllm_instance **仅通过函数接口交互**，不暴露 HTTP 控制面给 vllm_instance。
- 对 Serve 来说，vllm_instance 是路由与负载的最小单元（instance_alias），但其生命周期完全由 Worker 内部管理。

**vllm_instance 与 Worker 对接的函数接口（概念级）**
```
start_instance
stop_instance
set_sleep_level
wake_instance
get_instance_status
generate
```

---

## 10. 待讨论改进点（占位）

- 实例注册/启动失败的回滚策略
- 状态同步与服务重启后的恢复
- Sleep Monitor 与 Serve Autoscaler 的协作策略
- 统一错误码规范（model_not_ready / worker_busy / resource_insufficient）
