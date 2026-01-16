# 架构设计文档

## 系统架构

### 整体架构图

```
┌─────────────────────────────────────────────────────────────┐
│                         Client                               │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP (OpenAI API)
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    Router (Port 8000)                        │
│  - OpenAI 兼容 API                                           │
│  - 请求路由                                                   │
│  - 管理接口                                                   │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   Manager (Port 9000)                        │
│  - Worker 注册管理                                           │
│  - 模型路由表                                                 │
│  - 负载均衡决策                                               │
│  - 健康监控                                                   │
└───────┬──────────────────────┬──────────────────────────────┘
        │ HTTP                  │ HTTP
        ▼                       ▼
┌──────────────────┐    ┌──────────────────┐
│ Worker 1 (7001)  │    │ Worker 2 (7002)  │
│ ┌──────────────┐ │    │ ┌──────────────┐ │
│ │ vLLM Manager │ │    │ │ vLLM Manager │ │
│ └──────┬───────┘ │    │ └──────┬───────┘ │
│        │         │    │        │         │
│ ┌──────▼───────┐ │    │ ┌──────▼───────┐ │
│ │ vLLM Server  │ │    │ │ vLLM Server  │ │
│ │ (Port 8001)  │ │    │ │ (Port 8003)  │ │
│ └──────────────┘ │    │ └──────────────┘ │
│ ┌──────────────┐ │    │ ┌──────────────┐ │
│ │ vLLM Server  │ │    │ │ vLLM Server  │ │
│ │ (Port 8002)  │ │    │ │ (Port 8004)  │ │
│ └──────────────┘ │    │ └──────────────┘ │
└──────────────────┘    └──────────────────┘
```

## 核心流程

### 1. Worker 启动和注册流程

```
Worker 启动
    │
    ├─→ 检测 GPU 环境
    │   └─→ 使用 pynvml 获取 GPU 信息
    │
    ├─→ 启动 FastAPI 服务 (Port 7000)
    │
    └─→ 向 Manager 注册
        │
        ├─→ POST /workers/register
        │   {
        │     "worker_id": "worker-1",
        │     "worker_url": "http://worker1:7000",
        │     "gpu_info": {...}
        │   }
        │
        └─→ 启动心跳任务（每 30 秒）
            └─→ POST /workers/{id}/heartbeat
```

### 2. 模型注册和启动流程

```
Client 请求注册模型
    │
    ├─→ POST /admin/models/register
    │   {
    │     "alias": "qwen-vl-2b",
    │     "model_name": "Qwen/...",
    │     "gpu_memory_utilization": 0.9
    │   }
    │
    ▼
Router 转发到 Manager
    │
    ├─→ POST /models/register
    │
    ▼
Manager 处理
    │
    ├─→ 检查 alias 是否已存在
    │
    ├─→ 选择合适的 Worker
    │   └─→ 策略：实例数少 + 显存多
    │
    ├─→ 向 Worker 发送启动请求
    │   └─→ POST /instances/start
    │
    ▼
Worker 启动 vLLM 实例
    │
    ├─→ 查找可用端口（8001+）
    │
    ├─→ 启动 vLLM OpenAI 服务器
    │   └─→ python -m vllm.entrypoints.openai.api_server
    │       --model {model_path}
    │       --port {port}
    │
    ├─→ 等待服务启动（健康检查）
    │
    └─→ 返回实例信息
        │
        ▼
Manager 创建路由
    │
    └─→ 更新 model_routes 表
        {
          "alias": "qwen-vl-2b",
          "worker_id": "worker-1",
          "worker_url": "http://worker1:7000",
          "vllm_port": 8001
        }
```

### 3. 推理请求流程

```
Client 发送请求
    │
    ├─→ POST /v1/chat/completions
    │   {
    │     "model": "qwen-vl-2b",
    │     "messages": [...]
    │   }
    │
    ▼
Router 处理
    │
    ├─→ 从 Manager 查询路由
    │   └─→ GET /models/qwen-vl-2b
    │       返回: {
    │         "worker_url": "http://worker1:7000",
    │         "vllm_port": 8001
    │       }
    │
    ├─→ 构建代理 URL
    │   └─→ http://worker1:7000/proxy/qwen-vl-2b/v1/chat/completions
    │
    ▼
Worker 代理请求
    │
    ├─→ 验证实例存在且状态为 RUNNING
    │
    ├─→ 转发到 vLLM Server
    │   └─→ POST http://localhost:8001/v1/chat/completions
    │
    ├─→ 更新 last_used 时间
    │
    └─→ 返回响应（支持流式）
        │
        ▼
Router 返回给 Client
```

### 4. 健康监控流程

```
Manager 监控任务（每 30 秒）
    │
    ├─→ 检查所有 Worker 的心跳时间
    │
    ├─→ 超过 60 秒无心跳
    │   └─→ 标记 Worker 为 INACTIVE
    │
    └─→ 可选：清理不活跃 Worker 的路由

Worker 心跳任务（每 30 秒）
    │
    ├─→ 收集 GPU 信息
    │
    ├─→ 收集所有实例状态
    │
    └─→ 发送到 Manager
        └─→ POST /workers/{id}/heartbeat
            {
              "gpu_info": {...},
              "instances": {...}
            }
```

## 关键设计决策

### 1. 为什么使用 HTTP 而不是 Ray？

**优点：**
- 更简单的部署模型
- 跨机器通信更容易
- 不依赖 Ray 集群
- 更容易调试和监控
- 支持容器化部署

**缺点：**
- 需要自己实现心跳和健康检查
- 网络开销稍大

### 2. 为什么 Worker 管理 vLLM 进程？

**优点：**
- 动态端口分配
- 进程隔离更好
- 支持独立重启
- 便于资源控制

**实现方式：**
- 使用 `subprocess.Popen` 启动 vLLM server
- 监控进程状态
- 提供代理接口

### 3. 为什么需要 Router？

**优点：**
- 提供统一的 OpenAI 兼容接口
- 隐藏内部架构细节
- 便于添加认证、限流等功能
- 支持多个 Manager 负载均衡

### 4. 路由表设计

```python
model_routes = {
    "alias": ModelRouting(
        alias="qwen-vl-2b",
        model_name="Qwen/...",
        worker_id="worker-1",
        worker_url="http://worker1:7000",
        vllm_port=8001,
        created_at=timestamp
    )
}
```

**特点：**
- alias 作为唯一标识
- 记录完整的路由信息
- 支持快速查询

## 扩展性考虑

### 水平扩展

**Manager：**
- 使用共享存储（Redis/数据库）
- 多个 Manager 实例共享状态
- 通过 etcd 实现分布式锁

**Worker：**
- 无状态设计，易于扩展
- 每台 GPU 机器运行一个 Worker
- 支持动态添加/删除

**Router：**
- 完全无状态
- 可以运行多个实例
- 使用负载均衡器分发请求

### 性能优化

1. **连接池**
   - httpx.AsyncClient 复用
   - 减少连接建立开销

2. **缓存**
   - 路由信息缓存
   - 减少 Manager 查询

3. **异步处理**
   - 所有 I/O 操作异步化
   - 提高并发能力

## 容错设计

### Worker 故障

- Manager 通过心跳检测
- 超时标记为 INACTIVE
- Router 请求失败时重试其他 Worker

### Manager 故障

- Worker 继续运行，定期重试注册
- Router 使用缓存的路由信息
- 恢复后自动同步状态

### vLLM 实例崩溃

- Worker 检测到进程退出
- 更新实例状态为 ERROR
- 可选：自动重启

## 安全考虑

### 认证授权

- Router 层添加 API Key 验证
- Worker 到 Manager 的通信使用 Token
- 使用 HTTPS 加密传输

### 资源隔离

- 每个 vLLM 实例独立进程
- 限制 GPU 显存使用
- 设置请求超时

### 监控审计

- 记录所有 API 调用
- 监控资源使用情况
- 异常告警

## 未来改进

1. **动态伸缩**
   - 根据负载自动启动/停止实例
   - 实现模型休眠唤醒

2. **智能调度**
   - 考虑 GPU 负载
   - 考虑网络延迟
   - 实现亲和性调度

3. **高级功能**
   - 请求队列
   - 批处理优化
   - 模型预热
   - A/B 测试

4. **可观测性**
   - Prometheus 指标
   - 分布式追踪
   - 日志聚合
