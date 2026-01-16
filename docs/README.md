# Serverless vLLM - 分布式 LLM 服务池

基于 HTTP 通信的解耦架构，支持多机 GPU 资源池化和动态路由。

## 架构概述

### 组件

1. **Manager (中央管理器)** - 端口 9000
   - 接收 Worker 注册和心跳
   - 维护 Worker 和模型路由表
   - 决策模型部署位置
   - 监控 Worker 健康状态

2. **Worker (工作节点)** - 端口 7000+
   - 检测本机 GPU 环境
   - 管理 vLLM 实例生命周期
   - 提供动态路由代理
   - 定期向 Manager 发送心跳

3. **Router (API 网关)** - 端口 8000
   - 提供 OpenAI 兼容 API
   - 路由请求到对应的 Worker
   - 管理接口（注册/注销模型）

### 通信方式

- Manager ↔ Worker: HTTP (注册、心跳、指令)
- Router ↔ Manager: HTTP (查询路由)
- Router ↔ Worker: HTTP (转发推理请求)
- Worker ↔ vLLM: HTTP (本地 proxy)

### 关键特性

- ✅ 完全解耦，无 Ray 依赖
- ✅ HTTP 通信，易于跨机部署
- ✅ 动态启动 vLLM server
- ✅ 随机端口 + alias 路由
- ✅ 自动心跳和健康检查
- ✅ OpenAI 兼容 API
- ✅ 多 Worker 负载均衡

## 快速开始

### 本地开发模式

1. **安装依赖**
```bash
pip install -r requirements.txt
```

2. **启动 Manager**
```bash
bash scripts/start_manager.sh
```

3. **启动 Worker（新终端）**
```bash
export WORKER_ID=worker-1
export MANAGER_URL=http://localhost:9000
bash scripts/start_worker.sh
```

4. **启动 Router（新终端）**
```bash
export MANAGER_URL=http://localhost:9000
bash scripts/start_router.sh
```

5. **注册模型**
```bash
curl -X POST http://localhost:8000/admin/models/register \
  -H "Content-Type: application/json" \
  -d '{
    "alias": "qwen-vl-2b",
    "model_name": "Qwen/Qwen3-VL-2B-Instruct",
    "gpu_memory_utilization": 0.9,
    "max_model_len": 8192
  }'
```

6. **发送推理请求**
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-vl-2b",
    "messages": [
      {"role": "user", "content": "你好"}
    ]
  }'
```

### Docker 部署模式

1. **构建镜像**
```bash
docker-compose -f config/docker-compose.yaml build
```

2. **启动服务**
```bash
docker-compose -f config/docker-compose.yaml up -d
```

3. **查看日志**
```bash
docker-compose -f config/docker-compose.yaml logs -f
```

## API 文档

### OpenAI 兼容 API

**Chat Completions**
```bash
POST /v1/chat/completions
Content-Type: application/json

{
  "model": "qwen-vl-2b",
  "messages": [
    {"role": "user", "content": "Hello"}
  ],
  "max_tokens": 128,
  "temperature": 0.7
}
```

**List Models**
```bash
GET /v1/models
```

### 管理 API

**注册模型**
```bash
POST /admin/models/register
Content-Type: application/json

{
  "alias": "model-name",
  "model_name": "path/to/model",
  "worker_id": "worker-1"  # 可选
}
```

**注销模型**
```bash
DELETE /admin/models/{alias}
```

**列出 Workers**
```bash
GET /admin/workers
```

**系统状态**
```bash
GET /admin/status
```

### Worker API

**健康检查**
```bash
GET /health
```

**Worker 信息**
```bash
GET /info
```

**启动实例**
```bash
POST /instances/start
Content-Type: application/json

{
  "alias": "model-1",
  "model_name": "Qwen/Qwen3-VL-2B-Instruct"
}
```

**停止实例**
```bash
POST /instances/{alias}/stop
```

**代理请求**
```bash
POST /proxy/{alias}/v1/chat/completions
```

### Manager API

**Worker 注册**
```bash
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
