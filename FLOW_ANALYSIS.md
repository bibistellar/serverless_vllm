# 推理流程分析报告

## 问题概述

检查从模型注册到推理是否能正确使用 AsyncLLMEngine 实例。

## 完整流程分析

### 1. 模型注册流程

```bash
curl -X POST http://100.100.238.4:18000/v1/models/register \
  -H "Content-Type: application/json" \
  -d '{
    "alias": "qwen-vl-2b",
    "model_name": "Qwen/Qwen3-VL-2B-Instruct",
    "max_model_len": 8192,
    "gpu_memory_utilization": 0.5
  }'
```

**流程：**
1. 请求到达 Router (端口 18000) → `/v1/models/register` 端点
2. Router 调用 `manager.register_model.remote()` (Ray remote 调用)
3. Manager Actor 执行：
   - 选择可用的 Worker（`_select_worker_for_model()`）
   - 调用 Worker HTTP API：`POST {worker_url}/instances/start`
4. Worker 收到请求：
   - `vllm_manager.start_instance()` 被调用
   - 创建 `AsyncLLMEngine.from_engine_args()` **← 关键步骤**
   - 将 engine 保存到 `self.engines[alias]`
5. Manager 创建路由信息到 `model_routes`

### 2. 推理请求流程

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-vl-2b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

**流程：**
1. 请求到达 Router (端口 8000) → `/v1/chat/completions` 端点
2. Router 调用 `manager.get_model_routing.remote("qwen-vl-2b")`
3. Manager 返回路由信息：`{worker_url, alias}`
4. Router 转发请求到：`{worker_url}/proxy/{alias}/v1/chat/completions`
5. Worker 的 `/proxy/{alias}/v1/chat/completions` 处理：
   - 调用 `vllm_manager.generate(alias, prompt, sampling_params)` **← AsyncLLMEngine 调用**
   - 返回流式或非流式响应

## 潜在问题分析

### ❌ 问题 1：Router 端口不一致

**症状：**
- 注册时用 `http://100.100.238.4:18000`
- 推理时用 `http://localhost:8000`

**原因：**
- Ray Serve 启动在端口 18000
- 端口 8000 可能没有服务监听

**解决：** 两个请求都应该使用 **相同的端口 18000**

```bash
# 正确的推理请求
curl -X POST http://100.100.238.4:18000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-vl-2b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

### ⚠️ 问题 2：Worker 必须先启动并注册

**检查点：**
- Worker 服务是否运行？
- Worker 是否已向 Manager 注册？

**验证命令：**
```bash
# 检查 Worker 列表
curl http://100.100.238.4:18000/workers

# 检查健康状态
curl http://100.100.238.4:18000/health
```

**如果 Worker 未注册：**
- Manager 的 `register_model` 会失败
- 错误信息："No available worker found"

### ✅ 问题 3：AsyncLLMEngine 使用是否正确？

**代码检查结果：✅ 正确**

#### 3.1 Engine 创建 (vllm_manager.py:89-95)
```python
engine_args = AsyncEngineArgs(
    model=model_path,
    gpu_memory_utilization=gpu_memory_utilization,
    tensor_parallel_size=tensor_parallel_size,
    max_model_len=max_model_len,
    trust_remote_code=True,
)
engine = AsyncLLMEngine.from_engine_args(engine_args)
self.engines[alias] = engine
```
✅ **正确：** 使用标准的 vLLM API 创建 AsyncLLMEngine

#### 3.2 生成调用 (vllm_manager.py:117-145)
```python
async def generate(
    self,
    alias: str,
    prompt: str,
    sampling_params: SamplingParams
) -> AsyncIterator:
    engine = self.engines[alias]
    request_id = f"{alias}-{time.time()}"
    
    # 调用引擎生成
    async for output in engine.generate(prompt, sampling_params, request_id):
        yield output
```
✅ **正确：** 使用异步生成器，正确调用 `engine.generate()`

#### 3.3 Worker 服务调用 (service.py:225-255)

**流式响应：**
```python
if request.stream:
    engine_output = self.vllm_manager.generate(alias, prompt, sampling_params)
    return StreamingResponse(
        stream_chat_completion(engine_output, request_id, request.model),
        media_type="text/event-stream"
    )
```
✅ **正确：** 异步生成器传递给流式响应处理

**非流式响应：**
```python
else:
    full_text = ""
    finish_reason = "stop"
    
    async for output in self.vllm_manager.generate(alias, prompt, sampling_params):
        for completion_output in output.outputs:
            full_text = completion_output.text
            if completion_output.finish_reason:
                finish_reason = completion_output.finish_reason
    
    response = format_chat_completion_response(
        request_id=request_id,
        model=request.model,
        text=full_text,
        finish_reason=finish_reason
    )
    return JSONResponse(content=response)
```
✅ **正确：** 正确遍历异步生成器获取输出

### ⚠️ 问题 4：messages_to_prompt 实现简单

**当前实现 (openai_protocol.py:147-165)：**
```python
def messages_to_prompt(messages: List[ChatMessage]) -> str:
    """将 messages 转换为单个 prompt 字符串"""
    prompt_parts = []
    
    for msg in messages:
        role = msg.role
        content = msg.content
        
        if role == "system":
            prompt_parts.append(f"System: {content}")
        elif role == "user":
            prompt_parts.append(f"User: {content}")
        elif role == "assistant":
            prompt_parts.append(f"Assistant: {content}")
    
    prompt_parts.append("Assistant:")
    return "\n".join(prompt_parts)
```

**问题：**
- 这是一个简单的格式化，不使用模型的 chat template
- 对于 Qwen 等模型，应该使用其特定的对话格式

**建议改进：**
```python
def messages_to_prompt(messages: List[ChatMessage], tokenizer=None) -> str:
    """使用模型的 chat template"""
    if tokenizer and hasattr(tokenizer, 'apply_chat_template'):
        return tokenizer.apply_chat_template(
            [{"role": m.role, "content": m.content} for m in messages],
            tokenize=False,
            add_generation_prompt=True
        )
    # 否则使用简单格式
    # ... 当前实现
```

## 测试建议

### 1. 检查服务状态

```bash
# 1. 检查 Ray Serve 状态
curl http://100.100.238.4:18000/health

# 2. 检查 Worker 列表
curl http://100.100.238.4:18000/workers

# 预期输出：至少有一个 active 的 worker
```

### 2. 注册模型（确保使用正确端口）

```bash
curl -X POST http://100.100.238.4:18000/v1/models/register \
  -H "Content-Type: application/json" \
  -d '{
    "alias": "qwen-vl-2b",
    "model_name": "Qwen/Qwen3-VL-2B-Instruct",
    "max_model_len": 8192,
    "gpu_memory_utilization": 0.5
  }'

# 预期输出：{"status": "success", "routing": {...}}
```

### 3. 检查模型状态

```bash
# 列出所有模型
curl http://100.100.238.4:18000/v1/models

# 预期输出：包含 "qwen-vl-2b" 的模型列表
```

### 4. 发送推理请求（使用正确端口）

```bash
# 非流式请求
curl -X POST http://100.100.238.4:18000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-vl-2b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100,
    "stream": false
  }'

# 流式请求
curl -X POST http://100.100.238.4:18000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-vl-2b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100,
    "stream": true
  }'
```

## 结论

### ✅ 能否正确使用 AsyncLLMEngine？

**答案：可以，但需要满足以下条件：**

1. ✅ **代码实现正确**：AsyncLLMEngine 的创建和调用逻辑都正确
2. ⚠️ **端口统一**：注册和推理都需要使用端口 18000（不是 8000）
3. ⚠️ **Worker 必须运行**：确保至少有一个 Worker 注册到 Manager
4. ⚠️ **模型加载时间**：AsyncLLMEngine 创建需要时间（可能几秒到几十秒），在此期间推理请求会返回 503

### 常见错误和排查

| 错误 | 原因 | 解决方案 |
|------|------|----------|
| Connection refused (端口 8000) | Router 监听在 18000 而非 8000 | 使用 18000 端口 |
| Model not found | 模型未注册或路由表中不存在 | 先注册模型 |
| No available worker | Worker 未启动或未注册 | 启动 Worker 服务 |
| Instance not ready (503) | 模型正在加载 | 等待几秒后重试 |
| Worker URL unreachable | Manager 无法访问 Worker HTTP | 检查网络和防火墙 |

## 改进建议

### 1. 使用模型的 chat template
当前 `messages_to_prompt` 过于简单，建议使用 tokenizer 的 chat template。

### 2. 健康检查增强
Worker 可以定期检查 engine 状态，而不是仅在启动时检查。

### 3. 错误处理
在 AsyncLLMEngine 创建失败时，应该有更详细的错误信息。

### 4. 超时处理
推理请求应该有超时限制，避免长时间阻塞。
