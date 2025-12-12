import uuid
from fastapi import Request
from fastapi.responses import JSONResponse
from ray import logger, serve
import ray

@serve.deployment()
class VadLLMRouter:
    def __init__(self, pool_handle=None):
        # pool_handle 可能尚未准备好，空值时走假数据路径
        self.pool = pool_handle  # Ray actor: PoolManager

    async def chat_completions(self, request: Request):
        body = await request.json()
        model = body["model"]
        messages = body["messages"]

        # 1. 将 messages 拼成一个 prompt（简单策略）
        prompt = self._build_prompt(messages)
        max_tokens = body.get("max_tokens")
        temperature = body.get("temperature")

        text = None

        # 2. 尝试调用 PoolManager 选择 engine；失败时退回假数据
        if self.pool is not None:
            try:
                engine_ref = await self.pool.get_engine_for_request.remote(model)

                # 3. 构造内部请求
                internal_req = {
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }

                # 4. 调用 engine.generate
                output = await engine_ref.generate.remote(internal_req)
                text = output.get("text")
            except Exception as exc:  # noqa: BLE001
                logger.warning("Pool not ready, return fake response: %s", exc)

        if text is None:
            text = self._fake_completion(prompt, model)

        # 5. 组装 Chat Completion 响应
        resp = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": text
                    },
                    "finish_reason": "stop"
                }
            ]
        }
        return JSONResponse(resp)

    async def register_model(self, request: Request):
        """注册新模型到 PoolManager。

        请求体示例：
        {
          "alias": "qwen3-vl-2b",
          "full_name": "Qwen/Qwen3-VL-2B-Instruct"
        }
        """
        if self.pool is None:
            return JSONResponse(
                {"error": "PoolManager not available"},
                status_code=503,
            )

        body = await request.json()
        alias = body.get("alias")
        full_name = body.get("full_name")

        if not alias or not full_name:
            return JSONResponse(
                {"error": "alias and full_name are required"},
                status_code=400,
            )

        try:
            # Ray async actor 调用，返回 register_model 的结果字典
            result = await self.pool.register_model.remote(alias, full_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("register_model failed: %s", exc)
            return JSONResponse(
                {"error": f"register_model failed: {exc}"},
                status_code=500,
            )

        return JSONResponse(result)

    def _build_prompt(self, messages):
        # 简单拼接，可以后续换成更复杂的模板
        parts = []
        for m in messages:
            role = m["role"]
            content = m["content"]
            parts.append(f"{role.upper()}: {content}")
        return "\n".join(parts)

    def _fake_completion(self, prompt: str, model: str) -> str:
        # 提供一段可视化结果，便于前端联调
        snippet = (prompt or "")[:200]
        suffix = "..." if snippet and len(prompt) > len(snippet) else ""
        return f"[fake completion for {model}] {snippet}{suffix}"

    # Ray Serve 入口：映射到 /v1/chat/completions 路由
    async def __call__(self, request: Request):
        # 简易路由匹配
        path = request.url.path
        if path.endswith("/chat/completions"):
            return await self.chat_completions(request)
        if path.endswith("/models/register"):
            return await self.register_model(request)
        else:
            return JSONResponse({"error": "not found"}, status_code=404)
