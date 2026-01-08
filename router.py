import uuid
from fastapi import Request
from fastapi.responses import JSONResponse
from ray import logger, serve
from qwen_vl_utils import process_vision_info
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
        max_tokens = body.get("max_tokens")
        temperature = body.get("temperature")

        text = None

        # 2. 尝试调用 PoolManager 选择 engine；失败时退回假数据
        if self.pool is not None:
            try:
                engine_ref = await self.pool.get_engine_for_request.remote(model)

                # 3. 构造内部请求：只透传原始 messages，由后端引擎负责
                # 使用 chat_template 等构造真正的 prompt，避免在 Router
                # 中重复维护 prompt 逻辑。
                internal_req = {
                    "prompt": "",  # 兼容旧字段，不再在这里构造 prompt
                    # 传递原始 messages，交由 LLMEngineWorker 解析
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }

                # 4. 调用 engine.generate
                output = await engine_ref.generate.remote(internal_req)
                text = output.get("text")
            except Exception as exc:  # noqa: BLE001
                logger.warning("Pool not ready, return fake response: %s", exc)

        if text is None:
            text = self._fake_completion(messages, model)

        # 6. 组装 Chat Completion 响应
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

    async def unregister_model(self, request: Request):
        """从 PoolManager 中注销模型并释放资源。

        请求体示例：
        {
          "alias": "qwen3-vl-2b"
        }
        """
        if self.pool is None:
            return JSONResponse(
                {"error": "PoolManager not available"},
                status_code=503,
            )

        body = await request.json()
        alias = body.get("alias")

        if not alias:
            return JSONResponse(
                {"error": "alias is required"},
                status_code=400,
            )

        try:
            result = await self.pool.unregister_model.remote(alias)
        except Exception as exc:  # noqa: BLE001
            logger.warning("unregister_model failed: %s", exc)
            return JSONResponse(
                {"error": f"unregister_model failed: {exc}"},
                status_code=500,
            )

        return JSONResponse(result)

    def _fake_completion(self, messages, model: str) -> str:
        """构造一个简单的假响应，用于 Pool 不可用时的联调。

        这里不再尝试重建模型实际使用的 prompt，只粗略取首条
        user 文本内容，避免与 LLMEngineWorker 的 prompt 逻辑重复。
        """
        first_text = ""
        try:
            for m in messages or []:
                if m.get("role") != "user":
                    continue
                content = m.get("content")
                if isinstance(content, str):
                    first_text = content
                    break
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            txt = block.get("text")
                            if isinstance(txt, str):
                                first_text = txt
                                break
                    if first_text:
                        break
        except Exception:
            first_text = ""

        snippet = (first_text or "")[:200]
        suffix = "..." if snippet and len(first_text) > len(snippet) else ""
        return f"[fake completion for {model}] {snippet}{suffix}"

    # Ray Serve 入口：映射到 /v1/chat/completions 路由
    async def __call__(self, request: Request):
        # 简易路由匹配
        path = request.url.path
        if path.endswith("/chat/completions"):
            return await self.chat_completions(request)
        if path.endswith("/models/register"):
            return await self.register_model(request)
        if path.endswith("/models/unregister"):
            return await self.unregister_model(request)
        else:
            return JSONResponse({"error": "not found"}, status_code=404)
