"""OpenAI API 协议适配器 - Qwen3-VL 优化版本

将 OpenAI 格式的请求转换为 Qwen3-VL 所需格式
简化了多模态处理逻辑，专注于 Qwen3-VL 模型
"""
import time
import logging
from typing import Dict, List, Optional, AsyncIterator, Union
from pydantic import BaseModel
from vllm.sampling_params import SamplingParams

from .message_adapters import convert_messages_for_backend as _convert_messages_for_backend

logger = logging.getLogger(__name__)


# OpenAI API 请求/响应模型

class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[Dict]]  # 支持字符串或多模态content列表


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    base_model: Optional[str] = None
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    max_tokens: Optional[int] = 1024
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None
    # vLLM额外参数
    top_k: Optional[int] = -1
    repetition_penalty: Optional[float] = None


def convert_to_sampling_params(
    temperature: float = 0.7,
    top_p: float = 1.0,
    max_tokens: Optional[int] = 1024,
    stop: Optional[List[str]] = None,
    top_k: Optional[int] = -1,
    repetition_penalty: Optional[float] = None,
    **kwargs
) -> SamplingParams:
    """将 OpenAI 参数转换为 vLLM SamplingParams"""
    params = {
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens or 1024,
        "stop": stop or [],
        "top_k": top_k,
    }
    
    if repetition_penalty is not None:
        params["repetition_penalty"] = repetition_penalty
    
    return SamplingParams(**params)


async def convert_messages_for_backend(
    messages: List[ChatMessage],
    backend_type: str,
    base_model: Optional[str] = None,
) -> List[Dict]:
    """统一的消息转换入口，根据 backend/base_model 决定处理策略。"""
    return _convert_messages_for_backend(
        messages,
        backend_type,
        base_model=base_model,
    )


def format_chat_completion_response(
    request_id: str,
    model: str,
    text: str,
    finish_reason: str = "stop",
    usage: Optional[Dict] = None
) -> Dict:
    """格式化为 OpenAI chat completion 响应"""
    return {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text
                },
                "finish_reason": finish_reason
            }
        ],
        "usage": usage or {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
    }


def format_chat_completion_chunk(
    request_id: str,
    model: str,
    delta: str,
    finish_reason: Optional[str] = None
) -> str:
    """格式化为 OpenAI chat completion chunk (SSE 格式)"""
    import json
    
    chunk = {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"content": delta} if delta else {},
                "finish_reason": finish_reason
            }
        ]
    }
    
    return f"data: {json.dumps(chunk)}\n\n"


def compute_text_delta(previous_text: str, current_text: Optional[str]) -> tuple[str, str]:
    """Compute incremental delta and updated cumulative text.

    Supports two producer styles:
    1) cumulative text (current_text = full text so far)
    2) delta text (current_text = newly generated tokens)
    """
    if not current_text:
        return "", previous_text

    if previous_text and current_text.startswith(previous_text):
        # Cumulative output.
        delta = current_text[len(previous_text):]
        return delta, current_text

    # Assume delta output.
    return current_text, previous_text + current_text


async def stream_chat_completion(
    engine_output: AsyncIterator,
    request_id: str,
    model: str
) -> AsyncIterator[str]:
    """将 vLLM 流式输出转换为 OpenAI SSE 格式"""
    previous_text = ""
    
    async for output in engine_output:
        for completion_output in output.outputs:
            current_text = completion_output.text
            finish_reason = completion_output.finish_reason
            
            # 只输出增量文本
            delta, previous_text = compute_text_delta(previous_text, current_text)
            if delta:
                yield format_chat_completion_chunk(request_id, model, delta)
            
            if finish_reason:
                yield format_chat_completion_chunk(request_id, model, "", finish_reason)
    
    # 发送 [DONE] 标记
    yield "data: [DONE]\n\n"
