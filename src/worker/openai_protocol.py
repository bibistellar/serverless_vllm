"""OpenAI API 协议适配器

将 vLLM 的输出转换为 OpenAI 兼容的格式
"""
import time
import uuid
from typing import Dict, List, Optional, AsyncIterator
from pydantic import BaseModel
from vllm.sampling_params import SamplingParams


# OpenAI API 请求/响应模型

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None


class CompletionRequest(BaseModel):
    model: str
    prompt: str
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    max_tokens: Optional[int] = 16
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None


def convert_to_sampling_params(
    temperature: float = 0.7,
    top_p: float = 1.0,
    max_tokens: Optional[int] = None,
    stop: Optional[List[str]] = None,
    **kwargs
) -> SamplingParams:
    """将 OpenAI 参数转换为 vLLM SamplingParams"""
    return SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens or 512,
        stop=stop,
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


def format_completion_response(
    request_id: str,
    model: str,
    text: str,
    finish_reason: str = "stop",
    usage: Optional[Dict] = None
) -> Dict:
    """格式化为 OpenAI completion 响应"""
    return {
        "id": f"cmpl-{request_id}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "text": text,
                "index": 0,
                "logprobs": None,
                "finish_reason": finish_reason
            }
        ],
        "usage": usage or {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
    }


def messages_to_prompt(messages: List[ChatMessage]) -> str:
    """将 messages 转换为单个 prompt 字符串
    
    简单实现，实际使用时可能需要根据模型的 chat template 调整
    """
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


async def stream_chat_completion(
    engine_output: AsyncIterator,
    request_id: str,
    model: str
) -> AsyncIterator[str]:
    """将 vLLM 流式输出转换为 OpenAI SSE 格式"""
    async for output in engine_output:
        for completion_output in output.outputs:
            delta = completion_output.text
            finish_reason = completion_output.finish_reason
            
            if delta:
                yield format_chat_completion_chunk(request_id, model, delta)
            
            if finish_reason:
                yield format_chat_completion_chunk(request_id, model, "", finish_reason)
    
    # 发送 [DONE] 标记
    yield "data: [DONE]\n\n"
