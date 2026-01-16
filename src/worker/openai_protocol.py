"""OpenAI API 协议适配器 - Qwen3-VL 优化版本

将 OpenAI 格式的请求转换为 Qwen3-VL 所需格式
简化了多模态处理逻辑，专注于 Qwen3-VL 模型
"""
import time
import uuid
import base64
import io
import logging
from typing import Dict, List, Optional, AsyncIterator, Union
from pydantic import BaseModel
from vllm.sampling_params import SamplingParams
from PIL import Image

logger = logging.getLogger(__name__)


# OpenAI API 请求/响应模型

class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[Dict]]  # 支持字符串或多模态content列表


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    max_tokens: Optional[int] = 1024
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None
    # vLLM额外参数
    top_k: Optional[int] = -1
    repetition_penalty: Optional[float] = None


class CompletionRequest(BaseModel):
    """简化版本的 Completion 请求（主要使用 Chat Completion）"""
    model: str
    prompt: str
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = 1024
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None


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


async def process_messages_to_qwen_format(messages: List[ChatMessage]) -> List[Dict]:
    """将 OpenAI 格式的消息转换为 Qwen3-VL 格式
    
    处理多模态 content，支持文本和图像（base64 编码）
    
    Args:
        messages: OpenAI 格式的消息列表
        
    Returns:
        Qwen3 格式的消息列表
    """
    qwen_messages = []
    
    for msg in messages:
        message_dict = {"role": msg.role}
        
        # 处理 content
        if isinstance(msg.content, str):
            # 纯文本消息
            message_dict["content"] = msg.content
        else:
            # 多模态消息
            content_list = []
            for item in msg.content:
                if not isinstance(item, dict):
                    continue
                
                item_type = item.get("type")
                
                if item_type == "text":
                    content_list.append({
                        "type": "text",
                        "text": item.get("text", "")
                    })
                elif item_type == "image_url":
                    # 处理图像 URL
                    image_url_data = item.get("image_url", {})
                    if isinstance(image_url_data, dict):
                        url = image_url_data.get("url")
                    else:
                        url = image_url_data
                    
                    if url:
                        # 支持 base64 编码的图像
                        if url.startswith("data:image"):
                            # 提取 base64 数据
                            header, encoded = url.split(",", 1)
                            image_bytes = base64.b64decode(encoded)
                            image = Image.open(io.BytesIO(image_bytes))
                            
                            content_list.append({
                                "type": "image",
                                "image": image
                            })
                        elif url.startswith("http"):
                            # 支持 URL (Qwen3-VL 会自动下载)
                            content_list.append({
                                "type": "image",
                                "image": url
                            })
                        else:
                            # 本地文件路径
                            content_list.append({
                                "type": "image",
                                "image": url
                            })
            
            message_dict["content"] = content_list
        
        qwen_messages.append(message_dict)
    
    return qwen_messages


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
    """格式化为 OpenAI completion 响应（简化版本）"""
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
            delta = current_text[len(previous_text):]
            if delta:
                yield format_chat_completion_chunk(request_id, model, delta)
            
            previous_text = current_text
            
            if finish_reason:
                yield format_chat_completion_chunk(request_id, model, "", finish_reason)
    
    # 发送 [DONE] 标记
    yield "data: [DONE]\n\n"
