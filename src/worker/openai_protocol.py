"""OpenAI API 协议适配器

将 vLLM 的输出转换为 OpenAI 兼容的格式
"""
import time
import uuid
from typing import Dict, List, Optional, AsyncIterator
from pydantic import BaseModel
from vllm.sampling_params import SamplingParams


# OpenAI API 请求/响应模型

from typing import Union, List as ListType

class ChatMessage(BaseModel):
    role: str
    content: Union[str, ListType[Dict]]  # 支持字符串或多模态content列表


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None
    # vLLM额外参数
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None
    add_generation_prompt: Optional[bool] = True
    continue_final_message: Optional[bool] = False


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
    top_k: Optional[int] = None,
    repetition_penalty: Optional[float] = None,
    **kwargs
) -> SamplingParams:
    """将 OpenAI 参数转换为 vLLM SamplingParams"""
    params = {
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens or 512,
        "stop": stop,
    }
    
    # 添加可选参数
    if top_k is not None and top_k != -1:
        params["top_k"] = top_k
    if repetition_penalty is not None:
        params["repetition_penalty"] = repetition_penalty
    
    return SamplingParams(**params)


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


def messages_to_prompt(
    messages: List[ChatMessage],
    tokenizer=None,
    add_generation_prompt: bool = True,
    continue_final_message: bool = False
) -> str:
    """将 messages 转换为单个 prompt 字符串
    
    参考 vLLM 的实现，优先使用 tokenizer 的 chat_template
    
    Args:
        messages: 消息列表
        tokenizer: 模型的 tokenizer（可选）
        add_generation_prompt: 是否添加生成提示（默认True）
        continue_final_message: 是否继续最后一条消息（默认False）
    
    Returns:
        格式化后的 prompt
    """
    # 转换为字典格式（支持多模态content）
    conversation = []
    for msg in messages:
        message_dict = {"role": msg.role}
        # 如果content是字符串，直接使用；如果是列表，保持原样
        if isinstance(msg.content, str):
            message_dict["content"] = msg.content
        else:
            # 多模态content，保持列表格式
            message_dict["content"] = msg.content
        conversation.append(message_dict)
    
    # 如果有 tokenizer 且支持 chat_template，使用它
    if tokenizer is not None:
        try:
            # 尝试使用 tokenizer 的 apply_chat_template
            if hasattr(tokenizer, 'apply_chat_template'):
                prompt = tokenizer.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                    continue_final_message=continue_final_message
                )
                return prompt
        except Exception as e:
            # 如果失败，记录警告并使用默认方法
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to apply chat template: {e}, using fallback")
    
    # 默认方法：使用通用的 ChatML 格式（适用于 Qwen 等模型）
    prompt_parts = []
    
    for msg in conversation:
        role = msg["role"]
        content = msg["content"]
        
        # 如果content是列表（多模态），只提取文本部分
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
            content = " ".join(text_parts)
        
        if role == "system":
            prompt_parts.append(f"<|im_start|>system\n{content}<|im_end|>")
        elif role == "user":
            prompt_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
        elif role == "assistant":
            prompt_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
    
    # 根据参数决定是否添加 assistant 开始标记
    if add_generation_prompt and not continue_final_message:
        prompt_parts.append("<|im_start|>assistant\n")
    
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
