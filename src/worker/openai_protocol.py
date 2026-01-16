"""OpenAI API 协议适配器

将 vLLM 的输出转换为 OpenAI 兼容的格式
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
import httpx

logger = logging.getLogger(__name__)


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

async def load_image_from_base64(image_data: str) -> Image.Image:
    """从base64加载图片
    
    Args:
        image_data: base64编码的图片数据（可能包含data URI前缀）
        
    Returns:
        PIL.Image对象
    """
    # 如果是data URI格式，提取base64部分
    if image_data.startswith('data:image'):
        # 格式: data:image/jpeg;base64,/9j/4AAQ...
        header, encoded = image_data.split(',', 1)
        image_bytes = base64.b64decode(encoded)
    else:
        # 直接是base64字符串
        image_bytes = base64.b64decode(image_data)
    
    return Image.open(io.BytesIO(image_bytes))


async def process_multimodal_content(content: Union[str, List[Dict]]) -> tuple[str, Optional[List[Image.Image]]]:
    """处理多模态content，提取文本和图片
    
    Args:
        content: 字符串或包含text和image_url的字典列表
        
    Returns:
        (text, images) 元组
        
    Note:
        只支持base64编码的图片，不支持URL下载
    """
    if isinstance(content, str):
        return content, None
    
    # 处理列表格式的多模态content
    text_parts = []
    images = []
    
    for item in content:
        if not isinstance(item, dict):
            continue
            
        item_type = item.get('type')
        
        if item_type == 'text':
            text_parts.append(item.get('text', ''))
        elif item_type == 'image_url':
            image_url_data = item.get('image_url', {})
            if isinstance(image_url_data, dict):
                url = image_url_data.get('url')
            else:
                url = image_url_data
                
            if url:
                # 只处理base64格式的图片
                if not url.startswith('data:image'):
                    logger.warning(f"Skipping non-base64 image URL. Please encode images as base64.")
                    continue
                    
                try:
                    image = await load_image_from_base64(url)
                    images.append(image)
                except Exception as e:
                    logger.error(f"Failed to load image from base64: {e}")
                    raise ValueError(f"Failed to decode image: {e}")
    
    text = ' '.join(text_parts) if text_parts else ''
    return text, images if images else None

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


async def messages_to_prompt_and_images(
    messages: List[ChatMessage],
    tokenizer=None,
    add_generation_prompt: bool = True,
    continue_final_message: bool = False
) -> tuple[str, Optional[List[Image.Image]]]:
    """将 messages 转换为 prompt 字符串和图片列表
    
    参考 vLLM 的实现，优先使用 tokenizer 的 chat_template
    
    Args:
        messages: 消息列表
        tokenizer: 模型的 tokenizer（可选）
        add_generation_prompt: 是否添加生成提示（默认True）
        continue_final_message: 是否继续最后一条消息（默认False）
    
    Returns:
        (prompt, images) 元组
    """
    # 处理多模态content，提取文本和图片
    all_images = []
    conversation = []
    
    for msg in messages:
        message_dict = {"role": msg.role}
        
        # 处理content（可能是文本或多模态列表）
        if isinstance(msg.content, str):
            message_dict["content"] = msg.content
        else:
            # 多模态content
            text, images = await process_multimodal_content(msg.content)
            message_dict["content"] = text
            if images:
                all_images.extend(images)
        
        conversation.append(message_dict)
    
    # 如果有 tokenizer 且支持 chat_template，使用它
    if tokenizer is not None:
        try:
            if hasattr(tokenizer, 'apply_chat_template'):
                prompt = tokenizer.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                    continue_final_message=continue_final_message
                )
                return prompt, all_images if all_images else None
        except Exception as e:
            logger.warning(f"Failed to apply chat template: {e}, using fallback")
    
    # 默认方法：使用通用的 ChatML 格式（适用于 Qwen 等模型）
    prompt_parts = []
    
    for msg in conversation:
        role = msg["role"]
        content = msg["content"]
        
        if role == "system":
            prompt_parts.append(f"<|im_start|>system\n{content}<|im_end|>")
        elif role == "user":
            prompt_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
        elif role == "assistant":
            prompt_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
    
    # 根据参数决定是否添加 assistant 开始标记
    if add_generation_prompt and not continue_final_message:
        prompt_parts.append("<|im_start|>assistant\n")
    
    prompt = "\n".join(prompt_parts)
    return prompt, all_images if all_images else None


def messages_to_prompt(
    messages: List[ChatMessage],
    tokenizer=None,
    add_generation_prompt: bool = True,
    continue_final_message: bool = False
) -> str:
    """同步版本的messages_to_prompt（不支持图片）
    
    为了向后兼容保留此函数
    """
    # 转换为字典格式（只处理文本）
    conversation = []
    for msg in messages:
        message_dict = {"role": msg.role}
        if isinstance(msg.content, str):
            message_dict["content"] = msg.content
        else:
            # 多模态content，只提取文本部分
            text_parts = []
            for item in msg.content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
            message_dict["content"] = " ".join(text_parts)
        conversation.append(message_dict)
    
    # 如果有 tokenizer 且支持 chat_template，使用它
    if tokenizer is not None:
        try:
            if hasattr(tokenizer, 'apply_chat_template'):
                prompt = tokenizer.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                    continue_final_message=continue_final_message
                )
                return prompt
        except Exception as e:
            logger.warning(f"Failed to apply chat template: {e}, using fallback")
    
    # 默认方法：使用通用的 ChatML 格式
    prompt_parts = []
    for msg in conversation:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            prompt_parts.append(f"<|im_start|>system\n{content}<|im_end|>")
        elif role == "user":
            prompt_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
        elif role == "assistant":
            prompt_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
    
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
