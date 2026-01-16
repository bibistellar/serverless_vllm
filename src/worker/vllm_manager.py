"""vLLM 实例管理器 - 直接管理 vLLM AsyncLLMEngine 对象

不再启动独立的 HTTP 服务器进程，而是直接在进程内管理 LLMEngine 对象，
避免多一层 HTTP 调用的开销。

专门针对 Qwen3-VL 多模态模型优化。
"""
import asyncio
import logging
import time
import os
import torch
from typing import Dict, Optional, AsyncIterator
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.inputs.data import TextPrompt
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from src.common.models import VLLMInstanceInfo, InstanceStatus

logger = logging.getLogger(__name__)


class VLLMManager:
    """vLLM 引擎管理器 - 直接管理 AsyncLLMEngine 实例（针对 Qwen3-VL 优化）"""
    
    def __init__(self):
        # alias -> VLLMInstanceInfo
        self.instances: Dict[str, VLLMInstanceInfo] = {}
        # alias -> AsyncLLMEngine
        self.engines: Dict[str, AsyncLLMEngine] = {}
        # alias -> AutoProcessor (用于 Qwen3-VL 多模态处理)
        self.processors: Dict[str, AutoProcessor] = {}
    
    async def start_instance(
        self,
        alias: str,
        model_name: str,
        model_path: Optional[str] = None,
        gpu_memory_utilization: float = 0.75,
        max_model_len: Optional[int] = 4096,
        tensor_parallel_size: Optional[int] = None
    ) -> VLLMInstanceInfo:
        """启动一个新的 vLLM 引擎实例（Qwen3-VL 优化版本）
        
        Args:
            alias: 实例别名
            model_name: 模型名称（如 Qwen/Qwen3-VL-2B-Instruct）
            model_path: 模型路径，如果为 None 则使用 model_name
            gpu_memory_utilization: GPU 内存使用率
            max_model_len: 最大模型长度
            tensor_parallel_size: 张量并行大小（默认使用所有 GPU）
            
        Returns:
            VLLMInstanceInfo: 实例信息
        """
        if alias in self.instances:
            logger.warning(f"Instance {alias} already exists")
            return self.instances[alias]
        
        # 处理模型路径
        if model_path:
            model_path = os.path.expanduser(model_path)
            if not os.path.isdir(model_path):
                logger.warning(f"Model path {model_path} not found, using model_name: {model_name}")
                model_path = model_name
        else:
            model_path = model_name
        
        # 自动检测 tensor_parallel_size
        if tensor_parallel_size is None:
            tensor_parallel_size = torch.cuda.device_count()
            logger.info(f"Auto-detected tensor_parallel_size: {tensor_parallel_size}")
        
        logger.info(f"Starting vLLM engine '{alias}' with model: {model_path}")
        logger.info(f"  GPU memory utilization: {gpu_memory_utilization}")
        logger.info(f"  Tensor parallel size: {tensor_parallel_size}")
        logger.info(f"  Max model length: {max_model_len}")
        
        # 创建实例信息（先标记为 STARTING）
        instance = VLLMInstanceInfo(
            alias=alias,
            model_name=model_name,
            port=0,  # 不需要端口，直接调用
            status=InstanceStatus.STARTING,
            created_at=time.time(),
            base_url="",  # 不需要 URL
            pid=0  # 不需要独立进程
        )
        
        self.instances[alias] = instance
        
        try:
            # 加载 AutoProcessor (用于 Qwen3-VL 多模态处理)
            logger.info(f"Loading AutoProcessor for '{alias}'...")
            processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            self.processors[alias] = processor
            logger.info(f"✅ AutoProcessor loaded for '{alias}'")
            
            # 构建引擎参数（针对 Qwen3-VL 优化）
            engine_args = AsyncEngineArgs(
                model=model_path,
                mm_encoder_tp_mode="data",  # Qwen3-VL 推荐设置
                enable_expert_parallel=False,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                seed=0,
                trust_remote_code=True,
            )
            
            logger.info(f"Creating AsyncLLMEngine for '{alias}'...")
            start_time = time.time()
            
            # 创建引擎（这是一个异步操作，可能需要较长时间）
            engine = AsyncLLMEngine.from_engine_args(engine_args)
            
            self.engines[alias] = engine
            instance.status = InstanceStatus.RUNNING
            
            elapsed = time.time() - start_time
            logger.info(f"✅ vLLM engine '{alias}' is READY! (took {elapsed:.1f}s)")
            
            return instance
            
        except Exception as e:
            logger.error(f"❌ Failed to start vLLM engine '{alias}': {e}")
            instance.status = InstanceStatus.ERROR
            # 清理已加载的资源
            self.processors.pop(alias, None)
            raise
    
    async def generate(
        self,
        alias: str,
        messages: list,
        sampling_params: SamplingParams
    ) -> AsyncIterator:
        """使用指定引擎进行生成（异步生成器，针对 Qwen3-VL 优化）
        
        Args:
            alias: 实例别名
            messages: OpenAI 格式的消息列表
            sampling_params: 采样参数
            
        Yields:
            生成结果
        """
        if alias not in self.engines:
            raise ValueError(f"Engine {alias} not found")
        
        if alias not in self.processors:
            raise ValueError(f"Processor {alias} not found")
        
        engine = self.engines[alias]
        processor = self.processors[alias]
        
        # 更新最后使用时间
        self.update_last_used(alias)
        
        # 使用 processor 和 qwen_vl_utils 处理消息
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 使用 qwen_vl_utils 处理多模态信息
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            image_patch_size=processor.image_processor.patch_size,
            return_video_kwargs=True,
            return_video_metadata=True
        )
        
        # 构建多模态数据
        mm_data = {}
        if image_inputs is not None:
            mm_data['image'] = image_inputs
            logger.info(f"Processing {len(image_inputs)} image(s)")
        if video_inputs is not None:
            mm_data['video'] = video_inputs
            logger.info(f"Processing video input")
        
        # 生成唯一的请求 ID
        request_id = f"{alias}-{time.time()}"
        
        # 构建输入：使用 TextPrompt 数据结构
        if mm_data:
            # 多模态输入
            inputs = TextPrompt(
                prompt=text,
                multi_modal_data=mm_data,
                mm_processor_kwargs=video_kwargs
            )
        else:
            # 纯文本输入
            inputs = text
        
        # 调用引擎生成（AsyncLLMEngine.generate 返回异步生成器）
        async for output in engine.generate(inputs, sampling_params, request_id):
            yield output
    
    async def get_processor(self, alias: str):
        """获取指定引擎的 processor"""
        if alias not in self.processors:
            raise ValueError(f"Processor {alias} not found")
        return self.processors[alias]
    
    async def get_tokenizer(self, alias: str):
        """获取指定引擎的 tokenizer（已弃用，使用 processor 替代）"""
        # 为了向后兼容保留此方法
        if alias not in self.processors:
            raise ValueError(f"Processor {alias} not found")
        return self.processors[alias].tokenizer
    
    async def stop_instance(self, alias: str) -> bool:
        """停止 vLLM 引擎实例"""
        if alias not in self.instances:
            logger.warning(f"Instance {alias} not found")
            return False
        
        # 移除引擎
        engine = self.engines.pop(alias, None)
        if engine:
            # AsyncLLMEngine 没有显式的关闭方法，由 GC 处理
            logger.info(f"vLLM engine {alias} removed")
        
        # 移除 processor
        self.processors.pop(alias, None)
        
        # 清理实例信息
        self.instances.pop(alias, None)
        
        return True
    
    def get_instance(self, alias: str) -> Optional[VLLMInstanceInfo]:
        """获取实例信息"""
        return self.instances.get(alias)
    
    def list_instances(self) -> Dict[str, VLLMInstanceInfo]:
        """列出所有实例"""
        return self.instances.copy()
    
    def update_last_used(self, alias: str):
        """更新实例最后使用时间"""
        instance = self.instances.get(alias)
        if instance:
            instance.last_used = time.time()
