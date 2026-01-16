"""vLLM 实例管理器 - 直接管理 vLLM AsyncLLMEngine 对象

不再启动独立的 HTTP 服务器进程，而是直接在进程内管理 LLMEngine 对象，
避免多一层 HTTP 调用的开销。
"""
import asyncio
import logging
import time
from typing import Dict, Optional, AsyncIterator
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from src.common.models import VLLMInstanceInfo, InstanceStatus

logger = logging.getLogger(__name__)


class VLLMManager:
    """vLLM 引擎管理器 - 直接管理 AsyncLLMEngine 实例"""
    
    def __init__(self):
        # alias -> VLLMInstanceInfo
        self.instances: Dict[str, VLLMInstanceInfo] = {}
        # alias -> AsyncLLMEngine
        self.engines: Dict[str, AsyncLLMEngine] = {}
    
    async def start_instance(
        self,
        alias: str,
        model_name: str,
        model_path: Optional[str] = None,
        gpu_memory_utilization: float = 0.9,
        max_model_len: Optional[int] = None,
        tensor_parallel_size: int = 1
    ) -> VLLMInstanceInfo:
        """启动一个新的 vLLM 引擎实例（直接管理 AsyncLLMEngine）
        
        Args:
            alias: 实例别名
            model_name: 模型名称
            model_path: 模型路径，如果为 None 则使用 model_name
            gpu_memory_utilization: GPU 内存使用率
            max_model_len: 最大模型长度
            tensor_parallel_size: 张量并行大小
            
        Returns:
            VLLMInstanceInfo: 实例信息
        """
        if alias in self.instances:
            logger.warning(f"Instance {alias} already exists")
            return self.instances[alias]
        
        model_path = model_path or model_name
        
        logger.info(f"Starting vLLM engine '{alias}' with model: {model_path}")
        logger.info(f"  GPU memory utilization: {gpu_memory_utilization}")
        logger.info(f"  Tensor parallel size: {tensor_parallel_size}")
        if max_model_len:
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
            # 构建引擎参数
            engine_args = AsyncEngineArgs(
                model=model_path,
                gpu_memory_utilization=gpu_memory_utilization,
                tensor_parallel_size=tensor_parallel_size,
                max_model_len=max_model_len,
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
            raise
    
    async def generate(
        self,
        alias: str,
        prompt: str,
        sampling_params: SamplingParams,
        multi_modal_data: Optional[Dict] = None
    ) -> AsyncIterator:
        """使用指定引擎进行生成（异步生成器）
        
        Args:
            alias: 实例别名
            prompt: 输入文本（已经应用chat template）
            sampling_params: 采样参数
            multi_modal_data: 多模态数据（如图片），格式为 {"image": [PIL.Image, ...]}
            
        Yields:
            生成结果
        """
        if alias not in self.engines:
            raise ValueError(f"Engine {alias} not found")
        
        engine = self.engines[alias]
        
        # 更新最后使用时间
        self.update_last_used(alias)
        
        # 生成唯一的请求 ID
        request_id = f"{alias}-{time.time()}"
        
        # 调用引擎生成（AsyncLLMEngine.generate 返回异步生成器）
        # 如果有多模态数据，需要传递给引擎
        if multi_modal_data:
            async for output in engine.generate(
                {"prompt": prompt, "multi_modal_data": multi_modal_data},
                sampling_params,
                request_id
            ):
                yield output
        else:
            async for output in engine.generate(prompt, sampling_params, request_id):
                yield output
    
    async def get_tokenizer(self, alias: str):
        """获取指定引擎的tokenizer
        
        注意：AsyncLLM.get_tokenizer()是异步方法，需要await
        """
        if alias not in self.engines:
            raise ValueError(f"Engine {alias} not found")
        
        engine = self.engines[alias]
        
        # AsyncLLM.get_tokenizer()是异步方法
        return await engine.get_tokenizer()
    
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
