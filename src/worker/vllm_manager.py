"""vLLM 实例管理器 - 直接管理 vLLM AsyncLLMEngine 对象

不再启动独立的 HTTP 服务器进程，而是直接在进程内管理 LLMEngine 对象，
避免多一层 HTTP 调用的开销。

专门针对 Qwen3-VL 多模态模型优化。

支持三级缓存结构：
- Level 0 (ACTIVE): 正常运行状态
- Level 1 (SLEEP_1): Offload weights to CPU RAM, discard KV cache
- Level 2 (SLEEP_2): Discard both weights and KV cache
- Level 3 (UNLOADED): 完全释放 AsyncLLM 实例
"""
import asyncio
import logging
import time
import os
import torch
from enum import Enum
from typing import Dict, Optional, AsyncIterator
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.inputs.data import TextPrompt
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from src.common.models import VLLMInstanceInfo, InstanceStatus

logger = logging.getLogger(__name__)


class FakeInstance:
    """Fake model instance for load testing."""

    def __init__(self, response: str, delay_s: float, capacity: int) -> None:
        self.response = response
        self.delay_s = delay_s
        self.capacity = max(1, capacity)
        self.semaphore = asyncio.Semaphore(self.capacity)


class _FakeCompletionOutput:
    def __init__(self, text: str, finish_reason: Optional[str] = None) -> None:
        self.text = text
        self.finish_reason = finish_reason


class _FakeEngineOutput:
    def __init__(self, outputs: list[_FakeCompletionOutput]) -> None:
        self.outputs = outputs


class SleepLevel(Enum):
    """引擎睡眠级别"""
    ACTIVE = 0      # 正常运行
    SLEEP_1 = 1     # 权重卸载到 CPU，清除 KV cache
    SLEEP_2 = 2     # 清除权重和 KV cache
    UNLOADED = 3    # 完全释放引擎


class VLLMManager:
    """vLLM 引擎管理器 - 直接管理 AsyncLLMEngine 实例（针对 Qwen3-VL 优化）
    
    支持三级缓存：
    - 自动超时管理：一定时间无请求后自动降级
    - 手动控制：支持外部控制睡眠级别
    - 智能唤醒：根据当前睡眠级别自动执行唤醒操作
    """
    
    def __init__(
        self,
        enable_auto_sleep: bool = True,
        sleep_1_timeout: int = 300,    # 5分钟后进入 sleep level 1
        sleep_2_timeout: int = 900,    # 15分钟后进入 sleep level 2
        unload_timeout: int = 1800     # 30分钟后完全卸载
    ):
        # alias -> VLLMInstanceInfo
        self.instances: Dict[str, VLLMInstanceInfo] = {}
        # alias -> AsyncLLMEngine
        self.engines: Dict[str, AsyncLLMEngine] = {}
        # alias -> AutoProcessor (用于 Qwen3-VL 多模态处理)
        self.processors: Dict[str, AutoProcessor] = {}
        # alias -> SleepLevel (当前睡眠级别)
        self.sleep_levels: Dict[str, SleepLevel] = {}
        # alias -> 引擎启动参数（用于重新加载）
        self.engine_args: Dict[str, AsyncEngineArgs] = {}
        # alias -> lifecycle lock
        self._lifecycle_locks: Dict[str, asyncio.Lock] = {}
        # alias -> inflight request count
        self._inflight_requests: Dict[str, int] = {}
        # alias -> latency metrics
        self._latency_metrics: Dict[str, Dict] = {}
        # alias -> fake instance
        self.fake_instances: Dict[str, FakeInstance] = {}
        
        # 自动睡眠配置
        self.enable_auto_sleep = enable_auto_sleep
        self.sleep_1_timeout = sleep_1_timeout
        self.sleep_2_timeout = sleep_2_timeout
        self.unload_timeout = unload_timeout
        
        # 后台任务
        self._sleep_monitor_task: Optional[asyncio.Task] = None
        
    async def start_sleep_monitor(self):
        """启动睡眠监控任务"""
        if not self.enable_auto_sleep:
            logger.info("Auto-sleep is disabled")
            return
            
        if self._sleep_monitor_task is not None:
            logger.warning("Sleep monitor already running")
            return
            
        self._sleep_monitor_task = asyncio.create_task(self._monitor_sleep())
        logger.info("Sleep monitor started")
    
    async def stop_sleep_monitor(self):
        """停止睡眠监控任务"""
        if self._sleep_monitor_task:
            self._sleep_monitor_task.cancel()
            try:
                await self._sleep_monitor_task
            except asyncio.CancelledError:
                pass
            self._sleep_monitor_task = None
            logger.info("Sleep monitor stopped")
    
    async def _monitor_sleep(self):
        """监控引擎使用情况，自动降级到睡眠模式"""
        logger.info("Sleep monitor loop started")
        
        while True:
            try:
                await asyncio.sleep(60)  # 每分钟检查一次
                
                current_time = time.time()
                
                for alias, instance in list(self.instances.items()):
                    if instance.status != InstanceStatus.RUNNING:
                        continue
                    
                    idle_time = current_time - instance.last_used
                    current_level = self.sleep_levels.get(alias, SleepLevel.ACTIVE)
                    
                    # 根据空闲时间决定睡眠级别
                    if idle_time >= self.unload_timeout and current_level != SleepLevel.UNLOADED:
                        logger.info(f"Instance {alias} idle for {idle_time:.0f}s, unloading...")
                        await self.set_sleep_level(alias, SleepLevel.UNLOADED)
                    elif idle_time >= self.sleep_2_timeout and current_level == SleepLevel.SLEEP_1:
                        logger.info(f"Instance {alias} idle for {idle_time:.0f}s, entering sleep level 2...")
                        await self.set_sleep_level(alias, SleepLevel.SLEEP_2)
                    elif idle_time >= self.sleep_1_timeout and current_level == SleepLevel.ACTIVE:
                        logger.info(f"Instance {alias} idle for {idle_time:.0f}s, entering sleep level 1...")
                        await self.set_sleep_level(alias, SleepLevel.SLEEP_1)
                        
            except asyncio.CancelledError:
                logger.info("Sleep monitor cancelled")
                break
            except Exception as e:
                logger.error(f"Error in sleep monitor: {e}")
                import traceback
                traceback.print_exc()
    
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

        if alias in self.fake_instances:
            logger.warning(f"Fake instance {alias} already exists")
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
        self._lifecycle_locks[alias] = asyncio.Lock()
        self._inflight_requests[alias] = 0
        self._latency_metrics[alias] = {
            "count": 0,
            "ttft_last": None,
            "ttft_avg": None,
            "e2e_last": None,
            "e2e_avg": None,
        }
        
        try:
            def _build_engine_sync():
                logger.info(f"Loading AutoProcessor for '{alias}'...")
                processor = AutoProcessor.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                )
                logger.info(f"✅ AutoProcessor loaded for '{alias}'")

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
                engine = AsyncLLMEngine.from_engine_args(engine_args)
                return processor, engine, engine_args

            start_time = time.time()
            processor, engine, engine_args = await asyncio.to_thread(_build_engine_sync)

            self.processors[alias] = processor
            self.engines[alias] = engine
            self.engine_args[alias] = engine_args  # 保存引擎参数供重新加载使用
            self.sleep_levels[alias] = SleepLevel.ACTIVE  # 初始为激活状态
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

    async def start_fake_instance(
        self,
        alias: str,
        response: str = "OK",
        delay_s: float = 0.5,
        capacity: int = 1,
    ) -> VLLMInstanceInfo:
        """启动一个假模型实例，用于调度测试"""
        if alias in self.instances:
            logger.warning(f"Instance {alias} already exists")
            return self.instances[alias]

        fake = FakeInstance(response=response, delay_s=delay_s, capacity=capacity)
        self.fake_instances[alias] = fake

        instance = VLLMInstanceInfo(
            alias=alias,
            model_name="__fake__",
            port=0,
            status=InstanceStatus.RUNNING,
            created_at=time.time(),
            base_url="",
            pid=0,
        )
        self.instances[alias] = instance
        self.sleep_levels[alias] = SleepLevel.ACTIVE
        self._lifecycle_locks[alias] = asyncio.Lock()
        self._inflight_requests[alias] = 0
        self._latency_metrics[alias] = {
            "count": 0,
            "ttft_last": None,
            "ttft_avg": None,
            "e2e_last": None,
            "e2e_avg": None,
        }

        logger.info(f"✅ Fake instance '{alias}' ready (delay={delay_s}s, capacity={capacity})")
        return instance
    
    async def set_sleep_level(self, alias: str, level: SleepLevel) -> bool:
        """设置引擎的睡眠级别
        
        Args:
            alias: 实例别名
            level: 目标睡眠级别
            
        Returns:
            是否成功
        """
        if alias not in self.instances:
            logger.warning(f"Instance {alias} not found")
            return False

        if alias in self.fake_instances:
            self.sleep_levels[alias] = level
            instance = self.instances.get(alias)
            if instance:
                instance.status = InstanceStatus.RUNNING if level == SleepLevel.ACTIVE else InstanceStatus.SLEEPING
            return True
        
        if level != SleepLevel.ACTIVE:
            inflight = self._inflight_requests.get(alias, 0)
            if inflight > 0:
                logger.warning(
                    f"Cannot change {alias} to {level.name} while {inflight} request(s) in flight"
                )
                return False

        current_level = self.sleep_levels.get(alias, SleepLevel.ACTIVE)
        if alias not in self._lifecycle_locks:
            self._lifecycle_locks[alias] = asyncio.Lock()
        
        if current_level == level:
            logger.info(f"Instance {alias} already at sleep level {level.name}")
            return True
        
        logger.info(f"Setting {alias} from {current_level.name} to {level.name}")
        
        try:
            instance = self.instances.get(alias)
            if instance and level != SleepLevel.ACTIVE:
                instance.status = InstanceStatus.SLEEPING
            async with self._lifecycle_locks[alias]:
                if level == SleepLevel.ACTIVE:
                    if instance and current_level == SleepLevel.UNLOADED:
                        instance.status = InstanceStatus.STARTING
                    # 从任何睡眠级别唤醒到激活状态
                    await self._wake_up(alias, current_level)
                elif level == SleepLevel.SLEEP_1:
                    # 进入 sleep level 1
                    if current_level == SleepLevel.ACTIVE:
                        await self._sleep_level_1(alias)
                    elif current_level == SleepLevel.SLEEP_2:
                        # 从 level 2 唤醒到 level 1
                        await self._wake_up_from_sleep_2(alias)
                        await self._sleep_level_1(alias)
                    elif current_level == SleepLevel.UNLOADED:
                        # 从卸载状态重新加载到 sleep 1
                        await self._reload_and_sleep_1(alias)
                elif level == SleepLevel.SLEEP_2:
                    # 进入 sleep level 2
                    if current_level == SleepLevel.ACTIVE:
                        await self._sleep_level_1(alias)
                        await self._sleep_level_2(alias)
                    elif current_level == SleepLevel.SLEEP_1:
                        await self._sleep_level_2(alias)
                    elif current_level == SleepLevel.UNLOADED:
                        # 从卸载状态不能直接到 sleep 2
                        logger.warning(f"Cannot go from UNLOADED to SLEEP_2 directly")
                        return False
                elif level == SleepLevel.UNLOADED:
                    # 完全卸载
                    await self._unload(alias)
            
            self.sleep_levels[alias] = level
            if instance and level == SleepLevel.ACTIVE:
                instance.status = InstanceStatus.RUNNING
            logger.info(f"✅ Instance {alias} now at sleep level {level.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set sleep level for {alias}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def _sleep_level_1(self, alias: str):
        """进入 Sleep Level 1: offload weights to CPU RAM, discard KV cache"""
        engine = self.engines.get(alias)
        if engine:
            logger.info(f"Putting {alias} to sleep level 1...")
            await engine.sleep(level=1)
            logger.info(f"✅ {alias} entered sleep level 1")
    
    async def _sleep_level_2(self, alias: str):
        """进入 Sleep Level 2: discard both weights and KV cache"""
        engine = self.engines.get(alias)
        if engine:
            logger.info(f"Putting {alias} to sleep level 2...")
            await engine.sleep(level=2)
            logger.info(f"✅ {alias} entered sleep level 2")
    
    async def _unload(self, alias: str):
        """完全卸载引擎实例"""
        logger.info(f"Unloading {alias}...")
        
        # 移除引擎（但保留 processor 和实例信息）
        engine = self.engines.pop(alias, None)
        if engine:
            del engine  # 显式删除，让 GC 回收

        instance = self.instances.get(alias)
        if instance:
            instance.status = InstanceStatus.SLEEPING
            
        logger.info(f"✅ {alias} unloaded")
    
    async def _wake_up(self, alias: str, from_level: SleepLevel):
        """从睡眠状态唤醒到激活状态"""
        if from_level == SleepLevel.ACTIVE:
            return
        elif from_level == SleepLevel.SLEEP_1:
            await self._wake_up_from_sleep_1(alias)
        elif from_level == SleepLevel.SLEEP_2:
            await self._wake_up_from_sleep_2(alias)
        elif from_level == SleepLevel.UNLOADED:
            await self._reload_engine(alias)
    
    async def _wake_up_from_sleep_1(self, alias: str):
        """从 Sleep Level 1 唤醒"""
        engine = self.engines.get(alias)
        if engine:
            logger.info(f"Waking up {alias} from sleep level 1...")
            await engine.wake_up()
            logger.info(f"✅ {alias} woke up from sleep level 1")
    
    async def _wake_up_from_sleep_2(self, alias: str):
        """从 Sleep Level 2 唤醒"""
        engine = self.engines.get(alias)
        if engine:
            logger.info(f"Waking up {alias} from sleep level 2...")
            try:
                # Reallocate weights memory only
                await engine.wake_up(tags=["weights"])

                # Load weights in-place
                await engine.collective_rpc("reload_weights")

                # Reallocate KV cache
                await engine.wake_up(tags=["kv_cache"])

                logger.info(f"✅ {alias} woke up from sleep level 2")
            except Exception as e:
                logger.warning(f"Wake up from sleep level 2 failed for {alias}, reloading: {e}")
                await self._reload_engine(alias, replace_existing=True)
    
    async def _reload_engine(self, alias: str, replace_existing: bool = False):
        """从完全卸载状态重新加载引擎"""
        if alias not in self.engine_args:
            raise ValueError(f"No saved engine args for {alias}")
        
        logger.info(f"Reloading engine {alias}...")
        start_time = time.time()
        
        engine_args = self.engine_args[alias]
        if replace_existing:
            self.engines.pop(alias, None)
        instance = self.instances.get(alias)
        if instance:
            instance.status = InstanceStatus.STARTING
        engine = await asyncio.to_thread(AsyncLLMEngine.from_engine_args, engine_args)
        self.engines[alias] = engine
        if instance:
            instance.status = InstanceStatus.RUNNING
        
        elapsed = time.time() - start_time
        logger.info(f"✅ {alias} reloaded (took {elapsed:.1f}s)")
    
    async def _reload_and_sleep_1(self, alias: str):
        """重新加载引擎并进入 sleep level 1"""
        await self._reload_engine(alias)
        await self._sleep_level_1(alias)
    
    async def ensure_active(self, alias: str):
        """确保引擎处于激活状态（在生成前调用）"""
        if alias in self.fake_instances:
            self.sleep_levels[alias] = SleepLevel.ACTIVE
            return

        current_level = self.sleep_levels.get(alias, SleepLevel.ACTIVE)
        
        if current_level != SleepLevel.ACTIVE:
            logger.info(f"Engine {alias} is at {current_level.name}, waking up...")
            ok = await self.set_sleep_level(alias, SleepLevel.ACTIVE)
            if not ok:
                raise RuntimeError(f"Failed to wake up engine {alias} from {current_level.name}")
    
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
        if alias not in self.engines and alias not in self.instances:
            raise ValueError(f"Engine {alias} not found")

        if alias in self.fake_instances:
            # Fake model generation
            start_time = time.perf_counter()
            first_token_time: Optional[float] = None
            self._inflight_requests[alias] = self._inflight_requests.get(alias, 0) + 1
            fake = self.fake_instances[alias]
            try:
                async with fake.semaphore:
                    await asyncio.sleep(fake.delay_s)
                    first_token_time = time.perf_counter()
                    yield _FakeEngineOutput(
                        outputs=[_FakeCompletionOutput(text=fake.response, finish_reason="stop")]
                    )
            finally:
                end_time = time.perf_counter()
                if first_token_time is not None:
                    ttft = first_token_time - start_time
                    e2e = end_time - start_time
                    metrics = self._latency_metrics.setdefault(
                        alias,
                        {
                            "count": 0,
                            "ttft_last": None,
                            "ttft_avg": None,
                            "e2e_last": None,
                            "e2e_avg": None,
                        },
                    )
                    metrics["count"] += 1
                    count = metrics["count"]
                    metrics["ttft_last"] = ttft
                    metrics["e2e_last"] = e2e
                    if metrics["ttft_avg"] is None:
                        metrics["ttft_avg"] = ttft
                    else:
                        metrics["ttft_avg"] += (ttft - metrics["ttft_avg"]) / count
                    if metrics["e2e_avg"] is None:
                        metrics["e2e_avg"] = e2e
                    else:
                        metrics["e2e_avg"] += (e2e - metrics["e2e_avg"]) / count

                self._inflight_requests[alias] = max(self._inflight_requests.get(alias, 1) - 1, 0)
            return

        if alias not in self.processors:
            raise ValueError(f"Processor {alias} not found")
        
        # 确保引擎处于激活状态
        start_time = time.perf_counter()
        first_token_time: Optional[float] = None
        self._inflight_requests[alias] = self._inflight_requests.get(alias, 0) + 1
        try:
            await self.ensure_active(alias)
        except Exception:
            self._inflight_requests[alias] = max(self._inflight_requests.get(alias, 1) - 1, 0)
            raise
        
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
        try:
            async for output in engine.generate(inputs, sampling_params, request_id):
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                yield output
        finally:
            end_time = time.perf_counter()
            if first_token_time is not None:
                ttft = first_token_time - start_time
                e2e = end_time - start_time
                metrics = self._latency_metrics.setdefault(
                    alias,
                    {
                        "count": 0,
                        "ttft_last": None,
                        "ttft_avg": None,
                        "e2e_last": None,
                        "e2e_avg": None,
                    },
                )
                metrics["count"] += 1
                count = metrics["count"]
                metrics["ttft_last"] = ttft
                metrics["e2e_last"] = e2e
                if metrics["ttft_avg"] is None:
                    metrics["ttft_avg"] = ttft
                else:
                    metrics["ttft_avg"] += (ttft - metrics["ttft_avg"]) / count
                if metrics["e2e_avg"] is None:
                    metrics["e2e_avg"] = e2e
                else:
                    metrics["e2e_avg"] += (e2e - metrics["e2e_avg"]) / count

            self._inflight_requests[alias] = max(self._inflight_requests.get(alias, 1) - 1, 0)
    
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
        
        # 清理睡眠级别和引擎参数
        self.sleep_levels.pop(alias, None)
        self.engine_args.pop(alias, None)
        self._lifecycle_locks.pop(alias, None)
        self._inflight_requests.pop(alias, None)
        self._latency_metrics.pop(alias, None)
        self.fake_instances.pop(alias, None)
        
        # 清理实例信息
        self.instances.pop(alias, None)
        
        return True
    
    def get_instance(self, alias: str) -> Optional[VLLMInstanceInfo]:
        """获取实例信息"""
        return self.instances.get(alias)
    
    def get_sleep_level(self, alias: str) -> Optional[SleepLevel]:
        """获取实例的当前睡眠级别"""
        return self.sleep_levels.get(alias)
    
    def list_instances(self) -> Dict[str, VLLMInstanceInfo]:
        """列出所有实例"""
        return self.instances.copy()
    
    def get_instance_status(self, alias: str) -> Dict:
        """获取实例的详细状态（包括睡眠级别）"""
        instance = self.instances.get(alias)
        if not instance:
            return None
        
        sleep_level = self.sleep_levels.get(alias, SleepLevel.ACTIVE)
        metrics = self._latency_metrics.get(alias, {})
        
        return {
            "alias": alias,
            "status": instance.status.value,
            "model_name": instance.model_name,
            "sleep_level": sleep_level.name,
            "sleep_level_value": sleep_level.value,
            "last_used": instance.last_used,
            "created_at": instance.created_at,
            "idle_time": time.time() - instance.last_used if instance.last_used else 0,
            "inflight_requests": self._inflight_requests.get(alias, 0),
            "ttft_last": metrics.get("ttft_last"),
            "ttft_avg": metrics.get("ttft_avg"),
            "e2e_last": metrics.get("e2e_last"),
            "e2e_avg": metrics.get("e2e_avg"),
            "request_count": metrics.get("count", 0),
            "capacity": self.fake_instances.get(alias).capacity if alias in self.fake_instances else None,
            "is_fake": alias in self.fake_instances,
        }
    
    def update_last_used(self, alias: str):
        """更新实例最后使用时间"""
        instance = self.instances.get(alias)
        if instance:
            instance.last_used = time.time()
