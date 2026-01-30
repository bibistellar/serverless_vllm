"""Worker 端主服务 - 单机 GPU 管理服务

负责：
1. 检测本机 GPU 环境
2. 管理 vLLM 实例的启动、停止
3. 接收并路由推理请求到对应的 vLLM 实例
4. 向中央 Manager 注册并定期发送心跳
5. 提供动态路由，根据 alias 转发请求到对应的 vLLM server

专门针对 Qwen3-VL 多模态模型优化
"""
import asyncio
import logging
import time
import signal
import atexit
import os
from typing import Dict, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
import httpx
import uvicorn

from src.common.models import VLLMInstanceInfo, InstanceStatus, WorkerStatus
from src.common.utils import get_gpu_info
from src.worker.vllm_manager import VLLMManager
from src.worker.openai_protocol import (
    ChatCompletionRequest,
    CompletionRequest,
    convert_to_sampling_params,
    format_chat_completion_response,
    format_completion_response,
    stream_chat_completion
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WorkerService:
    """Worker 服务"""
    
    def __init__(
        self,
        worker_id: str,
        listen_host: str = "0.0.0.0",
        listen_port: int = 7000,
        manager_url: Optional[str] = None,
        public_url: Optional[str] = None
    ):
        self.worker_id = worker_id
        self.listen_host = listen_host
        self.listen_port = listen_port
        self.manager_url = manager_url
        self.public_url = public_url  # 公网访问 URL
        
        # vLLM 实例管理器
        self.vllm_manager = VLLMManager()
        
        # GPU 信息
        self.gpu_info = get_gpu_info()
        
        # FastAPI 应用
        self.app = FastAPI(title=f"Worker Service - {worker_id}")
        self._setup_routes()
        
        # 已废弃心跳机制（保留注册/注销）
    
    def _setup_routes(self):
        """设置 API 路由"""
        
        @self.app.get("/health")
        async def health():
            """健康检查"""
            return {
                "status": "ok",
                "worker_id": self.worker_id,
                "gpu_info": self.gpu_info.to_dict(),
                "instances": len(self.vllm_manager.instances)
            }
        
        @self.app.get("/info")
        async def info():
            """获取 Worker 信息"""
            return {
                "worker_id": self.worker_id,
                "gpu_info": self.gpu_info.to_dict(),
                "instances": {
                    alias: instance.to_dict()
                    for alias, instance in self.vllm_manager.list_instances().items()
                }
            }
        
        @self.app.post("/instances/start")
        async def start_instance(request: Request):
            """启动 vLLM 实例
            
            请求体：
            {
                "alias": "qwen-vl-2b",
                "model_name": "Qwen/Qwen3-VL-2B-Instruct",
                "model_path": "/path/to/model",  # 可选
                "gpu_memory_utilization": 0.9,  # 可选
                "max_model_len": 8192,  # 可选
                "tensor_parallel_size": 1  # 可选
            }
            """
            try:
                body = await request.json()
                alias = body.get("alias")
                model_name = body.get("model_name")
                
                if not alias or not model_name:
                    raise HTTPException(
                        status_code=400,
                        detail="alias and model_name are required"
                    )
                
                is_fake = model_name in {"__fake__", "fake", "mock", "dummy"} or str(model_name).startswith("fake:")
                if body.get("fake", False):
                    is_fake = True

                if is_fake:
                    fake_response = body.get("fake_response", "FAKE_RESPONSE")
                    fake_delay = body.get("fake_delay", None)
                    fake_delay_ms = body.get("fake_delay_ms", 500)
                    delay_s = float(fake_delay) if fake_delay is not None else float(fake_delay_ms) / 1000.0
                    fake_capacity = int(body.get("fake_capacity", 1))
                    instance = await self.vllm_manager.start_fake_instance(
                        alias=alias,
                        response=fake_response,
                        delay_s=delay_s,
                        capacity=fake_capacity,
                    )
                else:
                    instance = await self.vllm_manager.start_instance(
                        alias=alias,
                        model_name=model_name,
                        model_path=body.get("model_path"),
                        gpu_memory_utilization=body.get("gpu_memory_utilization", 0.9),
                        max_model_len=body.get("max_model_len"),
                        tensor_parallel_size=body.get("tensor_parallel_size", 1)
                    )
                
                return {
                    "status": "success",
                    "instance": instance.to_dict()
                }
                
            except Exception as e:
                logger.error(f"Failed to start instance: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/instances/{alias}/stop")
        async def stop_instance(alias: str):
            """停止 vLLM 实例"""
            success = await self.vllm_manager.stop_instance(alias)
            if success:
                return {"status": "success", "message": f"Instance {alias} stopped"}
            else:
                raise HTTPException(status_code=404, detail=f"Instance {alias} not found")
        
        @self.app.get("/instances")
        async def list_instances():
            """列出所有实例"""
            instances = self.vllm_manager.list_instances()
            return {
                "instances": {
                    alias: instance.to_dict()
                    for alias, instance in instances.items()
                }
            }
        
        @self.app.get("/instances/{alias}")
        async def get_instance(alias: str):
            """获取实例信息"""
            instance = self.vllm_manager.get_instance(alias)
            if instance:
                return instance.to_dict()
            else:
                raise HTTPException(status_code=404, detail=f"Instance {alias} not found")
        
        @self.app.get("/instances/{alias}/status")
        async def get_instance_status(alias: str):
            """获取实例详细状态（包括睡眠级别）"""
            status = self.vllm_manager.get_instance_status(alias)
            if status:
                status["gpu_info"] = get_gpu_info().to_dict()
                return status
            else:
                raise HTTPException(status_code=404, detail=f"Instance {alias} not found")
        
        @self.app.post("/instances/{alias}/sleep")
        async def set_sleep_level(alias: str, request: Request):
            """设置实例的睡眠级别
            
            请求体：
            {
                "level": 0,  # 0=ACTIVE, 1=SLEEP_1, 2=SLEEP_2, 3=UNLOADED
                "level_name": "SLEEP_1"  # 可选，使用名称
            }
            """
            try:
                body = await request.json()
                
                # 支持数字或名称
                if "level_name" in body:
                    from src.worker.vllm_manager import SleepLevel
                    level_name = body["level_name"].upper()
                    try:
                        level = SleepLevel[level_name]
                    except KeyError:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Invalid level_name: {level_name}. Must be one of: ACTIVE, SLEEP_1, SLEEP_2, UNLOADED"
                        )
                elif "level" in body:
                    from src.worker.vllm_manager import SleepLevel
                    level_value = body["level"]
                    try:
                        level = SleepLevel(level_value)
                    except ValueError:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Invalid level: {level_value}. Must be 0-3"
                        )
                else:
                    raise HTTPException(
                        status_code=400,
                        detail="Must provide 'level' (0-3) or 'level_name' (ACTIVE/SLEEP_1/SLEEP_2/UNLOADED)"
                    )
                
                success = await self.vllm_manager.set_sleep_level(alias, level)
                
                if success:
                    return {
                        "status": "success",
                        "message": f"Instance {alias} set to {level.name}",
                        "level": level.value,
                        "level_name": level.name
                    }
                else:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Failed to set sleep level for {alias}"
                    )
                    
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error setting sleep level for {alias}: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/instances/{alias}/wake")
        async def wake_instance(alias: str):
            """唤醒实例到激活状态"""
            from src.worker.vllm_manager import SleepLevel
            
            try:
                success = await self.vllm_manager.set_sleep_level(alias, SleepLevel.ACTIVE)
                
                if success:
                    return {
                        "status": "success",
                        "message": f"Instance {alias} woken up"
                    }
                else:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Failed to wake up {alias}"
                    )
            except Exception as e:
                logger.error(f"Error waking up {alias}: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/sleep/config")
        async def configure_auto_sleep(request: Request):
            """配置自动睡眠参数
            
            请求体：
            {
                "enable": true,
                "sleep_1_timeout": 300,
                "sleep_2_timeout": 900,
                "unload_timeout": 1800
            }
            """
            try:
                body = await request.json()
                
                if "enable" in body:
                    self.vllm_manager.enable_auto_sleep = body["enable"]
                
                if "sleep_1_timeout" in body:
                    self.vllm_manager.sleep_1_timeout = body["sleep_1_timeout"]
                
                if "sleep_2_timeout" in body:
                    self.vllm_manager.sleep_2_timeout = body["sleep_2_timeout"]
                
                if "unload_timeout" in body:
                    self.vllm_manager.unload_timeout = body["unload_timeout"]
                
                return {
                    "status": "success",
                    "config": {
                        "enable_auto_sleep": self.vllm_manager.enable_auto_sleep,
                        "sleep_1_timeout": self.vllm_manager.sleep_1_timeout,
                        "sleep_2_timeout": self.vllm_manager.sleep_2_timeout,
                        "unload_timeout": self.vllm_manager.unload_timeout
                    }
                }
                
            except Exception as e:
                logger.error(f"Error configuring auto-sleep: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/sleep/config")
        async def get_sleep_config():
            instance = self.vllm_manager.get_instance(alias)
            if not instance:
                raise HTTPException(status_code=404, detail=f"Instance {alias} not found")
            else:
                """获取当前的自动睡眠配置"""
                return {
                    "enable_auto_sleep": self.vllm_manager.enable_auto_sleep,
                    "sleep_1_timeout": self.vllm_manager.sleep_1_timeout,
                    "sleep_2_timeout": self.vllm_manager.sleep_2_timeout,
                    "unload_timeout": self.vllm_manager.unload_timeout,
                }

        
        @self.app.post("/proxy/{alias}/v1/chat/completions")
        async def chat_completions(alias: str, request: ChatCompletionRequest):
            """OpenAI Chat Completions API - Qwen3-VL 优化版本"""
            instance = self.vllm_manager.get_instance(alias)
            if not instance:
                raise HTTPException(status_code=404, detail=f"Instance {alias} not found")
            
            # 检查模型状态
            if instance.status == InstanceStatus.STARTING:
                return JSONResponse(
                    status_code=503,
                    content={
                        "error": {
                            "message": f"模型 {alias} 正在加载中，请稍候再试...",
                            "type": "model_loading",
                            "code": "model_not_ready",
                            "status": "starting"
                        }
                    }
                )
            elif instance.status != InstanceStatus.RUNNING:
                raise HTTPException(
                    status_code=503,
                    detail=f"Instance {alias} is not running (status: {instance.status.value})"
                )
            
            # 转换 OpenAI 格式的消息为 Qwen3-VL 格式
            from src.worker.openai_protocol import process_messages_to_qwen_format
            
            try:
                qwen_messages = await process_messages_to_qwen_format(request.messages)
                logger.debug(f"Converted {len(qwen_messages)} messages for {alias}")
            except Exception as e:
                logger.error(f"Failed to convert messages: {e}")
                raise HTTPException(status_code=400, detail=f"Invalid message format: {e}")
            
            # 构建采样参数
            sampling_params = convert_to_sampling_params(
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
                stop=request.stop,
                top_k=request.top_k,
                repetition_penalty=request.repetition_penalty
            )
            
            # 生成请求 ID
            import uuid
            request_id = str(uuid.uuid4())
            
            try:
                if request.stream:
                    # 流式响应
                    engine_output = self.vllm_manager.generate(
                        alias, qwen_messages, sampling_params
                    )
                    return StreamingResponse(
                        stream_chat_completion(engine_output, request_id, request.model),
                        media_type="text/event-stream"
                    )
                else:
                    # 非流式响应
                    full_text = ""
                    finish_reason = "stop"
                    
                    async for output in self.vllm_manager.generate(
                        alias, qwen_messages, sampling_params
                    ):
                        for completion_output in output.outputs:
                            full_text = completion_output.text
                            if completion_output.finish_reason:
                                finish_reason = completion_output.finish_reason
                    
                    response = format_chat_completion_response(
                        request_id=request_id,
                        model=request.model,
                        text=full_text,
                        finish_reason=finish_reason
                    )
                    
                    return JSONResponse(content=response)
            
            except Exception as e:
                logger.error(f"Error during generation for {alias}: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/proxy/{alias}/v1/completions")
        async def completions(alias: str, request: CompletionRequest):
            """OpenAI Completions API - 简化版本（不推荐使用，请使用 Chat Completions）"""
            instance = self.vllm_manager.get_instance(alias)
            if not instance:
                raise HTTPException(status_code=404, detail=f"Instance {alias} not found")
            
            # 检查模型状态
            if instance.status == InstanceStatus.STARTING:
                return JSONResponse(
                    status_code=503,
                    content={
                        "error": {
                            "message": f"模型 {alias} 正在加载中，请稍候再试...",
                            "type": "model_loading",
                            "code": "model_not_ready",
                            "status": "starting"
                        }
                    }
                )
            elif instance.status != InstanceStatus.RUNNING:
                raise HTTPException(
                    status_code=503,
                    detail=f"Instance {alias} is not running"
                )
            
            # 将 prompt 转换为消息格式
            messages = [{"role": "user", "content": request.prompt}]
            
            # 构建采样参数
            sampling_params = convert_to_sampling_params(
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
                stop=request.stop
            )
            
            # 生成请求 ID
            import uuid
            request_id = str(uuid.uuid4())
            
            try:
                # 使用统一的 generate 方法
                full_text = ""
                finish_reason = "stop"
                
                async for output in self.vllm_manager.generate(alias, messages, sampling_params):
                    for completion_output in output.outputs:
                        full_text = completion_output.text
                        if completion_output.finish_reason:
                            finish_reason = completion_output.finish_reason
                
                response = format_completion_response(
                    request_id=request_id,
                    model=request.model,
                    text=full_text,
                    finish_reason=finish_reason
                )
                
                return JSONResponse(content=response)
            
            except Exception as e:
                logger.error(f"Error during generation for {alias}: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    async def register_to_manager(self):
        """向中央 Manager 注册"""
        if not self.manager_url:
            logger.info("No manager URL configured, skipping registration")
            return
        
        register_url = f"{self.manager_url}/workers/register"
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                payload = {
                    "worker_id": self.worker_id,
                    "worker_url": f"http://{self.listen_host}:{self.listen_port}",
                    "gpu_info": self.gpu_info.to_dict()
                }
                
                # 如果配置了公网URL，则添加到请求中
                if self.public_url:
                    payload["public_worker_url"] = self.public_url
                
                response = await client.post(
                    register_url,
                    json=payload
                )
                
                if response.status_code == 200:
                    logger.info(f"Successfully registered to manager: {self.manager_url}")
                    if self.public_url:
                        logger.info(f"Public URL: {self.public_url}")
                else:
                    logger.error(f"Failed to register to manager: {response.status_code} - {response.text}")
                    
            except Exception as e:
                logger.error(f"Error registering to manager: {e}")
    
    async def unregister_from_manager(self):
        """从 Manager 注销"""
        if not self.manager_url:
            return
        
        unregister_url = f"{self.manager_url}/workers/{self.worker_id}/unregister"
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.delete(unregister_url)
                if response.status_code == 200:
                    logger.info(f"Successfully unregistered from manager")
                else:
                    logger.warning(f"Failed to unregister from manager: {response.status_code}")
            except Exception as e:
                logger.warning(f"Error unregistering from manager: {e}")
    
    async def cleanup(self):
        """清理资源"""
        logger.info("Cleaning up Worker resources...")
        
        # 停止睡眠监控任务
        await self.vllm_manager.stop_sleep_monitor()
        
        # 停止所有 vLLM 实例
        for alias in list(self.vllm_manager.instances.keys()):
            try:
                await self.vllm_manager.stop_instance(alias)
                logger.info(f"Stopped instance: {alias}")
            except Exception as e:
                logger.error(f"Error stopping instance {alias}: {e}")
        
        # 从 Manager 注销
        await self.unregister_from_manager()
        
        # 无心跳任务需要取消
    
    def run(self):
        """运行 Worker 服务"""
        # 注册启动和关闭事件
        @self.app.on_event("startup")
        async def on_startup():
            await self.register_to_manager()
            # 启动睡眠监控任务
            await self.vllm_manager.start_sleep_monitor()
            logger.info(f"Worker Service started: {self.worker_id}")
        
        @self.app.on_event("shutdown")
        async def on_shutdown():
            await self.cleanup()
            logger.info(f"Worker Service shut down: {self.worker_id}")
        
        logger.info(f"Starting Worker Service: {self.worker_id}")
        logger.info(f"Listening on {self.listen_host}:{self.listen_port}")
        logger.info(f"GPU Info: {self.gpu_info.to_dict()}")
        
        uvicorn.run(
            self.app,
            host=self.listen_host,
            port=self.listen_port,
            log_level="info"
        )


def main():
    """主函数"""
    import sys
    
    worker_id = os.getenv("WORKER_ID", "worker-1")
    listen_host = os.getenv("WORKER_HOST", "0.0.0.0")
    listen_port = int(os.getenv("WORKER_PORT", "7000"))
    manager_url = os.getenv("MANAGER_URL")  # 如 http://manager:9000
    public_url = os.getenv("PUBLIC_URL")  # 如 https://u557149-9507-6992150f.bjb2.seetacloud.com:8443
    
    service = WorkerService(
        worker_id=worker_id,
        listen_host=listen_host,
        listen_port=listen_port,
        manager_url=manager_url,
        public_url=public_url
    )
    
    # 运行服务（cleanup 会在 FastAPI 的 shutdown 事件中自动调用）
    try:
        service.run()
    except KeyboardInterrupt:
        logger.info("Received KeyboardInterrupt, shutting down...")
    except Exception as e:
        logger.error(f"Error running service: {e}")
        raise
    finally:
        logger.info("Worker service stopped")


if __name__ == "__main__":
    main()
