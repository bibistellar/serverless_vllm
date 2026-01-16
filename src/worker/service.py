"""Worker 端主服务 - 单机 GPU 管理服务

负责：
1. 检测本机 GPU 环境
2. 管理 vLLM 实例的启动、停止
3. 接收并路由推理请求到对应的 vLLM 实例
4. 向中央 Manager 注册并定期发送心跳
5. 提供动态路由，根据 alias 转发请求到对应的 vLLM server
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
    messages_to_prompt,
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
        heartbeat_interval: int = 30,
        public_url: Optional[str] = None
    ):
        self.worker_id = worker_id
        self.listen_host = listen_host
        self.listen_port = listen_port
        self.manager_url = manager_url
        self.heartbeat_interval = heartbeat_interval
        self.public_url = public_url  # 公网访问 URL
        
        # vLLM 实例管理器
        self.vllm_manager = VLLMManager()
        
        # GPU 信息
        self.gpu_info = get_gpu_info()
        
        # FastAPI 应用
        self.app = FastAPI(title=f"Worker Service - {worker_id}")
        self._setup_routes()
        
        # 心跳任务
        self._heartbeat_task: Optional[asyncio.Task] = None
    
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
            """获取实例状态（供 Manager 查询）"""
            instance = self.vllm_manager.get_instance(alias)
            if instance:
                return {
                    "alias": alias,
                    "status": instance.status.value,
                    "ready": instance.status == InstanceStatus.RUNNING,
                    "model_name": instance.model_name
                }
            else:
                raise HTTPException(status_code=404, detail=f"Instance {alias} not found")
        
        @self.app.post("/proxy/{alias}/v1/chat/completions")
        async def chat_completions(alias: str, request: ChatCompletionRequest):
            """OpenAI Chat Completions API - 直接调用 vLLM 引擎"""
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
            
            # 转换 messages 为 prompt
            prompt = messages_to_prompt(request.messages)
            
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
                if request.stream:
                    # 流式响应
                    engine_output = self.vllm_manager.generate(alias, prompt, sampling_params)
                    return StreamingResponse(
                        stream_chat_completion(engine_output, request_id, request.model),
                        media_type="text/event-stream"
                    )
                else:
                    # 非流式响应
                    full_text = ""
                    finish_reason = "stop"
                    
                    async for output in self.vllm_manager.generate(alias, prompt, sampling_params):
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
            """OpenAI Completions API - 直接调用 vLLM 引擎"""
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
                # 非流式响应（简化实现）
                full_text = ""
                finish_reason = "stop"
                
                async for output in self.vllm_manager.generate(alias, request.prompt, sampling_params):
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
    
    async def send_heartbeat(self):
        """发送心跳到 Manager"""
        if not self.manager_url:
            return
        
        heartbeat_url = f"{self.manager_url}/workers/{self.worker_id}/heartbeat"
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                instances_data = {
                    alias: instance.to_dict()
                    for alias, instance in self.vllm_manager.list_instances().items()
                }
                
                response = await client.post(
                    heartbeat_url,
                    json={
                        "gpu_info": self.gpu_info.to_dict(),
                        "instances": instances_data
                    }
                )
                
                if response.status_code != 200:
                    logger.warning(f"Heartbeat failed: {response.status_code}")
                    
            except Exception as e:
                logger.error(f"Error sending heartbeat: {e}")
    
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
        
        # 停止所有 vLLM 实例
        for alias in list(self.vllm_manager.instances.keys()):
            try:
                await self.vllm_manager.stop_instance(alias)
                logger.info(f"Stopped instance: {alias}")
            except Exception as e:
                logger.error(f"Error stopping instance {alias}: {e}")
        
        # 从 Manager 注销
        await self.unregister_from_manager()
        
        # 取消心跳任务
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
    
    async def _heartbeat_loop(self):
        """心跳循环"""
        # 首次注册
        await self.register_to_manager()
        
        # 定期发送心跳
        while True:
            await asyncio.sleep(self.heartbeat_interval)
            await self.send_heartbeat()
    
    async def start_heartbeat(self):
        """启动心跳任务"""
        if self._heartbeat_task is None or self._heartbeat_task.done():
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            logger.info("Heartbeat task started")
    
    def run(self):
        """运行 Worker 服务"""
        # 注册启动和关闭事件
        @self.app.on_event("startup")
        async def on_startup():
            await self.start_heartbeat()
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
