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
from typing import Dict, Optional, Tuple
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
import httpx
import uvicorn

from ..common.models import InstanceStatus
from ..common.utils import get_gpu_info
from .vllm_instance import VLLMManager, SleepLevel
from .llama_instance import LlamaManager
from .openai_protocol import (
    ChatCompletionRequest,
    convert_to_sampling_params,
    compute_text_delta,
    format_chat_completion_response,
    stream_chat_completion
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BACKEND_VLLM = "vllm"
BACKEND_LLAMA_CPP = "llama.cpp"
SUPPORTED_BACKENDS = {BACKEND_VLLM, BACKEND_LLAMA_CPP}


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
        # llama.cpp 实例管理器
        self.llama_manager = LlamaManager()
        
        # GPU 信息
        self.gpu_info = get_gpu_info()
        
        # FastAPI 应用
        self.app = FastAPI(title=f"Worker Service - {worker_id}")
        self._setup_routes()
        
        # 已废弃心跳机制（保留注册/注销）

    def _validate_backend_type(self, backend_type: str) -> str:
        if backend_type not in SUPPORTED_BACKENDS:
            raise HTTPException(
                status_code=400,
                detail=f"backend_type must be one of {sorted(SUPPORTED_BACKENDS)}",
            )
        return backend_type

    def _get_manager_by_backend(self, backend_type: str):
        if backend_type == BACKEND_VLLM:
            return self.vllm_manager
        if backend_type == BACKEND_LLAMA_CPP:
            return self.llama_manager
        raise HTTPException(status_code=400, detail=f"unsupported backend_type: {backend_type}")

    def _resolve_alias(self, alias: str) -> Optional[Tuple[str, object]]:
        if self.vllm_manager.get_instance(alias):
            return BACKEND_VLLM, self.vllm_manager
        if self.llama_manager.get_instance(alias):
            return BACKEND_LLAMA_CPP, self.llama_manager
        return None

    def _list_all_instances(self) -> Dict[str, Dict]:
        out: Dict[str, Dict] = {}
        for alias, instance in self.vllm_manager.list_instances().items():
            payload = instance.to_dict()
            payload["backend_type"] = BACKEND_VLLM
            out[alias] = payload
        for alias, instance in self.llama_manager.list_instances().items():
            payload = instance.to_dict()
            payload["backend_type"] = BACKEND_LLAMA_CPP
            out[alias] = payload
        return out
    
    def _setup_routes(self):
        """设置 API 路由"""
        
        @self.app.get("/health")
        async def health():
            """健康检查"""
            return {
                "status": "ok",
                "worker_id": self.worker_id,
                "gpu_info": self.gpu_info.to_dict(),
                "instances": len(self.vllm_manager.instances) + len(self.llama_manager.instances)
            }
        
        @self.app.get("/info")
        async def info():
            """获取 Worker 信息"""
            return {
                "worker_id": self.worker_id,
                "gpu_info": self.gpu_info.to_dict(),
                "instances": self._list_all_instances(),
            }
        
        @self.app.post("/instances/start")
        async def start_instance(request: Request):
            """启动实例
            
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
                backend_type_raw = body.get("backend_type")
                
                if not alias or not model_name or not isinstance(backend_type_raw, str):
                    raise HTTPException(
                        status_code=400,
                        detail="alias, model_name and backend_type are required"
                    )
                backend_type = self._validate_backend_type(backend_type_raw)
                if self._resolve_alias(alias):
                    raise HTTPException(status_code=409, detail=f"Instance {alias} already exists")

                is_fake = model_name in {"__fake__", "fake", "mock", "dummy"} or str(model_name).startswith("fake:")
                if body.get("fake", False):
                    is_fake = True

                manager = self._get_manager_by_backend(backend_type)
                if backend_type == BACKEND_VLLM:
                    instance = await manager.start_instance(
                        alias=alias,
                        model_name=model_name,
                        model_path=body.get("model_path"),
                        gpu_memory_utilization=body.get("gpu_memory_utilization", 0.9),
                        max_model_len=body.get("max_model_len"),
                        tensor_parallel_size=body.get("tensor_parallel_size", 1),
                        fake=is_fake,
                        fake_response=body.get("fake_response"),
                        fake_delay=body.get("fake_delay"),
                        fake_delay_ms=body.get("fake_delay_ms"),
                        fake_capacity=body.get("fake_capacity"),
                    )
                else:
                    instance = await manager.start_instance(
                        alias=alias,
                        model_name=model_name,
                        model_path=body.get("model_path"),
                        fake=is_fake,
                        fake_response=body.get("fake_response"),
                        fake_delay=body.get("fake_delay"),
                        fake_delay_ms=body.get("fake_delay_ms"),
                        fake_capacity=body.get("fake_capacity"),
                        llama_filename=body.get("llama_filename"),
                        llama_mmproj_path=body.get("llama_mmproj_path"),
                        llama_n_gpu_layers=int(body.get("llama_n_gpu_layers", -1)),
                        max_model_len=body.get("max_model_len"),
                    )
                
                payload = instance.to_dict()
                payload["backend_type"] = backend_type
                return {
                    "status": "success",
                    "instance": payload,
                }
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to start instance: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/instances/{alias}/stop")
        async def stop_instance(alias: str):
            """停止实例"""
            resolved = self._resolve_alias(alias)
            if not resolved:
                raise HTTPException(status_code=404, detail=f"Instance {alias} not found")
            _, manager = resolved
            success = await manager.stop_instance(alias)
            if success:
                return {"status": "success", "message": f"Instance {alias} stopped"}
            else:
                raise HTTPException(status_code=404, detail=f"Instance {alias} not found")
        
        @self.app.get("/instances")
        async def list_instances():
            """列出所有实例"""
            return {"instances": self._list_all_instances()}
        
        @self.app.get("/instances/{alias}/status")
        async def get_instance_status(alias: str):
            """获取实例详细状态（包括睡眠级别）"""
            resolved = self._resolve_alias(alias)
            if not resolved:
                raise HTTPException(status_code=404, detail=f"Instance {alias} not found")
            backend_type, manager = resolved
            status = manager.get_instance_status(alias)
            if status:
                status["backend_type"] = backend_type
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
                resolved = self._resolve_alias(alias)
                if not resolved:
                    raise HTTPException(status_code=404, detail=f"Instance {alias} not found")
                backend_type, _ = resolved
                if backend_type != BACKEND_VLLM:
                    raise HTTPException(
                        status_code=400,
                        detail=f"backend {backend_type} does not support sleep",
                    )

                body = await request.json()
                
                # 支持数字或名称
                if "level_name" in body:
                    level_name = body["level_name"].upper()
                    try:
                        level = SleepLevel[level_name]
                    except KeyError:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Invalid level_name: {level_name}. Must be one of: ACTIVE, SLEEP_1, SLEEP_2, UNLOADED"
                        )
                elif "level" in body:
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
            try:
                resolved = self._resolve_alias(alias)
                if not resolved:
                    raise HTTPException(status_code=404, detail=f"Instance {alias} not found")
                backend_type, _ = resolved
                if backend_type != BACKEND_VLLM:
                    raise HTTPException(
                        status_code=400,
                        detail=f"backend {backend_type} does not support wake",
                    )

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
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error waking up {alias}: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/proxy/{alias}/v1/chat/completions")
        async def chat_completions(alias: str, request: ChatCompletionRequest):
            """OpenAI Chat Completions API - Qwen3-VL 优化版本"""
            resolved = self._resolve_alias(alias)
            if not resolved:
                raise HTTPException(status_code=404, detail=f"Instance {alias} not found")
            backend_type, manager = resolved
            instance = manager.get_instance(alias)
            if instance is None:
                raise HTTPException(status_code=404, detail=f"Instance {alias} not found")

            status_payload = manager.get_instance_status(alias) or {}
            logger.info(
                "proxy chat: alias=%s backend=%s status=%s inflight=%s",
                alias,
                backend_type,
                instance.status.value,
                status_payload.get("inflight_requests", 0),
            )

            if backend_type == BACKEND_VLLM:
                # 自动唤醒（包括 UNLOADED）
                await self.vllm_manager.ensure_active(alias)
            
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
                    engine_output = manager.generate(
                        alias, request.messages, sampling_params, base_model=request.base_model
                    )
                    return StreamingResponse(
                        stream_chat_completion(engine_output, request_id, request.model),
                        media_type="text/event-stream"
                    )
                else:
                    # 非流式响应
                    full_text = ""
                    previous_text = ""
                    finish_reason = "stop"
                    
                    output_iter = manager.generate(
                        alias, request.messages, sampling_params, base_model=request.base_model
                    )

                    async for output in output_iter:
                        for completion_output in output.outputs:
                            delta, previous_text = compute_text_delta(
                                previous_text, completion_output.text
                            )
                            if delta:
                                full_text += delta
                            if completion_output.finish_reason:
                                finish_reason = completion_output.finish_reason
                    
                    response = format_chat_completion_response(
                        request_id=request_id,
                        model=request.model,
                        text=full_text,
                        finish_reason=finish_reason
                    )
                    
                    return JSONResponse(content=response)
            
            except (ValueError, TypeError, FileNotFoundError) as e:
                logger.error(f"Invalid request payload for {alias}: {e}")
                raise HTTPException(status_code=400, detail=f"Invalid message format: {e}")
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
        
        # 停止所有实例
        for alias in list(self.vllm_manager.instances.keys()):
            try:
                await self.vllm_manager.stop_instance(alias)
                logger.info(f"Stopped instance: {alias}")
            except Exception as e:
                logger.error(f"Error stopping instance {alias}: {e}")
        for alias in list(self.llama_manager.instances.keys()):
            try:
                await self.llama_manager.stop_instance(alias)
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
