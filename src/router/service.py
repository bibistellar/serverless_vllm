"""HTTP Router 服务 - Ray Serve Deployment

提供 OpenAI 兼容的 API 接口，通过 Ray remote 调用 Manager 获取路由信息
然后将推理请求转发到对应的 Worker
"""
import logging
from typing import Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
import httpx
import ray
from ray import serve

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_cpus": 1}
)
class RouterService:
    """Router 服务 - Ray Serve Deployment
    
    对外暴露 OpenAI 兼容的 API，内部通过 Ray remote 调用 Manager
    """
    
    def __init__(self, manager_actor_name: str = "manager"):
        """初始化 Router
        
        Args:
            manager_actor_name: Manager Actor 的名称
        """
        self.manager_actor_name = manager_actor_name
        self._manager = None
        
        # 创建 FastAPI 应用
        self.app = FastAPI(title="LLM Router Service")
        
        # 注册所有路由
        self._register_routes()
        
        logger.info(f"Router initialized, will connect to Manager: {manager_actor_name}")
    
    def _get_manager(self):
        """获取 Manager Actor 句柄"""
        if self._manager is None:
            try:
                self._manager = ray.get_actor(self.manager_actor_name)
                logger.info(f"Connected to Manager Actor: {self.manager_actor_name}")
            except ValueError as e:
                logger.error(f"Failed to get Manager Actor '{self.manager_actor_name}': {e}")
                raise HTTPException(
                    status_code=503,
                    detail=f"Manager service not available"
                )
        return self._manager
    
    def _register_routes(self):
        """注册所有 HTTP 路由"""
        
        @self.app.get("/health")
        async def health():
            """健康检查"""
            try:
                manager = self._get_manager()
                manager_health = await manager.health.remote()
                return {
                    "status": "ok",
                    "service": "router",
                    "manager": manager_health
                }
            except Exception as e:
                return {
                    "status": "error",
                    "service": "router",
                    "error": str(e)
                }
        
        @self.app.post("/v1/chat/completions")
        async def chat_completions(request: Request):
            """OpenAI Chat Completions API"""
            body = await request.json()
            model = body.get("model")
            
            if not model:
                raise HTTPException(status_code=400, detail="model is required")
            
            try:
                manager = self._get_manager()
                routing = await manager.get_model_routing.remote(model)
                
                if routing is None:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Model {model} not found or not available"
                    )
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to get routing for model {model}: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Internal error: {str(e)}"
                )
            
            worker_url = routing["worker_url"]
            alias = routing["alias"]
            target_url = f"{worker_url}/proxy/{alias}/v1/chat/completions"
            
            async with httpx.AsyncClient(timeout=300.0) as client:
                try:
                    is_stream = body.get("stream", False)
                    
                    response = await client.post(
                        target_url,
                        json=body,
                        headers={k: v for k, v in request.headers.items() 
                                if k.lower() not in ['host', 'content-length']}
                    )
                    
                    if is_stream:
                        async def stream_generator():
                            async for chunk in response.aiter_bytes():
                                yield chunk
                        
                        return StreamingResponse(
                            stream_generator(),
                            media_type="text/event-stream",
                            headers=dict(response.headers)
                        )
                    else:
                        return JSONResponse(
                            content=response.json(),
                            status_code=response.status_code
                        )
                
                except httpx.TimeoutException:
                    raise HTTPException(status_code=504, detail="Request timed out")
                except Exception as e:
                    logger.error(f"Error forwarding request: {e}")
                    raise HTTPException(
                        status_code=502,
                        detail=f"Error forwarding request: {str(e)}"
                    )
        
        @self.app.post("/v1/completions")
        async def completions(request: Request):
            """OpenAI Completions API"""
            body = await request.json()
            model = body.get("model")
            
            if not model:
                raise HTTPException(status_code=400, detail="model is required")
            
            try:
                manager = self._get_manager()
                routing = await manager.get_model_routing.remote(model)
                
                if routing is None:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Model {model} not found"
                    )
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to get routing for model {model}: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Internal error: {str(e)}"
                )
            
            worker_url = routing["worker_url"]
            alias = routing["alias"]
            target_url = f"{worker_url}/proxy/{alias}/v1/completions"
            
            async with httpx.AsyncClient(timeout=300.0) as client:
                try:
                    response = await client.post(
                        target_url,
                        json=body,
                        headers={k: v for k, v in request.headers.items() 
                                if k.lower() not in ['host', 'content-length']}
                    )
                    
                    return JSONResponse(
                        content=response.json(),
                        status_code=response.status_code
                    )
                
                except Exception as e:
                    logger.error(f"Error forwarding request: {e}")
                    raise HTTPException(
                        status_code=502,
                        detail=f"Error forwarding request: {str(e)}"
                    )
        
        @self.app.get("/v1/models")
        async def list_models():
            """列出所有可用模型"""
            try:
                manager = self._get_manager()
                result = await manager.list_models.remote()
                return result
            except Exception as e:
                logger.error(f"Error listing models: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Error listing models: {str(e)}"
                )
        
        @self.app.post("/v1/models/register")
        async def register_model(request: Request):
            """注册新模型（管理接口）"""
            body = await request.json()
            
            try:
                manager = self._get_manager()
                result = await manager.register_model.remote(
                    alias=body.get("alias"),
                    model_name=body.get("model_name"),
                    model_path=body.get("model_path"),
                    gpu_memory_utilization=body.get("gpu_memory_utilization", 0.9),
                    max_model_len=body.get("max_model_len"),
                    tensor_parallel_size=body.get("tensor_parallel_size", 1),
                    worker_id=body.get("worker_id")
                )
                return result
            except Exception as e:
                logger.error(f"Error registering model: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Error registering model: {str(e)}"
                )
        
        @self.app.delete("/v1/models/{alias}")
        async def unregister_model(alias: str):
            """注销模型"""
            try:
                manager = self._get_manager()
                result = await manager.unregister_model.remote(alias)
                return result
            except Exception as e:
                logger.error(f"Error unregistering model: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Error unregistering model: {str(e)}"
                )
        
        @self.app.get("/workers")
        async def list_workers():
            """列出所有 Worker（管理接口）"""
            try:
                manager = self._get_manager()
                result = await manager.list_workers.remote()
                return result
            except Exception as e:
                logger.error(f"Error listing workers: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Error listing workers: {str(e)}"
                )
        
        @self.app.post("/workers/register")
        async def register_worker(request: Request):
            """注册 Worker（由 Worker 自动调用）"""
            body = await request.json()
            
            try:
                manager = self._get_manager()
                result = await manager.register_worker.remote(
                    worker_id=body.get("worker_id"),
                    worker_url=body.get("worker_url"),
                    gpu_info=body.get("gpu_info", {}),
                    public_worker_url=body.get("public_worker_url")
                )
                return result
            except Exception as e:
                logger.error(f"Error registering worker: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Error registering worker: {str(e)}"
                )
        
        @self.app.post("/workers/{worker_id}/heartbeat")
        async def worker_heartbeat(worker_id: str, request: Request):
            """接收 Worker 心跳（由 Worker 自动调用）"""
            body = await request.json()
            
            try:
                manager = self._get_manager()
                result = await manager.worker_heartbeat.remote(
                    worker_id=worker_id,
                    gpu_info=body.get("gpu_info", {}),
                    instances=body.get("instances", {})
                )
                return result
            except Exception as e:
                logger.error(f"Error processing heartbeat: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Error processing heartbeat: {str(e)}"
                )
        
        @self.app.delete("/workers/{worker_id}/unregister")
        async def unregister_worker(worker_id: str):
            """注销 Worker（由 Worker 退出时调用）"""
            try:
                manager = self._get_manager()
                result = await manager.unregister_worker.remote(worker_id)
                return result
            except Exception as e:
                logger.error(f"Error unregistering worker: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Error unregistering worker: {str(e)}"
                )
    
    async def __call__(self, request):
        """Ray Serve HTTP 入口 - 直接调用 FastAPI app"""
        # Ray Serve 会传入一个 Starlette Request 对象
        # 我们需要将它转换为 ASGI scope/receive/send 并调用 FastAPI
        scope = request.scope
        receive = request.receive
        
        # 创建一个响应收集器
        response_started = False
        status_code = 200
        headers = []
        body_parts = []
        
        async def send(message):
            nonlocal response_started, status_code, headers
            if message["type"] == "http.response.start":
                response_started = True
                status_code = message["status"]
                headers = message.get("headers", [])
            elif message["type"] == "http.response.body":
                body_parts.append(message.get("body", b""))
        
        # 调用 FastAPI 应用
        await self.app(scope, receive, send)
        
        # 构造响应
        from starlette.responses import Response
        return Response(
            content=b"".join(body_parts),
            status_code=status_code,
            headers={k.decode(): v.decode() for k, v in headers}
        )
