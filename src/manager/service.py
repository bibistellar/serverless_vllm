"""中央 Manager 服务 - Ray Actor 版本

负责：
1. 维护多个 Worker 的信息（GPU 状态、运行的模型实例）
2. 接收 Worker 注册和心跳
3. 根据请求选择合适的 Worker 进行路由
4. 决定在哪个 Worker 上部署新的 vLLM 实例
5. 管理模型路由表

部署方式：作为 Ray Actor 在 Ray cluster 中常驻运行
通信方式：
- 与 Router: Ray remote 调用
- 与 Worker: HTTP 请求
"""
import asyncio
import logging
import time
from typing import Dict, List, Optional
import httpx
import ray

from src.common.models import WorkerInfo, ModelRouting, GPUInfo, VLLMInstanceInfo, WorkerStatus, InstanceStatus

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@ray.remote
class ManagerService:
    """中央管理器服务 - Ray Actor
    
    作为 Ray Actor 运行，提供以下方法供 Router 调用：
    - register_worker: 注册 Worker
    - worker_heartbeat: 接收 Worker 心跳
    - register_model: 注册模型
    - get_model_routing: 获取模型路由信息
    - list_workers: 列出所有 Worker
    - list_models: 列出所有模型
    """
    
    def __init__(self, heartbeat_timeout: int = 60):
        self.heartbeat_timeout = heartbeat_timeout
        
        # Worker 信息表: worker_id -> WorkerInfo
        self.workers: Dict[str, WorkerInfo] = {}
        
        # 模型路由表: alias -> ModelRouting
        self.model_routes: Dict[str, ModelRouting] = {}
        
        logger.info("Manager Actor initialized")
    
    async def health(self) -> Dict:
        """健康检查"""
        active_workers = [w for w in self.workers.values() if w.status == WorkerStatus.ACTIVE]
        return {
            "status": "ok",
            "workers": len(self.workers),
            "active_workers": len(active_workers),
            "models": len(self.model_routes)
        }
    
    async def register_worker(
        self,
        worker_id: str,
        worker_url: str,
        gpu_info: Dict
    ) -> Dict:
        """注册 Worker
        
        Args:
            worker_id: Worker ID
            worker_url: Worker HTTP 地址
            gpu_info: GPU 信息字典
            
        Returns:
            注册结果
        """
        if not worker_id or not worker_url:
            return {
                "status": "error",
                "message": "worker_id and worker_url are required"
            }
        
        # 构建 GPUInfo 对象
        gpu_info_obj = GPUInfo(
            gpu_count=gpu_info.get("gpu_count", 0),
            gpu_names=gpu_info.get("gpu_names", []),
            total_memory_gb=gpu_info.get("total_memory_gb", 0.0),
            available_memory_gb=gpu_info.get("available_memory_gb", 0.0),
            utilization=gpu_info.get("utilization", 0.0)
        )
        
        # 更新或创建 Worker 信息
        if worker_id in self.workers:
            worker = self.workers[worker_id]
            worker.worker_url = worker_url
            worker.gpu_info = gpu_info_obj
            worker.last_heartbeat = time.time()
            worker.status = WorkerStatus.ACTIVE
            logger.info(f"Worker {worker_id} re-registered")
        else:
            worker = WorkerInfo(
                worker_id=worker_id,
                worker_url=worker_url,
                gpu_info=gpu_info_obj,
                last_heartbeat=time.time(),
                status=WorkerStatus.ACTIVE
            )
            self.workers[worker_id] = worker
            logger.info(f"Worker {worker_id} registered: {worker_url}")
        
        return {
            "status": "success",
            "message": f"Worker {worker_id} registered successfully"
        }
    
    async def worker_heartbeat(
        self,
        worker_id: str,
        gpu_info: Dict,
        instances: Dict
    ) -> Dict:
        """接收 Worker 心跳
        
        Args:
            worker_id: Worker ID
            gpu_info: GPU 信息
            instances: 实例信息
            
        Returns:
            心跳响应
        """
        if worker_id not in self.workers:
            return {
                "status": "error",
                "message": f"Worker {worker_id} not found, please register first"
            }
        
        worker = self.workers[worker_id]
        
        # 更新 GPU 信息
        worker.gpu_info = GPUInfo(
            gpu_count=gpu_info.get("gpu_count", 0),
            gpu_names=gpu_info.get("gpu_names", []),
            total_memory_gb=gpu_info.get("total_memory_gb", 0.0),
            available_memory_gb=gpu_info.get("available_memory_gb", 0.0),
            utilization=gpu_info.get("utilization", 0.0)
        )
        
        # 更新实例信息
        worker.instances = {}
        for alias, inst_data in instances.items():
            worker.instances[alias] = VLLMInstanceInfo(
                alias=inst_data.get("alias"),
                model_name=inst_data.get("model_name"),
                port=inst_data.get("port"),
                status=InstanceStatus(inst_data.get("status", "running")),
                created_at=inst_data.get("created_at", time.time()),
                base_url=inst_data.get("base_url", ""),
                pid=inst_data.get("pid"),
                last_used=inst_data.get("last_used", 0.0)
            )
        
        # 更新心跳时间和状态
        worker.last_heartbeat = time.time()
        worker.status = WorkerStatus.ACTIVE
        
        return {"status": "ok"}
    
    async def unregister_worker(self, worker_id: str) -> Dict:
        """注销 Worker
        
        Args:
            worker_id: Worker ID
            
        Returns:
            注销结果
        """
        if worker_id not in self.workers:
            return {
                "status": "error",
                "message": f"Worker {worker_id} not found"
            }
        
        # 移除该 Worker 上的所有模型路由
        aliases_to_remove = [
            alias for alias, route in self.model_routes.items()
            if route.worker_id == worker_id
        ]
        
        for alias in aliases_to_remove:
            del self.model_routes[alias]
            logger.info(f"Removed model routing for {alias} (worker {worker_id})")
        
        # 移除 Worker
        del self.workers[worker_id]
        logger.info(f"Worker {worker_id} unregistered")
        
        return {
            "status": "ok",
            "message": f"Worker {worker_id} unregistered successfully"
        }
    
    async def list_workers(self) -> Dict:
        """列出所有 Worker"""
        return {
            "workers": [worker.to_dict() for worker in self.workers.values()]
        }
    
    async def get_worker(self, worker_id: str) -> Dict:
        """获取 Worker 信息"""
        if worker_id not in self.workers:
            return {
                "status": "error",
                "message": f"Worker {worker_id} not found"
            }
        
        return self.workers[worker_id].to_dict()
    
    async def register_model(
        self,
        alias: str,
        model_name: str,
        model_path: Optional[str] = None,
        gpu_memory_utilization: float = 0.9,
        max_model_len: Optional[int] = None,
        tensor_parallel_size: int = 1,
        worker_id: Optional[str] = None
    ) -> Dict:
        """注册模型并在合适的 Worker 上启动 vLLM 实例
        
        Args:
            alias: 模型别名
            model_name: 模型名称
            model_path: 模型路径（可选）
            gpu_memory_utilization: GPU 内存利用率
            max_model_len: 最大序列长度
            tensor_parallel_size: 张量并行度
            worker_id: 指定 Worker ID（可选）
            
        Returns:
            注册结果
        """
        if not alias or not model_name:
            return {
                "status": "error",
                "message": "alias and model_name are required"
            }
        
        # 检查是否已经存在
        if alias in self.model_routes:
            return {
                "status": "exists",
                "message": f"Model {alias} already registered",
                "routing": self.model_routes[alias].to_dict()
            }
        
        # 选择 Worker
        if worker_id:
            if worker_id not in self.workers:
                return {
                    "status": "error",
                    "message": f"Worker {worker_id} not found"
                }
            worker = self.workers[worker_id]
        else:
            worker = self._select_worker_for_model()
            if not worker:
                return {
                    "status": "error",
                    "message": "No available worker found"
                }
        
        # 在 Worker 上启动实例
        try:
            instance_info = await self._start_instance_on_worker(
                worker=worker,
                alias=alias,
                model_name=model_name,
                model_path=model_path,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                tensor_parallel_size=tensor_parallel_size
            )
            
            # 创建路由
            routing = ModelRouting(
                alias=alias,
                model_name=model_name,
                worker_id=worker.worker_id,
                worker_url=worker.worker_url,
                vllm_port=instance_info.get("port"),
                created_at=time.time()
            )
            self.model_routes[alias] = routing
            
            logger.info(f"Model {alias} registered on worker {worker.worker_id}")
            
            return {
                "status": "success",
                "message": f"Model {alias} registered successfully",
                "routing": routing.to_dict()
            }
            
        except Exception as e:
            logger.error(f"Failed to register model {alias}: {e}")
            return {
                "status": "error",
                "message": f"Failed to start instance: {str(e)}"
            }
    
    async def list_models(self) -> Dict:
        """列出所有已注册的模型"""
        return {
            "models": [route.to_dict() for route in self.model_routes.values()]
        }
    
    async def get_model_routing(self, alias: str) -> Optional[Dict]:
        """获取模型路由信息"""
        if alias not in self.model_routes:
            return None
        
        return self.model_routes[alias].to_dict()
    
    async def unregister_model(self, alias: str) -> Dict:
        """注销模型"""
        if alias not in self.model_routes:
            return {
                "status": "error",
                "message": f"Model {alias} not found"
            }
        
        routing = self.model_routes[alias]
        
        # 停止 Worker 上的实例
        try:
            await self._stop_instance_on_worker(
                worker_url=routing.worker_url,
                alias=alias
            )
        except Exception as e:
            logger.warning(f"Failed to stop instance {alias}: {e}")
        
        # 删除路由
        del self.model_routes[alias]
        
        # 从 Worker 的实例列表中删除
        if routing.worker_id in self.workers:
            worker = self.workers[routing.worker_id]
            if alias in worker.instances:
                del worker.instances[alias]
        
        logger.info(f"Model {alias} unregistered")
        
        return {
            "status": "success",
            "message": f"Model {alias} unregistered"
        }
    
    async def check_workers_health(self):
        """检查 Worker 健康状态（供定期调用）"""
        current_time = time.time()
        
        for worker_id, worker in list(self.workers.items()):
            if current_time - worker.last_heartbeat > self.heartbeat_timeout:
                if worker.status == WorkerStatus.ACTIVE:
                    worker.status = WorkerStatus.INACTIVE
                    logger.warning(f"Worker {worker_id} marked as inactive (timeout)")
                    
                    # 清理该 worker 上的路由
                    stale_routes = [
                        alias for alias, route in self.model_routes.items()
                        if route.worker_id == worker_id
                    ]
                    for alias in stale_routes:
                        del self.model_routes[alias]
                        logger.info(f"Removed stale route: {alias}")
    
    def _select_worker_for_model(self) -> Optional[WorkerInfo]:
        """选择合适的 Worker 部署模型
        
        策略：选择可用内存最多的 active worker
        """
        active_workers = [
            w for w in self.workers.values()
            if w.status == WorkerStatus.ACTIVE
        ]
        
        if not active_workers:
            return None
        
        # 按可用内存排序
        return max(active_workers, key=lambda w: w.gpu_info.available_memory_gb)
    
    async def _start_instance_on_worker(
        self,
        worker: WorkerInfo,
        alias: str,
        model_name: str,
        model_path: Optional[str],
        gpu_memory_utilization: float,
        max_model_len: Optional[int],
        tensor_parallel_size: int
    ) -> Dict:
        """在 Worker 上启动 vLLM 实例
        
        注意：模型加载可能需要较长时间（几分钟到十几分钟），所以使用较长的超时
        """
        url = f"{worker.worker_url}/instances/start"
        
        payload = {
            "alias": alias,
            "model_name": model_name,
            "model_path": model_path,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_model_len": max_model_len,
            "tensor_parallel_size": tensor_parallel_size
        }
        
        # 使用 30 分钟超时，因为大模型加载时间不确定（可能需要几分钟到十几分钟）
        async with httpx.AsyncClient(timeout=1800.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            return response.json()
    
    async def _stop_instance_on_worker(
        self,
        worker_url: str,
        alias: str
    ) -> Dict:
        """停止 Worker 上的 vLLM 实例"""
        url = f"{worker_url}/instances/{alias}/stop"
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.delete(url)
            response.raise_for_status()
            return response.json()
