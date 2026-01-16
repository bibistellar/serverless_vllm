"""数据模型定义"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


class WorkerStatus(str, Enum):
    """Worker 状态"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    UNHEALTHY = "unhealthy"


class InstanceStatus(str, Enum):
    """vLLM 实例状态"""
    STARTING = "starting"
    RUNNING = "running"
    SLEEPING = "sleeping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class GPUInfo:
    """GPU 信息"""
    gpu_count: int
    gpu_names: List[str]
    total_memory_gb: float
    available_memory_gb: float
    utilization: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "gpu_count": self.gpu_count,
            "gpu_names": self.gpu_names,
            "total_memory_gb": self.total_memory_gb,
            "available_memory_gb": self.available_memory_gb,
            "utilization": self.utilization
        }


@dataclass
class VLLMInstanceInfo:
    """vLLM 实例信息"""
    alias: str
    model_name: str
    port: int
    status: InstanceStatus
    created_at: float
    base_url: str  # 内部访问 URL，如 http://localhost:8001
    pid: Optional[int] = None
    last_used: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "alias": self.alias,
            "model_name": self.model_name,
            "port": self.port,
            "status": self.status.value,
            "created_at": self.created_at,
            "base_url": self.base_url,
            "pid": self.pid,
            "last_used": self.last_used
        }


@dataclass
class WorkerInfo:
    """Worker 信息"""
    worker_id: str
    worker_url: str
    gpu_info: GPUInfo
    last_heartbeat: float
    status: WorkerStatus = WorkerStatus.ACTIVE
    instances: Dict[str, VLLMInstanceInfo] = field(default_factory=dict)
    public_worker_url: Optional[str] = None  # Worker 的公网访问 URL
    
    def to_dict(self) -> Dict:
        return {
            "worker_id": self.worker_id,
            "worker_url": self.worker_url,
            "gpu_info": self.gpu_info.to_dict(),
            "last_heartbeat": self.last_heartbeat,
            "status": self.status.value,
            "instances": {k: v.to_dict() for k, v in self.instances.items()},
            "public_worker_url": self.public_worker_url
        }


@dataclass
class ModelRouting:
    """模型路由信息"""
    alias: str
    model_name: str
    worker_id: str
    worker_url: str
    vllm_port: int
    created_at: float
    
    def to_dict(self) -> Dict:
        return {
            "alias": self.alias,
            "model_name": self.model_name,
            "worker_id": self.worker_id,
            "worker_url": self.worker_url,
            "vllm_port": self.vllm_port,
            "created_at": self.created_at
        }
