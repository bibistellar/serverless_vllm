"""工具函数"""
import subprocess
import logging
import os
from typing import Optional, List

logger = logging.getLogger(__name__)


def _parse_visible_devices() -> Optional[List[int]]:
    env = os.getenv("CUDA_VISIBLE_DEVICES")
    if env is None:
        return None
    env = env.strip()
    if env == "":
        return None
    if env in {"-1", "none", "None"}:
        return []
    indices: List[int] = []
    for part in env.split(","):
        part = part.strip()
        if part.isdigit():
            indices.append(int(part))
    if not indices:
        return []
    return indices


def get_gpu_info():
    """获取 GPU 信息（仅返回 CUDA_VISIBLE_DEVICES 可见的卡）"""
    try:
        import pynvml
        pynvml.nvmlInit()
        
        gpu_count = pynvml.nvmlDeviceGetCount()
        visible_indices = _parse_visible_devices()
        if visible_indices is None:
            indices = list(range(gpu_count))
        else:
            indices = [i for i in visible_indices if 0 <= i < gpu_count]
        gpu_names = []
        total_memory = 0.0
        available_memory = 0.0
        utilization_sum = 0.0
        devices = []
        
        for i in indices:
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode('utf-8')
            gpu_names.append(name)
            
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_gb = mem_info.total / (1024**3)  # 转换为 GB
            free_gb = mem_info.free / (1024**3)
            total_memory += total_gb
            available_memory += free_gb
            util_rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
            utilization_sum += util_rates.gpu
            devices.append(
                {
                    "index": i,
                    "name": name,
                    "total_memory_gb": round(total_gb, 2),
                    "available_memory_gb": round(free_gb, 2),
                    "utilization": round(util_rates.gpu / 100.0, 4),
                }
            )
        
        pynvml.nvmlShutdown()
        
        from src.common.models import GPUInfo, GPUDeviceInfo
        utilization = 0.0
        visible_count = len(indices)
        if visible_count > 0:
            utilization = round(utilization_sum / visible_count / 100.0, 4)

        return GPUInfo(
            gpu_count=visible_count,
            gpu_names=gpu_names,
            total_memory_gb=round(total_memory, 2),
            available_memory_gb=round(available_memory, 2),
            utilization=utilization,
            devices=[
                GPUDeviceInfo(
                    index=device["index"],
                    name=device["name"],
                    total_memory_gb=device["total_memory_gb"],
                    available_memory_gb=device["available_memory_gb"],
                    utilization=device["utilization"],
                )
                for device in devices
            ],
        )
    except Exception as e:
        logger.error(f"Failed to get GPU info: {e}")
        from src.common.models import GPUInfo
        return GPUInfo(
            gpu_count=0,
            gpu_names=[],
            total_memory_gb=0.0,
            available_memory_gb=0.0,
            devices=[],
        )


def find_free_port(start_port: int = 8000, max_attempts: int = 100) -> Optional[int]:
    """查找可用端口"""
    import socket
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    return None


def check_port_in_use(port: int) -> bool:
    """检查端口是否被使用"""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('', port))
            return False
        except OSError:
            return True
