"""工具函数"""
import subprocess
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def get_gpu_info():
    """获取 GPU 信息"""
    try:
        import pynvml
        pynvml.nvmlInit()
        
        gpu_count = pynvml.nvmlDeviceGetCount()
        gpu_names = []
        total_memory = 0.0
        available_memory = 0.0
        
        for i in range(gpu_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode('utf-8')
            gpu_names.append(name)
            
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_memory += mem_info.total / (1024**3)  # 转换为 GB
            available_memory += mem_info.free / (1024**3)
        
        pynvml.nvmlShutdown()
        
        from src.common.models import GPUInfo
        return GPUInfo(
            gpu_count=gpu_count,
            gpu_names=gpu_names,
            total_memory_gb=round(total_memory, 2),
            available_memory_gb=round(available_memory, 2)
        )
    except Exception as e:
        logger.error(f"Failed to get GPU info: {e}")
        from src.common.models import GPUInfo
        return GPUInfo(gpu_count=0, gpu_names=[], total_memory_gb=0.0, available_memory_gb=0.0)


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
