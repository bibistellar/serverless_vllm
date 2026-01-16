#!/usr/bin/env python3
"""
启动 Manager + Router 在 Ray Cluster 上

这个脚本会：
1. 初始化或连接到 Ray cluster
2. 启动 Manager 作为 Ray Actor
3. 启动 Router 作为 Ray Serve Deployment
"""
import ray
from ray import serve
import logging
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.manager.service import ManagerService
from src.router.service import RouterService

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def cleanup_existing_services():
    """清理已存在的服务（在 Ray 已初始化的情况下）"""
    logger.info("Checking for existing services...")
    
    try:
        # 尝试获取现有的 Manager Actor
        try:
            existing_manager = ray.get_actor("manager")
            logger.warning("Found existing Manager Actor, killing it...")
            ray.kill(existing_manager)
            import time
            time.sleep(2)  # 等待清理完成
            logger.info("Existing Manager Actor killed")
        except ValueError:
            logger.info("No existing Manager Actor found")
        
        # 检查并清理 Serve deployment
        try:
            from ray.serve._private.api import _get_global_client
            client = _get_global_client()
            if client is not None:
                logger.warning("Found existing Ray Serve deployments, shutting down...")
                serve.shutdown()
                import time
                time.sleep(2)
                logger.info("Existing Serve deployments shut down")
        except Exception as e:
            logger.info(f"No existing Serve deployment: {e}")
            
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")


def main():
    """启动 Manager + Router 服务"""
    
    # 1. 初始化 Ray
    logger.info("Initializing Ray...")
    
    # 设置 runtime_env，将当前目录作为 working_dir
    current_dir = os.path.dirname(os.path.abspath(__file__))
    runtime_env = {
        "working_dir": current_dir,
        "pip": ["httpx", "fastapi", "pydantic"]  # 确保依赖包可用
    }
    
    try:
        # 尝试连接现有 cluster
        ray.init(
            address="auto", 
            namespace="serverless_vllm",
            runtime_env=runtime_env
        )
        logger.info("Connected to existing Ray cluster")
    except Exception:
        # 如果失败，启动本地 cluster
        ray.init(
            namespace="serverless_vllm",
            runtime_env=runtime_env
        )
        logger.info("Started local Ray cluster")
    
    # 2. 清理已存在的服务（在 Ray 已连接后）
    cleanup_existing_services()
    
    # 3. 启动 Ray Serve
    logger.info("Starting Ray Serve...")
    serve.start(
        http_options={
            "host": "0.0.0.0",
            "port": 18000,
            "location": "EveryNode"
        }
    )
    
    # 4. 启动 Manager Actor
    logger.info("Starting Manager Actor...")
    manager = ManagerService.options(
        name="manager",
        lifetime="detached",  # 即使脚本退出也保持运行
        max_concurrency=100   # 支持并发调用
    ).remote(heartbeat_timeout=60)
    
    logger.info("Manager Actor started with name 'manager'")
    
    # 4. 启动 Router Deployment
    logger.info("Starting Router Deployment...")
    router_app = RouterService.bind(manager_actor_name="manager")
    
    serve.run(
        router_app,
        name="router",
        route_prefix="/"
    )
    
    logger.info("=" * 60)
    logger.info("✅ Serverless vLLM services started successfully!")
    logger.info("=" * 60)
    logger.info("Router API: http://0.0.0.0:18000")
    logger.info("")
    logger.info("Available endpoints:")
    logger.info("  - Health: GET http://0.0.0.0:18000/health")
    logger.info("  - Chat: POST http://0.0.0.0:18000/v1/chat/completions")
    logger.info("  - Models: GET http://0.0.0.0:18000/v1/models")
    logger.info("  - Register Model: POST http://0.0.0.0:18000/v1/models/register")
    logger.info("  - Workers: GET http://0.0.0.0:18000/workers")
    logger.info("")
    logger.info("Worker registration endpoint:")
    logger.info("  - Register Worker: POST http://0.0.0.0:18000/workers/register")
    logger.info("  - Worker Heartbeat: POST http://0.0.0.0:18000/workers/{worker_id}/heartbeat")
    logger.info("=" * 60)
    logger.info("Press Ctrl+C to stop...")
    
    # 保持运行
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("\nShutting down...")
        serve.shutdown()
        ray.shutdown()
        logger.info("Services stopped")


if __name__ == "__main__":
    main()
