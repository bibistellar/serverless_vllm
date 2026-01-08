import ray
import os
from ray import logger, serve
from wandb import config
from pathlib import Path

from pool_manager import PoolManager
from router import VadLLMRouter

def main() -> None:
    """本地直接启动一个 Ray Serve HTTP 服务。"""

    # 优先连接已有 Ray 集群，失败则退回本地单机模式
    try:
        project_root = Path(__file__).resolve().parent
        ray.init(
            address="auto",
            namespace="serverless_vllm",
            runtime_env={
                "env_vars": {"HF_ENDPOINT": "https://hf-mirror.com"},
                "working_dir": str(project_root), 
                # 如需一起装依赖可加：
                # "pip": [],
            },
        )
    except Exception as exc:  # noqa: BLE001
        ray.logger.warning("ray.init(address='auto') failed (%s), fallback to local mode", exc)
        # 本地模式下也使用独立 namespace，避免与其他 Serve 服务共享默认空间。
        ray.init(namespace="serverless_vllm")
        serve.start(
            proxy_location="EveryNode", # 或者直接用字符串 "EveryNode"
            http_options={
                "host": "0.0.0.0",
                "port": 18000
            }
        )

    # 先创建 PoolManager actor（轻量），再显式初始化引擎，避免 __init__ 阶段超时。
    # 为了便于调试和从外部脚本获取 handle，这里为 actor 命名。
    pool = PoolManager.options(name="llm_pool_manager").remote(
        # config_path="model_config.yaml",
        config_path=None,
    )
    # 串行触发引擎初始化，确保完成后再对外提供服务
    init_stats = ray.get(pool.init_engines.remote())
    logger.info("PoolManager engines initialized: %s", init_stats)

    # 启动后台 watcher，周期性同步 Ray 集群与引擎副本（例如新节点加入时补齐）
    # 这里 fire-and-forget，不阻塞主线程
    pool.watch_cluster.remote(60.0)

    llm_pool = VadLLMRouter.bind(pool_handle=pool)
    serve.run(
        llm_pool,
        name="llm_pool",
        # 统一以 /v1 为前缀，内部再根据 path 做细分路由：
        # - /v1/chat/completions
        # - /v1/models/register
        route_prefix="/v1",
    )

    ray.logger.info(
        "LLM Pool Ray Serve 已启动，chat: http://0.0.0.0:18000/v1/chat/completions, "
        "register: http://0.0.0.0:18000/v1/models/register",
    )

    try:
        import time

        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        ray.logger.info("收到中断信号，正在关闭 llm_pool...")
        serve.delete("llm_pool") # 如果只想删这一个应用，用这个
        ray.shutdown()


if __name__ == "__main__":
    main()
