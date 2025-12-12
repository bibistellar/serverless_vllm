import ray
from ray import logger, serve
from wandb import config

from pool_manager import PoolManager
from router import VadLLMRouter

def main() -> None:
    """本地直接启动一个 Ray Serve HTTP 服务。"""

    # 优先连接已有 Ray 集群，失败则退回本地单机模式
    try:
        ray.init(address="auto")
    except Exception as exc:  # noqa: BLE001
        ray.logger.warning("ray.init(address='auto') failed (%s), fallback to local mode", exc)
        ray.init()
    serve.start(http_options={"host": "0.0.0.0", "port": 18000})

    # 若当前环境无可用 GPU，则退回假引擎，避免 vLLM 抛错
    resources = ray.available_resources()
    use_fake = resources.get("GPU", 0) < 0.01
    if use_fake:
        logger.warning("No GPUs detected by Ray; PoolManager will use FakeEngine")

    # 先创建 PoolManager actor（轻量），再显式初始化引擎，避免 __init__ 阶段超时
    pool = PoolManager.remote(
        # config_path="model_config.yaml",
        config_path=None,
        use_fake=use_fake,
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
        ray.logger.info("收到中断信号，正在关闭 Ray Serve...")
        serve.shutdown()
        ray.shutdown()


if __name__ == "__main__":
    main()
