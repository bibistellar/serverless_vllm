import asyncio
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import ray
from ray import logger
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from llm_engine import NodeLLMEngine
import yaml


@dataclass
class ModelConfig:
    name: str
    repo: Optional[str] = None
    max_model_len: Optional[int] = None
    gpu_memory_size: Optional[str] = None

    @classmethod
    def from_dict(cls, name: str, data: Dict[str, Any]) -> "ModelConfig":
        return cls(
            name=name,
            repo=data.get("repo"),
            max_model_len=data.get("max_model_len"),
            gpu_memory_size=data.get("gpu_memory_size"),
        )

@ray.remote
class PoolManager:
    """Manage engine actors per model; returns one for each request."""

    def __init__(
        self,
        config_path: Optional[str] = None,
    ):
        self.config_path = config_path
        self.engines_by_model: Dict[str, List[Any]] = {}
        # 每个模型的轮询调度下标
        self._rr_index: Dict[str, int] = {}
        self.model_configs: Dict[str, ModelConfig] = {}
        # 每个 Ray node 上的 NodeLLMEngine 管理器
        self.node_engines: Dict[str, Any] = {}
        # 记录已经为哪些 node_id 创建过 NodeLLMEngine，便于后续增量同步
        self._initialized_node_ids: set[str] = set()
        # 这里只做轻量操作，真正的引擎初始化由显式方法触发
        self._load_config(config_path)

    async def init_engines(self) -> Dict[str, int]:
        """Lazily create all engines across nodes.

        在 Ray actor 创建完成后，由外部显式调用，避免 __init__ 阶段超时。
        返回各模型初始化成功的引擎数量统计。

        异步 actor 内部不得使用 blocking 的 ray.get，这里通过
        await health.remote() 等待引擎就绪。
        """
        await self._init_engines_on_all_nodes_async()
        return {k: len(v) for k, v in self.engines_by_model.items()}

    def _default_configs(self) -> Dict[str, ModelConfig]:
        return {"test-model": ModelConfig(name="test-model")}

    def _parse_gpu_memory_size(self, value: Optional[str]) -> Optional[float]:
        """将配置中的 gpu_memory_size 字符串解析为 GiB 浮点数。

        支持示例："10G", "10GiB", "10", "10240M" 等，无法解析时返回 None。
        """

        if not value:
            return None

        v = value.strip().lower()
        # 提取数字部分
        num_chars: List[str] = []
        for ch in v:
            if ch.isdigit() or ch == ".":
                num_chars.append(ch)
        if not num_chars:
            return None

        try:
            num = float("".join(num_chars))
        except Exception:  # noqa: BLE001
            return None

        # 带 m/M 但不带 g 时按 MB 处理
        if "m" in v and "g" not in v:
            return num / 1024.0
        # 其它情况一律视为 GiB
        return num

    def _parse_models(self, raw_models: Any) -> Dict[str, ModelConfig]:
        configs: Dict[str, ModelConfig] = {}

        if isinstance(raw_models, dict):
            iterable = raw_models.items()
        elif isinstance(raw_models, list):
            iterable: List[Any] = []
            for item in raw_models:
                if isinstance(item, dict) and item.get("name"):
                    iterable.append((item["name"], item))
        else:
            return self._default_configs()

        for name, cfg in iterable:
            if not name:
                continue
            if isinstance(cfg, ModelConfig):
                configs[name] = cfg
            elif isinstance(cfg, dict):
                configs[name] = ModelConfig.from_dict(name, cfg)

        return configs or self._default_configs()

    def _load_config(self, path: Optional[str]):
        # 没有提供配置文件路径时，不加载任何模型配置，保持空集合，
        # 由后续 register_model 或外部逻辑动态注入模型。
        if not path:
            self.model_configs = {}
            logger.info(
                "PoolManager started without config_path; no models registered yet",
            )
            return
        
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        models_raw = data.get("models", data)
        self.model_configs = self._parse_models(models_raw)
        logger.info("PoolManager loaded configs for models: %s", list(self.model_configs))

    async def _ensure_node_engines(self) -> None:
        """确保每个存活的 Ray node 上都有一个 NodeLLMEngine 实例。"""

        try:
            nodes = [n for n in ray.nodes() if n.get("Alive")]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to query ray.nodes(): %s", exc)
            return

        # 1) 计算当前存活节点的 ID 集合
        alive_node_ids = {
            n.get("NodeID")
            for n in nodes
            if n.get("NodeID")
        }

        # 2) 清理已下线节点对应的 NodeLLMEngine 记录
        stale_ids = [nid for nid in self.node_engines.keys() if nid not in alive_node_ids]
        for nid in stale_ids:
            handle = self.node_engines.pop(nid, None)
            logger.info("Removed NodeLLMEngine for dead node %s: %s", nid, handle)

        for node in nodes:
            node_id = node.get("NodeID")
            if not node_id or node_id in self.node_engines:
                continue

            strategy = NodeAffinitySchedulingStrategy(
                node_id=node_id,
                soft=False,
            )
            handle = NodeLLMEngine.options(
                scheduling_strategy=strategy,
                name=f"llm_node_engine_{node_id[:8]}",
            ).remote(node_id)

            try:
                # 简单健康检查，确保 actor 已成功创建
                await handle.health.remote()
            except Exception as exc:  # noqa: BLE001
                logger.warning("NodeLLMEngine health check failed on node %s: %s", node_id, exc)
                continue

            self.node_engines[node_id] = handle
            self._initialized_node_ids.add(node_id)
            logger.info("Created NodeLLMEngine on node %s: %s", node_id, handle)

    async def _init_engines_on_all_nodes_async(self) -> None:
        """初始化阶段仅确保 NodeLLMEngine 存在，不预创建 LLMEngineWorker。
        只调用 _ensure_node_engines()，保证每个节点上都有一个 NodeLLMEngine，
        具体的模型实例在 register_model 等扩展逻辑中按需创建。
        """
        # 真实 vLLM 模式：只负责确保每个节点上存在 NodeLLMEngine。
        await self._ensure_node_engines()
        logger.info(
            "NodeLLMEngine instances: %s",
            list(self.node_engines.keys()),
        )

    async def _tick_sleep_state_on_all_nodes(self) -> None:
        """周期性驱动所有节点上模型 worker 的休眠状态迁移。"""

        if not self.node_engines:
            return

        for nid, ne in self.node_engines.items():
            try:
                # fire-and-forget 调用，无需等待返回值
                ne.tick_sleep_state.remote()
            except Exception as exc:  # noqa: BLE001
                logger.warning("tick_sleep_state failed on node %s: %s", nid, exc)

    def get_engine_for_request(self, model_name: str):
        engines = self.engines_by_model.get(model_name, [])
        if not engines:
            logger.warning("No engines for model %s; maybe not registered yet", model_name)
            raise RuntimeError(f"No engines for model {model_name}")
        # 简单轮询调度：对于同一模型，在其引擎列表上依次取用，
        # 避免单点过载，也让 QPS 在多个 worker 之间更均匀。
        idx = self._rr_index.get(model_name, 0)
        pos = idx % len(engines)
        engine = engines[pos]
        self._rr_index[model_name] = (idx + 1) % len(engines)

        # 记录调度决策，便于排查多节点下的负载分布情况。
        # engine 本身是一个 ActorHandle，这里直接打印其 repr。 
        logger.info(
            "Dispatch model %s to engine[%d/%d]: %s",
            model_name,
            pos,
            len(engines),
            engine,
        )
        return engine

    def list_models(self) -> List[str]:
        return list(self.model_configs)

    async def watch_cluster(self, interval_s: float = 60.0) -> None:
        """周期性检查 Ray 集群状态，为新增节点补齐引擎。

        该方法可在外部以 fire-and-forget 方式调用：
        `pool.watch_cluster.remote(60)`，内部会在新节点出现时自动
        为其创建与已有节点相同规格的引擎副本。
        """
        while True:
            try:
                await self._init_engines_on_all_nodes_async()
                # 同步完节点后，顺便驱动各 worker 的休眠状态迁移
                await self._tick_sleep_state_on_all_nodes()
            except Exception as exc:  # noqa: BLE001
                logger.warning("watch_cluster sync failed: %s", exc)
            await asyncio.sleep(interval_s)

    async def register_model(self, alias: str, full_name: str) -> Dict[str, Any]:
        """Register a new model at runtime.

        参数：
        - alias: 对外使用的简写名，例如 "qwen3-vl-8b"
        - full_name: vLLM / HF 模型名，例如 "Qwen/Qwen3-VL-8B-Instruct"

        其余参数按照当前配置文件中的默认策略：
        - max_model_len: 沿用已有模型的 max_model_len（若有），否则 8192
        - gpu_memory_size: 若模板中未设置则为 None，由 LLMEngineWorker
            根据当前 GPU 显存自动估算（约 16GiB 预算）

        调用后会在可用节点上创建一个引擎实例，完成后即可通过
        PoolManager.get_engine_for_request() 获取到该模型的引擎。

        TODO:自动扩展逻辑
        """
        if alias in self.model_configs:
            raise RuntimeError(f"Model alias already exists: {alias}")

        # 从已有配置里抽一份模板，避免硬编码过多 magic number
        template_cfg: Optional[ModelConfig] = None
        if self.model_configs:
            # 任意取一个现有模型作为模板（当前场景下都是 Qwen 系列）
            template_cfg = next(iter(self.model_configs.values()))

        max_model_len = template_cfg.max_model_len if template_cfg else 8192
        # 若模板中没有显式配置 gpu_memory_size，则保持为 None，交由
        # LLMEngineWorker 在每个 worker 进程内根据当前 GPU free/total 自动估算。
        if template_cfg and template_cfg.gpu_memory_size is not None:
            gpu_memory_size = template_cfg.gpu_memory_size
        else:
            gpu_memory_size = None

        cfg = ModelConfig(
            name=alias,
            repo=full_name,
            max_model_len=max_model_len,
            gpu_memory_size=gpu_memory_size,
        )

        self.model_configs[alias] = cfg

        # 为新模型创建单个引擎（串行，防止资源抢占）。
        # 1) 先确保每个节点都有 NodeLLMEngine；
        # 2) 查询所有节点的 GPU 余量；
        # 3) 选择可用显存最多的节点创建该模型的 LLMEngineWorker。
        await self._ensure_node_engines()
        if not self.node_engines:
            raise RuntimeError("No NodeLLMEngine instances available for register_model")

        node_ids = list(self.node_engines.keys())
        stats_list = await asyncio.gather(
            *[self.node_engines[nid].get_gpu_stats.remote() for nid in node_ids],
            return_exceptions=True,
        )

        best_node_id: Optional[str] = None
        best_free_gb = -1.0
        for nid, stats in zip(node_ids, stats_list):
            if isinstance(stats, Exception):
                logger.warning("get_gpu_stats failed on node %s: %s", nid, stats)
                continue
            gpus = stats.get("gpus", []) or []
            total_free = 0.0
            for g in gpus:
                try:
                    total_free += float(g.get("free_gb", 0.0))
                except Exception:  # noqa: BLE001
                    continue
            if total_free > best_free_gb:
                best_free_gb = total_free
                best_node_id = nid

        if best_node_id is None:
            # 所有节点的显存信息不可用时，退回随机选择一个节点。
            best_node_id = random.choice(node_ids)
            logger.warning(
                "All get_gpu_stats failed; fallback to random node %s for model %s",
                best_node_id,
                alias,
            )

        node_engine = self.node_engines[best_node_id]
        try:
            worker = await node_engine.create_model_engine.remote(
                model_name=cfg.name,
                model_repo=cfg.repo or cfg.name,
                max_model_len=cfg.max_model_len or 8192,
                tensor_parallel_size=1,
                gpu_memory_size_gb=self._parse_gpu_memory_size(cfg.gpu_memory_size),
            )
            await worker.health.remote()
            # 注册完成后立即让新模型进入深度休眠，尽快释放显存；
            # 实际调用后将遵循基于空闲时间的 0->1->2 休眠迁移策略。
            try:
                worker.force_sleep.remote(level=2)
            except Exception as exc:  # noqa: BLE001
                logger.warning("engine force_sleep after register failed: %s", exc)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to create engine for model %s on node %s: %s",
                alias,
                best_node_id,
                exc,
            )
            raise

        self.engines_by_model.setdefault(alias, []).append(worker)

        logger.info("Registered model %s (%s)", alias, full_name)
        return {
            "alias": alias,
            "full_name": full_name
        }

    async def unregister_model(self, alias: str) -> Dict[str, Any]:
        """Unregister an existing model and free its resources.

        - 删除内存中的模型配置和调度索引；
        - 通知各个 NodeLLMEngine 关闭并销毁对应的 LLMEngineWorker。
        """
        if alias not in self.model_configs and alias not in self.engines_by_model:
            raise RuntimeError(f"Model alias not found: {alias}")

        cfg_existed = alias in self.model_configs
        # 删除本地配置与调度信息
        self.model_configs.pop(alias, None)
        self.engines_by_model.pop(alias, None)
        self._rr_index.pop(alias, None)

        # 尝试在所有已知节点上卸载该模型的 worker
        try:
            await self._ensure_node_engines()
        except Exception as exc:  # noqa: BLE001
            logger.warning("_ensure_node_engines failed before unregister_model(%s): %s", alias, exc)

        killed_total = 0
        if self.node_engines:
            results = await asyncio.gather(
                *[ne.unregister_model.remote(alias) for ne in self.node_engines.values()],
                return_exceptions=True,
            )
            for r in results:
                if isinstance(r, Exception):
                    logger.warning("Node unregister_model(%s) failed: %s", alias, r)
                elif isinstance(r, int):
                    killed_total += r

        logger.info(
            "Unregistered model %s, removed_config=%s, killed_workers=%d",
            alias,
            cfg_existed,
            killed_total,
        )

        return {
            "alias": alias,
            "removed_config": cfg_existed,
            "killed_workers": killed_total,
        }