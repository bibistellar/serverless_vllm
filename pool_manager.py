import asyncio
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import ray
from ray import logger
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

try:
    from llm_engine import LLMEngineWorker
except Exception as exc:  # noqa: BLE001
    LLMEngineWorker = None
    _LLM_ENGINE_IMPORT_ERROR = exc
else:
    _LLM_ENGINE_IMPORT_ERROR = None

try:
    import yaml
except Exception:  # noqa: BLE001
    yaml = None


@dataclass
class ModelConfig:
    name: str
    repo: Optional[str] = None
    max_model_len: Optional[int] = None
    tensor_parallel_size: Optional[int] = None
    gpu_memory_utilization: Optional[float] = None
    num_engines_per_node: int = 1
    default_max_tokens: int = 128
    default_temperature: float = 0.2

    @classmethod
    def from_dict(cls, name: str, data: Dict[str, Any]) -> "ModelConfig":
        return cls(
            name=name,
            repo=data.get("repo"),
            max_model_len=data.get("max_model_len"),
            tensor_parallel_size=data.get("tensor_parallel_size"),
            gpu_memory_utilization=data.get("gpu_memory_utilization"),
            num_engines_per_node=int(data.get("num_engines_per_node", 1)),
            default_max_tokens=int(
                data.get("default_max_tokens", data.get("max_tokens", 128))
            ),
            default_temperature=float(
                data.get("default_temperature", data.get("temperature", 0.2))
            ),
        )


@ray.remote
class FakeEngine:
    """A lightweight fake engine actor for early integration tests."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    def generate(self, req: Dict[str, Any]):
        prompt = req.get("prompt", "") or ""
        snippet = prompt[:200]
        suffix = "..." if snippet and len(prompt) > len(snippet) else ""
        return {"text": f"[fake engine {self.model_name}] {snippet}{suffix}"}

    def health(self) -> Dict[str, Any]:
        return {"model": self.model_name, "fake": True}

    def maybe_sleep(self):
        """与真实引擎保持接口一致的 no-op sleep 钩子。"""
        return None


@ray.remote
class PoolManager:
    """Manage engine actors per model; returns one for each request."""

    def __init__(
        self,
        config_path: Optional[str] = None,
        use_fake: bool = False,
        engine_options: Optional[Dict[str, Any]] = None,
    ):
        self.config_path = config_path
        self.use_fake = use_fake
        self.engine_options = engine_options or {}
        self.engines_by_model: Dict[str, List[Any]] = {}
        # 每个模型的轮询调度下标
        self._rr_index: Dict[str, int] = {}
        self.model_configs: Dict[str, ModelConfig] = {}
        # 记录已经为哪些 node_id 创建过引擎，便于后续增量同步
        self._initialized_node_ids: set[str] = set()
        self.engine_cls = self._choose_engine_cls(use_fake)
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

    def _cluster_gpu_count(self) -> float:
        try:
            return float(ray.cluster_resources().get("GPU", 0.0))
        except Exception:  # noqa: BLE001
            return 0.0

    def _choose_engine_cls(self, use_fake: bool):
        if use_fake:
            logger.info("PoolManager configured to use FakeEngine (user flag)")
            return FakeEngine
        if LLMEngineWorker is None:
            logger.warning(
                "LLMEngineWorker unavailable (%s); falling back to FakeEngine",
                _LLM_ENGINE_IMPORT_ERROR,
            )
            return FakeEngine

        if self._cluster_gpu_count() < 0.5:
            logger.warning(
                "No Ray GPU resources; PoolManager falling back to FakeEngine"
            )
            return FakeEngine

        return LLMEngineWorker

    def _default_configs(self) -> Dict[str, ModelConfig]:
        return {"test-model": ModelConfig(name="test-model")}

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

        if yaml is None:
            logger.warning("PyYAML not installed, falling back to default configs")
            self.model_configs = self._default_configs()
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            models_raw = data.get("models", data)
            self.model_configs = self._parse_models(models_raw)
            logger.info("PoolManager loaded configs for models: %s", list(self.model_configs))
        except FileNotFoundError:
            logger.warning("Config file %s not found, using default configs", path)
            self.model_configs = self._default_configs()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load config %s (%s), using default configs", path, exc)
            self.model_configs = self._default_configs()

    async def _init_engines_on_all_nodes_async(self) -> None:
        # 没有任何模型配置时，不做初始化，也不标记节点，
        # 以便后续 register_model 注入模型后，watch_cluster 能正确补齐。
        if not self.model_configs:
            logger.info("No model configs defined; skip engine initialization for now")
            return

        nodes = [n for n in ray.nodes() if n.get("Alive")]
        if not nodes:
            nodes = [{"NodeID": None}]

        for node in nodes:
            node_id = node.get("NodeID") or "__LOCAL__"
            # 已经为该节点初始化过，引擎则跳过（保持幂等），仅为新节点补齐
            if node_id in self._initialized_node_ids:
                continue
            for cfg in self.model_configs.values():
                for _ in range(cfg.num_engines_per_node):
                    options: Dict[str, Any] = dict(self.engine_options)
                    if node_id:
                        options["scheduling_strategy"] = NodeAffinitySchedulingStrategy(
                            node_id=node_id,
                            soft=True,
                        )
                    worker = self._spawn_engine(cfg, options)
                    # 等待当前引擎完成初始化后再创建下一个，串行化以减少资源争用。
                    try:
                        await worker.health.remote()
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Engine init check failed: %s", exc)
                        raise
                    self.engines_by_model.setdefault(cfg.name, []).append(worker)

            self._initialized_node_ids.add(node_id)

        logger.info(
            "Initialized engines: %s",
            {k: len(v) for k, v in self.engines_by_model.items()},
        )

    def _spawn_engine(self, cfg: ModelConfig, options: Dict[str, Any]):
        if self.engine_cls is FakeEngine:
            return self.engine_cls.options(**options).remote(cfg.name)

        # Real LLM engine
        # Allocate GPU for the engine actor; 默认占用 1 个 GPU 资源，由 Ray
        # 决定具体设备；gpu_mem_util 若未在配置中显式给出，则传 None，
        # 交由 LLMEngineWorker 根据当前 GPU 显存自动估算（以 16GiB 为目标）。
        options.setdefault("num_gpus", 1)
        gpu_mem_util = cfg.gpu_memory_utilization
        return self.engine_cls.options(**options).remote(
            model_name=cfg.name,
            model_repo=cfg.repo or cfg.name,
            sleep_level=2,
            max_model_len=cfg.max_model_len or 16000,
            tensor_parallel_size=cfg.tensor_parallel_size or 1,
            gpu_mem_util=gpu_mem_util,
            default_max_tokens=cfg.default_max_tokens,
            default_temperature=cfg.default_temperature,
            top_p_default=0.9,
        )

    def get_engine_for_request(self, model_name: str):
        engines = self.engines_by_model.get(model_name, [])
        if not engines:
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
        - tensor_parallel_size: 1
                - gpu_memory_utilization: 若模板中未设置则为 None，由 LLMEngineWorker
                    根据当前 GPU 显存自动估算（约 16GiB 预算）
        - num_engines_per_node: 1
        - default_max_tokens: 512
        - default_temperature: 0.2

        调用时会在所有节点上同步创建引擎。
        """
        if alias in self.model_configs:
            raise RuntimeError(f"Model alias already exists: {alias}")

        # 从已有配置里抽一份模板，避免硬编码过多 magic number
        template_cfg: Optional[ModelConfig] = None
        if self.model_configs:
            # 任意取一个现有模型作为模板（当前场景下都是 Qwen 系列）
            template_cfg = next(iter(self.model_configs.values()))

        max_model_len = template_cfg.max_model_len if template_cfg else 8192
        tensor_parallel_size = template_cfg.tensor_parallel_size or 1 if template_cfg else 1
        # 若模板中没有显式配置 gpu_memory_utilization，则保持为 None，交由
        # LLMEngineWorker 在每个 worker 进程内根据当前 GPU free/total 自动估算。
        if template_cfg and template_cfg.gpu_memory_utilization is not None:
            gpu_memory_utilization = template_cfg.gpu_memory_utilization
        else:
            gpu_memory_utilization = None
        num_engines_per_node = template_cfg.num_engines_per_node if template_cfg else 1
        default_max_tokens = template_cfg.default_max_tokens if template_cfg else 512
        default_temperature = template_cfg.default_temperature if template_cfg else 0.2

        cfg = ModelConfig(
            name=alias,
            repo=full_name,
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            num_engines_per_node=num_engines_per_node,
            default_max_tokens=default_max_tokens,
            default_temperature=default_temperature,
        )

        self.model_configs[alias] = cfg

        # 为新模型创建引擎（串行，防止资源抢占）
        nodes = [n for n in ray.nodes() if n.get("Alive")]
        if not nodes:
            nodes = [{"NodeID": None}]

        for node in nodes:
            raw_node_id = node.get("NodeID")
            node_id = raw_node_id or "__LOCAL__"
            for _ in range(cfg.num_engines_per_node):
                options: Dict[str, Any] = dict(self.engine_options)
                # 调度策略仍然使用 Ray 实际的 NodeID；占位节点不设置亲和性
                if raw_node_id:
                    options["scheduling_strategy"] = NodeAffinitySchedulingStrategy(
                        node_id=raw_node_id,
                        soft=True,
                    )
                worker = self._spawn_engine(cfg, options)
                try:
                    await worker.health.remote()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Engine init check failed: %s", exc)
                    raise
                self.engines_by_model.setdefault(alias, []).append(worker)

            # 标记该节点已为当前所有模型初始化过，引擎补齐由 watch_cluster
            # 针对“新节点”来处理，避免后续周期性检查重复创建引擎。
            self._initialized_node_ids.add(node_id)

        logger.info("Registered model %s (%s)", alias, full_name)
        return {
            "alias": alias,
            "full_name": full_name
        }