from typing import Any, Dict, List, Optional

from io import BytesIO
import base64
import os
import time

import ray
from ray import logger
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

try:
    from PIL import Image
except Exception:  # noqa: BLE001
    Image = None

# Qwen-VL 多模态处理所需的可选依赖
try:  # noqa: SIM105
    from transformers import AutoProcessor
    from qwen_vl_utils import process_vision_info
except Exception:  # noqa: BLE001
    AutoProcessor = None
    process_vision_info = None

try:  # Lazy import to allow environments without vLLM to fail fast with message
    from vllm import LLM, SamplingParams
except Exception as exc:  # noqa: BLE001
    LLM = None
    SamplingParams = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@ray.remote
class LLMEngineWorker:
    """vLLM-backed engine worker with sleep / wake lifecycle."""

    def __init__(
        self,
        model_name: str,
        model_repo: Optional[str] = None,
        sleep_level: int = 2,
        max_model_len: int = 8192,
        tensor_parallel_size: int = 1,
        gpu_mem_util: Optional[float] = None,
        default_max_tokens: int = 128,
        default_temperature: float = 0.2,
        top_p_default: float = 0.9,
    ):
        # 默认使用 HuggingFace 离线模式：假定模型权重已存在本地缓存，
        # 避免在初始化时访问网络检查 md5 / etag。
        # os.environ.setdefault("HF_HUB_OFFLINE", "1")
        # os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

        if LLM is None or SamplingParams is None:
            raise RuntimeError(f"vLLM import failed: {_IMPORT_ERROR}")

        self.model_name = model_name
        self.model_repo = model_repo or model_name
        self.sleep_level = sleep_level
        self.max_model_len = max_model_len
        self.tensor_parallel_size = tensor_parallel_size

        # 若未显式配置 gpu_mem_util，则根据当前 GPU 的总显存和可用显存，
        # 以 16GiB 为目标预算自动推算一个合理的 gpu_memory_utilization。
        if gpu_mem_util is None or gpu_mem_util <= 0:
            self.gpu_mem_util = self._auto_gpu_mem_util(target_gb=10.0)
        else:
            self.gpu_mem_util = gpu_mem_util
        self.default_max_tokens = default_max_tokens
        self.default_temperature = default_temperature
        self.top_p_default = top_p_default

        # 当前休眠状态：0=活跃；1=轻度休眠(level=1)；2=深度休眠(level=2)
        self.sleeping = False
        self._current_sleep_level: int = 0
        # 记录最近一次完成推理的时间戳，用于按需进入 sleep1/sleep2
        self._last_used_ts: float = time.time()
        # 懒加载 Qwen-VL 的 Processor
        self._qwen_vl_processor = None

        self.llm = LLM(
            model=self.model_repo,
            trust_remote_code=True,
            enable_sleep_mode=True,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_mem_util,
            max_model_len=self.max_model_len,
            load_format="runai_streamer", 
            model_loader_extra_config={"concurrency": 16},  # 试 8/16/32
        )

        # 初始化阶段不再自动进入 sleep，由上层组件决定何时触发

    def _wake_if_needed(self):
        """根据当前 sleep_level 选择合适的唤醒路径。

        - level 1: 对应 vLLM `sleep(level=1)`，只需一次 `llm.wake_up()`；
        - level 2: 对应 vLLM `sleep(level=2)`，需要先恢复权重、reload，再恢复 KV。
        """
        if not self.sleeping and self._current_sleep_level <= 0:
            return

        try:
            if self._current_sleep_level == 1:
                # Sleep level 1: offload weights to CPU, discard KV cache
                self.llm.wake_up()
            else:
                # Sleep level 2: discard both weights and KV cache
                self.llm.wake_up(tags=["weights"])
                try:
                    self.llm.collective_rpc("reload_weights")
                except Exception:  # noqa: BLE001
                    pass
                self.llm.wake_up(tags=["kv_cache"])

            self.sleeping = False
            self._current_sleep_level = 0
            self._last_used_ts = time.time()
        except Exception as exc:  # noqa: BLE001
            logger.warning("wake_up failed: %s", exc)
            self.sleeping = False
            self._current_sleep_level = 0

    def _auto_gpu_mem_util(self, target_gb: float = 16.0, safety: float = 0.9) -> float:
        """根据当前 GPU 显存情况自动估算 gpu_memory_utilization。

        vLLM 内部使用: budget = total_mem * gpu_memory_utilization，
        并要求 budget <= free_mem。这里的策略是：

        - 目标希望大约预留 target_gb (默认 16GiB) 给模型+KV cache；
        - 同时不超过当前可用显存的 safety 比例（默认 90%）。

        因此：
        - total_gb = 总显存；free_gb = 当前可用显存；
        - budget_gb = min(target_gb, free_gb * safety)；
        - gpu_memory_utilization = budget_gb / total_gb。
        """
        try:
            import torch

            if not torch.cuda.is_available():
                logger.warning(
                    "CUDA not available in auto gpu_mem_util; fallback to 0.7",
                )
                return 0.7

            free_bytes, total_bytes = torch.cuda.mem_get_info()
            total_gb = total_bytes / (1024**3)
            free_gb = free_bytes / (1024**3)

            if total_gb <= 0 or free_gb <= 0:
                logger.warning(
                    "Invalid GPU memory info (total=%.3f, free=%.3f), fallback 0.7",
                    total_gb,
                    free_gb,
                )
                return 0.7

            budget_gb = min(target_gb, free_gb * safety)
            util = budget_gb / total_gb
            # 做一个简单的 clamp，避免取值过小或过大
            util = max(0.1, min(util, 0.95))

            logger.info(
                "Auto gpu_memory_utilization=%.3f (target=%.1fGiB, free=%.1fGiB, total=%.1fGiB)",
                util,
                target_gb,
                free_gb,
                total_gb,
            )
            return util
        except Exception as exc:  # noqa: BLE001
            logger.warning("auto gpu_mem_util failed (%s); fallback to 0.7", exc)
            return 0.7

    def _prepare_qwen_vl_inputs(self, messages: Any):
        """将 OpenAI 风格 messages 解析为 Qwen-VL vLLM 所需 inputs 结构。

        前提：
        - 模型为 Qwen3-VL 系列（model_repo 名称中包含 qwen 和 vl）；
        - 已安装 transformers + qwen_vl_utils + PIL。

        行为：
        - 解析 messages 中的 image_url（data:image;base64 等），解码为 PIL.Image；
        - 构造 Qwen-VL 期望的 messages 结构（type=image, image=PIL.Image）；
        - 调用 AutoProcessor.apply_chat_template 和 process_vision_info
          得到 prompt、multi_modal_data 和 mm_processor_kwargs；
        - 返回形如 [{"prompt", "multi_modal_data", "mm_processor_kwargs"}] 的列表；
        - 若任一步失败，返回 None，调用方应回退到纯文本路径。
        """

        if AutoProcessor is None or process_vision_info is None or Image is None:
            return None

        repo_lower = (self.model_repo or "").lower()
        if "qwen" not in repo_lower or "vl" not in repo_lower:
            return None

        # 懒加载 Processor
        if self._qwen_vl_processor is None:
            try:
                self._qwen_vl_processor = AutoProcessor.from_pretrained(self.model_repo)
            except Exception as exc:  # noqa: BLE001
                logger.warning("AutoProcessor.from_pretrained failed: %s", exc)
                return None

        processor = self._qwen_vl_processor
        qwen_messages = []
        for m in messages or []:
            role = m.get("role", "user")
            content = m.get("content")
            blocks: list[Any] = []

            if isinstance(content, str):
                blocks.append({"type": "text", "text": content})
            elif isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    btype = block.get("type", "text")
                    if btype == "image_url":
                        url_info = block.get("image_url") or {}
                        url = url_info.get("url")
                        if not isinstance(url, str):
                            continue
                        # data:image;base64,... -> 解码为 PIL.Image
                        if url.startswith("data:image") and ";base64," in url:
                            try:
                                _, b64 = url.split(",", 1)
                                raw = base64.b64decode(b64)
                                img = Image.open(BytesIO(raw)).convert("RGB")
                                blocks.append({"type": "image", "image": img})
                            except Exception as exc:  # noqa: BLE001
                                logger.warning("failed to decode image_url in Qwen-VL path: %s", exc)
                        else:
                            # 其它 URL / 路径：直接按字符串传递给 Qwen-VL 工具
                            blocks.append({"type": "image", "image": url})
                    elif btype == "image":
                        # 已经是 image 块，直接保留
                        blocks.append(block)
                    else:
                        # 文本等其它块原样保留
                        blocks.append(block)

            if blocks:
                qwen_messages.append({"role": role, "content": blocks})

        if not qwen_messages:
            return None

        try:
            text = processor.apply_chat_template(
                qwen_messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            image_inputs, video_inputs, video_kwargs = process_vision_info(
                qwen_messages,
                image_patch_size=processor.image_processor.patch_size,
                return_video_kwargs=True,
                return_video_metadata=True,
            )

            mm_data: Dict[str, Any] = {}
            if image_inputs is not None:
                mm_data["image"] = image_inputs
            if video_inputs is not None:
                mm_data["video"] = video_inputs

            inputs = [
                {
                    "prompt": text,
                    "multi_modal_data": mm_data,
                    "mm_processor_kwargs": video_kwargs,
                },
            ]
            return inputs
        except Exception as exc:  # noqa: BLE001
            logger.warning("prepare Qwen-VL inputs failed: %s", exc)
            return None

    def generate(self, req: Dict[str, Any]):
        """Generate text from either a raw prompt or chat messages.

        Args:
            req: {
              "prompt": str,
                            "messages": Optional[List[Dict[str, Any]]],
              "max_tokens": Optional[int],
              "temperature": Optional[float],
              "top_p": Optional[float]
            }
        Returns:
            {"text": str}
        """
        messages = req.get("messages")
        prompt = req.get("prompt", "") or ""
        vl_inputs = None

        # 若上游传入了 messages，优先尝试走 Qwen-VL 多模态路径；
        # 若失败则回退到原来的 chat_template 纯文本逻辑。
        if messages:
            vl_inputs = self._prepare_qwen_vl_inputs(messages)
            if vl_inputs is None:
                try:
                    tokenizer = self.llm.get_tokenizer()
                    prompt = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "apply_chat_template failed (%s); fallback to plain prompt",
                        exc,
                    )

        max_tokens = req.get("max_tokens") or self.default_max_tokens
        temperature = req.get("temperature") or self.default_temperature
        top_p = req.get("top_p") or self.top_p_default

        # 若当前处于休眠状态，则先唤醒
        self._wake_if_needed()

        text = ""
        try:
            sp = SamplingParams(
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )

            if vl_inputs is not None:
                # Qwen-VL 多模态路径：直接传 inputs 列表
                outputs = self.llm.generate(vl_inputs, sampling_params=sp)
            else:
                # 纯文本路径：仅传 prompt
                outputs = self.llm.generate(
                    prompts=[prompt],
                    sampling_params=sp,
                    use_tqdm=False,
                )

            text = outputs[0].outputs[0].text
        except Exception as exc:  # noqa: BLE001
            logger.warning("generate failed: %s", exc)
            text = f"[engine error] {exc}"  # keep router path alive

        # 记录最近一次完成推理的时间，用于空闲调度进入 sleep1/sleep2
        self._last_used_ts = time.time()
        self._current_sleep_level = 0
        return {"text": text}

    def health(self) -> Dict[str, Any]:
        return {
            "model": self.model_name,
            "sleeping": self.sleeping,
            "sleep_level": self._current_sleep_level,
        }

    def eval_sleep_state(self, now_ts: Optional[float] = None) -> int:
        """根据空闲时间在 0/1/2 级休眠之间迁移。

        - t < 60s: 保持活跃（level=0）；
        - 60s <= t < 120s: 进入 / 保持 level=1；
        - t >= 120s: 进入 / 保持 level=2。

        返回当前 sleep_level，便于上层调试或观测。
        """

        if now_ts is None:
            now_ts = time.time()

        idle = max(0.0, now_ts - self._last_used_ts)

        try:
            # 目标 level 由 idle 时长决定
            if idle < 60.0:
                target_level = 0
            elif idle < 120.0:
                target_level = 1
            else:
                target_level = 2

            # 若已处于目标 level，则无需额外操作
            if target_level == self._current_sleep_level:
                return self._current_sleep_level

            # 从更深的休眠往更浅迁移仅在有请求时通过 _wake_if_needed 完成，
            # 因此这里仅处理 0->1->2 的单向迁移。
            if target_level == 0:
                return self._current_sleep_level

            # 执行实际的 sleep 调用
            self.llm.sleep(level=target_level)
            self.sleeping = True
            self._current_sleep_level = target_level
            return self._current_sleep_level
        except Exception as exc:  # noqa: BLE001
            logger.warning("eval_sleep_state sleep transition failed: %s", exc)
            # 出错时保持当前状态
            return self._current_sleep_level

    def force_sleep(self, level: int = 2) -> int:
        """强制立即进入指定休眠等级，供注册后初始释放显存等场景使用。"""

        try:
            level_int = int(level)
            if level_int <= 0:
                # 非正数视为不休眠
                self.sleeping = False
                self._current_sleep_level = 0
                return self._current_sleep_level

            self.llm.sleep(level=level_int)
            self.sleeping = True
            self._current_sleep_level = level_int
            return self._current_sleep_level
        except Exception as exc:  # noqa: BLE001
            logger.warning("force_sleep(level=%s) failed: %s", level, exc)
            return self._current_sleep_level


@ray.remote
class NodeLLMEngine:
    """Per-node engine manager.

    每个 Ray node 上只运行一个 NodeLLMEngine actor，用来：

    - 查询该节点上的 GPU 资源信息，供 PoolManager 做调度决策；
    - 在本节点上创建并管理下属的 LLMEngineWorker（vLLM 实例）。

    注意：真正的推理仍然由 LLMEngineWorker 完成，NodeLLMEngine 只负责
    资源管理与子 actor 的生命周期管理。
    """

    def __init__(self, node_id: str):
        self.node_id = node_id
        # model_name -> List[ActorHandle]
        self._model_workers: Dict[str, List[Any]] = {}

    def health(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "models": list(self._model_workers),
        }

    def get_gpu_stats(self) -> Dict[str, Any]:
        """返回当前节点的 GPU 显存信息，供 PoolManager 选择扩容节点。

        结构示例：
        {
          "node_id": "...",
          "gpus": [
            {"index": 0, "free_gb": 10.5, "total_gb": 15.9},
            ...
          ]
        }
        """

        try:  # noqa: SIM105
            import torch

            if not torch.cuda.is_available():
                return {"node_id": self.node_id, "gpus": []}

            num_devices = torch.cuda.device_count()
            gpus: List[Dict[str, float]] = []
            for idx in range(num_devices):
                try:
                    torch.cuda.set_device(idx)
                    free_bytes, total_bytes = torch.cuda.mem_get_info()
                    gpus.append(
                        {
                            "index": idx,
                            "free_gb": free_bytes / (1024**3),
                            "total_gb": total_bytes / (1024**3),
                        },
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "NodeLLMEngine.get_gpu_stats failed on device %d: %s",
                        idx,
                        exc,
                    )

            return {"node_id": self.node_id, "gpus": gpus}
        except Exception as exc:  # noqa: BLE001
            logger.warning("NodeLLMEngine.get_gpu_stats failed: %s", exc)
            return {"node_id": self.node_id, "gpus": []}

    def list_models(self) -> List[str]:
        return list(self._model_workers)

    def get_model_workers(self, model_name: str) -> List[Any]:
        return self._model_workers.get(model_name, [])

    def tick_sleep_state(self) -> Dict[str, List[int]]:
        """由 PoolManager 周期性调用，驱动所有 worker 的休眠状态迁移。

        返回结构示例：
        {
          "qwen3-vl-2b": [0, 1],  # 各 worker 当前 sleep_level
          ...
        }
        以便调试和观测（调用方可忽略该返回值）。
        """

        now_ts = time.time()
        levels: Dict[str, List[int]] = {}
        for model_name, workers in self._model_workers.items():
            cur_levels: List[int] = []
            for w in workers:
                try:
                    lvl = ray.get(w.eval_sleep_state.remote(now_ts))
                    if isinstance(lvl, int):
                        cur_levels.append(lvl)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "NodeLLMEngine[%s] tick_sleep_state for %s failed: %s",
                        self.node_id,
                        model_name,
                        exc,
                    )
            if cur_levels:
                levels[model_name] = cur_levels
        return levels

    def unregister_model(self, model_name: str) -> int:
        """卸载指定模型并释放本节点上的相关 Worker 资源。

        返回实际被销毁的 worker 数量。
        """
        workers = self._model_workers.pop(model_name, [])
        removed = 0
        for w in workers:
            try:
                ray.kill(w, no_restart=True)
                removed += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "NodeLLMEngine[%s] failed to kill worker for model %s: %s",
                    self.node_id,
                    model_name,
                    exc,
                )

        if workers:
            logger.info(
                "NodeLLMEngine[%s] unregistered model %s, killed %d workers",
                self.node_id,
                model_name,
                removed,
            )
        return removed

    def create_model_engine(
        self,
        model_name: str,
        model_repo: Optional[str] = None,
        max_model_len: int = 8192,
        tensor_parallel_size: int = 1,
        gpu_memory_size_gb: Optional[float] = None,
        default_max_tokens: int = 128,
        default_temperature: float = 0.2,
        top_p_default: float = 0.9,
    ):
        """在当前节点上为指定模型创建（或返回已有的）LLMEngineWorker。

        若该模型在本节点上已存在 worker，则直接返回第一个；
        否则使用 NodeAffinitySchedulingStrategy 将新的 worker 固定调度
        到当前 node 上，并记录在本地表中。
        """

        # 已存在则直接复用，保持幂等
        existing = self._model_workers.get(model_name)
        if existing:
            return existing[0]

        # 根据目标显存预算计算 gpu_memory_utilization；
        # 若未指定目标显存，则交由 LLMEngineWorker 自行估算。
        gpu_mem_util: Optional[float] = None
        if gpu_memory_size_gb is not None and gpu_memory_size_gb > 0:
            stats = self.get_gpu_stats()
            gpus = stats.get("gpus", []) or []
            if gpus:
                # 简单地选择第一个 GPU 的信息来估算占用比例。
                g0 = gpus[0]
                try:
                    free_gb = float(g0.get("free_gb", 0.0))
                    total_gb = float(g0.get("total_gb", 0.0))
                    if total_gb > 0:
                        budget_gb = min(gpu_memory_size_gb, free_gb * 0.9)
                        util = budget_gb / total_gb
                        gpu_mem_util = max(0.1, min(util, 0.95))
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "NodeLLMEngine[%s] failed to compute gpu_mem_util: %s",
                        self.node_id,
                        exc,
                    )

        options: Dict[str, Any] = {
            "num_gpus": 1,
            "scheduling_strategy": NodeAffinitySchedulingStrategy(
                node_id=self.node_id,
                soft=False,
            ),
        }

        worker = LLMEngineWorker.options(**options).remote(
            model_name=model_name,
            model_repo=model_repo or model_name,
            sleep_level=2,
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size,
            gpu_mem_util=gpu_mem_util,
            default_max_tokens=default_max_tokens,
            default_temperature=default_temperature,
            top_p_default=top_p_default,
        )

        self._model_workers.setdefault(model_name, []).append(worker)
        logger.info(
            "NodeLLMEngine[%s] created worker for model %s: %s",
            self.node_id,
            model_name,
            worker,
        )
        return worker
