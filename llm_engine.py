from typing import Any, Dict, Optional

from io import BytesIO
import base64
import os

import ray
from ray import logger

try:
    from PIL import Image
except Exception:  # noqa: BLE001
    Image = None

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
        max_model_len: int = 16000,
        tensor_parallel_size: int = 1,
        gpu_mem_util: Optional[float] = None,
        default_max_tokens: int = 128,
        default_temperature: float = 0.2,
        top_p_default: float = 0.9,
    ):
        # 默认使用 HuggingFace 离线模式：假定模型权重已存在本地缓存，
        # 避免在初始化时访问网络检查 md5 / etag。
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
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
            self.gpu_mem_util = self._auto_gpu_mem_util(target_gb=16.0)
        else:
            self.gpu_mem_util = gpu_mem_util
        self.default_max_tokens = default_max_tokens
        self.default_temperature = default_temperature
        self.top_p_default = top_p_default

        self.sleeping = False

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
        if not self.sleeping:
            return
        try:
            self.llm.wake_up(tags=["weights"])
            try:
                self.llm.collective_rpc("reload_weights")
            except Exception:  # noqa: BLE001
                pass
            self.llm.wake_up(tags=["kv_cache"])
            self.sleeping = False
        except Exception as exc:  # noqa: BLE001
            logger.warning("wake_up failed: %s", exc)
            self.sleeping = False

    def _sleep_back(self):
        try:
            self.llm.sleep(level=self.sleep_level)
            self.sleeping = True
        except Exception as exc:  # noqa: BLE001
            logger.warning("sleep failed: %s", exc)
            self.sleeping = False

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
        images_batch = None

        # 若上游传入了 messages，并且模型 tokenizer 支持 chat_template，
        # 尝试用 HF 的 apply_chat_template 来构造真正的 prompt，以获得
        # 与 vLLM 官方 API server 一致的行为。同时解析其中的 image_url，
        # 准备好传递给 vLLM 的 images 参数，用于多模态模型（如 Qwen-VL）。
        if messages:
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

            # 解析 messages 中的图片，支持 OpenAI 风格的 data:image/jpeg;base64,...
            # 只构建单批次 images_batch，对应单条 prompt。
            images: list[Any] = []
            if Image is None:
                logger.warning(
                    "PIL not available; ignore image_url blocks in messages",
                )
            else:
                for m in messages:
                    content = m.get("content")
                    if not isinstance(content, list):
                        continue
                    for block in content:
                        if not isinstance(block, dict):
                            continue
                        if block.get("type") != "image_url":
                            continue
                        url_info = block.get("image_url") or {}
                        url = url_info.get("url")
                        if not isinstance(url, str):
                            continue
                        # 仅处理 data:image/...;base64, 前缀，HTTP(S) URL 可按需扩展
                        if url.startswith("data:image") and ";base64," in url:
                            try:
                                header, b64 = url.split(",", 1)
                                raw = base64.b64decode(b64)
                                img = Image.open(BytesIO(raw)).convert("RGB")
                                images.append(img)
                            except Exception as exc:  # noqa: BLE001
                                logger.warning("failed to decode image_url: %s", exc)

            if images:
                images_batch = [images]

        max_tokens = req.get("max_tokens") or self.default_max_tokens
        temperature = req.get("temperature") or self.default_temperature
        top_p = req.get("top_p") or self.top_p_default

        self._wake_if_needed()

        text = ""
        try:
            sp = SamplingParams(
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )
            generate_kwargs: Dict[str, Any] = {
                "prompts": [prompt],
                "sampling_params": sp,
                "use_tqdm": False,
            }
            # 仅在解析到有效图片且底层 LLM 支持 vision 时传入 images 参数，
            # 以便对接 vLLM 的多模态推理接口。
            if images_batch is not None:
                generate_kwargs["images"] = images_batch

            outputs = self.llm.generate(**generate_kwargs)
            text = outputs[0].outputs[0].text
        except Exception as exc:  # noqa: BLE001
            logger.warning("generate failed: %s", exc)
            text = f"[engine error] {exc}"  # keep router path alive

        return {"text": text}

    def maybe_sleep(self):
        """由上层显式调用的 sleep 钩子。

        Router / PoolManager 在本次推理完成、结果返回之后，可以
        以 fire-and-forget 方式调用 `engine.maybe_sleep.remote()`；
        该调用会在 actor 内同步执行 sleep，阻塞该 actor，
        由上层通过调用频率来控制是否接受这部分开销。
        """
        if self.sleep_level is not None and self.sleep_level > 0:
            self._sleep_back()

    def health(self) -> Dict[str, Any]:
        return {"model": self.model_name, "sleeping": self.sleeping}
