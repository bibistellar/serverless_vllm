from typing import Any, Dict, Optional

import ray
from ray import logger

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
        gpu_mem_util: float = 0.7,
        default_max_tokens: int = 128,
        default_temperature: float = 0.2,
        top_p_default: float = 0.9,
    ):
        if LLM is None or SamplingParams is None:
            raise RuntimeError(f"vLLM import failed: {_IMPORT_ERROR}")

        self.model_name = model_name
        self.model_repo = model_repo or model_name
        self.sleep_level = sleep_level
        self.max_model_len = max_model_len
        self.tensor_parallel_size = tensor_parallel_size
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
        )

        if self.sleep_level is not None and self.sleep_level > 0:
            self._sleep_back()

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

    def generate(self, req: Dict[str, Any]):
        """Generate text from a prompt.

        Args:
            req: {"prompt": str, "max_tokens": Optional[int], "temperature": Optional[float], "top_p": Optional[float]}
        Returns:
            {"text": str}
        """
        prompt = req.get("prompt", "") or ""
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
            outputs = self.llm.generate(
                prompts=[prompt],
                sampling_params=sp,
                use_tqdm=False,
            )
            text = outputs[0].outputs[0].text
        except Exception as exc:  # noqa: BLE001
            logger.warning("generate failed: %s", exc)
            text = f"[engine error] {exc}"  # keep router path alive
        finally:
            if self.sleep_level is not None and self.sleep_level > 0:
                self._sleep_back()

        return {"text": text}

    def health(self) -> Dict[str, Any]:
        return {"model": self.model_name, "sleeping": self.sleeping}
