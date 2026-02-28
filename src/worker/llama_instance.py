"""llama.cpp instance manager."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Optional

from ..common.models import InstanceStatus, VLLMInstanceInfo
from .message_adapters import BACKEND_LLAMA_CPP, convert_messages_for_backend

logger = logging.getLogger(__name__)

class FakeLlamaInstance:
    def __init__(self, response: str, delay_s: float, capacity: int) -> None:
        self.response = response
        self.delay_s = delay_s
        self.capacity = max(1, capacity)
        self.semaphore = asyncio.Semaphore(self.capacity)


class _FakeCompletionOutput:
    def __init__(self, text: str, finish_reason: Optional[str] = None) -> None:
        self.text = text
        self.finish_reason = finish_reason


class _FakeEngineOutput:
    def __init__(self, outputs: list[_FakeCompletionOutput]) -> None:
        self.outputs = outputs


@dataclass
class LlamaRuntime:
    llm: Any
    capacity: int
    semaphore: asyncio.Semaphore


class LlamaManager:
    """llama.cpp manager with real startup and generation support."""

    def __init__(self) -> None:
        self.instances: Dict[str, VLLMInstanceInfo] = {}
        self.fake_instances: Dict[str, FakeLlamaInstance] = {}
        self.runtimes: Dict[str, LlamaRuntime] = {}
        self._inflight_requests: Dict[str, int] = {}
        self._latency_metrics: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _default_metrics() -> Dict[str, Any]:
        return {
            "count": 0,
            "ttft_last": None,
            "ttft_avg": None,
            "e2e_last": None,
            "e2e_avg": None,
        }

    def _ensure_metrics(self, alias: str) -> Dict[str, Any]:
        metrics = self._latency_metrics.setdefault(alias, self._default_metrics())
        for key, default in self._default_metrics().items():
            metrics.setdefault(key, default)
        return metrics

    @staticmethod
    def _to_metric_float(value: Any) -> float:
        if value is None:
            return 0.0
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _init_metrics(self, alias: str) -> None:
        self._inflight_requests[alias] = 0
        self._latency_metrics[alias] = self._default_metrics()

    def _select_base_model(self, base_model: Optional[str], instance: VLLMInstanceInfo) -> str:
        return base_model or instance.model_name

    def _extract_text(self, output: Dict[str, Any]) -> str:
        choices = output.get("choices") or []
        if not choices:
            return ""
        choice = choices[0]
        message = choice.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content
        text = choice.get("text")
        if isinstance(text, str):
            return text
        return ""

    def _extract_stream_delta(self, chunk: Dict[str, Any]) -> tuple[str, Optional[str]]:
        """Extract incremental text and finish_reason from a llama.cpp streaming chunk.

        We keep the manager-level contract consistent with vLLM-style streaming used by
        `stream_chat_completion`: completion_output.text should be the cumulative text so far.
        """
        choices = chunk.get("choices") or []
        if not choices:
            return "", None
        choice = choices[0] if isinstance(choices[0], dict) else {}
        finish_reason = choice.get("finish_reason")

        # OpenAI-style chat streaming: {"delta": {"content": "..."}}
        delta = choice.get("delta")
        if isinstance(delta, dict):
            content = delta.get("content")
            if isinstance(content, str) and content:
                return content, finish_reason

        # Some implementations may use message/content.
        message = choice.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str) and content:
                return content, finish_reason

        # Fallback to plain text field.
        text = choice.get("text")
        if isinstance(text, str) and text:
            return text, finish_reason

        return "", finish_reason

    def _load_llm_sync(
        self,
        *,
        model_name: str,
        model_path: Optional[str],
        llama_filename: Optional[str],
        llama_mmproj_path: Optional[str],
        llama_n_gpu_layers: int,
        max_model_len: Optional[int],
    ) -> Any:
        from llama_cpp import Llama
        from llama_cpp.llama_chat_format import Qwen3VLChatHandler
        from huggingface_hub import hf_hub_download

        init_kwargs: Dict[str, Any] = {
            "n_gpu_layers": llama_n_gpu_layers,
            "n_ctx": int(max_model_len or 4096),
            "verbose": False,
        }
        if llama_mmproj_path:
            resolved_mmproj = llama_mmproj_path
            if model_path:
                candidate = Path(model_path) / llama_mmproj_path
                if candidate.exists():
                    resolved_mmproj = str(candidate)
            if not Path(resolved_mmproj).exists():
                try:
                    resolved_mmproj = hf_hub_download(
                        repo_id=model_name,
                        filename=llama_mmproj_path,
                    )
                except Exception:
                    resolved_mmproj = llama_mmproj_path
            init_kwargs["chat_handler"] = Qwen3VLChatHandler(
                clip_model_path=resolved_mmproj
            )

        if model_path:
            init_kwargs["model_path"] = model_path
            return Llama(**init_kwargs)

        if not llama_filename:
            raise RuntimeError("llama_filename is required when model_path is not provided")
        return Llama.from_pretrained(
            repo_id=model_name,
            filename=llama_filename,
            **init_kwargs,
        )

    async def start_instance(
        self,
        alias: str,
        model_name: str,
        model_path: Optional[str] = None,
        *,
        fake: bool = False,
        fake_response: Optional[str] = None,
        fake_delay: Optional[float] = None,
        fake_delay_ms: Optional[int] = None,
        fake_capacity: Optional[int] = None,
        llama_filename: Optional[str] = None,
        llama_mmproj_path: Optional[str] = None,
        llama_n_gpu_layers: int = -1,
        max_model_len: Optional[int] = 4096,
    ) -> VLLMInstanceInfo:
        if alias in self.instances:
            return self.instances[alias]

        instance = VLLMInstanceInfo(
            alias=alias,
            model_name=model_name,
            port=0,
            status=InstanceStatus.STARTING,
            created_at=time.time(),
            base_url="",
            pid=0,
        )
        self.instances[alias] = instance
        self._init_metrics(alias)

        if fake:
            delay_s = float(fake_delay if fake_delay is not None else 0.02)
            if fake_delay_ms is not None:
                delay_s = max(0.0, int(fake_delay_ms) / 1000.0)
            capacity = int(fake_capacity if fake_capacity is not None else 1)
            response = fake_response or "This is a fake llama.cpp response."
            self.fake_instances[alias] = FakeLlamaInstance(response=response, delay_s=delay_s, capacity=capacity)
            instance.status = InstanceStatus.RUNNING
            instance.last_used = time.time()
            return instance

        try:
            llm = await asyncio.to_thread(
                self._load_llm_sync,
                model_name=model_name,
                model_path=model_path,
                llama_filename=llama_filename,
                llama_mmproj_path=llama_mmproj_path,
                llama_n_gpu_layers=llama_n_gpu_layers,
                max_model_len=max_model_len,
            )
            self.runtimes[alias] = LlamaRuntime(
                llm=llm,
                capacity=1,
                semaphore=asyncio.Semaphore(1),
            )
            instance.status = InstanceStatus.RUNNING
            instance.last_used = time.time()
            return instance
        except Exception:
            self.instances.pop(alias, None)
            self.fake_instances.pop(alias, None)
            self.runtimes.pop(alias, None)
            self._inflight_requests.pop(alias, None)
            self._latency_metrics.pop(alias, None)
            raise

    async def stop_instance(self, alias: str) -> bool:
        if alias not in self.instances:
            return False
        runtime = self.runtimes.pop(alias, None)
        if runtime is not None:
            llm = runtime.llm
            try:
                if hasattr(llm, "close"):
                    await asyncio.to_thread(llm.close)
            except Exception as exc:
                logger.warning("failed to close llama runtime %s: %s", alias, exc)
        self.instances.pop(alias, None)
        self.fake_instances.pop(alias, None)
        self._inflight_requests.pop(alias, None)
        self._latency_metrics.pop(alias, None)
        return True

    def get_instance(self, alias: str) -> Optional[VLLMInstanceInfo]:
        return self.instances.get(alias)

    def list_instances(self) -> Dict[str, VLLMInstanceInfo]:
        return self.instances.copy()

    async def ensure_active(self, alias: str) -> None:
        if alias not in self.instances:
            raise RuntimeError(f"instance {alias} not found")

    def get_instance_status(self, alias: str) -> Optional[Dict[str, Any]]:
        instance = self.instances.get(alias)
        if not instance:
            return None
        metrics = self._ensure_metrics(alias)
        request_count = int(metrics.get("count") or 0)
        if alias in self.fake_instances:
            capacity = self.fake_instances[alias].capacity
            is_fake = True
        else:
            capacity = 1
            is_fake = False
        return {
            "alias": alias,
            "backend_type": "llama.cpp",
            "status": instance.status.value,
            "model_name": instance.model_name,
            "sleep_level": "ACTIVE",
            "sleep_level_value": 0,
            "last_used": instance.last_used,
            "created_at": instance.created_at,
            "idle_time": time.time() - instance.last_used if instance.last_used else 0,
            "inflight_requests": self._inflight_requests.get(alias, 0),
            "ttft_last": self._to_metric_float(metrics.get("ttft_last")),
            "ttft_avg": self._to_metric_float(metrics.get("ttft_avg")),
            "e2e_last": self._to_metric_float(metrics.get("e2e_last")),
            "e2e_avg": self._to_metric_float(metrics.get("e2e_avg")),
            "request_count": request_count,
            "has_metrics": request_count > 0,
            "capacity": capacity,
            "max_concurrency": float(capacity),
            "is_fake": is_fake,
        }

    def _metric_update(self, alias: str, ttft: float, e2e: float) -> None:
        metrics = self._ensure_metrics(alias)
        metrics["count"] += 1
        count = metrics["count"]
        metrics["ttft_last"] = ttft
        metrics["e2e_last"] = e2e
        if metrics["ttft_avg"] is None:
            metrics["ttft_avg"] = ttft
        else:
            metrics["ttft_avg"] += (ttft - metrics["ttft_avg"]) / count
        if metrics["e2e_avg"] is None:
            metrics["e2e_avg"] = e2e
        else:
            metrics["e2e_avg"] += (e2e - metrics["e2e_avg"]) / count

    async def _generate_fake(
        self,
        alias: str,
        messages: Any,
    ) -> AsyncIterator[_FakeEngineOutput]:
        fake = self.fake_instances.get(alias)
        if fake is None:
            raise RuntimeError(f"fake instance {alias} not found")
        async with fake.semaphore:
            await asyncio.sleep(fake.delay_s)
            prompt = ""
            if isinstance(messages, list):
                for msg in messages:
                    if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                        prompt = msg["content"]
            text = fake.response.replace("{prompt}", prompt)
            # Stream in small chunks but keep `text` cumulative.
            buf = ""
            chunk_size = 16
            for i in range(0, len(text), chunk_size):
                buf += text[i : i + chunk_size]
                yield _FakeEngineOutput(outputs=[_FakeCompletionOutput(text=buf, finish_reason=None)])
            yield _FakeEngineOutput(outputs=[_FakeCompletionOutput(text=buf, finish_reason="stop")])

    def _extract_sampling(self, sampling_params: Any) -> Dict[str, Any]:
        temperature = getattr(sampling_params, "temperature", 0.2)
        top_p = getattr(sampling_params, "top_p", 0.95)
        max_tokens = getattr(sampling_params, "max_tokens", 256)
        if temperature is None:
            temperature = 0.2
        if top_p is None:
            top_p = 0.95
        if max_tokens is None:
            max_tokens = 256
        return {
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_tokens": int(max_tokens),
        }

    async def generate(
        self,
        alias: str,
        messages: Any,
        sampling_params: Any,
        base_model: Optional[str] = None,
    ) -> AsyncIterator[_FakeEngineOutput]:
        instance = self.instances.get(alias)
        if instance is None:
            raise RuntimeError(f"instance {alias} not found")
        if instance.status != InstanceStatus.RUNNING:
            raise RuntimeError(f"instance {alias} is not running")

        self._inflight_requests[alias] = self._inflight_requests.get(alias, 0) + 1
        start = time.perf_counter()
        ttft = None
        e2e = None
        resolved_base_model = self._select_base_model(base_model, instance)
        backend_messages = convert_messages_for_backend(
            messages,
            BACKEND_LLAMA_CPP,
            base_model=resolved_base_model,
        )
        try:
            if alias in self.fake_instances:
                async for out in self._generate_fake(alias, backend_messages):
                    if ttft is None:
                        ttft = time.perf_counter() - start
                    yield out
            else:
                runtime = self.runtimes.get(alias)
                if runtime is None:
                    raise RuntimeError(f"runtime {alias} not found")
                async with runtime.semaphore:
                    sampling = self._extract_sampling(sampling_params)

                    # True streaming from llama.cpp. We convert delta chunks into cumulative text.
                    loop = asyncio.get_running_loop()
                    queue: asyncio.Queue[object] = asyncio.Queue()

                    def _producer() -> None:
                        try:
                            stream_iter = runtime.llm.create_chat_completion(
                                messages=backend_messages,
                                stream=True,
                                **sampling,
                            )
                            for chunk in stream_iter:
                                loop.call_soon_threadsafe(queue.put_nowait, chunk)
                        except Exception as exc:  # pragma: no cover
                            loop.call_soon_threadsafe(queue.put_nowait, exc)
                        finally:
                            loop.call_soon_threadsafe(queue.put_nowait, None)

                    producer_task = asyncio.create_task(asyncio.to_thread(_producer))

                    buf = ""
                    while True:
                        item = await queue.get()
                        if item is None:
                            break
                        if isinstance(item, Exception):
                            raise item
                        if not isinstance(item, dict):
                            continue

                        delta_text, finish_reason = self._extract_stream_delta(item)
                        if delta_text:
                            buf += delta_text

                        if ttft is None and (delta_text or finish_reason):
                            ttft = time.perf_counter() - start

                        # Only emit when something meaningful happens.
                        if delta_text or finish_reason:
                            yield _FakeEngineOutput(
                                outputs=[_FakeCompletionOutput(text=buf, finish_reason=finish_reason)]
                            )

                    # Ensure background producer is finished (and surface any unexpected errors).
                    await producer_task
            e2e = time.perf_counter() - start
        finally:
            self._inflight_requests[alias] = max(self._inflight_requests.get(alias, 1) - 1, 0)
            instance.last_used = time.time()
            if ttft is not None and e2e is not None:
                self._metric_update(alias, ttft=ttft, e2e=e2e)
