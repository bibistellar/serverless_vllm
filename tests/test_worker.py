"""Worker integration test for sequential backend streaming checks.

This script validates both backends under limited hardware:
1) Start one backend instance.
2) Verify text chat streaming.
3) Verify image-list chat streaming.
4) Stop instance and wait for release.
5) Repeat with the other backend.
"""

from __future__ import annotations

import asyncio
import base64
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI

BACKEND_VLLM = "vllm"
BACKEND_LLAMA_CPP = "llama.cpp"
SUPPORTED_BACKENDS = {BACKEND_VLLM, BACKEND_LLAMA_CPP}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value else default


class WorkerTester:
    def __init__(self, worker_url: str) -> None:
        self.worker_url = worker_url.rstrip("/")
        self.poll_interval_s = _env_float("POLL_INTERVAL_S", 1.0)
        self.poll_timeout_s = _env_float("POLL_TIMEOUT_S", 1200.0)
        self.release_wait_s = _env_float("TEST_RELEASE_WAIT_S", 8.0)
        self.image_count = _env_int("TEST_IMAGE_COUNT", 8)

        self.vllm_alias = os.getenv("TEST_VLLM_ALIAS", "worker-test-vllm-real")
        self.vllm_model_name = os.getenv("TEST_VLLM_MODEL_NAME", "Qwen/Qwen3-VL-2B-Instruct")
        self.vllm_model_path = os.getenv("TEST_VLLM_MODEL_PATH")
        self.vllm_gpu_memory_utilization = _env_float("TEST_VLLM_GPU_MEMORY_UTILIZATION", 0.6)
        self.vllm_max_model_len = _env_int("TEST_VLLM_MAX_MODEL_LEN", 4096)
        self.vllm_tp = _env_int("TEST_VLLM_TP", 1)

        self.llama_alias = os.getenv("TEST_LLAMA_ALIAS", "worker-test-llama-real")
        self.llama_repo_or_name = os.getenv("TEST_LLAMA_MODEL_NAME", "Qwen/Qwen3-VL-2B-Instruct-GGUF")
        self.llama_model_path = os.getenv("TEST_LLAMA_MODEL_PATH")
        self.llama_filename = os.getenv("TEST_LLAMA_FILENAME", "Qwen3VL-2B-Instruct-F16.gguf")
        self.llama_mmproj_path = os.getenv(
            "TEST_LLAMA_MMPROJ_PATH",
            "mmproj-Qwen3VL-2B-Instruct-F16.gguf",
        )
        self.llama_n_gpu_layers = _env_int("TEST_LLAMA_N_GPU_LAYERS", -1)
        self.llama_max_model_len = _env_int("TEST_LLAMA_MAX_MODEL_LEN", 4096)

        self.text_prompt = os.getenv("TEST_TEXT_PROMPT", "请简短介绍你自己，不超过两句话。")
        self.image_prompt = os.getenv("TEST_IMAGE_PROMPT", "请简短描述这些图像内容。")

    async def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        async with httpx.AsyncClient(timeout=300.0) as client:
            return await client.request(method, f"{self.worker_url}{path}", **kwargs)

    async def _expect(self, method: str, path: str, status: int, **kwargs: Any) -> httpx.Response:
        resp = await self._request(method, path, **kwargs)
        if resp.status_code != status:
            raise RuntimeError(
                f"{method} {path} expected={status} got={resp.status_code} body={resp.text}"
            )
        return resp

    def _openai_client(self, alias: str) -> OpenAI:
        base_url = f"{self.worker_url}/proxy/{alias}/v1"
        return OpenAI(base_url=base_url, api_key=os.getenv("OPENAI_API_KEY", "EMPTY"))

    def _resolve_alias(self, backend: str) -> str:
        if backend == BACKEND_VLLM:
            return self.vllm_alias
        if backend == BACKEND_LLAMA_CPP:
            return self.llama_alias
        raise ValueError(f"unsupported backend={backend}")

    def _resolve_base_model(self, backend: str) -> str:
        if backend == BACKEND_VLLM:
            return self.vllm_model_name
        if backend == BACKEND_LLAMA_CPP:
            return self.llama_repo_or_name
        raise ValueError(f"unsupported backend={backend}")

    async def test_health(self) -> None:
        print("\n=== health ===")
        resp = await self._expect("GET", "/health", 200)
        print(resp.json())

    async def start_vllm(self) -> None:
        payload: Dict[str, Any] = {
            "alias": self.vllm_alias,
            "backend_type": BACKEND_VLLM,
            "model_name": self.vllm_model_name,
            "gpu_memory_utilization": self.vllm_gpu_memory_utilization,
            "max_model_len": self.vllm_max_model_len,
            "tensor_parallel_size": self.vllm_tp,
        }
        if self.vllm_model_path:
            payload["model_path"] = self.vllm_model_path
        await self._expect("POST", "/instances/start", 200, json=payload)
        print(f"start vllm: {self.vllm_alias}")

    async def start_llama(self) -> None:
        payload: Dict[str, Any] = {
            "alias": self.llama_alias,
            "backend_type": BACKEND_LLAMA_CPP,
            "model_name": self.llama_repo_or_name,
            "llama_n_gpu_layers": self.llama_n_gpu_layers,
            "max_model_len": self.llama_max_model_len,
        }
        if self.llama_model_path:
            payload["model_path"] = self.llama_model_path
        else:
            payload["llama_filename"] = self.llama_filename
        if self.llama_mmproj_path:
            payload["llama_mmproj_path"] = self.llama_mmproj_path
        await self._expect("POST", "/instances/start", 200, json=payload)
        print(f"start llama: {self.llama_alias}")

    async def start_backend(self, backend: str) -> None:
        if backend == BACKEND_VLLM:
            await self.start_vllm()
            return
        if backend == BACKEND_LLAMA_CPP:
            await self.start_llama()
            return
        raise ValueError(f"unsupported backend={backend}")

    async def wait_running(self, alias: str, backend_type: Optional[str] = None) -> Dict[str, Any]:
        deadline = asyncio.get_event_loop().time() + self.poll_timeout_s
        while True:
            resp = await self._expect("GET", f"/instances/{alias}/status", 200)
            data = resp.json()
            if data.get("status") == "running":
                if backend_type and data.get("backend_type") != backend_type:
                    raise RuntimeError(f"{alias} backend mismatch: {data}")
                return data
            if data.get("status") == "error":
                raise RuntimeError(f"{alias} startup failed: {data}")
            if asyncio.get_event_loop().time() >= deadline:
                raise TimeoutError(f"{alias} wait running timeout: {data}")
            await asyncio.sleep(self.poll_interval_s)

    async def ensure_stopped(self, alias: str) -> None:
        status_resp = await self._request("GET", f"/instances/{alias}/status")
        if status_resp.status_code == 404:
            return
        if status_resp.status_code != 200:
            raise RuntimeError(
                f"GET /instances/{alias}/status unexpected={status_resp.status_code} body={status_resp.text}"
            )

        stop_resp = await self._request("POST", f"/instances/{alias}/stop")
        if stop_resp.status_code not in (200, 404):
            raise RuntimeError(
                f"POST /instances/{alias}/stop unexpected={stop_resp.status_code} body={stop_resp.text}"
            )

        deadline = asyncio.get_event_loop().time() + max(60.0, self.poll_timeout_s / 2)
        while True:
            resp = await self._request("GET", f"/instances/{alias}/status")
            if resp.status_code == 404:
                return
            if asyncio.get_event_loop().time() >= deadline:
                raise TimeoutError(f"{alias} stop timeout: {resp.status_code} {resp.text}")
            await asyncio.sleep(self.poll_interval_s)

    @staticmethod
    def _chunk_to_text(event: Any) -> str:
        if not getattr(event, "choices", None):
            return ""
        delta = event.choices[0].delta
        if not delta:
            return ""
        content = getattr(delta, "content", None)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                    continue
                txt = getattr(item, "text", None)
                if isinstance(txt, str):
                    parts.append(txt)
                    continue
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            return "".join(parts)
        return ""

    def _stream_chat(self, alias: str, base_model: str, messages: List[Dict[str, Any]], *, title: str) -> str:
        client = self._openai_client(alias)
        text = ""
        request_kwargs: Dict[str, Any] = {
            "model": alias,
            "messages": messages,
            "max_tokens": 2048,
            "stream": True,
        }
        if base_model:
            request_kwargs["extra_body"] = {"base_model": base_model}
        stream = client.chat.completions.create(**request_kwargs)
        print(f"\n--- {title} ---")
        for event in stream:
            chunk = self._chunk_to_text(event)
            if chunk:
                text += chunk
                print(chunk, end="", flush=True)
        if text:
            print("")
        return text

    def _resolve_test_image(self) -> Optional[Path]:
        image_path = os.getenv("TEST_IMAGE_PATH")
        if image_path:
            img_file = Path(image_path)
        else:
            repo_root = Path(__file__).resolve().parents[1]
            img_file = repo_root / "demo.png"
        if not img_file.exists():
            return None
        return img_file

    def _build_image_list_messages(self, prompt: str) -> Optional[List[Dict[str, Any]]]:
        img_file = self._resolve_test_image()
        if img_file is None:
            print("skip image-list stream test: TEST_IMAGE_PATH and demo.png both missing")
            return None

        mime, _ = mimetypes.guess_type(str(img_file))
        if not mime:
            mime = "application/octet-stream"
        encoded = base64.b64encode(img_file.read_bytes()).decode("utf-8")
        data_uri = f"data:{mime};base64,{encoded}"

        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    *[
                        {"type": "image_url", "image_url": {"url": data_uri}}
                        for _ in range(self.image_count)
                    ],
                ],
            }
        ]

    async def run_backend_stream_tests(self, backend: str) -> None:
        if backend not in SUPPORTED_BACKENDS:
            raise ValueError(f"unsupported backend={backend}")

        alias = self._resolve_alias(backend)
        base_model = self._resolve_base_model(backend)
        print(f"\n=== backend {backend} ({alias}) ===")

        await self.ensure_stopped(alias)
        await self.start_backend(backend)
        await self.wait_running(alias, backend)
        try:
            text_messages = [{"role": "user", "content": self.text_prompt}]
            text = self._stream_chat(
                alias,
                base_model,
                text_messages,
                title=f"{backend} text stream",
            )
            if not text.strip():
                raise RuntimeError(f"{backend} text stream returned empty output")
            print(f"\n{text[:200]}")

            image_messages = self._build_image_list_messages(self.image_prompt)
            if image_messages is not None:
                image_text = self._stream_chat(
                    alias,
                    base_model,
                    image_messages,
                    title=f"{backend} image-list stream ({self.image_count} images)",
                )
                if not image_text.strip():
                    raise RuntimeError(f"{backend} image-list stream returned empty output")
                print(f"\n{image_text[:200]}")
        finally:
            await self.ensure_stopped(alias)
            if self.release_wait_s > 0:
                print(f"wait {self.release_wait_s:.1f}s for resource release...")
                await asyncio.sleep(self.release_wait_s)

    async def run_all(self) -> None:
        await self.test_health()

        order_env = os.getenv("TEST_BACKEND_ORDER", f"{BACKEND_LLAMA_CPP},{BACKEND_VLLM}")
        order = [item.strip() for item in order_env.split(",") if item.strip()]
        if not order:
            raise RuntimeError("TEST_BACKEND_ORDER resolved to empty list")
        invalid = [b for b in order if b not in SUPPORTED_BACKENDS]
        if invalid:
            raise RuntimeError(f"TEST_BACKEND_ORDER has unsupported values: {invalid}")

        print(f"backend order: {order}")
        for backend in order:
            await self.run_backend_stream_tests(backend)

        print("\nall worker streaming checks passed")


async def main() -> None:
    worker_url = os.getenv("WORKER_URL", "http://localhost:7000")
    tester = WorkerTester(worker_url)
    await tester.run_all()


if __name__ == "__main__":
    asyncio.run(main())
