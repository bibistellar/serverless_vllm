"""Worker-level tests for model_pool_serve.

This script targets the worker service directly (instances, sleep/wake, etc.).
"""

import asyncio
import base64
import os
from pathlib import Path
from typing import Any, Dict, Optional

import httpx


class WorkerTester:
    def __init__(
        self,
        worker_url: str,
        alias: str,
        model_name: str,
        model_path: Optional[str],
        gpu_memory_utilization: Optional[float],
        max_model_len: Optional[int],
        tensor_parallel_size: Optional[int],
        image_path: Optional[str],
        poll_interval_s: float,
        poll_timeout_s: float,
        start_retries: int,
    ) -> None:
        self.worker_url = worker_url.rstrip("/")
        self.alias = alias
        self.model_name = model_name
        self.model_path = model_path
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.tensor_parallel_size = tensor_parallel_size
        self.image_path = image_path
        self.poll_interval_s = poll_interval_s
        self.poll_timeout_s = poll_timeout_s
        self.start_retries = start_retries

    async def test_health(self) -> None:
        print("\n=== Worker 健康检查 ===")
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{self.worker_url}/health")
            resp.raise_for_status()
            print(f"✓ Worker 健康: {resp.json()}")

    async def start_instance(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "alias": self.alias,
            "model_name": self.model_name,
        }
        if self.model_path:
            payload["model_path"] = self.model_path
        if self.gpu_memory_utilization is not None:
            payload["gpu_memory_utilization"] = self.gpu_memory_utilization
        if self.max_model_len is not None:
            payload["max_model_len"] = self.max_model_len
        if self.tensor_parallel_size is not None:
            payload["tensor_parallel_size"] = self.tensor_parallel_size

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(f"{self.worker_url}/instances/start", json=payload)
            resp.raise_for_status()
            data = resp.json()
            print(f"✓ 启动实例触发成功: status={data.get('status')}")
            return data

    async def wait_until_running(self) -> Dict[str, Any]:
        retries_left = self.start_retries
        deadline = asyncio.get_event_loop().time() + self.poll_timeout_s
        async with httpx.AsyncClient(timeout=30.0) as client:
            while True:
                resp = await client.get(f"{self.worker_url}/instances/{self.alias}/status")
                resp.raise_for_status()
                data = resp.json()
                status = data.get("status")
                sleep_level = data.get("sleep_level")
                if status == "running" and sleep_level == "ACTIVE":
                    print("✓ 实例已进入 RUNNING/ACTIVE")
                    return data
                if status == "error":
                    if retries_left > 0:
                        retries_left -= 1
                        print(f"✗ 启动失败，准备重试 start（剩余 {retries_left} 次）: {data}")
                        await self.start_instance()
                        deadline = asyncio.get_event_loop().time() + self.poll_timeout_s
                    else:
                        raise RuntimeError(f"实例启动失败: {data}")
                if asyncio.get_event_loop().time() >= deadline:
                    raise TimeoutError(f"等待实例 RUNNING 超时: {data}")
                await asyncio.sleep(self.poll_interval_s)

    async def _chat(self, content: Any) -> str:
        payload = {
            "model": self.alias,
            "messages": [
                {
                    "role": "user",
                    "content": content,
                }
            ],
            "max_tokens": 64,
            "stream": False,
        }
        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(
                f"{self.worker_url}/proxy/{self.alias}/v1/chat/completions",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return text

    async def text_test(self, label: str) -> None:
        print(f"\n--- 文本测试: {label} ---")
        content = "用一句话介绍你自己。"
        text = await self._chat(content)
        print(f"✓ 文本响应: {text}")

    def _load_image_data_url(self) -> str:
        if not self.image_path:
            raise FileNotFoundError("未配置 IMAGE_PATH，无法进行图片测试")
        path = Path(self.image_path)
        if not path.exists():
            raise FileNotFoundError(f"图片文件不存在: {path}")
        raw = path.read_bytes()
        encoded = base64.b64encode(raw).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"

    async def image_test(self) -> None:
        print("\n--- 图片测试 ---")
        image_url = self._load_image_data_url()
        content = [
            {"type": "text", "text": "请描述图片中的主要内容。"},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]
        text = await self._chat(content)
        print(f"✓ 图片响应: {text}")

    async def set_sleep(self, level: int, label: str) -> None:
        print(f"\n--- 强制 sleep={level}: {label} ---")
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{self.worker_url}/instances/{self.alias}/sleep",
                json={"level": level},
            )
            resp.raise_for_status()
            print(f"✓ 已设置 sleep={level}")

    async def wake(self) -> None:
        print("\n--- 唤醒实例 ---")
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(f"{self.worker_url}/instances/{self.alias}/wake")
            resp.raise_for_status()
            print("✓ 唤醒请求成功（进入 STARTING，需轮询确认 RUNNING）")

    async def wake_and_wait(self, label: str) -> None:
        await self.wake()
        await self.wait_until_running()
        await self.text_test(label)

    async def test_full_flow(self) -> None:
        print("\n=== vLLM 实例完整测试流程 ===")
        await self.start_instance()
        await self.wait_until_running()

        await self.text_test("首次 RUNNING")
        await self.image_test()

        await self.set_sleep(1, "SLEEP_1")
        await self.wake_and_wait("SLEEP_1 -> WAKE")

        await self.set_sleep(2, "SLEEP_2")
        await self.wake_and_wait("SLEEP_2 -> WAKE")

        await self.set_sleep(3, "UNLOADED")
        await self.wake_and_wait("UNLOADED -> WAKE")

    async def run_all(self) -> None:
        await self.test_health()
        await self.test_full_flow()


def _env_float(name: str) -> Optional[float]:
    value = os.getenv(name)
    if value is None or value == "":
        return None
    return float(value)


def _env_int(name: str) -> Optional[int]:
    value = os.getenv(name)
    if value is None or value == "":
        return None
    return int(value)


async def main() -> None:
    worker_url = os.getenv("WORKER_URL", "http://localhost:7000")
    alias = os.getenv("MODEL_ALIAS", "worker-test-vllm")
    model_name = os.getenv("MODEL_NAME", "Qwen/Qwen3-VL-2B-Instruct")
    model_path = os.getenv("MODEL_PATH")
    gpu_memory_utilization = 0.75
    max_model_len = 2048
    tensor_parallel_size = _env_int("TENSOR_PARALLEL_SIZE")
    image_path = os.getenv("IMAGE_PATH", "demo.png")
    poll_interval_s = float(os.getenv("POLL_INTERVAL_S", "2"))
    poll_timeout_s = float(os.getenv("POLL_TIMEOUT_S", "900"))
    start_retries = int(os.getenv("START_RETRY_COUNT", "1"))

    tester = WorkerTester(
        worker_url=worker_url,
        alias=alias,
        model_name=model_name,
        model_path=model_path,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        tensor_parallel_size=tensor_parallel_size,
        image_path=image_path,
        poll_interval_s=poll_interval_s,
        poll_timeout_s=poll_timeout_s,
        start_retries=start_retries,
    )
    await tester.run_all()


if __name__ == "__main__":
    asyncio.run(main())
