"""系统测试（Serve + Worker 协作版）"""
import asyncio
import json
import os
import sys
import time
from typing import Any, Dict, Optional, List

import httpx


class SystemTester:
    def __init__(
        self,
        serve_url: str,
        worker_url: str,
        model_alias: str,
        model_name: str,
        gpu_memory_gb: float,
        max_model_len: int,
    ) -> None:
        self.serve_url = serve_url.rstrip("/")
        self.worker_url = worker_url.rstrip("/")
        self.model_alias = model_alias
        self.model_name = model_name
        self.gpu_memory_gb = gpu_memory_gb
        self.max_model_len = max_model_len

    async def test_health(self) -> None:
        print("\n=== 健康检查 ===")
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{self.serve_url}/health")
            resp.raise_for_status()
            print(f"✓ Serve 健康: {resp.json()}")

    async def register_model(self) -> None:
        print("\n=== 模型注册 ===")
        payload = {
            "alias": self.model_alias,
            "model_name": self.model_name,
            "gpu_memory_gb": self.gpu_memory_gb,
            "max_model_len": self.max_model_len,
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(f"{self.serve_url}/admin/models/register", json=payload)
            resp.raise_for_status()
            result = resp.json()
            print(f"✓ 注册返回: {result}")

    async def list_info(self) -> None:
        print("\n=== 信息列举 ===")
        async with httpx.AsyncClient(timeout=10.0) as client:
            models = await client.get(f"{self.serve_url}/admin/models")
            workers = await client.get(f"{self.serve_url}/admin/workers")
            models.raise_for_status()
            workers.raise_for_status()
            print(f"✓ models: {models.json()}")
            print(f"✓ workers: {workers.json()}")

    async def _get_instance_record(self) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{self.serve_url}/admin/models")
            resp.raise_for_status()
            data = resp.json().get("models", [])
            model_entry = next((m for m in data if m.get("alias") == self.model_alias), None)
            if not model_entry:
                raise RuntimeError(f"未找到模型: {self.model_alias}")
            instances = model_entry.get("instances", [])
            if not instances:
                raise RuntimeError(f"模型无实例: {self.model_alias}")
            return instances[0]

    async def _chat_once(self) -> httpx.Response:
        payload = {
            "model": self.model_alias,
            "messages": [{"role": "user", "content": "用一句话介绍你自己。"}],
            "max_tokens": 64,
        }
        async with httpx.AsyncClient(timeout=300.0) as client:
            return await client.post(f"{self.serve_url}/v1/chat/completions", json=payload)

    async def chat_with_retry(self, retries: int, wait_s: int) -> bool:
        for attempt in range(retries + 1):
            resp = await self._chat_once()
            if resp.status_code == 200:
                data = resp.json()
                text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                print(f"✓ 推理成功: {text}")
                return True
            if resp.status_code == 503:
                try:
                    payload = resp.json()
                except json.JSONDecodeError:
                    payload = {}
                if payload.get("error", {}).get("code") == "model_not_ready":
                    if attempt == retries:
                        print(f"✗ 重试耗尽，仍未就绪: {payload}")
                        return False
                    print(f"⚠️ 模型未就绪，{wait_s}s 后重试（剩余 {retries - attempt} 次）")
                    await asyncio.sleep(wait_s)
                    continue
            print(f"✗ 推理失败: {resp.status_code} {resp.text}")
            return False
        return False

    async def force_sleep(self, level: int) -> None:
        instance = await self._get_instance_record()
        instance_alias = instance.get("instance_alias")
        control_url = instance.get("control_url") or self.worker_url
        if not instance_alias:
            raise RuntimeError("实例缺少 instance_alias")
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{control_url}/instances/{instance_alias}/sleep",
                json={"level": level},
            )
            resp.raise_for_status()
            print(f"✓ 强制 sleep={level} 成功: {resp.json()}")

    async def run(self) -> None:
        await self.test_health()
        await self.register_model()
        await self.list_info()

        print("\n=== 一次文本调用 ===")
        ok = await self.chat_with_retry(retries=3, wait_s=60)
        if not ok:
            raise RuntimeError("首次文本调用失败")

        # for level in (1, 2, 3):
        #     print(f"\n=== 强制进入 sleep{level} 并测试 ===")
        #     await self.force_sleep(level)
        #     ok = await self.chat_with_retry(retries=3, wait_s=60)
        #     if not ok:
        #         raise RuntimeError(f"sleep{level} 场景测试失败")

        if os.getenv("RUN_OVERLOAD_TEST", "0") == "1":
            await self.overload_test()

    @staticmethod
    def _percentile(values: List[float], percentile: float) -> float:
        if not values:
            return 0.0
        values = sorted(values)
        k = (len(values) - 1) * percentile
        f = int(k)
        c = min(f + 1, len(values) - 1)
        if f == c:
            return values[f]
        return values[f] + (values[c] - values[f]) * (k - f)

    async def overload_test(self) -> None:
        print("\n=== 过载测试 ===")
        concurrency = int(os.getenv("OVERLOAD_CONCURRENCY", "8"))
        total_requests = int(os.getenv("OVERLOAD_REQUESTS", "64"))
        timeout_s = float(os.getenv("OVERLOAD_TIMEOUT_S", "120"))

        max_tokens = int(os.getenv("OVERLOAD_MAX_TOKENS", "1024"))
        prompt_len = max(32, max_tokens)
        prompt = os.urandom(prompt_len // 2).hex()
        payload = {
            "model": self.model_alias,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }

        sem = asyncio.Semaphore(concurrency)
        ok_latencies: List[float] = []
        not_ready = 0
        errors = 0

        async def _one(client: httpx.AsyncClient) -> None:
            nonlocal not_ready, errors
            async with sem:
                start = time.perf_counter()
                try:
                    resp = await client.post(f"{self.serve_url}/v1/chat/completions", json=payload)
                    elapsed = time.perf_counter() - start
                    if resp.status_code == 200:
                        ok_latencies.append(elapsed)
                        return
                    if resp.status_code == 503:
                        try:
                            data = resp.json()
                        except json.JSONDecodeError:
                            data = {}
                        if data.get("error", {}).get("code") == "model_not_ready":
                            not_ready += 1
                            return
                    errors += 1
                except Exception:
                    errors += 1

        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_s)) as client:
            tasks = [asyncio.create_task(_one(client)) for _ in range(total_requests)]
            await asyncio.gather(*tasks, return_exceptions=True)

        print(f"总请求数: {total_requests}")
        print(f"成功数: {len(ok_latencies)}")
        print(f"model_not_ready: {not_ready}")
        print(f"其他错误: {errors}")
        if ok_latencies:
            p50 = self._percentile(ok_latencies, 0.5)
            p95 = self._percentile(ok_latencies, 0.95)
            p99 = self._percentile(ok_latencies, 0.99)
            print(f"成功延迟(秒): p50={p50:.3f}, p95={p95:.3f}, p99={p99:.3f}")


async def main() -> None:
    serve_url = os.getenv("SERVE_URL") or (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000")
    worker_url = os.getenv("WORKER_URL") or (sys.argv[2] if len(sys.argv) > 2 else "http://localhost:7000")
    model_alias = os.getenv("MODEL_ALIAS", "qwen3-vl-2b")
    model_name = os.getenv("MODEL_NAME", "Qwen/Qwen3-VL-2B-Instruct")
    gpu_memory_gb = float(os.getenv("GPU_MEMORY_GB", "12.0"))
    max_model_len = int(os.getenv("MAX_MODEL_LEN", "4096"))

    tester = SystemTester(
        serve_url=serve_url,
        worker_url=worker_url,
        model_alias=model_alias,
        model_name=model_name,
        gpu_memory_gb=gpu_memory_gb,
        max_model_len=max_model_len,
    )
    await tester.run()


if __name__ == "__main__":
    asyncio.run(main())
