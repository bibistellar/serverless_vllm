"""Worker-level tests for model_pool_serve.

This script targets the worker service directly (instances, sleep/wake, etc.).
"""

import asyncio
import json
import os

import httpx


class WorkerTester:
    def __init__(self, worker_url: str = "http://localhost:7000") -> None:
        self.worker_url = worker_url.rstrip("/")

    async def test_health(self):
        print("\n=== Worker 健康检查 ===")
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{self.worker_url}/health")
            print(f"✓ Worker 健康: {resp.json()}")

    async def test_fake_instance_lifecycle(self):
        print("\n=== Fake 实例生命周期测试 ===")
        alias = "worker-test-fake"
        payload = {
            "alias": alias,
            "model_name": "__fake__",
            "fake": True,
            "fake_response": "FAKE_OK",
            "fake_delay_ms": 200,
            "fake_capacity": 1,
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(f"{self.worker_url}/instances/start", json=payload)
            resp.raise_for_status()
            print("✓ 启动 fake 实例")

            status = await client.get(f"{self.worker_url}/instances/{alias}/status")
            status.raise_for_status()
            print(f"✓ 实例状态: {status.json().get('status')}")

            await client.post(f"{self.worker_url}/instances/{alias}/sleep", json={"level": 1})
            await asyncio.sleep(1)
            status = await client.get(f"{self.worker_url}/instances/{alias}/status")
            status.raise_for_status()
            print(f"✓ 休眠级别: {status.json().get('sleep_level_value')}")

            await client.post(f"{self.worker_url}/instances/{alias}/wake")
            await asyncio.sleep(1)
            status = await client.get(f"{self.worker_url}/instances/{alias}/status")
            status.raise_for_status()
            print(f"✓ 唤醒后状态: {status.json().get('status')}")

            await client.post(f"{self.worker_url}/instances/{alias}/stop")
            print("✓ 停止 fake 实例")

    async def run_all(self):
        await self.test_health()
        # await self.test_fake_instance_lifecycle()


async def main():
    worker_url = os.getenv("WORKER_URL", "http://100.66.213.127:7000")
    tester = WorkerTester(worker_url=worker_url)
    await tester.run_all()


if __name__ == "__main__":
    asyncio.run(main())
