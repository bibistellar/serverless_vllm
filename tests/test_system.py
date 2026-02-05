"""系统测试（Serve + Worker 协作版）"""
import asyncio
import base64
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

    def _load_image_data_uris(self) -> List[str]:
        paths_raw = os.getenv("IMAGE_PATHS", "")
        if paths_raw:
            paths = [p.strip() for p in paths_raw.split(",") if p.strip()]
        else:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            default_path = os.path.join("/root/Code/video_anomaly_analysis_system/model_pool_serve/demo.png")
            paths = [default_path] if os.path.exists(default_path) else []
        if not paths:
            raise RuntimeError("未找到可用图片，请设置 IMAGE_PATHS")

        data_uris: List[str] = []
        for path in paths:
            with open(path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("ascii")
            ext = os.path.splitext(path)[1].lower()
            if ext in {".png"}:
                mime = "image/png"
            elif ext in {".jpg", ".jpeg"}:
                mime = "image/jpeg"
            else:
                mime = "application/octet-stream"
            data_uris.append(f"data:{mime};base64,{encoded}")
        return data_uris

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

        # 真实模型扩缩容测试（并发/突发请求驱动）
        if os.getenv("RUN_REAL_AUTOSCALER_TEST", "1") == "1":
            await self.real_autoscaler_test()

        # 图片列表 + 流式回复测试
        if os.getenv("RUN_IMAGE_STREAM_TEST", "0") == "1":
            await self.image_list_stream_test()

        # autoscaler 唤醒测试（强制 sleep 后观察唤醒恢复）
        run_wake_test = os.getenv("RUN_AUTOSCALER_WAKE_TEST", "0") == "1"
        run_wake_test = run_wake_test or os.getenv("RUN_AUTOSCALER_TEST", "0") == "1"
        if run_wake_test:
            await self.autoscaler_wake_test()

        if os.getenv("RUN_FAKE_AUTOSCALER_TEST", "0") == "1":
            await self.fake_autoscaler_test()

        if os.getenv("RUN_OVERLOAD_TEST", "0") == "1":
            await self.overload_test()

    async def autoscaler_wake_test(self) -> None:
        print("\n=== 弹性调度唤醒测试 ===")
        level = int(os.getenv("AUTOSCALER_SLEEP_LEVEL", "2"))
        retries = int(os.getenv("AUTOSCALER_RETRY_COUNT", "3"))
        wait_s = int(os.getenv("AUTOSCALER_RETRY_WAIT_S", "60"))

        await self.force_sleep(level)
        ok = await self.chat_with_retry(retries=retries, wait_s=wait_s)
        if not ok:
            raise RuntimeError(f"autoscaler 唤醒测试失败（sleep={level}）")

    async def fake_autoscaler_test(self) -> None:
        print("\n=== Fake 模型弹性调度测试 ===")
        alias = os.getenv("FAKE_MODEL_ALIAS", "fake-model")
        fake_payload = {
            "alias": alias,
            "model_name": "__fake__",
            "fake": True,
            "fake_response": "FAKE_OK",
            "fake_delay_ms": int(os.getenv("FAKE_DELAY_MS", "2000")),
            "fake_capacity": int(os.getenv("FAKE_CAPACITY", "1")),
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(f"{self.serve_url}/admin/models/register", json=fake_payload)
            resp.raise_for_status()
            print(f"✓ Fake 注册: {resp.json()}")

        # 先确保 fake 模型至少成功一次，产生延迟样本
        retries = int(os.getenv("FAKE_READY_RETRY_COUNT", "5"))
        wait_s = int(os.getenv("FAKE_READY_RETRY_WAIT_S", "5"))
        for attempt in range(retries + 1):
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{self.serve_url}/v1/chat/completions",
                    json={
                        "model": alias,
                        "messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 4,
                    },
                )
            if resp.status_code == 200:
                print("✓ Fake 首次请求成功")
                break
            if resp.status_code == 503 and attempt < retries:
                print(f"⚠️ Fake 未就绪，{wait_s}s 后重试")
                await asyncio.sleep(wait_s)
                continue
            resp.raise_for_status()

        total = int(os.getenv("FAKE_BURST_REQUESTS", "24"))
        concurrency = int(os.getenv("FAKE_BURST_CONCURRENCY", "8"))
        sem = asyncio.Semaphore(concurrency)

        async def _send_one() -> None:
            async with sem:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    resp = await client.post(
                        f"{self.serve_url}/v1/chat/completions",
                        json={
                            "model": alias,
                            "messages": [{"role": "user", "content": "hi"}],
                            "max_tokens": 4,
                        },
                    )
                    resp.raise_for_status()

        tasks = [asyncio.create_task(_send_one()) for _ in range(total)]
        await asyncio.gather(*tasks, return_exceptions=True)

        # 简单观察实例数量变化
        await asyncio.sleep(int(os.getenv("FAKE_SCALE_WAIT_S", "10")))
        async with httpx.AsyncClient(timeout=30.0) as client:
            models = await client.get(f"{self.serve_url}/admin/models")
            models.raise_for_status()
            data = models.json().get("models", [])
            target = next((m for m in data if m.get("alias") == alias), None)
            instances = target.get("instances", []) if target else []
            print(f"Fake 模型当前实例数: {len(instances)}")
            min_instances = int(os.getenv("FAKE_MIN_INSTANCES", "2"))
            if len(instances) < min_instances:
                raise RuntimeError(f"Fake 扩容断言失败：实例数 {len(instances)} < {min_instances}")

        # 缩容观察（等待一段时间后再次查看）
        scale_down_wait = int(os.getenv("FAKE_SCALE_DOWN_WAIT_S", "60"))
        await asyncio.sleep(scale_down_wait)
        async with httpx.AsyncClient(timeout=30.0) as client:
            models = await client.get(f"{self.serve_url}/admin/models")
            models.raise_for_status()
            data = models.json().get("models", [])
            target = next((m for m in data if m.get("alias") == alias), None)
            instances_after = target.get("instances", []) if target else []
            print(f"Fake 模型缩容后实例数: {len(instances_after)}")

    async def real_autoscaler_test(self) -> None:
        print("\n=== 真实模型弹性调度测试 ===")
        ok = await self.chat_with_retry(retries=3, wait_s=60)
        if not ok:
            raise RuntimeError("真实模型未就绪，无法执行弹性调度测试")

        total = int(os.getenv("REAL_BURST_REQUESTS", "16"))
        concurrency = int(os.getenv("REAL_BURST_CONCURRENCY", "8"))
        max_tokens = int(os.getenv("REAL_BURST_MAX_TOKENS", "2048"))
        test_duration_s = int(os.getenv("REAL_TEST_DURATION_S", "600"))
        pause_s = float(os.getenv("REAL_BURST_PAUSE_S", "2"))
        prompt_len = max(32, max_tokens)
        prompt = os.urandom(prompt_len // 2).hex()
        payload = {
            "model": self.model_alias,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }
        sem = asyncio.Semaphore(concurrency)

        async def _send_one() -> None:
            async with sem:
                async with httpx.AsyncClient(timeout=300.0) as client:
                    resp = await client.post(f"{self.serve_url}/v1/chat/completions", json=payload)
                    if resp.status_code == 503:
                        return
                    resp.raise_for_status()

        print(f"真实模型弹性调度测试持续 {test_duration_s}s ...")
        end_time = time.time() + test_duration_s
        while time.time() < end_time:
            tasks = [asyncio.create_task(_send_one()) for _ in range(total)]
            await asyncio.gather(*tasks, return_exceptions=True)
            if pause_s > 0:
                await asyncio.sleep(pause_s)

        await asyncio.sleep(int(os.getenv("REAL_SCALE_WAIT_S", "10")))
        async with httpx.AsyncClient(timeout=30.0) as client:
            models = await client.get(f"{self.serve_url}/admin/models")
            models.raise_for_status()
            data = models.json().get("models", [])
            target = next((m for m in data if m.get("alias") == self.model_alias), None)
            instances = target.get("instances", []) if target else []
            print(f"真实模型当前实例数: {len(instances)}")
            min_instances = int(os.getenv("REAL_MIN_INSTANCES", "1"))
            if len(instances) < min_instances:
                raise RuntimeError(f"真实模型扩容断言失败：实例数 {len(instances)} < {min_instances}")

        scale_down_wait = int(os.getenv("REAL_SCALE_DOWN_WAIT_S", "60"))
        await asyncio.sleep(scale_down_wait)
        async with httpx.AsyncClient(timeout=30.0) as client:
            models = await client.get(f"{self.serve_url}/admin/models")
            models.raise_for_status()
            data = models.json().get("models", [])
            target = next((m for m in data if m.get("alias") == self.model_alias), None)
            instances_after = target.get("instances", []) if target else []
            print(f"真实模型缩容后实例数: {len(instances_after)}")

    async def image_list_stream_test(self) -> None:
        print("\n=== 图片列表流式回复测试 ===")
        data_uris = self._load_image_data_uris()
        content = [{"type": "text", "text": "请描述这些图片的主要内容。"}]
        for uri in data_uris:
            content.append({"type": "image_url", "image_url": {"url": uri}})

        payload = {
            "model": self.model_alias,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": int(os.getenv("IMAGE_STREAM_MAX_TOKENS", "128")),
            "stream": True,
        }

        retries = int(os.getenv("IMAGE_STREAM_RETRY_COUNT", "3"))
        wait_s = int(os.getenv("IMAGE_STREAM_RETRY_WAIT_S", "60"))
        for attempt in range(retries + 1):
            timeout = httpx.Timeout(None)
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream(
                    "POST",
                    f"{self.serve_url}/v1/chat/completions",
                    json=payload,
                ) as resp:
                    if resp.status_code == 503:
                        body = await resp.aread()
                        try:
                            data = json.loads(body.decode("utf-8"))
                        except json.JSONDecodeError:
                            data = {}
                        if data.get("error", {}).get("code") == "model_not_ready":
                            if attempt == retries:
                                raise RuntimeError(f"图片流式测试失败：模型未就绪 {data}")
                            print(f"⚠️ 模型未就绪，{wait_s}s 后重试")
                            await asyncio.sleep(wait_s)
                            continue
                    resp.raise_for_status()

                    text = ""
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        if not line.startswith("data:"):
                            continue
                        data_str = line.split("data:", 1)[1].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue
                        delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
                        if delta:
                            text += delta
                    if not text:
                        raise RuntimeError("图片流式回复为空")
                    print(f"✓ 图片流式回复: {text}")
                    return

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
    worker_url = os.getenv("WORKER_URL") or (sys.argv[2] if len(sys.argv) > 2 else "http://100.66.213.127:7000")
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
