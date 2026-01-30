"""测试脚本 - 验证系统功能（Ray Serve 版本）"""
import asyncio
import base64
import io
import json
import os
import time
from pathlib import Path

import httpx
from PIL import Image


class SystemTester:
    """系统测试器"""
    
    def __init__(
        self,
        serve_url: str = "http://localhost:8000",
        model_alias: str = "test-model",
        model_name: str = "Qwen/Qwen3-VL-2B-Instruct",
    ):
        self.serve_url = serve_url
        self.model_alias = model_alias
        self.model_name = model_name

    def _repo_root(self) -> Path:
        return Path(__file__).resolve().parents[1]

    def _load_image_data_uri(self) -> str:
        root = self._repo_root()
        candidates = [
            root / "demo.png",
            root / "demo2.jpg",
            root / "demo.jpg",
        ]
        for path in candidates:
            if path.exists():
                data = path.read_bytes()
                mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
                encoded = base64.b64encode(data).decode("ascii")
                return f"data:{mime};base64,{encoded}"

        image = Image.new("RGB", (256, 256), color=(64, 128, 192))
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        data = buf.getvalue()
        encoded = base64.b64encode(data).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    async def _wait_model_ready(self, alias: str, timeout_s: int = 600) -> bool:
        deadline = time.time() + timeout_s
        async with httpx.AsyncClient(timeout=10.0) as client:
            while time.time() < deadline:
                try:
                    response = await client.get(f"{self.serve_url}/admin/models")
                    if response.status_code == 200:
                        models = response.json().get("models", [])
                        for model in models:
                            if model.get("alias") == alias and model.get("instances"):
                                return True
                except Exception:
                    pass
                await asyncio.sleep(5)
        return False

    async def _stream_first_token(self, payload: dict, timeout_s: int = 600):
        start = time.perf_counter()
        first_token_latency = None
        content = []
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_s)) as client:
            async with client.stream(
                "POST",
                f"{self.serve_url}/v1/chat/completions",
                json=payload,
            ) as response:
                response.raise_for_status()
                try:
                    async for line in response.aiter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        chunk = json.loads(data)
                        delta = chunk["choices"][0].get("delta", {})
                        text = delta.get("content")
                        if text:
                            content.append(text)
                            if first_token_latency is None:
                                first_token_latency = time.perf_counter() - start
                except httpx.RemoteProtocolError as exc:
                    if first_token_latency is None:
                        raise
                    print(f"⚠️ 流式连接被提前关闭: {exc}")
        return first_token_latency, "".join(content)

    async def _single_token_latency(self, payload: dict, timeout_s: int = 600):
        start = time.perf_counter()
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_s)) as client:
            response = await client.post(
                f"{self.serve_url}/v1/chat/completions",
                json=payload,
            )
            response.raise_for_status()
            latency = time.perf_counter() - start
            result = response.json()
            content = result["choices"][0]["message"]["content"]
        return latency, content

    async def test_health_checks(self):
        """测试健康检查"""
        print("\n=== 测试健康检查 ===")
        
        async with httpx.AsyncClient() as client:
            # 测试 Serve
            try:
                response = await client.get(f"{self.serve_url}/health")
                print(f"✓ Serve 健康: {response.json()}")
            except Exception as e:
                print(f"✗ Serve 不可用: {e}")
    
    async def test_model_registration(self, alias: str = None, model_name: str = None):
        """测试模型注册"""
        alias = alias or self.model_alias
        model_name = model_name or self.model_name
        print(f"\n=== 测试模型注册: {alias} ===")
        
        async with httpx.AsyncClient(timeout=360.0) as client:
            try:
                response = await client.post(
                    f"{self.serve_url}/admin/models/register",
                    json={
                        "alias": alias,
                        "model_name": model_name,
                        "gpu_memory_gb": 6.0,
                        "max_model_len": 2048
                    }
                )
                result = response.json()
                print(f"✓ 模型注册结果: {result}")
                return True
            except Exception as e:
                print(f"✗ 模型注册失败: {e}")
                return False
    
    async def test_list_models(self):
        """测试列出模型"""
        print("\n=== 测试列出模型 ===")
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(f"{self.serve_url}/v1/models")
                models = response.json()
                print(f"可用模型数量: {len(models['data'])}")
                for model in models['data']:
                    print(f"  - {model['id']}")
            except Exception as e:
                print(f"✗ 列出模型失败: {e}")
    
    async def test_chat_completion(self, model: str = None, ensure_registered: bool = True):
        """测试对话补全"""
        model = model or self.model_alias
        print(f"\n=== 测试对话补全: {model} ===")
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                if ensure_registered:
                    await self.test_model_registration(alias=model)
                # 等待模型就绪
                print("等待模型就绪...")
                ready = await self._wait_model_ready(model, timeout_s=600)
                if not ready:
                    print("✗ 模型未在超时内就绪")
                    return False
                
                # 发送请求
                print("发送推理请求...")
                response = await client.post(
                    f"{self.serve_url}/v1/chat/completions",
                    json={
                        "model": model,
                        "messages": [
                            {"role": "user", "content": "Hello!"}
                        ],
                        "max_tokens": 32,
                        "temperature": 0.7
                    }
                )
                
                result = response.json()
                print(f"✓ 推理成功!")
                print(f"  响应: {result['choices'][0]['message']['content']}")
                return True
                
            except Exception as e:
                print(f"✗ 推理失败: {e}")
                return False
    
    async def test_system_status(self):
        """测试系统状态"""
        print("\n=== 测试系统状态 ===")
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(f"{self.serve_url}/admin/status")
                status = response.json()
                print(f"Serve 状态: {status['health']}")
                print(f"模型数量: {len(status['models']['models'])}")
            except Exception as e:
                print(f"✗ 获取系统状态失败: {e}")

    async def test_multimodal_single_image(self, model: str = None):
        model = model or self.model_alias
        print(f"\n=== 测试单图像多模态: {model} ===")
        image_data = self._load_image_data_uri()
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "请描述这张图片的主要内容。"},
                        {"type": "image_url", "image_url": {"url": image_data}},
                    ],
                }
            ],
            "max_tokens": 64,
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                response = await client.post(f"{self.serve_url}/v1/chat/completions", json=payload)
                response.raise_for_status()
                result = response.json()
                print("✓ 单图像请求成功")
                print(f"  响应: {result['choices'][0]['message']['content']}")
                return True
            except Exception as e:
                print(f"✗ 单图像请求失败: {e}")
                return False

    async def test_multimodal_image_list(self, model: str = None, frames: int = 4):
        model = model or self.model_alias
        print(f"\n=== 测试多帧图像列表: {model} ===")
        image_data = self._load_image_data_uri()
        content = [{"type": "text", "text": "以下是视频抽帧，请总结主要变化。"}]
        for _ in range(frames):
            content.append({"type": "image_url", "image_url": {"url": image_data}})
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 80,
        }
        async with httpx.AsyncClient(timeout=180.0) as client:
            try:
                response = await client.post(f"{self.serve_url}/v1/chat/completions", json=payload)
                response.raise_for_status()
                result = response.json()
                print("✓ 多帧图像请求成功")
                print(f"  响应: {result['choices'][0]['message']['content']}")
                return True
            except Exception as e:
                print(f"✗ 多帧图像请求失败: {e}")
                return False

    async def test_streaming_chat(self, model: str = None, ensure_registered: bool = True):
        model = model or self.model_alias
        print(f"\n=== 测试流式输出: {model} ===")
        if ensure_registered:
            await self.test_model_registration(alias=model)
        ready = await self._wait_model_ready(model, timeout_s=600)
        if not ready:
            print("✗ 模型未在超时内就绪，跳过流式测试")
            return False
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "用一句话介绍你自己。"}],
            "max_tokens": 64,
            "stream": True,
        }
        try:
            first_latency, content = await self._stream_first_token(payload, timeout_s=600)
            print("✓ 流式输出成功")
            if first_latency is not None:
                print(f"  首字符延迟: {first_latency:.3f}s")
            print(f"  响应: {content}")
            return True
        except Exception as e:
            print(f"✗ 流式输出失败: {e}")
            return False

    async def test_autoscale_fake_model(self):
        """测试自动扩容逻辑（假模型）"""
        print("\n=== 测试自动扩容（假模型） ===")
        alias = "fake-model"
        payload = {
            "alias": alias,
            "model_name": "__fake__",
            "fake": True,
            "fake_response": "FAKE_OK",
            "fake_delay_ms": 3000,
            "fake_capacity": 1,
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(f"{self.serve_url}/admin/models/register", json=payload)
            if response.status_code >= 400:
                print(f"✗ 假模型注册失败: {response.text}")
                return False
            result = response.json()
            if result.get("status") not in {"success", "exists"}:
                print(f"✗ 假模型注册失败: {result}")
                return False

        # 等待模型注册生效
        async with httpx.AsyncClient(timeout=30.0) as client:
            for _ in range(10):
                models = await client.get(f"{self.serve_url}/admin/models")
                data = models.json().get("models", [])
                if any(m.get("alias") == alias for m in data):
                    break
                await asyncio.sleep(1)

        async def _send_fake():
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    f"{self.serve_url}/v1/chat/completions",
                    json={
                        "model": alias,
                        "messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 4,
                    },
                )
                resp.raise_for_status()

        tasks = [asyncio.create_task(_send_fake()) for _ in range(6)]
        await asyncio.sleep(2)

        # 等待 autoscaler 触发扩容
        await asyncio.sleep(15)
        async with httpx.AsyncClient(timeout=30.0) as client:
            models = await client.get(f"{self.serve_url}/admin/models")
            data = models.json().get("models", [])
            target = next((m for m in data if m.get("alias") == alias), None)
            if not target:
                print("✗ 未找到假模型记录")
                return False
            instances = target.get("instances", [])
            print(f"当前实例数: {len(instances)}")
            if len(instances) < 2:
                print("✗ 未触发扩容")
                return False

        await asyncio.gather(*tasks, return_exceptions=True)
        print("✓ 自动扩容测试通过")
        return True

    async def test_autoscale_real_model(self):
        """测试自动扩容逻辑（真模型，逐步增加并发）"""
        print("\n=== 测试自动扩容（真模型） ===")
        alias = "qwen3-vl-2b-test"
        model_name = "Qwen/Qwen3-VL-2B-Instruct"
        gpu_memory_gb = float(os.getenv("REAL_MODEL_GPU_GB", "6.0"))
        max_model_len = int(os.getenv("REAL_MODEL_MAX_LEN", "2048"))
        max_concurrency = int(os.getenv("REAL_MODEL_MAX_CONCURRENCY", "8"))
        step = int(os.getenv("REAL_MODEL_CONCURRENCY_STEP", "2"))
        hold_seconds = int(os.getenv("REAL_MODEL_HOLD_SECONDS", "20"))
        scale_down_wait = int(os.getenv("REAL_MODEL_SCALE_DOWN_WAIT", "60"))

        # 注册模型（限制显存占用）
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.serve_url}/admin/models/register",
                json={
                    "alias": alias,
                    "model_name": model_name,
                    "gpu_memory_gb": gpu_memory_gb,
                    "max_model_len": max_model_len,
                },
            )
            if response.status_code >= 400:
                print(f"✗ 真模型注册失败: {response.text}")
                return False
            result = response.json()
            if result.get("status") not in {"success", "exists"}:
                print(f"✗ 真模型注册失败: {result}")
                return False

        ready = await self._wait_model_ready(alias, timeout_s=1200)
        if not ready:
            print("✗ 真模型未在超时内就绪")
            return False

        stop_event = asyncio.Event()

        async def _send_loop():
            async with httpx.AsyncClient(timeout=120.0) as client:
                while not stop_event.is_set():
                    resp = await client.post(
                        f"{self.serve_url}/v1/chat/completions",
                        json={
                            "model": alias,
                            "messages": [{"role": "user", "content": "Hi"}],
                            "max_tokens": 8,
                        },
                    )
                    if resp.status_code >= 400:
                        break

        tasks: list[asyncio.Task] = []

        async def _get_instances():
            async with httpx.AsyncClient(timeout=30.0) as client:
                models = await client.get(f"{self.serve_url}/admin/models")
                data = models.json().get("models", [])
                return next((m for m in data if m.get("alias") == alias), None)

        current = 0
        while current < max_concurrency:
            add = min(step, max_concurrency - current)
            for _ in range(add):
                tasks.append(asyncio.create_task(_send_loop()))
            current += add
            await asyncio.sleep(5)
            print(f"当前并发: {current}")

        # 维持负载一段时间，等待扩容
        await asyncio.sleep(hold_seconds)
        model_entry = await _get_instances()
        if not model_entry:
            print("✗ 未找到真模型记录")
            stop_event.set()
            await asyncio.gather(*tasks, return_exceptions=True)
            return False

        instances = model_entry.get("instances", [])
        print(f"扩容后实例数: {len(instances)}")

        # 停止压测，观察是否休眠
        stop_event.set()
        await asyncio.gather(*tasks, return_exceptions=True)

        await asyncio.sleep(scale_down_wait)
        print("⚠️ 休眠状态需要在 Worker 侧验证，已跳过")

        return True
    
    async def run_all_tests(self):
        """运行所有测试"""
        print("=" * 60)
        print("开始系统测试")
        print("=" * 60)
        
        # 基础测试
        await self.test_health_checks()
        await self.test_list_models()
        await self.test_system_status()
        
        # 功能测试（可选，需要较长时间）
        if os.getenv("RUN_FULL_TESTS") == "1":
            await self.test_model_registration()
            await self.test_list_models()
            await self.test_chat_completion(ensure_registered=False)
            await self.test_multimodal_single_image()
            await self.test_multimodal_image_list()
            await self.test_streaming_chat(ensure_registered=False)
            # await self.test_autoscale_fake_model()
            if os.getenv("RUN_REAL_AUTOSCALE_TEST") == "1":
                await self.test_autoscale_real_model()
        
        print("\n" + "=" * 60)
        print("测试完成")
        print("=" * 60)


async def main():
    """主函数"""
    import sys
    
    serve_url = sys.argv[1] if len(sys.argv) > 1 else "http://100.100.238.4:8000"
    
    tester = SystemTester(
        serve_url=serve_url
    )
    
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
