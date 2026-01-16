"""测试脚本 - 验证系统功能"""
import asyncio
import httpx
import time
import json


class SystemTester:
    """系统测试器"""
    
    def __init__(
        self,
        router_url: str = "http://localhost:8000",
        manager_url: str = "http://localhost:9000",
        worker_url: str = "http://localhost:7000"
    ):
        self.router_url = router_url
        self.manager_url = manager_url
        self.worker_url = worker_url
    
    async def test_health_checks(self):
        """测试健康检查"""
        print("\n=== 测试健康检查 ===")
        
        async with httpx.AsyncClient() as client:
            # 测试 Router
            try:
                response = await client.get(f"{self.router_url}/health")
                print(f"✓ Router 健康: {response.json()}")
            except Exception as e:
                print(f"✗ Router 不可用: {e}")
            
            # 测试 Manager
            try:
                response = await client.get(f"{self.manager_url}/health")
                print(f"✓ Manager 健康: {response.json()}")
            except Exception as e:
                print(f"✗ Manager 不可用: {e}")
            
            # 测试 Worker
            try:
                response = await client.get(f"{self.worker_url}/health")
                print(f"✓ Worker 健康: {response.json()}")
            except Exception as e:
                print(f"✗ Worker 不可用: {e}")
    
    async def test_worker_registration(self):
        """测试 Worker 注册"""
        print("\n=== 测试 Worker 列表 ===")
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(f"{self.manager_url}/workers")
                workers = response.json()
                print(f"已注册 Worker 数量: {len(workers['workers'])}")
                for worker in workers['workers']:
                    print(f"  - {worker['worker_id']}: {worker['status']}")
            except Exception as e:
                print(f"✗ 获取 Worker 列表失败: {e}")
    
    async def test_model_registration(self, alias: str = "test-model", model_name: str = "facebook/opt-125m"):
        """测试模型注册"""
        print(f"\n=== 测试模型注册: {alias} ===")
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                response = await client.post(
                    f"{self.router_url}/admin/models/register",
                    json={
                        "alias": alias,
                        "model_name": model_name,
                        "gpu_memory_utilization": 0.5,
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
                response = await client.get(f"{self.router_url}/v1/models")
                models = response.json()
                print(f"可用模型数量: {len(models['data'])}")
                for model in models['data']:
                    print(f"  - {model['id']}")
            except Exception as e:
                print(f"✗ 列出模型失败: {e}")
    
    async def test_chat_completion(self, model: str = "test-model"):
        """测试对话补全"""
        print(f"\n=== 测试对话补全: {model} ===")
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                # 等待模型启动
                print("等待模型启动...")
                for i in range(30):
                    try:
                        response = await client.get(f"{self.manager_url}/models/{model}")
                        if response.status_code == 200:
                            break
                    except:
                        pass
                    await asyncio.sleep(10)
                    print(f"  等待中... ({i+1}/30)")
                
                # 发送请求
                print("发送推理请求...")
                response = await client.post(
                    f"{self.router_url}/v1/chat/completions",
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
                response = await client.get(f"{self.router_url}/admin/status")
                status = response.json()
                print(f"Manager 状态: {status['manager']}")
                print(f"模型数量: {len(status['models']['models'])}")
                print(f"Worker 数量: {len(status['workers']['workers'])}")
            except Exception as e:
                print(f"✗ 获取系统状态失败: {e}")
    
    async def run_all_tests(self):
        """运行所有测试"""
        print("=" * 60)
        print("开始系统测试")
        print("=" * 60)
        
        # 基础测试
        await self.test_health_checks()
        await self.test_worker_registration()
        await self.test_list_models()
        await self.test_system_status()
        
        # 功能测试（可选，需要较长时间）
        # print("\n是否执行完整功能测试（包括模型注册和推理）？")
        # print("警告：这需要下载模型并启动 vLLM，可能需要几分钟")
        # response = input("输入 'yes' 继续，其他键跳过: ")
        # if response.lower() == 'yes':
        #     await self.test_model_registration()
        #     await self.test_list_models()
        #     await self.test_chat_completion()
        
        print("\n" + "=" * 60)
        print("测试完成")
        print("=" * 60)


async def main():
    """主函数"""
    import sys
    
    router_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    manager_url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:9000"
    worker_url = sys.argv[3] if len(sys.argv) > 3 else "http://localhost:7000"
    
    tester = SystemTester(
        router_url=router_url,
        manager_url=manager_url,
        worker_url=worker_url
    )
    
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
