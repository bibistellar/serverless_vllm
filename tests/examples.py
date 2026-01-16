"""示例：如何使用系统"""
import httpx
import asyncio


async def example_register_model():
    """示例：注册模型"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/admin/models/register",
            json={
                "alias": "qwen-vl-2b",
                "model_name": "Qwen/Qwen3-VL-2B-Instruct",
                "gpu_memory_utilization": 0.9,
                "max_model_len": 8192,
                "tensor_parallel_size": 1
            }
        )
        print("注册结果:", response.json())


async def example_chat_completion():
    """示例：对话补全"""
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "http://localhost:8000/v1/chat/completions",
            json={
                "model": "qwen-vl-2b",
                "messages": [
                    {"role": "system", "content": "你是一个有帮助的助手。"},
                    {"role": "user", "content": "你好，请介绍一下自己。"}
                ],
                "max_tokens": 256,
                "temperature": 0.7
            }
        )
        result = response.json()
        print("回复:", result["choices"][0]["message"]["content"])


async def example_list_models():
    """示例：列出所有模型"""
    async with httpx.AsyncClient() as client:
        response = await client.get("http://localhost:8000/v1/models")
        models = response.json()
        print("可用模型:")
        for model in models["data"]:
            print(f"  - {model['id']}")


async def example_get_status():
    """示例：获取系统状态"""
    async with httpx.AsyncClient() as client:
        response = await client.get("http://localhost:8000/admin/status")
        status = response.json()
        print("系统状态:", status)


async def example_stream_completion():
    """示例：流式对话补全"""
    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream(
            "POST",
            "http://localhost:8000/v1/chat/completions",
            json={
                "model": "qwen-vl-2b",
                "messages": [
                    {"role": "user", "content": "讲一个故事"}
                ],
                "stream": True
            }
        ) as response:
            print("流式响应:")
            async for line in response.aiter_lines():
                if line:
                    print(line)


if __name__ == "__main__":
    print("=== 示例 1: 注册模型 ===")
    # asyncio.run(example_register_model())
    
    print("\n=== 示例 2: 列出模型 ===")
    asyncio.run(example_list_models())
    
    print("\n=== 示例 3: 对话补全 ===")
    # asyncio.run(example_chat_completion())
    
    print("\n=== 示例 4: 获取状态 ===")
    asyncio.run(example_get_status())
    
    print("\n=== 示例 5: 流式补全 ===")
    # asyncio.run(example_stream_completion())
