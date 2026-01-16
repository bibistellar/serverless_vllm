#!/usr/bin/env python3
"""测试tokenizer获取"""

import asyncio
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs

async def test_tokenizer():
    """测试从AsyncLLMEngine获取tokenizer"""
    
    # 创建引擎参数
    engine_args = AsyncEngineArgs(
        model="Qwen/Qwen2.5-0.5B-Instruct",  # 使用小模型测试
        trust_remote_code=True
    )
    
    # 创建引擎
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    
    # 获取tokenizer (同步方法)
    tokenizer = engine.get_tokenizer()
    
    print(f"✅ 成功获取tokenizer: {type(tokenizer)}")
    print(f"词汇表大小: {tokenizer.vocab_size}")
    
    # 测试apply_chat_template
    messages = [
        {"role": "user", "content": "你好"}
    ]
    
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    print(f"\n生成的prompt:\n{prompt}")
    
    # 关闭引擎
    # Note: AsyncLLMEngine可能没有shutdown方法，直接退出即可

if __name__ == "__main__":
    asyncio.run(test_tokenizer())
