"""测试多模态（图片+文本）推理功能

使用本地图片测试Worker的多模态能力
"""
import base64
import time
from pathlib import Path
from openai import OpenAI


def encode_image_to_base64(image_path: str) -> str:
    """将本地图片编码为base64字符串
    
    Args:
        image_path: 图片文件路径
        
    Returns:
        data URI格式的base64字符串
    """
    with open(image_path, "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode('utf-8')
    
    # 根据文件扩展名确定MIME类型
    ext = Path(image_path).suffix.lower()
    mime_type = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp'
    }.get(ext, 'image/jpeg')
    
    return f"data:{mime_type};base64,{image_data}"


def test_single_image():
    """测试单张图片的理解"""
    print("\n" + "="*60)
    print("测试 1: 单张图片理解")
    print("="*60)
    
    # 创建客户端
    client = OpenAI(
        api_key="EMPTY",
        base_url="http://127.0.0.1:18000/v1",
        timeout=600
    )
    
    # 准备测试图片（这里使用一个示例，需要替换为实际图片）
    # 你可以下载一张测试图片，或者使用URL
    image_url = "https://ofasys-multimodal-wlcb-3-toshanghai.oss-accelerate.aliyuncs.com/wpf272043/keepme/image/receipt.png"
    
    # 或者使用本地图片（如果有的话）
    # local_image_path = "/path/to/your/image.jpg"
    # image_base64 = encode_image_to_base64(local_image_path)
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    }
                },
                {
                    "type": "text",
                    "text": "请描述这张图片的内容，尽可能详细。"
                }
            ]
        }
    ]
    
    print(f"发送请求到模型...")
    start = time.time()
    
    try:
        response = client.chat.completions.create(
            model="qwen-vl-2b",
            messages=messages,
            max_tokens=512
        )
        
        elapsed = time.time() - start
        print(f"\n✅ 请求成功 (耗时: {elapsed:.2f}秒)")
        print(f"\n模型回复:")
        print("-" * 60)
        print(response.choices[0].message.content)
        print("-" * 60)
        
    except Exception as e:
        print(f"\n❌ 请求失败: {e}")


def test_multiple_images():
    """测试多张图片的理解"""
    print("\n" + "="*60)
    print("测试 2: 多张图片理解")
    print("="*60)
    
    client = OpenAI(
        api_key="EMPTY",
        base_url="http://127.0.0.1:18000/v1",
        timeout=600
    )
    
    # 使用多张图片（URL或base64）
    images = [
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo2.jpeg"
    ]
    
    # 构造包含多张图片的消息
    content = []
    for i, img_url in enumerate(images, 1):
        content.append({
            "type": "image_url",
            "image_url": {"url": img_url}
        })
    
    content.append({
        "type": "text",
        "text": "这些图片有什么共同点？请对比分析。"
    })
    
    messages = [{"role": "user", "content": content}]
    
    print(f"发送包含 {len(images)} 张图片的请求...")
    start = time.time()
    
    try:
        response = client.chat.completions.create(
            model="qwen-vl-2b",
            messages=messages,
            max_tokens=512
        )
        
        elapsed = time.time() - start
        print(f"\n✅ 请求成功 (耗时: {elapsed:.2f}秒)")
        print(f"\n模型回复:")
        print("-" * 60)
        print(response.choices[0].message.content)
        print("-" * 60)
        
    except Exception as e:
        print(f"\n❌ 请求失败: {e}")


def test_text_only():
    """测试纯文本（确保向后兼容）"""
    print("\n" + "="*60)
    print("测试 3: 纯文本模式（向后兼容性）")
    print("="*60)
    
    client = OpenAI(
        api_key="EMPTY",
        base_url="http://127.0.0.1:18000/v1",
        timeout=600
    )
    
    messages = [
        {
            "role": "user",
            "content": "请用一句话介绍你自己。"
        }
    ]
    
    print(f"发送纯文本请求...")
    start = time.time()
    
    try:
        response = client.chat.completions.create(
            model="qwen-vl-2b",
            messages=messages,
            max_tokens=100
        )
        
        elapsed = time.time() - start
        print(f"\n✅ 请求成功 (耗时: {elapsed:.2f}秒)")
        print(f"\n模型回复:")
        print("-" * 60)
        print(response.choices[0].message.content)
        print("-" * 60)
        
    except Exception as e:
        print(f"\n❌ 请求失败: {e}")


def test_with_local_image():
    """使用本地图片测试（需要提供本地图片路径）"""
    print("\n" + "="*60)
    print("测试 4: 本地图片base64编码")
    print("="*60)
    
    # 检查是否有测试图片
    test_image_paths = [
        "tests/test_image.jpg",
        "tests/test_image.png",
        "demo.jpg",
        "demo.png"
    ]
    
    local_image = None
    for path in test_image_paths:
        if Path(path).exists():
            local_image = path
            break
    
    if not local_image:
        print("⚠️  未找到本地测试图片，跳过此测试")
        print(f"提示: 可以在以下位置放置测试图片:")
        for path in test_image_paths:
            print(f"  - {path}")
        return
    
    print(f"使用本地图片: {local_image}")
    
    client = OpenAI(
        api_key="EMPTY",
        base_url="http://127.0.0.1:18000/v1",
        timeout=600
    )
    
    # 编码图片
    image_base64 = encode_image_to_base64(local_image)
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_base64
                    }
                },
                {
                    "type": "text",
                    "text": "这张图片里有什么？"
                }
            ]
        }
    ]
    
    print(f"发送请求...")
    start = time.time()
    
    try:
        response = client.chat.completions.create(
            model="qwen-vl-2b",
            messages=messages,
            max_tokens=512
        )
        
        elapsed = time.time() - start
        print(f"\n✅ 请求成功 (耗时: {elapsed:.2f}秒)")
        print(f"\n模型回复:")
        print("-" * 60)
        print(response.choices[0].message.content)
        print("-" * 60)
        
    except Exception as e:
        print(f"\n❌ 请求失败: {e}")


def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("多模态推理测试")
    print("="*60)
    print("\n说明:")
    print("1. 确保已启动 Ray 服务和 Worker")
    print("2. 确保已注册 qwen-vl-2b 模型")
    print("3. 测试将依次进行")
    print("\n开始测试...\n")
    
    # 测试1: 单张图片
    try:
        test_single_image()
    except Exception as e:
        print(f"测试1失败: {e}")
    
    time.sleep(2)
    
    # 测试2: 多张图片
    try:
        test_multiple_images()
    except Exception as e:
        print(f"测试2失败: {e}")
    
    time.sleep(2)
    
    # 测试3: 纯文本
    try:
        test_text_only()
    except Exception as e:
        print(f"测试3失败: {e}")
    
    time.sleep(2)
    
    # 测试4: 本地图片
    try:
        test_with_local_image()
    except Exception as e:
        print(f"测试4失败: {e}")
    
    print("\n" + "="*60)
    print("所有测试完成！")
    print("="*60)


if __name__ == "__main__":
    main()
