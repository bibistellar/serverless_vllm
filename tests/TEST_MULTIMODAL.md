# 多模态推理测试

测试Worker对图片+文本多模态输入的支持。

## 功能说明

已实现以下多模态功能：

1. ✅ 支持图片URL（HTTP/HTTPS）
2. ✅ 支持base64编码的图片
3. ✅ 支持单张或多张图片
4. ✅ 向后兼容纯文本模式
5. ✅ 使用tokenizer的chat template
6. ✅ 自动下载和处理图片
7. ✅ 传递多模态数据给vLLM引擎

## 前提条件

1. 启动Ray服务和Worker：
```bash
# 终端1: 启动Ray服务
python3 start_ray_services.py

# 终端2: 启动Worker（如果独立部署）
python -m src.worker.service
```

2. 注册支持视觉的模型：
```bash
curl -X POST http://localhost:18000/v1/models/register \
  -H "Content-Type: application/json" \
  -d '{
    "alias": "qwen-vl-2b",
    "model_name": "Qwen/Qwen3-VL-2B-Instruct",
    "max_model_len": 8192,
    "gpu_memory_utilization": 0.5
  }'
```

## 使用方法

### 方法1: Python测试脚本（推荐）

```bash
# 安装依赖
pip install openai pillow httpx

# 运行测试
cd /root/Code/serverless_vllm
python tests/test_multimodal.py
```

测试内容：
- 测试1: 单张图片理解
- 测试2: 多张图片对比分析
- 测试3: 纯文本模式（确保向后兼容）
- 测试4: 本地图片base64编码

### 方法2: Shell脚本（curl）

```bash
cd /root/Code/serverless_vllm
./tests/test_multimodal.sh
```

### 方法3: 直接curl命令

**单张图片：**
```bash
curl -X POST http://localhost:18000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-vl-2b",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "image_url",
            "image_url": {
              "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
            }
          },
          {
            "type": "text",
            "text": "请描述这张图片的内容。"
          }
        ]
      }
    ],
    "max_tokens": 512
  }'
```

**多张图片：**
```bash
curl -X POST http://localhost:18000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-vl-2b",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "image_url",
            "image_url": {"url": "https://example.com/image1.jpg"}
          },
          {
            "type": "image_url",
            "image_url": {"url": "https://example.com/image2.jpg"}
          },
          {
            "type": "text",
            "text": "比较这两张图片的异同。"
          }
        ]
      }
    ],
    "max_tokens": 512
  }'
```

**使用base64编码的本地图片：**
```python
import base64
from openai import OpenAI

# 读取本地图片并编码
with open("image.jpg", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode()
    image_url = f"data:image/jpeg;base64,{image_base64}"

client = OpenAI(
    api_key="EMPTY",
    base_url="http://127.0.0.1:18000/v1"
)

response = client.chat.completions.create(
    model="qwen-vl-2b",
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "text", "text": "描述这张图片"}
        ]
    }]
)

print(response.choices[0].message.content)
```

## 支持的图片格式

- JPEG (.jpg, .jpeg)
- PNG (.png)
- GIF (.gif)
- WebP (.webp)

## OpenAI兼容格式

完全兼容OpenAI的Chat Completions API格式：

```json
{
  "model": "qwen-vl-2b",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "文本内容"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "https://example.com/image.jpg"
          }
        }
      ]
    }
  ]
}
```

## 技术实现

1. **图片处理** (`openai_protocol.py`)
   - `load_image_from_url()`: 从URL或base64加载图片
   - `process_multimodal_content()`: 处理混合content，提取文本和图片
   - `messages_to_prompt_and_images()`: 异步处理消息，返回prompt和图片列表

2. **vLLM集成** (`vllm_manager.py`)
   - `generate()`: 支持`multi_modal_data`参数
   - 传递格式: `{"image": [PIL.Image, ...]}`

3. **Worker服务** (`service.py`)
   - 使用异步函数处理图片
   - 构造multi_modal_data传递给vLLM引擎
   - 保持向后兼容纯文本模式

## 故障排查

**问题1: 模型不支持多模态**
```
错误: Model does not support multi-modal inputs
```
解决: 确保使用支持视觉的模型，如 Qwen3-VL 系列

**问题2: 图片下载失败**
```
错误: Failed to load image from URL
```
解决: 检查URL是否可访问，或使用base64编码的图片

**问题3: 内存不足**
```
错误: CUDA out of memory
```
解决: 降低`gpu_memory_utilization`或减少`max_model_len`

## 性能提示

- 大图片会增加处理时间，建议使用适当尺寸（如800x600）
- 多张图片会增加显存占用
- 首次请求需要加载模型，后续请求会更快
- 使用base64编码会增加请求大小，URL方式更高效
