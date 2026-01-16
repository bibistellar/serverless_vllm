#!/bin/bash
# 多模态推理测试 - curl 版本

echo "==========================================="
echo "测试 1: 使用图片URL进行推理"
echo "==========================================="

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

echo -e "\n\n==========================================="
echo "测试 2: 多张图片理解"
echo "==========================================="

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
            "type": "image_url",
            "image_url": {
              "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo2.jpeg"
            }
          },
          {
            "type": "text",
            "text": "这两张图片有什么不同？"
          }
        ]
      }
    ],
    "max_tokens": 512
  }'

echo -e "\n\n==========================================="
echo "测试 3: 纯文本（向后兼容性）"
echo "==========================================="

curl -X POST http://localhost:18000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-vl-2b",
    "messages": [
      {
        "role": "user",
        "content": "请用一句话介绍你自己。"
      }
    ],
    "max_tokens": 100
  }'

echo -e "\n\n==========================================="
echo "测试完成！"
echo "==========================================="
