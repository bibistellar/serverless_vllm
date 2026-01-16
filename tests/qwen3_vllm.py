# -*- coding: utf-8 -*-
import os
# 必须在导入其他库之前设置环境变量
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
# os.environ['HF_HUB_OFFLINE'] = '1'
# os.environ['TRANSFORMERS_OFFLINE'] = '1'
# os.environ['HF_DATASETS_OFFLINE'] = '1'

import asyncio
import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.inputs.data import TextPrompt

def prepare_inputs_for_vllm(messages, processor):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # qwen_vl_utils 0.0.14+ reqired
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True
    )
    print(f"video_kwargs: {video_kwargs}")

    mm_data = {}
    if image_inputs is not None:
        mm_data['image'] = image_inputs
    if video_inputs is not None:
        mm_data['video'] = video_inputs

    return {
        'prompt': text,
        'multi_modal_data': mm_data,
        'mm_processor_kwargs': video_kwargs
    }


async def main():
    # messages = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {
    #                 "type": "video",
    #                 "video": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-VL/space_woaudio.mp4",
    #             },
    #             {"type": "text", "text": "这段视频有多长"},
    #         ],
    #     }
    # ]

    messages = [
        {
            "role": "user",
            "content": [
              {
                  "type": "image",
                  "image": "demo.jpg",
              },
              {"type": "text", "text": "Describe this image."},
            ],
        }
    ]

    # TODO: change to your own checkpoint path
    checkpoint_path = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203/")
    model_path = checkpoint_path if os.path.isdir(checkpoint_path) else "Qwen/Qwen3-VL-2B-Instruct"
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=os.path.isdir(checkpoint_path))
    inputs = [prepare_inputs_for_vllm(message, processor) for message in [messages]]

    # Initialize AsyncEngineArgs
    engine_args = AsyncEngineArgs(
        model=model_path,
        mm_encoder_tp_mode="data",
        enable_expert_parallel=False,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.75,
        max_model_len=4096,
        seed=0,
        trust_remote_code=True
    )
    
    # Create AsyncLLMEngine
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=1024,
        top_k=-1,
        stop_token_ids=[],
    )

    for i, input_ in enumerate(inputs):
        print()
        print('=' * 40)
        print(f"Inputs[{i}]: {input_['prompt']=!r}")
    print('\n' + '>' * 40)

    # Generate for each input
    for i, input_ in enumerate(inputs):
        request_id = f"request-{i}"
        
        # Build TextPrompt for multimodal input
        if input_.get('multi_modal_data'):
            prompt_input = TextPrompt(
                prompt=input_['prompt'],
                multi_modal_data=input_['multi_modal_data'],
                mm_processor_kwargs=input_.get('mm_processor_kwargs')
            )
        else:
            prompt_input = input_['prompt']
        
        # Start generation
        results_generator = engine.generate(
            prompt_input,
            sampling_params,
            request_id
        )
        
        # Stream output in real-time
        print()
        print('=' * 40)
        print(f"Streaming output:")
        print('-' * 40)
        
        previous_text = ""
        async for request_output in results_generator:
            # Get the current generated text
            current_text = request_output.outputs[0].text
            
            # Print only the new tokens (delta)
            new_text = current_text[len(previous_text):]
            if new_text:
                print(new_text, end='', flush=True)
            
            previous_text = current_text
        
        # Print final newline
        print()
        print('=' * 40)


if __name__ == '__main__':
    asyncio.run(main())