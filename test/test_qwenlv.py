import os
from PIL import Image
from dataflow.serving import LocalModelVLMServing_vllm  # 按照你的模块路径修改
import base64
import io
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

def main():
    model_path = "/data0/models/Qwen2.5-VL-3B-Instruct"
    image_path = "./static/images/Face.jpg"  # 替换为你自己的测试图像路径

    # 初始化模型
    model = LocalModelVLMServing_vllm(
        hf_model_name_or_path=model_path,
        vllm_tensor_parallel_size=2,
        vllm_temperature=0.7,
        vllm_top_p=0.9,
        vllm_max_tokens=512,
        vllm_gpu_memory_utilization=0.6
    )

    # 准备输入
    messages = [
                {
                    "role": "system",
                    "content": "请描述这张图片的内容。"
                }, 
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_path,
                        }, {
                            "type": "text",
                            "text": "<t\nPlease provide a detailed and comprehensive description of the image."
                        },
                    ],
                },
            ]

    # 处理输入
    image_inputs_list = []
    video_inputs_list = []
    audio_inputs_list = []
    prompt_list = []
    for i in range(2): #改成自己需要的数据量
        image_inputs, video_inputs= process_vision_info(messages)
        
        print(image_inputs, video_inputs)

        processor = AutoProcessor.from_pretrained("/data0/models/Qwen2.5-VL-3B-Instruct")

        prompt = processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
        print(prompt)

        image_inputs_list.append(image_inputs)
        video_inputs_list.append(video_inputs)
        audio_inputs_list.append(None)
        prompt_list.append(prompt)
    
    outputs = model.generate_from_input(
        user_inputs=prompt_list,
        image_inputs=image_inputs_list,
        video_inputs=video_inputs_list,
        audio_inputs=audio_inputs_list
    )

    print("=== Model Output ===")
    print(outputs[0])

if __name__ == "__main__":
    main()
