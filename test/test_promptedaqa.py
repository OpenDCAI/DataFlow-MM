from dataflow.operators.core_audio import PromptedAQAGenerator
from dataflow.operators.conversations import Conversation2Message
from dataflow.serving import LocalModelVLMServing_vllm
from dataflow.utils.storage import FileStorage

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"  # 设置可见的GPU设备

class AQAGenerator():
    def __init__(self):
        self.storage = FileStorage(
            first_entry_file_name="./dataflow/example/audio_vqa/sample_data_local.jsonl",
            cache_path="./cache",
            file_name_prefix="audio_aqa",
            cache_type="jsonl",
        )
        self.model_cache_dir = './dataflow_cache'

        self.vlm_serving = LocalModelVLMServing_vllm(
            hf_model_name_or_path="/mnt/public/data/guotianyu/Models/Qwen2-Audio-7B-Instruct",
            hf_cache_dir=self.model_cache_dir,
            vllm_tensor_parallel_size=8,
            vllm_temperature=0.7,
            vllm_top_p=0.9,
            vllm_max_tokens=512,
            vllm_gpu_memory_utilization=0.6
        )
        # self.format_converter = Conversation2Message(
        #     image_list_key="image",
        #     video_list_key="video",
        #     audio_list_key="audio",
        #     system_prompt="你是个热心的智能助手，是Dataflow的一个组件，你的任务是回答用户的问题。",
        # )
        self.prompt_generator = PromptedAQAGenerator(
            vlm_serving = self.vlm_serving,
            system_prompt="You are a helpful agent."
        )

    def forward(self):
        # Initial filters
        # self.format_converter.run(
        #     storage= self.storage.step(),
        #     input_conversation_key="conversation",
        #     output_message_key="messages",
        # )

        self.prompt_generator.run(
            storage = self.storage.step(),
            input_audio_key="audio",
            input_conversation_key="conversation",
            output_answer_key="answer",
        )

if __name__ == "__main__":
    # This is the entry point for the pipeline
    model = AQAGenerator()
    model.forward()
