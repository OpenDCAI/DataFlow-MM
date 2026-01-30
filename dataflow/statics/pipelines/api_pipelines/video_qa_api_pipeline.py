import os

# 设置 API Key 环境变量
os.environ["DF_API_KEY"] = "your api-key"

from dataflow.operators.core_vision import PromptedVQAGenerator, VideoCaptionToQAGenerator
from dataflow.serving.api_vlm_serving_openai import APIVLMServing_openai
from dataflow.utils.storage import FileStorage
from dataflow.prompts.video import VideoCaptionGeneratorPrompt

class VideoVQAGenerator():
    def __init__(self):
        """
        Initialize VideoVQAGenerator with API model parameters.
        """
        self.storage = FileStorage(
            first_entry_file_name="./dataflow/example/video_caption/sample_data.json",
            cache_path="./cache",
            file_name_prefix="video_vqa_api",
            cache_type="json",
        )

        # Initialize VLM API serving for caption generation
        self.vlm_serving = APIVLMServing_openai(
            api_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            key_name_of_api_key="DF_API_KEY",
            model_name="qwen3-vl-8b-instruct",
            image_io=None,
            send_request_stream=False,
            max_workers=10,
            timeout=1800
        )

        self.prompt_template = VideoCaptionGeneratorPrompt()
        
        self.prompted_vqa_generator = PromptedVQAGenerator(
            serving=self.vlm_serving,
            system_prompt="You are a helpful assistant.",
            prompt_template=self.prompt_template
        )
        
        self.videocaption_to_qa_generator = VideoCaptionToQAGenerator(
            vlm_serving=self.vlm_serving,
            use_video_input=True,
        )

    def forward(self):
        # Step 1: Generate video captions using PromptedVQAGenerator
        self.prompted_vqa_generator.run(
            storage=self.storage.step(),
            input_image_key="image",
            input_video_key="video",
            input_conversation_key="conversation",
            output_answer_key="caption",
        )
        
        # Step 2: Generate QA from captions
        self.videocaption_to_qa_generator.run(
            storage = self.storage.step(),
            input_image_key="image",
            input_video_key="video",
            input_conversation_key="conversation",
            output_key="qa",
        )

if __name__ == "__main__":
    model = VideoVQAGenerator()
    model.forward()

