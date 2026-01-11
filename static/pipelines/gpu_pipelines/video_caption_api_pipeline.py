import os

# 设置 API Key 环境变量
os.environ["DF_API_KEY"] = "your api key"

from dataflow.operators.core_vision import VideoToCaptionGenerator
from dataflow.serving.api_vlm_serving_openai import APIVLMServing_openai
from dataflow.utils.storage import FileStorage

class VideoCaptionGenerator():
    def __init__(self):
        self.storage = FileStorage(
            first_entry_file_name="./dataflow/example/video_caption/sample_data.json",
            cache_path="./cache",
            file_name_prefix="video_caption_api",
            cache_type="json",
        )

        self.vlm_serving = APIVLMServing_openai(
            api_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            key_name_of_api_key="DF_API_KEY",
            model_name="qwen3-vl-8b-instruct",
            image_io=None,
            send_request_stream=False,
            max_workers=10,
            timeout=1800
        )

        self.video_to_caption_generator = VideoToCaptionGenerator(
            vlm_serving = self.vlm_serving,
        )

    def forward(self):
        self.video_to_caption_generator.run(
            storage = self.storage.step(),
            input_image_key="image",
            input_video_key="video",
            input_conversation_key="conversation",
            output_key="caption",
        )

if __name__ == "__main__":
    # This is the entry point for the pipeline
    model = VideoCaptionGenerator()
    model.forward()

