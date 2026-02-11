import os

# 设置 API Key 环境变量
os.environ["DF_API_KEY"] = "sk-xxx"

import argparse
from dataflow.utils.storage import FileStorage
from dataflow.serving.api_vlm_serving_openai import APIVLMServing_openai
from dataflow.operators.core_vision import PromptedVQAGenerator


class ImageCaptionPipeline:
    """
    一行命令即可完成图片批量 Caption 生成。
    """

    def __init__(
        self,
        first_entry_file: str = "dataflow/example/image_to_text_pipeline/capsbench_captions.jsonl",
        cache_path: str = "./cache_local_skvqa",
        file_name_prefix: str = "skvqa_cache_step",
        cache_type: str = "jsonl",
    ):
        # ---------- 1. Storage ----------
        self.storage = FileStorage(
            first_entry_file_name=first_entry_file,
            cache_path=cache_path,
            file_name_prefix=file_name_prefix,
            cache_type=cache_type,
        )

        # ---------- 2. Serving ----------
        self.vlm_serving = APIVLMServing_openai(
            api_url="http://172.96.141.132:3001/v1", # Any API platform compatible with OpenAI format
            key_name_of_api_key="DF_API_KEY", # Set the API key for the corresponding platform in the environment variable or line 4
            model_name="gpt-5-nano-2025-08-07",
            image_io=None,
            send_request_stream=False,
            max_workers=10,
            timeout=1800
        )

        # ---------- 3. Operator ----------
        self.vqa_generator = PromptedVQAGenerator(
            serving=self.vlm_serving,
            system_prompt= "You are a image caption generator. Your task is to generate a concise and informative caption for the given image content."
        )

    # ------------------------------------------------------------------ #
    def forward(self):
        input_image_key = "image"
        output_answer_key = "caption"

        self.vqa_generator.run(
            storage=self.storage.step(),
            input_conversation_key="conversation",
            input_image_key=input_image_key,
            output_answer_key=output_answer_key,
        )

# ---------------------------- CLI 入口 -------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch image caption generation with DataFlow")

    parser.add_argument("--images_file", default="data/image_caption/sample_data.json")
    parser.add_argument("--cache_path", default="./cache_local")
    parser.add_argument("--file_name_prefix", default="caption")
    parser.add_argument("--cache_type", default="json")

    args = parser.parse_args()

    pipe = ImageCaptionPipeline(
        first_entry_file=args.images_file,
        cache_path=args.cache_path,
        file_name_prefix=args.file_name_prefix,
        cache_type=args.cache_type,
    )
    pipe.forward()