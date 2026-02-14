import os
os.environ["DF_API_KEY"] = "sk-xxxx"

import argparse
from dataflow.operators.core_vision.generate.image_bbox_generator import (
    ImageBboxGenerator, 
    ExistingBBoxDataGenConfig
)
from dataflow.operators.core_vision.generate.prompted_vqa_generator import (
    PromptedVQAGenerator
)
from dataflow.utils.storage import FileStorage

from dataflow.serving.api_vlm_serving_openai import APIVLMServing_openai
class ImageRegionCaptionPipeline:
    def __init__(
        self,
        first_entry_file: str = "./data/image_region_caption/image_region_caption_demo.jsonl",
        cache_path: str = "./cache/image_region_caption",
        file_name_prefix: str = "region_caption",
        cache_type: str = "jsonl",
        input_image_key: str = "image",
        input_bbox_key: str = "bbox",
        image_with_bbox_path: str = 'image_with_bbox',
        max_boxes: int = 10,
        output_image_with_bbox_path: str = "./cache/image_region_caption/image_with_bbox_result.jsonl",
    ):
        self.bbox_storage = FileStorage(
            first_entry_file_name=first_entry_file,
            cache_path=cache_path,
            file_name_prefix=file_name_prefix,
            cache_type=cache_type
        )

        self.cfg = ExistingBBoxDataGenConfig(
            max_boxes=max_boxes,
            input_jsonl_path=first_entry_file,
            output_jsonl_path=output_image_with_bbox_path,
        )

        self.caption_storage = FileStorage(
            first_entry_file_name=output_image_with_bbox_path,
            cache_path=cache_path,
            file_name_prefix=file_name_prefix,
            cache_type=cache_type
        )
        self.vlm_serving = APIVLMServing_openai(
            api_url="https://dashscope.aliyuncs.com/compatible-mode/v1", # Any API platform compatible with OpenAI format
            model_name="gpt-4o-mini",
            image_io=None,
            send_request_stream=False,
            max_workers=10,
            timeout=1800
        )
        self.bbox_generator = ImageBboxGenerator(config=self.cfg)
        self.caption_generator = PromptedVQAGenerator(serving=self.vlm_serving,system_prompt="You are a helpful assistant.")
        self.input_image_key = input_image_key
        self.input_bbox_key = input_bbox_key
        self.image_with_bbox_path=image_with_bbox_path
        self.bbox_record=None

    def forward(self):
        self.bbox_generator.run(
            storage=self.bbox_storage.step(),
            input_image_key=self.input_image_key,
            input_bbox_key=self.input_bbox_key
        )

        self.caption_generator.run(
            storage=self.caption_storage.step(),
            input_image_key='image_with_bbox',
            input_prompt_key='prompt'
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image region caption with DataFlow")
    parser.add_argument("--first_entry_file", default="./data/image_region_caption/image_region_caption_demo.jsonl")
    parser.add_argument("--cache_path", default="./cache/image_region_caption")
    parser.add_argument("--file_name_prefix", default="region_caption")
    parser.add_argument("--cache_type", default="jsonl")
    parser.add_argument("--input_image_key", default="image")
    parser.add_argument("--input_bbox_key", default="bbox")

    parser.add_argument("--max_boxes", type=int, default=10)
    parser.add_argument("--output_image_with_bbox_path", default="./cache/image_region_caption/image_with_bbox_result.jsonl")

    args = parser.parse_args()

    pipe = ImageRegionCaptionPipeline(
        first_entry_file=args.first_entry_file,
        cache_path=args.cache_path,
        file_name_prefix=args.file_name_prefix,
        cache_type=args.cache_type,
        input_image_key=args.input_image_key,
        input_bbox_key=args.input_bbox_key,
        max_boxes=args.max_boxes,
        output_image_with_bbox_path=args.output_image_with_bbox_path,
    )
    pipe.forward()