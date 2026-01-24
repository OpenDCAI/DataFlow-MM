import os

# 设置 API Key 环境变量
os.environ["DF_API_KEY"] = "your api-key"

import argparse
from typing import Any, List

from dataflow.utils.storage import FileStorage
from dataflow.serving.api_vlm_serving_openai import APIVLMServing_openai
from dataflow.operators.core_vision import ImageCaptionGenerator


class ImageCaptioningPipeline:
    """
    一行命令即可完成图片批量 Caption 生成。
    """

    def __init__(
        self,
        model_path: str,
        *,
        hf_cache_dir: str | None = None,
        download_dir: str = "./ckpt",
        device: str = "cuda",
        first_entry_file: str = "dataflow/example/image_to_text_pipeline/capsbench_captions.jsonl",
        cache_path: str = "./cache_local",
        file_name_prefix: str = "dataflow_cache_step",
        cache_type: str = "jsonl",
        media_key: str = "image",
        output_key: str = "caption",
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
            api_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            key_name_of_api_key="DF_API_KEY",
            model_name="qwen3-vl-8b-instruct",
            image_io=None,
            send_request_stream=False,
            max_workers=10,
            timeout=1800
        )

        # ---------- 3. Operator ----------
        self.caption_generator = ImageCaptionGenerator(
            llm_serving=self.vlm_serving,
            system_prompt="You are a image caption generator. Your task is to generate a concise and informative caption for the given image content.",
        )
        
        self.media_key = media_key
        self.output_key = output_key

    # ------------------------------------------------------------------ #
    def forward(self):
        """
        一键跑完当前 pipeline 一个 step：
            图片 → Caption
        """
        self.caption_generator.run(
            storage=self.storage.step(),       # Pipeline 负责推进 step
            input_modal_key=self.media_key,
            output_key=self.output_key,
        )


# ---------------------------- CLI 入口 -------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch image-to-caption with DataFlow")

    parser.add_argument("--model_path", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--hf_cache_dir", default="~/.cache/huggingface")
    parser.add_argument("--download_dir", default="./ckpt")
    parser.add_argument("--device", choices=["cuda", "cpu", "mps"], default="cuda")

    parser.add_argument("--images_file", default="./dataflow/example/image_to_text_pipeline/capsbench_captions.json")
    parser.add_argument("--cache_path", default="./cache_local")
    parser.add_argument("--file_name_prefix", default="caption")
    parser.add_argument("--cache_type", default="json")
    parser.add_argument("--media_key", default="image")
    parser.add_argument("--output_key", default="caption")

    args = parser.parse_args()

    pipe = ImageCaptioningPipeline(
        model_path=args.model_path,
        hf_cache_dir=args.hf_cache_dir,
        download_dir=args.download_dir,
        device=args.device,
        first_entry_file=args.images_file,
        cache_path=args.cache_path,
        file_name_prefix=args.file_name_prefix,
        cache_type=args.cache_type,
        media_key=args.media_key,
        output_key=args.output_key,
    )
    pipe.forward()
