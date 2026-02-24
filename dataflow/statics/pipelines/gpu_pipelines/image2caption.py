import argparse
from typing import Any, List

from dataflow.utils.storage import FileStorage
from dataflow.core import LLMServingABC
from dataflow.serving.local_model_vlm_serving import LocalModelVLMServing_vllm
from dataflow.operators.core_vision import PromptedVQAGenerator


class ImageCaptioningPipeline:
    """
    一行命令即可完成图片批量 Caption 生成。
    """

    def __init__(self, llm_serving: LLMServingABC = None):

        # ---------- 1. Storage ----------
        self.storage = FileStorage(
            first_entry_file_name="./example_data/image_caption/sample_data.json",
            cache_path="./cache_local",
            file_name_prefix="caption",
            cache_type="json",
        )

        # ---------- 2. Serving ----------
        self.serving = LocalModelVLMServing_vllm(
            hf_model_name_or_path="/mnt/bn/yufei1900/public_weight/Qwen2.5-VL-3B-Instruct",
            hf_cache_dir="~/.cache/huggingface",
            hf_local_dir="./ckpt",
            vllm_tensor_parallel_size=1,
            vllm_temperature=0.7,
            vllm_top_p=0.9,
            vllm_max_tokens=256,
        )

        # ---------- 3. Operator ----------
        self.caption_generator = PromptedVQAGenerator(
            serving=self.serving,
            system_prompt="You are a image caption generator. Your task is to generate a concise and informative caption for the given image content.",
        )
        
        self.media_key = "image"
        self.output_key = "caption"

    # ------------------------------------------------------------------ #
    def forward(self):
        """
        一键跑完当前 pipeline 一个 step：
            图片 → Caption
        """
        self.caption_generator.run(
            storage=self.storage.step(),       # Pipeline 负责推进 step
            input_conversation_key="conversation",
            input_image_key=self.media_key,
            output_answer_key=self.output_key,
        )


# ---------------------------- CLI 入口 -------------------------------- #
if __name__ == "__main__":
    pipe = ImageCaptioningPipeline()
    pipe.forward()