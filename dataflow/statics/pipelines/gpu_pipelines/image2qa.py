import argparse
from typing import Any, List

from dataflow.utils.storage import FileStorage
from dataflow.core import LLMServingABC
from dataflow.serving.local_model_vlm_serving import LocalModelVLMServing_vllm   
from dataflow.operators.core_vision import PromptedVQAGenerator

class ImageQAPipeline:
    """批量图片 → QA 对 (自动出题 + 作答)"""

    def __init__(self, llm_serving: LLMServingABC = None):

        # ---------- 1. Storage ----------
        self.storage = FileStorage(
            first_entry_file_name="./example_data/image_vqa/sample_data.json",
            cache_path="./cache_local",
            file_name_prefix="qa",
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

        # 3. Operator
        self.qa_generator = PromptedVQAGenerator(
            serving=self.serving,
            system_prompt="You are a image question-answer generator. Your task is to generate a question-answer pair for the given image content.",
        )

        self.media_key = "image"
        self.output_key = "qa"

    # ------------------------- Pipeline 单步 ------------------------- #
    def forward(self):
        self.qa_generator.run(
            storage=self.storage.step(),
            input_conversation_key="conversation",
            input_image_key=self.media_key,
            output_answer_key=self.output_key,
        )

# ------------------------------ CLI ------------------------------ #
if __name__ == "__main__":
    pipe = ImageQAPipeline()
    pipe.forward()