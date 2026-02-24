import argparse
from dataflow.utils.storage import FileStorage
from dataflow.core import LLMServingABC
from dataflow.serving.local_model_vlm_serving import LocalModelVLMServing_vllm
from dataflow.operators.core_vision import FixPromptedVQAGenerator
from dataflow.operators.core_vision import WikiQARefiner


class ContextVQAPipeline:
    """
    一行命令即可完成图片批量 ContextVQA Caption 生成。
    """

    def __init__(self, llm_serving: LLMServingABC = None):
        # ---------- 1. Storage ----------
        self.storage = FileStorage(
            first_entry_file_name="./example_data/image_contextvqa/sample_data.json",
            cache_path="./cache_local",
            file_name_prefix="context_vqa",
            cache_type="json",
        )

        # ---------- 2. Serving ----------
        self.serving = LocalModelVLMServing_vllm(
            hf_model_name_or_path="Qwen/Qwen2.5-VL-3B-Instruct",
            hf_cache_dir="~/.cache/huggingface",
            hf_local_dir="./ckpt",
            vllm_tensor_parallel_size=1,
            vllm_temperature=0.7,
            vllm_top_p=0.9,
            vllm_max_tokens=512,
        )

        # ---------- 3. Operator ----------
        self.vqa_generator = FixPromptedVQAGenerator(
            serving=self.serving,
            system_prompt="You are a helpful assistant.",
            user_prompt= """
            Write a Wikipedia article related to this image without directly referring to the image. Then write question answer pairs. The question answer pairs should satisfy the following criteria.
            1: The question should refer to the image.
            2: The question should avoid mentioning the name of the object in the image.
            3: The question should be answered by reasoning over the Wikipedia article.
            4: The question should sound natural and concise.
            5: The answer should be extracted from the Wikipedia article.
            6: The answer should not be any objects in the image.
            7: The answer should be a single word or phrase and list all correct answers separated by commas.
            8: The answer should not contain 'and', 'or', rather you can split them into multiple answers.
            """
        )

        self.refiner = WikiQARefiner()
    # ------------------------------------------------------------------ #
    def forward(self):
        input_image_key = "image"
        output_answer_key = "vqa"
        output_wiki_key = "context_vqa"

        self.vqa_generator.run(
            storage=self.storage.step(),
            input_image_key=input_image_key,
            output_answer_key=output_answer_key
        )

        self.refiner.run(
            storage=self.storage.step(),
            input_key=output_answer_key,
            output_key=output_wiki_key
        )

# ---------------------------- CLI 入口 -------------------------------- #
if __name__ == "__main__":
    pipe = ContextVQAPipeline()
    pipe.forward()