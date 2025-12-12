import argparse
from dataflow.utils.storage import FileStorage
from dataflow.serving.local_model_vlm_serving import LocalModelVLMServing_vllm
from dataflow.operators.core_vision import FixPromptedVQAGenerator
from dataflow.operators.core_vision import WikiQARefine


class ContextVQAPipeline:
    """
    一行命令即可完成图片批量 ContextVQA Caption 生成。
    """

    def __init__(
        self,
        model_path: str,
        *,
        hf_cache_dir: str | None = None,
        download_dir: str = "./ckpt",
        device: str = "cuda",
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
        self.serving = LocalModelVLMServing_vllm(
            hf_model_name_or_path=model_path,
            hf_cache_dir=hf_cache_dir,
            hf_local_dir=download_dir,
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

        self.refiner = WikiQARefine()
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
    parser = argparse.ArgumentParser(description="Batch SKVQA caption generation with DataFlow")

    parser.add_argument("--model_path", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--hf_cache_dir", default="~/.cache/huggingface")
    parser.add_argument("--download_dir", default="./ckpt")
    parser.add_argument("--device", choices=["cuda", "cpu", "mps"], default="cuda")

    parser.add_argument("--images_file", default="dataflow/example/image_to_text_pipeline/capsbench_captions.jsonl")
    parser.add_argument("--cache_path", default="./cache_local")
    parser.add_argument("--file_name_prefix", default="context_vqa")
    parser.add_argument("--cache_type", default="jsonl")

    args = parser.parse_args()

    pipe = ContextVQAPipeline(
        model_path=args.model_path,
        hf_cache_dir=args.hf_cache_dir,
        download_dir=args.download_dir,
        device=args.device,
        first_entry_file=args.images_file,
        cache_path=args.cache_path,
        file_name_prefix=args.file_name_prefix,
        cache_type=args.cache_type,
    )
    pipe.forward()