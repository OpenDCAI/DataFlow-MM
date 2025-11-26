import argparse
from dataflow.utils.storage import FileStorage
from dataflow.serving.local_model_vlm_serving import LocalModelVLMServing_vllm
from dataflow.operators.core_vision import ImageSKVQAGenerate


class ImageSKVQAPipeline:
    """
    一行命令即可完成图片批量 SKVQA Caption 生成。
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
        media_key: str = "image",
        output_key: str = "skvqa",
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
        self.skvqa_generator = ImageSKVQAGenerate(
            llm_serving=self.serving
        )

        self.media_key = media_key
        self.output_key = output_key

    # ------------------------------------------------------------------ #
    def forward(self):
        """
        一键跑完当前 pipeline 一个 step：
            图片 → SKVQA Caption
        """
        self.skvqa_generator.run(
            storage=self.storage.step(),
            input_modal_key=self.media_key,
            output_key=self.output_key,
        )


# ---------------------------- CLI 入口 -------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch SKVQA caption generation with DataFlow")

    parser.add_argument("--model_path", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--hf_cache_dir", default="~/.cache/huggingface")
    parser.add_argument("--download_dir", default="./ckpt")
    parser.add_argument("--device", choices=["cuda", "cpu", "mps"], default="cuda")

    parser.add_argument("--images_file", default="dataflow/example/image_to_text_pipeline/capsbench_captions.jsonl")
    parser.add_argument("--cache_path", default="./cache")
    parser.add_argument("--file_name_prefix", default="skvqa_caption")
    parser.add_argument("--cache_type", default="jsonl")
    parser.add_argument("--media_key", default="image")
    parser.add_argument("--output_key", default="skvqa")

    args = parser.parse_args()

    pipe = ImageSKVQAPipeline(
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
