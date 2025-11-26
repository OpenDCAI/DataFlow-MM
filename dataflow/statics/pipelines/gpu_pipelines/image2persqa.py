import argparse
from dataflow.utils.storage import FileStorage
from dataflow.serving.local_model_vlm_serving import LocalModelVLMServing_vllm
from dataflow.prompts.image import QAGeneratorPrompt          
from dataflow.operators.core_vision import PersQAGenerate

class PersQAPipeline:
    """批量图片 → 定制化QA 对 (自动出题 + 作答)"""

    def __init__(
        self,
        model_path: str,
        *,
        hf_cache_dir: str | None = None,
        download_dir: str = "./ckpt/models",
        device: str = "cuda",
        first_entry_file: str = "./dataflow/example/Image2TextPipeline/test_image2qa.jsonl",
        cache_path: str = "./cache_local",
        file_name_prefix: str = "dataflow_cache_step",
        cache_type: str = "jsonl",
        media_key: str = "image",
        output_key: str = "qa",                    # 存放 {question:<>, answer:<>} 或字符串
    ):
        # 1. Storage
        self.storage = FileStorage(
            first_entry_file_name=first_entry_file,
            cache_path=cache_path,
            file_name_prefix=file_name_prefix,
            cache_type=cache_type
        )

        # 2. Serving
        self.serving = LocalModelVLMServing_vllm(
            hf_model_name_or_path=model_path,
            hf_cache_dir=hf_cache_dir,
            hf_local_dir=download_dir,
            vllm_tensor_parallel_size=1,
            vllm_temperature=0.7,
            vllm_top_p=0.9,
            vllm_max_tokens=128,          # 适度长度
        )

        # 3. Operator
        self.qa_generator = PersQAGenerate(
            llm_serving=self.serving,
        )

        self.media_key = media_key
        self.output_key = output_key

    # ------------------------- Pipeline 单步 ------------------------- #
    def forward(self):
        self.qa_generator.run(
            storage=self.storage.step(),
            input_modal_key=self.media_key,
            output_key=self.output_key,
        )

# ------------------------------ CLI ------------------------------ #
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Batch image → QA pair (auto)")

    parser.add_argument("--model_path", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--hf_cache_dir", default="~/.cache/huggingface")
    parser.add_argument("--download_dir", default="./ckpt/models")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")

    parser.add_argument("--qa_file", default="./dataflow/example/image_to_text_pipeline/capsbench_qas.jsonl")
    parser.add_argument("--cache_path", default="./cache_local")
    parser.add_argument("--file_name_prefix", default="persqa")
    parser.add_argument("--cache_type", default="jsonl")
    parser.add_argument("--media_key", default="image")
    parser.add_argument("--output_key", default="pers_qa")

    args = parser.parse_args()

    pipe = PersQAPipeline(
        model_path=args.model_path,
        hf_cache_dir=args.hf_cache_dir,
        download_dir=args.download_dir,
        device=args.device,
        first_entry_file=args.qa_file,
        cache_path=args.cache_path,
        file_name_prefix=args.file_name_prefix,
        cache_type=args.cache_type,
        media_key=args.media_key,
        output_key=args.output_key,
    )
    pipe.forward()
