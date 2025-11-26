import argparse
from typing import Any, List, Optional

from dataflow.utils.storage import FileStorage
from dataflow.serving.local_model_vlm_serving import LocalModelVLMServing_vllm
from dataflow.operators.core_vision import ImageScaleCaptionGenerate, ImageScaleCaptionGenerateConfig


class ScaleCapPipeline:
    """
    一行命令即可把 {"image": "..."} jsonl 处理成 ScaleCap 风格的长描述数据：
        初稿 → 句级对比打分（goldens）→ 对象/位置追问 → 回答（可选二次过滤）→ final_caption
    """

    def __init__(
        self,
        vlm_model_path: str,
        *,
        llm_model_path: Optional[str] = None,   # 可选：不传则与 VLM 共用同一服务
        hf_cache_dir: Optional[str] = None,
        download_dir: str = "./ckpt/models",
        device: str = "cuda",

        # Storage
        first_entry_file: str = "/path/to/your_images.jsonl",
        cache_path: str = "./cache_local",
        file_name_prefix: str = "scalecap",
        cache_type: str = "jsonl",

        # Keys
        image_key: str = "image",
        output_key: str = "scalecap_record",

        # 算子配置
        tau_sentence: float = 0.15,
        max_questions: int = 20,
        max_init_tokens: int = 1024,
        max_answer_tokens: int = 256,
        second_filter: bool = False,

        # vLLM 生成配置（按需暴露更多）
        vllm_tensor_parallel_size: int = 1,
        vllm_temperature: float = 0.0,
        vllm_top_p: float = 0.9,
        vllm_max_tokens: int = 512,
    ):
        # ---------- 1. Storage ----------
        self.storage = FileStorage(
            first_entry_file_name=first_entry_file,
            cache_path=cache_path,
            file_name_prefix=file_name_prefix,
            cache_type=cache_type,
        )

        # ---------- 2. Serving ----------
        # VLM（必须）
        self.vlm_serving = LocalModelVLMServing_vllm(
            hf_model_name_or_path=vlm_model_path,
            hf_cache_dir=hf_cache_dir,
            hf_local_dir=download_dir,
            vllm_tensor_parallel_size=vllm_tensor_parallel_size,
            vllm_temperature=vllm_temperature,
            vllm_top_p=vllm_top_p,
            vllm_max_tokens=vllm_max_tokens,
        )
        # LLM（可选；不传则用 VLM 服务做整合）
        self.llm_serving = None
        if llm_model_path:
            self.llm_serving = LocalModelVLMServing_vllm(
                hf_model_name_or_path=llm_model_path,
                hf_cache_dir=hf_cache_dir,
                hf_local_dir=download_dir,
                vllm_tensor_parallel_size=vllm_tensor_parallel_size,
                vllm_temperature=0.0,     # 整合建议用确定性
                vllm_top_p=1.0,
                vllm_max_tokens=2048,
            )

        # ---------- 3. Operator ----------
        cfg = ImageScaleCaptionGenerateConfig(
            tau_sentence=tau_sentence,
            max_questions=max_questions,
            max_init_tokens=max_init_tokens,
            max_answer_tokens=max_answer_tokens,
            second_filter=second_filter,
            input_jsonl_path=None,     # 用 storage 驱动，无需直传
            output_jsonl_path=None
        )

        self.operator = ImageScaleCaptionGenerate(
            vlm_serving=self.vlm_serving,
            config=cfg,
        )

        # 记录字段
        self.image_key = image_key
        self.output_key = output_key

    # ------------------------------------------------------------------ #
    def forward(self):
        """
        一键跑完当前 pipeline 一个 step：
            图片 → （init_caption → goldens → QAs → final_caption）→ 写回 storage
        """
        self.operator.run(
            storage=self.storage.step(),   
            input_image_key=self.image_key,
            output_key=self.output_key,
        )
        print("[ScaleCapPipeline] Done. Cached to:", self.storage.cache_path)


# ---------------------------- CLI 入口 -------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ScaleCap Dense Captioning Pipeline (DataFlow)")

    # 模型
    parser.add_argument("--vlm_model_path", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--llm_model_path", default="")  
    parser.add_argument("--hf_cache_dir", default="~/.cache/huggingface")
    parser.add_argument("--download_dir", default="./ckpt/models")
    parser.add_argument("--device", choices=["cuda", "cpu", "mps"], default="cuda")

    # 输入/缓存
    parser.add_argument("--images_file", default="./dataflow/example/image_to_text_pipeline/scale_cap_captions.jsonl")
    parser.add_argument("--cache_path", default="./cache_local")
    parser.add_argument("--file_name_prefix", default="scalecap")
    parser.add_argument("--cache_type", default="jsonl")

    # 字段名
    parser.add_argument("--image_key", default="image")
    parser.add_argument("--output_key", default="scalecap_record")

    # 算子配置
    parser.add_argument("--tau_sentence", type=float, default=0.15)
    parser.add_argument("--max_questions", type=int, default=20)
    parser.add_argument("--max_init_tokens", type=int, default=1024)
    parser.add_argument("--max_answer_tokens", type=int, default=256)
    parser.add_argument("--second_filter", action="store_true")

    # vLLM 生成配置（按需扩展）
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=512)

    args = parser.parse_args()

    pipe = ScaleCapPipeline(
        vlm_model_path=args.vlm_model_path,
        llm_model_path=(args.llm_model_path or None),
        hf_cache_dir=args.hf_cache_dir,
        download_dir=args.download_dir,
        device=args.device,

        first_entry_file=args.images_file,
        cache_path=args.cache_path,
        file_name_prefix=args.file_name_prefix,
        cache_type=args.cache_type,

        image_key=args.image_key,
        output_key=args.output_key,

        tau_sentence=args.tau_sentence,
        max_questions=args.max_questions,
        max_init_tokens=args.max_init_tokens,
        max_answer_tokens=args.max_answer_tokens,
        second_filter=args.second_filter,

        vllm_tensor_parallel_size=args.tp,
        vllm_temperature=args.temperature,
        vllm_top_p=args.top_p,
        vllm_max_tokens=args.max_tokens,
    )
    pipe.forward()
