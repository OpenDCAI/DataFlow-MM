# test_caprl_mcq_build.py
import argparse
from dataflow.utils.storage import FileStorage
from dataflow.serving.local_model_vlm_serving import LocalModelVLMServing_vllm
from dataflow.operators.core_vision import CapRLMCQGenerate, CapRLMCQConfig

class CapRLPipeline:
    def __init__(
        self,
        vlm_model_path: str,
        *,
        first_entry_file: str,
        cache_path: str = "./cache_local",
        file_name_prefix: str = "caprl",
        cache_type: str = "jsonl",
        image_key: str = "image",
        output_key: str = "cap_rl_qa",
        rotate_num: int = 4,
        pass_visual_min: float = 1.0,
        pass_textual_max: float = 0.0,
        add_none_above_for_visual: bool = True,
        tp: int = 1,
        temperature: float = 0.1,
        top_p: float = 0.9,
        max_tokens: int = 512,
    ):
        self.storage = FileStorage(
            first_entry_file_name=first_entry_file,
            cache_path=cache_path,
            file_name_prefix=file_name_prefix,
            cache_type=cache_type,
        )

        self.serving = LocalModelVLMServing_vllm(
            hf_model_name_or_path=vlm_model_path,
            vllm_tensor_parallel_size=tp,
            vllm_temperature=temperature,
            vllm_top_p=top_p,
            vllm_max_tokens=max_tokens,
        )

        cfg = CapRLMCQConfig(
            rotate_num=rotate_num,
            pass_visual_min=pass_visual_min,
            pass_textual_max=pass_textual_max,
            add_none_above_for_visual=add_none_above_for_visual,
            input_jsonl_path=None,
            output_jsonl_path=None,
        )
        self.op = CapRLMCQGenerate(self.serving, cfg)
        self.image_key = image_key
        self.output_key = output_key

    def forward(self):
        self.op.run(
            storage=self.storage.step(),
            input_image_key=self.image_key,
            output_key=self.output_key,
        )
        print("[CapRLPipeline] Done →", self.storage.cache_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vlm_model_path", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--images_file", default="./dataflow/example/image_to_text_pipeline/capsbench_captions.jsonl")
    parser.add_argument("--cache_path", default="./cache_local")
    parser.add_argument("--file_name_prefix", default="caprl")
    parser.add_argument("--cache_type", default="jsonl")
    parser.add_argument("--rotate_num", type=int, default=4)
    parser.add_argument("--pass_visual_min", type=float, default=1.0)
    parser.add_argument("--pass_textual_max", type=float, default=0.0)
    args = parser.parse_args()

    pipe = CapRLPipeline(
        vlm_model_path=args.vlm_model_path,
        first_entry_file=args.images_file,
        cache_path=args.cache_path,
        file_name_prefix=args.file_name_prefix,
        cache_type=args.cache_type,
        rotate_num=args.rotate_num,
        pass_visual_min=args.pass_visual_min,
        pass_textual_max=args.pass_textual_max,
    )
    pipe.forward()
