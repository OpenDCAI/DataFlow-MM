import argparse
from dataflow.serving.local_model_vlm_serving import LocalModelVLMServing_vllm
from dataflow.operators.core_vision.generate.image_bbox_generator import (
    ImageBboxGenerator, 
    ExistingBBoxDataGenConfig
)
from dataflow.operators.core_vision.generate.prompted_vqa_generator import (
    PromptedVQAGenerator
)
from dataflow.utils.storage import FileStorage


class ImageRegionCaptioningPipeline:
    def __init__(
        self,
        model_path: str,
        *,
        hf_cache_dir: str | None = None,
        download_dir: str = "./ckpt/models",
        device: str = "cuda",
        first_entry_file: str = "./dataflow/example/image_to_text_pipeline/region_captions.jsonl",
        cache_path: str = "./dataflow/example/cache",
        file_name_prefix: str = "region_caption",
        cache_type: str = "jsonl",
        input_image_key: str = "image",
        input_bbox_key: str = "bbox",
        image_with_bbox_path: str = 'image_with_bbox',
        output_key: str = "mdvp_record",
        max_boxes: int = 10,
        input_jsonl_path: str = "./dataflow/example/image_to_text_pipeline/region_captions.jsonl",
        output_jsonl_path: str = "./dataflow/example/image_to_text_pipeline/region_captions_results_v1.jsonl",
        output_image_with_bbox_path: str = "./dataflow/example/image_to_text_pipeline/image_with_bbox_results_v1.jsonl",
        draw_visualization: bool = True
    ):
        self.bbox_storage = FileStorage(
            first_entry_file_name=first_entry_file,
            cache_path=cache_path,
            file_name_prefix=file_name_prefix,
            cache_type=cache_type
        )

        self.cfg = ExistingBBoxDataGenConfig(
            max_boxes=max_boxes,
            input_jsonl_path=input_jsonl_path,
            output_jsonl_path=output_image_with_bbox_path,
        )

        self.caption_storage = FileStorage(
            first_entry_file_name=output_image_with_bbox_path,
            cache_path=cache_path,
            file_name_prefix=file_name_prefix,
            cache_type=cache_type
        )
        self.serving = LocalModelVLMServing_vllm(
            hf_model_name_or_path=model_path,
            hf_cache_dir=hf_cache_dir,
            hf_local_dir=download_dir,
            vllm_tensor_parallel_size=1,
            vllm_temperature=0.7,
            vllm_top_p=0.9,
            vllm_max_tokens=512,
        )
        self.bbox_generator = ImageBboxGenerator(config=self.cfg)
        self.caption_generator = PromptedVQAGenerator(serving=self.serving,)
        self.input_image_key = input_image_key
        self.input_bbox_key = input_bbox_key
        self.output_key = output_key
        self.image_with_bbox_path=image_with_bbox_path
        self.bbox_record=None

    def forward(self):
        self.bbox_generator.run(
            storage=self.bbox_storage.step(),
            input_image_key=self.input_image_key,
            input_bbox_key=self.input_bbox_key,
            output_key=self.image_with_bbox_path,
        )
        # print(self.bbox_record)
        self.caption_generator.run(
            storage=self.caption_storage.step(),
            input_image_key='image_with_bbox'
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image region captioning with DataFlow")
 
    parser.add_argument("--model_path", default="/data0/happykeyan/Models/Qwen2.5-VL-3B-Instruct")
    # parser.add_argument("--model_path", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--hf_cache_dir", default="~/.cache/huggingface")
    parser.add_argument("--download_dir", default="./ckpt/models")
    parser.add_argument("--device", choices=["cuda", "cpu", "mps"], default="cuda")

    parser.add_argument("--first_entry_file", default="./dataflow/example/image_to_text_pipeline/region_captions.jsonl")
    parser.add_argument("--cache_path", default="./dataflow/example/cache")
    parser.add_argument("--file_name_prefix", default="region_caption")
    parser.add_argument("--cache_type", default="jsonl")
    parser.add_argument("--input_image_key", default="image")
    parser.add_argument("--input_bbox_key", default="bbox")
    parser.add_argument("--output_key", default="mdvp_record")

    parser.add_argument("--max_boxes", type=int, default=10)
    parser.add_argument("--input_jsonl_path", default="./dataflow/example/image_to_text_pipeline/region_captions.jsonl")
    parser.add_argument("--output_jsonl_path", default="./dataflow/example/image_to_text_pipeline/region_captions_results_v1.jsonl")
    parser.add_argument("--output_image_with_bbox_path", default="./dataflow/example/image_to_text_pipeline/image_with_bbox_results_v1.jsonl")
    parser.add_argument("--draw_visualization", type=bool, default=True)

    args = parser.parse_args()

    pipe = ImageRegionCaptioningPipeline(
        model_path=args.model_path,
        hf_cache_dir=args.hf_cache_dir,
        download_dir=args.download_dir,
        device=args.device,
        first_entry_file=args.first_entry_file,
        cache_path=args.cache_path,
        file_name_prefix=args.file_name_prefix,
        cache_type=args.cache_type,
        input_image_key=args.input_image_key,
        input_bbox_key=args.input_bbox_key,
        output_key=args.output_key,
        max_boxes=args.max_boxes,
        input_jsonl_path=args.input_jsonl_path,
        output_image_with_bbox_path=args.output_image_with_bbox_path,
        draw_visualization=args.draw_visualization
    )
    pipe.forward()