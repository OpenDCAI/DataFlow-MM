from dataflow.serving.local_model_vlm_serving import LocalModelVLMServing_vllm
from dataflow.operators.core_vision.generate.image_bbox_generator import (
    ImageBboxGenerator, 
    ExistingBBoxDataGenConfig
)
from dataflow.operators.core_vision.generate.prompted_vqa_generator import (
    PromptedVQAGenerator
)
from dataflow.utils.storage import FileStorage


class ImageRegionCaptionPipeline:
    def __init__(
        self,
        model_path: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        hf_cache_dir: str = "~/.cache/huggingface",
        download_dir: str = "./ckpt/models",
        first_entry_file: str = "./data/image_region_caption/image_region_caption_demo.jsonl",
        cache_path: str = "./cache/image_region_caption",
        file_name_prefix: str = "region_caption",
        cache_type: str = "jsonl",
        input_image_key: str = "image",
        input_bbox_key: str = "bbox",
<<<<<<< HEAD
=======
        image_with_bbox_path: str = 'image_with_bbox',
>>>>>>> 59b89e3c5635df9109a0ece2565ac7dc4562ea1d
        max_boxes: int = 10,
        output_image_with_bbox_path: str = "./cache/image_region_caption/image_with_bbox_result.jsonl",
    ):
        self.bbox_storage = FileStorage(
            first_entry_file_name=first_entry_file,
            cache_path=cache_path,
            file_name_prefix=file_name_prefix,
            cache_type=cache_type
        )

        self.cfg = ExistingBBoxDataGenConfig(
            max_boxes=max_boxes,
            input_jsonl_path=first_entry_file,
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
<<<<<<< HEAD
=======
        self.image_with_bbox_path=image_with_bbox_path
>>>>>>> 59b89e3c5635df9109a0ece2565ac7dc4562ea1d
        self.bbox_record=None

    def forward(self):
        self.bbox_generator.run(
            storage=self.bbox_storage.step(),
            input_image_key=self.input_image_key,
            input_bbox_key=self.input_bbox_key,
        )

        self.caption_generator.run(
            storage=self.caption_storage.step(),
            input_image_key='image_with_bbox',
            input_prompt_key='prompt'
        )


if __name__ == "__main__":
<<<<<<< HEAD
    pipe = ImageRegionCaptionPipeline()
=======
    parser = argparse.ArgumentParser(description="Image region caption with DataFlow")
    parser.add_argument("--model_path", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--hf_cache_dir", default="~/.cache/huggingface")
    parser.add_argument("--download_dir", default="./ckpt/models")
    parser.add_argument("--first_entry_file", default="./data/image_region_caption/image_region_caption_demo.jsonl")
    parser.add_argument("--cache_path", default="./cache/image_region_caption")
    parser.add_argument("--file_name_prefix", default="region_caption")
    parser.add_argument("--cache_type", default="jsonl")
    parser.add_argument("--input_image_key", default="image")
    parser.add_argument("--input_bbox_key", default="bbox")
    parser.add_argument("--max_boxes", type=int, default=10)
    parser.add_argument("--output_image_with_bbox_path", default="./cache/image_region_caption/image_with_bbox_result.jsonl")

    args = parser.parse_args()

    pipe = ImageRegionCaptionPipeline(
        model_path=args.model_path,
        hf_cache_dir=args.hf_cache_dir,
        download_dir=args.download_dir,
        first_entry_file=args.first_entry_file,
        cache_path=args.cache_path,
        file_name_prefix=args.file_name_prefix,
        cache_type=args.cache_type,
        input_image_key=args.input_image_key,
        input_bbox_key=args.input_bbox_key,
        max_boxes=args.max_boxes,
        output_image_with_bbox_path=args.output_image_with_bbox_path
    )
>>>>>>> 59b89e3c5635df9109a0ece2565ac7dc4562ea1d
    pipe.forward()