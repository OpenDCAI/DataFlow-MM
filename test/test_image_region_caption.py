from dataflow.serving.local_model_vlm_serving import LocalModelVLMServing_vllm
from dataflow.operators.core_vision.generate.image_region_caption_generator import (
    ImageRegionCaptionGenerate, 
    ExistingBBoxDataGenConfig
)
from dataflow.utils.storage import FileStorage

storage = FileStorage(
    first_entry_file_name="./dataflow/example/image_to_text_pipeline/region_captions.jsonl",
    cache_path="./dataflow/example/cache",
    file_name_prefix="region_caption",
    cache_type="jsonl"
)

model = LocalModelVLMServing_vllm(
    hf_model_name_or_path="/mnt/public/model/huggingface/Qwen2.5-VL-3B-Instruct",
    hf_cache_dir="~/.cache/huggingface",
    hf_local_dir="./ckpt",
    vllm_tensor_parallel_size=1,
    vllm_temperature=0.0,
    vllm_top_p=0.9,
    vllm_max_tokens=1024
)

cfg = ExistingBBoxDataGenConfig(
    max_boxes=10,
    input_jsonl_path="./dataflow/example/image_to_text_pipeline/region_captions.jsonl",
    output_jsonl_path="./dataflow/example/image_to_text_pipeline/region_captions_results.jsonl",
    draw_visualization=True
)

operator = ImageRegionCaptionGenerate(llm_serving=model, config=cfg)

operator.run(
    storage=storage.step(),
    image_key="image",
    bbox_key="bbox",
    output_key="mdvp_record"
)
