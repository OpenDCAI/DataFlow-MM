import os
from dataflow.operators.core_vision import PromptedImageGenerator
from dataflow.serving.local_image_gen_serving import LocalImageGenServing
from dataflow.utils.storage import FileStorage
from dataflow.io import ImageIO


class ImageGenerationPipeline():
    def __init__(self):
        # 使用绝对路径确保能找到文件
        from pathlib import Path
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent.parent.parent
        prompts_file = project_root / "dataflow" / "example" / "image_gen" / "text2image" / "prompts.jsonl"
        prompts_file = str(prompts_file)
        
        self.storage = FileStorage(
            first_entry_file_name=prompts_file,
            cache_path="./cache_local/text2image_local",
            file_name_prefix="dataflow_cache_step",
            cache_type="jsonl"
        )

        image_save_path = str(project_root / "cache_local" / "text2image_local")
        
        self.serving = LocalImageGenServing(
            image_io=ImageIO(save_path=image_save_path),
            batch_size=4,
            hf_model_name_or_path="/mnt/DataFlow/hzy/lqh/models/FLUX.1-dev",   # "black-forest-labs/FLUX.1-dev"
            hf_cache_dir="./cache_local",
            hf_local_dir="./ckpt/models/",
            diffuser_num_inference_steps=20,  # 减少推理步数从50到20，加快速度
            diffuser_image_height=512,  # 降低分辨率可以加快速度
            diffuser_image_width=512,
        )

        self.text_to_image_generator = PromptedImageGenerator(
            t2i_serving=self.serving,
            save_interval=10
        )
    
    def forward(self):
        self.text_to_image_generator.run(
            storage=self.storage.step(),
            input_conversation_key="conversations",
            output_image_key="images",
        )

if __name__ == "__main__":
    # This is the entry point for the pipeline
    model = ImageGenerationPipeline()
    model.forward()
