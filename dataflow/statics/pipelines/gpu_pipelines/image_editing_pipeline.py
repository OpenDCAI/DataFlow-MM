import os
from pathlib import Path
from dataflow.operators.core_vision import PromptedImageEditGenerator
from dataflow.serving.local_image_gen_serving import LocalImageGenServing
from dataflow.utils.storage import FileStorage
from dataflow.io import ImageIO


class ImageGenerationPipeline():
    def __init__(self):
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent.parent.parent
        
        data_file = project_root / "dataflow" / "example" / "image_gen" / "image_edit" / "prompts_local.jsonl"
        
        self.storage = FileStorage(
            first_entry_file_name=str(data_file),
            cache_path="./cache_local/image_edit_local",
            file_name_prefix="dataflow_cache_step",
            cache_type="jsonl"
        )

        image_save_path = str(project_root / "cache_local" / "image_edit_local")
        
        self.serving = LocalImageGenServing(
            image_io=ImageIO(save_path=image_save_path),
            hf_model_name_or_path="/mnt/DataFlow/hzy/lqh/models/FLUX.1-Kontext-dev",
            hf_cache_dir="./cache_local",
            hf_local_dir="./ckpt/models/",
            Image_gen_task="imageedit",
            batch_size=4,
            diffuser_model_name="FLUX-Kontext",
            diffuser_num_inference_steps=28,
            diffuser_guidance_scale=3.5,
        )

        self.text_to_image_generator = PromptedImageEditGenerator(
            image_edit_serving=self.serving,
            save_interval=10
        )
    
    def forward(self):
        self.text_to_image_generator.run(
            storage=self.storage.step(),
            input_image_key="images",
            input_conversation_key="conversations",
            output_image_key="output_image",
        )

if __name__ == "__main__":
    # This is the entry point for the pipeline
    model = ImageGenerationPipeline()
    model.forward()
