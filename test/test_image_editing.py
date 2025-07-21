import os
from dataflow.operators.generate import ImageEditor
from dataflow.serving.local_image_gen_serving import LocalImageGenServing
from dataflow.utils.storage import FileStorage
from dataflow.io import ImageIO


class ImageGenerationPipeline():
    def __init__(self):
        self.storage = FileStorage(
            first_entry_file_name="./dataflow/example/image_edit/prompts.jsonl",
            cache_path="./cache_local/image_edit",
            file_name_prefix="dataflow_cache_step",
            cache_type="jsonl"
        )

        self.serving = LocalImageGenServing(
            image_io=ImageIO(save_path=os.path.join(self.storage.cache_path, "images")),
            hf_model_name_or_path="/ytech_m2v5_hdd/workspace/kling_mm/Models/FLUX.1-Kontext-dev",   # "black-forest-labs/FLUX.1-Kontext-dev"
            hf_cache_dir="./cache_local",
            hf_local_dir="./ckpt/models/",
            Image_gen_task="imageedit",
            diffuser_model_name="FLUX-Kontext",
            diffuser_num_inference_steps=28,
            diffuser_guidance_scale=3.5,
        )

        self.text_to_image_generator = ImageEditor(
            image_edit_serving=self.serving,
            save_interval=10
        )
    
    def forward(self):
        self.text_to_image_generator.run(
            storage=self.storage.step(),
            input_image_key="images",
            input_conversation_key="conversations",
            output_image_key="edited_images",
        )

if __name__ == "__main__":
    # This is the entry point for the pipeline
    model = ImageGenerationPipeline()
    model.forward()
