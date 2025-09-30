import os
from dataflow.operators.core_vision import PromptedImageGenerator
from dataflow.serving.local_image_gen_serving import LocalImageGenServing
from dataflow.utils.storage import FileStorage
from dataflow.io import ImageIO


class ImageGenerationPipeline():
    def __init__(self):
        self.storage = FileStorage(
            first_entry_file_name="./dataflow/example/image_gen/text2image/prompts.jsonl",
            cache_path="./cache_local",
            file_name_prefix="dataflow_cache_step",
            cache_type="jsonl"
        )

        self.serving = LocalImageGenServing(
            image_io=ImageIO(save_path=os.path.join(self.storage.cache_path, "images")),
            batch_size=4,
            hf_model_name_or_path="/ytech_m2v5_hdd/CheckPoints/FLUX.1-dev",   # "black-forest-labs/FLUX.1-dev"
            hf_cache_dir="./cache_local",
            hf_local_dir="./ckpt/models/"
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
