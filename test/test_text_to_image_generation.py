from dataflow.operators.generate import Text2ImageGenerator
from dataflow.serving.local_image_gen_serving import LocalImageGenServing
from dataflow.utils.storage import FileStorage


class ImageGenerationPipeline():
    def __init__(self):
        self.storage = FileStorage(
            first_entry_file_name="./dataflow/example/text2image/prompts.jsonl",
            cache_path="./cache_local",
            file_name_prefix="dataflow_cache_step",
            cache_type="jsonl"
        )

        self.serving = LocalImageGenServing(
            hf_model_name_or_path="/ytech_m2v5_hdd/CheckPoints/FLUX.1-dev",   # "black-forest-labs/FLUX.1-dev"
            hf_cache_dir="./cache_local",
            hf_local_dir="./ckpt/models/",
            device="cuda"
        )

        self.text_to_image_generator = Text2ImageGenerator(
            pipe = self.serving
        )
    
    def forward(self):
        self.text_to_image_generator.run(
            storage = self.storage.step(),
            input_image_key="images",
            input_video_key="videos",
            input_audio_key="audios",
            input_conversation_key="conversations",
            output_key="gen_images",
        )

if __name__ == "__main__":
    # This is the entry point for the pipeline
    model = ImageGenerationPipeline()
    model.forward()
