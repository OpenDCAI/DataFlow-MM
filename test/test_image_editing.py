import os
import argparse
from dataflow.operators.core_vision import PromptedImageEditGenerator
from dataflow.serving.local_image_gen_serving import LocalImageGenServing
from dataflow.serving.api_vlm_serving_openai import APIVLMServing_openai
from dataflow.utils.storage import FileStorage
from dataflow.io import ImageIO


class ImageGenerationPipeline():
    def __init__(self, serving_type="local", api_key="", api_url="http://123.129.219.111:3000/v1/"):
        self.storage = FileStorage(
            first_entry_file_name="./cache_local/text2image_condition/dataflow_cache_step_step1.jsonl",
            cache_path="./cache_local/multi2single_gen",
            file_name_prefix="dataflow_cache_step",
            cache_type="jsonl"
        )

        if serving_type == "local":
            self.serving = LocalImageGenServing(
                image_io=ImageIO(save_path=os.path.join(self.storage.cache_path, "images")),
                hf_model_name_or_path="black-forest-labs/FLUX.1-Kontext-dev",   # "black-forest-labs/FLUX.1-Kontext-dev"
                hf_cache_dir="./cache_local",
                hf_local_dir="./ckpt/models/",
                Image_gen_task="imageedit",
                batch_size=4,
                diffuser_model_name="FLUX-Kontext",
                diffuser_num_inference_steps=28,
                diffuser_guidance_scale=3.5,
            )
        elif serving_type == "api":
            self.serving = APIVLMServing_openai(
                api_url=api_url,
                model_name="gemini-2.5-flash-image-preview",               # try nano-banana
                image_io=ImageIO(save_path=os.path.join(self.storage.cache_path, "images")),
                send_request_stream=True,    # if utilize http://35.220.164.252:3888/v1/, delete this line
            )

        self.text_to_image_generator = PromptedImageEditGenerator(
            image_edit_serving=self.serving,
            save_interval=10
        )
    
    def forward(self):
        self.text_to_image_generator.run(
            storage=self.storage.step(),
            input_image_key="input_image",
            input_conversation_key="instruction",
            output_image_key="output_image",
        )

if __name__ == "__main__":
    # This is the entry point for the pipeline
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--serving_type',
        choices=['local', 'api'],
        default='api',
    )
    parser.add_argument(
        '--api_key', type=str, default='',
    )
    parser.add_argument(
        '--api_url', type=str, default='http://123.129.219.111:3000/v1/',
    )
    args = parser.parse_args()
    os.environ['DF_API_KEY'] = args.api_key
    model = ImageGenerationPipeline(serving_type=args.serving_type, api_key=args.api_key, api_url=args.api_url)
    model.forward()
