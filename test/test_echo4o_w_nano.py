# provide a image editing dataset

# text-to-image compose multiple objects

# task_type = "image editing", "multi-turn editing", "multi-image conditions generation", "text-to-image with multiple objects"


# step 1: utilize a local model to geenrate base images

# step 2: follow different task settings to realize image generation

import os
import argparse
from dataflow.operators.core_vision import PromptedImageEditGenerator
from dataflow.operators.core_vision import PromptedImageGenerator
from dataflow.serving.local_image_gen_serving import LocalImageGenServing
from dataflow.serving.api_vlm_serving_openai import APIVLMServing_openai
from dataflow.utils.storage import FileStorage
from dataflow.io import ImageIO


class uni_image_gen_pipeline():
    def __init__(self, api_url="http://123.129.219.111:3000/v1/"):
        self.storage = FileStorage(
            first_entry_file_name="./dataflow/example/image_gen/unified_image_gen/prompts.jsonl",
            cache_path="./cache_local",
            file_name_prefix="dataflow_cache_step",
            cache_type="jsonl"
        )

        self.t2i_local_serving = LocalImageGenServing(
            image_io=ImageIO(save_path=os.path.join(self.storage.cache_path, "condition_images")),
            batch_size=4,
            hf_model_name_or_path="/ytech_m2v5_hdd/CheckPoints/FLUX.1-dev",   # "black-forest-labs/FLUX.1-dev"
            hf_cache_dir="./cache_local",
            hf_local_dir="./ckpt/models/"
        )

        self.edit_api_serving = APIVLMServing_openai(
                api_url=api_url,
                model_name="gemini-2.5-flash-image-preview",               # try nano-banana
                image_io=ImageIO(save_path=os.path.join(self.storage.cache_path, "target_images")),
                send_request_stream=True,
            )

        self.text_to_image_generator = PromptedImageGenerator(
            t2i_serving=self.t2i_local_serving,
            save_interval=10
        )

        self.image_edit_generator = PromptedImageEditGenerator(
            image_edit_serving=self.edit_api_serving,
            save_interval=10
        )

    def forward(self):
        self.text_to_image_generator.run(
            storage=self.storage.step(),
            input_conversation_key="image_condition_input",
            output_image_key="condition_images",
        )
        self.image_edit_generator.run(
            storage=self.storage.step(),
            input_image_key="condition_images",
            input_conversation_key="image_edit_input",
            output_image_key="target_images",
        )

