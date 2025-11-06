import os
import argparse
from dataflow.operators.image_generation import PromptedT2ITextGenerator
from dataflow.operators.core_vision import PromptedImageGenerator
from dataflow.operators.core_vision import PromptedImageEditGenerator
from dataflow.serving.api_llm_serving_request import APILLMServing_request
from dataflow.serving.api_vlm_serving_openai import APIVLMServing_openai
from dataflow.serving.local_image_gen_serving import LocalImageGenServing
from dataflow.prompts.image_gen_prompt_generator import MultiImagesToImagePromptGenerator
from dataflow.utils.storage import FileStorage
from dataflow.io import ImageIO


class MultiImages2ImagePipeline():
    def __init__(
        self, 
        serving_type="api", 
        api_key="", 
        api_url="https://api.openai.com/v1/", 
        api_vlm_url="https://api.openai.com/v1/", 
        ip_condition_num=1, 
        repeat_times=1
    ):
        self.storage = FileStorage(
            first_entry_file_name="./dataflow/example/image_gen/multi_image_input_gen/prompts.jsonl",
            cache_path="./cache_local/multi_images_to_image_gen",
            file_name_prefix="dataflow_cache_step",
            cache_type="jsonl"
        )

        os.environ["DF_API_KEY"] = api_key
        self.serving = APILLMServing_request(
            api_url=api_url,
            model_name="gpt-4o",               # try gpt-4o, gpt-4o-mini, gpt-4-turbo
            max_workers=5,
        )
        
        self.t2i_serving = LocalImageGenServing(
            image_io=ImageIO(save_path=os.path.join(self.storage.cache_path, "condition_images")),
            batch_size=8,
            hf_model_name_or_path="/ytech_m2v5_hdd/CheckPoints/FLUX.1-dev",   # "black-forest-labs/FLUX.1-dev"
            hf_cache_dir="./cache_local",
            hf_local_dir="./ckpt/models/"
        )

        self.vlm_serving = APIVLMServing_openai(
            api_url=api_vlm_url,
            model_name="gemini-2.5-flash-image-preview",               # try nano-banana
            image_io=ImageIO(save_path=os.path.join(self.storage.cache_path, "target_images")),
            # send_request_stream=True,    # if use ip http://123.129.219.111:3000/ add this line
        )

        self.t2i_text_prompt_generator = MultiImagesToImagePromptGenerator()

        self.text_to_image_sample_generator = PromptedT2ITextGenerator(
            llm_serving=self.serving,
        )

        self.text_to_image_generator = PromptedImageGenerator(
            t2i_serving=self.t2i_serving,
        )

        self.image_editing_generator = PromptedImageEditGenerator(
            image_edit_serving=self.vlm_serving,
        )

    def forward(self):
        self.text_to_image_sample_generator.run(
            storage=self.storage.step(),
            prompt_generator=self.t2i_text_prompt_generator,
            input_style_key = "input_style",
            input_prompt_key = "input_text",
            output_prompt_key = "instruction",
            output_prompt_key_2 = "output_img_discript",
        )

        self.text_to_image_generator.run(
            storage=self.storage.step(),
            input_conversation_key="input_text",
            output_image_key="input_image",
        )

        self.image_editing_generator.run(
            storage=self.storage.step(),
            input_image_key="input_image",
            input_conversation_key="output_img_discript",
            output_image_key="output_image",
        )


if __name__ == "__main__":
    # This is the entry point for the pipeline
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--serving_type',
        choices=['api'],
        default='api',
    )
    parser.add_argument(
        '--api_key', type=str, default='',
    )
    parser.add_argument(
        '--api_url', type=str, default='https://api.openai.com/v1/',
    )
    parser.add_argument(
        '--api_vlm_url', type=str, default='https://api.openai.com/v1/',
    )
    parser.add_argument(
        '--ip_condition_num', type=int, default=1,
        help="Number of input condition elements to consider when generating prompts."
    )
    parser.add_argument(
        '--repeat_times', type=int, default=1,
        help="Number of times to repeat the prompt generation process."
    )
    args = parser.parse_args()

    pipeline = MultiImages2ImagePipeline(
        serving_type=args.serving_type,
        api_key=args.api_key,
        api_url=args.api_url,
        api_vlm_url=args.api_vlm_url,
        ip_condition_num=args.ip_condition_num,
        repeat_times=args.repeat_times
    )
    pipeline.forward()
