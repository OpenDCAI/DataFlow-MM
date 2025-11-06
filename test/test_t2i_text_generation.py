import os
import argparse
from dataflow.operators.image_generation import PromptedT2ITextGenerator
from dataflow.serving.api_llm_serving_request import APILLMServing_request
from dataflow.utils.storage import FileStorage


class TextPromptExtendPipeline():
    def __init__(
        self, 
        serving_type="api", 
        api_key="", 
        api_url="https://api.openai.com/v1/", 
        ip_condition_num=1, 
        repeat_times=1
    ):
        if os.path.exists("./cache_local/multi2single_image_gen/dataflow_cache_step_step1.jsonl") is False:
            entry_file_path = "./cache_local/text_prompt_extend/dataflow_cache_step_step1.jsonl"
        else:
            entry_file_path = "./cache_local/multi2single_image_gen/dataflow_cache_step_step1.jsonl"
        self.storage = FileStorage(
            first_entry_file_name=entry_file_path,
            cache_path="./cache_local/multi2single_image_gen",
            file_name_prefix="dataflow_cache_step",
            cache_type="jsonl"
        )

        if serving_type == "api":
            os.environ["DF_API_KEY"] = api_key
            self.serving = APILLMServing_request(
                api_url=api_url,
                model_name="gpt-4o",               # try gpt-4o, gpt-4o-mini, gpt-4-turbo
                max_workers=5,
            )
        else:
            raise ValueError("Currently only 'api' serving_type is supported.")

        self.text_to_image_generator = PromptedT2ITextGenerator(
            llm_serving=self.serving,
            ip_condition_num=ip_condition_num,
            repeat_times=repeat_times
        )
    
    def forward(self):
        self.text_to_image_generator.run(
            storage=self.storage.step(),
            input_style_key = "input_style",
            input_prompt_key = "input_text",
            output_prompt_key = "instruction",
            output_prompt_key_2 = "output_img_discript"
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
        '--ip_condition_num', type=int, default=1,
        help="Number of input condition elements to consider when generating prompts."
    )
    parser.add_argument(
        '--repeat_times', type=int, default=1,
        help="Number of times to repeat the prompt generation process."
    )
    args = parser.parse_args()

    pipeline = TextPromptExtendPipeline(
        serving_type=args.serving_type,
        api_key=args.api_key,
        api_url=args.api_url,
        ip_condition_num=args.ip_condition_num,
        repeat_times=args.repeat_times
    )
    pipeline.forward()
