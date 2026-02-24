import os

# 设置 API Key 环境变量
os.environ["DF_API_KEY"] = "sk-xxx"

from dataflow.utils.storage import FileStorage
from dataflow.core import LLMServingABC
from dataflow.serving.api_vlm_serving_openai import APIVLMServing_openai
from dataflow.operators.core_vision import PromptedVQAGenerator, WikiQARefiner


class ContextVQAPipeline:
    """
    一行命令即可完成图片批量 ContextVQA Caption 生成。
    """

    def __init__(self, llm_serving: LLMServingABC = None):
        # ---------- 1. Storage ----------
        self.storage = FileStorage(
            first_entry_file_name="./example_data/image_contextvqa/sample_data.json",
            cache_path="./cache_local",
            file_name_prefix="context_vqa",
            cache_type="json",
        )

        # ---------- 2. Serving ----------
        self.vlm_serving = APIVLMServing_openai(
            api_url="http://172.96.141.132:3001/v1", # Any API platform compatible with OpenAI format
            key_name_of_api_key="DF_API_KEY", # Set the API key for the corresponding platform in the environment variable or line 4
            model_name="gpt-5-nano-2025-08-07",
            image_io=None,
            send_request_stream=False,
            max_workers=10,
            timeout=1800
        )

        # ---------- 3. Operator ----------
        self.vqa_generator = PromptedVQAGenerator(
            serving=self.vlm_serving,
            system_prompt= "You are a helpful assistant."
        )

        self.refiner = WikiQARefiner()

    # ------------------------------------------------------------------ #
    def forward(self):
        input_image_key = "image"
        output_answer_key = "vqa"
        output_wiki_key = "context_vqa"

        self.vqa_generator.run(
            storage=self.storage.step(),
            input_conversation_key="conversation",
            input_image_key=input_image_key,
            output_answer_key=output_answer_key,
        )

        self.refiner.run(
            storage=self.storage.step(),
            input_key=output_answer_key,
            output_key=output_wiki_key
        )

# ---------------------------- CLI 入口 -------------------------------- #
if __name__ == "__main__":
    pipe = ContextVQAPipeline()
    pipe.forward()