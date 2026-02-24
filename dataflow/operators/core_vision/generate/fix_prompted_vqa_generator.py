import os

# 设置 API Key 环境变量
os.environ["DF_API_KEY"] = "sk-iaY19LU7WMT5QlK8LujFIG7RjI2omHLWYiCs4Do6imieLKOg"

import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger

from dataflow.utils.storage import FileStorage, DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC
from dataflow.serving.local_model_vlm_serving import LocalModelVLMServing_vllm
from dataflow.serving.api_vlm_serving_openai import APIVLMServing_openai


@OPERATOR_REGISTRY.register()
class FixPromptedVQAGenerator(OperatorABC):
    '''
    FixPromptedVQAGenerator generate answers for questions based on provided context. 
    The context can be image or video.
    '''

    def __init__(self, 
                 serving: LLMServingABC, 
                 system_prompt: str = "You are a helpful assistant.",
                 user_prompt: str = "Please caption the media in detail."):
        self.logger = get_logger()
        self.serving = serving
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
            
    @staticmethod
    def get_desc(lang: str = "zh"):
        return "基于给定的 system prompt 和 user prompt，并读取 image/video 生成答案" \
            if lang == "zh" else \
            "Generate answers for questions based on provided context. The context can be image/video."
    
    def run(self, 
            storage: DataFlowStorage,
            input_image_key: str = "image", 
            input_video_key: str = "video",
            output_answer_key: str = "answer",
            ):

        if output_answer_key is None:
            raise ValueError("output_answer_key must be provided.")

        self.logger.info("Running FixPromptedVQA...")

        dataframe = storage.read('dataframe')
        self.logger.info(f"Loading, number of rows: {len(dataframe)}")

        image_column = dataframe.get(input_image_key, pd.Series([])).tolist()
        video_column = dataframe.get(input_video_key, pd.Series([])).tolist()

        # 标准化为 list[list[path]]
        image_column = [path if isinstance(path, list) else [path] for path in image_column]
        video_column = [path if isinstance(path, list) else [path] for path in video_column]

        if len(image_column) == 0:
            image_column = None
        if len(video_column) == 0:
            video_column = None

        if image_column is None and video_column is None:
            raise ValueError("At least one of input_image_key or input_video_key must be provided.")

        if image_column is not None and video_column is not None:
            raise ValueError("Only one of input_image_key or input_video_key must be provided.")

        conversations_list = []
        image_inputs_list = image_column
        video_inputs_list = video_column

        for idx in range(len(dataframe)):
            conversation = []

            # 构造 user message
            user_message = {
                "from": "human",
                "value": self.user_prompt
            }

            # 本地模式需要注入 <image> / <video> token
            tokens = []

            if image_inputs_list and idx < len(image_inputs_list):
                valid_images = [img for img in image_inputs_list[idx] if img is not None]
                if valid_images:
                    tokens.extend(["<image>"] * len(valid_images))

            if video_inputs_list and idx < len(video_inputs_list):
                valid_videos = [vid for vid in video_inputs_list[idx] if vid is not None]
                if valid_videos:
                    tokens.extend(["<video>"] * len(valid_videos))

            if tokens:
                user_message["value"] = "".join(tokens) + user_message["value"]

            conversation.append(user_message)
            conversations_list.append(conversation)

        # 直接使用 generate_from_input_messages
        outputs = self.serving.generate_from_input_messages(
            conversations=conversations_list,
            image_list=image_inputs_list,
            video_list=video_inputs_list,
            system_prompt=self.system_prompt
        )

        dataframe[output_answer_key] = outputs
        output_file = storage.write(dataframe)
        self.logger.info(f"Results saved to {output_file}")

        return output_answer_key


if __name__ == "__main__":

    model = APIVLMServing_openai(
        api_url="http://172.96.141.132:3001/v1", # Any API platform compatible with OpenAI format
        key_name_of_api_key="DF_API_KEY", # Set the API key for the corresponding platform in the environment variable or line 4
        model_name="gpt-5-nano-2025-08-07",
        image_io=None,
        send_request_stream=False,
        max_workers=10,
        timeout=1800
    )

    # model = LocalModelVLMServing_vllm(
    #     hf_model_name_or_path="Qwen/Qwen2.5-VL-3B-Instruct",
    #     vllm_tensor_parallel_size=1,
    #     vllm_temperature=0.7,
    #     vllm_top_p=0.9,
    #     vllm_max_tokens=512,
    # )

    generator = FixPromptedVQAGenerator(
        serving=model,
        system_prompt="You are a helpful assistant.",
        user_prompt="Please caption the media in detail."
    )

    storage = FileStorage(
        first_entry_file_name="./dataflow/example/test_data/image_data.json", 
        cache_path="./cache_prompted_vqa",
        file_name_prefix="fix_prompted_vqa",
        cache_type="json",
    )

    storage.step()

    generator.run(
        storage=storage,
        input_image_key="image",
        input_video_key="video",
        output_answer_key="answer",
    )