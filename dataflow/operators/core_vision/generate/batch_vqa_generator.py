import os

# 设置 API Key 环境变量
os.environ["DF_API_KEY"] = "sk-iaY19LU7WMT5QlK8LujFIG7RjI2omHLWYiCs4Do6imieLKOg"

from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage, FileStorage
from dataflow.core import OperatorABC, LLMServingABC
from dataflow import get_logger
from dataflow.serving.api_vlm_serving_openai import APIVLMServing_openai


@OPERATOR_REGISTRY.register()
class BatchVQAGenerator(OperatorABC):
    """
    批量视觉问答生成器 (One Image, Many Questions)
    """

    def __init__(self, serving: LLMServingABC, system_prompt: str = "You are a helpful assistant."):
        self.serving = serving
        self.system_prompt = system_prompt
        self.logger = get_logger()

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "批量视觉问答生成算子。\n"
                "单张图片对应多个问题，输出答案列表。\n"
                "自动广播图片到多个问题。"
            )
        else:
            return (
                "Batch VQA Generator.\n"
                "One image with multiple questions.\n"
                "Automatically broadcasts image to prompts."
            )

    def run(
        self,
        storage: DataFlowStorage,
        input_prompts_key: str,
        input_image_key: str,
        output_key: str,
    ):
        self.logger.info(f"Running BatchVQAGenerator on {input_prompts_key}...")
        df = storage.read("dataframe")

        all_answers_nested = []

        for _, row in df.iterrows():
            questions = row.get(input_prompts_key, [])
            
            image_path = row.get(input_image_key)
            # 统一处理成 list[str]
            if isinstance(image_path, str):
                image_paths = [image_path]
            elif isinstance(image_path, list):
                image_paths = image_path
            else:
                image_paths = None


            if not questions or not isinstance(questions, list) or not image_path:
                all_answers_nested.append([])
                continue

            conversations = []
            image_list = []

            image_list = []

            for q in questions:

                user_value = "<image>" + q

                conversation = [
                    {
                        "from": "human",
                        "value": user_value
                    }
                ]

                conversations.append(conversation)

                # 关键修复点
                image_list.append(image_paths)

            if not conversations:
                all_answers_nested.append([])
                continue
            
            print(f"conversations: {conversations}")
            print(f"image_list: {image_list}")
            
            # 批量调用
            row_answers = self.serving.generate_from_input_messages(
                conversations=conversations,
                image_list=image_list,
                video_list=None,
                system_prompt=self.system_prompt
            )

            all_answers_nested.append(row_answers)

        df[output_key] = all_answers_nested
        storage.write(df)

        return [output_key]

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

    generator = BatchVQAGenerator(
        serving=model,
        system_prompt="You are a helpful assistant.",
    )

    storage = FileStorage(
        first_entry_file_name="./dataflow/example/test_data/batched_prompt_data.json", 
        cache_path="./cache_prompted_vqa",
        file_name_prefix="batch_prompted_vqa",
        cache_type="json",
    )

    storage.step()

    generator.run(
        storage=storage,
        input_image_key="image",
        input_prompts_key="question",
        output_key="answer",
    )