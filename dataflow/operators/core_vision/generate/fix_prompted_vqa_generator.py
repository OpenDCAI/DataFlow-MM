import pandas as pd
from typing import List

from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import FileStorage, DataFlowStorage
from dataflow.core import OperatorABC, LLMServingABC
from dataflow import get_logger

from dataflow.serving.local_model_vlm_serving import LocalModelVLMServing_vllm
from dataflow.serving.api_vlm_serving_openai import APIVLMServing_openai


# 提取判断是否为 API Serving 的辅助函数
def is_api_serving(serving):
    return isinstance(serving, APIVLMServing_openai)


@OPERATOR_REGISTRY.register()
class FixPromptedVQAGenerator(OperatorABC):
    """
    FixPromptedVQAGenerator generate answers for questions based on provided context. The context can be image/video.
    """
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
        if lang == "zh":
            return (
                "固定提示词视觉问答生成算子 (FixPromptedVQAGenerator)。\n"
                "基于给定的 system prompt 和 user prompt，读取 image/video 生成答案。\n\n"
                "特点：\n"
                "  - 支持图像或视频模态\n"
                "  - 统一支持 API 和本地 Local 模型部署模式\n"
                "  - 自动管理底层的 <image> 或 <video> 占位符\n"
            )
        else:
            return "Generate answers for questions based on provided context. The context can be image/video."

    def run(self, 
            storage: DataFlowStorage,
            input_image_key: str = "image", 
            input_video_key: str = "video",
            output_answer_key: str = "answer",
            ):
        if not output_answer_key:
            raise ValueError("'output_answer_key' must be provided.")

        self.logger.info("Running FixPromptedVQA...")
        
        # 加载 DataFrame
        dataframe: pd.DataFrame = storage.read('dataframe')
        self.logger.info(f"Loaded dataframe with {len(dataframe)} rows")

        # 提取并清洗多模态列数据
        image_column = dataframe.get(input_image_key, pd.Series([None] * len(dataframe))).tolist()
        video_column = dataframe.get(input_video_key, pd.Series([None] * len(dataframe))).tolist()

        # 统一转为 List 格式
        image_column = [path if isinstance(path, list) else [path] if pd.notna(path) else [] for path in image_column]
        video_column = [path if isinstance(path, list) else [path] if pd.notna(path) else [] for path in video_column]
        
        # 判断当前生效的模态
        has_images = any(len(p) > 0 for p in image_column)
        has_videos = any(len(p) > 0 for p in video_column)

        if has_images and has_videos:
            raise ValueError("Only one of input_image_key or input_video_key can be provided with valid data.")
        if not has_images and not has_videos:
            raise ValueError("At least one of input_image_key or input_video_key must contain valid media paths.")

        use_api_mode = is_api_serving(self.serving)
        if use_api_mode:
            self.logger.info("Using API serving mode")
        else:
            self.logger.info("Using local serving mode")

        # 构造对话与输入列表
        conversations_list = []
        image_inputs_list = None
        video_inputs_list = None

        if has_images:
            image_inputs_list = image_column
            for paths in image_column:
                valid_media_count = len([p for p in paths if p])
                
                if use_api_mode:
                    content = self.user_prompt
                else:
                    content = ("<image>" * valid_media_count) + self.user_prompt
                    
                conversations_list.append([{"role": "user", "content": content}])
                
        elif has_videos:
            video_inputs_list = video_column
            for paths in video_column:
                valid_media_count = len([p for p in paths if p])
                
                if use_api_mode:
                    content = self.user_prompt
                else:
                    content = ("<video>" * valid_media_count) + self.user_prompt
                    
                conversations_list.append([{"role": "user", "content": content}])

        # 统一调用基类的消息生成接口
        outputs = self.serving.generate_from_input_messages(
            conversations=conversations_list,
            image_list=image_inputs_list,
            video_list=video_inputs_list,
            system_prompt=self.system_prompt
        )

        # 保存结果
        dataframe[output_answer_key] = outputs
        output_file = storage.write(dataframe)
        self.logger.info(f"Results saved to {output_file}")

        return [output_answer_key]
    

# ==========================================
# 测试用例 (Main Block)
# ==========================================
if __name__ == "__main__":
    # 使用 API 模式进行测试
    model = APIVLMServing_openai(
        api_url="http://172.96.141.132:3001/v1",
        key_name_of_api_key="DF_API_KEY",
        model_name="gpt-5-nano-2025-08-07",
        image_io=None,
        send_request_stream=False,
        max_workers=10,
        timeout=1800
    )

    # 如需使用本地模式，解开下方注释：
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

    # 准备输入数据
    storage = FileStorage(
        first_entry_file_name="./dataflow/example/image_to_text_pipeline/fix_prompted_vqa.jsonl", 
        cache_path="./cache_prompted_vqa",
        file_name_prefix="fix_prompted_vqa",
        cache_type="jsonl",
    )
    storage.step()  # 加载数据

    generator.run(
        storage=storage,
        input_image_key="image",
        input_video_key="video",
        output_answer_key="answer",
    )