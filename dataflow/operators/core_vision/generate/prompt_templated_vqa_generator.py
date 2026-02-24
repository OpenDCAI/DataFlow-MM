import pandas as pd
from typing import List

from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import FileStorage, DataFlowStorage

from dataflow.core import OperatorABC, LLMServingABC
from dataflow.serving.local_model_vlm_serving import LocalModelVLMServing_vllm
from dataflow.serving.api_vlm_serving_openai import APIVLMServing_openai
from dataflow.prompts.prompt_template import NamedPlaceholderPromptTemplate


# 提取判断是否为 API Serving 的辅助函数
def is_api_serving(serving):
    return isinstance(serving, APIVLMServing_openai)


@OPERATOR_REGISTRY.register()
class PromptTemplatedVQAGenerator(OperatorABC):
    """
    PromptTemplatedVQAGenerator:
    1) 从 DataFrame 读取若干字段（由 input_keys 指定）
    2) 使用 prompt_template.build_prompt(...) 生成纯文本 prompt
    3) 将该 prompt 与 image/video 一起输入多模态模型，生成答案
    """

    def __init__(
        self,
        serving: LLMServingABC,
        prompt_template: NamedPlaceholderPromptTemplate,
        system_prompt: str = "You are a helpful assistant.",
    ):
        self.logger = get_logger()
        self.serving = serving
        self.system_prompt = system_prompt
        self.prompt_template = prompt_template

        if self.prompt_template is None:
            raise ValueError(
                "prompt_template cannot be None for PromptTemplatedVQAGenerator."
            )

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "基于模板的动态多模态问答算子 (PromptTemplatedVQAGenerator)。\n"
                "JSONL/DataFrame 中包含若干字段，通过 input_keys 将 DataFrame 列映射到模板字段，\n"
                "由 prompt_template 生成最终的文本 Prompt，再结合 image/video 进行多模态问答。\n\n"
                "特点：\n"
                "  - 支持动态组装复杂的 Prompt\n"
                "  - 统一支持 API 和本地 Local 模型部署模式\n"
                "  - 自动管理底层的 <image> 或 <video> 占位符\n"
            )
        else:
            return (
                "PromptTemplatedVQAGenerator: a multimodal VQA operator that first builds "
                "text prompts from a prompt template and multiple input fields, then "
                "performs VQA with image/video."
            )

    def run(
        self,
        storage: DataFlowStorage,
        input_image_key: str = "image",
        input_video_key: str = "video",
        output_answer_key: str = "answer",
        **input_keys,
    ):
        """
        参数：
        - storage: DataFlowStorage
        - input_image_key / input_video_key: 存放图片/视频路径的列名（只允许其一存在）
        - output_answer_key: 输出答案列名
        - **input_keys: 模板字段名 -> DataFrame 列名
            例如：descriptions="descriptions_col", type="type_col"
        """
        if not output_answer_key:
            raise ValueError("'output_answer_key' must be provided.")

        if len(input_keys) == 0:
            raise ValueError(
                "PromptTemplatedVQAGenerator requires at least one input key "
                "to fill the prompt template (e.g., descriptions='descriptions')."
            )

        self.logger.info("Running PromptTemplatedVQAGenerator...")

        # 1. 加载 DataFrame
        dataframe: pd.DataFrame = storage.read("dataframe")
        self.logger.info(f"Loaded dataframe with {len(dataframe)} rows")

        # 2. 动态生成 Prompt 文本
        need_fields = set(input_keys.keys())
        prompt_column = []

        for idx, row in dataframe.iterrows():
            key_dict = {}
            for key in need_fields:
                col_name = input_keys[key]  # 模板字段名 -> DataFrame 列名
                key_dict[key] = row.get(col_name, "")
            prompt_text = self.prompt_template.build_prompt(need_fields, **key_dict)
            prompt_column.append(prompt_text)

        self.logger.info(
            f"Built {len(prompt_column)} prompts using fields: {need_fields}"
        )

        # 3. 提取并清洗多模态列数据
        image_column = dataframe.get(input_image_key, pd.Series([None] * len(dataframe))).tolist()
        video_column = dataframe.get(input_video_key, pd.Series([None] * len(dataframe))).tolist()

        image_column = [path if isinstance(path, list) else [path] if pd.notna(path) else [] for path in image_column]
        video_column = [path if isinstance(path, list) else [path] if pd.notna(path) else [] for path in video_column]

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

        # 4. 构造多模态对话结构
        conversations_list = []
        image_inputs_list = None
        video_inputs_list = None

        if has_images:
            image_inputs_list = image_column
            for prompt_text, paths in zip(prompt_column, image_column):
                valid_media_count = len([p for p in paths if p])
                
                if use_api_mode:
                    content = prompt_text
                else:
                    img_tokens = "<image>" * valid_media_count
                    content = f"{img_tokens}\n{prompt_text}" if img_tokens else prompt_text
                    
                conversations_list.append([{"role": "user", "content": content}])
                
        elif has_videos:
            video_inputs_list = video_column
            for prompt_text, paths in zip(prompt_column, video_column):
                valid_media_count = len([p for p in paths if p])
                
                if use_api_mode:
                    content = prompt_text
                else:
                    vid_tokens = "<video>" * valid_media_count
                    content = f"{vid_tokens}\n{prompt_text}" if vid_tokens else prompt_text
                    
                conversations_list.append([{"role": "user", "content": content}])

        # 5. 统一调用基类接口
        outputs = self.serving.generate_from_input_messages(
            conversations=conversations_list,
            image_list=image_inputs_list,
            video_list=video_inputs_list,
            system_prompt=self.system_prompt,
        )

        dataframe[output_answer_key] = outputs
        output_file = storage.write(dataframe)
        self.logger.info(f"Results saved to {output_file}")

        return [output_answer_key]


# ==========================================
# 测试用例 (Main Block)
# ==========================================
if __name__ == "__main__":
    
    # 使用 API 模式测试
    model = APIVLMServing_openai(
        api_url="http://172.96.141.132:3001/v1",
        key_name_of_api_key="DF_API_KEY",
        model_name="gpt-5-nano-2025-08-07",
        image_io=None,
        send_request_stream=False,
        max_workers=10,
        timeout=1800
    )

    # 如需测试 Local 模型，请解开注释
    # model = LocalModelVLMServing_vllm(
    #     hf_model_name_or_path="Qwen/Qwen2.5-VL-3B-Instruct",
    #     vllm_tensor_parallel_size=1,
    #     vllm_temperature=0.7,
    #     vllm_top_p=0.9,
    #     vllm_max_tokens=512,
    # )
    
    TEMPLATE = (
        "Descriptions:\n"
        "{descriptions}\n\n"
        "Collect all details for {type} in the scene, including detailed appearance, "
        "structure, material, and special marks or logos. Do not include any analysis "
        "or your opinions, and then update the Description field with the collected details."
        "If there are no {type}s in the scene, simply state 'No {type}s found.'."
    )
    prompt_template = NamedPlaceholderPromptTemplate(template=TEMPLATE, join_list_with="\n\n")

    generator = PromptTemplatedVQAGenerator(
        serving=model,
        system_prompt="You are a helpful assistant.",
        prompt_template=prompt_template,
    )

    # 准备数据流存储
    storage = FileStorage(
        first_entry_file_name="./dataflow/example/image_to_text_pipeline/prompttemplated_vqa.jsonl", 
        cache_path="./cache_prompted_vqa",
        file_name_prefix="prompttemplated_vqa",
        cache_type="jsonl",
    )
    storage.step()  # 加载数据

    generator.run(
        storage=storage,
        input_image_key="image",
        input_video_key="video",
        output_answer_key="answer",
        # 下方为 input_keys 参数，表示：模板中 {descriptions} 对应 DataFrame 中的 "descriptions" 列
        descriptions="descriptions",
        type="type",
    )
    