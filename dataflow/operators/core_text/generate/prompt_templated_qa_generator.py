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
class PromptTemplatedQAGenerator(OperatorABC):
    """
    PromptTemplatedQAGenerator:
    1) 从 DataFrame 读取若干字段（由 input_keys 指定）
    2) 使用 prompt_template.build_prompt(...) 生成纯文本 prompt
    3) 将该 prompt 输入大语言模型，生成纯文本答案
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
                "prompt_template cannot be None for PromptTemplatedQAGenerator."
            )

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "基于模板的纯文本问答算子 (PromptTemplatedQAGenerator)。\n"
                "JSONL/DataFrame 中包含若干字段，通过 input_keys 将列映射到模板字段，\n"
                "由 prompt_template 动态生成纯文本 Prompt，进行批量问答。\n\n"
                "特点：\n"
                "  - 支持动态组装复杂的纯文本 Prompt\n"
                "  - 统一支持 API 和本地 Local 模型部署模式\n"
                "  - 全局 Batch 处理，极简代码结构\n"
            )
        else:
            return (
                "PromptTemplatedQAGenerator: a pure text QA operator that builds "
                "text prompts from a template and multiple input fields, then "
                "performs QA inference."
            )

    def run(
        self,
        storage: DataFlowStorage,
        output_answer_key: str = "answer",
        **input_keys,
    ):
        """
        参数：
        - storage: DataFlowStorage
        - output_answer_key: 输出答案列名
        - **input_keys: 模板字段名 -> DataFrame 列名
            例如：descriptions="descriptions_col", type="type_col"
        """
        if not output_answer_key:
            raise ValueError("'output_answer_key' must be provided.")

        if len(input_keys) == 0:
            raise ValueError(
                "PromptTemplatedQAGenerator requires at least one input key "
                "to fill the prompt template (e.g., descriptions='descriptions')."
            )

        self.logger.info("Running PromptTemplatedQAGenerator...")

        # 1. 加载 DataFrame
        dataframe: pd.DataFrame = storage.read("dataframe")
        self.logger.info(f"Loaded dataframe with {len(dataframe)} rows")

        use_api_mode = is_api_serving(self.serving)
        if use_api_mode:
            self.logger.info("Using API serving mode")
        else:
            self.logger.info("Using local serving mode")

        # 2. 动态生成 Prompt 文本并组装标准对话结构
        need_fields = set(input_keys.keys())
        conversations_list = []

        for idx, row in dataframe.iterrows():
            key_dict = {}
            for key in need_fields:
                col_name = input_keys[key]  # 模板字段名 -> DataFrame 列名
                # 安全获取值，防止 NaN 导致字符串格式化异常
                val = row.get(col_name)
                key_dict[key] = val if pd.notna(val) else ""
                
            prompt_text = self.prompt_template.build_prompt(need_fields, **key_dict)
            
            # 统一组装为基类所需的消息格式
            conversations_list.append([{"role": "user", "content": prompt_text}])

        self.logger.info(
            f"Built {len(conversations_list)} prompts using fields: {need_fields}"
        )

        # 3. 统一调用基类接口进行纯文本推理 (无需传入 image_list/video_list)
        outputs = self.serving.generate_from_input_messages(
            conversations=conversations_list,
            system_prompt=self.system_prompt,
        )

        # 4. 保存结果
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

    # 如需测试 Local 模型，请解开注释 (VLM 模型同样能处理纯文本)
    # model = LocalModelVLMServing_vllm(
    #     hf_model_name_or_path="Qwen/Qwen2.5-VL-7B-Instruct",
    #     vllm_tensor_parallel_size=1,
    #     vllm_temperature=0.7,
    #     vllm_top_p=0.9,
    #     vllm_max_tokens=512,
    # )
    
    TEMPLATE = (
        "Descriptions:\n"
        "{descriptions}\n\n"
        "Please collect all details for {type} in the Descriptions and print them out. "
        "Eg: If the Descriptions are 'A red car is driving on the road, with a blue sky in the background.', "
        "then the output should be 'Car: red, sky: blue'.\n\n"
        "If there are no {type}s in the Descriptions, simply state 'No {type}s found.'."
    )
    prompt_template = NamedPlaceholderPromptTemplate(template=TEMPLATE, join_list_with="\n\n")

    generator = PromptTemplatedQAGenerator(
        serving=model,
        system_prompt="You are a helpful assistant.",
        prompt_template=prompt_template,
    )

    # 准备输入数据
    storage = FileStorage(
        first_entry_file_name="./dataflow/example/text_to_text/prompt_templated_qa.jsonl", 
        cache_path="./cache_prompted_qa",
        file_name_prefix="prompt_templated_qa",
        cache_type="jsonl",
    )
    storage.step()  # 加载数据

    generator.run(
        storage=storage,
        output_answer_key="answer",
        descriptions="descriptions",
        type="type",
    )