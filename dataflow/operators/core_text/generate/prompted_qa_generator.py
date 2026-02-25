import pandas as pd
from typing import List

from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import FileStorage, DataFlowStorage

from dataflow.core import OperatorABC, LLMServingABC
from dataflow.serving.local_model_vlm_serving import LocalModelVLMServing_vllm
from dataflow.serving.api_vlm_serving_openai import APIVLMServing_openai


# 提取判断是否为 API Serving 的辅助函数
def is_api_serving(serving):
    return isinstance(serving, APIVLMServing_openai)


@OPERATOR_REGISTRY.register()
class PromptedQAGenerator(OperatorABC):
    """
    PromptedQAGenerator read prompt and generate answers.
    """
    def __init__(self, 
                 serving: LLMServingABC, 
                 system_prompt: str = "You are a helpful assistant."):
        self.logger = get_logger()
        self.serving = serving
        self.system_prompt = system_prompt
            
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "基础文本问答算子 (PromptedQAGenerator)。\n"
                "直接读取指定列作为 prompt，生成纯文本答案。\n\n"
                "特点：\n"
                "  - 极简纯文本问答\n"
                "  - 统一支持 API 和本地 Local 模型部署模式\n"
                "  - 全局 Batch 处理，极简代码结构\n"
            )
        else:
            return "Read prompt to generate answers."
    
    def run(self, 
            storage: DataFlowStorage,
            input_prompt_key: str = "prompt",
            output_answer_key: str = "answer",
            ):
        if not output_answer_key:
            raise ValueError("'output_answer_key' must be provided.")

        self.logger.info("Running PromptedQAGenerator...")

        # 1. 加载 DataFrame
        dataframe: pd.DataFrame = storage.read('dataframe')
        self.logger.info(f"Loaded dataframe with {len(dataframe)} rows")

        use_api_mode = is_api_serving(self.serving)
        if use_api_mode:
            self.logger.info("Using API serving mode")
        else:
            self.logger.info("Using local serving mode")

        # 2. 提取并清洗 Prompt 数据
        prompt_column = dataframe.get(input_prompt_key, pd.Series([None] * len(dataframe))).tolist()
        
        # 组装为基类所需的消息格式，同时处理可能存在的 NaN 空值
        conversations_list = []
        for p in prompt_column:
            safe_prompt = str(p) if pd.notna(p) else ""
            conversations_list.append([{"role": "user", "content": safe_prompt}])

        # 3. 统一调用基类接口进行推理
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

    # 如需使用本地模型，请解开注释
    # model = LocalModelVLMServing_vllm(
    #     hf_model_name_or_path="/data0/happykeyan/Models/Qwen2.5-VL-3B-Instruct",
    #     vllm_tensor_parallel_size=1,
    #     vllm_temperature=0.7,
    #     vllm_top_p=0.9,
    #     vllm_max_tokens=512,
    # )

    generator = PromptedQAGenerator(
        serving=model,
        system_prompt="You are a helpful assistant. Return the value of the math expression in the user prompt.",
    )

    # Prepare input
    storage = FileStorage(
        first_entry_file_name="./dataflow/example/text_to_text/prompted_qa.jsonl", 
        cache_path="./cache_prompted_qa",
        file_name_prefix="prompted_qa",
        cache_type="jsonl",
    )
    storage.step()  # Load the data

    generator.run(
        storage=storage,
        input_prompt_key="prompt",
        output_answer_key="answer",
    )