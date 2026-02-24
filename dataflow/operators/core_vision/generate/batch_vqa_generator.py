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
class BatchVQAGenerator(OperatorABC):
    """
    [Generate] 批量视觉问答生成器。
    输入：问题列表 (Questions) + 图片。
    输出：答案列表 (New Text Content)。
    """
    def __init__(self, serving: LLMServingABC, system_prompt: str = "You are a helpful assistant."):
        self.serving = serving
        self.system_prompt = system_prompt
        self.logger = get_logger()

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "批量视觉问答生成算子 (BatchVQAGenerator)。\n"
                "该算子用于针对单张图片回答列表中的多个问题 (One Image, Many Questions)。\n\n"
                "输入参数：\n"
                "  - input_prompts_key: 问题列表列 (List[str])\n"
                "  - input_image_key: 图像列\n"
                "输出参数：\n"
                "  - output_key: 生成的答案列表列 (List[str])\n"
                "功能特点：\n"
                "  - 自动进行广播 (Broadcasting)，将单图映射到多个问题\n"
                "  - 统一支持 API 和本地 Local 模型部署模式\n"
                "  - 支持全局批处理加速推理\n"
            )
        else:
            return (
                "Batch VQA Generator (BatchVQAGenerator).\n"
                "This operator answers multiple questions based on a single image (One Image, Many Questions).\n\n"
                "Input Parameters:\n"
                "  - input_prompts_key: Column containing the list of questions\n"
                "  - input_image_key: Column containing the image\n"
                "Output Parameters:\n"
                "  - output_key: Column storing the list of generated answers\n"
                "Features:\n"
                "  - Automatically broadcasts one image to multiple prompts\n"
                "  - Unifies support for API and Local model deployment modes\n"
                "  - Supports global batch processing for faster inference\n"
            )

    def run(self, storage: DataFlowStorage, input_prompts_key: str, input_image_key: str, output_key: str):
        self.logger.info(f"Running BatchVQAGenerator on {input_prompts_key}...")
        df: pd.DataFrame = storage.read("dataframe")
        
        use_api_mode = is_api_serving(self.serving)
        if use_api_mode:
            self.logger.info("Using API serving mode")
        else:
            self.logger.info("Using local serving mode")

        # 1. 展平数据阶段 (Flatten Data)
        # 将 [ [q1, q2], [q3] ] 展平为 [q1, q2, q3]，以便一次性送入大模型获得最高并发性能
        flat_conversations = []
        flat_images = []
        row_question_counts = [] # 记录每一行有几个问题，用于后续重组答案

        for idx, row in df.iterrows():
            questions = row.get(input_prompts_key, [])
            image_path = row.get(input_image_key)
            
            # 统一将图片路径处理为 List 格式
            if isinstance(image_path, str):
                image_path = [image_path]
            elif not image_path:
                image_path = []

            if not isinstance(questions, list):
                questions = []

            row_question_counts.append(len(questions))

            for q in questions:
                # 构造标准对话格式
                if use_api_mode:
                    # API 模式通常只需要标准文本，图片通过 image_list 单独传入
                    conversation = [{"role": "user", "content": q}]
                else:
                    # Local 模式（如 vLLM）通常需要手动在文本前拼接 <image> 占位符
                    img_tokens = "<image>" * len(image_path)
                    conversation = [{"role": "user", "content": img_tokens + q}]

                flat_conversations.append(conversation)
                flat_images.append(image_path)

        # 2. 批量推理阶段 (Batch Inference)
        if flat_conversations:
            flat_outputs = self.serving.generate_from_input_messages(
                conversations=flat_conversations,
                image_list=flat_images,
                system_prompt=self.system_prompt,
            )
        else:
            flat_outputs = []

        # 3. 重组数据阶段 (Unflatten Data)
        # 将展平的输出 [a1, a2, a3] 根据 row_question_counts 重组回 [ [a1, a2], [a3] ]
        all_answers_nested = []
        current_idx = 0
        for count in row_question_counts:
            row_answers = flat_outputs[current_idx : current_idx + count]
            all_answers_nested.append(row_answers)
            current_idx += count
            
        df[output_key] = all_answers_nested
        output_file = storage.write(df)
        
        self.logger.info("Results saved to %s", output_file)
        return [output_key]


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

    # 如果需要测试本地模型，可以解开注释：
    # model = LocalModelVLMServing_vllm(
    #     hf_model_name_or_path="Qwen/Qwen2.5-VL-3B-Instruct",
    #     vllm_tensor_parallel_size=1,
    #     ...
    # )

    generator = BatchVQAGenerator(
        serving=model,
        system_prompt="You are a helpful visual assistant."
    )

    storage = FileStorage(
        first_entry_file_name="./dataflow/example/image_to_text_pipeline/sample_data.json",
        cache_path="./cache_local",
        file_name_prefix="batch_vqa",
        cache_type="json",
    )
    
    storage.step()

    generator.run(
        storage=storage,
        input_prompts_key="questions",  # 假设输入列包含多个问题
        input_image_key="image",
        output_key="answers",           # 输出列表
    )