import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger

from dataflow.utils.storage import FileStorage, DataFlowStorage
from dataflow.core.Operator import OperatorABC
from dataflow.core import LLMServingABC
from dataflow.serving.local_model_vlm_serving import LocalModelVLMServing_vllm

from dataflow.operators.core_vision import PromptedVQAGenerator

@OPERATOR_REGISTRY.register()
class ImageQAGenerator(OperatorABC):
    '''
    QA Generator is a class that generates QA pairs for given images.
    '''
    def __init__(self, llm_serving: LLMServingABC, system_prompt: str):
        self.logger = get_logger()
        self.generator = PromptedVQAGenerator(
            serving=llm_serving,
            system_prompt=system_prompt,
        )

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "该算子用于调用视觉语言大模型生成图像问答对。\n\n"
                "输入参数：\n"
                "  - multi_modal_key: 多模态数据字段名 (默认: 'image')\n"
                "  - output_key: 输出问答对字段名 (默认: 'output')\n"
                "输出参数：\n"
                "  - output_key: 生成的问答对文本\n"
                "功能特点：\n"
                "  - 支持批量处理多张图像\n"
                "  - 基于视觉语言模型自动生成相关问答\n"
                "  - 可应用于视觉问答数据集构建和模型训练\n"
                "  - 自动处理图像输入和问答提示词构建\n"
            )
        elif lang == "en":
            return (
                "This operator calls large vision-language models to generate question-answer pairs from images.\n\n"
                "Input Parameters:\n"
                "  - multi_modal_key: Multi-modal data field name (default: 'image')\n"
                "  - output_key: Output QA pairs field name (default: 'output')\n"
                "Output Parameters:\n"
                "  - output_key: Generated question-answer pairs text\n"
                "Features:\n"
                "  - Supports batch processing of multiple images\n"
                "  - Automatically generates relevant QA pairs using vision-language models\n"
                "  - Applicable for visual question-answering dataset construction and model training\n"
                "  - Automatically handles image inputs and QA prompt construction\n"
            )
        else:
            return "ImageQAGenerate produces question-answer pairs for given images using vision-language models."

    def run(
        self,
        storage: DataFlowStorage,
        input_modal_key: str = "image", 
        output_key: str = "output"
    ):
        """
        Runs the QA generation process in batch mode.
        """
        output_answer_key = self.generator.run(
            storage=storage,
            input_conversation_key="conversation",
            input_image_key=input_modal_key,
            output_answer_key=output_key,
        )
        return [output_answer_key]

if __name__ == "__main__":
    model = LocalModelVLMServing_vllm(
        hf_model_name_or_path="Qwen/Qwen2.5-VL-3B-Instruct",
        vllm_tensor_parallel_size=1,
        vllm_temperature=0.7,
        vllm_top_p=0.9,
        vllm_max_tokens=512,
    )

    qa_generator = ImageQAGenerator(
        llm_serving=model,
        system_prompt="You are a image question-answer generator. Your task is to generate a question-answer pair for the given image content.",
    )

    storage = FileStorage(
        first_entry_file_name="./dataflow/example/image_to_text_pipeline/capsbench_qas.json",
        cache_path="./cache_local",
        file_name_prefix="qa",
        cache_type="json",
    )
    storage.step()  # Load the data

    qa_generator.run(
        storage=storage,
        input_modal_key="image",
        output_key="qa"
    )