import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger

from dataflow.utils.storage import FileStorage, DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC
from dataflow.serving.local_model_vlm_serving import LocalModelVLMServing_vllm

from dataflow.operators.core_vision import PromptedVQAGenerator

@OPERATOR_REGISTRY.register()
class ImageCaptionGenerator(OperatorABC):
    '''
    Caption Generator is a class that generates captions for given images.
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
                "该算子用于调用视觉语言大模型生成图像描述。\n\n"
                "输入参数：\n"
                "  - multi_modal_key: 多模态数据字段名 (默认: 'image')\n"
                "  - output_key: 输出描述字段名 (默认: 'output')\n"
                "输出参数：\n"
                "  - output_key: 生成的图像描述文本\n"
                "功能特点：\n"
                "  - 支持批量处理多张图像\n"
                "  - 基于Qwen等视觉语言模型生成高质量描述\n"
                "  - 自动处理图像输入和提示词构建\n"
            )
        elif lang == "en":
            return (
                "This operator calls large vision-language models to generate image captions.\n\n"
                "Input Parameters:\n"
                "  - multi_modal_key: Multi-modal data field name (default: 'image')\n"
                "  - output_key: Output caption field name (default: 'output')\n"
                "Output Parameters:\n"
                "  - output_key: Generated image description text\n"
                "Features:\n"
                "  - Supports batch processing of multiple images\n"
                "  - Generates high-quality captions using models like Qwen\n"
                "  - Automatically handles image inputs and prompt construction\n"
            )
        else:
            return "ImageCaptionGenerate produces textual descriptions for given images using vision-language models."

    def run(
        self,
        storage: DataFlowStorage,
        input_modal_key: str = "image", 
        output_key: str = "output"
    ):
        """
        Runs the caption generation process in batch mode, reading from the input file and saving results to output.
        """
        output_answer_key = self.generator.run(
            storage=storage,
            input_conversation_key="conversation",
            input_image_key=input_modal_key,
            output_answer_key=output_key,
        )
        return [output_answer_key]

if __name__ == "__main__":
    # Initialize model
    model = LocalModelVLMServing_vllm(
        hf_model_name_or_path="Qwen/Qwen2.5-VL-3B-Instruct",
        vllm_tensor_parallel_size=1,
        vllm_temperature=0.7,
        vllm_top_p=0.9,
        vllm_max_tokens=512,
    )

    caption_generator = ImageCaptionGenerator(
        llm_serving=model,
        system_prompt="You are a image caption generator. Your task is to generate a concise and informative caption for the given image content.",
    )

    # Prepare input
    storage = FileStorage(
        first_entry_file_name="./dataflow/example/image_to_text_pipeline/capsbench_captions.json", 
        cache_path="./cache_local",
        file_name_prefix="caption",
        cache_type="json",
    )
    storage.step()  # Load the data

    caption_generator.run(
        storage=storage,
        input_modal_key="image",
        output_key="caption"
    )