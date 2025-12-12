from dataflow.core.Operator import OperatorABC

from dataflow.prompts.image import QAGeneratorPrompt
import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger

from dataflow.utils.storage import FileStorage, DataFlowStorage
from dataflow.core import LLMServingABC
from dataflow.serving.local_model_vlm_serving import LocalModelVLMServing_vllm

from qwen_vl_utils import process_vision_info


@OPERATOR_REGISTRY.register()
class ImageQAGenerator(OperatorABC):
    '''
    QA Generator is a class that generates QA pairs for given images.
    '''
    def __init__(self, llm_serving: LLMServingABC):
        self.logger = get_logger()
        self.prompt_generator = QAGeneratorPrompt()
        self.llm_serving = llm_serving

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

    def _validate_dataframe(self, dataframe: pd.DataFrame):
        required_keys = [self.multi_modal_key]
        forbidden_keys = [self.output_key]

        missing = [k for k in required_keys if k not in dataframe.columns]
        conflict = [k for k in forbidden_keys if k in dataframe.columns]

        if missing:
            raise ValueError(f"Missing required column(s): {missing}")
        if conflict:
            raise ValueError(f"The following column(s) already exist and would be overwritten: {conflict}")

    def _prepare_batch_inputs(self, media_paths):
        """
        Construct batched prompts and multimodal inputs from media paths.
        """
        prompts, system_prompt = self.prompt_generator.build_prompt()

        prompt_list = []
        image_inputs_list = []

        for paths in media_paths:
            for path in paths:
                raw_prompt = [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": path},
                            {"type": "text", "text": prompts},
                        ],
                    },
                ]
                # Get multimodal inputs
                image_inputs, _ = process_vision_info(raw_prompt)
                prompt = self.llm_serving.processor.apply_chat_template(
                    raw_prompt, tokenize=False, add_generation_prompt=True
                )

                image_inputs_list.append(image_inputs)
                prompt_list.append(prompt)

        return prompt_list, image_inputs_list

    def run(
        self,
        storage: DataFlowStorage,
        input_modal_key: str = "image",
        output_key: str = "output"
    ):
        """
        Runs the QA generation process in batch mode.
        """
        self.multi_modal_key, self.output_key = input_modal_key, output_key
        # storage.step()
        dataframe = storage.read("dataframe")
        self._validate_dataframe(dataframe)
        # media_paths = storage.media_paths  # [[img1], [img2], ...]
        media_paths = dataframe.get(self.multi_modal_key, pd.Series([])).tolist()
        # 将media_paths中的非list类型的路径转换为list
        media_paths = [path if isinstance(path, list) else [path] for path in media_paths]
        
        prompt_list, image_inputs_list = self._prepare_batch_inputs(media_paths)

        outputs = self.llm_serving.generate_from_input(
            user_inputs=prompt_list,
            image_inputs=image_inputs_list
        )

        dataframe[self.output_key] = outputs
        output_file = storage.write(dataframe)
        self.logger.info(f"Results saved to {output_file}")

        return [output_key]


# if __name__ == "__main__":
#     model_path = ".Qwen/Qwen2.5-VL-3B-Instruct"

#     model = LocalModelVLMServing_vllm(
#         hf_model_name_or_path=model_path,
#         vllm_tensor_parallel_size=1,
#         vllm_temperature=0.7,
#         vllm_top_p=0.9,
#         vllm_max_tokens=512,
#     )

#     qa_generator = ImageQAGenerator(
#         llm_serving=model
#     )

#     storage = FileStorage(
#         first_entry_file_name="dataflow/example/Image2TextPipeline/test_image2qa.jsonl",
#         cache_type="jsonl",
#         media_key="image",
#         media_type="image"
#     )

#     storage.step()  # Load the data

#     qa_generator.run(
#         storage=storage,
#         multi_modal_key="image",
#         output_key="qa"
#     )