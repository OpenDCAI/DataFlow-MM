<<<<<<< HEAD
=======
import os

# 设置 API Key 环境变量
os.environ["DF_API_KEY"] = "sk-xxx"

>>>>>>> 59b89e3c5635df9109a0ece2565ac7dc4562ea1d
import pandas as pd
import random
from typing import List

from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger

from dataflow.utils.storage import FileStorage, DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC

<<<<<<< HEAD
from dataflow.prompts.image import PersQAGeneratorPrompt
from dataflow.operators.core_vision import PromptedVQAGenerator

import random
from typing import List, Optional
=======
from dataflow.serving.local_model_vlm_serving import LocalModelVLMServing_vllm
from dataflow.serving.api_vlm_serving_openai import APIVLMServing_openai
from dataflow.prompts.image import PersQAGeneratorPrompt


def is_api_serving(serving):
    return isinstance(serving, APIVLMServing_openai)

>>>>>>> 59b89e3c5635df9109a0ece2565ac7dc4562ea1d

@OPERATOR_REGISTRY.register()
class PersQAGenerator(OperatorABC):
    """
    Personalized QA generator.
    """

    def __init__(self, llm_serving: LLMServingABC):
        self.logger = get_logger()
<<<<<<< HEAD
        prompt_generator = PersQAGeneratorPrompt()
        self.prompt_template, self.qa_template, system_prompt = prompt_generator.build_prompt()
        self.generator = PromptedVQAGenerator(
            serving=llm_serving,
            system_prompt=system_prompt,
        )
=======
        self.serving = llm_serving
>>>>>>> 59b89e3c5635df9109a0ece2565ac7dc4562ea1d

        prompt_generator = PersQAGeneratorPrompt()
        self.prompt_template, self.qa_template, self.system_prompt = (
            prompt_generator.build_prompt()
        )
    
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "该算子用于调用视觉语言大模型生成个性化图像问答。\n\n"
                "输入参数：\n"
                "  - multi_modal_key: 多模态数据字段名 (默认: 'image')\n"
                "  - output_key: 输出问答对字段名 (默认: 'output')\n"
                "输出参数：\n"
                "  - output_key: 生成的个性化问答文本，格式为'Question: ..., Answer: ...'\n"
                "功能特点：\n"
                "  - 支持为图像中的特定人物生成个性化问答\n"
                "  - 自动为主人公分配名称标签（如'<mam>'）\n"
                "  - 从预定义问题模板中随机选择相关问题\n"
                "  - 要求模型回答时以主人公名称开头\n"
                "  - 支持批量处理多张图像\n"
                "  - 输出包含完整的问题-答案对\n"
                "应用场景：\n"
                "  - 个性化视觉问答数据集构建\n"
                "  - 人物中心的多模态对话生成\n"
                "  - 视觉语言模型的角色理解能力评估\n"
            )
        elif lang == "en":
            return (
                "This operator calls large vision-language models to generate personalized image QA pairs.\n\n"
                "Input Parameters:\n"
                "  - multi_modal_key: Multi-modal data field name (default: 'image')\n"
                "  - output_key: Output QA pairs field name (default: 'output')\n"
                "Output Parameters:\n"
                "  - output_key: Generated personalized QA text in 'Question: ..., Answer: ...' format\n"
                "Features:\n"
                "  - Generates personalized QA for specific characters in images\n"
                "  - Automatically assigns name tags (e.g., '<mam>') to main characters\n"
                "  - Randomly selects relevant questions from predefined templates\n"
                "  - Requires model to start answers with character name\n"
                "  - Supports batch processing of multiple images\n"
                "  - Output includes complete question-answer pairs\n"
                "Applications:\n"
                "  - Personalized visual QA dataset construction\n"
                "  - Character-centric multimodal dialogue generation\n"
                "  - Evaluation of role understanding in vision-language models\n"
            )
        else:
            return "PersQAGenerate produces personalized question-answer pairs for images with character focus."

<<<<<<< HEAD
    # ----------------------------- Helpers -----------------------------------
    def _build_prompts(self, df: pd.DataFrame, sks: str) -> List[str]:
        """Build one prompt per row using the configured template.

        Unlike the QA variant, this template does not depend on row fields, so
        we simply create ``len(df)`` prompts by calling ``build_prompt()``.
        """
        human_qs = self.qa_template["human_qs"]
        prompt_list = []
        for _ in range(len(df)):
            query = random.choice(human_qs).replace("<sks>", f"<{sks}>")
            prompt = self.prompt_template.format(sks=sks, query=query)
            prompt_list.append(prompt)

        return prompt_list

    @staticmethod
    def _set_first_user_message(conversation: object, value: str) -> object:
        """Safely set the first user message's 'value' in a conversation.

        Expected format: a list of messages (dicts), where the first item
        represents the user's message and has a 'value' field.
        """
=======

    def _build_prompts(self, df: pd.DataFrame, sks: str) -> List[str]:
        human_qs = self.qa_template["human_qs"]
        prompt_list = []

        for _ in range(len(df)):
            query = random.choice(human_qs).replace("<sks>", f"<{sks}>")
            prompt = self.prompt_template.format(sks=sks, query=query)
            prompt_list.append(prompt)

        return prompt_list

    @staticmethod
    def _set_first_user_message(conversation: object, value: str) -> object:
>>>>>>> 59b89e3c5635df9109a0ece2565ac7dc4562ea1d
        try:
            if isinstance(conversation, list) and conversation:
                first = conversation[0]
                if isinstance(first, dict) and "value" in first:
                    first["value"] = value
        except Exception:
<<<<<<< HEAD
            # Be defensive but don't fail the whole run
            pass
        return conversation
=======
            pass
        return conversation

>>>>>>> 59b89e3c5635df9109a0ece2565ac7dc4562ea1d

    def run(
        self,
        storage: DataFlowStorage,
        input_modal_key: str = "image",
        output_key: str = "output",
    ):
<<<<<<< HEAD
        """
        Runs the caption generation process in batch mode, reading from the input file and saving results to output.
        """
        if not output_key:
            raise ValueError("'output_key' must be a non-empty string.")

        df: pd.DataFrame = storage.read("dataframe")
        self.logger.info("Loaded dataframe with %d rows", len(df))
        
        sks = 'mam'
        prompts = self._build_prompts(df, sks)

        df["conversation"] = [
            self._set_first_user_message(conv, prompt)
            for conv, prompt in zip(df["conversation"].tolist(), prompts)
        ]

        # Write the modified dataframe back to storage
        storage.write(df)

        output_answer_key = self.generator.run(
            storage=storage.step(),
            input_conversation_key="conversation",
            input_image_key=input_modal_key,
            output_answer_key=output_key,
        )
        return [output_answer_key]
=======
        if not output_key:
            raise ValueError("'output_key' must be a non-empty string.")

        self.logger.info("Running PersQAGenerator...")

        df: pd.DataFrame = storage.read("dataframe")
        self.logger.info("Loaded dataframe with %d rows", len(df))

        sks = "mam"
        prompts = self._build_prompts(df, sks)

        df["conversation"] = [
            self._set_first_user_message(conv, prompt)
            for conv, prompt in zip(df["conversation"].tolist(), prompts)
        ]

        conversations_raw = df["conversation"].tolist()

        if input_modal_key not in df.columns:
            image_column = None
        else:
            image_column = df[input_modal_key].tolist()
            image_column = [
                path if isinstance(path, list) else [path]
                for path in image_column
            ]

            if len(image_column) == 0 or all(p is None for p in image_column):
                image_column = None

        image_inputs_list = image_column

        use_api_mode = is_api_serving(self.serving)

        if use_api_mode:
            self.logger.info("Using API serving mode")

            conversations_list = []
            for conv_raw in conversations_raw:
                conversation = []
                if isinstance(conv_raw, list):
                    for turn in conv_raw:
                        if isinstance(turn, dict):
                            role = (
                                "user"
                                if turn.get("from") == "human"
                                else "assistant"
                            )
                            content = turn.get("value", "")
                            conversation.append(
                                {"role": role, "content": content}
                            )
                conversations_list.append(conversation)

            outputs = self.serving.generate_from_input_messages(
                conversations=conversations_list,
                image_list=image_inputs_list,
                system_prompt=self.system_prompt,
            )

        else:
            self.logger.info("Using local serving mode")

            conversations_with_tokens = []

            for idx, conv_raw in enumerate(conversations_raw):
                conversation = []

                for turn_idx, turn in enumerate(conv_raw):
                    if isinstance(turn, dict):
                        is_first_user = (
                            turn.get("from") == "human" and turn_idx == 0
                        )

                        if is_first_user:
                            value = turn.get("value", "")
                            tokens = []

                            if (
                                image_inputs_list
                                and idx < len(image_inputs_list)
                                and image_inputs_list[idx]
                            ):
                                valid_images = [
                                    img
                                    for img in image_inputs_list[idx]
                                    if img is not None
                                ]
                                if valid_images:
                                    tokens.extend(
                                        ["<image>"] * len(valid_images)
                                    )

                            if tokens:
                                new_value = "".join(tokens) + value
                                turn = {**turn, "value": new_value}

                        conversation.append(turn)

                conversations_with_tokens.append(conversation)

            outputs = self.serving.generate_from_input_messages(
                conversations=conversations_with_tokens,
                image_list=image_inputs_list,
                system_prompt=self.system_prompt,
            )

        df[output_key] = outputs
        output_file = storage.write(df)

        self.logger.info("Results saved to %s", output_file)
        return [output_key]
>>>>>>> 59b89e3c5635df9109a0ece2565ac7dc4562ea1d


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

    generator = PersQAGenerator(
        llm_serving=model
    )

    storage = FileStorage(
<<<<<<< HEAD
        first_entry_file_name="./dataflow/example/image_to_text_pipeline/sample_data.json", 
=======
        first_entry_file_name="./dataflow/example/image_to_text_pipeline/sample_data.json",
>>>>>>> 59b89e3c5635df9109a0ece2565ac7dc4562ea1d
        cache_path="./cache_local",
        file_name_prefix="pers_qa",
        cache_type="json",
    )

    storage.step()

    generator.run(
        storage=storage,
        input_modal_key="image",
        output_key="pers_qa",
    )