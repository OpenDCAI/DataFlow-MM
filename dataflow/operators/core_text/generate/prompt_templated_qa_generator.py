import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger

from dataflow.utils.storage import FileStorage, DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC

from dataflow.serving.local_model_vlm_serving import LocalModelVLMServing_vllm
from dataflow.prompts.prompt_template import NamedPlaceholderPromptTemplate


@OPERATOR_REGISTRY.register()
class PromptTemplatedQAGenerator(OperatorABC):
    """
    PromptTemplatedQAGenerator:
    1) 从 DataFrame 读取若干字段（由 input_keys 指定）
    2) 使用 prompt_template.build_prompt(...) 生成纯文本 prompt
    3) 将该 prompt 与 image/video 一起输入多模态模型，生成答案

    其中 prompt_template 需要实现：
        build_prompt(self, need_fields: set[str], **kwargs) -> str
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
                "PromptTemplatedQAGenerator：先用模板填充文本 prompt，再"
                "进行问答的算子。\n"
                "JSONL/DataFrame 中包含若干字段（例如 descriptions、type 等），"
                "通过 input_keys 将 DataFrame 列映射到模板字段，由 prompt_template 生成最终的文本 Prompt。"
            )
        else:
            return (
                "PromptTemplatedQAGenerator: a QA operator that first builds "
                "text prompts from a prompt template and multiple input fields, then "
                "performs QA."
            )

    def _prepare_batch_inputs(self, prompts):

        prompt_list = []

        for p in prompts:
            raw_prompt = [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": p},
                    ],
                },
            ]
            prompt = self.serving.processor.apply_chat_template(
                raw_prompt, tokenize=False, add_generation_prompt=True
            )

            prompt_list.append(prompt)

        return prompt_list

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
            例如：
                descriptions="descriptions", type="type"

        逻辑：
        1. 从 DataFrame 每行抽取 input_keys 对应列，形成 key_dict
        2. 用 prompt_template.build_prompt(need_fields, **key_dict) 得到文本 prompt
        """
        if output_answer_key is None:
            raise ValueError("output_answer_key must be provided.")

        if len(input_keys) == 0:
            raise ValueError(
                "PromptTemplatedVQAGenerator requires at least one input key "
                "to fill the prompt template (e.g., descriptions='descriptions')."
            )

        self.logger.info("Running PromptTemplatedQAGenerator...")
        self.output_answer_key = output_answer_key

        dataframe = storage.read("dataframe")
        self.logger.info(f"Loading, number of rows: {len(dataframe)}")

        need_fields = set(input_keys.keys())
        prompt_column = []

        for idx, row in dataframe.iterrows():
            key_dict = {}
            for key in need_fields:
                col_name = input_keys[key]  # 模板字段名 -> DataFrame 列名
                key_dict[key] = row[col_name]
            prompt_text = self.prompt_template.build_prompt(need_fields, **key_dict)
            prompt_column.append(prompt_text)

        self.logger.info(
            f"Using prompt_template to build prompts with fields {need_fields}, "
            f"prepared {len(prompt_column)} prompts."
        )

        prompt_list = self._prepare_batch_inputs(prompt_column)

        outputs = self.serving.generate_from_input(
            system_prompt=self.system_prompt,
            user_inputs=prompt_list
        )

        dataframe[self.output_answer_key] = outputs
        output_file = storage.write(dataframe)
        self.logger.info(f"Results saved to {output_file}")

        return output_answer_key


if __name__ == "__main__":
    model = LocalModelVLMServing_vllm(
        hf_model_name_or_path="Qwen/Qwen2.5-VL-7B-Instruct",
        vllm_tensor_parallel_size=1,
        vllm_temperature=0.7,
        vllm_top_p=0.9,
        vllm_max_tokens=512,
    )
    
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

    # Prepare input
    storage = FileStorage(
        first_entry_file_name="./dataflow/example/text_to_text/prompt_templated_qa.jsonl", 
        cache_path="./cache_prompted_qa",
        file_name_prefix="prompt_templated_qa",
        cache_type="jsonl",
    )
    storage.step()  # Load the data

    generator.run(
        storage=storage,
        output_answer_key="answer",
        descriptions="descriptions",
        type="type",
    )
