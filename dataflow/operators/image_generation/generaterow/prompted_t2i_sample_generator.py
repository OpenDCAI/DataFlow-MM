# generate extended text prompts based on provided text category
import os
import pandas as pd
import random
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger

from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import VLMServingABC


@OPERATOR_REGISTRY.register()
class PromptedT2ISampleGenerator(OperatorABC):
    def __init__(
        self,
        llm_serving: VLMServingABC,
        ip_condition_num: int = 1,
        repeat_times: int = 1
    ):
        self.llm_serving = llm_serving
        self.ip_condition_num = ip_condition_num
        self.repeat_times = repeat_times
        super().__init__()

    @staticmethod
    def get_desc(lang: str = "en") -> str:
        return (
            ""
            if lang != "zh"
            else "基于提供的文本元素内容，生成对应的文本instruction"
        )
    
    def run(
        self,
        storage: DataFlowStorage,
        input_element_key: str = "input_element",
        input_style_key: str = "input_style",
        input_prompt_key: str = "input_text",
    ):
        # 读取并基本校验
        df = storage.read(output_type="dict")
        df = pd.DataFrame(df)
        if not isinstance(df, pd.DataFrame):
            raise ValueError("storage.read must return a pandas DataFrame")

        if input_element_key not in df.columns:
            raise KeyError(f"Missing column in storage: {input_element_key}")
        if input_style_key not in df.columns:
            raise KeyError(f"Missing column in storage: {input_style_key}")

        logger = get_logger(__name__)

        def build_one_input_text(elements_groups: list[list[str]]) -> str:
            chosen = []
            for gi, group in enumerate(elements_groups):
                if not isinstance(group, (list, tuple)) or len(group) == 0:
                    raise ValueError("Each element group must be a non-empty list")
                item = random.choice(group)
                if gi == 0:
                    item = f"a photo of {item}"
                chosen.append(item)
            return ", ".join(chosen)

        def pick_style(styles: list[str]) -> str:
            if not isinstance(styles, (list, tuple)) or len(styles) == 0:
                raise ValueError("input_style must be a non-empty list")
            return random.choice(styles)

        # 构造所有 system prompts 与其元数据（仅用于回配 global_outputs）
        row_entries = []
        sample_id = 0

        for idx, row in df.iterrows():
            elements_groups = row[input_element_key]
            styles_list = row[input_style_key]
            if not isinstance(elements_groups, (list, tuple)) or len(elements_groups) == 0:
                raise ValueError(f"Row {idx}: input_element must be list[list[str]]")
            if not isinstance(styles_list, (list, tuple)) or len(styles_list) == 0:
                raise ValueError(f"Row {idx}: input_style must be list[str]")
            
            for _ in range(self.repeat_times):
                condition_texts = []
                for __ in range(self.ip_condition_num):
                    condition_texts.append(build_one_input_text(elements_groups))

                chosen_style = pick_style(styles_list)

                row_entries.append({
                    "idx": sample_id,
                    input_prompt_key: condition_texts,  # 用于 input_prompt_key
                    input_style_key: chosen_style,               # 若不需要可移除
                })
                sample_id += 1

        expanded_df = pd.DataFrame(row_entries)

        storage.write(expanded_df)

        logger.info(f"PromptedT2ITextGenerator finished. new rows: {len(expanded_df)}")
