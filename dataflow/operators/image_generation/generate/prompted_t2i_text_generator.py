# generate extended text prompts based on provided text category
import os
import pandas as pd
import random
import numpy as np
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger

from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import VLMServingABC


@OPERATOR_REGISTRY.register()
class PromptedT2ITextGenerator(OperatorABC):
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
        prompt_generator,
        storage: DataFlowStorage,
        input_style_key: str = "input_style",
        input_prompt_key: str = "input_text",
        output_prompt_key: str = "instruction",
        output_prompt_key_2: str = "output_img_discript",
    ):
        if output_prompt_key is None:
            raise ValueError("At least one of output_key must be provided.")

        # 读取并基本校验
        df = storage.read(output_type="dict")
        df = pd.DataFrame(df)
        if not isinstance(df, pd.DataFrame):
            raise ValueError("storage.read must return a pandas DataFrame")

        if input_prompt_key not in df.columns:
            raise KeyError(f"Missing column in storage: {input_prompt_key}")
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

        # 构造所有 system prompts
        per_row_meta = []          # 保存每行的条目（不回写原 df）
        global_system_prompts = [] # 全部 prompt 顺序列表
        nano_global_system_prompts = []

        for idx, row in df.iterrows():
            if output_prompt_key in row.keys():
                judge_flag = pd.isna(row[output_prompt_key])
                if bool(judge_flag == np.array([False])):
                    continue
            
            condition_texts, sys_prompt, nano_prompt = prompt_generator.generate_prompt(
                row,
                input_prompt_key=input_prompt_key,
                input_style_key=input_style_key,
            )

            row_entries = {
                "condition_texts": condition_texts,  # 用于 input_prompt_key
                "system_prompt": sys_prompt,         # 若不需要可移除
            }
            global_system_prompts.append(sys_prompt)
            nano_global_system_prompts.append(nano_prompt)
            # if len(global_system_prompts) >= 200:
            #     break

            per_row_meta.append({
                "row_idx": row["idx"],
                "entries": row_entries,
            })
        # 批量推理
        try:
            global_outputs = self.llm_serving.generate_from_input(global_system_prompts)
            nano_global_outputs = self.llm_serving.generate_from_input(nano_global_system_prompts)
        except Exception:
            logger.exception("llm_serving.generate_from_input failed")
            raise

        if not isinstance(global_outputs, (list, tuple)):
            raise ValueError("llm_serving.generate_from_input must return a list of strings")
        if len(global_outputs) != len(global_system_prompts):
            raise ValueError(
                f"LLM output length mismatch: got {len(global_outputs)} != {len(global_system_prompts)}"
            )

        # 组装“全新 df”：仅保留 condition_texts -> input_prompt_key 与 对应输出 -> output_prompt_key
        cursor = 0
        for bundle in per_row_meta:
            entries = bundle["entries"]
            out_text = global_outputs[cursor]
            nano_out_text = nano_global_outputs[cursor]
            cursor += 1

            # 统一输出前缀
            try:
                output_prompt = out_text.replace("[output_prompt]:", "", 1).strip()
                output_prompt_2 = nano_out_text.replace("[output_prompt]:", "", 1).strip()
            except:
                continue

            output_prompt = [{"content": output_prompt}]
            output_prompt_2 = [{"content": "Generate a naturalistic image based on the following description. The central elements must be seamlessly integrated, maintaining visual continuity without separation: " + output_prompt_2}]
            df.at[bundle["row_idx"], output_prompt_key] = output_prompt
            df.at[bundle["row_idx"], output_prompt_key_2] = output_prompt_2

        storage.write(df)
