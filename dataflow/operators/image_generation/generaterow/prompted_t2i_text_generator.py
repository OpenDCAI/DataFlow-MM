# generate extended text prompts based on provided text category
import os
import pandas as pd
import random
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger

from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import VLMServingABC


# 暂时弄成这样，先实现一下
system_prompt = lambda input_elements, input_style, template: f'''
Please help me generate a comprehensive image description based on the input elements prompts, style prompts, and the corresponding requirement template. The specific style of the generated prompts can be adjusted according to the requirement template. My requirements are as follows:

[input_elements]: {input_elements}

[input_style]: {input_style}

[template]: {template}

Please generate a comprehensive image description based on the [input_elements] and [input_style] content, following the corresponding format. Return your response in the format: '[output_prompt]: ...'
'''

# the first image
TEMPLATE_REFERENCE_1 = """
if
[input_elements]: 'a cup', 'a person'
then
[output_prompt]: 'In the target image, place the cup from the first image on a table and have the person from the second image standing next to it, holding the cup.'
(hint: Please use direct references like "the second image" to refer to input images and ensure the order corresponds to [input_elements])
"""

# Image 1
TEMPLATE_REFERENCE_2 = """
if
[input_elements]: 'a cup', 'a person'
then
[output_prompt]: 'In the target image, place the cup in Image 1 on a table and have the person in Image 2 standing next to it, holding the cup.'
(hint: Please use direct references like "Image 1" to refer to input images and ensure the order corresponds to [input_elements])
"""

# <|image_1|>
TEMPLATE_REFERENCE_3 = """
if
[input_elements]: 'a cup', 'a person'
then
[output_prompt]: 'In the target image, place the cup in <|image_1|> on a table and have the person in <|image_2|> standing next to it, holding the cup.'
(hint: Please use <|image_<idx>|> and ensure the order corresponds to [input_elements])
"""

# <img><|image_1|></img>
TEMPLATE_X2I = """
if
[input_elements]: 'a simple white cup'. 'a man wearing striking blue suit'
[input_style]: 'professional, elegant, focused'
then
[output_prompt]: 'A distinguished man sits confidently at a polished wooden table, wearing a striking blue suit that commands attention. The suit is tailored to perfection, enhancing his authoritative presence while exuding a sense of professionalism. His expression is thoughtful, with a hint of determination in his eyes, as he rests one hand on the table. In front of him, a simple white cup sits, steam gently rising from it, hinting at a moment of contemplation or perhaps a brief pause during a busy day. The scene encapsulates a blend of elegance and focus, reflecting the man's commitment to his work and the importance of the moment. The cup is the cup in <img><|image_1|></img>. The man is the one in <img><|image_2|></img>.'
(hint: Please use <img><|image_<idx>|></img> and ensure the order corresponds to [input_elements])
"""

# w/o reference to input images

# combine directly, w/o specific items
TEMPLATE_SHORT = """
if
[input_elements]: 'a cup of coffee', 'a person in a red sweater'
then
[output_prompt]: 'Combine these items in a picture.'
(hint: Do not specify specific items.)
"""

# short action and interaction, w/ specific items
TEMPLATE_ECHO4O = """
if
[input_elements]: 'person stand at a wooden counter', 'scissors', 'towel', 'lavender stem'
then
[output_prompt]: 'Have the person stand at a wooden counter, holding scissors in their right hand with the blade angled upward, while their left hand lightly touches a towel, focusing intently on cutting a lavender stem.'
(hint: Try not to make the [output_prompt] too long, brief and straightforward is preferred.)
"""

# complex action and interaction, w/ specific items
TEMPLATE_COMPLEX_ACTION = """
if
[input_elements]: 'a basketball', 'a player in a red jersey'
[input_style]: 'dynamic, energetic, sporty'
then
[output_prompt]: 'A basketball player in a vivid red jersey leaps powerfully off the polished court, gripping the basketball tightly as he soars toward the hoop. His body twists mid-air, eyes locked on the rim, while the ball is poised for a dramatic slam dunk. The scene captures the intense energy and athleticism of the moment, with the player's muscles tensed and the crowd in the background reacting to the action. The interaction between the player and the basketball is the focal point, emphasizing motion, anticipation, and the excitement of the game.'
(hint: Focus on capturing the action and interaction between the elements.)
"""


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
        storage: DataFlowStorage,
        input_element_key: str = "input_element",
        input_style_key: str = "input_style",
        input_prompt_key: str = "input_text",
        output_prompt_key: str = "instruction"
    ):
        if output_prompt_key is None:
            raise ValueError("At least one of output_key must be provided.")

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

        templates = [TEMPLATE_ECHO4O, TEMPLATE_X2I]

        # 构造所有 system prompts 与其元数据（仅用于回配 global_outputs）
        per_row_meta = []          # 保存每行的条目（不回写原 df）
        global_system_prompts = [] # 全部 prompt 顺序列表

        for idx, row in df.iterrows():
            elements_groups = row[input_element_key]
            styles_list = row[input_style_key]
            if not isinstance(elements_groups, (list, tuple)) or len(elements_groups) == 0:
                raise ValueError(f"Row {idx}: input_element must be list[list[str]]")
            if not isinstance(styles_list, (list, tuple)) or len(styles_list) == 0:
                raise ValueError(f"Row {idx}: input_style must be list[str]")

            row_entries = []
            for _ in range(self.repeat_times):
                condition_texts = []
                for __ in range(self.ip_condition_num):
                    condition_texts.append(build_one_input_text(elements_groups))

                chosen_style = pick_style(styles_list)
                template = random.choice(templates)

                sys_prompt = system_prompt(
                    input_elements="; ".join(condition_texts),
                    input_style=chosen_style,
                    template=template,
                ).strip()

                row_entries.append({
                    "condition_texts": condition_texts,  # 用于 input_prompt_key
                    "style": chosen_style,               # 若不需要可移除
                    "template": "ECHO4O" if template is TEMPLATE_ECHO4O else "X2I",  # 若不需要可移除
                    "system_prompt": sys_prompt,         # 若不需要可移除
                })
                global_system_prompts.append(sys_prompt)

            per_row_meta.append({
                "row_idx": idx,
                "entries": row_entries,
            })

        # 批量推理
        try:
            global_outputs = self.llm_serving.generate_from_input(global_system_prompts)
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
        expanded_rows = []
        cursor = 0
        for bundle in per_row_meta:
            entries = bundle["entries"]
            for entry in entries:
                out_text = global_outputs[cursor]
                cursor += 1

                # 统一输出前缀
                if isinstance(out_text, str):
                    stripped = out_text.strip()
                    if stripped.lower().startswith("[output_prompt]:"):
                        stripped = stripped.replace("[output_prompt]:", "", 1).strip()
                else:
                    # stripped = "[output_prompt]: "
                    stripped = "; ".join(entry["condition_texts"])

                # input_text_value = "; ".join(entry["condition_texts"])
                input_text_value = [{"content": f"{t}, white background"} for t in entry["condition_texts"]]
                stripped = [{"content": stripped}]

                expanded_rows.append({
                    input_prompt_key: input_text_value,
                    output_prompt_key: stripped,
                    # 如无需要，可删除以下元信息字段
                    # "_style": entry["style"],
                    # "_template": entry["template"],
                    # "_system_prompt": entry["system_prompt"],
                })

        expanded_df = pd.DataFrame(expanded_rows)

        storage.write(expanded_df)

        logger.info(f"PromptedT2ITextGenerator finished. new rows: {len(expanded_df)}")
