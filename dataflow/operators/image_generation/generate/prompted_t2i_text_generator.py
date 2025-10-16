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


# 暂时弄成这样，先实现一下
system_prompt = lambda input_elements, input_style, template: f'''
Please help me generate a comprehensive image description based on the input elements prompts, style prompts, and the corresponding requirement template. The specific style of the generated prompts can be adjusted according to the requirement template. My requirements are as follows:

[input_elements]: {input_elements}

[input_style]: {input_style}

[template]: {template}

Please generate a comprehensive image description based on the [input_elements] and [input_style] content, following the corresponding format. Return your response in the format: '[output_prompt]: ...'
Do not include any extraneous content, and strictly adhere to the specified format in your response.
'''

nano_system_prompt = lambda input_elements, input_style: f'''
Please generate an image description based on the provided [input_elements] and [input_style]:

[input_elements]: {input_elements}

[input_style]: {input_style}

please generate an extra [output_prompt] that describes the spatial relationships and interactions between different input_elements as comprehensively as possible. If the [input_style], [output_prompt] please follow:
if
[input_elements]: 'a basketball', 'a player in a red jersey'
[input_style]: 'dynamic, energetic, sporty'
then
[output_prompt]: 'A basketball player in a vivid red jersey leaps powerfully off the polished court, gripping the basketball tightly as he soars toward the hoop. His body twists mid-air, eyes locked on the rim, while the ball is poised for a dramatic slam dunk. The scene captures the intense energy and athleticism of the moment, with the player's muscles tensed and the crowd in the background reacting to the action. The interaction between the player and the basketball is the focal point, emphasizing motion, anticipation, and the excitement of the game. The image style is dynamic, energetic, and sporty.'

otherwise, [output_prompt] please follow the template below:
if
[input_elements]: 'a basketball', 'a player in a red jersey'
then
[output_prompt]: 'A basketball player in a vivid red jersey leaps powerfully off the polished court, gripping the basketball tightly as he soars toward the hoop.  His body twists mid-air, eyes locked on the rim, while the ball is poised for a slam dunk.  The scene captures the player's muscles tensed and the crowd in the background reacting to the action.  The interaction between the player and the basketball is the focal point.'

Note: [output_prompt] must include spatial relationships between objects and interaction mechanisms. Even if there are only items, try to establish connections between objects as much as possible.
Do not include any extraneous content, and strictly adhere to the specified format in your response.
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

        templates = [TEMPLATE_ECHO4O, TEMPLATE_X2I, TEMPLATE_REFERENCE_1, TEMPLATE_REFERENCE_2, TEMPLATE_REFERENCE_3, TEMPLATE_SHORT, TEMPLATE_COMPLEX_ACTION]
        weights = [1, 2, 2, 2, 2, 2, 2]

        # 构造所有 system prompts 与其元数据（仅用于回配 global_outputs）
        per_row_meta = []          # 保存每行的条目（不回写原 df）
        global_system_prompts = [] # 全部 prompt 顺序列表
        nano_global_system_prompts = []

        for idx, row in df.iterrows():
            if output_prompt_key in row.keys():
                judge_flag = pd.isna(row[output_prompt_key])
                if bool(judge_flag == np.array([False])):
                    continue
            condition_texts = [cond_text["content"].replace(", white background", "", 1).strip() for cond_text in row[input_prompt_key]]
            styles = row[input_style_key]

            template = random.choices(templates, weights=weights, k=1)[0]
            if template in [TEMPLATE_REFERENCE_1, TEMPLATE_REFERENCE_2, TEMPLATE_REFERENCE_3, TEMPLATE_SHORT, TEMPLATE_ECHO4O]:
                styles = "N/A"

            sys_prompt = system_prompt(
                input_elements="; ".join(condition_texts),
                input_style=styles,
                template=template,
            ).strip()
            nano_prompt = nano_system_prompt(
                input_elements="; ".join(condition_texts),
                input_style=styles,
            ).strip()

            row_entries = {
                "condition_texts": condition_texts,  # 用于 input_prompt_key
                "template": "ECHO4O" if template is TEMPLATE_ECHO4O else "X2I",  # 若不需要可移除
                "system_prompt": sys_prompt,         # 若不需要可移除
            }
            global_system_prompts.append(sys_prompt)
            nano_global_system_prompts.append(nano_prompt)

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
            output_prompt = out_text.replace("[output_prompt]:", "", 1).strip()
            output_prompt_2 = nano_out_text.replace("[output_prompt]:", "", 1).strip()

            output_prompt = [{"content": output_prompt}]
            output_prompt_2 = [{"content": output_prompt_2}]
            df.at[bundle["row_idx"], output_prompt_key] = output_prompt
            df.at[bundle["row_idx"], output_prompt_key_2] = output_prompt_2

        storage.write(df)
