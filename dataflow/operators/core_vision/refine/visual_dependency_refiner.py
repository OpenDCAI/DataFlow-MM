import re
import random
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple

from dataflow import get_logger
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC, LLMServingABC
from qwen_vl_utils import process_vision_info


def shuffle_options_logic(qa_item: Dict[str, Any], add_none_option: bool = False) -> Tuple[str, str]:
    options = qa_item.get("options", {})
    correct_letter = qa_item.get("answer")
    correct_text = options.get(correct_letter)
    
    items = list(options.items()) 
    if not items or not correct_text:
        return qa_item["question"], correct_letter

    texts = [v for k, v in items]
    random.shuffle(texts)
    
    new_labels = ["A", "B", "C", "D", "E", "F"]
    new_answer_letter = None
    
    q_lines = [qa_item["question_title"]]
    
    current_idx = 0
    for i, txt in enumerate(texts):
        lbl = new_labels[i]
        q_lines.append(f"   - {lbl}) {txt}")
        if txt == correct_text:
            new_answer_letter = lbl
        current_idx = i
            
    if add_none_option:
        next_lbl = new_labels[current_idx + 1]
        q_lines.append(f"   - {next_lbl}) None of the above")
        
    return "\n".join(q_lines), new_answer_letter

def extract_letter_only(model_out: str) -> Optional[str]:
    if not model_out: return None
    m = re.search(r"\b([A-Fa-f])\b", model_out)
    if m: return m.group(1).upper()
    m2 = re.search(r"(?:answer|option)\s*[:：]\s*([A-Fa-f])", model_out, re.I)
    if m2: return m2.group(1).upper()
    return None


@OPERATOR_REGISTRY.register()
class VisualDependencyRefiner(OperatorABC):
    """
    [Refine] 视觉依赖性校验器。
    对 MCQ 列表进行“旋转 + 双盲测试 (Visual vs Text-Only)”：
    筛选出 必须依赖视觉信息 (High Visual Acc) 且 不能仅凭常识 (Low Text Acc) 的问题。
    """
    def __init__(
        self, 
        serving: LLMServingABC, 
        instruction_template: str,
        rotate_num: int = 4,
        pass_visual_min: float = 1.0,
        pass_textual_max: float = 0.25, 
        add_none_above_visual: bool = True
    ):
        self.serving = serving
        self.inst_template = instruction_template
        self.rotate_num = max(1, rotate_num)
        self.pass_visual_min = pass_visual_min
        self.pass_textual_max = pass_textual_max
        self.add_none = add_none_above_visual
        self.logger = get_logger()

    @staticmethod
    def get_desc(lang: str = "zh"):
        return (
            "视觉依赖性校验算子 (VisualDependencyRefiner)。\n"
            "通过多次旋转选项并进行 有图/无图 对比测试，筛选出必须依赖视觉信息才能回答的高质量 MCQ。"
        ) if lang == "zh" else "Visual Dependency Refiner: Filters MCQs requiring visual info via rotation checks."

    def run(self, storage: DataFlowStorage, input_list_key: str, input_image_key: str, output_key: str):
        self.logger.info(f"Running VisualDependencyRefiner on {input_list_key}...")
        df = storage.read("dataframe")
        
        filtered_results = []
        
        for idx, row in df.iterrows():
            qa_list = row.get(input_list_key, [])
            image_path = row.get(input_image_key)
            
            if not qa_list or not isinstance(qa_list, list) or not image_path:
                filtered_results.append([])
                continue
            
            kept_qas = []
            
            # 遍历该图生成的每一道题
            for qa_item in qa_list:
                
                # --- 分离 Batch ---
                # 我们不再把 VQA 和 QA 混在一起发，而是攒成两个独立的 Batch
                visual_prompts = []
                visual_images = []
                visual_answers = [] # 记录对应的正确答案
                
                text_prompts = []
                text_answers = []
                
                # 准备数据 (rotate_num 次)
                for _ in range(self.rotate_num):
                    # 1. Visual Case
                    q_v, ans_v = shuffle_options_logic(qa_item, add_none_option=self.add_none)
                    raw_v = [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": [
                            {"type": "image", "image": image_path},
                            {"type": "text", "text": self.inst_template.format(q_v)}
                        ]}
                    ]
                    img_inp, _ = process_vision_info(raw_v)
                    p_v = self.serving.processor.apply_chat_template(raw_v, tokenize=False, add_generation_prompt=True)
                    
                    # Qwen 防御性 Patch
                    if "<|image_pad|>" not in p_v and "<image>" not in p_v:
                         p_v = "<|vision_start|><|image_pad|><|vision_end|>" + p_v
                    
                    visual_prompts.append(p_v)
                    visual_images.append(img_inp)
                    visual_answers.append(ans_v)
                    
                    # 2. Text-Only Case
                    q_t, ans_t = shuffle_options_logic(qa_item, add_none_option=False)
                    # 纯文本请求不需要 System Prompt，或者保持一致
                    raw_t = [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": [{"type": "text", "text": self.inst_template.format(q_t)}]}
                    ]
                    # 纯文本不用 process_vision_info
                    p_t = self.serving.processor.apply_chat_template(raw_t, tokenize=False, add_generation_prompt=True)
                    
                    text_prompts.append(p_t)
                    text_answers.append(ans_t)
                
                if not visual_prompts:
                    continue

                # --- 分别调用 ---
                
                # 1. Visual Batch (image_inputs != None)
                # 触发 Server 的“多模态模式”分支
                vis_outputs = self.serving.generate_from_input(
                    user_inputs=visual_prompts,
                    image_inputs=visual_images
                )
                
                # 2. Text Batch (image_inputs == None)
                # 触发 Server 的“纯文本模式”分支
                txt_outputs = self.serving.generate_from_input(
                    user_inputs=text_prompts,
                    image_inputs=None  # 显式传 None
                )
                
                # --- 统计结果 ---
                v_correct = 0
                l_correct = 0
                
                for i in range(self.rotate_num):
                    # 提取 Visual 结果
                    pred_v = extract_letter_only(vis_outputs[i])
                    if pred_v == visual_answers[i]:
                        v_correct += 1
                        
                    # 提取 Text 结果
                    pred_t = extract_letter_only(txt_outputs[i])
                    if pred_t == text_answers[i]:
                        l_correct += 1
                
                v_acc = v_correct / self.rotate_num
                l_acc = l_correct / self.rotate_num
                
                if v_acc >= self.pass_visual_min and l_acc <= self.pass_textual_max:
                    qa_item["stats"] = {"v_acc": v_acc, "t_acc": l_acc}
                    kept_qas.append(qa_item)
            
            filtered_results.append(kept_qas)
            
        df[output_key] = filtered_results
        storage.write(df)
        return [output_key]
    