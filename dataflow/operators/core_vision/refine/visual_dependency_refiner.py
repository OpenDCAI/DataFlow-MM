import re
import random
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple

from dataflow import get_logger
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import FileStorage, DataFlowStorage
from dataflow.core import OperatorABC, LLMServingABC

from dataflow.serving.local_model_vlm_serving import LocalModelVLMServing_vllm
from dataflow.serving.api_vlm_serving_openai import APIVLMServing_openai


# 提取判断是否为 API Serving 的辅助函数
def is_api_serving(serving):
    return isinstance(serving, APIVLMServing_openai)


def shuffle_options_logic(qa_item: Dict[str, Any], add_none_option: bool = False) -> Tuple[str, str]:
    """混淆选项逻辑 (保持原版业务逻辑不变)"""
    options = qa_item.get("options", {})
    correct_letter = qa_item.get("answer")
    correct_text = options.get(correct_letter)
    
    items = list(options.items()) 
    if not items or not correct_text:
        return qa_item.get("question", ""), correct_letter

    texts = [v for k, v in items]
    random.shuffle(texts)
    
    new_labels = ["A", "B", "C", "D", "E", "F"]
    new_answer_letter = None
    
    q_lines = [qa_item.get("question_title", "")]
    
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
    """提取大模型回复中的选项字母"""
    if not model_out: 
        return None
    model_out = str(model_out)
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
        self.system_prompt = "You are a helpful assistant."
        self.logger = get_logger()

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "视觉依赖性校验算子 (VisualDependencyRefiner)。\n"
                "通过多次旋转选项并进行 有图/无图 对比测试，筛选出必须依赖视觉信息才能回答的高质量 MCQ。\n\n"
                "特点：\n"
                "  - 双盲精度测试：$V_{acc}$ 与 $T_{acc}$ 联合校验\n"
                "  - 全局并行批处理：成百上千倍提升过滤吞吐量\n"
                "  - 统一 API 与本地模型接口，自动管理多模态 Token\n"
            )
        else:
            return "Visual Dependency Refiner: Filters MCQs requiring visual info via global rotation checks."

    def run(self, storage: DataFlowStorage, input_list_key: str, input_image_key: str, output_key: str):
        if not output_key:
            raise ValueError("'output_key' must be provided.")

        self.logger.info(f"Running VisualDependencyRefiner on {input_list_key}...")
        df: pd.DataFrame = storage.read("dataframe")
        
        use_api_mode = is_api_serving(self.serving)
        
        # =========================================================
        # 1. 全局展平阶段 (Global Flattening)
        # =========================================================
        vis_conversations, vis_images, vis_mappings = [], [], []
        txt_conversations, txt_mappings = [], []

        for row_idx, row in df.iterrows():
            qa_list = row.get(input_list_key, [])
            image_path = row.get(input_image_key)

            # 清洗图片路径
            if isinstance(image_path, str):
                image_path = [image_path]
            elif not image_path:
                image_path = []

            if not qa_list or not isinstance(qa_list, list) or not image_path:
                continue

            for qa_idx, qa_item in enumerate(qa_list):
                # 对每一道题，生成 rotate_num 次 有图 & 无图 变体
                for _ in range(self.rotate_num):
                    
                    # --- 1. Visual Case (有图分支) ---
                    q_v, ans_v = shuffle_options_logic(qa_item, add_none_option=self.add_none)
                    prompt_v = self.inst_template.format(q_v)
                    
                    if use_api_mode:
                        content_v = prompt_v
                    else:
                        img_tokens = "<image>" * len(image_path)
                        content_v = f"{img_tokens}\n{prompt_v}" if img_tokens else prompt_v
                        
                    vis_conversations.append([{"role": "user", "content": content_v}])
                    vis_images.append(image_path)
                    vis_mappings.append({"row_idx": row_idx, "qa_idx": qa_idx, "expected": ans_v})

                    # --- 2. Text-Only Case (纯文本无图分支) ---
                    q_t, ans_t = shuffle_options_logic(qa_item, add_none_option=False)
                    prompt_t = self.inst_template.format(q_t)
                    
                    txt_conversations.append([{"role": "user", "content": prompt_t}])
                    txt_mappings.append({"row_idx": row_idx, "qa_idx": qa_idx, "expected": ans_t})

        # =========================================================
        # 2. 全局双轨推理阶段 (Parallel Batch Inference)
        # =========================================================
        vis_outputs = []
        if vis_conversations:
            self.logger.info(f"Running VISUAL batch inference for {len(vis_conversations)} items...")
            vis_outputs = self.serving.generate_from_input_messages(
                conversations=vis_conversations,
                image_list=vis_images,
                system_prompt=self.system_prompt
            )

        txt_outputs = []
        if txt_conversations:
            self.logger.info(f"Running TEXT-ONLY batch inference for {len(txt_conversations)} items...")
            txt_outputs = self.serving.generate_from_input_messages(
                conversations=txt_conversations,
                # 显式传入 None，触发模型纯文本分支
                image_list=None,
                system_prompt=self.system_prompt
            )

        # =========================================================
        # 3. 计分与回填过滤阶段 (Scoring & Filtering)
        # =========================================================
        # 计分板格式: (row_idx, qa_idx) -> {"v_correct": 0, "t_correct": 0}
        qa_stats = {}

        # 统计 Visual 得分
        for mapping, out_text in zip(vis_mappings, vis_outputs):
            key = (mapping["row_idx"], mapping["qa_idx"])
            if key not in qa_stats:
                qa_stats[key] = {"v_correct": 0, "t_correct": 0}
                
            pred = extract_letter_only(out_text)
            if pred == mapping["expected"]:
                qa_stats[key]["v_correct"] += 1

        # 统计 Text 得分
        for mapping, out_text in zip(txt_mappings, txt_outputs):
            key = (mapping["row_idx"], mapping["qa_idx"])
            pred = extract_letter_only(out_text)
            if pred == mapping["expected"]:
                qa_stats[key]["t_correct"] += 1

        # 最终回填
        filtered_results = [[] for _ in range(len(df))]
        
        for row_idx, row in df.iterrows():
            qa_list = row.get(input_list_key, [])
            if not isinstance(qa_list, list):
                continue
                
            for qa_idx, qa_item in enumerate(qa_list):
                stats = qa_stats.get((row_idx, qa_idx))
                if not stats:
                    continue
                
                # 计算通过率
                v_acc = stats["v_correct"] / self.rotate_num
                t_acc = stats["t_correct"] / self.rotate_num
                
                # 核心筛选逻辑：必须看图才能做对，且盲猜做不对
                if v_acc >= self.pass_visual_min and t_acc <= self.pass_textual_max:
                    qa_item["stats"] = {"v_acc": v_acc, "t_acc": t_acc}
                    filtered_results[row_idx].append(qa_item)

        df[output_key] = filtered_results
        output_file = storage.write(df)
        self.logger.info(f"Refinement complete. Results saved to {output_file}")
        
        return [output_key]


# ==========================================
# 测试用例 (Main Block)
# ==========================================
if __name__ == "__main__":
    
    # API 模式
    model = APIVLMServing_openai(
        api_url="http://172.96.141.132:3001/v1",
        key_name_of_api_key="DF_API_KEY",
        model_name="gpt-5-nano-2025-08-07",
        image_io=None,
        send_request_stream=False,
        max_workers=10,
        timeout=1800
    )

    # 本地模型
    # model = LocalModelVLMServing_vllm(
    #     hf_model_name_or_path="Qwen/Qwen2.5-VL-3B-Instruct",
    #     vllm_tensor_parallel_size=1,
    #     vllm_temperature=0.7,
    #     vllm_top_p=0.9,
    #     vllm_max_tokens=512,
    # )

    # 模板需与用户的占位符匹配 (这里通过 .format(q_v) 直接传参，所以用 {0})
    refiner = VisualDependencyRefiner(
        serving=model,
        instruction_template="Please answer the following multiple-choice question.\n{0}",
        rotate_num=4,
        pass_visual_min=0.75,   # 有图准确率需 >= 75%
        pass_textual_max=0.30,  # 无图瞎猜准确率需 <= 30%
        add_none_above_visual=True
    )

    storage = FileStorage(
        first_entry_file_name="./dataflow/example/image_to_text_pipeline/dependency_sample.jsonl", 
        cache_path="./cache_dependency",
        file_name_prefix="visual_dependency",
        cache_type="jsonl",
    )
    storage.step()

    refiner.run(
        storage=storage,
        input_list_key="mcq_candidates", 
        input_image_key="image",
        output_key="high_quality_mcqs"
    )