# -*- coding: utf-8 -*-
# CapRL-style MCQ generation + filtering operator for DataFlow
from __future__ import annotations

import os
import re
import json
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from dataflow.core import OperatorABC, VLMServingABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY

# -----------------------------
# Prompts
# -----------------------------
SYS_PROMPT_MCQ = (
    "Your task is to generate five multiple-choice questions and their answers about the object "
    "based on the provided image. The questions should be challenging and focus on the image content.\n"
    "You must strictly follow the format below and must not output irrelevant sentences:\n"
    "#### 1. **Example question?**\n"
    "   - A) Option A\n"
    "   - B) Option B\n"
    "   - C) Option C\n"
    "   - D) Option D\n\n"
    "**Answer:** D) Option D\n"
    "------\n"
    "#### 2. **Another example?**\n"
    "   - A) ...\n"
    "   - B) ...\n"
    "   - C) ...\n"
    "   - D) ...\n\n"
    "**Answer:** B) ...\n"
    "------\n"
    "All questions must be answerable from the image alone."
)
USER_PROMPT_MCQ = "Here is the image"

# 过滤阶段：只返回字母
ANSWER_LETTER_INSTRUCTION = "{}. Answer the question with only the correct letter"

# -----------------------------
# Config
# -----------------------------
@dataclass
class CapRLMCQConfig:
    # 生成与解析
    expected_mcq_num: int = 5
    max_mcq_tokens: int = 2048  
    # 旋转与双重验证
    rotate_num: int = 4
    add_none_above_for_visual: bool = True  # 可视问答时在选项末尾加 “E) None of the above”
    # 通过阈值（和官方逻辑等价/更稳健）：可视准确率需高、无图像准确率需低
    pass_visual_min: float = 1.0    # 1.0 表示所有旋转都答对
    pass_textual_max: float = 0.0   # 0.0 表示所有旋转都答错
    # I/O
    input_jsonl_path: Optional[str] = None   # 非 DataFrame 模式
    output_jsonl_path: Optional[str] = None  # 非 DataFrame 模式
    # 抽样/去重
    dedup_questions: bool = True


# -----------------------------
# 工具函数
# -----------------------------
_Q_BLOCK_SPLIT = re.compile(r"^####\s*\d+\.\s*\*\*(.*?)\*\*\s*$", re.M)
_OPT_LINE_RE = re.compile(r"^\s*-\s*([A-F])\)\s*(.+?)\s*$")
_ANS_LINE_RE = re.compile(r"^\s*\*\*Answer:\*\*\s*([A-F])\)\s*(.+?)\s*$", re.I)

def _split_blocks(mcq_text: str) -> List[str]:
    """把整段 MCQ 文本按 '#### N. **...' 分成题块（包含题干、选项、答案）。"""
    indices = [m.start() for m in _Q_BLOCK_SPLIT.finditer(mcq_text)]
    if not indices:
        return []
    indices.append(len(mcq_text))
    blocks = [mcq_text[indices[i]:indices[i+1]].strip() for i in range(len(indices)-1)]
    return [b for b in blocks if b]

def _parse_one_block(block: str) -> Optional[Dict[str, Any]]:
    """
    解析单题块为：
      {
        "question": "Q + options (带换行)",
        "options": {"A": "...", "B": "...", ...},
        "answer": "A"/"B"/...（字母）,
        "answer_text": "option text"
      }
    """
    lines = [ln.rstrip() for ln in block.splitlines() if ln.strip()]
    # 题干行（#### N. **...**）
    q_title_m = _Q_BLOCK_SPLIT.search(block)
    if not q_title_m:
        return None
    q_title = q_title_m.group(1).strip()

    # 收集合规的选项
    options: Dict[str, str] = {}
    for ln in lines:
        m = _OPT_LINE_RE.match(ln)
        if m:
            options[m.group(1)] = m.group(2).strip()

    # 答案行
    ans_letter, ans_text = None, None
    for ln in lines:
        m = _ANS_LINE_RE.match(ln)
        if m:
            ans_letter = m.group(1).upper()
            ans_text = m.group(2).strip()
            break

    # 基本校验
    if not options or ans_letter is None or ans_letter not in options:
        return None

    # 组装 “问题+选项” 文本（用于后续发送给模型作答）
    q_lines = [f"{q_title}"]
    # 保持 A..F 顺序
    for lbl in ["A", "B", "C", "D", "E", "F"]:
        if lbl in options:
            q_lines.append(f"   - {lbl}) {options[lbl]}")
    q_text = "\n".join(q_lines)

    return {
        "question": q_text,
        "options": options,
        "answer": ans_letter,
        "answer_text": ans_text,
    }

def _parse_mcq_text(mcq_text: str, expected: int = 5) -> List[Dict[str, Any]]:
    blocks = _split_blocks(mcq_text)
    parsed = []
    for b in blocks:
        item = _parse_one_block(b)
        if item:
            parsed.append(item)
    # 截断/保留预期数
    if expected > 0:
        parsed = parsed[:expected]
    # 去重（按题面+答案）
    uniq = []
    seen = set()
    for it in parsed:
        key = (it["question"], it["answer"])
        if key not in seen:
            seen.add(key)
            uniq.append(it)
    return uniq

def _shuffle_options(question_with_opts: str, correct_letter: str) -> Tuple[str, str]:
    """
    随机打乱选项标签，并返回 (新问题文本, 新答案字母)。
    question_with_opts 的第二行起应为 '   - X) text' 格式。
    """
    lines = question_with_opts.splitlines()
    if not lines:
        return question_with_opts, correct_letter
    q_text = lines[0]
    opt_lines = [ln for ln in lines[1:] if ln.strip()]
    # 抽取原选项
    original = []
    correct_text = None
    for ln in opt_lines:
        m = _OPT_LINE_RE.match(ln)
        if m:
            lbl, txt = m.group(1), m.group(2).strip()
            original.append((lbl, txt))
            if lbl == correct_letter:
                correct_text = txt
    if not original or correct_text is None:
        return question_with_opts, correct_letter

    random.shuffle(original)
    new_labels = ["A", "B", "C", "D", "E", "F"]
    reassigned = []
    new_answer = None
    for i, (_lbl, txt) in enumerate(original):
        lbl = new_labels[i]
        reassigned.append((lbl, txt))
        if txt == correct_text:
            new_answer = lbl

    out_lines = [q_text]
    for lbl, txt in reassigned:
        out_lines.append(f"   - {lbl}) {txt}")
    return "\n".join(out_lines), (new_answer or correct_letter)

def _extract_letter_only(model_out: str) -> Optional[str]:
    """尽量稳健地从模型输出里取 A-F 的单字母（允许大小写）。"""
    if not model_out:
        return None
    m = re.search(r"\b([A-Fa-f])\b", model_out)
    if m:
        return m.group(1).upper()
    # 兜底：常见形式 like "Answer: C"
    m2 = re.search(r"[Aa]nswer\s*[:：]\s*([A-Fa-f])", model_out)
    if m2:
        return m2.group(1).upper()
    return None


# -----------------------------
# 算子
# -----------------------------
@OPERATOR_REGISTRY.register()
class CapRLMCQGenerate(OperatorABC):
    """
    从 {"image": "..."} 构建用于 CapRL 的 MCQ 数据，并执行可视/文本双重过滤：
      1) 由 VLM 生成 5 道 MCQ；
      2) 每题做 N 次“旋转（选项随机重排）”；
      3) 可视条件：给图像 + “E) None of the above”（可选）；
         文本条件：不带图像（仅题面），都要求只回单个字母；
      4) 通过条件：可视答对率 >= pass_visual_min 且 无图像答对率 <= pass_textual_max；
    输出写回 DataFrame 指定列，或另存为 jsonl。
    """

    def __init__(self, vlm_serving: VLMServingABC, config: Optional[CapRLMCQConfig] = None):
        self.serving = vlm_serving
        self.cfg = config or CapRLMCQConfig()

    @staticmethod
    def get_desc(lang: str = "zh") -> str:
        zh = (
            "从 {image} jsonl 生成 MCQ 并做“有图/无图”双重过滤，确保问题需依赖图像才能作答；"
            "输出结构写入 cap_rl_qa 字段，可直接用于后续 RL 的可验证奖励。"
        )
        en = (
            "Generate MCQs from images and apply dual filtering (with/without image) to ensure visual dependency; "
            "the resulting cap_rl_qa record is ready for verifiable-reward RL."
        )
        return zh if str(lang).lower().startswith("zh") else en

    # ---------- 调用 VLM ----------
    def _gen_mcq_raw(self, image_path: str) -> str:
        """用系统提示 + <image> 让 VLM 生成 5 道 MCQ 的原始文本。"""
        conversations = [[{"from": "human", "value": f"<image>\n{USER_PROMPT_MCQ}"}]]
        outs = self.serving.generate_from_input_messages(
            conversations=conversations,
            image_list=[[image_path]],
            system_prompt=SYS_PROMPT_MCQ
        )
        return outs[0] if outs else ""

    def _ask_letter_with_image(self, prompt_text: str, image_path: str) -> str:
        conversations = [[{"from": "human", "value": f"<image>\n{prompt_text}"}]]
        outs = self.serving.generate_from_input_messages(
            conversations=conversations,
            image_list=[[image_path]],
        )
        return outs[0] if outs else ""

    def _ask_letter_text_only(self, prompt_text: str) -> str:
        conversations = [[{"from": "human", "value": prompt_text}]]
        outs = self.serving.generate_from_input_messages(conversations=conversations)
        return outs[0] if outs else ""

    # ---------- 主流程 ----------
    def _filter_one_qa(self, qa: Dict[str, Any], image_path: str) -> Dict[str, Any]:
        """
        对单题做多次旋转与双重验证，返回统计与是否保留。
        """
        rotate_num = max(1, self.cfg.rotate_num)
        v_correct, l_correct = 0, 0
        trials: List[Dict[str, Any]] = []

        for _ in range(rotate_num):
            q_rot, ans_letter = _shuffle_options(qa["question"], qa["answer"])

            # 可视：可选添加 “E) None of the above”
            q_vis = q_rot
            if self.cfg.add_none_above_for_visual and not re.search(r"^\s*-\s*E\)", q_rot, re.M):
                q_vis = q_vis + "\n   - E) None of the above"

            vis_prompt = ANSWER_LETTER_INSTRUCTION.format(q_vis)
            txt_prompt = ANSWER_LETTER_INSTRUCTION.format(q_rot)

            v_out = self._ask_letter_with_image(vis_prompt, image_path)
            l_out = self._ask_letter_text_only(txt_prompt)

            v_pred = _extract_letter_only(v_out)
            l_pred = _extract_letter_only(l_out)

            v_ok = (v_pred == ans_letter)
            l_ok = (l_pred == ans_letter)

            v_correct += 1 if v_ok else 0
            l_correct += 1 if l_ok else 0

            trials.append({
                "rotated_answer": ans_letter,
                "visual_output": v_out,
                "visual_pred": v_pred,
                "visual_correct": bool(v_ok),
                "text_output": l_out,
                "text_pred": l_pred,
                "text_correct": bool(l_ok),
            })

        v_acc = v_correct / rotate_num
        l_acc = l_correct / rotate_num
        keep = (v_acc >= self.cfg.pass_visual_min) and (l_acc <= self.cfg.pass_textual_max)

        return {
            "qa": qa,
            "trials": trials,
            "visual_acc": v_acc,
            "text_acc": l_acc,
            "keep": keep,
        }

    def _dedup_mcqs(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self.cfg.dedup_questions:
            return items
        out, seen = [], set()
        for it in items:
            key = (it["question"].strip(), it["answer"])
            if key in seen:
                continue
            seen.add(key)
            out.append(it)
        return out

    # ---------- 入口 ----------
    def run(
        self,
        storage: DataFlowStorage,
        image_key: str = "image",
        output_key: str = "cap_rl_qa",
    ):
        """
        输入：
          1) DataFrame：storage.read('dataframe') 含 image 列
          2) JSONL：config.input_jsonl_path，每行 {"image": "..."}
        输出：
          - DataFrame：写入列 `output_key`（json.dumps(record)）
          - JSONL：写入 config.output_jsonl_path（默认 *.caprl_mcq.jsonl）
        """
        # 读取数据
        use_df = False
        df = None
        try:
            df = storage.read("dataframe")
            use_df = image_key in df.columns
        except Exception:
            use_df = False

        if use_df:
            rows = [{image_key: v} for v in df[image_key].tolist()]
        else:
            if not self.cfg.input_jsonl_path:
                raise ValueError("No input found. Provide DataFrame[image] or config.input_jsonl_path.")
            with open(self.cfg.input_jsonl_path, "r", encoding="utf-8") as f:
                rows = [json.loads(ln) for ln in f if ln.strip()]

        outputs: List[Dict[str, Any]] = []

        for i, row in enumerate(rows):
            image_path = row.get(image_key, "")
            if not image_path:
                continue

            # 1) 生成 MCQ 原文并解析
            raw_text = self._gen_mcq_raw(image_path)
            qa_list = _parse_mcq_text(raw_text, expected=self.cfg.expected_mcq_num)
            qa_list = self._dedup_mcqs(qa_list)

            # 2) 对每题做旋转与双重过滤
            per_qa_stats = [self._filter_one_qa(qa, image_path) for qa in qa_list]

            # 3) 通过题集合
            kept = [s for s in per_qa_stats if s["keep"]]
            kept_qas = [s["qa"] for s in kept]

            # 4) 汇总记录（用于 RL 的“可验证奖励”后续阶段）
            rec = {
                "image": image_path,
                "raw_mcq_text": raw_text,
                "parsed_qa_list": qa_list,          # 解析后的 5 题（含题干、选项、答案）
                "filter_stats": per_qa_stats,        # 每题旋转与双重验证的详细结果
                "kept_qas": kept_qas,                # 通过过滤的题
                "num_kept": len(kept_qas),
                "num_all": len(qa_list),
                "config": {
                    "rotate_num": self.cfg.rotate_num,
                    "pass_visual_min": self.cfg.pass_visual_min,
                    "pass_textual_max": self.cfg.pass_textual_max,
                    "add_none_above_for_visual": self.cfg.add_none_above_for_visual,
                }
            }

            outputs.append({"image": image_path, output_key: rec})
            if use_df:
                df.at[i, output_key] = rec

        # 落盘
        if use_df:
            storage.write(df)
        else:
            out_path = self.cfg.output_jsonl_path or os.path.splitext(self.cfg.input_jsonl_path)[0] + ".caprl_mcq.jsonl"
            self.cfg.output_jsonl_path = out_path
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                for r in outputs:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

        return [output_key]
