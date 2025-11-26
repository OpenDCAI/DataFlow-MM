# -*- coding: utf-8 -*-
# ScaleCap-style dense caption generation operator (correct message schema)
from __future__ import annotations

import os
import re
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from dataflow.core import OperatorABC, VLMServingABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY

from tqdm import tqdm


VLM_PROMPT_1 = (
    "Describe the fine-grained content of the image, including scenes, objects, "
    "relationships, instance location, and any text present."
)

LLM_PROMPT_1 = '''Your task is to convert each Object mentioned in a given sentence into a corresponding instruction, and all the resulting instructions are output as "Describe more details about the [Object]". Ensure your instructions do not cover the raw question, options, or thought process of answering the instructions. You should ignore the Objects that appear in some inferences, such as the sentences that begins with 'it might be' or 'there are probably'.
Sentence: 
The image depicts a man in a suit and tie jumping in the air above a bed in a bedroom
Instructions:
Describe more details about the man.
Describe more details about the suit.
Describe more details about the tie.
Describe more details about the bed.
Describe more details about the bedroom.

Sentence:
The train appears to be the main subject of the image, showcasing its sleek design and modern appearance
Instructions:
Describe more details about the train.

Sentence:
The table has a few other items on it, including a camera, a jar of jam, and a spoon, suggesting that there might be some people ready to eat
Instructions:
Describe more details about the table.
Describe more details about the camera.
Describe more details about the jam.
Describe more details about the spoon.

Sentence:
The text "You see the world as you are!" is a playful and thought-provoking statement, encouraging viewers to appreciate their unique qualities and perspectives
Instructions:
Describe more details about the text.

Sentence:
1. **Preheat the Oven**: Preheat your oven to 350\u00b0F (175\u00bC).
Instructions:
Describe more details about the oven.
Describe more details about the preheat temperature.

Sentence:
{}
Instructions:
'''

LLM_PROMPT_2 = '''Descriptions:
{}

Collect all details about each object from the descriptions, including detailed appearance, structure, material, and special marks or logos. Do not include any analysis or your opinions.'''

LLM_PROMPT_3 = '''Descriptions:
{}

Extract and abstract only the position information about each object from the decriptions. Do not include any analysis or your opinions.'''

LLM_PROMPT_4 = '''Basic Context:
{}

Object Information:
{}

Position Information:
{}

Following the logic of the above Basic Context, organize all details provided in Object Information and Position Information to give a very comprehensive description about the image. Do not include any analysis or your opinions.'''


# -----------------------------
# Helpers
# -----------------------------
_SENT_SPLIT = re.compile(r"(?<=[.!?。！？])\s+")

def split_into_sentences(text: str) -> List[str]:
    parts = [p.strip() for p in _SENT_SPLIT.split(text) if p.strip()]
    return parts or ([text.strip()] if text.strip() else [])

def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

def _write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def _uniq(xs: List[str]) -> List[str]:
    seen, out = set(), []
    for x in xs:
        if x not in seen:
            out.append(x); seen.add(x)
    return out


# -----------------------------
# Config
# -----------------------------
@dataclass
class ImageScaleCaptionGenerateConfig:
    tau_sentence: float = 0.15          # 预留给严格“对比打分”版本；当前自检版不直接使用
    max_questions: int = 20
    max_init_tokens: int = 1024         # 由 serving 的 max_tokens 控制，这里仅留作注释/记录
    max_answer_tokens: int = 256        # 同上
    second_filter: bool = False         # 对回答再做一轮 yes/no 自检
    input_jsonl_path: Optional[str] = None
    output_jsonl_path: Optional[str] = None


# -----------------------------
# Operator（使用正确的 conversation schema）
# -----------------------------
@OPERATOR_REGISTRY.register()
class ImageScaleCaptionGenerate(OperatorABC):
    """
    从 {"image": "..."} jsonl 生成 ScaleCap 风格的长描述：
      初稿 → 句级自检（goldens） → 对象/位置追问 → 回答（可选二次过滤） → 整合 final_caption
    仅使用 VLMServingABC.generate_from_input_messages，
    且对话格式为：{"from": "human"/"assistant"/"system", "value": "..."}；
    带图时在 human 的 value 中包含 "<image>" 占位并通过 image_list 传图。
    """

    def __init__(self, vlm_serving: VLMServingABC, config: Optional[ImageScaleCaptionGenerateConfig] = None):
        self.serving = vlm_serving
        self.cfg = config or ImageScaleCaptionGenerateConfig()

    @staticmethod
    def get_desc(lang: str = "zh") -> str:
        if str(lang).lower().startswith("zh"):
            return (
                "  从包含 {\"image\": \"/path/to/img\"} 的 JSONL 或 DataFrame 列批量生成高信息密度的图像长描述。\n"
                "  流程为：初稿 caption → 句级自检筛选（golden sentences）→ 对象/位置追问 → 回答（可选二次过滤）→ 整合为 final_caption。\n"
                "\n"
                "背景论文：\n"
                "  arXiv:2506.19848  https://arxiv.org/abs/2506.19848\n"
                "\n"
                "输入参数（使用方式与说明）：\n"
                "  • Config（ScaleCapSingleConfig）：\n"
                "    - tau_sentence: float = 0.15\n"
                "        句级筛选阈值（预留给严格对比打分版本；当前自检版不直接使用）。\n"
                "    - max_questions: int = 20\n"
                "        每张图基于 golden sentences 生成的对象/位置追问总预算。\n"
                "    - max_init_tokens: int = 1024\n"
                "        初稿 caption 的期望最大长度（由 serving 的采样参数实际控制，这里作为约定与注释）。\n"
                "    - max_answer_tokens: int = 256\n"
                "        每条追问回答的期望最大长度（同上，由 serving 控制）。\n"
                "    - second_filter: bool = False\n"
                "        是否对追问回答再做一次句级自检（yes/no）以过滤泛化/幻觉。\n"
                "    - input_jsonl_path: Optional[str]\n"
                "        非 DataFrame 模式下，读取的输入 JSONL 路径（每行至少含 {\"image\": ...}）。\n"
                "    - output_jsonl_path: Optional[str]\n"
                "        非 DataFrame 模式下，落盘输出路径（默认写为 *.scalecap.jsonl）。\n"
                "  • run(storage, image_key=\"image\", output_key=\"scalecap_record\")：\n"
                "    - image_key：输入图像路径列名（或 JSONL 的字段名）。\n"
                "    - output_key：将生成记录以 JSON 字符串写回到该列；JSONL 模式下写文件。\n"
                "\n"
                "新增条目解释（输出结构，写入 output_key）：\n"
                "  {\n"
                "    \"image\": \"/path/to/img.jpg\",                     # 原始图像路径\n"
                "    \"init_caption\": \"...\",                          # 细粒度初稿描述\n"
                "    \"golden_sentences\": [\"...\", \"...\"],           # 句级自检保留下来的关键句\n"
                "    \"object_questions\": [\"Describe more details about the ...\", ...],\n"
                "    \"position_questions\": [\"Describe more details about the position of ...\", ...],\n"
                "    \"qa_answers_filtered\": [\"...\", \"...\"],         # 追问回答，经可选自检过滤与去重\n"
                "    \"final_caption\": \"...\"                           # 汇总融合后的最终长描述\n"
                "  }\n"
            )
        else:
            return (
                "  Build dense, faithful long captions for an image JSONL or a DataFrame column.\n"
                "  Pipeline: init caption → sentence self-check (goldens) → object/position QAs →\n"
                "  optional second filtering → final caption integration.\n"
                "\n"
                "Background paper:\n"
                "  arXiv:2506.19848  https://arxiv.org/abs/2506.19848\n"
                "\n"
                "Inputs & Usage:\n"
                "  • Config (ScaleCapSingleConfig):\n"
                "    - tau_sentence (float, 0.15): reserved for strict contrastive scoring; not used in the light self-check.\n"
                "    - max_questions (int, 20): total budget for object/position follow-up questions per image.\n"
                "    - max_init_tokens (int, 1024): expected max length of the initial caption (actual length controlled by serving).\n"
                "    - max_answer_tokens (int, 256): expected max length per answer (also controlled by serving).\n"
                "    - second_filter (bool, False): enable a second yes/no self-check on answers to filter generic/hallucinated text.\n"
                "    - input_jsonl_path (Optional[str]): input JSONL path when not using DataFrame (each line has at least {\"image\": ...}).\n"
                "    - output_jsonl_path (Optional[str]): output JSONL path in non-DataFrame mode (defaults to *.scalecap.jsonl).\n"
                "  • run(storage, image_key=\"image\", output_key=\"scalecap_record\"):\n"
                "    - image_key: field/column name of the image path.\n"
                "    - output_key: where the JSON record is written (as a string) or saved to file in JSONL mode.\n"
                "\n"
                "Produced record (written to output_key):\n"
                "  {\n"
                "    \"image\": \"/path/to/img.jpg\",\n"
                "    \"init_caption\": \"...\",\n"
                "    \"golden_sentences\": [\"...\", \"...\"],\n"
                "    \"object_questions\": [\"Describe more details about the ...\", ...],\n"
                "    \"position_questions\": [\"Describe more details about the position of ...\", ...],\n"
                "    \"qa_answers_filtered\": [\"...\", \"...\"],\n"
                "    \"final_caption\": \"...\"\n"
                "  }\n"
            )


    # ---------- Serving 调用（全部走 generate_from_input_messages） ----------
    def _gen_text(self, text: str) -> str:
        # 纯文本：human 一轮
        convs = [[{"from": "human", "value": text}]]
        outs = self.serving.generate_from_input_messages(conversations=convs)
        return outs[0] if outs else ""

    def _gen_with_image(self, prompt: str, image_path: str) -> str:
        # 带图：human 的 value 里放 <image> 占位，再跟上文本
        convs = [[{"from": "human", "value": f"<image>\n{prompt}"}]]
        outs = self.serving.generate_from_input_messages(
            conversations=convs,
            image_list=[[image_path]],
        )
        return outs[0] if outs else ""

    def _batch_yesno_with_image(self, sentences: List[str], image_path: str, template: str) -> List[str]:
        # 批量带图 yes/no 判断
        conversations, images = [], []
        for s in sentences:
            q = template.format(sentence=s)
            conversations.append([{"from": "human", "value": f"<image>\n{q}"}])
            images.append([image_path])
        outs = self.serving.generate_from_input_messages(
            conversations=conversations,
            image_list=images,
        )
        return [(o or "").strip().lower() for o in outs]

    # ---------- Pipeline 步骤 ----------
    def _gen_init_caption(self, image_path: str) -> str:
        return self._gen_with_image(VLM_PROMPT_1, image_path)

    def _pick_golden_sentences(self, image_path: str, init_caption: str) -> List[str]:
        """
        轻量自检版 goldens：对每个句子询问“是否直接由图像支持（yes/no）”，保留 yes。
        后续若切换到“对比打分（with/without image logits）”，替换本函数即可。
        """
        sents = split_into_sentences(init_caption) or [init_caption]
        tpl = ("Given the image and the following sentence:\n\n"
               "'{sentence}'\n\n"
               "Is this sentence directly supported by the visual evidence? "
               "Answer strictly with 'yes' or 'no'.")
        flags = self._batch_yesno_with_image(sents, image_path, tpl)
        goldens = [s for s, f in zip(sents, flags) if f.startswith("y")]
        return goldens if goldens else [init_caption]

    def _gen_instructions_per_sentence(self, sentence: str) -> List[str]:
        out = self._gen_text(LLM_PROMPT_1.format(sentence))
        # 规整为行，并去重，只保留“Describe more details about ...”模板
        out = out[: out.rfind(".") + 1] if "." in out else out
        lines = [t.strip() for t in out.split("\n") if t.strip()]
        uniq = _uniq([(t.split(".")[0] + ".") for t in lines])
        return [t for t in uniq if t.startswith("Describe more details about")]

    def _gen_questions(self, golden_sentences: List[str]) -> Tuple[List[str], List[str]]:
        obj_ins: List[str] = []
        for s in golden_sentences:
            obj_ins += self._gen_instructions_per_sentence(s)
        obj_ins = _uniq(obj_ins)[: self.cfg.max_questions]
        pos_ins = [
            "Describe more details about the position of" + ins.split("Describe more details about")[-1]
            for ins in obj_ins
        ]
        return obj_ins, pos_ins

    def _vlm_batch_answers(self, prompts: List[str], image_path: str) -> List[str]:
        if not prompts:
            return []
        conversations, images = [], []
        for p in prompts:
            conversations.append([{"from": "human", "value": f"<image>\n{p}"}])
            images.append([image_path])
        outs = self.serving.generate_from_input_messages(
            conversations=conversations,
            image_list=images,
        )
        return [(" ".join(o.split("\n")) if o else "") for o in outs]

    def _filter_answers_yesno(self, answers: List[str], image_path: str) -> List[str]:
        if not self.cfg.second_filter or not answers:
            return answers
        tpl = ("Given the image and the following answer text:\n\n"
               "'{sentence}'\n\n"
               "Is this answer grounded in the image (not generic)? Answer strictly 'yes' or 'no'.")
        flags = self._batch_yesno_with_image(answers, image_path, tpl)
        return [a for a, f in zip(answers, flags) if f.startswith("y")]

    def _integrate(self, goldens: List[str], obj_details: List[str], pos_details: List[str]) -> str:
        obj_caption = self._gen_text(LLM_PROMPT_2.format("\n".join(obj_details)))
        pos_caption = self._gen_text(LLM_PROMPT_3.format("\n".join(pos_details)))
        final_caption = self._gen_text(LLM_PROMPT_4.format("\n".join(goldens), obj_caption, pos_caption))
        return final_caption

    # ---------- 主入口 ----------
    def run(
        self,
        storage: DataFlowStorage,
        input_image_key: str = "image",
        output_key: str = "scalecap_record",
    ):
        """
        输入：
          1) DataFrame：storage.read('dataframe') 含 image 列
          2) JSONL：config.input_jsonl_path，每行 {"image": "..."}
        输出：
          - DataFrame：写入列 `output_key`（json.dumps(record)）
          - JSONL：写入 config.output_jsonl_path（默认 *.scalecap.jsonl）
        """
        # 读取
        use_df = False
        df = None
        try:
            df = storage.read("dataframe")
            use_df = image_key in df.columns
        except Exception:
            use_df = False

        rows = _read_jsonl(self.cfg.input_jsonl_path) if (not use_df and self.cfg.input_jsonl_path) \
               else ([{image_key: v} for v in df[image_key].tolist()] if use_df else [])
        if not rows and not use_df:
            raise ValueError("No input found. Provide DataFrame[image] or config.input_jsonl_path.")

        out_records: List[Dict[str, Any]] = []

        for i, row in enumerate(rows):
            print(f"Processing {i + 1}/{len(rows)} images...", end="\r")
            image_path = row.get(image_key, "")
            if not image_path:
                continue

            # 1) 初稿
            try:
                init_caption = self._gen_init_caption(image_path)

                # 2) goldens
                goldens = self._pick_golden_sentences(image_path, init_caption)

                # 3) 追问
                obj_qs, pos_qs = self._gen_questions(goldens)

                # 4) 回答 + 可选过滤
                obj_ans_raw = self._vlm_batch_answers(obj_qs, image_path)
                pos_ans_raw = self._vlm_batch_answers(pos_qs, image_path)
                obj_ans = self._filter_answers_yesno(obj_ans_raw, image_path)
                pos_ans = self._filter_answers_yesno(pos_ans_raw, image_path)

                qa_filtered = _uniq(obj_ans + pos_ans)

                # 5) 整合
                final_caption = self._integrate(goldens=goldens, obj_details=obj_ans, pos_details=pos_ans)

                record = {
                    "image": image_path,
                    "init_caption": init_caption,
                    "golden_sentences": goldens,
                    "object_questions": obj_qs,
                    "position_questions": pos_qs,
                    "qa_answers_filtered": qa_filtered,
                    "final_caption": final_caption,
                }
            except Exception as e:
                print(f"\n[Error] Failed processing image {image_path}: {e}")
                record = {
                    "image": image_path,
                    "init_caption": "error",
                    "golden_sentences": [],
                    "object_questions": [],
                    "position_questions": [],
                    "qa_answers_filtered": [],
                    "final_caption": "",
                }
            out_records.append(record)

            if use_df:
                df.at[i, output_key] = json.dumps(record, ensure_ascii=False)

        # 写回
        if use_df:
            storage.write(df)
        else:
            out_path = self.cfg.output_jsonl_path or os.path.splitext(self.cfg.input_jsonl_path)[0] + ".scalecap.jsonl"
            self.cfg.output_jsonl_path = out_path
            _write_jsonl(out_path, out_records)

        return [output_key]
