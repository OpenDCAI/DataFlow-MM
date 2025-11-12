import os
import ast
import json
import base64
import random
import re
from io import BytesIO
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

from dataflow import get_logger
from dataflow.core import OperatorABC, LLMServingABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY
from qwen_vl_utils import process_vision_info

try:
    import wandb
except Exception:
    wandb = None

def _process_text_chain(chain: List[str]) -> Tuple[str, str]:
    if chain and (chain[0].startswith("<image>") or chain[0].endswith("<image>")):
        chain = chain[1:]
    final_answer = chain[-1].replace("<answer>", "").replace("</answer>", "").strip()
    chain = chain[:-1]
    cleaned = []
    for line in chain:
        line = line.replace("<think>", "").replace("</think>", "")
        line = line.replace("<answer>", "").replace("</answer>", "")
        cleaned.append(line.strip())
    joined_chain = " ".join(cleaned)
    return joined_chain, final_answer

def _build_reasoning_chains_from_rollouts(
    node_data: Dict[str, Any],
    backtrack_message: str = "Wait, this seems off. Let's try something else.",
    thought_start_tag: str = "<think>",
    thought_end_tag: str = "</think>",
    answer_start_tag: str = "<answer>",
    answer_end_tag: str = "</answer>",
) -> List[str]:
    rollouts = node_data.get("rollouts", [])
    correct_rollouts, wrong_rollouts = [], []
    for r in rollouts:
        (correct_rollouts if r.get("reward", 0.0) >= 1.0 else wrong_rollouts).append(r)
    child_nodes = node_data.get("children", [])
    is_terminal = node_data.get("is_terminal", False)
    all_chains = []
    for wr in wrong_rollouts:
        wc, _ = _process_text_chain(wr["ephemeral_texts"])
        wc += f"\n{backtrack_message}"
        for cr in correct_rollouts:
            cc, ca = _process_text_chain(cr["ephemeral_texts"])
            c = f"{thought_start_tag}\n{wc}\n{cc}\n{thought_end_tag}\n{answer_start_tag} {ca} {answer_end_tag}"
            all_chains.append(c)
    for cr in correct_rollouts:
        cc, ca = _process_text_chain(cr["ephemeral_texts"])
        c = f"{thought_start_tag}\n{cc}\n{thought_end_tag}\n{answer_start_tag} {ca} {answer_end_tag}"
        all_chains.append(c)
    if not is_terminal:
        for child in child_nodes:
            all_chains.extend(_build_reasoning_chains_from_rollouts(child, backtrack_message))
    return all_chains

def _build_all_reasoning_for_sample(sample_json: Dict[str, Any]) -> Tuple[List[str], float]:
    root_node = sample_json["tree"]
    root_node_value = root_node.get("value", 0.0)
    chains = _build_reasoning_chains_from_rollouts(root_node)
    chains = list(set(chains))
    return chains, root_node_value

def _extract_think_points(chain_text: str):
    think_blocks = re.findall(r"<think>(.*?)</think>", chain_text, flags=re.DOTALL)
    pts, idx = [], 1
    for block in think_blocks:
        coords = re.findall(r"\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)", block)
        for xs, ys in coords:
            try:
                x, y = float(xs), float(ys)
                pts.append((x, y, idx))
                idx += 1
            except Exception:
                pass
    return pts

_SYSTEM_PROMPTS = {
    "web_grounding": (
        "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. "
        "The Assistant systematically reasons through the problem step by step, verifying each step and grounding every step to a specific point in the image.\n\n"
        "All reasoning processes must be enclosed within a single set of '<think>' tags, with each reasoning step explicitly referencing a coordinate:\n\n"
        "<think>\n[Reasoning text with grounded points inline] (x1, y1). [Further reasoning] (x2, y2), [Final refinement] (x3, y3).\n</think>\n\n"
        "The final answer should be enclosed in '<answer>' tags in the format:\n<answer> (xf, yf) </answer>\n\n"
        "Your task is to help the user identify the precise coordinates (x, y) of a specific area/element/object on the screen based on a description.\n"
        "- Aim to point to the center or a representative point within the described area/element/object as accurately as possible.\n"
        "- If the description is unclear or ambiguous, infer the most relevant area or element based on its likely context or purpose.\n"
        "- The final output should be the single most precise coordinate for the requested element.\n"
        "- The Assistant should verify each step and check multiple possible solutions before selecting the final answer."
    ),
    "spatial": (
        "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. "
        "The Assistant systematically reasons through the problem step by step by checking and verifying possible solutions and image regions, "
        "while grounding reasoning steps to specific objects and their relationships in the image using (x,y) coordinates. "
        "There may be one image or two images concatenated together.\n\n"
        "All reasoning processes must be enclosed within a single set of '<think>' tags.\n\n"
        "The final answer should be enclosed in '<answer>' tags in the format:\n<answer> {text of selected answer choice} </answer>\n"
        "- Your answer should be the exact text of the selected option."
    ),
    "web_action": (
        "You are a helpful Assistant tasked with navigating a web browser. "
        "Each reasoning step must be enclosed within '<think>' tags and reference exactly one specific coordinate (x, y). "
        "When ready, provide exactly one final action in <answer>...</answer>."
    ),
    "vstar": (
        "You are an assistant answering a visual question by reasoning through image regions. "
        "All reasoning in one <think>...</think>; final answer in <answer>...</answer>."
    ),
}

@OPERATOR_REGISTRY.register()
class VisionMCTSReasoningSFTGenerate(OperatorABC):
    def __init__(
        self,
        llm_serving: LLMServingABC,
        prompt_type: str = "web_grounding",
        val_size: float = 0.05,
        log_to_wandb: bool = False,
        max_samples_per_file: int = 10000,
        draw_points: bool = True,
        seed: int = 42,
    ):
        self.logger = get_logger()
        self.llm_serving = llm_serving
        self.prompt_type = prompt_type
        self.val_size = val_size
        self.log_to_wandb = log_to_wandb
        self.max_samples_per_file = max_samples_per_file
        self.draw_points = draw_points
        random.seed(seed)

    @staticmethod
    def get_desc(lang: str = "zh") -> str:
        zh = (
            "从 MCTS 搜索树或直接调用 VLM，生成带 <think>/<answer> 且内含坐标落点的推理链，用于后续训练（如 SFT / RL 热身）。生成的推理链统一包含单段 <think>…</think> 与单段 <answer>…</answer>，并在思维过程中显式引用一个或多个 (x, y)。\n"
            "背景论文 arXiv:2505.23678  https://arxiv.org/abs/2505.23678\n"
            "典型场景 Web Grounding / Spatial Reasoning / Web Action / Visual Search 等需要“显式坐标落点”的推理数据构建。\n"
            "系统提示模板（prompt_type） \n"
            "  · web_grounding：屏幕/GUI元素定位；最终答案为坐标 (x, y)。\n"
            "  · spatial：空间推理/多图对比；最终答案为选项文本或指向性描述。\n"
            "  · web_action：网页动作预测；最终答案为一次动作（含类型与参数）。\n"
            "  · vstar：通用视觉问答的坐标化思维链形式。\n"
            "工作机制\n"
            "  · 若样本含 MCTS 树：解析各 rollout，将“错误→回溯→正确”的路径线性化为多样式推理链；自动去重并可限量采样。\n"
            "  · 若无 MCTS 树：以所选系统提示直接调用 VLM 生成带 <think>/<answer> 的链。\n"
            "  · 可选可视化：解析链中 (x, y) 并在原图上叠加标注（绿色=思维步骤，红色=预测，蓝色=参考）。\n"
            "  · 可选日志：将若干样例以 HTML 方式发送至 W&B，便于快速质检。\n"
            "输入参数\n"
            "  · prompt_type: {'web_grounding','spatial','web_action','vstar'}\n"
            "  · val_size: 验证比例，默认 0.05\n"
            "  · log_to_wandb: 是否记录到 Weights & Biases\n"
            "  · max_samples_per_file: 单样本最多保留的推理链数量\n"
            "  · draw_points: 是否在图像上绘制坐标点\n"
            "  · seed: 随机种子\n"
        )

        en = (
            "Build coordinate-grounded chains-of-thought (<think>/<answer>) from MCTS trees or via direct VLM prompting—ready for training stages (e.g., SFT / RL warm-up). Produced chains contain a single <think>…</think> and a single <answer>…</answer>, with explicit (x, y) references in the reasoning.\n"
            "Paper arXiv:2505.23678  https://arxiv.org/abs/2505.23678\n"
            "Use Cases Web grounding, spatial reasoning, web action prediction, and visual search where explicit (x, y) grounding is required.\n"
            "System Prompts (prompt_type)\n"
            "  · web_grounding: screen/GUI localization; final answer is a coordinate (x, y).\n"
            "  · spatial: spatial reasoning / multi-image; final answer is an option text or pointer.\n"
            "  · web_action: next web action; final answer is one action with type & args.\n"
            "  · vstar: generic VQA with coordinate-grounded CoT format.\n"
            "Behavior\n"
            "  · With MCTS: parse rollouts, compose corrective→direct chains, deduplicate, optionally subsample.\n"
            "  · Without MCTS: prompt the VLM with the selected system prompt to produce <think>/<answer> chains.\n"
            "  · Optional visualization: overlay (x, y) steps on the image (green=steps, red=pred, blue=ref).\n"
            "  · Optional logging: HTML previews to Weights & Biases for quick QC.\n"
            "Input Args\n"
            "  · prompt_type ∈ {'web_grounding','spatial','web_action','vstar'}\n"
            "  · val_size (default 0.05), log_to_wandb, max_samples_per_file, draw_points, seed\n"
        )

        return zh if lang.lower().startswith("zh") else en

    def _choose_system_prompt(self) -> str:
        if self.prompt_type not in _SYSTEM_PROMPTS:
            raise ValueError(f"Invalid prompt_type: {self.prompt_type}")
        return _SYSTEM_PROMPTS[self.prompt_type]

    def _gen_with_llm(self, question: str, image_path: Optional[str]) -> str:
        sys_prompt = self._choose_system_prompt()
        content = []
        if image_path:
            content.append({"type":"image","image":image_path})
        content.append({"type":"text","text":question})
        raw_prompt = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": content},
        ]
        image_inputs, _ = process_vision_info(raw_prompt)
        prompt = self.llm_serving.processor.apply_chat_template(
            raw_prompt, tokenize=False, add_generation_prompt=True
        )
        outputs = self.llm_serving.generate_from_input(
            user_inputs=[prompt],
            image_inputs=[image_inputs]
        )
        return outputs[0] if isinstance(outputs, list) else outputs

    def _to_sft_entry(self, chain_text: str, question: str, image: Optional[str], gt_answer: Optional[str], idx: str):
        return {
            "id": idx,
            "metadata": {},
            "messages": [
                {"role":"system","content": self._choose_system_prompt()},
                {"role":"user","content": question},
                {"role":"assistant","content": chain_text}
            ],
            "images": [image] if image else [],
            "gt_answer": gt_answer if gt_answer is not None else "",
        }

    def _maybe_draw_overlay(self, image_path: str, chain_text: str, true_answer: str) -> Optional[str]:
        if not (self.draw_points and image_path and os.path.exists(image_path)):
            return None
        try:
            with Image.open(image_path) as im:
                draw = ImageDraw.Draw(im)
                r = 7
                for (x,y,i) in _extract_think_points(chain_text):
                    draw.ellipse((x-r,y-r,x+r,y+r), fill="green")
                    draw.text((x+15,y), str(i), fill="green")
                try:
                    fa = chain_text.split("<answer>")[1].split("</answer>")[0].strip()
                    pa = ast.literal_eval(fa)
                    if isinstance(pa,(list,tuple)) and len(pa)==2:
                        px,py=float(pa[0]),float(pa[1])
                        draw.ellipse((px-r,py-r,px+r,py+r), fill="red")
                        draw.text((px+15,py), "PRED", fill="red")
                except Exception:
                    pass
                try:
                    ga = ast.literal_eval(true_answer) if true_answer else None
                    if isinstance(ga,(list,tuple)) and len(ga)==2:
                        gx,gy=float(ga[0]),float(ga[1])
                        draw.ellipse((gx-r,gy-r,gx+r,gy+r), fill="blue")
                        draw.text((gx+15,gy), "GT", fill="blue")
                except Exception:
                    pass
                if im.mode!="RGB":
                    im=im.convert("RGB")
                buf=BytesIO()
                im.save(buf, format="PNG")
                enc=base64.b64encode(buf.getvalue()).decode("utf-8")
                return f'<img src="data:image/png;base64,{enc}" style="max-width:600px;" />'
        except Exception:
            return None

    def run(
        self,
        storage: DataFlowStorage,
        question_key: str = "question",
        image_key: str = "image",
        tree_key: Optional[str] = "tree",
        true_answer_key: str = "true_answer",
        output_key: str = "sft_entry",
    ):
        df = storage.read("dataframe")
        need_cols = [question_key]
        for c in need_cols:
            if c not in df.columns:
                raise ValueError(f"Missing required column: {c}")

        chains_txt_all = []
        sft_train, sft_val = [], []
        all_images = set()
        chain_global_id = 0
        correct_count = 0

        n = len(df)
        cut = int(n * (1.0 - self.val_size))
        train_idx_set = set(range(cut))

        for i, row in enumerate(tqdm(df.itertuples(), desc=f"Implementing {self.__class__.__name__}")):
            q = getattr(row, question_key, "")
            img = getattr(row, image_key, None) if image_key in df.columns else None
            gt = getattr(row, true_answer_key, "")
            chain_list: List[str] = []

            if tree_key and tree_key in df.columns and getattr(row, tree_key, None):
                try:
                    tree_obj = getattr(row, tree_key)
                    if isinstance(tree_obj, str):
                        tree_obj = json.loads(tree_obj)
                    sample_json = {"tree": tree_obj}
                    chain_list, root_val = _build_all_reasoning_for_sample(sample_json)
                    if root_val > 0:
                        correct_count += 1
                except Exception:
                    chain_list = []

            if not chain_list:
                try:
                    chain_text = self._gen_with_llm(q, img)
                    chain_list = [chain_text]
                except Exception as e:
                    chain_list = []

            if len(chain_list) > self.max_samples_per_file:
                chain_list = random.sample(chain_list, self.max_samples_per_file)

            sft_entries_cur = []
            for ctext in chain_list:
                idx = f"{i}_{chain_global_id}"
                entry = self._to_sft_entry(ctext, q, img, gt, idx)
                sft_entries_cur.append(entry)
                chains_txt_all.append(ctext)
                chain_global_id += 1

            if i in train_idx_set:
                sft_train.extend(sft_entries_cur)
            else:
                sft_val.extend(sft_entries_cur)

            df.at[i, output_key] = json.dumps(sft_entries_cur, ensure_ascii=False)
            if img:
                all_images.add(img)

        base_dir = os.path.dirname(storage.first_entry_file_name) if hasattr(storage, "first_entry_file_name") else "."
        out_dir = os.path.join(base_dir, "reasoning_chains")
        os.makedirs(out_dir, exist_ok=True)

        txt_path = os.path.join(out_dir, "reasoning_chains.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            sep = "\n\n----------------------\n\n----------------------\n\n"
            for ch in chains_txt_all:
                f.write(ch + sep)

        train_json_path = os.path.join(out_dir, "reasoning_chains_train.json")
        val_json_path = os.path.join(out_dir, "reasoning_chains_val.json")
        with open(train_json_path, "w", encoding="utf-8") as f:
            json.dump(sft_train, f, indent=2, ensure_ascii=False)
        with open(val_json_path, "w", encoding="utf-8") as f:
            json.dump(sft_val, f, indent=2, ensure_ascii=False)

        if self.log_to_wandb and wandb is not None and (sft_train or sft_val):
            wandb.init(project="vlm-search", name="mcts_reasoning_chains_sft")
            sft_entries = sft_train + sft_val
            max_samples = min(50, len(sft_entries))
            rnd_idx = random.sample(range(len(sft_entries)), max_samples)
            html = "<html><body>"
            for idx in rnd_idx:
                e = sft_entries[idx]
                q = e["messages"][1]["content"]
                ctext = e["messages"][2]["content"]
                img_path = e["images"][0] if e.get("images") else ""
                gt = e.get("gt_answer", "")
                overlay = self._maybe_draw_overlay(img_path, ctext, gt) if img_path else ""
                esc = ctext.replace("<","&lt;").replace(">","&gt;")
                html += f'<div style="border:1px solid #ddd; padding:10px; margin:10px 0;">'
                html += f"<p><b>Question:</b> {q}</p>{overlay}<p><b>Chain:</b> {esc}</p></div>"
            html += "</body></html>"
            wandb.log({"mcts_reasoning_chains": wandb.Html(html)})
            wandb.finish()

        storage.write(df)
        self.logger.info(f"Train SFT: {train_json_path} ({len(sft_train)}) | Val SFT: {val_json_path} ({len(sft_val)}) | Chains TXT: {txt_path}")
        return [output_key]
