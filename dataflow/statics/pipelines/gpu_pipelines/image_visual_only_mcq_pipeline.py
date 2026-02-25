import argparse
from dataflow.utils.storage import FileStorage
from dataflow.serving.local_model_vlm_serving import LocalModelVLMServing_vllm

from dataflow.operators.core_vision import FixPromptedVQAGenerator, VisualDependencyRefiner
from dataflow.operators.core_text import FunctionalRefiner
from dataflow.prompts.image import ImageCaprlPrompt

import re
from typing import List, Dict, Any

_Q_BLOCK_SPLIT = re.compile(r"^####\s*\d+\.\s*\*\*(.*?)\*\*\s*$", re.M)
_OPT_LINE_RE = re.compile(r"^\s*-\s*([A-F])\)\s*(.+?)\s*$")
_ANS_LINE_RE = re.compile(r"^\s*\*\*Answer:\*\*\s*([A-F])\)\s*(.+?)\s*$", re.I)

def parse_mcq_text_logic(mcq_text: str, expected: int = 5) -> List[Dict[str, Any]]:
    if not mcq_text or not isinstance(mcq_text, str): return []
    
    indices = [m.start() for m in _Q_BLOCK_SPLIT.finditer(mcq_text)]
    if not indices: return []
    indices.append(len(mcq_text))
    blocks = [mcq_text[indices[i]:indices[i+1]].strip() for i in range(len(indices)-1)]
    
    parsed = []
    for block in blocks:
        lines = [ln.rstrip() for ln in block.splitlines() if ln.strip()]
        q_title_m = _Q_BLOCK_SPLIT.search(block)
        if not q_title_m: continue
        
        q_title = q_title_m.group(1).strip()
        options = {}
        ans_letter, ans_text = None, None
        
        for ln in lines:
            m_opt = _OPT_LINE_RE.match(ln)
            if m_opt:
                options[m_opt.group(1)] = m_opt.group(2).strip()
                continue
            m_ans = _ANS_LINE_RE.match(ln)
            if m_ans:
                ans_letter = m_ans.group(1).upper()
                ans_text = m_ans.group(2).strip()
                break
        
        if options and ans_letter and ans_letter in options:
            q_lines = [q_title]
            for lbl in ["A", "B", "C", "D", "E", "F"]:
                if lbl in options:
                    q_lines.append(f"   - {lbl}) {options[lbl]}")
            
            parsed.append({
                "question": "\n".join(q_lines),
                "question_title": q_title,
                "options": options,
                "answer": ans_letter,
                "answer_text": ans_text
            })
            
    if expected > 0:
        parsed = parsed[:expected]
        
    uniq = []
    seen = set()
    for it in parsed:
        key = (it["question_title"], it["answer"])
        if key not in seen:
            seen.add(key)
            uniq.append(it)
    return uniq


class VisualOnlyMCQPipeline:
    def __init__(
        self,
        model_path: str,
        *,
        first_entry_file: str,
        hf_cache_dir: str | None = None,
        download_dir: str = "./ckpt/models",
        cache_path: str = "../cache/cache_mcq",
        file_name_prefix: str = "vis_mcq",
        # Config
        rotate_num: int = 4,
        pass_visual_min: float = 1.0,
        pass_textual_max: float = 0.25,
        add_none_above: bool = True,
        # Keys
        input_image_key: str = "image",
        output_key: str = "final_mcqs",
        # VLLM
        device: str = "cuda",
        vllm_max_tokens: int = 2048
    ):
        self.storage = FileStorage(
            first_entry_file_name=first_entry_file,
            cache_path=cache_path,
            file_name_prefix=file_name_prefix,
            cache_type="jsonl"
        )
        
        self.serving = LocalModelVLMServing_vllm(
            hf_cache_dir=hf_cache_dir,
            hf_local_dir=download_dir,
            hf_model_name_or_path=model_path,
            vllm_tensor_parallel_size=1,
            vllm_temperature=0.1, 
            vllm_max_tokens=vllm_max_tokens
        )
        
        # Keys
        self.keys = {
            "img": input_image_key,
            "raw_text": "raw_mcq_text",
            "parsed_list": "parsed_mcq_list",
            "final": output_key
        }
        
        # --- Prompts ---
        self.prompts_db = ImageCaprlPrompt().build_prompt()

        # ================== Operators ==================
        
        # 1. Generate Raw MCQs (FixPromptedVQAGenerator)
        # 直接使用 prompt 类中的字符串
        self.op_gen_raw = FixPromptedVQAGenerator(
            serving=self.serving,
            system_prompt=self.prompts_db["SYS_PROMPT_MCQ"],
            user_prompt=self.prompts_db["USER_PROMPT_MCQ"]
        )
        
        # 2. Parse MCQs (Refine)
        self.op_parse = FunctionalRefiner(func=parse_mcq_text_logic)
        
        # 3. Verify Visual Dependency (Refine)
        # 传入 prompt 模板
        self.op_verify = VisualDependencyRefiner(
            serving=self.serving,
            instruction_template=self.prompts_db["ANSWER_INSTRUCTION"],
            rotate_num=rotate_num,
            pass_visual_min=pass_visual_min,
            pass_textual_max=pass_textual_max,
            add_none_above_visual=add_none_above
        )

    def forward(self):
        print(">>> [Pipeline] Step 1: Generating Raw MCQs (FixPrompted)...")
        self.op_gen_raw.run(
            self.storage.step(),
            input_image_key=self.keys["img"],
            output_answer_key=self.keys["raw_text"]
        )
        
        print(">>> [Pipeline] Step 2: Parsing MCQs...")
        self.op_parse.run(
            self.storage.step(),
            output_key=self.keys["parsed_list"],
            mcq_text=self.keys["raw_text"], 
            expected=5
        )
        
        print(">>> [Pipeline] Step 3: Verifying Visual Dependency (Rotation Check)...")
        self.op_verify.run(
            self.storage.step(),
            input_list_key=self.keys["parsed_list"],
            input_image_key=self.keys["img"],
            output_key=self.keys["final"]
        )
        
        print(f">>> [Pipeline] Done. Results in: {self.keys['final']}")

if __name__ == "__main__":
    pipe = VisualOnlyMCQPipeline(
        model_path="Qwen/Qwen2.5-VL-3B-Instruct",
        first_entry_file="../example_data/capsbench_images/image_visual_only_mcq_demo.jsonl",
        hf_cache_dir="~/.cache/huggingface",
        download_dir="../ckpt/models/Qwen2.5-VL-3B-Instruct",
        rotate_num=4,
        pass_visual_min=1.0,
        pass_textual_max=0.25
    )
    pipe.forward()