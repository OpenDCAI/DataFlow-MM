import re
import argparse
from typing import Callable, Any, List

from dataflow.utils.storage import FileStorage

from dataflow.serving.local_model_vlm_serving import LocalModelVLMServing_vllm

from dataflow.prompts.prompt_template import NamedPlaceholderPromptTemplate
from dataflow.prompts.image import ImageScaleCaptionPrompt

from dataflow.operators.core_vision import PromptedVQAGenerator, BatchVQAGenerator, VisualGroundingRefiner
from dataflow.operators.core_text import PromptTemplatedQAGenerator, FunctionalRefiner


def split_sentences(text: str) -> List[str]:
    """将文本拆分为句子列表"""
    if not text or not isinstance(text, str):
        return []
    # 使用正则按标点符号分割 (. ! ? 。 ！ ？)
    _SENT_SPLIT = re.compile(r"(?<=[.!?。！？])\s+")
    parts = [p.strip() for p in _SENT_SPLIT.split(text) if p.strip()]
    return parts or ([text.strip()] if text.strip() else [])

def join_list(data: Any, separator: str = "\n") -> str:
    """将列表连接为字符串"""
    if isinstance(data, list):
        # 过滤掉非字符串元素或空字符串
        valid_items = [str(x) for x in data if x]
        return separator.join(valid_items)
    return str(data) if data is not None else ""

def parse_questions_logic(text: str, max_q: int = 20) -> List[str]:
    """
    解析 LLM 生成的 "Describe more details about..." 文本，
    并自动扩展 position 问题。
    """
    if not text or not isinstance(text, str):
        return []

    lines = [t.strip() for t in text.split("\n") if t.strip()]
    obj_qs = []
    
    for line in lines:
        # 提取包含 "Describe more details about" 的行
        if "Describe more details about" in line:
            # 去除可能的序号 (如 "1. Describe...")
            try:
                start_idx = line.find("Describe")
                clean = line[start_idx:]
                # 去除句末多余内容，保留到第一个句号
                if "." in clean:
                    clean = clean.split(".")[0] + "."
                obj_qs.append(clean)
            except Exception:
                continue
    
    # 去重并保持顺序
    seen = set()
    unique_obj_qs = []
    for q in obj_qs:
        if q not in seen:
            unique_obj_qs.append(q)
            seen.add(q)
    
    # 截断
    unique_obj_qs = unique_obj_qs[:max_q]
    
    # 扩展 Position 问题
    pos_qs = [
        q.replace("Describe more details about", "Describe more details about the position of")
        for q in unique_obj_qs
    ]
    
    # 返回合并后的列表 (对象问题 + 位置问题)
    return unique_obj_qs + pos_qs


class ImageScaleCaptionPipeline:
    def __init__(
        self,
        model_path: str,
        *,
        hf_cache_dir: str | None = None,
        download_dir: str = "./ckpt/models",
        device: str = "cuda",
        # Storage params
        first_entry_file: str = "images.jsonl",
        cache_path: str = "./cache_scalecap",
        file_name_prefix: str = "scalecap",
        cache_type: str = "jsonl",
        # Keys
        input_image_key: str = "image",
        output_key: str = "final_caption",
        # VLLM Config
        vllm_tensor_parallel_size: int = 1,
        vllm_temperature: float = 0.7,
        vllm_top_p: float = 0.9,
        vllm_max_tokens: int = 512,
    ):
        # 1. Storage
        self.storage = FileStorage(
            first_entry_file_name=first_entry_file,
            cache_path=cache_path,
            file_name_prefix=file_name_prefix,
            cache_type=cache_type,
        )

        # 2. Serving
        self.serving = LocalModelVLMServing_vllm(
            hf_model_name_or_path=model_path,
            hf_cache_dir=hf_cache_dir,
            hf_local_dir=download_dir,
            vllm_tensor_parallel_size=vllm_tensor_parallel_size,
            vllm_temperature=vllm_temperature,
            vllm_top_p=vllm_top_p,
            vllm_max_tokens=vllm_max_tokens,
        )

        # 3. Prompts
        self.prompts_db = ImageScaleCaptionPrompt().build_prompt()

        # 4. Keys
        self.input_image_key = input_image_key
        self.output_key = output_key

        # ================== Operator Initialization ==================

        # --- Step A: Generate Init Caption ---
        # 构造固定 Prompt 列
        self.refine_const_prompt = FunctionalRefiner(func=lambda: self.prompts_db["VLM_PROMPT_1"])
        
        # 生成初稿 (使用通用 PromptedVQAGenerator)
        self.gen_init_caption = PromptedVQAGenerator(
            serving=self.serving,
            system_prompt="You are a helpful assistant."
        )

        # --- Step B: Refine Golden Sentences ---
        # 分句
        self.refine_split = FunctionalRefiner(func=split_sentences)
        
        # 视觉自检 (保留 Yes 的句子)
        self.refine_golden = VisualGroundingRefiner(
            serving=self.serving,
            prompt_template="Given the image, is the description '{text}' directly supported by visual evidence? Answer strictly yes or no."
        )

        # --- Step C: Generate Questions ---
        # 列表转字符串
        self.refine_join = FunctionalRefiner(func=join_list)
        
        # 文本生成问题 (Text-to-Text)
        tpl_q = NamedPlaceholderPromptTemplate(
            template=self.prompts_db["LLM_PROMPT_1"], 
            join_list_with="\n"
        )
        self.gen_questions_text = PromptTemplatedQAGenerator(
            serving=self.serving,
            prompt_template=tpl_q
        )
        
        # 解析问题文本为列表
        self.refine_parse_qs = FunctionalRefiner(func=parse_questions_logic)

        # --- Step D: Generate Answers ---
        # 批量回答 (One Image -> Many Qs)
        self.gen_answers = BatchVQAGenerator(serving=self.serving)
        
        # 回答过滤
        self.refine_answers = VisualGroundingRefiner(
            serving=self.serving,
            prompt_template="Given the image, is the statement '{text}' grounded in the image and not generic? Answer strictly yes or no."
        )

        # --- Step E: Integrate Final Caption ---
        # 融合 (Text-to-Text)
        tpl_final = NamedPlaceholderPromptTemplate(
            template=self.prompts_db["LLM_PROMPT_4"], 
            join_list_with="\n"
        )
        self.gen_final_caption = PromptTemplatedQAGenerator(
            serving=self.serving,
            prompt_template=tpl_final
        )

    def forward(self):
        print(">>> [Pipeline] Step 0: Preparing Prompts...")
        # 构造 init_prompt 列
        self.refine_const_prompt.run(
            self.storage.step(), 
            output_key="init_prompt"
        )

        print(">>> [Pipeline] Step 1: Generating Initial Caption...")
        self.gen_init_caption.run(
            self.storage.step(),
            input_prompt_key="init_prompt",
            input_image_key=self.input_image_key,
            output_answer_key="init_caption"
        )

        print(">>> [Pipeline] Step 2: Refining Golden Sentences...")
        self.refine_split.run(
            self.storage.step(), 
            output_key="sentences", 
            text="init_caption"
        )
        self.refine_golden.run(
            self.storage.step(), 
            input_list_key="sentences", 
            input_image_key=self.input_image_key, 
            output_key="golden_sentences"
        )

        print(">>> [Pipeline] Step 3: Generating Details Questions...")
        self.refine_join.run(
            self.storage.step(), 
            output_key="golden_str", 
            data="golden_sentences"
        )
        
        # template: "{sentence}" -> map to col "golden_str"
        self.gen_questions_text.run(
            self.storage.step(), 
            output_answer_key="raw_q_text", 
            sentence="golden_str"
        )
        
        self.refine_parse_qs.run(
            self.storage.step(), 
            output_key="q_list", 
            text="raw_q_text"
        )

        print(">>> [Pipeline] Step 4: Generating & Filtering Answers...")
        self.gen_answers.run(
            self.storage.step(), 
            input_prompts_key="q_list", 
            input_image_key=self.input_image_key, 
            output_key="raw_answers"
        )
        
        self.refine_answers.run(
            self.storage.step(), 
            input_list_key="raw_answers", 
            input_image_key=self.input_image_key, 
            output_key="final_details"
        )

        print(">>> [Pipeline] Step 5: Integrating Final Caption...")
        self.refine_join.run(
            self.storage.step(), 
            output_key="details_str", 
            data="final_details"
        )
        
        # template keys: context, object_info, position_info
        self.gen_final_caption.run(
            self.storage.step(),
            output_answer_key=self.output_key,
            context="golden_str",
            object_info="details_str",
            position_info="details_str" # 简化：同时作为 object 和 position 信息
        )

        print(f">>> [Pipeline] All Done. Result saved to: {self.storage.cache_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ScaleCap Dense Captioning Pipeline")
    
    # Model
    parser.add_argument("--model_path", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--hf_cache_dir", default="~/.cache/huggingface")
    parser.add_argument("--download_dir", default="./ckpt/models")
    parser.add_argument("--device", default="cuda")

    # Storage / IO
    parser.add_argument("--input_jsonl", default="./dataflow/example/image_to_text_pipeline/capsbench_captions.jsonl", help="Input file with images")
    parser.add_argument("--cache_path", default="./cache_scalecap_results")
    parser.add_argument("--file_name_prefix", default="scalecap")
    parser.add_argument("--input_image_key", default="image")
    parser.add_argument("--output_key", default="final_caption")

    # vLLM Config
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--max_tokens", type=int, default=1024)

    args = parser.parse_args()

    pipe = ImageScaleCaptionPipeline(
        model_path=args.model_path,
        hf_cache_dir=args.hf_cache_dir,
        download_dir=args.download_dir,
        device=args.device,
        
        first_entry_file=args.input_jsonl,
        cache_path=args.cache_path,
        file_name_prefix=args.file_name_prefix,
        
        input_image_key=args.input_image_key,
        output_key=args.output_key,
        
        vllm_tensor_parallel_size=args.tp,
        vllm_max_tokens=args.max_tokens
    )
    
    pipe.forward()