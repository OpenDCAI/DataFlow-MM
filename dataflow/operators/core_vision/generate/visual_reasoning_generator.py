import pandas as pd
from typing import Optional, List, Dict, Any

from dataflow import get_logger
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import FileStorage, DataFlowStorage
from dataflow.core import OperatorABC, LLMServingABC

from dataflow.serving.local_model_vlm_serving import LocalModelVLMServing_vllm
from dataflow.serving.api_vlm_serving_openai import APIVLMServing_openai
from dataflow.prompts.image import MCTReasoningPrompt


# 提取判断是否为 API Serving 的辅助函数
def is_api_serving(serving):
    return isinstance(serving, APIVLMServing_openai)


@OPERATOR_REGISTRY.register()
class VisualReasoningGenerator(OperatorABC):
    """
    [Generate] 调用 VLM 生成推理链。
    支持 Fallback：如果 input_existing_chains_key 中已有数据，则直接使用，不进行生成。
    """
    def __init__(self, serving: LLMServingABC, prompt_type: str = "web_grounding"):
        self.serving = serving
        self.prompt_type = prompt_type
        self.prompt_generator = MCTReasoningPrompt()
        self.system_prompt = self._get_sys_prompt()
        self.logger = get_logger()

    def _get_sys_prompt(self):
        prompts = self.prompt_generator.build_prompt()
        if self.prompt_type not in prompts:
            self.logger.warning(f"Prompt type '{self.prompt_type}' not found. Using fallback system prompt.")
            return "You are a helpful assistant capable of deep visual reasoning."
        return prompts[self.prompt_type]

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "视觉推理生成算子 (VisualReasoningGenerator)。\n"
                "调用 VLM 生成带 <think> 和 <answer> 的视觉推理链 (MCT)。\n\n"
                "特点：\n"
                "  - 支持 Fallback 机制，断点续传跳过已生成的行\n"
                "  - 统一支持 API 和本地 Local 模型部署模式\n"
                "  - 全局 Batch 处理未命中缓存的数据，保证最高吞吐量\n"
            )
        else:
            return "Generates visual reasoning chains using VLM with fallback support."

    def run(
        self, 
        storage: DataFlowStorage, 
        input_question_key: str, 
        input_image_key: str, 
        output_key: str,
        input_existing_chains_key: Optional[str] = None
    ):
        if not output_key:
            raise ValueError("'output_key' must be provided.")

        self.logger.info("Running VisualReasoningGenerator...")
        df: pd.DataFrame = storage.read("dataframe")

        use_api_mode = is_api_serving(self.serving)
        if use_api_mode:
            self.logger.info("Using API serving mode")
        else:
            self.logger.info("Using local serving mode")

        # 初始化最终结果列表 (用 None 占位，长度与 DataFrame 相同)
        final_results = [None] * len(df)
        
        # 1. 过滤与展平阶段 (Filter & Flatten Data)
        flat_conversations = []
        flat_images = []
        indices_to_generate = []  # 记录真正需要跑大模型的行索引

        for idx, row in df.iterrows():
            # --- 处理 Fallback (断点续传缓存) ---
            existing = row.get(input_existing_chains_key) if input_existing_chains_key else None
            if existing and isinstance(existing, list) and len(existing) > 0:
                final_results[idx] = existing
                continue
            
            # --- 提取正常数据 ---
            q = row.get(input_question_key, "")
            img_path = row.get(input_image_key)
            
            if not isinstance(q, str) or not q.strip():
                final_results[idx] = []
                continue

            # 清洗图片路径
            if isinstance(img_path, str):
                img_path = [img_path]
            elif not img_path:
                img_path = []
            
            valid_img_paths = [p for p in img_path if p and isinstance(p, str)]

            # 构造输入 Content
            if use_api_mode:
                content = q
            else:
                img_tokens = "<image>" * len(valid_img_paths)
                content = f"{img_tokens}\n{q}" if img_tokens else q

            flat_conversations.append([{"role": "user", "content": content}])
            flat_images.append(valid_img_paths)
            indices_to_generate.append(idx)

        # 2. 批量推理阶段 (Batch Inference)
        if flat_conversations:
            self.logger.info(f"Generating reasoning chains for {len(flat_conversations)} samples "
                             f"({len(df) - len(flat_conversations)} skipped due to Fallback or empty input)...")
            
            outputs = self.serving.generate_from_input_messages(
                conversations=flat_conversations,
                image_list=flat_images,
                system_prompt=self.system_prompt
            )
            
            # 3. 数据重组回填阶段 (Reconstruct Data)
            for df_idx, out_text in zip(indices_to_generate, outputs):
                final_results[df_idx] = [out_text] if out_text else []

        # 扫尾：把跳过大模型且没有缓存的 None 替换为空列表
        final_results = [res if res is not None else [] for res in final_results]
        
        # 写入结果
        df[output_key] = final_results
        output_file = storage.write(df)
        self.logger.info(f"Results saved to {output_file}")
        
        return [output_key]


# ==========================================
# 测试用例 (Main Block)
# ==========================================
if __name__ == "__main__":
    
    # 使用 API 模式测试
    model = APIVLMServing_openai(
        api_url="http://172.96.141.132:3001/v1",
        key_name_of_api_key="DF_API_KEY",
        model_name="gpt-5-nano-2025-08-07",
        image_io=None,
        send_request_stream=False,
        max_workers=10,
        timeout=1800
    )

    # 如需使用本地模型，请解开注释
    # model = LocalModelVLMServing_vllm(
    #     hf_model_name_or_path="Qwen/Qwen2.5-VL-3B-Instruct",
    #     vllm_tensor_parallel_size=1,
    #     vllm_temperature=0.7,
    #     vllm_top_p=0.9,
    #     vllm_max_tokens=512,
    # )

    generator = VisualReasoningGenerator(
        serving=model,
        prompt_type="web_grounding"
    )

    storage = FileStorage(
        first_entry_file_name="./dataflow/example/image_to_text_pipeline/reasoning_sample.jsonl", 
        cache_path="./cache_reasoning",
        file_name_prefix="visual_reasoning",
        cache_type="jsonl",
    )
    storage.step()

    generator.run(
        storage=storage,
        input_question_key="question",
        input_image_key="image",
        output_key="reasoning_chain",
        input_existing_chains_key="cached_reasoning" # 测试时可以在 jsonl 里留几行带这个字段的数据看看效果
    )