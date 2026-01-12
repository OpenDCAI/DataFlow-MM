from typing import Optional, List, Dict, Any

from dataflow import get_logger
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC, LLMServingABC
from qwen_vl_utils import process_vision_info

from dataflow.prompts.image import MCTReasoningPrompt

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
            # 如果没有匹配的，默认给一个基础的，防止报错
            return "You are a helpful assistant."
        return prompts[self.prompt_type]

    @staticmethod
    def get_desc(lang: str = "zh"):
        return "调用 VLM 生成带 <think> 和 <answer> 的视觉推理链 (支持 Fallback)。" if lang == "zh" else "Generates visual reasoning chains using VLM with fallback support."

    def run(
        self, 
        storage: DataFlowStorage, 
        input_question_key: str, 
        input_image_key: str, 
        output_key: str,
        input_existing_chains_key: Optional[str] = None
    ):
        df = storage.read("dataframe")
        final_results = []
        
        batch_prompts = []
        batch_images = []
        indices_to_generate = []

        for idx, row in df.iterrows():
            # Check Fallback
            existing = row.get(input_existing_chains_key) if input_existing_chains_key else None
            
            if existing and isinstance(existing, list) and len(existing) > 0:
                final_results.append(existing)
                continue
            
            q = row.get(input_question_key, "")
            img_path = row.get(input_image_key)
            
            if not q:
                final_results.append([])
                continue

            content = []
            if img_path:
                content.append({"type": "image", "image": img_path})
            content.append({"type": "text", "text": q})

            raw_prompt = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": content}
            ]
            
            # --- [Fix: 正确使用 process_vision_info] ---
            image_inputs, _ = process_vision_info(raw_prompt)
            final_p = self.serving.processor.apply_chat_template(
                raw_prompt, tokenize=False, add_generation_prompt=True
            )
            
            batch_prompts.append(final_p)
            batch_images.append(image_inputs)
            indices_to_generate.append(idx)
            final_results.append(None) # 占位符

        if batch_prompts:
            self.logger.info(f"Generating reasoning chains for {len(batch_prompts)} samples...")
            outputs = self.serving.generate_from_input(
                user_inputs=batch_prompts,
                image_inputs=batch_images
            )
            
            ptr = 0
            for i, res in enumerate(final_results):
                if res is None:
                    # 包装成 List[str] 保持统一
                    final_results[i] = [outputs[ptr]]
                    ptr += 1
        
        df[output_key] = final_results
        storage.write(df)
        return [output_key]
