from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC, LLMServingABC
from dataflow import get_logger
from qwen_vl_utils import process_vision_info


@OPERATOR_REGISTRY.register()
class VisualGroundingRefiner(OperatorABC):
    """
    [Refine] 视觉一致性精炼器。
    输入：文本列表 (sentences/answers) + 图片。
    行为：对列表中每一项进行视觉验证 (Yes/No)，去除幻觉内容。
    """
    def __init__(self, serving: LLMServingABC, prompt_template: str, system_prompt: str = "You are a helpful assistant."):
        self.serving = serving
        self.template = prompt_template  # 必须包含 {text} 占位符
        self.system_prompt = system_prompt
        self.logger = get_logger()

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "视觉一致性精炼算子 (VisualGroundingRefiner)。\n"
                "该算子用于对文本列表进行逐项视觉验证，过滤掉与图像不符的内容。\n\n"
                "输入参数：\n"
                "  - input_list_key: 待验证的文本列表列 (如句子列表)\n"
                "  - input_image_key: 对应的图像列\n"
                "  - prompt_template: 验证用的 Prompt 模板 (需含 {text} 占位符)\n"
                "输出参数：\n"
                "  - output_key: 过滤后保留的文本列表\n"
                "功能特点：\n"
                "  - 自动构造 'Yes/No' 判别问题\n"
                "  - 批量并行推理，高效过滤幻觉 (Hallucination)\n"
            )
        else:
            return (
                "Visual Grounding Refiner (VisualGroundingRefiner).\n"
                "This operator verifies a list of texts against an image and filters out unsupported content.\n\n"
                "Input Parameters:\n"
                "  - input_list_key: Column containing the list of texts to verify\n"
                "  - input_image_key: Column containing the image\n"
                "  - prompt_template: Prompt template for verification (must include {text})\n"
                "Output Parameters:\n"
                "  - output_key: The filtered list of texts\n"
                "Features:\n"
                "  - Automatically constructs 'Yes/No' verification questions\n"
                "  - Batch parallel inference for efficient hallucination filtering\n"
            )

    def run(self, storage: DataFlowStorage, input_list_key: str, input_image_key: str, output_key: str):
        self.logger.info(f"Running VisualGroundingRefiner on {input_list_key}...")
        df = storage.read("dataframe")
        
        refined_results = []
        
        for idx, row in df.iterrows():
            items = row.get(input_list_key, [])
            image_path = row.get(input_image_key)
            
            if not items or not isinstance(items, list) or not image_path:
                refined_results.append([])
                continue
            
            # 1. 构造 Batch Prompts
            batch_prompts = []
            batch_images = []
            
            for item in items:
                text_prompt = self.template.format(text=item)
                raw_prompt = [
                    {"role": "system", "content": self.system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": text_prompt},
                            {"type": "image", "image": image_path}
                        ]
                    }
                ]
                image_inputs, _ = process_vision_info(raw_prompt)
                final_prompt = self.serving.processor.apply_chat_template(
                    raw_prompt, tokenize=False, add_generation_prompt=True
                )
                batch_prompts.append(final_prompt)
                batch_images.append(image_inputs)

            if not batch_prompts:
                refined_results.append([])
                continue

            # 2. 批量推理
            flags = self.serving.generate_from_input(
                system_prompt=self.system_prompt,
                user_inputs=batch_prompts,
                image_inputs=batch_images
            )
            
            # 3. 过滤逻辑 (保留 'yes')
            kept = [item for item, flag in zip(items, flags) if "yes" in flag.lower()]
            refined_results.append(kept)

        df[output_key] = refined_results
        storage.write(df)
        return [output_key]
