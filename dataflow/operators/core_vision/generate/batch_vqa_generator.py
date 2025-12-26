from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC, LLMServingABC
from dataflow import get_logger
from qwen_vl_utils import process_vision_info


@OPERATOR_REGISTRY.register()
class BatchVQAGenerator(OperatorABC):
    """
    [Generate] 批量视觉问答生成器。
    输入：问题列表 (Questions) + 图片。
    输出：答案列表 (New Text Content)。
    """
    def __init__(self, serving: LLMServingABC, system_prompt: str = "You are a helpful assistant."):
        self.serving = serving
        self.system_prompt = system_prompt
        self.logger = get_logger()

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "批量视觉问答生成算子 (BatchVQAGenerator)。\n"
                "该算子用于针对单张图片回答列表中的多个问题 (One Image, Many Questions)。\n\n"
                "输入参数：\n"
                "  - input_prompts_key: 问题列表列 (List[str])\n"
                "  - input_image_key: 图像列\n"
                "输出参数：\n"
                "  - output_key: 生成的答案列表列 (List[str])\n"
                "功能特点：\n"
                "  - 自动进行广播 (Broadcasting)，将单图映射到多个问题\n"
                "  - 适用于由粗到细 (Coarse-to-Fine) 的密集描述生成场景\n"
            )
        else:
            return (
                "Batch VQA Generator (BatchVQAGenerator).\n"
                "This operator answers multiple questions based on a single image (One Image, Many Questions).\n\n"
                "Input Parameters:\n"
                "  - input_prompts_key: Column containing the list of questions\n"
                "  - input_image_key: Column containing the image\n"
                "Output Parameters:\n"
                "  - output_key: Column storing the list of generated answers\n"
                "Features:\n"
                "  - Automatically broadcasts one image to multiple prompts\n"
                "  - Ideal for coarse-to-fine dense captioning scenarios\n"
            )

    def run(self, storage: DataFlowStorage, input_prompts_key: str, input_image_key: str, output_key: str):
        self.logger.info(f"Running BatchVQAGenerator on {input_prompts_key}...")
        df = storage.read("dataframe")
        
        all_answers_nested = []
        
        for idx, row in df.iterrows():
            questions = row.get(input_prompts_key, [])
            image_path = row.get(input_image_key)
            
            if not questions or not isinstance(questions, list) or not image_path:
                all_answers_nested.append([])
                continue

            batch_prompts = []
            batch_images = []
            
            for q in questions:
                raw = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": q}
                    ]}
                ]
                image_inputs, _ = process_vision_info(raw)
                final_p = self.serving.processor.apply_chat_template(raw, tokenize=False, add_generation_prompt=True)
                
                batch_prompts.append(final_p)
                batch_images.append(image_inputs)
            
            if not batch_prompts:
                all_answers_nested.append([])
                continue

            # 批量调用
            row_answers = self.serving.generate_from_input(
                system_prompt=self.system_prompt,
                user_inputs=batch_prompts,
                image_inputs=batch_images
            )
            all_answers_nested.append(row_answers)
            
        df[output_key] = all_answers_nested
        storage.write(df)
        return [output_key]