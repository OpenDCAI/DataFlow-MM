import pandas as pd
from typing import List

from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import FileStorage, DataFlowStorage
from dataflow.core import OperatorABC, LLMServingABC
from dataflow import get_logger

from dataflow.serving.local_model_vlm_serving import LocalModelVLMServing_vllm
from dataflow.serving.api_vlm_serving_openai import APIVLMServing_openai


# 提取判断是否为 API Serving 的辅助函数
def is_api_serving(serving):
    return isinstance(serving, APIVLMServing_openai)


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
                "  - 全局 Batch 展平处理，支持超大规模并发过滤\n"
                "  - 统一支持 API 和本地 Local 模型部署模式\n"
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
                "  - Global batch flattening for massive concurrent filtering\n"
                "  - Unifies support for API and Local model deployment modes\n"
            )

    def run(self, storage: DataFlowStorage, input_list_key: str, input_image_key: str, output_key: str):
        if not output_key:
            raise ValueError("'output_key' must be provided.")

        self.logger.info(f"Running VisualGroundingRefiner on {input_list_key}...")
        df: pd.DataFrame = storage.read("dataframe")
        
        use_api_mode = is_api_serving(self.serving)
        if use_api_mode:
            self.logger.info("Using API serving mode")
        else:
            self.logger.info("Using local serving mode")

        # ---------------------------------------------------------
        # 1. 展平数据阶段 (Flatten Data)
        # 将 N 张图片和对应 M 个待验证文本展平为一维请求列表
        # ---------------------------------------------------------
        flat_conversations = []
        flat_images = []
        row_mappings = []  # 记录这道 prompt 属于哪一行以及它的原始文本：{"row_idx": int, "item": str}

        for idx, row in df.iterrows():
            items = row.get(input_list_key, [])
            image_path = row.get(input_image_key)
            
            # 清洗图片路径
            if isinstance(image_path, str):
                image_path = [image_path]
            elif not image_path:
                image_path = []

            if not isinstance(items, list) or not items or not image_path:
                continue
            
            # 为每一张图的每一个待验证文本构造对话
            for item in items:
                # 安全校验，防止传入非字符串导致 format 报错
                if not isinstance(item, str):
                    item = str(item)
                    
                prompt_text = self.template.format(text=item)
                
                if use_api_mode:
                    content = prompt_text
                else:
                    img_tokens = "<image>" * len(image_path)
                    content = f"{img_tokens}\n{prompt_text}" if img_tokens else prompt_text
                
                flat_conversations.append([{"role": "user", "content": content}])
                flat_images.append(image_path)
                row_mappings.append({"row_idx": idx, "item": item})

        # ---------------------------------------------------------
        # 2. 批量推理阶段 (Batch Inference)
        # ---------------------------------------------------------
        if flat_conversations:
            self.logger.info(f"Verifying {len(flat_conversations)} items globally...")
            flat_outputs = self.serving.generate_from_input_messages(
                conversations=flat_conversations,
                image_list=flat_images,
                system_prompt=self.system_prompt
            )
        else:
            flat_outputs = []

        # ---------------------------------------------------------
        # 3. 重组解析阶段 (Unflatten & Parse Data)
        # ---------------------------------------------------------
        # 初始化一个与 df 等长的空列表字典
        refined_results = [[] for _ in range(len(df))]
        
        for mapping, out_text in zip(row_mappings, flat_outputs):
            idx = mapping["row_idx"]
            original_item = mapping["item"]
            
            # 过滤逻辑 (模型回复中包含 'yes' 视为验证通过，保留该文本)
            if out_text and "yes" in str(out_text).lower():
                refined_results[idx].append(original_item)

        df[output_key] = refined_results
        output_file = storage.write(df)
        self.logger.info(f"Results saved to {output_file}")
        
        return [output_key]
