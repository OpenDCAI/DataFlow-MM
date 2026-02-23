import re
import pandas as pd
from typing import List, Dict, Any

from dataflow import get_logger
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import FileStorage, DataFlowStorage
from dataflow.core import OperatorABC, LLMServingABC
from dataflow.serving.local_model_vlm_serving import LocalModelVLMServing_vllm
from dataflow.serving.api_vlm_serving_openai import APIVLMServing_openai


# 提取判断是否为 API Serving 的辅助函数
def is_api_serving(serving):
    return isinstance(serving, APIVLMServing_openai)


def parse_bbox_logic(text: str) -> List[List[float]]:
    """解析模型生成的 BBox 文本 (x1, y1), (x2, y2)"""
    if not text: 
        return []
    
    bboxes = []
    # 兼容 (0.1, 0.1), (0.2, 0.2) 格式
    pattern = r'\(\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\)\s*,\s*\(\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\)'
    
    for match in re.finditer(pattern, text):
        try:
            coords = list(map(float, match.groups()))
            x1, y1, x2, y2 = coords
            
            # 归一化处理 (适配 0-1000 输出，转换为 0-1 的相对坐标)
            if any(c > 1.05 for c in coords):
                x1, y1, x2, y2 = x1/1000, y1/1000, x2/1000, y2/1000
                
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            bboxes.append([x1, y1, x2, y2])
        except Exception: 
            continue
            
    return bboxes


@OPERATOR_REGISTRY.register()
class VLMBBoxGenerator(OperatorABC):
    """
    [Generate] 使用通用 VLM Serving 生成 BBox 数据。
    输入：Image + Keywords List
    输出：BBox Map
    """
    def __init__(self, serving: LLMServingABC, prompt_template: str = 'Detect "{keyword}".'):
        self.serving = serving
        self.prompt_tmpl = prompt_template
        self.logger = get_logger()
        self.system_prompt = "You are a helpful assistant capable of visual grounding."

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "视觉定位 BBox 生成算子 (VLMBBoxGenerator)。\n"
                "输入图像和关键词列表，使用 VLM 模型检测并输出目标的边界框。\n\n"
                "特点：\n"
                "  - 自动过滤和去重关键词\n"
                "  - 全局 Batch 展平处理，极大提升吞吐量\n"
                "  - 统一支持 API 和本地 Local 模型部署模式，自动管理多模态占位符\n"
                "  - 自动归一化坐标并提取前 3 个置信度最高的候选框\n"
            )
        else:
            return "Uses a VLM to detect bounding boxes for a list of keywords (Supports batching)."

    def run(self, storage: DataFlowStorage, input_image_key: str, input_kws_key: str, output_key: str):
        if not output_key:
            raise ValueError("'output_key' must be provided.")

        self.logger.info("Running VLMBBoxGenerator...")
        df: pd.DataFrame = storage.read("dataframe")
        
        use_api_mode = is_api_serving(self.serving)
        if use_api_mode:
            self.logger.info("Using API serving mode")
        else:
            self.logger.info("Using local serving mode")

        # ---------------------------------------------------------
        # 1. 展平数据阶段 (Flatten Data)
        # 将 N 张图片和 M 个关键词展平为 N*M 的一维请求列表
        # ---------------------------------------------------------
        flat_conversations = []
        flat_images = []
        row_mappings = []  # 记录这道 prompt 属于哪一行的哪个关键词：{"row_idx": int, "keyword": str}

        for idx, row in df.iterrows():
            img_path = row.get(input_image_key)
            keywords = row.get(input_kws_key, [])
            
            # 清洗图片路径
            if isinstance(img_path, str):
                img_path = [img_path]
            elif not img_path:
                img_path = []

            # 校验数据有效性
            if not isinstance(keywords, list) or not img_path:
                continue
            
            # 针对单张图片，去重关键词
            unique_kws = list(set([str(k) for k in keywords if k]))
            if not unique_kws:
                continue
            
            for kw in unique_kws:
                safe_kw = kw.replace('"', '\\"')
                text_prompt = self.prompt_tmpl.format(keyword=safe_kw)
                
                if use_api_mode:
                    content = text_prompt
                else:
                    img_tokens = "<image>" * len(img_path)
                    content = f"{img_tokens}\n{text_prompt}" if img_tokens else text_prompt
                
                flat_conversations.append([{"role": "user", "content": content}])
                flat_images.append(img_path)
                row_mappings.append({"row_idx": idx, "keyword": kw})

        # ---------------------------------------------------------
        # 2. 批量推理阶段 (Batch Inference)
        # 一次性将所有组合送入大模型，最大化利用显存和并发
        # ---------------------------------------------------------
        if flat_conversations:
            self.logger.info(f"Generating BBox for {len(flat_conversations)} image-keyword pairs...")
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
        # 初始化一个与 df 等长的空字典列表
        bbox_maps = [{} for _ in range(len(df))]
        
        for mapping, out_text in zip(row_mappings, flat_outputs):
            idx = mapping["row_idx"]
            kw = mapping["keyword"]
            
            # 检查是否包含 "not found"
            if not out_text or "not found" in str(out_text).lower():
                continue
            
            boxes = parse_bbox_logic(str(out_text))
            if boxes:
                # 格式化为字符串列表，仅保留前 3 个
                box_strs = [f"[{b[0]:.3f}, {b[1]:.3f}, {b[2]:.3f}, {b[3]:.3f}]" for b in boxes]
                bbox_maps[idx][kw] = box_strs[:3]

        df[output_key] = bbox_maps
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

    # 如需测试 Local 模型，请解开注释
    # model = LocalModelVLMServing_vllm(
    #     hf_model_name_or_path="Qwen/Qwen2.5-VL-3B-Instruct",
    #     vllm_tensor_parallel_size=1,
    #     vllm_temperature=0.7,
    #     vllm_top_p=0.9,
    #     vllm_max_tokens=512,
    # )

    generator = VLMBBoxGenerator(
        serving=model,
        prompt_template='Detect "{keyword}". Please provide the bounding boxes in (x1, y1), (x2, y2) format.'
    )

    storage = FileStorage(
        first_entry_file_name="./dataflow/example/image_to_text_pipeline/bbox_sample.jsonl", 
        cache_path="./cache_bbox",
        file_name_prefix="bbox_gen",
        cache_type="jsonl",
    )
    storage.step()

    generator.run(
        storage=storage,
        input_image_key="image",
        input_kws_key="keywords",  # 假设这列的数据格式为: ["cat", "dog", "car"]
        output_key="bbox_map",     # 输出将被存为: {"cat": ["[0.1, 0.2, 0.3, 0.4]"], "dog": [...]}
    )