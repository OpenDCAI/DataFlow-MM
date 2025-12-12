import torch
import gc
import re
from PIL import Image
from typing import List, Dict, Any

from dataflow import get_logger
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC, LLMServingABC
from qwen_vl_utils import process_vision_info

def parse_bbox_logic(text: str) -> List[List[float]]:
    """解析模型生成的 BBox 文本 (x1, y1, x2, y2)"""
    if not text: return []
    bboxes = []
    # 兼容 (0.1, 0.1), (0.2, 0.2) 格式
    pattern = r'\(\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\)\s*,\s*\(\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\)'
    for match in re.finditer(pattern, text):
        try:
            coords = list(map(float, match.groups()))
            x1, y1, x2, y2 = coords
            # 归一化处理 (适配 0-1000 输出)
            if any(c > 1.05 for c in coords):
                x1, y1, x2, y2 = x1/1000, y1/1000, x2/1000, y2/1000
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            bboxes.append([x1, y1, x2, y2])
        except: continue
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

    @staticmethod
    def get_desc(lang: str = "zh"):
        return "使用 VLM 模型检测关键词的 BBox (支持批量处理)。"

    def run(self, storage: DataFlowStorage, input_image_key: str, input_kws_key: str, output_key: str):
        self.logger.info("Running VLMBBoxGenerator...")
        df = storage.read("dataframe")
        bbox_maps = []
        
        for idx, row in df.iterrows():
            img_path = row.get(input_image_key)
            keywords = row.get(input_kws_key, [])
            row_map = {}
            
            # 校验数据有效性
            if not keywords or not isinstance(keywords, list) or not img_path:
                bbox_maps.append({})
                continue
            
            # 针对单张图片，去重关键词
            unique_kws = list(set(keywords))
            if not unique_kws:
                bbox_maps.append({})
                continue

            # --- 构造 Batch Request (One Image vs N Keywords) ---
            batch_prompts = []
            batch_images = []
            
            for kw in unique_kws:
                safe_kw = kw.replace('"', '\\"')
                text_prompt = self.prompt_tmpl.format(keyword=safe_kw)
                
                # 构造符合 Serving 接口的 raw prompt
                raw_prompt = [
                    {"role": "system", "content": "You are a helpful assistant capable of visual grounding."},
                    {"role": "user", "content": [
                        {"type": "image", "image": img_path},
                        {"type": "text", "text": text_prompt}
                    ]}
                ]
                
                # 处理 Vision Info
                try:
                    img_inp, _ = process_vision_info(raw_prompt)
                    prompt_str = self.serving.processor.apply_chat_template(
                        raw_prompt, tokenize=False, add_generation_prompt=True
                    )
                    
                    # Qwen2.5-VL 防御性补丁 (防止 template 没加占位符)
                    if "<|image_pad|>" not in prompt_str and "<image>" not in prompt_str:
                        prompt_str = "<|vision_start|><|image_pad|><|vision_end|>" + prompt_str
                        
                    batch_prompts.append(prompt_str)
                    batch_images.append(img_inp)
                except Exception as e:
                    self.logger.warning(f"Failed to prepare prompt for '{kw}': {e}")
            
            # --- 批量调用 Serving ---
            if not batch_prompts:
                bbox_maps.append({})
                continue

            try:
                outputs = self.serving.generate_from_input(
                    user_inputs=batch_prompts,
                    image_inputs=batch_images
                )
                
                # --- 解析结果 ---
                for kw, out_text in zip(unique_kws, outputs):
                    # 检查是否包含 "not found"
                    if "not found" in out_text.lower():
                        continue
                    
                    boxes = parse_bbox_logic(out_text)
                    if boxes:
                        # 格式化为字符串列表
                        box_strs = [f"[{b[0]:.3f}, {b[1]:.3f}, {b[2]:.3f}, {b[3]:.3f}]" for b in boxes]
                        row_map[kw] = box_strs[:3] # 保留前3个
                        
            except Exception as e:
                self.logger.error(f"Serving generation error at row {idx}: {e}")
            
            bbox_maps.append(row_map)
        
        df[output_key] = bbox_maps
        storage.write(df)
        return [output_key]