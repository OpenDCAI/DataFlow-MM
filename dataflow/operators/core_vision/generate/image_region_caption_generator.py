from __future__ import annotations
import os
import json
import re
import cv2
import numpy as np
from dataclasses import dataclass

from typing import Any, Dict, List, Optional, Tuple
from dataflow.prompts.image import CaptionGeneratorPrompt
import pandas as pd

from dataflow.core.Operator import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger

from dataflow.utils.storage import FileStorage, DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC
from dataflow.serving.local_model_vlm_serving import LocalModelVLMServing_vllm

from qwen_vl_utils import process_vision_info

def vp_normalize(in_p, pad_x, pad_y, width, height):
    if len(in_p) == 2:
        x0, y0 = in_p
        x0 = x0 + pad_x
        y0 = y0 + pad_y
        sx0 = round(x0 / width, 3)
        sy0 = round(y0 / height,3)
        return [sx0, sy0, -1, -1]
    elif len(in_p) == 4:
        x0, y0, w, h = in_p
        x0 = x0 + pad_x
        y0 = y0 + pad_y
        sx0 = round(x0 / width, 3)
        sy0 = round(y0 / height, 3)
        sx1 = round((x0 + w) / width, 3)
        sy1 = round((y0 + h) / height, 3)
        return [sx0, sy0, sx1, sy1]


def paint_text_box(image_path, bbox, vis_path = None, rgb=(0, 255, 0), rect_thickness=2):
    image = cv2.imread(image_path)
    image_name = image_path.split('/')[-1].split('.')[0] + ".jpg"
    h, w, channels = image.shape

    # 创建一个与原始图像大小相同的黑色图像 (所有像素值为0)
    pre_alpha_image = np.zeros_like(image)
    alpha = 0.8
    beta = 1.0 - alpha
    image = cv2.addWeighted(image, alpha, pre_alpha_image, beta, 0)

    for i, (x, y, box_w, box_h) in enumerate(bbox, start=1):
        # 画矩形框
        x,y,box_w,box_h = int(x), int(y),int(box_w), int(box_h)
        cv2.rectangle(image, (x, y), (x + box_w, y + box_h), rgb, rect_thickness)

        # 初始文本位置
        text_x, text_y = x + 4, y + 20
        # 调整文本位置以防止出界
        if text_x < 0:  # 如果文本超出左边界
            text_x = 0
        if text_y < 0:  # 如果文本超出上边界
            text_y = y + box_h + 15
        if text_y > h:  # 如果文本超出下边界
            text_y = h - 5

        thickness = 2
        # 获取文本宽度和高度
        text = str(i)
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, thickness)
        # 计算文本位置
        text_x = x + 4
        text_y = y + 20
        # 绘制文本矩形背景
        cv2.rectangle(image, (text_x, text_y - text_height - baseline), (text_x + text_width, text_y + baseline), (0, 0, 0), -1)
        # 绘制文本
        cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), thickness)

    save_path = vis_path
    # 保存图像
    signal=cv2.imwrite(save_path, image)

    return save_path
def conv():
    conv = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."
    return conv

def non_max_suppression(boxes, overlap_thresh=0.3):
    """
    非极大值抑制 NMS 算法，去除重叠过多的框
    """
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]  # x2 = x + w
    y2 = boxes[:, 1] + boxes[:, 3]  # y2 = y + h

    areas = boxes[:, 2] * boxes[:, 3]
    idxs = np.argsort(areas)[::-1]

    keep = []
    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[1:]])
        yy1 = np.maximum(y1[i], y1[idxs[1:]])
        xx2 = np.minimum(x2[i], x2[idxs[1:]])
        yy2 = np.minimum(y2[i], y2[idxs[1:]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        intersection_area = w * h

        overlap = intersection_area / areas[idxs[1:]]

        idxs = np.delete(idxs, np.concatenate(([0], np.where(overlap > overlap_thresh)[0] + 1)))

    return boxes[keep].tolist()

def extract_boxes_from_image(image_path, min_area_ratio=0.01, max_area_ratio=0.8, min_aspect_ratio=0.1, max_aspect_ratio=5):
    """
    从图片中提取主体物体（矩形）坐标，使用形态学和自适应阈值进行优化。
        min_area_ratio (float): Minimum box area as a ratio
        max_area_ratio (float): Maximum box area as a ratio
        min_aspect_ratio (float): Minimum aspect ratio (w/h)
        max_aspect_ratio (float): Maximum aspect ratio (w/h)
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return []
    
    h_img, w_img, _ = image.shape
    min_area = w_img * h_img * min_area_ratio
    max_area = w_img * h_img * max_area_ratio 

    # 灰度化和去噪
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 自适应阈值
    thresh = cv2.adaptiveThreshold(
        blurred, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        11, 
        2 
    )

    # 形态学闭运算
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # 查找轮廓
    contours, _ = cv2.findContours(closed.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour) 

        # 过滤
        if h == 0: continue
        aspect_ratio = w / h
        
        if area < min_area or area > max_area:
            continue
            
        if min_aspect_ratio > aspect_ratio or aspect_ratio > max_aspect_ratio:
            continue
            
        if w > w_img * 0.9 or h > h_img * 0.9:
            continue

        boxes.append((x, y, w, h))
        
    # 非极大值抑制
    boxes = non_max_suppression(boxes, overlap_thresh=0.3)
            
    return boxes

@dataclass
class ExistingBBoxDataGenConfig:
    max_boxes: int = 10  # 单图最大框数量（与模型输入对齐）
    input_jsonl_path: Optional[str] = None  # 输入含框数据路径（每行含image和bbox字段）
    output_jsonl_path: Optional[str] = None  # 输出处理后数据路径
    draw_visualization: bool = True  # 是否生成带框可视化图


@OPERATOR_REGISTRY.register()
class ImageRegionCaptionGenerate(OperatorABC):
    '''
    Caption Generator is a class that generates captions for given images.
    '''
    def __init__(self, llm_serving: LLMServingABC, config: Optional[ExistingBBoxDataGenConfig] = None):
        self.logger = get_logger()
        self.prompt_generator = CaptionGeneratorPrompt()
        self.llm_serving = llm_serving
        self.cfg = config or ExistingBBoxDataGenConfig()

    @staticmethod
    def get_desc(lang: str = "zh"):
        if str(lang).lower().startswith("zh"):
            return (
                "  从包含 {\"image\": \"/path/to/img\"}（可选含 {\"bbox\": [[x,y,w,h],...]}）的 JSONL 数据中，批量生成图像各标记区域的描述。\n"
                "  支持两种框来源：输入自带bbox / 自动从图像提取bbox（使用边缘检测+轮廓拟合技术）；核心流程为：bbox处理（提取/标准化）→ 带框可视化生成 → VLM提示构造 → 区域描述生成 → 结果整合。\n"
                "\n"
                "依赖工具与服务：\n"
                "  • 框提取：extract_boxes_from_image（边缘检测+轮廓拟合，过滤小面积/异常宽高比框）\n"
                "  • 框标准化：vp_normalize（坐标归一化，适配模型输入）\n"
                "  • 可视化：paint_text_box（绘制带数字标记的彩色框）\n"
                "  • VLM服务：LocalModelVLMServing_vllm（基于qwen_vl_utils处理视觉输入）\n"
                "\n"
                "输入参数（使用方式与说明）：\n"
                "  • Config（ExistingBBoxDataGenConfig）：\n"
                "    - max_boxes: int = 10\n"
                "        单图最大处理框数量（超出截断，不足补零，与模型输入对齐）。\n"
                "    - input_jsonl_path: Optional[str]\n"
                "        输入JSONL路径（每行需含\"image\"字段，可选含\"bbox\"字段）。\n"
                "    - output_jsonl_path: Optional[str]\n"
                "        输出JSONL路径（存储含区域描述的完整记录）。\n"
                "    - draw_visualization: bool = True\n"
                "        是否生成带数字标记框的可视化图像（默认生成，路径存于storage.cache_path）。\n"
                "  • run(storage, image_key=\"image\", bbox_key=\"bbox\", output_key=\"mdvp_record\")：\n"
                "    - storage: DataFlowStorage\n"
                "        数据存储实例，用于获取可视化图像保存路径（cache_path）。\n"
                "    - image_key：输入数据中「图像路径」的字段名（默认\"image\"）。\n"
                "    - bbox_key：输入数据中「原始bbox」的字段名（默认\"bbox\"，无则自动从图像提取框）。\n"
                "    - output_key：输出数据中「区域描述记录」的字段名（默认\"mdvp_record\"）。\n"
                "\n"
                "处理流程：\n"
                "  1. 读取输入数据，检查是否包含bbox字段\n"
                "  2. 若无bbox字段，自动从图像中提取边界框（使用边缘检测技术）\n"
                "  3. 对bbox坐标进行标准化处理，适配模型输入格式\n"
                "  4. 生成带数字标记的可视化图像（可选）\n"
                "  5. 构造VLM提示词，请求生成区域描述\n"
                "  6. 整合结果并输出完整记录\n"
                "\n"
                "输出结构（写入 output_key 对应记录）：\n"
                "  {\n"
                "    \"image\": \"/path/to/img.jpg\",                     # 原始图像路径\n"
                "    \"mdvp_record\": \"<region1>: 描述1; <region2>: 描述2\", # VLM生成的各区域描述\n"
                "    \"meta_info\": {\n"
                "        \"type\": \"with_bbox/without_bbox\",          # 框来源类型（输入自带/图像自动提取）\n"
                "        \"bbox\": [[x0,y0,w0,h0], [x1,y1,w1,h1], ...], # 原始bbox坐标（未标准化）\n"
                "        \"normalized_bbox\": [[0.1,0.2,0.3,0.4], ...], # 标准化后bbox（适配模型，补零至max_boxes）\n"
                "        \"image_with_bbox\": \"/path/to/cache/1_bbox_vis.jpg\" # 带框可视化图像路径\n"
                "    }\n"
                "  }\n"
            )
        else:
            return (
                "  Generate region-specific captions for images from JSONL data containing {\"image\": \"/path/to/img\"} (optionally {\"bbox\": [[x,y,w,h],...]}).\n"
                "  Supports two bbox sources: input-provided bboxes / auto-extracted bboxes from images using edge detection. Pipeline: bbox processing (extraction/normalization) → boxed visualization → VLM prompt construction → region caption generation → result integration.\n"
                "\n"
                "Dependent Tools & Services:\n"
                "  • Bbox Extraction: extract_boxes_from_image (edge detection + contour fitting, filters small/abnormal-aspect-ratio boxes)\n"
                "  • Bbox Normalization: vp_normalize (coordinate normalization for model input)\n"
                "  • Visualization: paint_text_box (draws colored boxes with numeric labels)\n"
                "  • VLM Service: LocalModelVLMServing_vllm (visual input processed via qwen_vl_utils)\n"
                "\n"
                "Inputs & Usage:\n"
                "  • Config (ExistingBBoxDataGenConfig):\n"
                "    - max_boxes (int, 10): Max number of bboxes processed per image (truncates excess, pads zeros to match model input).\n"
                "    - input_jsonl_path (Optional[str]): Path to input JSONL (each line requires \"image\" field, optional \"bbox\" field).\n"
                "    - output_jsonl_path (Optional[str]): Path to output JSONL (stores complete records with region captions).\n"
                "    - draw_visualization (bool, True): Whether to generate boxed visualization images (saved to storage.cache_path by default).\n"
                "  • run(storage, image_key=\"image\", bbox_key=\"bbox\", output_key=\"mdvp_record\"):\n"
                "    - storage (DataFlowStorage): Storage instance for retrieving visualization save path (cache_path).\n"
                "    - image_key: Field name of \"image path\" in input data (default: \"image\").\n"
                "    - bbox_key: Field name of \"raw bboxes\" in input data (default: \"bbox\"; auto-extracts from image if missing).\n"
                "    - output_key: Field name of \"region caption record\" in output data (default: \"mdvp_record\").\n"
                "\n"
                "Processing Pipeline:\n"
                "  1. Read input data and check for bbox field\n"
                "  2. If no bbox field, automatically extract bounding boxes from image using edge detection\n"
                "  3. Normalize bbox coordinates for model input\n"
                "  4. Generate visualization with numbered boxes (optional)\n"
                "  5. Construct VLM prompts and generate region captions\n"
                "  6. Integrate results and output complete records\n"
                "\n"
                "Produced Record (written to output_key-related structure):\n"
                "  {\n"
                "    \"image\": \"/path/to/img.jpg\",\n"
                "    \"mdvp_record\": \"<region1>: description1; <region2>: description2\",\n"
                "    \"meta_info\": {\n"
                "        \"type\": \"with_bbox/without_bbox\",\n"
                "        \"bbox\": [[x0,y0,w0,h0], [x1,y1,w1,h1], ...],\n"
                "        \"normalized_bbox\": [[0.1,0.2,0.3,0.4], ...],\n"
                "        \"image_with_bbox\": \"/path/to/cache/1_bbox_vis.jpg\"\n"
                "    }\n"
                "  }\n"
            )
    
    # 框坐标标准化（适配模型输入）
    def _normalize_bboxes(self, bboxes: List[List[float]], img_width: int, img_height: int,row=None) -> List[List[float]]:
        normalized = []
        for bbox in bboxes[:self.cfg.max_boxes]:  # 截断超出最大数量的框
            # 代码库中vp_normalize函数：处理填充和归一化
            norm_bbox = vp_normalize(bbox, pad_x=0, pad_y=0, width=img_width, height=img_height)
            normalized.append(norm_bbox)
        # 不足max_boxes则补零
        while len(normalized) < self.cfg.max_boxes:
            normalized.append([0.0, 0.0, 0.0, 0.0])
        return normalized

    # 生成带框可视化图片
    def _generate_visualization(self, image_path: str, bboxes: List[List[float]],vispath,counter) -> str:
        if not self.cfg.draw_visualization:
            return ""
        # 调用代码库中绘制框标记的工具
        vis_path = os.path.join(vispath , f"{counter}_bbox_vis.jpg")
        paint_text_box(image_path, bboxes,vis_path)  # 绘制带数字标签的框
        return vis_path

    # 生成模型输入prompt（基于框标记的描述指令）
    def _gen_prompt(self, bbox_count: int) -> str:
        return (f"Describe the content of each marked region in the image. "
                f"There are {bbox_count} regions: <region1> to <region{bbox_count}>.")

    def run(self, storage: DataFlowStorage, image_key: str = "image", bbox_key: str = "bbox", output_key: str = "mdvp_record"):
        rows = []
        if self.cfg.input_jsonl_path:
            with open(self.cfg.input_jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    rows.append(json.loads(line.strip()))
        out_records=[]
        counter=0
        for row in rows:
            counter+=1
            image_path = row[image_key]
            if bbox_key in row:
                raw_bboxes = row[bbox_key]
                typ='with_bbox'
            else:
                raw_bboxes=extract_boxes_from_image(image_path)
                typ='without_bbox'
            if not image_path or not raw_bboxes:
                print(f'Error! {row} has no {image_key} or {bbox_key}!')
                continue

            img = cv2.imread(image_path)
            h, w = img.shape[:2]

            normalized_bboxes = self._normalize_bboxes(raw_bboxes, w, h,row)

            vis_path = self._generate_visualization(image_path, raw_bboxes[:self.cfg.max_boxes],storage
            .cache_path,counter)

            valid_bbox_count = sum(1 for b in normalized_bboxes if sum(b) != 0)
            prompt = self._gen_prompt(valid_bbox_count)
            system_prompt=conv()
            raw_prompt = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": vis_path},
                        {"type": "text", "text": prompt},
                    ],
                },
            ]
            image_inputs, _ = process_vision_info(raw_prompt)
            final_prompt = self.llm_serving.processor.apply_chat_template(
                    raw_prompt, tokenize=False, add_generation_prompt=True
                )
            output = self.llm_serving.generate_from_input(
                user_inputs=[final_prompt],
                image_inputs=[image_inputs]
                )
            record = {
                "image": image_path,
                "mdvp_record": output,
                "meta_info":{
                    "type":typ,
                    "bbox": raw_bboxes,
                    "normalized_bbox": normalized_bboxes,
                    "image_with_bbox": vis_path
                }
                
            }
            out_records.append(record)

        if self.cfg.output_jsonl_path:
            os.makedirs(os.path.dirname(self.cfg.output_jsonl_path), exist_ok=True)
            with open(self.cfg.output_jsonl_path, "w", encoding="utf-8") as f:
                for r in out_records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return [output_key]

# if __name__ == "__main__":
#     model_path = "/data0/mt/.cache/huggingface/hub/Qwen2.5-VL-3B-Instruct"

#     # Initialize model
#     model = LocalModelVLMServing_vllm(
#         hf_model_name_or_path=model_path,
#         vllm_tensor_parallel_size=1,
#         vllm_temperature=0.7,
#         vllm_top_p=0.9,
#         vllm_max_tokens=512,
#     )

#     caption_generator = ImageCaptionGenerate(
#         llm_serving=model
#     )

#     # Prepare input
#     storage = FileStorage(
#         first_entry_file_name="dataflow/example/Image2TextPipeline/test_image2caption.jsonl", 
#         cache_type="jsonl", 
#         media_key="image", 
#         media_type="image"
#     )
#     storage.step()  # Load the data

#     caption_generator.run(
#         storage=storage,
#         multi_modal_key="image",
#         output_key="caption"
#     )