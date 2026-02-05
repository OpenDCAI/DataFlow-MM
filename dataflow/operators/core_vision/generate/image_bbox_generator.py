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

    pre_alpha_image = np.zeros_like(image)
    alpha = 0.8
    beta = 1.0 - alpha
    image = cv2.addWeighted(image, alpha, pre_alpha_image, beta, 0)

    for i, (x, y, box_w, box_h) in enumerate(bbox, start=1):
        x,y,box_w,box_h = int(x), int(y),int(box_w), int(box_h)
        cv2.rectangle(image, (x, y), (x + box_w, y + box_h), rgb, rect_thickness)

        text_x, text_y = x + 4, y + 20
        if text_x < 0:
            text_x = 0
        if text_y < 0: 
            text_y = y + box_h + 15
        if text_y > h:
            text_y = h - 5

        thickness = 2
        text = str(i)
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, thickness)
        text_x = x + 4
        text_y = y + 20
        cv2.rectangle(image, (text_x, text_y - text_height - baseline), (text_x + text_width, text_y + baseline), (0, 0, 0), -1)
        cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), thickness)

    save_path = vis_path
    signal=cv2.imwrite(save_path, image)

    return save_path

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
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return []
    
    h_img, w_img, _ = image.shape
    min_area = w_img * h_img * min_area_ratio
    max_area = w_img * h_img * max_area_ratio 

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.adaptiveThreshold(
        blurred, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        11, 
        2 
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour) 
        if h == 0: continue
        aspect_ratio = w / h
        if area < min_area or area > max_area:
            continue
        if min_aspect_ratio > aspect_ratio or aspect_ratio > max_aspect_ratio:
            continue
        if w > w_img * 0.9 or h > h_img * 0.9:
            continue
        boxes.append((x, y, w, h))
    boxes = non_max_suppression(boxes, overlap_thresh=0.3)
            
    return boxes

@dataclass
class ExistingBBoxDataGenConfig:
    max_boxes: int = 10
    input_jsonl_path: Optional[str] = None
    output_jsonl_path: Optional[str] = None


@OPERATOR_REGISTRY.register()
class ImageBboxGenerator(OperatorABC):
    '''
    Caption Generator is a class that generates captions for given images.
    '''
    def __init__(self, config: Optional[ExistingBBoxDataGenConfig] = None):
        self.logger = get_logger()
        self.prompt_generator = CaptionGeneratorPrompt()
        self.cfg = config or ExistingBBoxDataGenConfig()

    @staticmethod
    def get_desc(lang: str = "zh"):
        if str(lang).lower().startswith("zh"):
            return (
                "功能说明：\n"
                "  用于图像区域描述任务的数据准备阶段。从 JSONL 读取图像，处理边界框（自动提取或读取已有），生成带框的可视化图像，并构建 VLM 所需的 Prompt。\n"
                "\n"
                "处理流程：\n"
                "  1. 读取输入：从 input_jsonl_path 读取数据。\n"
                "  2. BBox 处理：若无 bbox 字段，使用 extract_boxes_from_image 自动提取；对坐标进行归一化（normalized_bbox）并补零对齐。\n"
                "  3. 可视化：调用 paint_text_box 生成绘制了编号方框的图像，保存至 cache 路径。\n"
                "  4. Prompt 构造：根据有效框数量生成提示词（例如：'Describe the content... There are N regions...'）。\n"
                "  5. 输出：将处理后的元数据写入 output_jsonl_path。\n"
                "\n"
                "输出数据结构（Flat JSON）：\n"
                "  {\n"
                "    \"image\": \"/path/to/img.jpg\",             // 原始图像路径\n"
                "    \"type\": \"with_bbox/without_bbox\",      // 框来源类型\n"
                "    \"bbox\": [[x,y,w,h], ...],                // 原始坐标\n"
                "    \"normalized_bbox\": [[0.1,0.2...], ...],  // 归一化坐标\n"
                "    \"image_with_bbox\": \"/path/to/vis.jpg\", // 可视化结果图路径\n"
                "    \"valid_bboxes_num\": 3,                   // 有效框数量\n"
                "    \"prompt\": \"Describe the content...\"    // 生成的提示词\n"
                "  }\n"
            )
        else:
            return (
                "Function Description:\n"
                "  Data preparation stage for image region captioning. Reads images from JSONL, processes bounding boxes (auto-extract or existing), generates visualized images with boxes, and constructs Prompts for VLM.\n"
                "\n"
                "Pipeline:\n"
                "  1. Input: Reads data from input_jsonl_path.\n"
                "  2. BBox Processing: Auto-extracts bboxes via extract_boxes_from_image if missing; Normalizes coordinates (normalized_bbox) with padding.\n"
                "  3. Visualization: Generates images with numbered bounding boxes via paint_text_box and saves to cache.\n"
                "  4. Prompt: Generates a prompt string based on the number of valid boxes.\n"
                "  5. Output: Writes processed metadata to output_jsonl_path.\n"
                "\n"
                "Output Data Structure (Flat JSON):\n"
                "  {\n"
                "    \"image\": \"/path/to/img.jpg\",             // Original image path\n"
                "    \"type\": \"with_bbox/without_bbox\",      // Source type\n"
                "    \"bbox\": [[x,y,w,h], ...],                // Raw coordinates\n"
                "    \"normalized_bbox\": [[0.1,0.2...], ...],  // Normalized coordinates\n"
                "    \"image_with_bbox\": \"/path/to/vis.jpg\", // Visualized image path\n"
                "    \"valid_bboxes_num\": 3,                   // Number of valid boxes\n"
                "    \"prompt\": \"Describe the content...\"    // Generated prompt string\n"
                "  }\n"
            )
    
    def _normalize_bboxes(self, bboxes: List[List[float]], img_width: int, img_height: int,row=None) -> List[List[float]]:
        normalized = []
        for bbox in bboxes[:self.cfg.max_boxes]:
            norm_bbox = vp_normalize(bbox, pad_x=0, pad_y=0, width=img_width, height=img_height)
            normalized.append(norm_bbox)
        while len(normalized) < self.cfg.max_boxes:
            normalized.append([0.0, 0.0, 0.0, 0.0])
        return normalized

    def _generate_visualization(self, image_path: str, bboxes: List[List[float]],vispath,counter) -> str:
        vis_path = os.path.join(vispath , f"{counter}_bbox_vis.jpg")
        os.makedirs(vispath, exist_ok=True)
        paint_text_box(image_path, bboxes,vis_path)
        return vis_path

    def _gen_prompt(self, bbox_count: int) -> str:
        return (f"Describe the content of each marked region in the image. "
                f"There are {bbox_count} regions: <region1> to <region{bbox_count}>.")

    def run(self, storage: DataFlowStorage, input_image_key: str = "image", input_bbox_key: str = "bbox"):
        rows = []
        if self.cfg.input_jsonl_path:
            with open(self.cfg.input_jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():  # <--- 新增：如果是空行，直接跳过
                        continue
                    try:
                        rows.append(json.loads(line.strip()))
                    except json.JSONDecodeError as e:
                        print(f"Skipping invalid line: {line[:50]}... Error: {e}")
        out_records=[]
        counter=0
        for row in rows:
            counter+=1
            image_path = row[input_image_key]
            if input_bbox_key in row:
                raw_bboxes = row[input_bbox_key]
                typ='with_bbox'
            else:
                raw_bboxes=extract_boxes_from_image(image_path)
                typ='without_bbox'
            if not image_path or not raw_bboxes:
                print(f'Error! {row} has no {input_image_key} or {input_bbox_key}!')
                continue

            img = cv2.imread(image_path)
            h, w = img.shape[:2]

            normalized_bboxes = self._normalize_bboxes(raw_bboxes, w, h,row)

            vis_path = self._generate_visualization(image_path, raw_bboxes[:self.cfg.max_boxes],storage
            .cache_path,counter)

            valid_bbox_count = sum(1 for b in normalized_bboxes if sum(b) != 0)
            
            record={
                'image':image_path,
                'type':typ,
                'bbox':raw_bboxes,
                'normalized_bbox':normalized_bboxes,
                'result_file':storage.cache_path,
                'image_with_bbox':vis_path,
                'valid_bboxes_num':valid_bbox_count,
                'prompt':self._gen_prompt(valid_bbox_count)
            }
            out_records.append(record)

        if self.cfg.output_jsonl_path:
            os.makedirs(os.path.dirname(self.cfg.output_jsonl_path), exist_ok=True)
            with open(self.cfg.output_jsonl_path, "w", encoding="utf-8") as f:
                for r in out_records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return