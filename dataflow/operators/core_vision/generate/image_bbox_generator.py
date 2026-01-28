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
                "  从包含 {\"image\": \"/path/to/img\"}（可选携带 {\"bbox\": [[x,y,w,h],...]}）的 JSONL 格式数据中，批量生成图像内各标记区域的文本描述。\n"
                "  支持两种边界框来源：输入数据自带 bbox / 自动从图像提取 bbox（基于边缘检测+轮廓拟合技术）；核心流程为：bbox 处理（提取/标准化）→ 带框可视化图生成 → VLM 提示词构造 → 区域描述生成 → 结果整合输出。\n"
                "\n"
                "依赖组件：\n"
                "  • 框提取：extract_boxes_from_image（边缘检测+轮廓拟合，过滤小面积/异常宽高比框）\n"
                "  • 框标准化：vp_normalize（坐标归一化，适配 VLM 模型输入格式）\n"
                "  • 可视化：paint_text_box（绘制带数字编号的彩色边界框）\n"
                "  • VLM 服务：LocalModelVLMServing_vllm（基于 qwen_vl_utils 处理视觉输入）\n"
                "\n"
                "输入参数说明：\n"
                "  1. 配置类（ExistingBBoxDataGenConfig）：\n"
                "     - max_boxes (int, 默认10)：单张图像最大处理框数量（超出截断，不足补零，与模型输入维度对齐）\n"
                "     - input_jsonl_path (Optional[str])：输入 JSONL 文件路径（每行必须含\"image\"字段，可选含\"bbox\"字段）\n"
                "     - output_jsonl_path (Optional[str])：输出 JSONL 文件路径（存储包含区域描述的完整记录）\n"
                "     - draw_visualization (bool, 默认True)：是否生成带数字标记框的可视化图像（默认保存至 storage.cache_path）\n"
                "  2. run 方法参数：\n"
                "     - storage (DataFlowStorage)：数据存储实例，用于获取可视化图像缓存路径\n"
                "     - input_image_key (str, 默认\"image\")：输入数据中「图像路径」的字段名\n"
                "     - input_bbox_key (str, 默认\"bbox\")：输入数据中「原始边界框」的字段名（无此字段则自动从图像提取）\n"
                "     - output_key (str, 默认\"mdvp_record\")：输出数据中「区域描述记录」的字段名\n"
                "\n"
                "处理流程：\n"
                "  1. 读取输入 JSONL 数据，检查是否包含 bbox 字段\n"
                "  2. 若无 bbox 字段，调用 extract_boxes_from_image 从图像自动提取边界框\n"
                "  3. 对原始 bbox 坐标执行标准化处理，适配 VLM 模型输入格式\n"
                "  4. 生成带数字编号的边界框可视化图像（可选）\n"
                "  5. 构造 VLM 提示词，调用模型生成各区域文本描述\n"
                "  6. 整合原始信息、标准化框、可视化路径、区域描述等结果并输出\n"
                "\n"
                "输出数据结构（output_key 对应字段）：\n"
                "  {\n"
                "    \"image\": \"/path/to/img.jpg\",                     // 原始图像路径\n"
                "    \"mdvp_record\": \"<region1>: 描述1; <region2>: 描述2\", // VLM 生成的各区域文本描述\n"
                "    \"meta_info\": {\n"
                "        \"type\": \"with_bbox/without_bbox\",          // 框来源类型（输入自带/图像提取）\n"
                "        \"bbox\": [[x0,y0,w0,h0], [x1,y1,w1,h1], ...], // 原始未标准化的 bbox 坐标\n"
                "        \"normalized_bbox\": [[0.1,0.2,0.3,0.4], ...], // 标准化后 bbox（补零至 max_boxes 数量）\n"
                "        \"image_with_bbox\": \"/path/to/cache/1_bbox_vis.jpg\" // 带框可视化图像路径\n"
                "    }\n"
                "  }\n"
            )
        else:
            return (
                "Function Description:\n"
                "  Batch generate text descriptions for each marked region in images from JSONL format data containing {\"image\": \"/path/to/img\"} (optionally with {\"bbox\": [[x,y,w,h],...]}).\n"
                "  Supports two bounding box sources: input-provided bboxes / auto-extracted bboxes from images (based on edge detection + contour fitting); core pipeline: bbox processing (extraction/normalization) → boxed visualization generation → VLM prompt construction → region description generation → result integration and output.\n"
                "\n"
                "Dependent Components:\n"
                "  • Bbox Extraction: extract_boxes_from_image (edge detection + contour fitting, filters small-area/abnormal aspect ratio boxes)\n"
                "  • Bbox Normalization: vp_normalize (coordinate normalization to adapt to VLM model input format)\n"
                "  • Visualization: paint_text_box (draw colored bounding boxes with numeric labels)\n"
                "  • VLM Service: LocalModelVLMServing_vllm (process visual input via qwen_vl_utils)\n"
                "\n"
                "Input Parameter Description:\n"
                "  1. Configuration Class (ExistingBBoxDataGenConfig):\n"
                "     - max_boxes (int, default 10): Maximum number of boxes processed per image (truncate excess, pad zeros to align with model input dimensions)\n"
                "     - input_jsonl_path (Optional[str]): Path to input JSONL file (each line must contain \"image\" field, optionally \"bbox\" field)\n"
                "     - output_jsonl_path (Optional[str]): Path to output JSONL file (store complete records with region descriptions)\n"
                "     - draw_visualization (bool, default True): Whether to generate visualization images with numbered boxes (saved to storage.cache_path by default)\n"
                "  2. run Method Parameters:\n"
                "     - storage (DataFlowStorage): Data storage instance for getting visualization cache path\n"
                "     - input_image_key (str, default \"image\"): Field name of \"image path\" in input data\n"
                "     - input_bbox_key (str, default \"bbox\"): Field name of \"raw bounding boxes\" in input data (auto-extract from image if missing)\n"
                "     - output_key (str, default \"mdvp_record\"): Field name of \"region description record\" in output data\n"
                "\n"
                "Processing Pipeline:\n"
                "  1. Read input JSONL data and check for the presence of bbox field\n"
                "  2. If no bbox field, call extract_boxes_from_image to auto-extract bounding boxes from images\n"
                "  3. Normalize raw bbox coordinates to adapt to VLM model input format\n"
                "  4. Generate visualization images with numbered bounding boxes (optional)\n"
                "  5. Construct VLM prompts and call model to generate text descriptions for each region\n"
                "  6. Integrate raw info, normalized boxes, visualization paths, region descriptions and output\n"
                "\n"
                "Output Data Structure (corresponding to output_key):\n"
                "  {\n"
                "    \"image\": \"/path/to/img.jpg\",                     // Original image path\n"
                "    \"mdvp_record\": \"<region1>: description1; <region2>: description2\", // VLM-generated region descriptions\n"
                "    \"meta_info\": {\n"
                "        \"type\": \"with_bbox/without_bbox\",          // Box source type (input-provided/extracted from image)\n"
                "        \"bbox\": [[x0,y0,w0,h0], [x1,y1,w1,h1], ...], // Raw unnormalized bbox coordinates\n"
                "        \"normalized_bbox\": [[0.1,0.2,0.3,0.4], ...], // Normalized bboxes (padded to max_boxes count)\n"
                "        \"image_with_bbox\": \"/path/to/cache/1_bbox_vis.jpg\" // Path to boxed visualization image\n"
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
        # 调用代码库中绘制框标记的工具
        vis_path = os.path.join(vispath , f"{counter}_bbox_vis.jpg")
        paint_text_box(image_path, bboxes,vis_path)  # 绘制带数字标签的框
        return vis_path

    # 生成模型输入prompt（基于框标记的描述指令）
    def _gen_prompt(self, bbox_count: int) -> str:
        return (f"Describe the content of each marked region in the image. "
                f"There are {bbox_count} regions: <region1> to <region{bbox_count}>.")

    def run(self, storage: DataFlowStorage, input_image_key: str = "image", input_bbox_key: str = "bbox", output_key: str = "mdvp_record"):
        rows = []
        if self.cfg.input_jsonl_path:
            with open(self.cfg.input_jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    rows.append(json.loads(line.strip()))
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
            print(record)

        if self.cfg.output_jsonl_path:
            os.makedirs(os.path.dirname(self.cfg.output_jsonl_path), exist_ok=True)
            with open(self.cfg.output_jsonl_path, "w", encoding="utf-8") as f:
                for r in out_records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return [output_key]