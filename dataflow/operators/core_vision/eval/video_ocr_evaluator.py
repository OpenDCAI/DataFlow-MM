import os
import queue
import concurrent.futures
from typing import Any, Dict, List, Optional
import paddleocr
import torch
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from tqdm import tqdm
from PIL import Image
import cv2
import pandas as pd

# ----------------------------
# OCR Processing Logic
# ----------------------------

def process_single_row(row, figure_root, model) -> float:
    """Process a single row to calculate OCR text area ratio."""
    video_path = row["video"]
    
    # Generate the clip_id based on video_path and clip_idx
    clip_id = f"{_video_stem(video_path)}_{row['clip_idx']}"  # 假设 `clip_idx` 是从数据中的 `video_scene` 提供的
    
    img_dir = os.path.join(figure_root, os.path.splitext(os.path.basename(video_path))[0], clip_id, "img")
    img_list = sorted(glob.glob(f"{img_dir}/*.jpg"))[:3]  # Limit to first 3 images

    if not img_list:
        return 0.0  # If no images found, return 0

    # Load images and perform OCR
    images = [cv2.imread(img_path) for img_path in img_list]
    result = model.predict(input=images)

    area = row["height"] * row["width"]
    area_list = []

    for res in result:
        total_text_area = 0  # Initialize total text area
        for rec_box in res["rec_boxes"]:
            x_min, y_min, x_max, y_max = (
                float(rec_box[0]),
                float(rec_box[1]),
                float(rec_box[2]),
                float(rec_box[3]),
            )  # Extract top-left and bottom-right coordinates
            text_area = (x_max - x_min) * (y_max - y_min)  # Calculate text area
            total_text_area += text_area
        ratio = total_text_area / area
        area_list.append(ratio)

    return max(area_list) if area_list else 0.0


def worker(task_queue, result_queue, args, id):
    """Worker function for OCR inference with PaddleOCR."""
    gpu_id = id % args.gpu_num
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  # Bind to specific GPU

    model = paddleocr.PaddleOCR(
        device="gpu" if gpu_id >= 0 else "cpu",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    while True:
        try:
            index, row = task_queue.get_nowait()
        except queue.Empty:
            break

        ocr_score = process_single_row(row, args, model)
        result_queue.put((index, ocr_score))


def run_ocr_inference(
    dataframe: pd.DataFrame,
    figure_root: str,
    input_video_key: str = "video",
    video_clips_key: str = "video_clips",
    load_num: int = 3,
    batch_size: int = 64,
    num_workers: int = 4,
    gpu_num: int = 1,
    disable_parallel: bool = False,
    skip_if_existing: bool = False
) -> pd.DataFrame:
    """
    Compute OCR scores and inject them into video clips in the dataframe.
    """
    results = []

    # Single-process processing
    if disable_parallel:
        model = paddleocr.PaddleOCR(
            device="gpu:0" if gpu_num > 0 else "cpu",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

        for index, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc="OCR Processing (Serial)"):
            ocr_score = process_single_row(row=row, figure_root=figure_root, model=model)
            results.append((index, ocr_score))

    # Multi-process processing
    else:
        manager = Manager()
        task_queue = manager.Queue()
        result_queue = manager.Queue()

        # Populate the task queue
        for index, row in dataframe.iterrows():
            task_queue.put((index, row, figure_root, gpu_num))

        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for worker_id in range(num_workers):
                futures.append(
                    executor.submit(
                        worker,
                        task_queue=task_queue,
                        result_queue=result_queue,
                        figure_root=figure_root,
                        gpu_num=gpu_num,
                        worker_id=worker_id
                    )
                )

            # Wait for all workers to finish
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="OCR Workers Finishing"
            ):
                try:
                    future.result()
                except Exception as e:
                    print(f"Worker {future} failed with error: {str(e)}")

        # Collect multi-process results
        while not result_queue.empty():
            index, ocr_score = result_queue.get()
            results.append((index, ocr_score))

    # Inject OCR scores into the dataframe
    dataframe_with_ocr = dataframe.copy()
    dataframe_with_ocr["ocr_score"] = [score for _, score in results]

    # Insert OCR scores into video clips dictionary
    for idx, ocr_score in results:
        video_clip_data = dataframe_with_ocr.at[idx, video_clips_key]
        if isinstance(video_clip_data, dict) and "clips" in video_clip_data:
            for clip in video_clip_data["clips"]:
                clip["ocr_score"] = ocr_score

    return dataframe_with_ocr


# ----------------------------
# DataFlow Operator
# ----------------------------

@OPERATOR_REGISTRY.register()
class VideoOCREvaluator(OperatorABC):
    """
    OCR analysis operator: Computes text area ratios for video clips using PaddleOCR.
    """

    def __init__(self,
                 figure_root: str = "img",
                 num_workers: int = 16,
                 gpu_num: int = 1,
                 disable_parallel: bool = True,
                 skip_if_existing: bool = False):
        """
        Initialize OCR analysis operator
        
        Args:
            figure_root: Image directory
            num_workers: Number of workers
            gpu_num: GPU number for workers
            disable_parallel: Disable multi-processing
            skip_if_existing: Skip if output exists
        """
        self.logger = get_logger()
        self.figure_root = figure_root
        self.num_workers = num_workers
        self.gpu_num = gpu_num
        self.disable_parallel = disable_parallel
        self.skip_if_existing = skip_if_existing

    @staticmethod
    def get_desc(lang: str = "zh"):
        """Operator description"""
        if lang == "zh":
            return "基于 PaddleOCR 的视频OCR分析"
        else:
            return "Video OCR analysis based on PaddleOCR"

    def run(self,
            storage: DataFlowStorage,
            input_video_key: str = "video",
            video_clips_key: str = "video_clips",  
            output_key: str = "video_clips",  
            ):
        """
        Execute OCR analysis (main interface for DataFlowStorage)
        
        Args:
            storage: DataFlowStorage object for data read/write
            csv_path: Input DataFrame Key in Storage
            output_csv_key: Output results Key in Storage
            out_csv_path: Optional, path to save the CSV
        
        Returns:
            str: Output key in storage
        """
        self.logger.info("Starting OCRAnalyzer...")

        # Read input data
        df = storage.read("dataframe")
        self.logger.info(f"Loaded dataframe: {len(df)} rows")

        # Run OCR inference
        scored = run_ocr_inference(
            dataframe=df,
            figure_root=self.figure_root,
            num_workers=self.num_workers,
            gpu_num=self.gpu_num,
            disable_parallel=self.disable_parallel,
            skip_if_existing=self.skip_if_existing
        )

        # If caller wants a different column name to hold updated clips, mirror it.
        if output_key != video_clips_key:
            scored[output_key] = scored[video_clips_key]

        storage.write(scored)
        self.logger.info(f"Luminance scoring complete. Stats injected into column: '{video_clips_key}'")
