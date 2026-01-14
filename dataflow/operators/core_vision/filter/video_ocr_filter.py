"""
Video OCR Filter - combines OCR analysis and filtering.

This operator:
1. Calls VideoOCREvaluator to compute OCR text ratio
2. Filters clips based on ocr_min/ocr_max thresholds
3. Adds/updates 'filtered' field in each clip
4. Preserves all clips (doesn't remove), just marks them as filtered or not

Pipeline integration:
- Reads upstream dataframe with `video_clips` column
- Computes ocr_score (text area ratio) for each clip
- Updates clip["filtered"] based on ocr_min/ocr_max thresholds
- Writes back updated clips to dataframe
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.operators.core_vision.eval.video_ocr_evaluator import (
    score_ocr_over_dataframe
)


def apply_ocr_filter(
    dataframe: pd.DataFrame,
    video_clips_key: str = "video_clips",
    ocr_min: Optional[float] = None,
    ocr_max: Optional[float] = None,
) -> pd.DataFrame:
    """
    Apply OCR score filtering to clips.
    
    Args:
        dataframe: DataFrame with video_clips column containing ocr_score
        video_clips_key: Column name for video clips data
        ocr_min: Minimum OCR score threshold
        ocr_max: Maximum OCR score threshold
        
    Returns:
        DataFrame with updated filtered field in clips
    """
    if video_clips_key not in dataframe.columns:
        raise KeyError(f"Column '{video_clips_key}' not found in dataframe.")
    
    df = dataframe.copy()
    
    for idx, row in df.iterrows():
        cell = row.get(video_clips_key)
        if not isinstance(cell, dict) or "clips" not in cell:
            continue
        
        clips = cell.get("clips", [])
        if not isinstance(clips, list):
            continue
        
        for clip in clips:
            if not isinstance(clip, dict):
                continue
            
            # Initialize filtered field if not exists (default to True = pass)
            if "filtered" not in clip:
                clip["filtered"] = True
            
            # Skip if already filtered by previous operators
            if clip.get("filtered") is False:
                continue
            
            # Apply OCR filter
            ocr_score = clip.get("ocr_score")
            
            if ocr_min is not None:
                if ocr_score is None:
                    # If score doesn't exist, mark as filtered
                    clip["filtered"] = False
                elif ocr_score < ocr_min:
                    clip["filtered"] = False
            
            if ocr_max is not None:
                if ocr_score is None:
                    # If score doesn't exist, mark as filtered
                    clip["filtered"] = False
                elif ocr_score > ocr_max:
                    clip["filtered"] = False
        
        # Write back updated clips
        df.at[idx, video_clips_key] = cell
    
    return df


@OPERATOR_REGISTRY.register()
class VideoOCRFilter(OperatorABC):
    """
    DataFlow operator: compute OCR scores and filter clips based on thresholds.
    
    This operator combines VideoOCREvaluator's analysis functionality with filtering logic.
    It computes OCR text area ratio for each clip and marks clips as filtered if they don't meet
    the minimum/maximum OCR score thresholds.
    """

    def __init__(
        self,
        figure_root: str = "extract_frames",
        input_video_key: str = "video",
        video_clips_key: str = "video_clips",
        load_num: int = 3,
        batch_size: int = 8,
        num_workers: int = 4,
        gpu_num: int = 0,
        init_distributed: bool = False,
        output_key: str = "video_clips",
        det_model_dir: str = None,
        rec_model_dir: str = None,
        ocr_min: Optional[float] = None,
        ocr_max: Optional[float] = None,
    ):
        """
        Initialize VideoOCRFilter operator.
        
        Args:
            figure_root: Directory where frames were extracted
            input_video_key: Column name for video paths
            video_clips_key: Column name for video clips data
            load_num: Number of frames to load per clip for analysis
            batch_size: Batch size for processing
            num_workers: Number of worker processes for data loading
            gpu_num: GPU ID (0+ for GPU, -1 for CPU)
            init_distributed: Whether to use distributed processing
            output_key: Output column name
            det_model_dir: Path to PaddleOCR detection model directory
            rec_model_dir: Path to PaddleOCR recognition model directory
            ocr_min: Minimum OCR score threshold (clips below this are filtered)
            ocr_max: Maximum OCR score threshold (clips above this are filtered)
        """
        self.logger = get_logger()
        self.figure_root = figure_root
        self.input_video_key = input_video_key
        self.video_clips_key = video_clips_key
        self.load_num = load_num
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.gpu_num = gpu_num
        self.init_distributed = init_distributed
        self.output_key = output_key
        self.det_model_dir = det_model_dir
        self.rec_model_dir = rec_model_dir
        self.ocr_min = ocr_min
        self.ocr_max = ocr_max

    @staticmethod
    def get_desc(lang: str = "zh") -> str:
        return "计算视频片段OCR文字比例并根据阈值过滤" if lang == "zh" else "Compute OCR text ratio for video clips and filter based on thresholds."

    def run(
        self,
        storage: DataFlowStorage,
        figure_root: Optional[str] = None,
        input_video_key: Optional[str] = None,
        video_clips_key: Optional[str] = None,
        load_num: Optional[int] = None,
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
        gpu_num: Optional[int] = None,
        init_distributed: Optional[bool] = None,
        output_key: Optional[str] = None,
        det_model_dir: Optional[str] = None,
        rec_model_dir: Optional[str] = None,
        ocr_min: Optional[float] = None,
        ocr_max: Optional[float] = None,
    ):
        """
        Execute OCR analysis and filtering.
        
        Args:
            storage: DataFlow storage object
            figure_root: Directory where frames were extracted
            input_video_key: Column name for video paths
            video_clips_key: Column name for video clips data
            load_num: Number of frames to load per clip
            batch_size: Batch size for processing
            num_workers: Number of worker processes
            gpu_num: GPU ID
            init_distributed: Whether to use distributed processing
            output_key: Output column name
            det_model_dir: Path to PaddleOCR detection model directory
            rec_model_dir: Path to PaddleOCR recognition model directory
            ocr_min: Minimum OCR score threshold
            ocr_max: Maximum OCR score threshold
        """
        # Resolve runtime config
        figure_root = figure_root or self.figure_root
        input_video_key = input_video_key or self.input_video_key
        video_clips_key = video_clips_key or self.video_clips_key
        load_num = load_num or self.load_num
        batch_size = batch_size or self.batch_size
        num_workers = num_workers or self.num_workers
        gpu_num = self.gpu_num if gpu_num is None else gpu_num
        init_distributed = self.init_distributed if init_distributed is None else init_distributed
        output_key = output_key or self.output_key
        det_model_dir = det_model_dir or self.det_model_dir
        rec_model_dir = rec_model_dir or self.rec_model_dir
        ocr_min = self.ocr_min if ocr_min is None else ocr_min
        ocr_max = self.ocr_max if ocr_max is None else ocr_max

        self.logger.info("Running VideoOCRFilter...")
        df = storage.read("dataframe")
        self.logger.info(f"Loaded dataframe: {len(df)} rows")

        # Step 1: Compute OCR scores
        self.logger.info("Computing OCR scores...")
        scored = score_ocr_over_dataframe(
            dataframe=df,
            figure_root=figure_root,
            input_video_key=input_video_key,
            video_clips_key=video_clips_key,
            load_num=load_num,
            batch_size=batch_size,
            num_workers=num_workers,
            gpu_num=gpu_num,
            init_distributed=init_distributed,
            det_model_dir=det_model_dir,
            rec_model_dir=rec_model_dir,
        )
        self.logger.info("OCR scores computed")

        # Step 2: Apply filtering based on ocr_min/ocr_max
        if ocr_min is not None or ocr_max is not None:
            self.logger.info(f"Applying OCR filter (ocr_min={ocr_min}, ocr_max={ocr_max})...")
            filtered = apply_ocr_filter(
                dataframe=scored,
                video_clips_key=video_clips_key,
                ocr_min=ocr_min,
                ocr_max=ocr_max,
            )
            self.logger.info("OCR filtering complete")
        else:
            # No filtering, but ensure filtered field exists
            self.logger.info("No OCR threshold specified, initializing filtered field...")
            filtered = apply_ocr_filter(
                dataframe=scored,
                video_clips_key=video_clips_key,
                ocr_min=None,
                ocr_max=None,
            )

        # Write back
        if output_key != video_clips_key:
            filtered[output_key] = filtered[video_clips_key]

        storage.write(filtered)
        self.logger.info(f"VideoOCRFilter complete. Output column: '{output_key}'")

