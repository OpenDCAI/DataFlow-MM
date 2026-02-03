"""
Video Luminance Filter - combines luminance analysis and filtering.

This operator:
1. Calls VideoLuminanceEvaluator to compute luminance statistics
2. Filters clips based on lum_min/lum_max thresholds
3. Adds/updates 'filtered' field in each clip
4. Preserves all clips (doesn't remove), just marks them as filtered or not

Pipeline integration:
- Reads upstream dataframe with `video_clips` column
- Computes luminance_mean, luminance_min, luminance_max for each clip
- Updates clip["filtered"] based on lum_min/lum_max thresholds
- Writes back updated clips to dataframe
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.operators.core_vision.eval.video_luminance_evaluator import (
    score_luminance_over_dataframe
)


def apply_luminance_filter(
    dataframe: pd.DataFrame,
    video_clips_key: str = "video_clips",
    lum_min: Optional[float] = None,
    lum_max: Optional[float] = None,
) -> pd.DataFrame:
    """
    Apply luminance filtering to clips.
    
    Args:
        dataframe: DataFrame with video_clips column containing luminance_mean
        video_clips_key: Column name for video clips data
        lum_min: Minimum luminance threshold
        lum_max: Maximum luminance threshold
        
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
            
            # Apply luminance filter
            lum_mean = clip.get("luminance_mean")
            
            if lum_min is not None:
                if lum_mean is None:
                    # If score doesn't exist, mark as filtered
                    clip["filtered"] = False
                elif lum_mean < lum_min:
                    clip["filtered"] = False
            
            if lum_max is not None:
                if lum_mean is None:
                    # If score doesn't exist, mark as filtered
                    clip["filtered"] = False
                elif lum_mean > lum_max:
                    clip["filtered"] = False
        
        # Write back updated clips
        df.at[idx, video_clips_key] = cell
    
    return df


@OPERATOR_REGISTRY.register()
class VideoLuminanceFilter(OperatorABC):
    """
    DataFlow operator: compute luminance statistics and filter clips based on thresholds.
    
    This operator combines VideoLuminanceEvaluator's analysis functionality with filtering logic.
    It computes luminance statistics for each clip and marks clips as filtered if they don't meet
    the minimum/maximum luminance thresholds.
    """

    def __init__(
        self,
        figure_root: str = "extract_frames",
        input_video_key: str = "video",
        video_clips_key: Optional[str] = None,
        load_num: int = 3,
        batch_size: int = 64,
        num_workers: int = 4,
        init_distributed: bool = False,
        output_key: str = "video_clips",
        lum_min: Optional[float] = None,
        lum_max: Optional[float] = None,
    ):
        """
        Initialize VideoLuminanceFilter operator.
        
        Args:
            figure_root: Directory where frames were extracted
            input_video_key: Column name for video paths
            video_clips_key: Column name for video clips data
            load_num: Number of frames to load per clip for analysis
            batch_size: Batch size for processing
            num_workers: Number of worker processes for data loading
            init_distributed: Whether to use distributed processing
            output_key: Output column name
            lum_min: Minimum luminance threshold (clips below this are filtered)
            lum_max: Maximum luminance threshold (clips above this are filtered)
        """
        self.logger = get_logger()
        self.figure_root = figure_root
        self.input_video_key = input_video_key
        self.video_clips_key = video_clips_key
        self.load_num = load_num
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.init_distributed = init_distributed
        self.output_key = output_key
        self.lum_min = lum_min
        self.lum_max = lum_max

    @staticmethod
    def get_desc(lang: str = "zh") -> str:
        return "计算视频片段亮度统计并根据阈值过滤" if lang == "zh" else "Compute luminance statistics for video clips and filter based on thresholds."

    def run(
        self,
        storage: DataFlowStorage,
        figure_root: Optional[str] = None,
        input_video_key: Optional[str] = None,
        video_clips_key: Optional[str] = None,
        load_num: Optional[int] = None,
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
        init_distributed: Optional[bool] = None,
        output_key: Optional[str] = None,
        lum_min: Optional[float] = None,
        lum_max: Optional[float] = None,
    ):
        """
        Execute luminance analysis and filtering.
        
        Args:
            storage: DataFlow storage object
            figure_root: Directory where frames were extracted
            input_video_key: Column name for video paths
            video_clips_key: Column name for video clips data
            load_num: Number of frames to load per clip
            batch_size: Batch size for processing
            num_workers: Number of worker processes
            init_distributed: Whether to use distributed processing
            output_key: Output column name
            lum_min: Minimum luminance threshold
            lum_max: Maximum luminance threshold
        """
        # Resolve runtime config
        import os
        figure_root = figure_root or os.path.join(storage.cache_path, self.figure_root)
        input_video_key = input_video_key or self.input_video_key
        video_clips_key = self.video_clips_key if video_clips_key is None else video_clips_key
        load_num = load_num or self.load_num
        batch_size = batch_size or self.batch_size
        num_workers = num_workers or self.num_workers
        init_distributed = self.init_distributed if init_distributed is None else init_distributed
        output_key = output_key or self.output_key
        lum_min = self.lum_min if lum_min is None else lum_min
        lum_max = self.lum_max if lum_max is None else lum_max

        self.logger.info("Running VideoLuminanceFilter...")
        df = storage.read("dataframe")
        self.logger.info(f"Loaded dataframe: {len(df)} rows")

        # Step 1: Compute luminance statistics
        self.logger.info("Computing luminance statistics...")
        scored = score_luminance_over_dataframe(
            dataframe=df,
            figure_root=figure_root,
            input_video_key=input_video_key,
            video_clips_key=video_clips_key,
            load_num=load_num,
            batch_size=batch_size,
            num_workers=num_workers,
            init_distributed=init_distributed,
        )
        self.logger.info("Luminance statistics computed")

        # Step 2: Apply filtering based on lum_min/lum_max
        if video_clips_key is None:
            # Whole-video mode: stats stored at row level (luminance_mean/min/max)
            mean_col = "luminance_mean"
            min_col = "luminance_min"
            max_col = "luminance_max"
            for c in [mean_col, min_col, max_col]:
                if c not in scored.columns:
                    scored[c] = None

            if lum_min is not None or lum_max is not None:
                def _pass(v: Optional[float]) -> bool:
                    if v is None:
                        return False
                    try:
                        vv = float(v)
                    except Exception:
                        return False
                    if lum_min is not None and vv < float(lum_min):
                        return False
                    if lum_max is not None and vv > float(lum_max):
                        return False
                    return True

                scored[output_key] = scored.apply(
                    lambda r: {
                        "luminance_mean": r.get(mean_col),
                        "luminance_min": r.get(min_col),
                        "luminance_max": r.get(max_col),
                        "filtered": _pass(r.get(mean_col)),
                    },
                    axis=1,
                )
            else:
                scored[output_key] = scored.apply(
                    lambda r: {
                        "luminance_mean": r.get(mean_col),
                        "luminance_min": r.get(min_col),
                        "luminance_max": r.get(max_col),
                    },
                    axis=1,
                )
            filtered = scored
        else:
            if lum_min is not None or lum_max is not None:
                self.logger.info(f"Applying luminance filter (lum_min={lum_min}, lum_max={lum_max})...")
                filtered = apply_luminance_filter(
                    dataframe=scored,
                    video_clips_key=video_clips_key,
                    lum_min=lum_min,
                    lum_max=lum_max,
                )
                self.logger.info("Luminance filtering complete")
            else:
                # No filtering, but ensure filtered field exists
                self.logger.info("No luminance threshold specified, initializing filtered field...")
                filtered = apply_luminance_filter(
                    dataframe=scored,
                    video_clips_key=video_clips_key,
                    lum_min=None,
                    lum_max=None,
                )

        # Write back
        if video_clips_key is not None and output_key != video_clips_key:
            filtered[output_key] = filtered[video_clips_key]

        storage.write(filtered)
        self.logger.info(f"VideoLuminanceFilter complete. Output column: '{output_key}'")

