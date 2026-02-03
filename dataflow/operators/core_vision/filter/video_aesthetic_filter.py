"""
Video Aesthetic Filter - combines scoring and filtering based on aesthetic quality.

This operator:
1. Calls VideoAestheticEvaluator to compute aesthetic scores
2. Filters clips based on aes_min threshold
3. Adds/updates 'filtered' field in each clip
4. Preserves all clips (doesn't remove), just marks them as filtered or not

Pipeline integration:
- Reads upstream dataframe with `video_clips` column
- Computes aesthetic_score for each clip
- Updates clip["filtered"] based on aes_min threshold
- Writes back updated clips to dataframe
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.operators.core_vision.eval.video_aesthetic_evaluator import (
    score_aesthetics_over_dataframe
)


def apply_aesthetic_filter(
    dataframe: pd.DataFrame,
    video_clips_key: str = "video_clips",
    aes_min: Optional[float] = None,
) -> pd.DataFrame:
    """
    Apply aesthetic score filtering to clips.
    
    Args:
        dataframe: DataFrame with video_clips column containing aesthetic_score
        video_clips_key: Column name for video clips data
        aes_min: Minimum aesthetic score threshold
        
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
            
            # Apply aesthetic filter
            if aes_min is not None:
                aes_score = clip.get("aesthetic_score")
                if aes_score is None:
                    # If score doesn't exist, mark as filtered
                    clip["filtered"] = False
                elif aes_score < aes_min:
                    clip["filtered"] = False
        
        # Write back updated clips
        df.at[idx, video_clips_key] = cell
    
    return df


@OPERATOR_REGISTRY.register()
class VideoAestheticFilter(OperatorABC):
    """
    DataFlow operator: compute aesthetic scores and filter clips based on threshold.
    
    This operator combines VideoAestheticEvaluator's scoring functionality with filtering logic.
    It computes aesthetic scores for each clip and marks clips as filtered if they don't meet
    the minimum aesthetic score threshold.
    """

    def __init__(
        self,
        figure_root: str = "extract_frames",
        input_video_key: str = "video",
        video_clips_key: Optional[str] = None,
        clip_model: str = "ViT-L/14",
        mlp_checkpoint: Optional[str] = None,
        load_num: int = 3,
        batch_size: int = 64,
        num_workers: int = 4,
        init_distributed: bool = False,
        output_key: str = "video_clips",
        aes_min: Optional[float] = None,
    ):
        """
        Initialize VideoAestheticFilter operator.
        
        Args:
            figure_root: Directory where frames were extracted
            input_video_key: Column name for video paths
            video_clips_key: Column name for video clips data
            clip_model: CLIP model name or path
            mlp_checkpoint: Path to MLP checkpoint for aesthetic prediction
            load_num: Number of frames to load per clip for scoring
            batch_size: Batch size for processing
            num_workers: Number of worker processes for data loading
            init_distributed: Whether to use distributed processing
            output_key: Output column name
            aes_min: Minimum aesthetic score threshold (clips below this are filtered)
        """
        self.logger = get_logger()
        self.figure_root = figure_root
        self.input_video_key = input_video_key
        self.video_clips_key = video_clips_key
        self.clip_model = clip_model
        self.mlp_checkpoint = mlp_checkpoint
        self.load_num = load_num
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.init_distributed = init_distributed
        self.output_key = output_key
        self.aes_min = aes_min

    @staticmethod
    def get_desc(lang: str = "zh") -> str:
        return "计算视频片段美学分数并根据阈值过滤" if lang == "zh" else "Compute aesthetic scores for video clips and filter based on threshold."

    def run(
        self,
        storage: DataFlowStorage,
        figure_root: Optional[str] = None,
        input_video_key: Optional[str] = None,
        video_clips_key: Optional[str] = None,
        clip_model: Optional[str] = None,
        mlp_checkpoint: Optional[str] = None,
        load_num: Optional[int] = None,
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
        init_distributed: Optional[bool] = None,
        output_key: Optional[str] = None,
        aes_min: Optional[float] = None,
    ):
        """
        Execute aesthetic scoring and filtering.
        
        Args:
            storage: DataFlow storage object
            figure_root: Directory where frames were extracted
            input_video_key: Column name for video paths
            video_clips_key: Column name for video clips data
            clip_model: CLIP model name or path
            mlp_checkpoint: Path to MLP checkpoint
            load_num: Number of frames to load per clip
            batch_size: Batch size for processing
            num_workers: Number of worker processes
            init_distributed: Whether to use distributed processing
            output_key: Output column name
            aes_min: Minimum aesthetic score threshold
        """
        # Resolve runtime config
        import os
        figure_root = figure_root or os.path.join(storage.cache_path, self.figure_root)
        input_video_key = input_video_key or self.input_video_key
        video_clips_key = self.video_clips_key if video_clips_key is None else video_clips_key
        clip_model = clip_model or self.clip_model
        mlp_checkpoint = mlp_checkpoint or self.mlp_checkpoint
        load_num = load_num or self.load_num
        batch_size = batch_size or self.batch_size
        num_workers = num_workers or self.num_workers
        init_distributed = self.init_distributed if init_distributed is None else init_distributed
        output_key = output_key or self.output_key
        aes_min = self.aes_min if aes_min is None else aes_min

        self.logger.info("Running VideoAestheticFilter...")
        df = storage.read("dataframe")
        self.logger.info(f"Loaded dataframe: {len(df)} rows")

        # Step 1: Compute aesthetic scores
        self.logger.info("Computing aesthetic scores...")
        scored = score_aesthetics_over_dataframe(
            dataframe=df,
            figure_root=figure_root,
            input_video_key=input_video_key,
            video_clips_key=video_clips_key,
            clip_model=clip_model,
            mlp_checkpoint=mlp_checkpoint,
            load_num=load_num,
            batch_size=batch_size,
            num_workers=num_workers,
            init_distributed=init_distributed,
        )
        self.logger.info("Aesthetic scores computed")

        # Step 2: Apply filtering based on aes_min
        if video_clips_key is None:
            # Whole-video mode: score stored at row level (`aesthetic_score`)
            score_col = "aesthetic_score"
            if score_col not in scored.columns:
                scored[score_col] = None

            if aes_min is not None:
                def _pass(v: Optional[float]) -> bool:
                    if v is None:
                        return False
                    try:
                        vv = float(v)
                    except Exception:
                        return False
                    return vv >= float(aes_min)

                scored[output_key] = scored[score_col].apply(lambda v: {"aesthetic_score": v, "filtered": _pass(v)})
            else:
                scored[output_key] = scored[score_col].apply(lambda v: {"aesthetic_score": v})
            filtered = scored
        else:
            if aes_min is not None:
                self.logger.info(f"Applying aesthetic filter (aes_min={aes_min})...")
                filtered = apply_aesthetic_filter(
                    dataframe=scored,
                    video_clips_key=video_clips_key,
                    aes_min=aes_min,
                )
                self.logger.info("Aesthetic filtering complete")
            else:
                # No filtering, but ensure filtered field exists
                self.logger.info("No aesthetic threshold specified, initializing filtered field...")
                filtered = apply_aesthetic_filter(
                    dataframe=scored,
                    video_clips_key=video_clips_key,
                    aes_min=None,
                )

        # Write back
        if video_clips_key is not None and output_key != video_clips_key:
            filtered[output_key] = filtered[video_clips_key]

        storage.write(filtered)
        self.logger.info(f"VideoAestheticFilter complete. Output column: '{output_key}'")

