from __future__ import annotations

from typing import Any, Dict, List, Optional
from collections import defaultdict

import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC


def _timecode_to_seconds(timecode) -> float:
    """Convert timecode to seconds. Accepts integer seconds or 'HH:MM:SS.mmm' string format."""
    if isinstance(timecode, (int, float)):
        return float(timecode)
    try:
        h, m, s = timecode.split(":")
        return float(h) * 3600.0 + float(m) * 60.0 + float(s)
    except Exception:
        return 0.0


def _extract_video_name(clip_id: str) -> str:
    """
    Extract original video name from clip ID.
    Example: "trailer_0" -> "trailer", "video_name_5" -> "video_name"
    """
    if "_" not in clip_id:
        return clip_id
    parts = clip_id.split("_")
    # Check if last part is a number (clip index)
    try:
        int(parts[-1])
        return "_".join(parts[:-1])
    except ValueError:
        return clip_id


def merge_video_captions(
    dataframe: pd.DataFrame,
    caption_key: str = "caption",
    id_key: str = "id",
    timestamp_start_key: str = "timestamp_start",
    timestamp_end_key: str = "timestamp_end",
    duration_key: str = "duration_sec",
) -> List[Dict[str, Any]]:
    """
    Merge captions from multiple clips into text format for LLM input.
    
    Args:
        dataframe: DataFrame containing clip information and captions
        caption_key: Column name for caption text
        id_key: Column name for clip ID
        timestamp_start_key: Column name for start timestamp
        timestamp_end_key: Column name for end timestamp  
        duration_key: Column name for clip duration
        
    Returns:
        List of dicts with id, captions (text format), num_clips
    """
    video_captions = defaultdict(list)
    logger = get_logger()
    
    for _, row in dataframe.iterrows():
        clip_id = row.get(id_key, "")
        caption_text = row.get(caption_key, "")
        
        # Extract original video name
        video_name = _extract_video_name(clip_id)
        
        # Get timestamps - require them to be present
        if timestamp_start_key not in row or row[timestamp_start_key] is None:
            logger.warning(f"Clip {clip_id}: Missing {timestamp_start_key}, skipping")
            continue
        
        start_time = _timecode_to_seconds(row[timestamp_start_key])
        
        # Get end time from timestamp or duration
        if timestamp_end_key in row and row[timestamp_end_key] is not None:
            end_time = _timecode_to_seconds(row[timestamp_end_key])
        elif duration_key in row and row[duration_key]:
            end_time = start_time + float(row[duration_key])
        else:
            logger.warning(f"Clip {clip_id}: Missing {timestamp_end_key} and {duration_key}, skipping")
            continue
        
        # Clean caption text
        clean_caption = caption_text.replace('\n', ' ').strip() if caption_text else ""
        if not clean_caption:
            continue
        
        video_captions[video_name].append({
            "start_time": start_time,
            "end_time": end_time,
            "caption": clean_caption,
        })
    
    # Format results
    merged_data = []
    for video_name, captions in video_captions.items():
        # Sort by start_time
        captions.sort(key=lambda x: x["start_time"])
        
        # Format as text for LLM input
        captions_text = ""
        for item in captions:
            caption = item['caption']
            # Lowercase first character (matching step4 logic)
            caption = caption[0].lower() + caption[1:] if len(caption) > 0 else ""
            # Round timestamps to integer seconds for readability
            start_sec = int(item['start_time'])
            end_sec = int(item['end_time'])
            captions_text += f"From {start_sec} to {end_sec}, {caption}\n"
        
        merged_data.append({
            "id": video_name,
            "captions": captions_text.strip(),
            "num_clips": len(captions),
        })
    
    return merged_data


@OPERATOR_REGISTRY.register()
class VideoMergedCaptionGenerator(OperatorABC):
    """
    DataFlow operator: generate merged caption structures from multiple clips.
    
    Input: DataFrame with individual clip captions
    Output: Updated DataFrame in storage with merged captions (one row per video)
    """
    
    def __init__(
        self,
        caption_key: str = "caption",
        id_key: str = "id",
        timestamp_start_key: str = "timestamp_start",
        timestamp_end_key: str = "timestamp_end",
        duration_key: str = "duration_sec",
    ):
        """
        Initialize VideoMergedCaptionGenerator operator.
        
        Args:
            caption_key: Column name for caption text
            id_key: Column name for clip ID
            timestamp_start_key: Column name for start timestamp
            timestamp_end_key: Column name for end timestamp
            duration_key: Column name for clip duration
        """
        self.logger = get_logger()
        self.caption_key = caption_key
        self.id_key = id_key
        self.timestamp_start_key = timestamp_start_key
        self.timestamp_end_key = timestamp_end_key
        self.duration_key = duration_key
    
    @staticmethod
    def get_desc(lang: str = "zh") -> str:
        if lang == "zh":
            return (
                "该算子将同一视频的多个片段字幕合并为LLM可直接使用的文本格式。\n\n"
                "输入要求：\n"
                "  - caption_key: 字幕字段名 (默认: 'caption')\n"
                "  - 必须有 timestamp_start 和 timestamp_end 字段\n"
                "输出格式：\n"
                "  - 'From X to Y, caption...' 格式的文本\n"
                "功能特点：\n"
                "  - 按原视频分组片段字幕\n"
                "  - 按时间顺序排序\n"
                "  - 直接可用作LLM输入\n"
            )
        elif lang == "en":
            return (
                "This operator merges captions from clips into LLM-ready text format.\n\n"
                "Input Requirements:\n"
                "  - caption_key: Caption field name (default: 'caption')\n"
                "  - Must have timestamp_start and timestamp_end fields\n"
                "Output Format:\n"
                "  - Text format: 'From X to Y, caption...'\n"
                "Features:\n"
                "  - Groups clips by original video\n"
                "  - Sorts by time order\n"
                "  - Ready for LLM input\n"
            )
        else:
            return "VideoMergedCaptionGenerator merges captions into LLM-ready text format."
    
    def run(
        self,
        storage: DataFlowStorage,
        caption_key: Optional[str] = None,
        id_key: Optional[str] = None,
    ):
        """
        Execute merged caption generation.
        
        Args:
            storage: DataFlow storage object
            caption_key: Override caption field name
            id_key: Override clip ID field name
        """
        caption_key = caption_key or self.caption_key
        id_key = id_key or self.id_key
        
        self.logger.info("Running VideoMergedCaptionGenerator...")
        df = storage.read("dataframe")
        self.logger.info(f"Loaded dataframe: {len(df)} clips")
        
        # Merge captions into text format
        merged_captions = merge_video_captions(
            dataframe=df,
            caption_key=caption_key,
            id_key=id_key,
            timestamp_start_key=self.timestamp_start_key,
            timestamp_end_key=self.timestamp_end_key,
            duration_key=self.duration_key,
        )
        
        # Write merged captions to storage for downstream operators
        # Convert to DataFrame for consistency with other operators
        merged_df = pd.DataFrame(merged_captions)
        storage.write(merged_df)
        
        self.logger.info(f"✓ Merged captions written to storage")
        self.logger.info(f"  Total videos: {len(merged_captions)}")
        for video_data in merged_captions:
            self.logger.info(
                f"  - {video_data['id']}: {video_data['num_clips']} clips"
            )
        
        self.logger.info("Merged caption generation complete.")

