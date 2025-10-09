"""
Video clip information extraction with clean engineering structure.

Reads inputs from prior pipeline dataframe:
- `video` (str or [str]): video path
- `video_info` (dict): expects keys like {height, width, fps}
- `video_scene` (dict): expects {"scenes": [{"start","end","start_frame","end_frame","duration_sec"}, ...], "fps": optional}

Outputs:
- Adds column `video_clips` with {"success": bool, "error": Optional[str], "clips": [ ... ]}

Design:
- Safe parallel execution via ProcessPoolExecutor.map
- Robust per-row error handling (no global failure)
- English comments & type hints
"""

from __future__ import annotations

import os
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from tqdm import tqdm

from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC


# ----------------------------
# Helpers
# ----------------------------

def _aspect_ratio(width: Optional[int], height: Optional[int]) -> Optional[float]:
    if not width or not height or width <= 0 or height <= 0:
        return None
    return float(width) / float(height)

def _resolution_str(width: Optional[int], height: Optional[int]) -> Optional[str]:
    if not width or not height:
        return None
    return f"{int(width)}x{int(height)}"

def _to_secs(tc: str) -> Optional[float]:
    """Parse 'HH:MM:SS.mmm' -> seconds; return None if invalid."""
    try:
        h, m, s = tc.split(":")
        return float(h) * 3600.0 + float(m) * 60.0 + float(s)
    except Exception:
        return None

def _to_frame_idx(sec: Optional[float], fps: Optional[float]) -> Optional[int]:
    if sec is None or not fps or fps <= 0:
        return None
    return int(math.floor(sec * float(fps)))


# ----------------------------
# Job model & worker
# ----------------------------

@dataclass(frozen=True)
class _ClipJob:
    video_path: Optional[str]
    height: Optional[int]
    width: Optional[int]
    fps: Optional[float]
    id_ori: Optional[str]
    scenes: List[Dict[str, Any]]  # each: {"start","end","start_frame","end_frame","duration_sec"}

def _make_clips_for_job(job: _ClipJob) -> Dict[str, Any]:
    """
    Build per-row clip entries from scene boundaries and video_info.
    Returns {"success": bool, "error": Optional[str], "clips": [clip dicts]}
    """
    # Validate path existence but don't hard fail; still produce clip metrics if possible.
    if not job.video_path:
        return {"success": False, "error": "missing_video_path", "clips": []}

    # Dimensions / fps
    H, W = job.height, job.width
    fps = job.fps
    ar = _aspect_ratio(W, H)
    res = _resolution_str(W, H)

    clips: List[Dict[str, Any]] = []
    try:
        fname = os.path.basename(str(job.video_path))
        stem = os.path.splitext(fname)[0]

        for idx, sc in enumerate(job.scenes):
            start_tc = sc.get("start")
            end_tc = sc.get("end")
            # Prefer provided start/end_frame if valid; else compute from timecodes + fps.
            sf = sc.get("start_frame")
            ef = sc.get("end_frame")

            if sf is None or ef is None:
                # Compute from tc if possible
                s_sec = _to_secs(start_tc) if isinstance(start_tc, str) else None
                e_sec = _to_secs(end_tc) if isinstance(end_tc, str) else None
                sf = _to_frame_idx(s_sec, fps)
                ef = _to_frame_idx(e_sec, fps)

            # Duration/frames
            if isinstance(sf, int) and isinstance(ef, int) and ef >= sf:
                num_frames = ef - sf
            else:
                # Fallback to duration in seconds (if present) * fps
                dur_sec = sc.get("duration_sec")
                num_frames = int(dur_sec * fps) if dur_sec and fps and fps > 0 else None

            clip_id = f"{stem}_{idx}"
            clips.append({
                "video_path": job.video_path,
                "id": clip_id,
                "num_frames": num_frames,
                "height": H,
                "width": W,
                "aspect_ratio": ar,
                "fps": fps,
                "resolution": res,
                "timestamp_start": start_tc,
                "timestamp_end": end_tc,
                "frame_start": sf,
                "frame_end": ef,
                "duration_sec": sc.get("duration_sec"),
                "id_ori": job.id_ori,
            })

        return {"success": True, "error": None, "clips": clips}

    except Exception as e:
        return {"success": False, "error": f"build_failed: {e}", "clips": []}


# ----------------------------
# Public API
# ----------------------------

def extract_video_clips_dataframe(
    dataframe: pd.DataFrame,
    input_video_key: str = "video",
    video_info_key: str = "video_info",
    video_scene_key: str = "video_scene",
    output_key: str = "video_clips",
    drop_invalid_timestamps: bool = False,
    disable_parallel: bool = False,
    num_workers: Optional[int] = None,
) -> pd.DataFrame:
    """
    Build clip metadata per row from prior pipeline columns.

    Source columns:
      - `input_video_key`: str or [str]
      - `video_info_key`: dict with (height,width,fps)
      - `video_scene_key`: dict with `scenes` list:
          each scene dict should contain at least {"start","end"}; optionally "start_frame","end_frame","duration_sec".

    Args:
      drop_invalid_timestamps: if True, drop rows where scenes are missing or malformed.
      disable_parallel / num_workers: control parallel execution.

    Returns:
      A new DataFrame with an added/updated `output_key` column shaped like:
        {"success": bool, "error": Optional[str], "clips": [ {clip fields...}, ... ]}
    """
    for col in [input_video_key, video_info_key, video_scene_key]:
        if col not in dataframe.columns:
            raise KeyError(f"Column '{col}' not found in dataframe.")

    df = dataframe.copy()
    df[input_video_key] = df[input_video_key].astype(object)

    def _get_path(cell: Any) -> Optional[str]:
        if isinstance(cell, (list, tuple)) and cell:
            return str(cell[0])
        if isinstance(cell, str):
            return cell
        return None

    def _get_hwf(cell: Any) -> Tuple[Optional[int], Optional[int], Optional[float]]:
        if not isinstance(cell, dict):
            return None, None, None
        h = cell.get("height")
        w = cell.get("width")
        fps = cell.get("fps")
        try:
            h = int(h) if h is not None else None
        except Exception:
            h = None
        try:
            w = int(w) if w is not None else None
        except Exception:
            w = None
        try:
            fps = float(fps) if fps is not None and float(fps) > 0 else None
        except Exception:
            fps = None
        return h, w, fps

    def _get_scenes(cell: Any) -> List[Dict[str, Any]]:
        """
        Normalize scenes list:
          - Prefer dict format from `video_scene['scenes']`
          - If cell is malformed or empty, return []
        """
        if isinstance(cell, dict) and isinstance(cell.get("scenes"), list):
            # Ensure each scene has start/end as strings
            out: List[Dict[str, Any]] = []
            for sc in cell["scenes"]:
                if not isinstance(sc, dict):
                    continue
                start = sc.get("start")
                end = sc.get("end")
                if isinstance(start, str) and isinstance(end, str):
                    out.append(sc)
            return out
        return []

    jobs: List[_ClipJob] = []
    for _, row in df.iterrows():
        path = _get_path(row[input_video_key])
        H, W, fps = _get_hwf(row[video_info_key])
        scenes = _get_scenes(row[video_scene_key])

        if drop_invalid_timestamps and not scenes:
            # mark as an empty job; downstream will produce success=False and can be filtered if needed
            jobs.append(_ClipJob(path, H, W, fps, row.get("id"), []))
        else:
            jobs.append(_ClipJob(path, H, W, fps, row.get("id"), scenes))

    # Execute
    results: List[Dict[str, Any]] = []
    if disable_parallel:
        for jb in tqdm(jobs, total=len(jobs), desc="Build clips (serial)"):
            results.append(_make_clips_for_job(jb))
    else:
        from concurrent.futures import ProcessPoolExecutor
        max_workers = num_workers or os.cpu_count() or 1
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            for out in tqdm(ex.map(_make_clips_for_job, jobs),
                            total=len(jobs),
                            desc="Build clips (parallel)"):
                results.append(out)

    # Attach column
    df[output_key] = results

    # Optionally drop invalid rows
    if drop_invalid_timestamps:
        mask = df[output_key].apply(lambda d: isinstance(d, dict) and d.get("success") and d.get("clips"))
        df = df[mask].reset_index(drop=True)

    return df


# ----------------------------
# DataFlow Operator
# ----------------------------

@OPERATOR_REGISTRY.register()
class VideoClipFilter(OperatorABC):
    """
    DataFlow operator: consume video path/info/scene columns and produce per-row clip lists.
    Output column is configurable (default: 'video_clips').
    """

    def __init__(
        self,
        input_video_key: str = "video",
        video_info_key: str = "video_info",
        video_scene_key: str = "video_scene",
        output_key: str = "video_clips",
        drop_invalid_timestamps: bool = False,
        disable_parallel: bool = False,
        num_workers: int = 16,
    ):
        self.logger = get_logger()
        self.input_video_key = input_video_key
        self.video_info_key = video_info_key
        self.video_scene_key = video_scene_key
        self.output_key = output_key
        self.drop_invalid_timestamps = drop_invalid_timestamps
        self.disable_parallel = disable_parallel
        self.num_workers = num_workers

    @staticmethod
    def get_desc(lang: str = "zh") -> str:
        return "从上游DataFrame生成视频片段元数据" if lang == "zh" else "Generate per-row video clip metadata from upstream dataframe."

    def run(
        self,
        storage: DataFlowStorage,
        input_video_key: Optional[str] = None,
        video_info_key: Optional[str] = None,
        video_scene_key: Optional[str] = None,
        output_key: Optional[str] = None,
    ):
        input_video_key = input_video_key or self.input_video_key
        video_info_key = video_info_key or self.video_info_key
        video_scene_key = video_scene_key or self.video_scene_key
        output_key = output_key or self.output_key

        if output_key is None:
            raise ValueError("Parameter 'output_key' must not be None.")

        self.logger.info("Running VideoClipOperator...")
        df = storage.read("dataframe")
        self.logger.info(f"Loaded dataframe: {len(df)} rows")

        processed = extract_video_clips_dataframe(
            dataframe=df,
            input_video_key=input_video_key,
            video_info_key=video_info_key,
            video_scene_key=video_scene_key,
            output_key=output_key,
            drop_invalid_timestamps=self.drop_invalid_timestamps,
            disable_parallel=self.disable_parallel,
            num_workers=self.num_workers if not self.disable_parallel else 1,
        )

        storage.write(processed)
        self.logger.info(f"Clip extraction complete. Output column: '{output_key}'")
