"""
Video frame extraction utility with a clean engineering structure.

Inputs expected in the upstream dataframe:
- `video`: str or [str] (video path)
- `video_info`: dict with keys like {height, width, fps} (optional but improves scheduling)
- `video_clips`: dict {"clips": [ { "id", "frame_start", "frame_end", "num_frames", "fps", ... }, ... ]}

Output:
- Adds column `video_frame_export`:
  {
    "success": bool,
    "error": Optional[str],
    "output_dir": str,
    "total_clips": int,
    "total_saved_frames": int,
    "clips": [
      {
        "clip_id": str,
        "dir": str,                 # directory where frames were saved
        "saved": int,               # number of frames saved
        "frame_indices": [int, ...] # (optional) list of local indices within the clip
      }, ...
    ]
  }

Design:
- Safe parallel execution via ProcessPoolExecutor.map (top-level picklable worker)
- Robust per-row error handling
- English comments & type hints
"""

from __future__ import annotations

import os
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2  # type: ignore
import pandas as pd
from tqdm import tqdm

from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC


# ----------------------------
# Helpers
# ----------------------------

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _parse_target_size(target_size: Optional[str | Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    """
    Accepts "640*360" or (640, 360) or None. Returns (w, h) or None.
    """
    if target_size is None:
        return None
    if isinstance(target_size, tuple):
        return (int(target_size[0]), int(target_size[1]))
    if isinstance(target_size, str):
        if "*" in target_size:
            w, h = target_size.split("*")
            return (int(w), int(h))
        if "x" in target_size.lower():
            w, h = target_size.lower().split("x")
            return (int(w), int(h))
    return None

def _first_path(cell: Any) -> Optional[str]:
    if isinstance(cell, (list, tuple)) and cell:
        return str(cell[0])
    if isinstance(cell, str):
        return cell
    return None

def _fps_from_info(cell: Any) -> Optional[float]:
    if isinstance(cell, dict):
        fps = cell.get("fps")
        try:
            fps = float(fps)
            return fps if fps > 0 else None
        except Exception:
            return None
    return None

def _clips_from_cell(cell: Any) -> List[Dict[str, Any]]:
    if isinstance(cell, dict) and isinstance(cell.get("clips"), list):
        return [c for c in cell["clips"] if isinstance(c, dict)]
    return []


# ----------------------------
# Job model & worker
# ----------------------------

@dataclass(frozen=True)
class _FrameJob:
    video_path: str
    clip_id: str
    frame_start: int
    num_frames: int
    fps: Optional[float]
    out_dir: str
    target_size: Optional[Tuple[int, int]]
    interval_sec: Optional[float]  # if None, pick evenly 3 frames

def _compute_indices(num_frames: int, fps: Optional[float], interval_sec: Optional[float]) -> List[int]:
    """
    Determine which local indices (0..num_frames-1) to save.
    - If interval_sec provided & fps>0 -> stride by round(interval_sec*fps), min 1.
    - Else pick ~3 evenly spaced frames within the clip (0, mid, last).
    """
    if num_frames <= 0:
        return []
    if interval_sec is not None and fps and fps > 0:
        stride = max(1, int(round(interval_sec * fps)))
        return [i for i in range(0, num_frames, stride)]
    # Default: 3 evenly
    if num_frames == 1:
        return [0]
    if num_frames == 2:
        return [0, 1]
    return list({0, num_frames // 2, num_frames - 1})

def _extract_frames_for_job(job: _FrameJob) -> Dict[str, Any]:
    """
    Top-level picklable worker that extracts frames for a single clip within a video.
    Returns a dict containing per-clip results.
    """
    try:
        _ensure_dir(job.out_dir)
        # Precompute which local indices to save
        local_indices = sorted(_compute_indices(job.num_frames, job.fps, job.interval_sec))
        if not local_indices:
            return {"clip_id": job.clip_id, "dir": job.out_dir, "saved": 0, "frame_indices": []}

        cap = cv2.VideoCapture(job.video_path)
        if not cap.isOpened():
            return {"clip_id": job.clip_id, "dir": job.out_dir, "saved": 0, "frame_indices": [], "error": "open_failed"}

        # Seek to clip start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(job.frame_start)))

        saved = 0
        local_pos = 0  # counts from 0 to num_frames-1 as we read
        wanted = set(local_indices)

        # Iterate sequentially for num_frames steps
        while local_pos < job.num_frames:
            ret, img = cap.read()
            if not ret:
                break

            if local_pos in wanted:
                if job.target_size is not None:
                    img = cv2.resize(img, job.target_size)
                fname = os.path.join(job.out_dir, f"frame_{local_pos:06d}.jpg")
                cv2.imwrite(fname, img)
                saved += 1

            local_pos += 1

        cap.release()
        return {"clip_id": job.clip_id, "dir": job.out_dir, "saved": saved, "frame_indices": local_indices}
    except Exception as e:
        return {"clip_id": job.clip_id, "dir": job.out_dir, "saved": 0, "frame_indices": [], "error": f"exception:{e}"}


# ----------------------------
# Public API
# ----------------------------

def extract_video_frames_dataframe(
    dataframe: pd.DataFrame,
    input_video_key: str = "video",
    video_info_key: str = "video_info",
    video_clips_key: str = "video_clips",
    output_key: str = "video_frame_export",
    output_dir: str = "extract_frames",
    interval_sec: Optional[float] = None,
    target_size: Optional[str | Tuple[int, int]] = None,
    disable_parallel: bool = False,
    num_workers: Optional[int] = None,
) -> pd.DataFrame:
    """
    Extract frames for each clip in the upstream dataframe.

    Args:
        input_video_key: where to read video path(s).
        video_info_key: where to read fps (optional).
        video_clips_key: where to read clips list produced by previous operator.
        output_key: column to store per-row extraction summary.
        output_dir: root directory to save frames (subdirs per video/clip).
        interval_sec: sampling interval in seconds; if None, save 3 evenly spaced frames per clip.
        target_size: "W*H" string or (W, H) tuple for resizing; None keeps original frame size.
    """
    for col in [input_video_key, video_info_key, video_clips_key]:
        if col not in dataframe.columns:
            raise KeyError(f"Column '{col}' not found in dataframe.")

    df = dataframe.copy()
    df[input_video_key] = df[input_video_key].astype(object)

    tsize = _parse_target_size(target_size)
    _ensure_dir(output_dir)

    # Build jobs per-clip
    jobs: List[Tuple[int, _FrameJob]] = []  # (row_index, job)
    for idx, row in df.iterrows():
        video_path = _first_path(row[input_video_key])
        fps_row = _fps_from_info(row[video_info_key])
        clips = _clips_from_cell(row[video_clips_key])

        # Each clip dict is expected to contain id, frame_start, frame_end, num_frames.
        if not video_path or not isinstance(clips, list) or len(clips) == 0:
            continue

        # Make a video-level parent dir: <output_dir>/<stem>
        stem = os.path.splitext(os.path.basename(video_path))[0]
        video_parent = os.path.join(output_dir, stem)

        for clip in clips:
            clip_id = str(clip.get("id") or f"{stem}_clip")
            frame_start = int(clip.get("frame_start") or 0)
            frame_end = int(clip.get("frame_end") or frame_start)
            num_frames = int(clip.get("num_frames") or max(0, frame_end - frame_start))
            fps_clip = clip.get("fps")
            try:
                fps_clip = float(fps_clip) if fps_clip is not None else None
            except Exception:
                fps_clip = None
            fps = fps_clip or fps_row  # prefer clip-level fps, fallback to row-level

            clip_dir = os.path.join(video_parent, clip_id, "img")
            jobs.append((
                idx,
                _FrameJob(
                    video_path=video_path,
                    clip_id=clip_id,
                    frame_start=frame_start,
                    num_frames=num_frames,
                    fps=fps,
                    out_dir=clip_dir,
                    target_size=tsize,
                    interval_sec=interval_sec,
                )
            ))

    # Run extraction
    per_row_outputs: Dict[int, Dict[str, Any]] = {}
    if disable_parallel:
        for idx, job in tqdm(jobs, total=len(jobs), desc="Extract frames (serial)"):
            res = _extract_frames_for_job(job)
            acc = per_row_outputs.setdefault(idx, {"success": True, "error": None, "clips": [], "total_saved_frames": 0})
            acc["clips"].append(res)
            acc["total_saved_frames"] += int(res.get("saved", 0))
            if "error" in res:
                acc["success"] = False
                acc["error"] = res["error"]
    else:
        from concurrent.futures import ProcessPoolExecutor

        max_workers = num_workers or os.cpu_count() or 1
        jobs_idx = [idx for idx, _ in jobs]
        jobs_only = [jb for _, jb in jobs]

        results = []
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            # 注意：不使用 lambda，直接映射顶层函数
            results_iter = ex.map(_extract_frames_for_job, jobs_only, chunksize=1)
            for row_idx, res in tqdm(zip(jobs_idx, results_iter),
                                    total=len(jobs),
                                    desc="Extract frames (parallel)"):
                acc = per_row_outputs.setdefault(row_idx, {"success": True, "error": None, "clips": [], "total_saved_frames": 0})
                acc["clips"].append(res)
                acc["total_saved_frames"] += int(res.get("saved", 0))
                if "error" in res:
                    acc["success"] = False
                    acc["error"] = res["error"]


    # Attach per-row summaries
    outputs: List[Dict[str, Any]] = []
    for idx in range(len(df)):
        summary = per_row_outputs.get(idx, {"success": False, "error": "no_jobs", "clips": [], "total_saved_frames": 0})
        # annotated context
        summary["output_dir"] = output_dir
        summary["total_clips"] = len(summary.get("clips", []))
        outputs.append(summary)

    df[output_key] = outputs
    return df


# ----------------------------
# DataFlow Operator
# ----------------------------

@OPERATOR_REGISTRY.register()
class VideoFrameFilter(OperatorABC):
    """
    DataFlow operator: extract frames for each clip from the upstream dataframe.
    Consumes `video`, `video_info`, `video_clips`; produces `video_frame_export`.
    """

    def __init__(
        self,
        input_video_key: str = "video",
        video_info_key: str = "video_info",
        video_clips_key: str = "video_clips",
        output_key: str = "video_frame_export",
        output_dir: str = "./cache/extract_frames",
        interval_sec: Optional[float] = None,
        target_size: Optional[str] = "640*360",
        disable_parallel: bool = False,
        num_workers: int = 16,
    ):
        self.logger = get_logger()
        self.input_video_key = input_video_key
        self.video_info_key = video_info_key
        self.video_clips_key = video_clips_key
        self.output_key = output_key
        self.output_dir = output_dir
        self.interval_sec = interval_sec
        self.target_size = target_size
        self.disable_parallel = disable_parallel
        self.num_workers = num_workers

    @staticmethod
    def get_desc(lang: str = "zh") -> str:
        return "按上游片段元数据抽取视频帧" if lang == "zh" else "Extract frames per clip from upstream metadata."

    def run(
        self,
        storage: DataFlowStorage,
        input_video_key: Optional[str] = None,
        video_info_key: Optional[str] = None,
        video_clips_key: Optional[str] = None,
        output_key: Optional[str] = None,
        output_dir: Optional[str] = None,
    ):
        input_video_key = input_video_key or self.input_video_key
        video_info_key = video_info_key or self.video_info_key
        video_clips_key = video_clips_key or self.video_clips_key
        output_key = output_key or self.output_key
        output_dir = output_dir or os.path.join(storage.cache_path, self.output_dir)

        if output_key is None:
            raise ValueError("Parameter 'output_key' must not be None.")

        self.logger.info("Running ExtractFramesOperator...")
        df = storage.read("dataframe")
        self.logger.info(f"Loaded dataframe: {len(df)} rows")

        processed = extract_video_frames_dataframe(
            dataframe=df,
            input_video_key=input_video_key,
            video_info_key=video_info_key,
            video_clips_key=video_clips_key,
            output_key=output_key,
            output_dir=output_dir,
            interval_sec=self.interval_sec,
            target_size=self.target_size,
            disable_parallel=self.disable_parallel,
            num_workers=self.num_workers if not self.disable_parallel else 1,
        )

        storage.write(processed)
        self.logger.info(f"Frame extraction complete. Output column: '{output_key}', root dir: {output_dir}")
