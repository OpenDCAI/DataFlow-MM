"""
Video scene detection & timestamp processing with clean engineering structure.

Features:
- PySceneDetect-based scene detection (ContentDetector + AdaptiveDetector)
- Timestamp trimming/splitting with min/max duration constraints
- Safe parallel execution via ProcessPoolExecutor.map
- Reads video path & (optional) fps from previous dataframe output
- Clear type hints and English comments
- Output column: 'video_scene'
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import math
import pandas as pd
from tqdm import tqdm

# PySceneDetect imports (error-handled at call sites)
from scenedetect import AdaptiveDetector, ContentDetector, SceneManager, open_video

from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
from itertools import repeat

@dataclass(frozen=True)
class _SceneJobConfig:
    frame_skip: int
    start_remove_sec: float
    end_remove_sec: float
    min_seconds: float
    max_seconds: float

def _process_job(
    job: Tuple[Optional[str], Optional[float]],
    cfg: _SceneJobConfig,
) -> Dict[str, Any]:
    """
    Top-level picklable worker for ProcessPoolExecutor.
    """
    path, fps_hint = job
    return _process_single_video(
        video_path=path or "",
        frame_skip=cfg.frame_skip,
        start_remove_sec=cfg.start_remove_sec,
        end_remove_sec=cfg.end_remove_sec,
        min_seconds=cfg.min_seconds,
        max_seconds=cfg.max_seconds,
        fps_hint=fps_hint,
    )

# ----------------------------
# Helpers & data containers
# ----------------------------

def timecode_to_seconds(tc: str) -> float:
    """Convert 'HH:MM:SS.mmm' timecode into seconds (float)."""
    h, m, s = tc.split(":")
    return float(h) * 3600.0 + float(m) * 60.0 + float(s)


def seconds_to_timecode(seconds: float) -> str:
    """Convert seconds (float) to 'HH:MM:SS.mmm' timecode."""
    if seconds < 0:
        seconds = 0.0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds - (h * 3600 + m * 60)
    return f"{h:02d}:{m:02d}:{s:06.3f}"  # zero-padded with ms


def seconds_to_frame_index(seconds: float, fps: float) -> int:
    """Convert seconds to integer frame index using fps."""
    if fps is None or fps <= 0:
        return 0
    return int(math.floor(seconds * fps))


@dataclass(frozen=True)
class SceneSegment:
    """A single processed scene segment."""
    start_sec: float
    end_sec: float
    fps: float

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start": seconds_to_timecode(self.start_sec),
            "end": seconds_to_timecode(self.end_sec),
            "start_frame": seconds_to_frame_index(self.start_sec, self.fps),
            "end_frame": seconds_to_frame_index(self.end_sec, self.fps),
            "duration_sec": round(self.duration_sec, 6),
        }


# ----------------------------
# Core scene detection
# ----------------------------

def _detect_raw_scenes(
    video_path: str,
    frame_skip: int = 0,
    content_threshold: float = 21.0,
    adaptive_threshold: float = 3.0,
    min_scene_len_frames: int = 15,
) -> Tuple[List[Tuple[float, float]], Optional[float]]:
    """
    Run PySceneDetect to get raw scene boundaries (in seconds).

    Returns:
        (scene_pairs_in_seconds, fps)
    Notes:
        - Uses two detectors (Content + Adaptive) via a SceneManager.
        - If fps cannot be read from PySceneDetect, returns None and caller may override.
    """
    video = open_video(video_path)
    fps = float(video.frame_rate) if video.frame_rate else None

    sm = SceneManager()
    sm.add_detector(ContentDetector(threshold=content_threshold, min_scene_len=min_scene_len_frames))
    sm.add_detector(AdaptiveDetector(adaptive_threshold=adaptive_threshold,
                                     min_scene_len=min_scene_len_frames,
                                     luma_only=True))
    sm.detect_scenes(video=video, frame_skip=frame_skip)
    scene_list = sm.get_scene_list()

    pairs: List[Tuple[float, float]] = []
    for s, e in scene_list:
        pairs.append((timecode_to_seconds(s.get_timecode()),
                      timecode_to_seconds(e.get_timecode())))
    return pairs, fps


def _trim_and_split_scenes(
    pairs_sec: List[Tuple[float, float]],
    start_remove_sec: float,
    end_remove_sec: float,
    min_seconds: float,
    max_seconds: float,
) -> List[Tuple[float, float]]:
    """
    Apply head/tail trimming and split long scenes into chunks within [min, max].
    Returns a list of (start_sec, end_sec).
    """
    out: List[Tuple[float, float]] = []
    total_remove = max(0.0, start_remove_sec) + max(0.0, end_remove_sec)
    min_seconds = max(0.0, float(min_seconds))
    max_seconds = max(min_seconds, float(max_seconds))

    for s, e in pairs_sec:
        if e <= s:
            continue
        duration = e - s
        if duration <= 0:
            continue

        # If original duration too short to keep after trimming, skip.
        if duration < total_remove:
            continue

        ns = s + start_remove_sec
        ne = e - end_remove_sec
        if ne <= ns:
            continue

        nd = ne - ns
        if nd <= max_seconds:
            if nd >= min_seconds:
                out.append((ns, ne))
            continue

        # Split long scene into chunks of max_seconds
        cur = ns
        while cur + max_seconds <= ne:
            out.append((cur, cur + max_seconds))
            cur += max_seconds

        # Remainder
        if ne - cur >= min_seconds:
            out.append((cur, ne))

    return out


def _process_single_video(
    video_path: str,
    frame_skip: int,
    start_remove_sec: float,
    end_remove_sec: float,
    min_seconds: float,
    max_seconds: float,
    fps_hint: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Process one video path:
      - detect scenes
      - trim/split
      - build output dict with fps & scene entries
    """
    if not os.path.exists(video_path):
        return {"success": False, "error": "not_found", "fps": fps_hint, "scenes": []}

    try:
        raw_pairs, fps_detected = _detect_raw_scenes(
            video_path=video_path,
            frame_skip=frame_skip,
            content_threshold=21.0,
            adaptive_threshold=3.0,
            min_scene_len_frames=15,
        )
    except Exception as e:
        return {"success": False, "error": f"detect_failed: {e}", "fps": fps_hint, "scenes": []}

    fps = fps_hint if (fps_hint is not None and fps_hint > 0) else (fps_detected if fps_detected and fps_detected > 0 else None)

    processed_pairs = _trim_and_split_scenes(
        pairs_sec=raw_pairs,
        start_remove_sec=start_remove_sec,
        end_remove_sec=end_remove_sec,
        min_seconds=min_seconds,
        max_seconds=max_seconds,
    )

    scenes: List[Dict[str, Any]] = []
    if fps is None or fps <= 0:
        # Without fps, we can still return timecodes & durations; frame indices will be 0.
        # Users can backfill frames later if fps becomes available.
        for s, e in processed_pairs:
            seg = SceneSegment(start_sec=s, end_sec=e, fps=0.0)
            scenes.append(seg.to_dict())
    else:
        for s, e in processed_pairs:
            seg = SceneSegment(start_sec=s, end_sec=e, fps=float(fps))
            scenes.append(seg.to_dict())

    return {"success": True, "error": None, "fps": fps, "scenes": scenes}


# ----------------------------
# Public API
# ----------------------------

def extract_video_scenes_dataframe(
    dataframe: pd.DataFrame,
    input_video_key: str = "video",
    video_info_key: str = "video_info",
    output_key: str = "video_scene",
    frame_skip: int = 0,
    start_remove_sec: float = 0.0,
    end_remove_sec: float = 0.0,
    min_seconds: float = 2.0,
    max_seconds: float = 15.0,
    disable_parallel: bool = False,
    num_workers: Optional[int] = None,
) -> pd.DataFrame:
    """
    Compute scene segments for each row.

    Args:
        dataframe: input dataframe that already contains video path column,
                   and optionally 'video_info' dict with key 'fps'.
        input_video_key: column name containing video path(s) (str or [str]).
        video_info_key: column name containing dict with 'fps' if available.
        output_key: column to store results (dict with 'fps' and 'scenes' list).
        frame_skip: pass-through to SceneManager.detect_scenes (sampling stride).
        *_remove_sec/min/max: trimming/splitting config (seconds).
        disable_parallel: run in serial if True.
        num_workers: process pool size when parallel.

    Returns:
        New dataframe with an added/updated `output_key` column.
    """
    if input_video_key not in dataframe.columns:
        raise KeyError(f"Column '{input_video_key}' not found.")
    data = dataframe.copy()
    data[input_video_key] = data[input_video_key].astype(object)

    # Normalize to concrete path string
    def _path_from_cell(v: Any) -> Optional[str]:
        if isinstance(v, (list, tuple)) and v:
            return str(v[0])
        if isinstance(v, str):
            return v
        return None

    # Extract fps hint from 'video_info' if present
    def _fps_from_info(v: Any) -> Optional[float]:
        if isinstance(v, dict):
            fps = v.get("fps", None)
            try:
                return float(fps) if fps is not None and float(fps) > 0 else None
            except Exception:
                return None
        return None

    paths: List[Optional[str]] = data[input_video_key].apply(_path_from_cell).tolist()
    fps_hints: List[Optional[float]] = (
        data[video_info_key].apply(_fps_from_info).tolist() if video_info_key in data.columns else [None] * len(data)
    )

    jobs = list(zip(paths, fps_hints))

    if disable_parallel:
        results: List[Dict[str, Any]] = []
        for (p, f) in tqdm(jobs, total=len(jobs), desc="Scene detect (serial)"):
            res = _process_single_video(
                video_path=p or "",
                frame_skip=frame_skip,
                start_remove_sec=start_remove_sec,
                end_remove_sec=end_remove_sec,
                min_seconds=min_seconds,
                max_seconds=max_seconds,
                fps_hint=f,
            )
            results.append(res)
    else:
        from concurrent.futures import ProcessPoolExecutor

        max_workers = num_workers or os.cpu_count() or 1
        cfg = _SceneJobConfig(
            frame_skip=frame_skip,
            start_remove_sec=start_remove_sec,
            end_remove_sec=end_remove_sec,
            min_seconds=min_seconds,
            max_seconds=max_seconds,
        )

        results = []
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            # ex.map 支持多个可迭代参数；用 repeat(cfg) 给每个任务同一个配置对象
            for out in tqdm(
                ex.map(_process_job, jobs, repeat(cfg)),
                total=len(jobs),
                desc="Scene detect (parallel)",
            ):
                results.append(out)

    # Attach results to dataframe
    data[output_key] = results

    return data


# ----------------------------
# DataFlow Operator
# ----------------------------

@OPERATOR_REGISTRY.register()
class VideoSceneFilter(OperatorABC):
    """
    DataFlow operator: detect scenes for each video path from the input dataframe,
    and write the processed dataframe (adds 'video_scene' column) back to storage.
    """

    def __init__(
        self,
        frame_skip: int = 0,
        start_remove_sec: float = 0.0,
        end_remove_sec: float = 0.0,
        min_seconds: float = 2.0,
        max_seconds: float = 15.0,
        disable_parallel: bool = False,
        num_workers: int = 16,
        input_video_key: str = "video",
        video_info_key: str = "video_info",
        output_key: str = "video_scene",
    ):
        self.logger = get_logger()
        self.frame_skip = frame_skip
        self.start_remove_sec = start_remove_sec
        self.end_remove_sec = end_remove_sec
        self.min_seconds = min_seconds
        self.max_seconds = max_seconds
        self.disable_parallel = disable_parallel
        self.num_workers = num_workers
        self.input_video_key = input_video_key
        self.video_info_key = video_info_key
        self.output_key = output_key

    @staticmethod
    def get_desc(lang: str = "zh") -> str:
        return "基于 PySceneDetect 的视频场景切分" if lang == "zh" else "Video scene splitting with PySceneDetect."

    def run(
        self,
        storage: DataFlowStorage,
        input_video_key: Optional[str] = None,
        video_info_key: Optional[str] = None,
        output_key: Optional[str] = None,
    ):
        """
        Read dataframe from storage, compute scenes, write dataframe back.
        Column names can be overridden via run() params.
        """
        input_video_key = input_video_key or self.input_video_key
        video_info_key = video_info_key or self.video_info_key
        output_key = output_key or self.output_key

        if output_key is None:
            raise ValueError("Parameter 'output_key' must not be None.")

        self.logger.info("Running SceneDetectorOperator...")
        df = storage.read("dataframe")
        self.logger.info(f"Loaded dataframe: {len(df)} rows")

        processed = extract_video_scenes_dataframe(
            dataframe=df,
            input_video_key=input_video_key,
            video_info_key=video_info_key,
            output_key=output_key,
            frame_skip=self.frame_skip,
            start_remove_sec=self.start_remove_sec,
            end_remove_sec=self.end_remove_sec,
            min_seconds=self.min_seconds,
            max_seconds=self.max_seconds,
            disable_parallel=self.disable_parallel,
            num_workers=self.num_workers if not self.disable_parallel else 1,
        )

        storage.write(processed)
        self.logger.info(f"Scene detection complete. Output column: '{output_key}'")
