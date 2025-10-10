"""
Video information extraction utility with clean engineering structure.

Features:
- Multiple backends: OpenCV, TorchVision, PyAV
- Safe parallel execution with ProcessPoolExecutor.map
- Robust error handling per row (no global failure)
- Clear type hints and English comments
- Added fps and duration_sec
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

# Optional imports guarded at call time
import cv2  # type: ignore

from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC


class Backend(str, Enum):
    OPENCV = "opencv"
    TORCHVISION = "torchvision"
    AV = "av"


@dataclass(frozen=True)
class VideoInfo:
    """Container for a single video's probed metadata."""
    success: bool
    num_frames: Optional[int]
    height: Optional[int]
    width: Optional[int]
    aspect_ratio: Optional[float]     # width / height
    resolution: Optional[int]         # width * height
    fps: Optional[float]              # average frames per second
    duration_sec: Optional[float]     # total duration in seconds


# ----------------------------
# Low-level helpers
# ----------------------------

def _get_video_length_opencv(cap: "cv2.VideoCapture", method: str = "header") -> int:
    """
    Get video frame count using OpenCV.

    method:
        - "header": use metadata header (fast, may be unreliable in some codecs)
        - "set": seek to end then read current frame index (slower, more robust)
    """
    assert method in {"header", "set"}
    if method == "header":
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
    return int(cap.get(cv2.CAP_PROP_POS_FRAMES))


def _probe_with_opencv(path: str) -> Tuple[int, int, int, Optional[float]]:
    """
    Probe basic video info using OpenCV.

    Returns:
        (num_frames, height, width, fps)
    Raises:
        ValueError if the video cannot be opened.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        raise ValueError("Failed to open video with OpenCV.")

    # Frame count: try header first; if 0, try robust method
    num_frames = _get_video_length_opencv(cap, method="header")
    if num_frames <= 0:
        num_frames = _get_video_length_opencv(cap, method="set")

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    fps_val = cap.get(cv2.CAP_PROP_FPS)
    fps: Optional[float] = float(fps_val) if fps_val and fps_val > 0 else None

    cap.release()
    return num_frames, height, width, fps


def _probe_with_torchvision(path: str) -> Tuple[int, int, int, Optional[float]]:
    """
    Probe basic video info using TorchVision's read_video.

    Returns:
        (num_frames, height, width, fps)

    Notes:
        Assumes your local wrapper `.read_video` returns (vframes, infos),
        where infos may contain a key like "video_fps". If absent, fps=None.
    """
    try:
        from .read_video import read_video  # your local helper wrapper
    except Exception as e:
        raise ImportError(
            "TorchVision backend is not available (read_video import failed)."
        ) from e

    vframes, infos = read_video(path)
    num_frames = int(vframes.shape[0])
    height = int(vframes.shape[2])
    width = int(vframes.shape[3])

    fps: Optional[float] = None
    # Try common keys; stay permissive to your wrapper format
    if isinstance(infos, dict):
        for key in ("video_fps", "fps", "video_fps_avg"):
            val = infos.get(key)
            if val and float(val) > 0:
                fps = float(val)
                break

    return num_frames, height, width, fps


def _probe_with_av(path: str) -> Tuple[int, int, int, Optional[float], Optional[float]]:
    """
    Probe basic video info using PyAV.

    Returns:
        (num_frames, height, width, fps, duration_sec)

    Notes:
        PyAV can provide duration via stream.duration * stream.time_base.
        If that is missing, duration is computed from frames/fps when possible.
    """
    try:
        import av  # type: ignore
    except Exception as e:
        raise ImportError("PyAV backend is not installed.") from e

    container = av.open(path)
    try:
        stream = container.streams.video[0]

        # Frame metrics
        num_frames = int(stream.frames) if stream.frames is not None else 0
        height = int(stream.height)
        width = int(stream.width)

        # FPS from average_rate (Fraction)
        fps: Optional[float] = None
        if stream.average_rate:
            try:
                fps = float(stream.average_rate)
            except Exception:
                fps = None

        # Duration from stream where available
        duration_sec: Optional[float] = None
        if stream.duration is not None and stream.time_base is not None:
            try:
                duration_sec = float(stream.duration * stream.time_base)
            except Exception:
                duration_sec = None

        # Fallback: compute duration if frames and fps exist
        if duration_sec is None and num_frames and fps:
            duration_sec = float(num_frames) / float(fps)

    finally:
        container.close()

    return num_frames, height, width, fps, duration_sec


def _aspect_ratio(width: Optional[int], height: Optional[int]) -> Optional[float]:
    """Compute width/height safely."""
    if width is None or height is None or width <= 0 or height <= 0:
        return None
    return float(width) / float(height)


def _resolution(width: Optional[int], height: Optional[int]) -> Optional[int]:
    """Compute pixel count safely."""
    if width is None or height is None:
        return None
    return int(width) * int(height)


def _safe_duration(num_frames: Optional[int], fps: Optional[float]) -> Optional[float]:
    """Compute duration in seconds from num_frames and fps if both are valid."""
    if num_frames is None or fps is None:
        return None
    if fps <= 0:
        return None
    return float(num_frames) / float(fps)


# ----------------------------
# Row worker (safe)
# ----------------------------

def _probe_single(path: Union[str, List[str], Tuple[str, ...]], backend: Backend) -> VideoInfo:
    """
    Probe a single video path with the chosen backend.
    This is exception-safe and returns a structured VideoInfo.

    Notes:
        - If 'path' is a list/tuple, the first element is used.
        - Any error is captured and yields success=False.
    """
    # Normalize path
    if isinstance(path, (list, tuple)):
        path = path[0]
    path = str(path)

    if not os.path.exists(path):
        return VideoInfo(False, None, None, None, None, None, None, None)

    try:
        if backend == Backend.OPENCV:
            n, h, w, fps = _probe_with_opencv(path)
            duration = _safe_duration(n, fps)

        elif backend == Backend.TORCHVISION:
            n, h, w, fps = _probe_with_torchvision(path)
            duration = _safe_duration(n, fps)

        elif backend == Backend.AV:
            n, h, w, fps, duration = _probe_with_av(path)

        else:
            raise ValueError(f"Unknown backend: {backend}")

        ar = _aspect_ratio(w, h)
        res = _resolution(w, h)
        return VideoInfo(True, n, h, w, ar, res, fps, duration)

    except Exception:
        # Intentionally swallow exceptions to keep batch robust
        return VideoInfo(False, None, None, None, None, None, None, None)


# ----------------------------
# Public API
# ----------------------------

def extract_video_info_dataframe(
    dataframe: pd.DataFrame,
    backend: str = Backend.OPENCV.value,
    input_video_key: str = "video",
    output_key: str = "video_info",
    ext: bool = False,
    disable_parallel: bool = False,
    num_workers: Optional[int] = None,
) -> pd.DataFrame:
    """
    Extract video info for each row in a dataframe.

    Args:
        dataframe: Input dataframe (will not be mutated).
        backend: One of {"opencv", "torchvision", "av"}.
        input_video_key: Column name containing video path(s).
        output_key: Column name to store results as dicts.
        ext: If True, filter rows where the input path does not exist.
        disable_parallel: If True, run in serial.
        num_workers: Process pool size; defaults to os.cpu_count() when parallel.

    Returns:
        A new dataframe with an added/updated column `output_key` containing dicts:
            {
                "success": bool,
                "num_frames": Optional[int],
                "height": Optional[int],
                "width": Optional[int],
                "aspect_ratio": Optional[float],  # width / height
                "resolution": Optional[int],      # width * height
                "fps": Optional[float],
                "duration_sec": Optional[float],
            }
        Sorted by num_frames (ascending, None at bottom).
    """
    if input_video_key not in dataframe.columns:
        raise KeyError(f"Column '{input_video_key}' not found in dataframe.")

    data = dataframe.copy()
    data[input_video_key] = data[input_video_key].astype(object)

    if ext:
        # Filter in advance to avoid probing non-existent files
        def _exists(v: Any) -> bool:
            p = v[0] if isinstance(v, (list, tuple)) else v
            return isinstance(p, str) and os.path.exists(p)

        data = data[data[input_video_key].apply(_exists)].reset_index(drop=True)

    be = Backend(backend)
    paths: List[Union[str, List[str], Tuple[str, ...]]] = data[input_video_key].tolist()

    if disable_parallel:
        results: List[VideoInfo] = [
            _probe_single(p, be) for p in tqdm(paths, desc="Processing videos (serial)")
        ]
    else:
        from concurrent.futures import ProcessPoolExecutor

        max_workers = num_workers or os.cpu_count() or 1
        results = []
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            for vi in tqdm(ex.map(_probe_single, paths, [be] * len(paths)),
                           total=len(paths),
                           desc="Processing videos (parallel)"):
                results.append(vi)

    data[output_key] = [
        {
            "success": vi.success,
            "num_frames": vi.num_frames,
            "height": vi.height,
            "width": vi.width,
            "aspect_ratio": vi.aspect_ratio,
            "resolution": vi.resolution,
            "fps": vi.fps,
            "duration_sec": vi.duration_sec,
        }
        for vi in results
    ]

    def _sort_key(rec: Dict[str, Any]) -> float:
        val = rec.get("num_frames")
        return float("inf") if val is None else float(val)

    data = data.sort_values(by=output_key, key=lambda s: s.apply(_sort_key), ascending=True)
    data = data.reset_index(drop=True)
    return data


@OPERATOR_REGISTRY.register()
class VideoInfoFilter(OperatorABC):
    """
    DataFlow operator: scan dataframe for video files, probe basic info,
    and write the processed dataframe back to storage.
    """

    def __init__(
        self,
        backend: str = Backend.OPENCV.value,
        disable_parallel: bool = False,
        num_workers: int = 16,
        seed: int = 42,
        ext: bool = False,
    ):
        self.logger = get_logger()
        self.backend = backend
        self.disable_parallel = disable_parallel
        self.num_workers = num_workers
        self.seed = seed
        self.ext = ext

    @staticmethod
    def get_desc(lang: str = "zh") -> str:
        return "生成Video原始信息" if lang == "zh" else "Generate video info metadata."

    def run(
        self,
        storage: DataFlowStorage,
        input_video_key: str = "video",
        output_key: str = "video_info",
    ):
        if output_key is None:
            raise ValueError("Parameter 'output_key' must not be None.")

        self.logger.info("Running VideoInfoFilter...")
        dataframe = storage.read("dataframe")
        self.logger.info(f"Loaded dataframe: {len(dataframe)} rows")

        processed = extract_video_info_dataframe(
            dataframe=dataframe,
            backend=self.backend,
            input_video_key=input_video_key,
            output_key=output_key,
            ext=self.ext,
            disable_parallel=self.disable_parallel,
            num_workers=self.num_workers if not self.disable_parallel else 1,
        )

        storage.write(processed)
        self.logger.info(
            f"Video info extraction complete. Output column: '{output_key}' "
            f"(includes fps and duration_sec)"
        )
