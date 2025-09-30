"""
Video reading utilities with memory optimization and multiple backend support.
"""

import gc
import math
import os
import re
import warnings
from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple, Union
from tools.logger import test_lg
import av
import cv2
import numpy as np
import torch
from torchvision import get_video_backend
from torchvision.io.video import _check_av_available

MAX_NUM_FRAMES = 2500


def read_video_av(
    filename: str,
    start_pts: Union[float, Fraction] = 0,
    end_pts: Optional[Union[float, Fraction]] = None,
    pts_unit: str = "pts",
    output_format: str = "THWC",
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Read video frames using PyAV backend with memory optimization.
    
    Modified from torchvision.io.video.read_video with improvements:
    - No audio extraction (returns empty aframes)
    - PyAV backend only
    - Added container.close() and gc.collect() to prevent memory leaks
    - Optimized for memory efficiency
    """
    # Validate format
    output_format = output_format.upper()
    if output_format not in ("THWC", "TCHW"):
        raise ValueError(f"output_format should be either 'THWC' or 'TCHW', got {output_format}.")
    
    # Check file existence
    if not os.path.exists(filename):
        raise RuntimeError(f"File not found: {filename}")
    
    # Validate backend
    assert get_video_backend() == "pyav", "pyav backend is required for read_video_av"
    _check_av_available()
    
    # Validate time range
    if end_pts is None:
        end_pts = float("inf")
    if end_pts < start_pts:
        raise ValueError(f"end_pts should be larger than start_pts, got start_pts={start_pts} and end_pts={end_pts}")

    # Extract video metadata
    info = {}
    container = av.open(filename, metadata_errors="ignore")
    video_fps = container.streams.video[0].average_rate
    if video_fps is not None:
        info["video_fps"] = float(video_fps)
    
    # Get frame dimensions
    iter_video = container.decode(**{"video": 0})
    frame = next(iter_video).to_rgb().to_ndarray()
    height, width = frame.shape[:2]
    total_frames = container.streams.video[0].frames
    if total_frames == 0:
        total_frames = MAX_NUM_FRAMES
        warnings.warn(f"total_frames is 0, using {MAX_NUM_FRAMES} as a fallback")
    container.close()
    del container

    # Pre-allocate frame buffer (np.zeros doesn't actually allocate memory)
    video_frames = np.zeros((total_frames, height, width, 3), dtype=np.uint8)

    # Read video frames
    try:
        container = av.open(filename, metadata_errors="ignore")
        assert container.streams.video is not None
        video_frames = _read_from_stream(
            video_frames,
            container,
            start_pts,
            end_pts,
            pts_unit,
            container.streams.video[0],
            {"video": 0},
            filename=filename,
        )
    except av.AVError as e:
        print(f"[Warning] Error while reading video {filename}: {e}")

    # Convert to tensor and adjust format
    vframes = torch.from_numpy(video_frames).clone()
    del video_frames
    if output_format == "TCHW":
        # Convert [T,H,W,C] to [T,C,H,W]
        vframes = vframes.permute(0, 3, 1, 2)

    aframes = torch.empty((1, 0), dtype=torch.float32)
    return vframes, aframes, info


def _read_from_stream(
    video_frames,
    container: "av.container.Container",
    start_offset: float,
    end_offset: float,
    pts_unit: str,
    stream: "av.stream.Stream",
    stream_name: Dict[str, Optional[Union[int, Tuple[int, ...], List[int]]]],
    filename: Optional[str] = None,
) -> List["av.frame.Frame"]:
    """Read frames from video stream with proper buffering and seeking"""
    # Convert time units
    if pts_unit == "sec":
        start_offset = int(math.floor(start_offset * (1 / stream.time_base)))
        if end_offset != float("inf"):
            end_offset = int(math.ceil(end_offset * (1 / stream.time_base)))
    else:
        warnings.warn("The pts_unit 'pts' gives wrong results. Please use pts_unit 'sec'.")

    # Check if buffering is needed for DivX packed B-frames
    should_buffer = True
    max_buffer_size = 5
    if stream.type == "video":
        extradata = stream.codec_context.extradata
        if extradata and b"DivX" in extradata:
            pos = extradata.find(b"DivX")
            d = extradata[pos:]
            o = re.search(rb"DivX(\d+)Build(\d+)(\w)", d)
            if o is None:
                o = re.search(rb"DivX(\d+)b(\d+)(\w)", d)
            if o is not None:
                should_buffer = o.group(3) == b"p"
    
    # Calculate seek offset with safety margin
    seek_offset = start_offset
    seek_offset = max(seek_offset - 1, 0)  # Safety margin for seeking
    if should_buffer:
        seek_offset = max(seek_offset - max_buffer_size, 0)
    
    # Seek to start position
    try:
        container.seek(seek_offset, any_frame=False, backward=True, stream=stream)
    except av.AVError as e:
        print(f"[Warning] Error while seeking video {filename}: {e}")
        return []

    # Read frames from stream
    buffer_count = 0
    frames_pts = []
    cnt = 0
    try:
        for _idx, frame in enumerate(container.decode(**stream_name)):
            frames_pts.append(frame.pts)
            video_frames[cnt] = frame.to_rgb().to_ndarray()
            cnt += 1
            if cnt >= len(video_frames):
                break
            if frame.pts >= end_offset:
                if should_buffer and buffer_count < max_buffer_size:
                    buffer_count += 1
                    continue
                break
    except av.AVError as e:
        print(f"[Warning] Error while reading video {filename}: {e}")

    # Clean up resources to prevent memory leaks
    container.close()
    del container
    gc.collect()  # Force garbage collection for PyAV threads

    # ensure that the results are sorted wrt the pts
    # NOTE: here we assert frames_pts is sorted
    start_ptr = 0
    end_ptr = cnt
    while start_ptr < end_ptr and frames_pts[start_ptr] < start_offset:
        start_ptr += 1
    while start_ptr < end_ptr and frames_pts[end_ptr - 1] > end_offset:
        end_ptr -= 1
    if start_offset > 0 and start_offset not in frames_pts[start_ptr:end_ptr]:
        # if there is no frame that exactly matches the pts of start_offset
        # add the last frame smaller than start_offset, to guarantee that
        # we will have all the necessary data. This is most useful for audio
        if start_ptr > 0:
            start_ptr -= 1
    result = video_frames[start_ptr:end_ptr].copy()
    return result


def read_video_cv2(filename, start_pts=None, end_pts=None, pts_unit="pts"):
    """
    Read video using OpenCV backend.
    """
    if pts_unit != "frames":
        warnings.warn("Using pts_unit other than 'frames' is not supported for cv2 backend")
    
    cap = cv2.VideoCapture(filename)
    
    # Get video metadata
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame range
    if start_pts is None:
        start_pts = 0
    if end_pts is None:
        end_pts = frame_count
    
    # Limit frame range to video bounds
    start_pts = max(0, start_pts)
    end_pts = min(frame_count, end_pts)
    num_frames = end_pts - start_pts
    
    if num_frames <= 0:
        return torch.zeros(0, 3, 0, 0), None, {"video_fps": fps}
    
    # Seek to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_pts)
    
    # Read frames
    frames = []
    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB and change HWC to CHW format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = torch.from_numpy(frame).permute(2, 0, 1).float()
        frames.append(frame)
    
    cap.release()
    
    if frames:
        video_tensor = torch.stack(frames)
    else:
        video_tensor = torch.zeros(0, 3, 0, 0)
    
    metadata = {"video_fps": fps}
    return video_tensor, None, metadata


def read_video(video_path, backend="av"):
    """
    Read video using specified backend.
    """
    if backend == "cv2":
        vframes, vinfo = read_video_cv2(video_path)
    elif backend == "av":
        vframes, _, vinfo = read_video_av(filename=video_path, pts_unit="sec", output_format="TCHW")
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    return vframes, vinfo
