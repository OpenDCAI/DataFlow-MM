"""
Luminance analysis for video clips with optional distributed execution.

Pipeline integration:
- Reads upstream dataframe columns:
    * `video` (str or [str]) : absolute path(s) to the video file
    * `video_clips` (dict)   : {"clips": [ { "id", "frame_start", "frame_end", "num_frames", ... }, ... ]}
      Frames are expected under: <figure_root>/<stem>/<clip_id>/img/*.jpg
      where stem = basename(video_path) without extension.

- Writes per-clip statistics back into each clip dict:
    * clip["luminance_mean"], clip["luminance_min"], clip["luminance_max"]

Design highlights:
- Safe, picklable top-level functions (no local lambdas/closures)
- Optional torch.distributed with DistributedSampler + all_gather_object
- Robust against missing frames (skips or returns NaN, never crashes)
"""

from __future__ import annotations

import os
import glob
import gc
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import cv2  # type: ignore

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision.transforms.functional import pil_to_tensor
from tqdm import tqdm

from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC


# ----------------------------
# Small utilities
# ----------------------------

def _first_path(cell: Any) -> Optional[str]:
    """Normalize a cell that may be str or [str] into a plain path string."""
    if isinstance(cell, (list, tuple)) and cell:
        return str(cell[0])
    if isinstance(cell, str):
        return cell
    return None

def _video_stem(video_path: str) -> str:
    return os.path.splitext(os.path.basename(video_path))[0]

def _gather_all(obj: Any, world_size: int) -> List[Any]:
    """Gather python objects across ranks into a list of length `world_size`."""
    out = [None for _ in range(world_size)]
    dist.all_gather_object(out, obj)
    return out


# ----------------------------
# Dataset
# ----------------------------

@dataclass(frozen=True)
class ClipSample:
    row_idx: int
    clip_idx: int
    clip_id: str
    image_paths: List[str]
    video_path: Optional[str] = None

class ClipFrameDataset(Dataset):
    """
    Dataset that loads up to `load_num` frames per clip.
    Missing/short frame sets are padded by repeating the last frame;
    If no frame is found, a single black frame is injected to keep shapes stable.
    """
    def __init__(self, samples: List[ClipSample], load_num: int = 3):
        self.samples = samples
        self.load_num = max(1, int(load_num))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        smp = self.samples[idx]
        paths = sorted(smp.image_paths)[: self.load_num]

        frames = []
        if len(paths) == 0:
            if smp.video_path:
                frames = _sample_frames_from_video_uint8(smp.video_path, self.load_num)
            if len(frames) == 0:
                # single black frame as fallback
                frames = [torch.zeros(3, 224, 224, dtype=torch.uint8)]
            # pad to load_num
            while len(frames) < self.load_num:
                frames.append(frames[-1].clone())
        else:
            for p in paths:
                with Image.open(p).convert("RGB") as im:
                    frames.append(pil_to_tensor(im))  # (C,H,W) uint8
            while len(frames) < self.load_num:
                frames.append(frames[-1].clone())

        images = torch.stack(frames, dim=0)  # (N, C, H, W) uint8
        return {
            "row_idx": smp.row_idx,
            "clip_idx": smp.clip_idx,
            "clip_id": smp.clip_id,
            "images": images,  # (N, C, H, W) uint8
        }


# ----------------------------
# Core luminance logic
# ----------------------------

def _sample_frames_from_video_uint8(video_path: str, load_num: int) -> List[torch.Tensor]:
    """
    Sample up to `load_num` frames from the whole video (evenly spaced).
    Returns RGB uint8 tensors with shape (C,H,W).
    Used when we don't have extracted frames on disk (e.g. video_clips_key=None mode).
    """
    load_num = max(1, int(load_num))
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 0:
            frames: List[torch.Tensor] = []
            for _ in range(load_num):
                ret, img = cap.read()
                if not ret or img is None:
                    break
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                t = torch.from_numpy(rgb).permute(2, 0, 1).contiguous().to(dtype=torch.uint8)
                frames.append(t)
            return frames

        if total == 1:
            indices = [0]
        else:
            indices = np.linspace(0, total - 1, num=min(load_num, total), dtype=int).tolist()

        frames: List[torch.Tensor] = []
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ret, img = cap.read()
            if not ret or img is None:
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(rgb).permute(2, 0, 1).contiguous().to(dtype=torch.uint8)
            frames.append(t)
        return frames
    finally:
        cap.release()

def _discover_clip_images(figure_root: str, video_path: str, clip_id: str) -> List[str]:
    """
    Discover image files for a given clip:
        <figure_root>/<stem>/<clip_id>/img/*.jpg
    """
    stem = _video_stem(video_path)
    # import ipdb; ipdb.set_trace()
    clip_dir = os.path.join(figure_root, stem, clip_id, "img")
    return sorted(glob.glob(os.path.join(clip_dir, "*.jpg")))

def _build_samples_from_df(
    df: pd.DataFrame,
    input_video_key: str,
    video_clips_key: str,
    figure_root: str,
) -> List[ClipSample]:
    samples: List[ClipSample] = []
    for ridx, row in df.iterrows():
        vpath = _first_path(row.get(input_video_key))
        if not vpath:
            continue
        vc = row.get(video_clips_key)
        if not isinstance(vc, dict) or "clips" not in vc or not isinstance(vc["clips"], list):
            continue
        for cidx, clip in enumerate(vc["clips"]):
            clip_id = str(clip.get("id", f"{_video_stem(vpath)}_{cidx}"))
            imgs = _discover_clip_images(figure_root, vpath, clip_id)
            
            samples.append(ClipSample(ridx, cidx, clip_id, imgs, vpath))
    return samples

def _build_samples_from_df_whole_video(
    df: pd.DataFrame,
    input_video_key: str,
) -> List[ClipSample]:
    """
    Build samples for whole-video luminance scoring (no clip splitting required).
    Each row produces a single sample; frames are sampled from video_path.
    """
    samples: List[ClipSample] = []
    for ridx, row in df.iterrows():
        vpath = _first_path(row.get(input_video_key))
        if not vpath:
            continue
        samples.append(ClipSample(ridx, 0, _video_stem(vpath), [], vpath))
    return samples

@torch.no_grad()
def _compute_luminance_stats(images_uint8: torch.Tensor) -> Tuple[float, float, float]:
    """
    Compute luminance statistics for a stack of frames.
    Args:
        images_uint8: (N, C, H, W) uint8 in [0,255], C=3 RGB
    Returns:
        (mean, min, max) luminance over the N frames (each frame averaged spatially)
    """
    # Convert to float32 to avoid overflow and for precise weighting
    imgs = images_uint8.to(dtype=torch.float32)  # [0..255]
    
    R = imgs[:, 0]  # (N,H,W)
    G = imgs[:, 1]
    B = imgs[:, 2]
    # ITU-R BT.709 luma coefficients
    Y = 0.2126 * R + 0.7152 * G + 0.0722 * B  # (N,H,W)
    # Per-frame average luminance
    per_frame = Y.mean(dim=(1, 2))  # (N,)
    return float(per_frame.mean().item()), float(per_frame.min().item()), float(per_frame.max().item())


def _inject_luminance_into_dataframe(
    df: pd.DataFrame,
    video_clips_key: str,
    row_indices: List[int],
    clip_indices: List[int],
    stats_triplets: List[Tuple[float, float, float]],
) -> pd.DataFrame:
    """
    Write luminance stats back into df[video_clips_key]["clips"][clip_idx]:
        clip["luminance_mean"], clip["luminance_min"], clip["luminance_max"]
    """
    out = df.copy()
    for r, c, (m, mn, mx) in zip(row_indices, clip_indices, stats_triplets):
        cell = out.at[r, video_clips_key]
        if isinstance(cell, dict) and "clips" in cell and 0 <= c < len(cell["clips"]):
            cell["clips"][c]["luminance_mean"] = float(m)
            cell["clips"][c]["luminance_min"] = float(mn)
            cell["clips"][c]["luminance_max"] = float(mx)
            out.at[r, video_clips_key] = cell
    return out

def _inject_row_level_luminance_into_dataframe(
    df: pd.DataFrame,
    row_indices: List[int],
    stats_triplets: List[Tuple[float, float, float]],
) -> pd.DataFrame:
    """
    Write row-level stats back into df columns:
      - luminance_mean, luminance_min, luminance_max
    """
    out = df.copy()
    for r, (m, mn, mx) in zip(row_indices, stats_triplets):
        out.at[r, "luminance_mean"] = float(m)
        out.at[r, "luminance_min"] = float(mn)
        out.at[r, "luminance_max"] = float(mx)
    return out


# ----------------------------
# Public API
# ----------------------------

def score_luminance_over_dataframe(
    dataframe: pd.DataFrame,
    figure_root: str,
    input_video_key: str = "video",
    video_clips_key: Optional[str] = "video_clips",
    load_num: int = 3,
    batch_size: int = 64,
    num_workers: int = 4,
    init_distributed: bool = False,
) -> pd.DataFrame:
    """
    Compute luminance stats (mean/min/max) per clip and inject them into `video_clips`.

    Args:
        dataframe: upstream df with `video` and `video_clips` columns.
        figure_root: root directory containing extracted frames.
        input_video_key/video_clips_key: column names.
        load_num: number of frames per clip to evaluate (padded if fewer).
        batch_size: DataLoader batch size (clips per batch).
        num_workers: DataLoader workers.
        init_distributed: if True (or WORLD_SIZE>1), use DistributedSampler.

    Returns:
        A new DataFrame with stats written into `video_clips["clips"][i]`.
    """
    if input_video_key not in dataframe.columns:
        raise KeyError(f"Column '{input_video_key}' not found.")

    df = dataframe.copy()

    row_level_mode = video_clips_key is None
    if row_level_mode:
        samples = _build_samples_from_df_whole_video(df, input_video_key)
        # ensure output columns exist
        for col in ["luminance_mean", "luminance_min", "luminance_max"]:
            if col not in df.columns:
                df[col] = np.nan
    else:
        if video_clips_key not in dataframe.columns:
            raise KeyError(f"Column '{video_clips_key}' not found.")
        samples = _build_samples_from_df(df, input_video_key, video_clips_key, figure_root)
    if len(samples) == 0:
        return df  # nothing to do

    # Distributed config
    world_size_env = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = init_distributed or world_size_env > 1
    rank = 0
    world_size = 1

    if distributed and dist.is_available() and not dist.is_initialized():
        # Basic single-node defaults if user didn't set envs
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29501")
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if torch.cuda.is_available():
            torch.cuda.set_device(rank % torch.cuda.device_count())

    dataset = ClipFrameDataset(samples, load_num=load_num)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False) if dist.is_initialized() else None
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, sampler=sampler, shuffle=False, pin_memory=True)

    row_ids: List[int] = []
    clip_ids: List[int] = []
    stats: List[Tuple[float, float, float]] = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for batch in tqdm(loader, disable=(rank != 0), desc="Luminance scoring"):
            images = batch["images"].to(device)  # (B, N, C, H, W) uint8
            # Compute per-sample stats independently
            B = images.shape[0]
            for b in range(B):
                mean_v, min_v, max_v = _compute_luminance_stats(images[b])
                stats.append((mean_v, min_v, max_v))
            row_ids.extend(batch["row_idx"].tolist())
            clip_ids.extend(batch["clip_idx"].tolist())

    # Aggregate across ranks
    if dist.is_initialized():
        gathered_rows = _gather_all(row_ids, world_size)
        gathered_clips = _gather_all(clip_ids, world_size)
        gathered_stats = _gather_all(stats, world_size)

        if rank == 0:
            rows_flat = [x for sub in gathered_rows for x in sub]
            clips_flat = [x for sub in gathered_clips for x in sub]
            stats_flat = [x for sub in gathered_stats for x in sub]
            if row_level_mode:
                df_scored = _inject_row_level_luminance_into_dataframe(df, rows_flat, stats_flat)
            else:
                df_scored = _inject_luminance_into_dataframe(df, video_clips_key, rows_flat, clips_flat, stats_flat)
        else:
            df_scored = df
    else:
        if row_level_mode:
            df_scored = _inject_row_level_luminance_into_dataframe(df, row_ids, stats)
        else:
            df_scored = _inject_luminance_into_dataframe(df, video_clips_key, row_ids, clip_ids, stats)

    # Cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return df_scored


# ----------------------------
# DataFlow Operator
# ----------------------------

@OPERATOR_REGISTRY.register()
class VideoLuminanceEvaluator(OperatorABC):
    """
    DataFlow operator: compute luminance statistics per clip and write them into `video_clips`.
    """

    def __init__(
        self,
        figure_root: str = "extract_frames",   # where frames were exported by your extractor
        input_video_key: str = "video",
        video_clips_key: str = "video_clips",
        load_num: int = 3,
        batch_size: int = 64,
        num_workers: int = 4,
        init_distributed: bool = False,
        output_key: str = "video_clips",       # same column; we update in place by default
    ):
        self.logger = get_logger()
        self.figure_root = figure_root
        self.input_video_key = input_video_key
        self.video_clips_key = video_clips_key
        self.load_num = load_num
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.init_distributed = init_distributed
        self.output_key = output_key

    @staticmethod
    def get_desc(lang: str = "zh") -> str:
        return "为每个视频片段计算亮度统计并写回 video_clips" if lang == "zh" else "Compute per-clip luminance stats and write back into video_clips."

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
    ):
        figure_root = figure_root or self.figure_root
        input_video_key = input_video_key or self.input_video_key
        video_clips_key = video_clips_key or self.video_clips_key
        load_num = load_num or self.load_num
        batch_size = batch_size or self.batch_size
        num_workers = num_workers or self.num_workers
        init_distributed = self.init_distributed if init_distributed is None else init_distributed
        output_key = output_key or self.output_key

        self.logger.info("Running LuminanceScoringOperator...")
        df = storage.read("dataframe")
        self.logger.info(f"Loaded dataframe: {len(df)} rows")

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

        # If caller wants a different column name to hold updated clips, mirror it.
        if output_key != video_clips_key:
            scored[output_key] = scored[video_clips_key]

        storage.write(scored)
        self.logger.info(f"Luminance scoring complete. Stats injected into column: '{video_clips_key}'")
