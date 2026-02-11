"""
OCR analysis for video clips using PaddleOCR.

Pipeline integration:
- Reads upstream dataframe columns:
    * `video` (str or [str]) : absolute path(s) to the video file
    * `video_clips` (dict)   : {"clips": [ { "id", "frame_start", "frame_end", ... }, ... ]}
      Frames are expected under: <figure_root>/<stem>/<clip_id>/img/*.jpg
      where stem = basename(video_path) without extension.

- Writes per-clip OCR score back into each clip dict:
    * clip["ocr_score"] : ratio of text area to frame area

Design:
- Supports batch processing and multi-worker data loading via PyTorch DataLoader
- Similar architecture to video_aesthetic_evaluator for consistency
"""

from __future__ import annotations

import os
import glob
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
import paddleocr

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC


# ----------------------------
# Utilities
# ----------------------------

def _first_path(cell: Any) -> Optional[str]:
    """Normalize a cell that may be str or [str] into a plain path string."""
    if isinstance(cell, (list, tuple)) and cell:
        return str(cell[0])
    if isinstance(cell, str):
        return cell
    return None

def _video_stem(video_path: str) -> str:
    """Get video filename without extension."""
    return os.path.splitext(os.path.basename(video_path))[0]

def _gather_all(obj: Any, world_size: int) -> List[Any]:
    """Gather python objects across ranks into a list of length `world_size`."""
    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, obj)
    return gathered


# ----------------------------
# Dataset
# ----------------------------

@dataclass(frozen=True)
class ClipSample:
    """Sample representing a single clip to be processed."""
    row_idx: int
    clip_idx: int
    clip_id: str
    image_paths: List[str]
    clip_height: int
    clip_width: int
    video_path: Optional[str] = None


class ClipFrameDataset(Dataset):
    """
    Dataset that loads up to `load_num` frames per clip for OCR processing.
    """
    def __init__(self, samples: List[ClipSample], load_num: int = 3):
        self.samples = samples
        self.load_num = max(1, int(load_num))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        smp = self.samples[idx]
        paths = sorted(smp.image_paths)[: self.load_num]

        # Load images using cv2; if no paths, sample frames directly from the video.
        images: List[np.ndarray] = []
        if paths:
            for p in paths:
                if os.path.exists(p):
                    img = cv2.imread(p)
                    if img is not None:
                        images.append(img)
        elif smp.video_path:
            images = _sample_frames_from_video(smp.video_path, self.load_num)
        
        # Determine actual clip dimensions
        # Priority: 1) from actual images, 2) from clip metadata
        if len(images) > 0:
            actual_height, actual_width = images[0].shape[:2]
            clip_height = actual_height
            clip_width = actual_width
        else:
            # Fallback to metadata if no images loaded
            clip_height = smp.clip_height
            clip_width = smp.clip_width
            # Create placeholder image
            images = [np.zeros((clip_height, clip_width, 3), dtype=np.uint8)]
        
        return {
            "row_idx": smp.row_idx,
            "clip_idx": smp.clip_idx,
            "clip_id": smp.clip_id,
            "images": images,  # List of numpy arrays
            "clip_height": clip_height,
            "clip_width": clip_width,
        }


# ----------------------------
# Core OCR Logic
# ----------------------------

def _discover_clip_images(figure_root: str, video_path: str, clip_id: str) -> List[str]:
    """
    Discover image files for a given clip:
        <figure_root>/<stem>/<clip_id>/img/*.jpg
    Returns all found images.
    """
    stem = _video_stem(video_path)
    clip_dir = os.path.join(figure_root, stem, clip_id, "img")
    all_images = sorted(glob.glob(os.path.join(clip_dir, "*.jpg")))
    return all_images


def _sample_frames_from_video(video_path: str, load_num: int) -> List[np.ndarray]:
    """
    Sample up to `load_num` frames from the whole video (evenly spaced).
    Used when we don't have extracted frames on disk (e.g. video_clips_key=None mode).
    """
    load_num = max(1, int(load_num))
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 0:
            # fallback: try reading sequentially
            frames: List[np.ndarray] = []
            for _ in range(load_num):
                ret, img = cap.read()
                if not ret or img is None:
                    break
                frames.append(img)
            return frames

        if total == 1:
            indices = [0]
        else:
            # evenly spaced across [0, total-1]
            indices = np.linspace(0, total - 1, num=min(load_num, total), dtype=int).tolist()

        frames = []
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ret, img = cap.read()
            if ret and img is not None:
                frames.append(img)
        return frames
    finally:
        cap.release()


def _build_samples_from_df(
    df: pd.DataFrame,
    input_video_key: str,
    video_clips_key: str,
    figure_root: str,
) -> List[ClipSample]:
    """
    Build a list of ClipSample objects from the dataframe.
    Similar to video_aesthetic_evaluator._build_samples_from_df.
    """
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
            
            # Try to get dimensions from clip metadata, fallback to 720p if not available
            # Note: In practice, clips should always have height/width from upstream operators
            clip_height = clip.get("height", 720)  # 720p default
            clip_width = clip.get("width", 1280)   # 720p default
            
            samples.append(ClipSample(ridx, cidx, clip_id, imgs, clip_height, clip_width, vpath))
    return samples


def _build_samples_from_df_whole_video(
    df: pd.DataFrame,
    input_video_key: str,
) -> List[ClipSample]:
    """
    Build samples for whole-video OCR (no clip splitting required).
    Each row produces a single sample with frames sampled directly from the video.
    """
    samples: List[ClipSample] = []
    for ridx, row in df.iterrows():
        vpath = _first_path(row.get(input_video_key))
        if not vpath:
            continue

        # Try to get dimensions from video metadata
        cap = cv2.VideoCapture(vpath)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            cap.release()
        else:
            w, h = 0, 0

        clip_width = w if w > 0 else 1280
        clip_height = h if h > 0 else 720
        clip_id = _video_stem(vpath)
        samples.append(ClipSample(ridx, 0, clip_id, [], clip_height, clip_width, vpath))
    return samples


def _compute_ocr_score_for_images(images: List[np.ndarray], clip_height: int, clip_width: int, model) -> float:
    """
    Compute OCR text area ratio for a list of images.
    
    Args:
        images: List of image arrays (numpy arrays from cv2)
        clip_height: Height of the clip
        clip_width: Width of the clip
        model: PaddleOCR model
        
    Returns:
        OCR score (ratio of text area to frame area)
    """
    if not images:
        return 0.0
    
    # Run OCR prediction
    try:
        results = model.predict(input=images)
    except Exception as e:
        print(f"OCR prediction failed: {e}")
        return 0.0
    
    # Get frame area from clip metadata or from actual image
    area = clip_height * clip_width
    if area <= 0 and len(images) > 0:
        area = images[0].shape[0] * images[0].shape[1]
    if area <= 0:
        return 0.0
    
    # Calculate text area ratios for each frame
    area_list = []
    for res in results:
        total_text_area = 0.0
        
        # Check if rec_boxes exists and has data
        if "rec_boxes" in res and res["rec_boxes"] is not None and len(res["rec_boxes"]) > 0:
            for rec_box in res["rec_boxes"]:
                try:
                    x_min, y_min, x_max, y_max = (
                        float(rec_box[0]),
                        float(rec_box[1]),
                        float(rec_box[2]),
                        float(rec_box[3]),
                    )
                    text_area = (x_max - x_min) * (y_max - y_min)
                    total_text_area += text_area
                except (IndexError, ValueError, TypeError):
                    continue
        
        ratio = total_text_area / area
        area_list.append(ratio)
    
    return max(area_list) if area_list else 0.0


def _custom_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for DataLoader since we have variable-length image lists.
    We don't stack images; we just return lists.
    """
    return {
        "row_idx": [item["row_idx"] for item in batch],
        "clip_idx": [item["clip_idx"] for item in batch],
        "clip_id": [item["clip_id"] for item in batch],
        "images": [item["images"] for item in batch],  # List of lists of images
        "clip_height": [item["clip_height"] for item in batch],
        "clip_width": [item["clip_width"] for item in batch],
    }


def _inject_ocr_into_dataframe(
    df: pd.DataFrame,
    video_clips_key: str,
    row_indices: List[int],
    clip_indices: List[int],
    ocr_scores: List[float],
) -> pd.DataFrame:
    """
    Write OCR scores back into df[video_clips_key]["clips"][clip_idx]:
        clip["ocr_score"]
    """
    out = df.copy()
    for r, c, score in zip(row_indices, clip_indices, ocr_scores):
        cell = out.at[r, video_clips_key]
        if isinstance(cell, dict) and "clips" in cell and 0 <= c < len(cell["clips"]):
            cell["clips"][c]["ocr_score"] = float(score)
            out.at[r, video_clips_key] = cell
    return out


def _inject_ocr_into_dataframe_row_level(
    df: pd.DataFrame,
    output_key: str,
    row_indices: List[int],
    ocr_scores: List[float],
) -> pd.DataFrame:
    """
    Write row-level OCR score back into df[output_key].
    """
    out = df.copy()
    for r, score in zip(row_indices, ocr_scores):
        out.at[r, output_key] = float(score)
    return out


# ----------------------------
# Public API
# ----------------------------

def score_ocr_over_dataframe(
    dataframe: pd.DataFrame,
    figure_root: str,
    input_video_key: str = "video",
    video_clips_key: Optional[str] = "video_clips",
    load_num: int = 3,
    batch_size: int = 8,
    num_workers: int = 4,
    gpu_num: int = 0,
    init_distributed: bool = False,
    det_model_dir: str = None,
    rec_model_dir: str = None,
) -> pd.DataFrame:
    """
    Compute OCR scores per clip and inject them into `video_clips`.
    Uses DataLoader for efficient batch processing with multi-worker data loading.
    Supports distributed processing across multiple GPUs.

    Args:
        dataframe: upstream df with `video` and `video_clips` columns.
        figure_root: root directory containing extracted frames.
        input_video_key/video_clips_key: column names.
        load_num: number of frames per clip to evaluate.
        batch_size: number of clips to process in each batch.
        num_workers: number of worker processes for data loading.
        gpu_num: GPU ID to use (0+ for GPU, -1 for CPU).
        init_distributed: if True (or WORLD_SIZE>1), use DistributedSampler.
        det_model_dir: Path to PaddleOCR detection model directory.
        rec_model_dir: Path to PaddleOCR recognition model directory.

    Returns:
        A new DataFrame with OCR scores written into `video_clips["clips"][i]`.
    """
    if input_video_key not in dataframe.columns:
        raise KeyError(f"Column '{input_video_key}' not found.")

    df = dataframe.copy()

    # Build samples from dataframe
    row_level_mode = video_clips_key is None
    if row_level_mode:
        # whole-video mode: no need for extracted frames or clip metadata
        samples = _build_samples_from_df_whole_video(df, input_video_key)
        # ensure output column exists
        if "ocr_score" not in df.columns:
            df["ocr_score"] = np.nan
    else:
        if video_clips_key not in dataframe.columns:
            raise KeyError(f"Column '{video_clips_key}' not found.")
        samples = _build_samples_from_df(df, input_video_key, video_clips_key, figure_root)
    if len(samples) == 0:
        # Nothing to score; return df unchanged
        return df

    # Determine distributed mode
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = init_distributed or world_size > 1
    rank = 0

    if distributed and not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()

    # Initialize PaddleOCR model
    device = "gpu:0" if gpu_num >= 0 else "cpu"
    ocr_kwargs = {
        "device": device,
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": False,
    }
    if det_model_dir:
        ocr_kwargs["det_model_dir"] = det_model_dir
    if rec_model_dir:
        ocr_kwargs["rec_model_dir"] = rec_model_dir
    model = paddleocr.PaddleOCR(**ocr_kwargs)

    # Build dataset & dataloader
    dataset = ClipFrameDataset(samples, load_num=load_num)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False) if dist.is_initialized() else None
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=sampler,
        shuffle=False,
        collate_fn=_custom_collate_fn,
        pin_memory=False,  # OCR doesn't need GPU tensor transfer
    )

    # Process batches
    row_ids: List[int] = []
    clip_ids: List[int] = []
    scores: List[float] = []

    for batch in tqdm(loader, disable=(rank != 0), desc="OCR Processing"):
        # batch contains lists of items
        for i in range(len(batch["row_idx"])):
            images = batch["images"][i]
            clip_height = batch["clip_height"][i]
            clip_width = batch["clip_width"][i]
            
            # Compute OCR score for this clip
            ocr_score = _compute_ocr_score_for_images(images, clip_height, clip_width, model)
            
            row_ids.append(batch["row_idx"][i])
            clip_ids.append(batch["clip_idx"][i])
            scores.append(ocr_score)

    # Distributed aggregation (gather to rank 0)
    if dist.is_initialized():
        gathered_rows = _gather_all(row_ids, world_size)
        gathered_clips = _gather_all(clip_ids, world_size)
        gathered_scores = _gather_all(scores, world_size)

        if rank == 0:
            # Flatten in rank order
            rows_flat = [x for sub in gathered_rows for x in sub]
            clips_flat = [x for sub in gathered_clips for x in sub]
            scores_flat = [x for sub in gathered_scores for x in sub]
            if row_level_mode:
                df_scored = _inject_ocr_into_dataframe_row_level(df, "ocr_score", rows_flat, scores_flat)
            else:
                df_scored = _inject_ocr_into_dataframe(df, video_clips_key, rows_flat, clips_flat, scores_flat)
        else:
            df_scored = df  # non-root ranks return original df; caller should only use rank 0 result
    else:
        if row_level_mode:
            df_scored = _inject_ocr_into_dataframe_row_level(df, "ocr_score", row_ids, scores)
        else:
            df_scored = _inject_ocr_into_dataframe(df, video_clips_key, row_ids, clip_ids, scores)

    return df_scored


# ----------------------------
# DataFlow Operator
# ----------------------------

@OPERATOR_REGISTRY.register()
class VideoOCREvaluator(OperatorABC):
    """
    DataFlow operator: compute OCR scores per clip and write them into `video_clips`.
    Supports batch processing and multi-worker data loading.
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
    ):
        """
        Initialize OCR analysis operator
        
        Args:
            figure_root: Directory where frames were extracted
            input_video_key: Column name for video paths
            video_clips_key: Column name for video clips data
            load_num: Number of frames to process per clip
            batch_size: Number of clips to process in each batch
            num_workers: Number of worker processes for data loading
            gpu_num: GPU ID (0+ for GPU, -1 for CPU)
            init_distributed: if True (or WORLD_SIZE>1), use DistributedSampler
            output_key: Output column name
            det_model_dir: Path to PaddleOCR detection model directory
            rec_model_dir: Path to PaddleOCR recognition model directory
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

    @staticmethod
    def get_desc(lang: str = "zh") -> str:
        return "为每个视频片段计算OCR文字比例并写回 video_clips" if lang == "zh" else "Compute per-clip OCR text ratio and write back into video_clips."

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
    ):
        """
        Execute OCR analysis (main interface for DataFlowStorage)
        """
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

        self.logger.info("Running OCRAnalyzer...")
        df = storage.read("dataframe")
        self.logger.info(f"Loaded dataframe: {len(df)} rows")

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

        # If caller wants a different column name to hold updated clips, mirror it.
        if output_key != video_clips_key:
            scored[output_key] = scored[video_clips_key]

        storage.write(scored)
        self.logger.info(f"OCR scoring complete. Scores injected into column: '{video_clips_key}'")
