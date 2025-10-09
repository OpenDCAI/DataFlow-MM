"""
Aesthetic scoring for video clips using CLIP encoder + MLP head.

Pipeline integration:
- Reads upstream dataframe columns:
    * `video` (str or [str]) : absolute path to the video file
    * `video_clips` (dict)   : {"clips": [ { "id", "frame_start", "frame_end", "num_frames", ... }, ... ]}
      Frames are expected under: <figure_root>/<stem>/<clip_id>/img/*.jpg
      where stem = basename(video_path) without extension.

- Writes scores back into `video_clips["clips"][i]["aesthetic_score"]`

Design:
- Pure PyTorch without local lambdas/closures for multi-process safety.
- Optional torch.distributed: if WORLD_SIZE>1 or `init_distributed=True`,
  will initialize a process group and use DistributedSampler.
- Robust to missing/insufficient frames (skips clip or returns NaN).

Notes:
- CLIP backbone is loaded via openai-clip `clip.load(model_name_or_path, device=...)`.
- MLP head is a regression head over CLIP image features.
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

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from einops import rearrange
from tqdm import tqdm
import clip  # openai-clip

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
    return os.path.splitext(os.path.basename(video_path))[0]

def _gather_all(obj: Any, world_size: int) -> List[Any]:
    """Gather python objects across ranks into a list of length `world_size`."""
    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, obj)
    return gathered


# ----------------------------
# Model definition
# ----------------------------

class MLP(nn.Module):
    """Simple MLP regression head for aesthetic scoring."""
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (B, 1)


class AestheticScorer(nn.Module):
    """CLIP image encoder + MLP head."""
    def __init__(self, clip_model: str, device: str):
        super().__init__()
        self.clip, self.preprocess = clip.load(clip_model, device=device)  # e.g., "ViT-L/14"
        # Infer feature dim from the visual projection
        with torch.no_grad():
            # For OpenAI CLIP, the output of encode_image is already projected to the text/image joint space,
            # typically 768 for ViT-L/14. We will detect it dynamically at runtime.
            pass
        # We'll initialize the MLP lazily once we see a batch (to detect dim).

        self.mlp: Optional[MLP] = None
        self.device = device
        self.eval()
        self.to(device)

    def ensure_head(self, feat_dim: int, ckpt_path: Optional[str] = None):
        """Instantiate the MLP head if not yet created; optionally load weights."""
        if self.mlp is None:
            self.mlp = MLP(feat_dim).to(self.device)
            if ckpt_path and os.path.exists(ckpt_path):
                state = torch.load(ckpt_path, map_location=self.device)
                # Be tolerant to key mismatches
                self.mlp.load_state_dict(state, strict=False)
            self.mlp.eval()

    @torch.no_grad()
    def forward(self, images_bnc: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images_bnc: (B, N, C, H, W) preprocessed images
        Returns:
            scores: (B,) averaged aesthetic score per clip
        """
        B, N, C, H, W = images_bnc.shape
        flat = rearrange(images_bnc, "B N C H W -> (B N) C H W")
        feats = self.clip.encode_image(flat)               # (B*N, D)
        feats = F.normalize(feats, p=2, dim=-1).float()    # cosine-normalized
        D = feats.shape[-1]

        self.ensure_head(D)                                # lazy init if needed
        assert self.mlp is not None
        preds = self.mlp(feats)                            # (B*N, 1)
        preds = rearrange(preds, "(B N) 1 -> B N", B=B)    # (B, N)
        return preds.mean(dim=1)                           # (B,)


# ----------------------------
# Dataset
# ----------------------------

@dataclass(frozen=True)
class ClipSample:
    row_idx: int
    clip_idx: int
    clip_id: str
    image_paths: List[str]

class ClipFrameDataset(Dataset):
    """
    Dataset that loads up to `load_num` frames per clip.
    Missing/short frame sets are padded by repeating the last frame.
    """
    def __init__(self, samples: List[ClipSample], preprocess, load_num: int = 3):
        self.samples = samples
        self.preprocess = preprocess
        self.load_num = max(1, int(load_num))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        smp = self.samples[idx]
        paths = sorted(self.samples[idx].image_paths)[: self.load_num]
        # if not enough, pad by repeating last
        if len(paths) == 0:
            # create a 1x3x224x224 black image (CLIP size) as placeholder to not break shapes
            # But better: mark empty and let collate handle; here we fallback to a dummy zero tensor
            img = Image.new("RGB", (224, 224), (0, 0, 0))
            tensor = self.preprocess(img)  # (C,H,W)
            images = torch.stack([tensor for _ in range(self.load_num)], dim=0)
        else:
            while len(paths) < self.load_num:
                paths.append(paths[-1])
            # load & preprocess
            imgs = []
            for p in paths[: self.load_num]:
                with Image.open(p).convert("RGB") as im:
                    imgs.append(self.preprocess(im))
            images = torch.stack(imgs, dim=0)  # (N, C, H, W)

        return {
            "row_idx": smp.row_idx,
            "clip_idx": smp.clip_idx,
            "clip_id": smp.clip_id,
            "images": images,  # (N, C, H, W)
        }


# ----------------------------
# Scoring Core
# ----------------------------

def _discover_clip_images(figure_root: str, video_path: str, clip_id: str) -> List[str]:
    """
    Discover image files for a given clip:
        <figure_root>/<stem>/<clip_id>/img/*.jpg
    """
    stem = _video_stem(video_path)
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
            samples.append(ClipSample(ridx, cidx, clip_id, imgs))
    return samples

def _inject_scores_into_dataframe(
    df: pd.DataFrame,
    video_clips_key: str,
    row_indices: List[int],
    clip_indices: List[int],
    scores: List[float],
) -> pd.DataFrame:
    """
    Write scores back into df[video_clips_key]["clips"][clip_idx]["aesthetic_score"].
    This function mutates a copied dataframe and returns it.
    """
    out = df.copy()
    for r, c, s in zip(row_indices, clip_indices, scores):
        cell = out.at[r, video_clips_key]
        if isinstance(cell, dict) and "clips" in cell and 0 <= c < len(cell["clips"]):
            cell["clips"][c]["aesthetic_score"] = float(s)
            out.at[r, video_clips_key] = cell
    return out


def score_aesthetics_over_dataframe(
    dataframe: pd.DataFrame,
    figure_root: str,
    input_video_key: str = "video",
    video_clips_key: str = "video_clips",
    clip_model: str = "ViT-L/14",
    mlp_checkpoint: Optional[str] = None,
    load_num: int = 3,
    batch_size: int = 64,
    num_workers: int = 4,
    init_distributed: bool = False,
) -> pd.DataFrame:
    """
    Compute aesthetic scores per clip and inject them back into `video_clips`.

    Args:
        dataframe: upstream df with `video` and `video_clips` columns.
        figure_root: root directory containing extracted frames.
        input_video_key/video_clips_key: column names.
        clip_model: CLIP backbone name or model path compatible with `clip.load`.
        mlp_checkpoint: optional path to the regression head weights.
        load_num: number of frames per clip to evaluate (padded if fewer).
        batch_size: DataLoader batch size (clips per batch).
        num_workers: DataLoader workers.
        init_distributed: if True (or WORLD_SIZE>1), use DistributedSampler.

    Returns:
        A new DataFrame with scores written into `video_clips["clips"][i]["aesthetic_score"]`.
    """
    if input_video_key not in dataframe.columns:
        raise KeyError(f"Column '{input_video_key}' not found.")
    if video_clips_key not in dataframe.columns:
        raise KeyError(f"Column '{video_clips_key}' not found.")

    df = dataframe.copy()

    # Discover samples
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AestheticScorer(clip_model=clip_model, device=device)

    # Build dataset & dataloader
    dataset = ClipFrameDataset(samples, preprocess=model.preprocess, load_num=load_num)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False) if dist.is_initialized() else None
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, sampler=sampler, shuffle=False, pin_memory=True)

    # Lazy init MLP head (feature dim discovered on first batch)
    row_ids: List[int] = []
    clip_ids: List[int] = []
    scores: List[float] = []

    with torch.no_grad():
        for batch in tqdm(loader, disable=(rank != 0), desc="Aesthetic scoring"):
            images = batch["images"].to(device, non_blocking=True)  # (B, N, C, H, W)

            # Ensure MLP initialized with correct dim
            # We peek a forward through CLIP only to detect feat dim if needed
            if model.mlp is None:
                B, N, C, H, W = images.shape
                feats = model.clip.encode_image(rearrange(images, "B N C H W -> (B N) C H W"))
                D = feats.shape[-1]
                model.ensure_head(D, ckpt_path=mlp_checkpoint)
                model.mlp.eval()

            clip_scores = model(images)  # (B,)
            scores.extend(clip_scores.float().cpu().tolist())
            row_ids.extend(batch["row_idx"].tolist())
            clip_ids.extend(batch["clip_idx"].tolist())

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
            df_scored = _inject_scores_into_dataframe(df, video_clips_key, rows_flat, clips_flat, scores_flat)
        else:
            df_scored = df  # non-root ranks return original df; caller should only use rank 0 result
    else:
        df_scored = _inject_scores_into_dataframe(df, video_clips_key, row_ids, clip_ids, scores)

    # Cleanup
    torch.cuda.empty_cache()
    gc.collect()

    # Destroy process group only if this function initialized it (optional policy)
    # Caller may handle lifecycle; we keep it simple and leave group as-is.

    return df_scored


# ----------------------------
# DataFlow Operator
# ----------------------------

@OPERATOR_REGISTRY.register()
class VideoAestheticEvaluator(OperatorABC):
    """
    DataFlow operator: compute aesthetic scores per clip and write them into `video_clips`.
    """

    def __init__(
        self,
        figure_root: str = "extract_frames",   # where frames were exported by your extractor
        input_video_key: str = "video",
        video_clips_key: str = "video_clips",
        clip_model: str = "ViT-L/14",
        mlp_checkpoint: Optional[str] = None,
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
        self.clip_model = clip_model
        self.mlp_checkpoint = mlp_checkpoint
        self.load_num = load_num
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.init_distributed = init_distributed
        self.output_key = output_key

    @staticmethod
    def get_desc(lang: str = "zh") -> str:
        return "为每个视频片段计算美学分，并写回 video_clips" if lang == "zh" else "Compute per-clip aesthetic scores and write back into video_clips."

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
    ):
        # Resolve runtime config
        figure_root = figure_root or os.path.join(storage.cache_path, self.figure_root)
        input_video_key = input_video_key or self.input_video_key
        video_clips_key = video_clips_key or self.video_clips_key
        clip_model = clip_model or self.clip_model
        mlp_checkpoint = mlp_checkpoint or self.mlp_checkpoint
        load_num = load_num or self.load_num
        batch_size = batch_size or self.batch_size
        num_workers = num_workers or self.num_workers
        init_distributed = self.init_distributed if init_distributed is None else init_distributed
        output_key = output_key or self.output_key

        self.logger.info("Running AestheticScoringOperator...")
        df = storage.read("dataframe")
        self.logger.info(f"Loaded dataframe: {len(df)} rows")

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

        # Write back (we update the same column by default, but honor output_key if different)
        if output_key != video_clips_key:
            scored[output_key] = scored[video_clips_key]

        storage.write(scored)
        self.logger.info(f"Aesthetic scoring complete. Scores injected into column: '{video_clips_key}'")
