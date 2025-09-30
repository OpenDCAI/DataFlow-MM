"""
Aesthetic scoring script for video frames using CLIP and MLP models.
Adapted from https://github.com/christophschuhmann/improved-aesthetic-predictor/blob/main/simple_inference.py
Calculates aesthetic scores for video clips using distributed processing.
"""

# adapted from https://github.com/christophschuhmann/improved-aesthetic-predictor/blob/main/simple_inference.py
import argparse
import gc
import os
from glob import glob
from datetime import timedelta
from PIL import Image
import clip
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm


def merge_scores(gathered_list: list, csv: pd.DataFrame, column):
    """Merge aesthetic scores from all distributed processes."""
    # Reorder results from all processes
    indices_list = list(map(lambda x: x[0], gathered_list))
    scores_list = list(map(lambda x: x[1], gathered_list))

    flat_indices = []
    for x in zip(*indices_list):
        flat_indices.extend(x)
    flat_scores = []
    for x in zip(*scores_list):
        flat_scores.extend(x)
    flat_indices = np.array(flat_indices)
    flat_scores = np.array(flat_scores)

    # Filter duplicates from distributed processing
    unique_indices, unique_indices_idx = np.unique(flat_indices, return_index=True)
    csv.loc[unique_indices, column] = flat_scores[unique_indices_idx]

    # Drop indices in csv not in unique_indices
    csv = csv.loc[unique_indices]
    return csv


class VideoTextDataset(torch.utils.data.Dataset):
    """Dataset for loading video frames for aesthetic scoring."""
    
    def __init__(self, csv_path, fig_load_dir, transform=None):
        self.csv_path = csv_path
        self.csv = pd.read_csv(csv_path)
        self.transform = transform
        self.fig_load_dir = fig_load_dir

    def __getitem__(self, index):
        """Load and transform video frames for a single sample."""
        sample = self.csv.iloc[index]

        # Load first 3 frames from video clip
        images_dir = os.path.join(self.fig_load_dir, sample["id"])
        images = sorted(glob(f"{images_dir}/img/*.jpg"))[:3]

        # Apply CLIP preprocessing transforms
        images = [self.transform(Image.open(img).convert("RGB")) for img in images]

        # Stack images into tensor
        images = torch.stack(images)

        return dict(index=index, images=images)

    def __len__(self):
        return len(self.csv)


class MLP(nn.Module):
    """Multi-layer perceptron for aesthetic score prediction."""
    
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)


class AestheticScorer(nn.Module):
    """Combined CLIP + MLP model for aesthetic scoring."""
    
    def __init__(self, input_size, device):
        super().__init__()
        self.mlp = MLP(input_size)
        self.clip, self.preprocess = clip.load("ViT-L/14", device=device)

        self.eval()
        self.to(device)

    def forward(self, x):
        """Extract CLIP features and predict aesthetic scores."""
        image_features = self.clip.encode_image(x)
        image_features = F.normalize(image_features, p=2, dim=-1).float()
        return self.mlp(image_features)


def parse_args():
    """Parse command line arguments for aesthetic scoring."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, help="Path to the input CSV file")
    parser.add_argument(
        "--load_num", type=int, default=4, help="Number of frames to load"
    )
    parser.add_argument("--bs", type=int, default=1024, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of workers")
    parser.add_argument(
        "--fig_load_dir",
        type=str,
        required=True,
        help="Directory to load the extracted frames",
    )
    parser.add_argument(
        "--prefetch_factor", type=int, default=3, help="Prefetch factor"
    )
    parser.add_argument("--skip_if_existing", action="store_true")
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    csv_path = args.csv_path
    if not os.path.exists(csv_path):
        print(f"CSV file '{csv_path}' not found. Exit.")
        exit()

    wo_ext, ext = os.path.splitext(csv_path)
    out_path = f"{wo_ext}_aes{ext}"
    if args.skip_if_existing and os.path.exists(out_path):
        print(f"Output CSV file '{out_path}' already exists. Exit.")
        exit()

    # Initialize distributed processing
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=24))
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())

    # Build aesthetic scoring model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = AestheticScorer(768, device)
    model.mlp.load_state_dict(
        torch.load("checkpoints/aesthetic.pth", map_location=device)
    )
    preprocess = model.preprocess

    # Build dataset and dataloader
    dataset = VideoTextDataset(
        args.csv_path, transform=preprocess, fig_load_dir=args.fig_load_dir
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.bs,
        num_workers=args.num_workers,
        sampler=DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=False,
            drop_last=False,
        ),
    )

    # Compute aesthetic scores for all batches
    indices_list = []
    scores_list = []
    model.eval()
    for batch in tqdm(
        dataloader, disable=(dist.get_rank() != 0), position=dist.get_rank()
    ):
        indices = batch["index"]
        images = batch["images"].to(device, non_blocking=True)

        B = images.shape[0]
        images = rearrange(images, "B N C H W -> (B N) C H W")

        # Compute aesthetic scores using CLIP + MLP
        with torch.no_grad():
            scores = model(images)

        # Average scores across frames for each video
        scores = rearrange(scores, "(B N) 1 -> B N", B=B)
        scores = scores.mean(dim=1)
        scores_np = scores.to(torch.float32).cpu().numpy()

        indices_list.extend(indices.tolist())
        scores_list.extend(scores_np.tolist())

    # Wait for all ranks to finish data processing
    dist.barrier()

    # Gather results from all processes and save
    torch.cuda.empty_cache()
    gc.collect()
    gathered_list = [None] * dist.get_world_size()
    dist.all_gather_object(gathered_list, (indices_list, scores_list))
    if dist.get_rank() == 0:
        csv_new = merge_scores(gathered_list, dataset.csv, column="aesthetic score")
        csv_new.to_csv(out_path, index=False)
        print(f"New csv with aesthetic scores saved to '{out_path}'.")


if __name__ == "__main__":
    main()
