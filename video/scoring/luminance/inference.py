"""
Luminance analysis script for video frames using distributed processing.
Calculates mean, min, and max luminance scores for video clips using PyTorch distributed computing.
"""

import argparse
import os
import gc
from glob import glob
from datetime import timedelta
from PIL import Image
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.transforms.functional import pil_to_tensor
from tqdm import tqdm


def merge_scores(gathered_list: list, csv: pd.DataFrame):
    """Merge luminance scores from all distributed processes."""
    # Reorder results from all processes
    indices_list = list(map(lambda x: x[0], gathered_list))
    mean_scores_list = list(map(lambda x: x[1], gathered_list))
    min_scores_list = list(map(lambda x: x[2], gathered_list))
    max_scores_list = list(map(lambda x: x[3], gathered_list))

    flat_indices = []
    for x in zip(*indices_list):
        flat_indices.extend(x)
    flat_mean_scores = []
    for x in zip(*mean_scores_list):
        flat_mean_scores.extend(x)
    flat_min_scores = []
    for x in zip(*min_scores_list):
        flat_min_scores.extend(x)
    flat_max_scores = []
    for x in zip(*max_scores_list):
        flat_max_scores.extend(x)
    flat_indices = np.array(flat_indices)
    flat_mean_scores = np.array(flat_mean_scores)
    flat_min_scores = np.array(flat_min_scores)
    flat_max_scores = np.array(flat_max_scores)

    # Filter duplicates from distributed processing
    unique_indices, unique_indices_idx = np.unique(flat_indices, return_index=True)
    csv.loc[unique_indices, "luminance mean"] = flat_mean_scores[unique_indices_idx]
    csv.loc[unique_indices, "luminance min"] = flat_min_scores[unique_indices_idx]
    csv.loc[unique_indices, "luminance max"] = flat_max_scores[unique_indices_idx]

    # Drop indices in csv not in unique_indices
    csv = csv.loc[unique_indices]
    return csv


class VideoDataset(torch.utils.data.Dataset):
    """Dataset to handle video luminance computation."""

    def __init__(self, csv_path, fig_load_dir):
        self.csv_path = csv_path
        self.csv = pd.read_csv(csv_path)
        self.fig_load_dir = fig_load_dir

    def __getitem__(self, index):
        """Get video frames and compute luminance for a single sample."""
        sample = self.csv.iloc[index]

        # Load first 3 frames from video clip
        images_dir = os.path.join(self.fig_load_dir, sample["id"])
        images = sorted(glob(f"{images_dir}/img/*.jpg"))[:3]

        # Transform images to tensors
        images = torch.stack(
            [pil_to_tensor(Image.open(img).convert("RGB")) for img in images]
        )

        return {"index": index, "images": images}

    def __len__(self):
        return len(self.csv)


def parse_args():
    """Parse command line arguments for luminance analysis."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, help="Path to the input CSV file")
    parser.add_argument("--bs", type=int, default=4, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of workers")
    parser.add_argument(
        "--fig_load_dir",
        type=str,
        required=True,
        help="Directory to load the extracted frames",
    )
    parser.add_argument("--skip_if_existing", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    csv_path = args.csv_path

    if not os.path.exists(csv_path):
        print(f"csvdata file '{csv_path}' not found. Exiting.")
        return

    output_path = csv_path.replace(".csv", "_lum.csv")
    if args.skip_if_existing and os.path.exists(output_path):
        print(f"Output '{output_path}' already exists. Exiting.")
        return

    # Initialize distributed processing
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=24))
    (
        torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
        if torch.cuda.is_available()
        else None
    )

    # Setup dataset and distributed dataloader
    dataset = VideoDataset(csv_path, fig_load_dir=args.fig_load_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=args.bs,
        num_workers=args.num_workers,
        sampler=DistributedSampler(
            dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank()
        ),
    )

    # Process batches and calculate luminance scores
    indices_list = []
    mean_scores_list = []
    max_scores_list = []
    min_scores_list = []
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    for batch in tqdm(
        dataloader, disable=(dist.get_rank() != 0), position=dist.get_rank()
    ):
        indices = batch["index"]
        images = batch["images"].to(device, non_blocking=True)  # [B, N, C, H, W]

        # Calculate luminance using standard RGB weights
        R, G, B = images[:, :, 0], images[:, :, 1], images[:, :, 2]
        luminance = 0.2126 * R + 0.7152 * G + 0.0722 * B
        scores = luminance.mean(dim=[2, 3])

        # Compute statistics across frames
        mean_scores = scores.mean(dim=1).cpu().numpy()
        max_scores = scores.max(dim=1)[0].cpu().numpy()
        min_scores = scores.min(dim=1)[0].cpu().numpy()

        indices_list.extend(indices.tolist())
        mean_scores_list.extend(mean_scores.tolist())
        max_scores_list.extend(max_scores.tolist())
        min_scores_list.extend(min_scores.tolist())

    # Wait for all ranks to finish data processing
    dist.barrier()

    # Gather results from all processes and save
    torch.cuda.empty_cache()
    gc.collect()
    gathered_list = [None] * dist.get_world_size()
    dist.all_gather_object(
        gathered_list,
        (indices_list, mean_scores_list, min_scores_list, max_scores_list),
    )
    if dist.get_rank() == 0:
        csv_new = merge_scores(gathered_list, dataset.csv)
        csv_new.to_csv(output_path, index=False)
        print(f"New csv with luminance scores saved to '{output_path}'")


if __name__ == "__main__":
    main()
