"""
Dataset filtering utility for video metadata with various quality metrics.
"""

import argparse
import os
import random
from glob import glob
import numpy as np
import pandas as pd


def main(args):
    """Apply filtering criteria to dataset"""
    # Load data
    data = pd.read_csv(args.csv_path)

    # Apply filters based on various metrics
    if args.frames_min is not None:
        assert "num_frames" in data.columns
        data = data[data["num_frames"] >= args.frames_min]
    if args.frames_max is not None:
        assert "num_frames" in data.columns
        data = data[data["num_frames"] <= args.frames_max]
    if args.fps_max is not None:
        assert "fps" in data.columns
        data = data[(data["fps"] <= args.fps_max) | np.isnan(data["fps"])]
    if args.fps_min is not None:
        assert "fps" in data.columns
        data = data[(data["fps"] >= args.fps_min) | np.isnan(data["fps"])]
    if args.resolution_max is not None:
        if "resolution" not in data.columns:
            height = data["height"]
            width = data["width"]
            data["resolution"] = height * width
        data = data[data["resolution"] <= args.resolution_max]
    if args.aes_min is not None:
        assert "aesthetic score" in data.columns
        data = data[data["aesthetic score"] >= args.aes_min]
    if args.ocr_max is not None:
        assert "ocr score" in data.columns
        data = data[data["ocr score"] <= args.ocr_max]
    if args.ocr_min is not None:
        assert "ocr score" in data.columns
        data = data[data["ocr score"] >= args.ocr_min]
    if args.lum_min is not None:
        assert "luminance mean" in data.columns
        data = data[data["luminance mean"] >= args.lum_min]
    if args.lum_max is not None:
        assert "luminance mean" in data.columns
        data = data[data["luminance mean"] <= args.lum_max]
    if args.motion_min is not None:
        assert "motion score" in data.columns
        data = data[data["motion score"] >= args.motion_min]
    if args.motion_max is not None:
        assert "motion score" in data.columns
        data = data[data["motion score"] <= args.motion_max]

    # Save filtered data
    data.to_csv(args.csv_save_path, index=False)
    print(f"Saved {len(data)} samples to {args.csv_save_path}.")


def parse_args():
    """Parse command line arguments for dataset filtering"""
    parser = argparse.ArgumentParser(
        description="Filter video dataset by quality metrics"
    )
    parser.add_argument(
        "--csv_path", type=str, required=True, help="Path to input CSV file"
    )
    parser.add_argument(
        "--csv_save_path", type=str, default=None, help="Path to save output CSV file"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Video property filters
    parser.add_argument(
        "--frames_min", type=int, default=None, help="Minimum number of frames"
    )
    parser.add_argument(
        "--frames_max", type=int, default=None, help="Maximum number of frames"
    )
    parser.add_argument(
        "--resolution_max", type=int, default=None, help="Maximum resolution"
    )
    parser.add_argument("--fps_max", type=float, default=None, help="Maximum FPS")
    parser.add_argument("--fps_min", type=float, default=None, help="Minimum FPS")

    # Quality metric filters
    parser.add_argument(
        "--aes_min", type=float, default=None, help="Minimum aesthetic score"
    )
    parser.add_argument(
        "--flow_min", type=float, default=None, help="Minimum optical flow score"
    )
    parser.add_argument(
        "--flow_max", type=float, default=None, help="Maximum optical flow score"
    )
    parser.add_argument("--ocr_max", type=float, default=None, help="Maximum OCR score")
    parser.add_argument("--ocr_min", type=float, default=None, help="Minimum OCR score")
    parser.add_argument(
        "--lum_min", type=float, default=None, help="Minimum luminance score"
    )
    parser.add_argument(
        "--lum_max", type=float, default=None, help="Maximum luminance score"
    )
    parser.add_argument(
        "--blur_max", type=float, default=None, help="Maximum blur score"
    )
    parser.add_argument(
        "--motion_min", type=float, default=None, help="Minimum motion score"
    )
    parser.add_argument(
        "--motion_max", type=float, default=None, help="Maximum motion score"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Set random seeds for reproducibility
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    main(args)
