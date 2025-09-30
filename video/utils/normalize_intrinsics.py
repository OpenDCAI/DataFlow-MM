"""
Camera intrinsics normalization utility.

This module provides functionality for:
- Normalizing camera intrinsics to standard format
- Converting focal length to normalized coordinates
- Parallel processing of multiple camera files
- Support for both threaded and sequential processing
"""

import numpy as np
import os
import pandas as pd
import argparse
import concurrent.futures
import multiprocessing as mp
from multiprocessing import Manager
import queue
from tqdm import tqdm


def possess_single_row(row, args):
    """
    Process a single row to normalize camera intrinsics.
    """
    id = row["id"]
    dir_path = os.path.join(args.dir_path, id, "reconstructions")
    cam_intrinsics_file = os.path.join(dir_path, "intrinsics.npy")
    
    # Load and normalize intrinsics
    intrinsics = np.load(cam_intrinsics_file)
    intrinsics[:, 0] /= intrinsics[:, 2] * 2  # Normalize focal length x
    intrinsics[:, 1] /= intrinsics[:, 3] * 2  # Normalize focal length y
    intrinsics[:, 2] = 0.5  # Set principal point x to center
    intrinsics[:, 3] = 0.5  # Set principal point y to center
    
    # Save normalized intrinsics
    np.save(cam_intrinsics_file, intrinsics)


def worker(task_queue, args, pbar):
    """
    Worker function for parallel processing of intrinsics normalization.
    """
    while True:
        try:
            index, row = task_queue.get(timeout=1)
        except queue.Empty:
            break
        possess_single_row(row, args)
        task_queue.task_done()
        pbar.update(1)


def parse_args():
    """Parse command line arguments for intrinsics normalization."""
    parser = argparse.ArgumentParser(description="Normalize camera intrinsics to standard format")
    parser.add_argument("--csv_path", type=str, help="Path to the csv file")
    parser.add_argument("--dir_path", type=str, default="./outputs")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of workers for parallel processing",
    )
    parser.add_argument(
        "--disable_parallel", action="store_true", help="Disable parallel processing"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    df = pd.read_csv(args.csv_path)

    if args.disable_parallel:
        # Sequential processing
        for index, row in tqdm(df.iterrows(), total=len(df)):
            possess_single_row(row, index, args)
    else:
        # Parallel processing using thread pool
        manager = Manager()
        task_queue = manager.Queue()
        
        # Add all tasks to queue
        for index, row in df.iterrows():
            task_queue.put((index, row))

        with tqdm(total=len(df), desc="Finished tasks") as pbar:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=args.num_workers
            ) as executor:
                futures = []
                for _ in range(args.num_workers):
                    futures.append(executor.submit(worker, task_queue, args, pbar))
                for future in concurrent.futures.as_completed(futures):
                    future.result()
