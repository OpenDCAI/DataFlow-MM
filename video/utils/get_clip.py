"""
Video clip information extraction utility with timestamp parsing.
"""

import argparse
import os
import queue
import concurrent.futures
from functools import partial
import pandas as pd
from scenedetect import FrameTimecode
import re
from tqdm import tqdm


def process_single_row(row, args):
    """Process a single video row to extract clip information"""
    video_path = row["video_path"]
    new_rows = []

    try:
        if "timestamp" in row:
            timestamp_str = row["timestamp"]
            # Parse timestamps using regex
            timestamp_pattern = (
                r"\('(\d{2}:\d{2}:\d{2}\.\d+)', '(\d{2}:\d{2}:\d{2}\.\d+)'\)"
            )
            matches = re.findall(timestamp_pattern, timestamp_str)
            scene_list = [
                (FrameTimecode(s, fps=row["fps"]), FrameTimecode(t, fps=row["fps"]))
                for s, t in matches
            ]
        else:
            scene_list = [None]
        if args.drop_invalid_timestamps:
            return new_rows, True
    except Exception as e:
        if args.drop_invalid_timestamps:
            return new_rows, False

    height = row["height"]
    width = row["width"]
    fps = row["fps"]

    # Extract clip information for each scene
    for idx, scene in enumerate(scene_list):
        if scene is not None:
            s, t = scene  # FrameTimecode objects

            fname = os.path.basename(video_path)
            fname_wo_ext = os.path.splitext(fname)[0]

            # Calculate clip metrics
            num_frames = t.frame_num - s.frame_num
            aspect_ratio = width / height if height != 0 else 0
            resolution = f"{width}x{height}"
            timestamp_start = s.get_timecode()
            timestamp_end = t.get_timecode()
            frame_start = s.frame_num
            frame_end = t.frame_num
            id_ori = row["id"] if "id" in row else ""
            id = f"{fname_wo_ext}_{idx}"
            new_rows.append(
                [
                    video_path,
                    id,
                    num_frames,
                    height,
                    width,
                    aspect_ratio,
                    fps,
                    resolution,
                    timestamp_start,
                    timestamp_end,
                    frame_start,
                    frame_end,
                    id_ori,
                ]
            )

    return (new_rows, True)


def worker(task_queue, results_queue, args):
    """Worker function for parallel processing"""
    while True:
        try:
            index, row = task_queue.get(timeout=1)
        except queue.Empty:
            break
        result = process_single_row(row, args)
        results_queue.put((index, result))
        task_queue.task_done()


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Extract video clip information from csvdata"
    )
    parser.add_argument("--csv_path", type=str, help="Path to the input CSV file")
    parser.add_argument(
        "--csv_save_dir",
        type=str,
        required=True,
        help="Directory to save output CSV file",
    )
    parser.add_argument(
        "--num_workers", type=int, default=None, help="Number of parallel workers"
    )
    parser.add_argument(
        "--disable_parallel", action="store_true", help="Disable parallel processing"
    )
    parser.add_argument(
        "--drop_invalid_timestamps",
        action="store_true",
        help="Drop rows with invalid timestamps",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    csv_path = args.csv_path
    if not os.path.exists(csv_path):
        print(f"csv file '{csv_path}' not found. Exit.")
        return

    os.makedirs(args.csv_save_dir, exist_ok=True)

    # Load csvdata
    csv = pd.read_csv(args.csv_path)

    # Setup multiprocessing
    from multiprocessing import Manager

    manager = Manager()
    task_queue = manager.Queue()
    results_queue = manager.Queue()

    for index, row in csv.iterrows():
        task_queue.put((index, row))

    if args.disable_parallel:
        # Sequential processing
        results = []
        for index, row in tqdm(
            csv.iterrows(), total=len(csv), desc="Processing rows"
        ):
            result = process_single_row(row, args)
            results.append(result)
    else:
        # Parallel processing
        num_workers = args.num_workers if args.num_workers else os.cpu_count()

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_workers
        ) as executor:
            futures = []
            for _ in range(num_workers):
                future = executor.submit(worker, task_queue, results_queue, args)
                futures.append(future)

            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="Workers completing",
            ):
                future.result()

        # Collect results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())

    # Process results
    results.sort(key=lambda x: x[0])
    new_rows = []
    valid_rows = []
    for index, (rows, valid) in results:
        valid_rows.append(index)
        new_rows.extend(rows)

    # Save corrected timestamps if needed
    if args.drop_invalid_timestamps:
        csv = csv[valid_rows]
        assert args.csv_path.endswith("timestamp.csv"), "Only support *timestamp.csv"
        csv.to_csv(
            args.csv_path.replace("timestamp.csv", "correct_timestamp.csv"),
            index=False,
        )
        print(
            f"Corrected timestamp file saved to '{args.csv_path.replace('timestamp.csv', 'correct_timestamp.csv')}'"
        )

    # Create and save clip information DataFrame
    columns = [
        "video_path",
        "id",
        "num_frames",
        "height",
        "width",
        "aspect_ratio",
        "fps",
        "resolution",
        "timestamp_start",
        "timestamp_end",
        "frame_start",
        "frame_end",
        "id_ori",
    ]
    new_df = pd.DataFrame(new_rows, columns=columns)
    new_csv_path = os.path.join(args.csv_save_dir, "clips_info.csv")
    new_df.to_csv(new_csv_path, index=False)
    print(f"Saved {len(new_df)} clip information to {new_csv_path}.")


if __name__ == "__main__":
    main()
