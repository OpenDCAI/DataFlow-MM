"""
Video cutting utility using FFmpeg with GPU acceleration support.
"""

import argparse
import os
import queue
import concurrent.futures
from functools import partial
import pandas as pd
import subprocess
from scenedetect import FrameTimecode
from tqdm import tqdm

# Default FFmpeg installation path
FFMPEG_PATH = "/usr/local/bin/ffmpeg"


def get_ffmpeg_acceleration():
    """Auto-detect FFmpeg acceleration: NVIDIA GPU > CPU"""
    try:
        output = subprocess.check_output(
            [FFMPEG_PATH, "-encoders"], stderr=subprocess.DEVNULL
        ).decode("utf-8")
        if "hevc_nvenc" in output:
            return "nvidia"
        else:
            return "cpu"
    except Exception as e:
        print(f"FFmpeg acceleration detection failed: {e}")
        return "cpu"


# Detect and set the acceleration type
ACCELERATION_TYPE = get_ffmpeg_acceleration()
print(f"FFmpeg acceleration type: {ACCELERATION_TYPE}")


def process_single_row(row, args, process_id):
    """Process a single video row to extract clip using FFmpeg"""
    video_path = row["video_path"]
    save_dir = args.video_save_dir

    # Skip resizing if video is already smaller (no upscaling)
    shorter_size = args.shorter_size
    if (shorter_size is not None) and ("height" in row) and ("width" in row):
        min_size = min(row["height"], row["width"])
        if min_size <= shorter_size:
            shorter_size = None

    # Parse timestamps
    seg_start = FrameTimecode(timecode=row["timestamp_start"], fps=row["fps"])
    seg_end = FrameTimecode(timecode=row["timestamp_end"], fps=row["fps"])

    id = row["id"]

    save_path = os.path.join(save_dir, f"{id}.mp4")
    if os.path.exists(save_path):
        # Use existing clip
        row["video_path"] = save_path
    else:
        try:
            # Build FFmpeg command
            cmd = [FFMPEG_PATH, "-nostdin", "-y"]
            # GPU acceleration setup
            if ACCELERATION_TYPE == "nvidia":
                cmd += [
                    "-hwaccel",
                    "cuda",
                    "-hwaccel_output_format",
                    "cuda",
                    "-hwaccel_device",
                    str(process_id % args.gpu_num),
                ]

            # Input and time range
            cmd += [
                "-i",
                video_path,
                "-ss",
                str(seg_start.get_seconds()),
                "-to",
                str(seg_end.get_seconds()),
            ]

            # Video encoder
            if ACCELERATION_TYPE == "nvidia":
                cmd += ["-c:v", "hevc_nvenc", "-preset", "fast"]
            else:
                cmd += ["-c:v", "libx264", "-preset", "fast"]

            # Frame rate
            if args.target_fps is not None:
                cmd += ["-r", str(args.target_fps)]

            # Video scaling
            if shorter_size is not None:
                if ACCELERATION_TYPE == "nvidia":
                    cmd += [
                        "-vf",
                        f"scale_cuda='if(gt(iw,ih),-2,{shorter_size})':'if(gt(iw,ih),{shorter_size},-2)'",
                    ]
                else:
                    cmd += [
                        "-vf",
                        f"scale='if(gt(iw,ih),-2,{shorter_size})':'if(gt(iw,ih),{shorter_size},-2)'",
                    ]

            cmd += ["-map", "0:v", save_path]

            subprocess.run(cmd, check=True, stderr=subprocess.PIPE)
            row["video_path"] = save_path
        except subprocess.CalledProcessError as e:
            print(f"ffmpeg command failed: {e.stderr.decode('utf-8')}")

    return row.values.tolist(), True


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Cut video clips from source videos")
    parser.add_argument("--csv_path", type=str, help="Path to input CSV file")
    parser.add_argument(
        "--csv_save_path", type=str, required=True, help="Path to save CSV file"
    )
    parser.add_argument(
        "--video_save_dir",
        type=str,
        required=True,
        help="Directory to save video clips",
    )
    parser.add_argument(
        "--target_fps", type=int, default=None, help="Target fps of output clips"
    )
    parser.add_argument(
        "--shorter_size",
        type=int,
        default=None,
        help="Resize shorter side keeping aspect ratio",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of workers for parallel processing",
    )
    parser.add_argument(
        "--disable_parallel", action="store_true", help="Disable parallel processing"
    )
    parser.add_argument(
        "--drop_invalid_timestamps",
        action="store_true",
        help="Drop rows with invalid timestamps",
    )
    parser.add_argument(
        "--gpu_num", type=int, default=1, help="Number of GPUs available"
    )
    return parser.parse_args()


def worker(task_queue, results_queue, args, process_id):
    """Worker function for parallel video processing"""
    while True:
        try:
            index, row = task_queue.get(timeout=1)
        except queue.Empty:
            break
        row, valid = process_single_row(row, args, process_id)
        results_queue.put((index, row, valid))
        task_queue.task_done()


def main():
    """Main function to process video cutting"""
    args = parse_args()
    csv_path = args.csv_path
    if not os.path.exists(csv_path):
        print(f"csv file '{csv_path}' not found. Exit.")
        return

    csv = pd.read_csv(args.csv_path)

    # Setup multiprocessing
    from multiprocessing import Manager

    manager = Manager()
    task_queue = manager.Queue()
    results_queue = manager.Queue()

    for index, row in csv.iterrows():
        task_queue.put((index, row))

    process_single_row_partial = partial(process_single_row, args=args)

    if args.disable_parallel:
        results = []
        for index, row in tqdm(
            csv.iterrows(), total=len(csv), desc="Processing rows"
        ):
            result = process_single_row_partial(row, index)
            results.append(result)
    else:
        num_workers = args.num_workers if args.num_workers else os.cpu_count()

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_workers
        ) as executor:
            futures = []
            for id in range(num_workers):
                futures.append(
                    executor.submit(worker, task_queue, results_queue, args, id)
                )

            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="Finished workers",
            ):
                future.result()

    # Collect results
    results = []
    while not results_queue.empty():
        results.append(results_queue.get())

    results.sort(key=lambda x: x[0])
    results = [x[1:] for x in results]

    new_rows = []
    valid_rows = []
    for new_row_list, valid in results:
        new_rows.append(new_row_list)
        valid_rows.append(valid)

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

    # Save results to CSV
    columns = csv.columns
    new_df = pd.DataFrame(new_rows, columns=columns)
    new_df = new_df.drop(columns=["timestamp_start"])
    new_df = new_df.drop(columns=["timestamp_end"])
    new_df = new_df.drop(columns=["frame_start"])
    new_df = new_df.drop(columns=["frame_end"])
    new_df.to_csv(args.csv_save_path, index=False)
    print(f"Saved {len(new_df)} clip information to {args.csv_save_path}.")


if __name__ == "__main__":
    main()
