"""
Video scene detection and timestamp processing utility.

This module provides functionality for:
- Scene detection using PySceneDetect library
- Timestamp processing and filtering
- Scene duration management
- Parallel processing of video files
"""

import argparse
import os
import concurrent.futures
import queue
import numpy as np
import pandas as pd
from tqdm import tqdm
from scenedetect import (
    AdaptiveDetector,
    detect,
    ContentDetector,
    SceneManager,
    open_video,
)
from multiprocessing import Manager


def timecode_to_seconds(timecode):
    """Convert timecode string to seconds."""
    h, m, s = map(float, timecode.split(":"))
    return h * 3600 + m * 60 + s


def seconds_to_timecode(seconds):
    """Convert seconds to timecode string format."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def process_single_row(
    row,
    frame_skip=0,
    start_remove_sec=0,
    end_remove_sec=0,
    min_seconds=2,
    max_seconds=15,
):
    """
    Process a single video file for scene detection.
    """
    video_path = row["video_path"]
    detector1 = ContentDetector(threshold=21, min_scene_len=15)
    detector2 = AdaptiveDetector(
        adaptive_threshold=3.0, min_scene_len=15, luma_only=True
    )
    detector = [detector1, detector2]

    try:
        if isinstance(detector, list):
            scene_manager = SceneManager()
            for i in detector:
                scene_manager.add_detector(i)
            video = open_video(video_path)
            # Get video frame rate
            fps = video.frame_rate
            scene_manager.detect_scenes(video=video, frame_skip=frame_skip)
            scene_list = scene_manager.get_scene_list()
        else:
            video = open_video(video_path)
            # Get video frame rate
            fps = video.frame_rate
            scene_list = detect(video_path, detector, start_in_scene=True)
        timestamp = [(s.get_timecode(), t.get_timecode()) for s, t in scene_list]

        # Process timestamps: remove specified seconds from start/end, filter by duration
        new_timestamp = []
        total_remove_sec = start_remove_sec + end_remove_sec
        for start_timecode, end_timecode in timestamp:
            start_seconds = timecode_to_seconds(start_timecode)
            end_seconds = timecode_to_seconds(end_timecode)
            duration = end_seconds - start_seconds
            # Only record scenes longer than total removal time
            if duration >= total_remove_sec:
                new_start_seconds = start_seconds + start_remove_sec
                new_end_seconds = end_seconds - end_remove_sec
                new_duration = new_end_seconds - new_start_seconds
                if new_duration <= max_seconds:
                    # Duration within max_seconds, check if meets min_seconds
                    if min_seconds <= new_duration:
                        new_start_timecode = seconds_to_timecode(new_start_seconds)
                        new_end_timecode = seconds_to_timecode(new_end_seconds)
                        new_timestamp.append((new_start_timecode, new_end_timecode))
                else:
                    # Duration exceeds max_seconds, split into segments
                    current_start = new_start_seconds
                    while current_start + max_seconds <= new_end_seconds:
                        new_start_timecode = seconds_to_timecode(current_start)
                        new_end_timecode = seconds_to_timecode(
                            current_start + max_seconds
                        )
                        new_timestamp.append((new_start_timecode, new_end_timecode))
                        current_start += max_seconds

                    # Handle remaining segment
                    last_duration = new_end_seconds - current_start
                    if last_duration >= min_seconds:
                        new_start_timecode = seconds_to_timecode(current_start)
                        new_end_timecode = seconds_to_timecode(new_end_seconds)
                        new_timestamp.append((new_start_timecode, new_end_timecode))

        return True, str(new_timestamp), float(fps)
    except Exception as e:
        print(f"Video '{video_path}' with error {e}")
        return False, "", None


def timecode_to_frames(timecode, fps):
    """Convert timecode to frame number using fps."""
    h, m, s = map(float, timecode.split(":"))
    total_seconds = h * 3600 + m * 60 + s
    return int(total_seconds * fps)


def worker(task_queue, results_queue, args):
    """
    Worker function for parallel scene detection processing.
    """
    while True:
        try:
            index, row = task_queue.get(timeout=1)
        except queue.Empty:
            break
        result = process_single_row(
            row,
            frame_skip=args.frame_skip,
            start_remove_sec=args.start_remove_sec,
            end_remove_sec=args.end_remove_sec,
            min_seconds=args.min_seconds,
            max_seconds=args.max_seconds,
        )
        results_queue.put((index, result))
        task_queue.task_done()


def parse_args():
    """Parse command line arguments for scene detection."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to the input CSV file containing video paths.",
    )
    parser.add_argument(
        "--num_workers", type=int, default=1, help="#workers for concurrent.futures"
    )
    parser.add_argument(
        "--frame_skip", type=int, default=0, help="skip frame for detect_scenes"
    )
    parser.add_argument(
        "--start_remove_sec",
        type=float,
        default=0,
        help="Seconds to remove from the start of each timestamp",
    )
    parser.add_argument(
        "--end_remove_sec",
        type=float,
        default=0,
        help="Seconds to remove from the end of each timestamp",
    )
    parser.add_argument(
        "--min_seconds",
        type=float,
        default=2,
        help="Minimum duration of a scene in seconds",
    )
    parser.add_argument(
        "--max_seconds",
        type=float,
        default=15,
        help="Maximum duration of a scene in seconds",
    )
    parser.add_argument(
        "--disable_parallel", action="store_true", help="Disable parallel processing"
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    csv_path = args.csv_path
    if not os.path.exists(csv_path):
        print(f"csv file '{csv_path}' not found. Exit.")
        return

    csv = pd.read_csv(csv_path)
    ret = []

    if args.disable_parallel:
        for index, row in tqdm(csv.iterrows(), total=len(csv)):
            succ, timestamps, fps = process_single_row(
                row,
                frame_skip=args.frame_skip,
                start_remove_sec=args.start_remove_sec,
                end_remove_sec=args.end_remove_sec,
                min_seconds=args.min_seconds,
                max_seconds=args.max_seconds,
            )
            csv.at[index, "fps"] = fps
            csv.at[index, "timestamp"] = timestamps
            ret.append((index, (succ, timestamps, fps)))
    else:
        manager = Manager()
        task_queue = manager.Queue()
        results_queue = manager.Queue()

        # Add all tasks to queue
        for index, row in csv.iterrows():
            task_queue.put((index, row))

        # Set number of workers
        if args.num_workers is not None:
            num_workers = args.num_workers
        else:
            num_workers = os.cpu_count()

        # Process videos in parallel
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for _ in range(num_workers):
                future = executor.submit(worker, task_queue, results_queue, args)
                futures.append(future)

            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="Finished workers",
            ):
                future.result()

        # Collect results
        while not results_queue.empty():
            ret.append(results_queue.get())

    # Sort results by index
    ret.sort(key=lambda x: x[0])
    succ, timestamps, fps_list = list(zip(*[result for _, result in ret]))

    csv["fps"] = fps_list
    csv["timestamp"] = timestamps
    csv = csv[np.array(succ)]

    def calculate_frame_numbers(row):
        """Calculate frame numbers from timestamps and fps."""
        timestamp = eval(row["timestamp"])
        fps = row["fps"]
        frame_numbers = [
            (timecode_to_frames(start, fps), timecode_to_frames(end, fps))
            for start, end in timestamp
        ]
        return str(frame_numbers)

    csv["frame_numbers"] = csv.apply(calculate_frame_numbers, axis=1)

    # Save results to new CSV file
    wo_ext, ext = os.path.splitext(csv_path)
    out_path = f"{wo_ext}_timestamp{ext}"
    csv.to_csv(out_path, index=False)
    print(
        f"New csv (shape={csv.shape}) with timestamp and frame numbers saved to '{out_path}'."
    )


if __name__ == "__main__":
    main()
