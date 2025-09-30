"""
Video file conversion utility for the SpatialVid project.

This module provides functionality to scan directories for video files,
process them, and generate CSV metadata files containing video information.
"""

import argparse
import os
import time
import pandas as pd


# Supported video file extensions
VID_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".m2ts", ".webm")


def scan_recursively(root):
    """
    Recursively scan a directory tree and yield all entries.
    """
    num = 0
    for entry in os.scandir(root):
        if entry.is_file():
            yield entry
        elif entry.is_dir():
            num += 1
            if num % 100 == 0:
                print(f"Scanned {num} directories.")
            yield from scan_recursively(entry.path)


def get_filelist(file_path, exts=None):
    """
    Get a list of files from a directory tree, optionally filtered by extensions.
    """
    filelist = []
    time_start = time.time()

    # Use recursive scanning to find all files
    obj = scan_recursively(file_path)
    for entry in obj:
        if entry.is_file():
            ext = os.path.splitext(entry.name)[-1].lower()
            if exts is None or ext in exts:
                filelist.append(entry.path)

    time_end = time.time()
    print(f"Scanned {len(filelist)} files in {time_end - time_start:.2f} seconds.")
    return filelist


def split_by_capital(name):
    """
    Split a camelCase or PascalCase string by capital letters.
    """
    new_name = ""
    for i in range(len(name)):
        if name[i].isupper() and i != 0:
            new_name += " "
        new_name += name[i]
    return new_name


def process_general_videos(root, output):
    """
    Process video files in a directory and generate a CSV metadata file.
    """
    # Expand user path (e.g., ~ to home directory)
    root = os.path.expanduser(root)
    if not os.path.exists(root):
        return

    # Get list of video files with supported extensions
    path_list = get_filelist(root, VID_EXTENSIONS)
    # Note: In some cases (like realestate dataset), you might want to use:
    # path_list = get_filelist(root)  # without extension filtering

    path_list = list(set(path_list))  # Remove duplicate entries

    # Extract filename without extension as ID
    fname_list = [os.path.splitext(os.path.basename(x))[0] for x in path_list]
    # Get relative paths from root directory
    relpath_list = [os.path.relpath(x, root) for x in path_list]

    # Create DataFrame with video metadata
    df = pd.DataFrame(dict(video_path=path_list, id=fname_list, relpath=relpath_list))

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output), exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Saved {len(df)} samples to {output}.")


if __name__ == "__main__":
    # Set up command line argument parser
    parser = argparse.ArgumentParser(
        description="Convert video directory structure to CSV metadata file"
    )
    parser.add_argument("--video_dir", type=str, help="Root directory containing video files")
    parser.add_argument("--split", type=str, default="train", help="Dataset split name")
    parser.add_argument("--info", type=str, default=None, help="Additional info file")
    parser.add_argument(
        "--output", type=str, default=None, required=True, help="Output CSV file path"
    )
    args = parser.parse_args()

    # Process videos and generate metadata CSV
    process_general_videos(args.video_dir, args.output)
