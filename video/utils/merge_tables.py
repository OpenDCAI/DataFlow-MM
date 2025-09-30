"""
CSV table merging utility for combining multiple clip information files.
"""

import os
import glob
import argparse
import pandas as pd


def read_csv_file(file_path):
    """Read a single CSV file"""
    return pd.read_csv(file_path)


def merge_tables_from_files(file_list, output_file, merge_on=None):
    """
    Merge multiple CSV files using common columns as merge keys.

    Args:
        file_list: List of CSV file paths to merge
        output_file: Output path for merged CSV file
        merge_on: List of column names for merging (defaults to first 13 columns)
    """
    if not file_list:
        raise ValueError("File list is empty!")

    # Read all CSV files
    dfs = [read_csv_file(f) for f in file_list]

    # Auto-select merge keys: first 13 columns
    if merge_on is None:
        merge_on = dfs[0].columns[:13].tolist()

    # Merge dataframes
    df_merged = dfs[0]
    for df in dfs[1:]:
        # Check if merge keys are consistent
        if merge_on != df.columns[:13].tolist():
            raise ValueError(
                f"Common columns in one file are inconsistent with previous files!"
            )
        # Merge based on specified keys
        df_merged = pd.merge(df_merged, df, on=merge_on)

    # Save merged result
    df_merged.to_csv(output_file, index=False)
    print(f"Merge completed. Saved to {output_file}")
    return df_merged


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple CSV files from a folder"
    )
    parser.add_argument("--csv_dir", type=str, help="Path to folder containing CSV files")
    parser.add_argument(
        "--output", type=str, required=True, help="Output path for merged CSV file"
    )

    args = parser.parse_args()

    # Match CSV files with 'clips_info_' prefix
    pattern = os.path.join(args.csv_dir, "clips_info_*.csv")
    file_list = glob.glob(pattern)
    file_list.sort()  # Sort to ensure consistent merge order

    if not file_list:
        raise ValueError(f"No matching CSV files found in folder {args.csv_dir}!")

    print(f"Found {len(file_list)} CSV files:")
    for f in file_list:
        print(f"  {f}")

    # Perform merge
    merge_tables_from_files(file_list, args.output)


if __name__ == "__main__":
    main()
