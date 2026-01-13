
import argparse
import os
import sys
import pandas as pd

from dataflow.utils.storage import FileStorage
from dataflow.operators.core_vision import ImageDiversityFilter


class ImageDiversityFilterPipeline:
    def __init__(
        self,
        *,
        input_file: str,
        cache_path: str,
        file_name_prefix: str = "imgtext_diversity_step",
        cache_type: str = "jsonl",
        input_image_key: str = "image",
        input_text_key: str = "caption",
        text_thresh: float = 0.8,
        hash_size: int = 8,
        img_dist_thresh: int = 5,
    ):
        self.storage = FileStorage(
            first_entry_file_name=input_file,
            cache_path=cache_path,
            file_name_prefix=file_name_prefix,
            cache_type=cache_type,
        )
        self.op = ImageDiversityFilter(
            text_thresh=text_thresh,
            hash_size=hash_size,
            img_dist_thresh=img_dist_thresh,
        )
        self.input_image_key = input_image_key
        self.input_text_key = input_text_key

    def forward(self):
        s = self.storage
        s.step()
        _ = s.read(output_type="dataframe")
        self.op.run(
            storage=s,
            input_image_key=self.input_image_key,
            input_text_key=self.input_text_key,
        )
        s.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test ImageDiversityFilter with single JSONL input")
    parser.add_argument("--input_file", default="./dataflow/example/test_image_filter/test_image_filter.jsonl")
    parser.add_argument("--output_file", default="./cache_filter/diversity_filtered.jsonl")
    parser.add_argument("--cache_path", default="./cache_filter/cache_tmp")
    parser.add_argument("--file_name_prefix", default="imgtext_diversity_step")
    parser.add_argument("--cache_type", default="jsonl")
    parser.add_argument("--input_image_key", default="image")
    parser.add_argument("--input_text_key", default="caption")
    parser.add_argument("--text_thresh", type=float, default=0.8)
    parser.add_argument("--hash_size", type=int, default=8)
    parser.add_argument("--img_dist_thresh", type=int, default=5)

    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Input file not found: {args.input_file}")
        sys.exit(1)

    df0 = pd.read_json(args.input_file, lines=True)
    needed_cols = {args.input_image_key, args.input_text_key}
    if not needed_cols.issubset(df0.columns):
        raise KeyError(f"需要列: {needed_cols}，实际列：{list(df0.columns)}")

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    os.makedirs(args.cache_path, exist_ok=True)

    pipe = ImageDiversityFilterPipeline(
        input_file=args.input_file,
        cache_path=args.cache_path,
        file_name_prefix=args.file_name_prefix,
        cache_type=args.cache_type,
        input_image_key=args.input_image_key,
        input_text_key=args.input_text_key,
        text_thresh=args.text_thresh,
        hash_size=args.hash_size,
        img_dist_thresh=args.img_dist_thresh,
    )

    print("Running ImageDiversityFilter...")
    pipe.forward()

    df_final = pipe.storage.read(output_type="dataframe")
    df_final.to_json(args.output_file, orient="records", lines=True, force_ascii=False)

    cols_to_show = [c for c in [args.input_image_key, args.input_text_key] if c in df_final.columns]
    if cols_to_show:
        print(df_final[cols_to_show].head(10).to_string(index=False))
    else:
        print(df_final.head(10).to_string(index=False))

    print(f"\nImageDiversityFilter result saved to: {args.output_file}")
