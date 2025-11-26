
import argparse
import os
import sys
import pandas as pd

from dataflow.utils.storage import FileStorage
from dataflow.operators.core_vision import ImageCatFilter


class ImageCatFilterPipeline:
    def __init__(
        self,
        *,
        input_file: str,
        cache_path: str,
        file_name_prefix: str = "imgtext_cat_step",
        cache_type: str = "jsonl",
        input_image_key: str = "image",
        input_caption_key: str = "caption",
        model_name: str = "facebook/bart-large-mnli",
        complexity_thresh: float = 0.4,
        min_caps: int = 2,
        action_thresh: float = 0.4,
        ocr_overlap_threshold: float = 0.2,
        ocr_nli_thresh: float = 0.6,
        device: str | None = None,
    ):
        self.storage = FileStorage(
            first_entry_file_name=input_file,
            cache_path=cache_path,
            file_name_prefix=file_name_prefix,
            cache_type=cache_type,
        )
        self.op = ImageCatFilter(
            model_name=model_name,
            complexity_thresh=complexity_thresh,
            min_caps=min_caps,
            action_thresh=action_thresh,
            ocr_overlap_threshold=ocr_overlap_threshold,
            ocr_nli_thresh=ocr_nli_thresh,
            device=device,
        )
        self.input_image_key = input_image_key
        self.input_caption_key = input_caption_key

    def forward(self):
        s = self.storage
        s.step()
        _ = s.read(output_type="dataframe")
        self.op.run(
            storage=s,
            input_image_key=self.input_image_key,
            input_caption_key=self.input_caption_key,
        )
        s.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default="./dataflow/example/test_image_filter/test_image_filter.jsonl")
    parser.add_argument("--output_file", default="./cache_filter/cat_filtered.jsonl")
    parser.add_argument("--cache_path", default="./cache_filter/tmp_cache")
    parser.add_argument("--file_name_prefix", default="imgtext_cat_step")
    parser.add_argument("--cache_type", default="jsonl")
    parser.add_argument("--input_image_key", default="image")
    parser.add_argument("--input_caption_key", default="caption")
    parser.add_argument("--model_name", default="facebook/bart-large-mnli")
    parser.add_argument("--complexity_thresh", type=float, default=0.4)
    parser.add_argument("--min_caps", type=int, default=2)
    parser.add_argument("--action_thresh", type=float, default=0.4)
    parser.add_argument("--ocr_overlap_threshold", type=float, default=0.2)
    parser.add_argument("--ocr_nli_thresh", type=float, default=0.6)
    parser.add_argument("--device", default=None)

    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Input file not found: {args.input_file}")
        sys.exit(1)

    df0 = pd.read_json(args.input_file, lines=True)
    needed_cols = {args.input_image_key, args.input_caption_key}
    if not needed_cols.issubset(df0.columns):
        raise KeyError(f"需要列: {needed_cols}，实际列：{list(df0.columns)}")

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    os.makedirs(args.cache_path, exist_ok=True)

    pipe = ImageCatFilterPipeline(
        input_file=args.input_file,
        cache_path=args.cache_path,
        file_name_prefix=args.file_name_prefix,
        cache_type=args.cache_type,
        input_image_key=args.input_image_key,
        input_caption_key=args.input_caption_key,
        model_name=args.model_name,
        complexity_thresh=args.complexity_thresh,
        min_caps=args.min_caps,
        action_thresh=args.action_thresh,
        ocr_overlap_threshold=args.ocr_overlap_threshold,
        ocr_nli_thresh=args.ocr_nli_thresh,
        device=args.device,
    )

    print("Running ImageCatFilter...")
    pipe.forward()

    df_final = pipe.storage.read(output_type="dataframe")
    df_final.to_json(args.output_file, orient="records", lines=True, force_ascii=False)

    cols_to_show = [c for c in [args.input_image_key, args.input_caption_key] if c in df_final.columns]
    if cols_to_show:
        print(df_final[cols_to_show].head(10).to_string(index=False))
    else:
        print(df_final.head(10).to_string(index=False))

    print(f"\nImageCatFilter result saved to: {args.output_file}")
