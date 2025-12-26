import argparse
import os
import sys
import pandas as pd

from dataflow.utils.storage import FileStorage
from dataflow.operators.core_vision import ImageComplexityFilter


class ComplexityFilterPipeline:
    def __init__(
        self,
        *,
        input_file: str,
        cache_path: str,
        file_name_prefix: str = "imgtext_complexity_step",
        cache_type: str = "jsonl",
        input_caption_key: str = "caption",
        model_name: str = "facebook/bart-large-mnli",
        threshold: float = 0.4,
        min_k: int = 2,
    ):
        self.storage = FileStorage(
            first_entry_file_name=input_file,
            cache_path=cache_path,
            file_name_prefix=file_name_prefix,
            cache_type=cache_type,
        )
        self.op = ImageComplexityFilter(
            model_name=model_name,
            threshold=threshold,
            min_k=min_k,
        )
        self.input_caption_key = input_caption_key

    def forward(self):
        s = self.storage
        s.step()
        _ = s.read(output_type="dataframe")
        self.op.run(
            storage=s,
            input_caption_key=self.input_caption_key,
        )
        s.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test ComplexityFilter with single JSONL input")
    parser.add_argument("--input_file", default="./dataflow/example/test_image_filter/test_image_filter.jsonl")
    parser.add_argument("--output_file", default="./cache_filter/complexity_filtered.jsonl")
    parser.add_argument("--cache_path", default="./cache_filter/cache_tmp")
    parser.add_argument("--file_name_prefix", default="imgtext_complexity_step")
    parser.add_argument("--cache_type", default="jsonl")
    parser.add_argument("--input_caption_key", default="caption")
    parser.add_argument("--input_image_key", default="image")
    parser.add_argument("--model_name", default="facebook/bart-large-mnli")
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--min_k", type=int, default=2)

    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Input file not found: {args.input_file}")
        sys.exit(1)

    df0 = pd.read_json(args.input_file, lines=True)
    if args.input_caption_key not in df0.columns:
        raise KeyError(f"需要列: '{args.input_caption_key}'，实际列：{list(df0.columns)}")

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    os.makedirs(args.cache_path, exist_ok=True)

    pipe = ComplexityFilterPipeline(
        input_file=args.input_file,
        cache_path=args.cache_path,
        file_name_prefix=args.file_name_prefix,
        cache_type=args.cache_type,
        input_caption_key=args.input_caption_key,
        model_name=args.model_name,
        threshold=args.threshold,
        min_k=args.min_k,
    )

    print("Running ComplexityFilter...")
    pipe.forward()

    df_final = pipe.storage.read(output_type="dataframe")
    df_final.to_json(args.output_file, orient="records", lines=True, force_ascii=False)

    cols_to_show = [c for c in [args.input_image_key, args.input_caption_key] if c in df_final.columns]
    if cols_to_show:
        print(df_final[cols_to_show].head(10).to_string(index=False))
    else:
        print(df_final.head(10).to_string(index=False))

    print(f"\nComplexityFilter result saved to: {args.output_file}")
