import argparse
import os
import sys
import pandas as pd

from dataflow.utils.storage import FileStorage
from dataflow.operators.core_vision import ImageSensitiveFilter


class ImageSensitiveFilterPipeline:
    def __init__(
        self,
        *,
        input_file: str,
        cache_path: str,
        file_name_prefix: str = "imgtext_sensitive_step",
        cache_type: str = "jsonl",
        input_image_key: str = "image",
        input_text_keys=("caption", "question", "answer"),
        model_name: str = "facebook/bart-large-mnli",
        threshold: float = 0.5,
    ):
        self.storage = FileStorage(
            first_entry_file_name=input_file,
            cache_path=cache_path,
            file_name_prefix=file_name_prefix,
            cache_type=cache_type,
        )
        self.op = ImageSensitiveFilter(
            model_name=model_name,
            threshold=threshold,
        )
        self.input_image_key = input_image_key
        self.input_text_keys = list(input_text_keys)

    def forward(self):
        s = self.storage
        s.step()
        _ = s.read(output_type="dataframe")
        self.op.run(
            storage=s,
            input_image_key=self.input_image_key,
            input_text_keys=self.input_text_keys,
        )
        s.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test ImageSensitiveFilter (NLI-based) with single JSONL input")

    parser.add_argument("--input_file", default="./dataflow/example/test_image_filter/test_image_filter.jsonl")
    parser.add_argument("--output_file", default="./cache_filter/sensitive_filtered.jsonl")
    parser.add_argument("--cache_path", default="./cache_filter/tmp_cache")
    parser.add_argument("--file_name_prefix", default="imgtext_sensitive_step")
    parser.add_argument("--cache_type", default="jsonl")

    # Data keys
    parser.add_argument("--input_image_key", default="image")
    parser.add_argument("--input_caption_key", default="caption")
    parser.add_argument("--input_question_key", default="question")
    parser.add_argument("--input_answer_key", default="answer")

    # NLI model params
    parser.add_argument("--model_name", default="facebook/bart-large-mnli")
    parser.add_argument("--threshold", type=float, default=0.5)

    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Input file not found: {args.input_file}")
        sys.exit(1)

    df0 = pd.read_json(args.input_file, lines=True)
    needed_cols = {
        args.input_image_key,
        args.input_caption_key,
        args.input_question_key,
        args.input_answer_key,
    }
    if not needed_cols.issubset(df0.columns):
        raise KeyError(f"需要列: {needed_cols}，实际列：{list(df0.columns)}")

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    os.makedirs(args.cache_path, exist_ok=True)

    pipe = ImageSensitiveFilterPipeline(
        input_file=args.input_file,
        cache_path=args.cache_path,
        file_name_prefix=args.file_name_prefix,
        cache_type=args.cache_type,
        input_image_key=args.input_image_key,
        input_text_keys=(
            args.input_caption_key,
            args.input_question_key,
            args.input_answer_key,
        ),
        model_name=args.model_name,
        threshold=args.threshold,
    )

    print("Running ImageSensitiveFilter (NLI)...")
    pipe.forward()

    df_final = pipe.storage.read(output_type="dataframe")
    df_final.to_json(args.output_file, orient="records", lines=True, force_ascii=False)
    print(df_final[[args.input_image_key, args.input_caption_key]].head(10).to_string(index=False))
    print(f"\nImageSensitiveFilter result saved to: {args.output_file}")
