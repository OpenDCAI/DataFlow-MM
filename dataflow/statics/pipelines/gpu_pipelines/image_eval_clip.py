import argparse
import os

import pandas as pd
from dataflow.utils.storage import FileStorage
from dataflow.operators.core_vision import ImageCLIPEvaluator


class ClipEvalPipeline:
    """
    使用 CLIPEvaluator 对 image + caption 进行图文对齐打分。
    """

    def __init__(
        self,
        *,
        input_file: str = "./dataflow/example/test_image_eval/test_image_eval.jsonl",
        cache_path: str = "./cache_eval",
        file_name_prefix: str = "imgtext_eval_clip",
        cache_type: str = "jsonl",
        model_name: str = "openai/clip-vit-base-patch32",
        image_key: str = "image",
        caption_key: str = "caption",
        output_key: str = "clip_score",
    ):
        self.storage = FileStorage(
            first_entry_file_name=input_file,
            cache_path=cache_path,
            file_name_prefix=file_name_prefix,
            cache_type=cache_type,
        )
        self.clip_evaluator = ImageCLIPEvaluator(model_name=model_name)
        self.image_key = image_key
        self.caption_key = caption_key
        self.output_key = output_key

    def forward(self):
        s = self.storage
        s.step()                               # step=0，读取原始数据
        _ = s.read(output_type="dataframe")

        self.clip_evaluator.run(
            storage=s,
            input_image_key=self.image_key,
            input_text_key=self.caption_key,
            output_key=self.output_key,
        )

        s.step()                               # 进入下一步，读取上一步输出


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate image-text alignment with CLIP")
    parser.add_argument("--input_file", default="./dataflow/example/test_image_eval/test_image_eval.jsonl")
    parser.add_argument("--output_file", default="./cache_eval/clip_eval_result.jsonl")
    parser.add_argument("--cache_path", default="./cache_eval")
    parser.add_argument("--file_name_prefix", default="imgtext_eval_clip")
    parser.add_argument("--cache_type", default="jsonl")
    parser.add_argument("--model_name", default="openai/clip-vit-base-patch32")
    parser.add_argument("--image_key", default="image")
    parser.add_argument("--caption_key", default="caption")
    parser.add_argument("--output_key", default="clip_score")

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    os.makedirs(args.cache_path, exist_ok=True)

    df0 = pd.read_json(args.input_file, lines=True)
    if args.image_key not in df0.columns or args.caption_key not in df0.columns:
        raise KeyError(f"需要列: '{args.image_key}' 和 '{args.caption_key}'，实际列：{list(df0.columns)}")

    pipe = ClipEvalPipeline(
        input_file=args.input_file,
        cache_path=args.cache_path,
        file_name_prefix=args.file_name_prefix,
        cache_type=args.cache_type,
        model_name=args.model_name,
        image_key=args.image_key,
        caption_key=args.caption_key,
        output_key=args.output_key,
    )

    print("开始运行 CLIP 评估算子...")
    pipe.forward()

    df_final = pipe.storage.read(output_type="dataframe")
    df_final.to_json(args.output_file, orient="records", lines=True, force_ascii=False)
    show_cols = [args.image_key, args.caption_key, args.output_key]
    print(df_final[show_cols].head(10).to_string(index=False))
    print(f"\nCLIP 评估结果已保存到: {args.output_file}")
