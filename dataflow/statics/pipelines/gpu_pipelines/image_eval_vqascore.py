import argparse
import os

import pandas as pd
from dataflow.utils.storage import FileStorage
from dataflow.operators.core_vision import ImageVQAScoreEvaluator


class VQAScoreEvalPipeline:
    """
    使用 VQAScoreEvaluator（BLIP VQA）对 image + caption 计算“匹配概率”得分。
    """

    def __init__(
        self,
        *,
        input_file: str = "./dataflow/example/test_image_eval/test_image_eval.jsonl",
        cache_path: str = "./cache_eval",
        file_name_prefix: str = "imgtext_eval_vqascore",
        cache_type: str = "jsonl",
        model_name: str = "Salesforce/blip-vqa-base",
        image_key: str = "image",
        caption_key: str = "caption",
        output_key: str = "vqa_score",
        local_only: bool = True,
    ):
        self.storage = FileStorage(
            first_entry_file_name=input_file,
            cache_path=cache_path,
            file_name_prefix=file_name_prefix,
            cache_type=cache_type,
        )
        self.vqa_evaluator = ImageVQAScoreEvaluator(
            model_name=model_name,
            local_only=local_only,
        )
        self.image_key = image_key
        self.caption_key = caption_key
        self.output_key = output_key

    def forward(self):
        s = self.storage
        s.step()
        _ = s.read(output_type="dataframe")

        # 注意这里参数名要跟你最终的 VQAScoreEvaluator.run 定义一致
        self.vqa_evaluator.run(
            storage=s,
            input_image_key=self.image_key,
            input_text_key=self.caption_key,
            output_key=self.output_key,
        )

        s.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate image-text match score with BLIP VQA")

    parser.add_argument("--input_file", default="./dataflow/example/test_image_eval/test_image_eval.jsonl")
    parser.add_argument("--output_file", default="./cache_eval/vqascore_eval_result.jsonl")
    parser.add_argument("--cache_path", default="./cache_eval")
    parser.add_argument("--file_name_prefix", default="imgtext_eval_vqascore")
    parser.add_argument("--cache_type", default="jsonl")
    parser.add_argument("--model_name", default="Salesforce/blip-vqa-base")
    parser.add_argument("--image_key", default="image")
    parser.add_argument("--caption_key", default="caption")
    parser.add_argument("--output_key", default="vqa_score")
    parser.add_argument("--local_only", action="store_true", default=True)

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    os.makedirs(args.cache_path, exist_ok=True)

    df0 = pd.read_json(args.input_file, lines=True)
    if args.image_key not in df0.columns or args.caption_key not in df0.columns:
        raise KeyError(f"需要列: '{args.image_key}' 和 '{args.caption_key}'，实际列：{list(df0.columns)}")

    pipe = VQAScoreEvalPipeline(
        input_file=args.input_file,
        cache_path=args.cache_path,
        file_name_prefix=args.file_name_prefix,
        cache_type=args.cache_type,
        model_name=args.model_name,
        image_key=args.image_key,
        caption_key=args.caption_key,
        output_key=args.output_key,
        local_only=args.local_only,
    )

    print("开始运行 VQA Score 评估算子...")
    pipe.forward()

    df_final = pipe.storage.read(output_type="dataframe")
    df_final.to_json(args.output_file, orient="records", lines=True, force_ascii=False)
    show_cols = [args.image_key, args.caption_key, args.output_key]
    print(df_final[show_cols].head(10).to_string(index=False))
    print(f"\nVQA Score 评估结果已保存到: {args.output_file}")
