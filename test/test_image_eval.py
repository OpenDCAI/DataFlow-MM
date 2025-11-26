import os
import sys
import pandas as pd

from dataflow.utils.storage import FileStorage
from dataflow.operators.core_vision import (
    CLIPEvaluator,
    LongCLIPEvaluator,
    VQAScoreEvaluator,
)

IN_PATH = "./dataflow/example/test_eval_image/test_eval_image.jsonl"
CACHE_DIR = "./cache_eval"
PREFIX = "imgtext_eval_step"

IMAGE_CANDIDATES = ["image_path", "image", "img_path", "path"]
TEXT_CANDIDATES = ["text", "caption", "question"]

CLIP_MODEL_NAME = "/data0/happykeyan/workspace/ckpt/clip-vit-base-patch32"
LONGCLIP_MODEL_PATH = "/data0/happykeyan/workspace/ckpt/LongCLIP-L-336px"
VQA_MODEL_PATH = "/data0/happykeyan/workspace/ckpt/blip-vqa-base"


def pick_columns(df: pd.DataFrame):
    image_key = next((c for c in IMAGE_CANDIDATES if c in df.columns), None)
    text_key = next((c for c in TEXT_CANDIDATES if c in df.columns), None)
    if image_key is None:
        raise KeyError(f"未找到图像列，期望列名之一：{IMAGE_CANDIDATES}，实际列：{list(df.columns)}")
    if text_key is None:
        raise KeyError(f"未找到文本列，期望列名之一：{TEXT_CANDIDATES}，实际列：{list(df.columns)}")
    return image_key, text_key


def main():
    if not os.path.exists(IN_PATH):
        print(f"输入文件不存在：{IN_PATH}")
        sys.exit(1)

    df0 = pd.read_json(IN_PATH, lines=True)
    image_key, text_key = pick_columns(df0)

    storage = FileStorage(
        first_entry_file_name=IN_PATH,
        cache_path=CACHE_DIR,
        file_name_prefix=PREFIX,
        cache_type="jsonl",
    )

    clip_evaluator = CLIPEvaluator(model_name=CLIP_MODEL_NAME)
    longclip_evaluator = LongCLIPEvaluator(model_name=LONGCLIP_MODEL_PATH)
    vqa_evaluator = VQAScoreEvaluator(model_name=VQA_MODEL_PATH, local_only=True)

    storage.step()
    _ = storage.read(output_type="dataframe")

    clip_evaluator.run(
        storage,
        input_image_key=image_key,
        input_text_key=text_key,
        output_key="clip_score",
    )
    storage.step()

    longclip_evaluator.run(
        storage,
        input_image_key=image_key,
        input_text_key=text_key,
        output_key="longclip_score",
    )
    storage.step()

    vqa_evaluator.run(
        storage,
        input_image_key=image_key,
        input_text_key=text_key,
        output_key="vqa_score",
    )
    storage.step()

    df_final = storage.read(output_type="dataframe")
    out_path = os.path.join(CACHE_DIR, f"{PREFIX}_final.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_final.to_json(out_path, orient="records", lines=True, force_ascii=False)

    show_cols = [image_key, text_key, "clip_score", "longclip_score", "vqa_score"]
    show_cols = [c for c in show_cols if c in df_final.columns]
    print(df_final[show_cols].head(10).to_string(index=False))
    print(f"\n保存结果到: {out_path}")


if __name__ == "__main__":
    main()
