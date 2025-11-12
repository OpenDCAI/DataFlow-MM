import os
import sys
import pandas as pd

from dataflow.utils.storage import FileStorage
from dataflow.operators.core_vision import (
    CLIPEvaluator,
    LongCLIPEvaluator,
    VQAScoreEvaluator,
)

IN_PATH = "./dataflow/example/image_to_text_pipeline/caption_result.jsonl"
CACHE_DIR = "./cache_eval"
PREFIX = "imgtext_eval_step"

IMAGE_CANDIDATES = ["image_path", "image", "img_path", "path"]
TEXT_CANDIDATES = ["text", "caption", "question"]

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
        # media_key=image_key,
        # media_type="image",
    )

    storage.step()  # step=0，读取原始数据
    _ = storage.read(output_type="dataframe")

    CLIPEvaluator().run(
        storage,
        image_key=image_key,
        text_key=text_key,
        output_key="clip_score",
    )
    storage.step()  # 进入下一步，读取上一步输出

    LongCLIPEvaluator().run(
        storage,
        image_key=image_key,
        text_key=text_key,
        output_key="longclip_score",
    )
    storage.step()

    VQAScoreEvaluator().run(
        storage,
        image_key=image_key,
        text_key=text_key,
        output_key="vqa_score",
    )
    storage.step()  # 最终结果读取

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
