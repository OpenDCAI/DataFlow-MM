import os
import sys
import pandas as pd
import colorlog
import warnings

from dataflow import get_logger
from dataflow.utils.storage import FileStorage
from dataflow.operators.core_vision import (
    ImageSensitiveFilter,
    ImageAestheticFilter,
    ImageClipFilter,
    ImageComplexityFilter,
    ImageConsistencyFilter,
    ImageDiversityFilter,
)

warnings.filterwarnings("ignore")


CAPTION_PATH = "./dataflow/example/image_to_text_pipeline/caption_result.jsonl"
QA_PATH = "./dataflow/example/image_to_text_pipeline/qa_result.jsonl"
CACHE_DIR = "./cache_filter"
PREFIX = "imgtext_filter_step"
MERGED_INPUT_PATH = os.path.join(CACHE_DIR, f"{PREFIX}_merged_input.jsonl")
FINAL_SAVE_PATH = "./cache_local/final_filtered.jsonl"


logger = get_logger()
for h in logger.handlers:
    h.setFormatter(colorlog.ColoredFormatter(
        "%(log_color)s[%(levelname)s] %(message)s",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "white",
            "SUCCESS": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
    ))


def prepare_merged_input():
    if not os.path.exists(CAPTION_PATH):
        logger.error(f"Caption file not found: {CAPTION_PATH}")
        sys.exit(1)
    if not os.path.exists(QA_PATH):
        logger.error(f"QA file not found: {QA_PATH}")
        sys.exit(1)

    logger.info(f"Reading caption data from: {CAPTION_PATH}")
    cap_df = pd.read_json(CAPTION_PATH, lines=True)

    logger.info(f"Reading QA data from: {QA_PATH}")
    qa_df = pd.read_json(QA_PATH, lines=True)

    df = pd.merge(cap_df, qa_df, on="image", how="inner")
    logger.info(f"Merged samples: {len(df)}")

    df["question"] = df["qa"].str.split("Answer:").str[0].str.replace("Question:", "").str.strip()
    df["answer"] = df["qa"].str.split("Answer:").str[1].str.strip()
    df.drop(columns=["qa"], inplace=True)

    os.makedirs(os.path.dirname(MERGED_INPUT_PATH), exist_ok=True)
    df.to_json(MERGED_INPUT_PATH, orient="records", lines=True, force_ascii=False)
    logger.info(f"Merged input saved to: {MERGED_INPUT_PATH}")


def main():
    prepare_merged_input()

    storage = FileStorage(
        first_entry_file_name=MERGED_INPUT_PATH,
        cache_path=CACHE_DIR,
        file_name_prefix=PREFIX,
        cache_type="jsonl",
    )

    storage.step()
    df0 = storage.read(output_type="dataframe")
    logger.info(f"Pipeline starts with {len(df0)} samples")

    sensitive_filter = ImageSensitiveFilter()
    aesthetic_filter = ImageAestheticFilter()
    clip_filter = ImageClipFilter(model_name="../ckpt/clip-vit-base-patch32")
    complexity_filter = ComplexityFilter(threshold=0.3, min_k=1)
    consistency_filter = ImageConsistencyFilter(threshold=0.35)
    diversity_filter = ImageDiversityFilter(text_thresh=0.75, hash_size=8, img_dist_thresh=5)

    logger.info("Running ImageSensitiveFilter...")
    sensitive_filter.run(
        storage,
        input_image_key="image",
        input_text_keys=["caption", "question", "answer"],
    )
    storage.step()
    df1 = storage.read(output_type="dataframe")
    logger.info(f"After ImageSensitiveFilter: {len(df1)} samples left")

    logger.info("Running ImageAestheticFilter...")
    aesthetic_filter.run(
        storage,
        input_image_key="image",
    )
    storage.step()
    df2 = storage.read(output_type="dataframe")
    logger.info(f"After ImageAestheticFilter: {len(df2)} samples left")

    logger.info("Running ImageClipFilter...")
    clip_filter.run(
        storage,
        input_image_key="image",
        input_caption_key="caption",
        threshold=0.23,
    )
    storage.step()
    df3 = storage.read(output_type="dataframe")
    logger.info(f"After ImageClipFilter: {len(df3)} samples left")

    logger.info("Running ComplexityFilter...")
    complexity_filter.run(
        storage,
        input_caption_key="caption",
    )
    storage.step()
    df4 = storage.read(output_type="dataframe")
    logger.info(f"After ComplexityFilter: {len(df4)} samples left")

    logger.info("Running ImageConsistencyFilter...")
    consistency_filter.run(
        storage,
        input_caption_key="caption",
        input_question_key="question",
        input_answer_key="answer",
    )
    storage.step()
    df5 = storage.read(output_type="dataframe")
    logger.info(f"After ImageConsistencyFilter: {len(df5)} samples left")

    logger.info("Running ImageDiversityFilter...")
    diversity_filter.run(
        storage,
        input_image_key="image",
        input_text_key="caption",
    )
    storage.step()
    df_final = storage.read(output_type="dataframe")
    logger.info(f"After ImageDiversityFilter: {len(df_final)} samples left")

    os.makedirs(os.path.dirname(FINAL_SAVE_PATH), exist_ok=True)
    df_final.to_json(FINAL_SAVE_PATH, lines=True, orient="records", force_ascii=False)
    logger.success(f"All done! Final filtered data saved to: {FINAL_SAVE_PATH}")


if __name__ == "__main__":
    main()
