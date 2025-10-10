import pandas as pd
from transformers import CLIPProcessor, CLIPModel
from dataflow.operators.core_vision import (
    SensitiveFilter, AestheticFilter, CatFilter,
    ComplexityFilter, ConsistencyFilter, DiversityFilter
)
from dataflow import get_logger
from PIL import Image
import torch, colorlog, warnings
warnings.filterwarnings("ignore")


logger = get_logger()
for h in logger.handlers:
    h.setFormatter(colorlog.ColoredFormatter(
        '%(log_color)s[%(levelname)s] %(message)s',
        log_colors={
            'DEBUG': 'cyan', 'INFO': 'white', 'SUCCESS': 'green',
            'WARNING': 'yellow', 'ERROR': 'red', 'CRITICAL': 'red,bg_white'
        }
    ))

def batch_clip_filter(df, processor, model, batch_size=32, threshold=0.23):
    keep = []
    model.eval()
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        imgs = [
            processor.image_processor(
                images=Image.open(p).convert("RGB"),
                return_tensors="pt"
            )["pixel_values"][0]
            for p in batch["image"]
        ]
        caps = batch["caption"].tolist()
        inputs = processor(
            text=caps,
            images=imgs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        ).to("cuda")
        with torch.no_grad():
            txt_feats = model.get_text_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )
            img_feats = model.get_image_features(pixel_values=inputs["pixel_values"])
        sims = (txt_feats * img_feats).sum(dim=-1).cpu().tolist()
        keep.extend([s >= threshold for s in sims])
    return df[keep]

def main():  
    caption_path = "/data0/happykeyan/workspace/DataFlow-MM/dataflow/example/image_to_text_pipeline/caption_result.jsonl"
    qa_path = "/data0/happykeyan/workspace/DataFlow-MM/dataflow/example/image_to_text_pipeline/qa_result.jsonl"
    save_path = "/data0/happykeyan/workspace/DataFlow-MM/dataflow/example/test_image_filter/final_filtered.jsonl"

    logger.info(f"Reading caption data from: {caption_path}")
    cap_df = pd.read_json(caption_path, lines=True)
    logger.info(f"Reading QA data from: {qa_path}")
    qa_df = pd.read_json(qa_path, lines=True)
    df = pd.merge(cap_df, qa_df, on="image", how="inner")
    logger.info(f"Loaded {len(df)} merged samples")

    df["question"] = df["qa"].str.split("Answer:").str[0].str.replace("Question:", "").str.strip()
    df["answer"] = df["qa"].str.split("Answer:").str[1].str.strip()
    df.drop(columns=["qa"], inplace=True)

    logger.info("Running SensitiveFilter…")
    sr = SensitiveFilter()
    df = df[df.apply(lambda r: sr.is_safe(r["image"], r["caption"], r["question"], r["answer"]), axis=1)]
    logger.info(f"After SensitiveFilter: {len(df)} samples left")

    logger.info("Running AestheticFilter…")
    af = AestheticFilter()
    df = df[df["image"].apply(af.is_quality)]
    logger.info(f"After AestheticFilter: {len(df)} samples left")

    logger.info("Running CLIPFilter…")
    processor = CLIPProcessor.from_pretrained("/data0/happykeyan/Models/clip-vit-base-patch32", use_fast=True)
    processor.image_processor._valid_processor_keys = []
    model = CLIPModel.from_pretrained("/data0/happykeyan/Models/clip-vit-base-patch32").cuda()
    df = batch_clip_filter(df, processor, model)
    logger.info(f"After CLIPFilter: {len(df)} samples left")

    logger.info("Running CatFilter…")
    cat = CatFilter(min_triples=1, ocr_overlap_threshold=0.2)
    cat.is_not_ocr_only = lambda image, caption: True
    df = df[df.apply(lambda r: cat.is_consistent(r["image"], r["caption"]), axis=1)]
    logger.info(f"After CatFilter: {len(df)} samples left")

    logger.info("Running ComplexityFilter…")
    nli = ComplexityFilter(threshold=0.4, min_k=1)
    df = df[df.apply(lambda r: nli.is_valid(r["image"], r["caption"]), axis=1)]
    logger.info(f"After ComplexityFilter: {len(df)} samples left")

    logger.info("Running ConsistencyFilter…")
    qa_ref = ConsistencyFilter(threshold=0.35)
    df = df[df.apply(lambda r: qa_ref.is_consistent(r["caption"], r["question"], r["answer"])[0], axis=1)]
    logger.info(f"After ConsistencyFilter: {len(df)} samples left")

    logger.info("Running DiversityFilter…")
    dr = DiversityFilter(text_thresh=0.75, hash_size=8, img_dist_thresh=5)
    keep = [dr.check_diversity(r["image"], r["caption"])[0] for _, r in df.iterrows()]
    df = df[keep]
    logger.info(f"After DiversityFilter: {len(df)} samples left")

    df.to_json(save_path, lines=True, orient="records")
    logger.success(f"All done! Final filtered data saved to: {save_path}")

if __name__ == "__main__":
    main()