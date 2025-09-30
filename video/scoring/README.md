# Scoring

## Aesthetic Score

To evaluate the aesthetic quality of videos, we use the scoring model from [CLIP+MLP Aesthetic Score Predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor). This model is trained on 176K SAC (Simulacra Aesthetic Captions) pairs, 15K LAION-Logos (Logos) pairs, and 250K AVA (The Aesthetic Visual Analysis) image-text pairs.

The aesthetic score is between 1 and 10, where 5.5 can be considered as the threshold for fair aesthetics, and 6.5 for high aesthetics. Good text-to-image models can achieve a score of 7.0 or higher.

First, download the scoring model to `./checkpoints/aesthetic.pth`. Skip this step if you already follow the installation instructions in [README](../README.md).

```bash
wget https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac+logos+ava1-l14-linearMSE.pth -O checkpoints/aesthetic.pth
wget https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt -O ViT-L-14.pt
```

Then, run the following command to compute aesthetic scores.
```bash
torchrun --nproc_per_node ${GPU_NUM} scoring/aesthetic/inference.py \
  ${ROOT_META}/clips_info.csv \
  --bs 16 \
  --num_workers ${NUM_WORKERS} \
  --fig_load_dir ${ROOT_FIG}
```

## Luminance Score

Luminance was calculated for the first, middle, and last frames using the standard formula $L = 0.2126 R + 0.7152 G + 0.0722 B$, where $R$, $G$, and $B$ are the respective channel values. Clips with average luminance outside the range [20, 140], either too dark or too bright, were excluded, ensuring that only videos with proper exposure were retained.

Run the following command to compute luminance scores.
```bash
torchrun --nproc_per_node ${GPU_NUM} scoring/luminance/inference.py \
  ${ROOT_META}/clips_info.csv \
  --bs 16 \
  --num_workers ${NUM_WORKERS} \
  --fig_load_dir ${ROOT_FIG}
```

## Motion Score
Conventional motion analysis using optical flow is computationally expensive and less effective for videos with complex motion patterns. Inspired by Open-Sora 2.0, we adopted a lightweight VMAF-based motion analysis method integrated into FFMPEG. This method yields a motion score between 0 and 20.

Clips with scores outside the valid range of [2, 14], either too static (scores $<$ 2) or excessively chaotic (scores $>$ 14), were filtered out.

Run the following command to compute motion scores.
```bash
python scoring/motion/inference.py ${ROOT_META}/clips_info.csv \
  --temp_save_dir ${ROOT_TEMP} \
  --num_workers $((GPU_NUM * 4)) \
  --gpu_num ${GPU_NUM}
```

## OCR
For text detection, we used the latest release of PaddleOCR, which offers high accuracy and robust multilingual support. We processed the first, middle, and last frames of each clip to detect text regions, computing the ratio of text area to frame size. Clips where the text area exceeded 30% were removed, as these were considered informational rather than visual.

Run the following command to compute OCR scores.
```bash
python scoring/ocr/inference.py ${ROOT_META}/clips_info.csv \
  --fig_load_dir ${ROOT_FIG} \
  --num_workers $((GPU_NUM * 4)) \
  --gpu_num ${GPU_NUM}
```