import logging
from typing import List, Optional

import torch
import torch.nn.functional as F
import pandas as pd
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor, BlipProcessor, BlipForQuestionAnswering

from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY


def _cos_sim_unit(a: torch.Tensor, b: torch.Tensor) -> float:
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    v = float((a * b).sum())
    return (v + 1.0) / 2.0


def _safe_open(path: str, logger: logging.Logger) -> Optional[Image.Image]:
    try:
        return Image.open(path).convert("RGB")
    except (FileNotFoundError, UnidentifiedImageError) as e:
        logger.warning(f"Failed to load image: {path}. Reason: {e}")
        return None


class _BaseCLIP(OperatorABC):
    _processor = None
    _model = None

    def __init__(self, model_name: str, device: Optional[str] = None, local_only: bool = True):
        self.logger = get_logger()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if _BaseCLIP._processor is None or _BaseCLIP._model is None:
            _BaseCLIP._processor = CLIPProcessor.from_pretrained(model_name)
            _BaseCLIP._model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = _BaseCLIP._processor
        self.model = _BaseCLIP._model

    @torch.no_grad()
    def _clip_sim(self, image: Image.Image, text: str) -> float:
        if image is None or not text:
            return 0.0
        inputs = self.processor(text=[text], images=[image], return_tensors="pt", padding=True).to(self.device)
        out = self.model(**inputs)
        return _cos_sim_unit(out.image_embeds[0], out.text_embeds[0])


@OPERATOR_REGISTRY.register()
class CLIPEvaluator(_BaseCLIP):
    def __init__(
        self,
        model_name: str = "/data0/happykeyan/DataFlow-MM/Dataflow-MM-Preview/ckpt/clip-vit-base-patch32",
        device: Optional[str] = None,
        local_only: bool = True,
    ):
        super().__init__(model_name=model_name, device=device, local_only=local_only)

    @staticmethod
    def get_desc(lang: str = "zh"):
        return "CLIP 图文对齐分数" if lang == "zh" else "CLIP image-text alignment score."

    def run(self, storage: DataFlowStorage, image_key: str = "image_path", text_key: str = "text", output_key: str = "clip_score"):
        df = storage.read("dataframe")
        scores: List[float] = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="CLIPEvaluator"):
            img = _safe_open(row[image_key], self.logger)
            text = str(row[text_key])
            scores.append(self._clip_sim(img, text))
        df[output_key] = scores
        storage.write(df)
        return [output_key]


@OPERATOR_REGISTRY.register()
class LongCLIPEvaluator(OperatorABC):
    def __init__(
        self,
        ckpt_path: str = "/data0/happykeyan/DataFlow-MM/Dataflow-MM-Preview/ckpt/LongCLIP-L-336px/longclip-L@336px.pt",
        device: Optional[str] = None,
    ):
        self.logger = get_logger()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        import sys
        # sys.path.insert(0, "/data0/happykeyan/DataFlow-MM/Dataflow-MM-Preview/Long-CLIP")
        from .model import longclip
        self.longclip = longclip
        self.model, self.preprocess = longclip.load(ckpt_path, device=self.device)
        self.tokenizer = longclip.tokenize
        self.model.eval()

    @staticmethod
    def get_desc(lang: str = "zh"):
        return "LongCLIP 长文本图文对齐分数" if lang == "zh" else "LongCLIP long-text image-text score."

    @torch.no_grad()
    def _clip_sim_longclip(self, image: Image.Image, text: str) -> float:
        if image is None or not text:
            return 0.0
        img_t = self.preprocess(image).unsqueeze(0).to(self.device)
        txt_t = self.tokenizer([text]).to(self.device)
        img_feat = self.model.encode_image(img_t)
        txt_feat = self.model.encode_text(txt_t)
        return _cos_sim_unit(img_feat[0], txt_feat[0])

    def run(self, storage: DataFlowStorage, image_key: str = "image_path", text_key: str = "text", output_key: str = "longclip_score"):
        df = storage.read("dataframe")
        scores: List[float] = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="LongCLIPEvaluator(LongCLIP)"):
            img = _safe_open(row[image_key], self.logger)
            text = str(row[text_key])
            scores.append(self._clip_sim_longclip(img, text))
        df[output_key] = scores
        storage.write(df)
        return [output_key]


@OPERATOR_REGISTRY.register()
class VQAScoreEvaluator(OperatorABC):
    def __init__(
        self,
        model_name: str = "/data0/happykeyan/DataFlow-MM/Dataflow-MM-Preview/ckpt/blip-vqa-base",
        device: Optional[str] = None,
        local_only: bool = True,
    ):
        self.logger = get_logger()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = BlipProcessor.from_pretrained(model_name, local_files_only=local_only)
        self.model = BlipForQuestionAnswering.from_pretrained(model_name, local_files_only=local_only).to(self.device).eval()
        tok = self.processor.tokenizer
        self.yes_ids = tok("yes", add_special_tokens=True, return_tensors="pt").input_ids.to(self.device)
        self.no_ids = tok("no", add_special_tokens=True, return_tensors="pt").input_ids.to(self.device)

    @staticmethod
    def get_desc(lang: str = "zh"):
        return "VQA Score（BLIP-Yes/No 概率）" if lang == "zh" else "VQA Score via BLIP Yes/No probability."

    @torch.no_grad()
    def _score_yesno(self, image: Image.Image, text: str) -> float:
        if image is None or not text:
            return 0.0
        question = f"Does this image match the description: {text}? Answer yes or no."
        inputs = self.processor(images=image, text=question, return_tensors="pt").to(self.device)
        out_yes = self.model(**inputs, labels=self.yes_ids)
        out_no = self.model(**inputs, labels=self.no_ids)
        ly = float(out_yes.loss.item())
        ln = float(out_no.loss.item())
        py = torch.exp(torch.tensor(-ly))
        pn = torch.exp(torch.tensor(-ln))
        p = float((py / (py + pn + 1e-8)).item())
        return p

    def run(self, storage: DataFlowStorage, image_key: str = "image_path", text_key: str = "text", output_key: str = "vqa_score"):
        df = storage.read("dataframe")
        scores: List[float] = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="VQAScoreEvaluator(BLIP)"):
            img = _safe_open(row[image_key], self.logger)
            text = str(row[text_key])
            scores.append(self._score_yesno(img, text))
        df[output_key] = scores
        storage.write(df)
        return [output_key]


__all__ = [
    "CLIPEvaluator",
    "LongCLIPEvaluator",
    "VQAScoreEvaluator",
]
