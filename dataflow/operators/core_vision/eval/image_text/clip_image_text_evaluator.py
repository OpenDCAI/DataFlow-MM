import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY


def _cos_sim_unit(a: torch.Tensor, b: torch.Tensor) -> float:
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    v = float((a * b).sum())
    return (v + 1.0) / 2.0


@OPERATOR_REGISTRY.register()
class CLIPEvaluator(OperatorABC):
    def __init__(
        self,
        model_name: str = "/data0/happykeyan/workspace/ckpt/clip-vit-base-patch32",
        device: str = None
    ):
        self.logger = get_logger()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = CLIPProcessor.from_pretrained(model_name, use_safetensors=True)
        self.model = CLIPModel.from_pretrained(model_name, use_safetensors=True).to(self.device).eval()

    @staticmethod
    def get_desc(lang="zh"):
        return "CLIP 图文对齐分数" if lang == "zh" else "CLIP image-text alignment score."

    @torch.no_grad()
    def compute_similarity(self, image_path: str, text: str) -> float:
        try:
            image = Image.open(image_path).convert("RGB")
        except (FileNotFoundError, UnidentifiedImageError) as e:
            self.logger.warning(f"Failed to load image: {image_path}. Reason: {e}")
            return 0.0
        if not text or text.strip() == "":
            return 0.0
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=77
        ).to(self.device)
        outputs = self.model(**inputs)
        sim = _cos_sim_unit(outputs.image_embeds[0], outputs.text_embeds[0])
        if not (0.0 <= sim <= 1.0):
            sim = max(min(sim, 1.0), 0.0)
        return float(sim)

    def run(self, storage: DataFlowStorage, image_key: str = "image_path", text_key: str = "text", output_key: str = "clip_score"):
        df = storage.read("dataframe")
        scores = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Implementing {self.__class__.__name__}"):
            img_path = row[image_key]
            text = str(row[text_key])
            score = self.compute_similarity(img_path, text)
            scores.append(score)
        df[output_key] = scores
        storage.write(df)
        return [output_key]
