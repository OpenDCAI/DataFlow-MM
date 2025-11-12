import torch
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY

@OPERATOR_REGISTRY.register()
class ClipFilter(OperatorABC):
    def __init__(
        self,
        model_name: str = "../ckpt/clip-vit-base-patch32",
        device: str = None
    ):
        self.logger = get_logger()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = CLIPProcessor.from_pretrained(model_name, use_safetensors=True, weights_only=False)
        self.model = CLIPModel.from_pretrained(model_name, use_safetensors=True, weights_only=False).to(self.device).eval()

    @staticmethod
    def get_desc(lang="zh"):
        return "图文一致性过滤（CLIP相似度）" if lang == "zh" else "Image-text consistency filter (CLIP similarity)."

    def compute_similarity(self, image_path: str, text: str) -> float:
        try:
            image = Image.open(image_path).convert("RGB")
        except (FileNotFoundError, UnidentifiedImageError):
            self.logger.warning(f"Failed to load image: {image_path}")
            return 0.0
        if not text or text.strip() == "":
            self.logger.warning(f"Empty text for image: {image_path}")
            return 0.0
        inputs = self.processor(text=[text], images=[image], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            img_emb = outputs.image_embeds
            txt_emb = outputs.text_embeds
        img_norm = img_emb / img_emb.norm(p=2, dim=-1, keepdim=True)
        txt_norm = txt_emb / txt_emb.norm(p=2, dim=-1, keepdim=True)
        sim = (img_norm @ txt_norm.T).cpu().item()
        if not (0.0 <= sim <= 1.0):
            sim = max(min(sim, 1.0), 0.0)
        return sim

    def is_consistent(self, image_path: str, caption: str, threshold: float = 0.25) -> bool:
        return self.compute_similarity(image_path, caption) >= threshold

    def run(self, storage: DataFlowStorage, image_key: str = "image", caption_key: str = "caption", threshold: float = 0.25):
        self.image_key = image_key
        self.caption_key = caption_key
        dataframe = storage.read("dataframe")
        refined_mask = []
        for i, row in enumerate(tqdm(dataframe.itertuples(), desc=f"Implementing {self.__class__.__name__}")):
            img_path = getattr(row, self.image_key)
            cap = getattr(row, self.caption_key)
            sim = self.compute_similarity(img_path, cap)
            ok = sim >= threshold
            refined_mask.append(ok)
            if not ok:
                self.logger.debug(f"CLIP failed at row {i}: sim={sim:.3f}, img={img_path}, cap={cap}")
        dataframe = dataframe[refined_mask].reset_index(drop=True)
        storage.write(dataframe)
        return [self.image_key, self.caption_key]
