import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

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
class LongCLIPEvaluator(OperatorABC):
    def __init__(
        self,
        ckpt_path: str = "/data0/happykeyan/DataFlow-MM/Dataflow-MM-Preview/ckpt/LongCLIP-L-336px/longclip-L@336px.pt",
        device: str = None,
    ):
        self.logger = get_logger()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        from .model import longclip
        self.longclip = longclip
        self.model, self.preprocess = longclip.load(ckpt_path, device=self.device)
        self.tokenizer = longclip.tokenize
        self.model.eval()
        self.context_length = 248

    @staticmethod
    def get_desc(lang="zh"):
        return "LongCLIP 长文本图文对齐分数" if lang == "zh" else "LongCLIP long-text image-text score."

    def _tokenize_safe(self, text: str):
        t = " ".join(str(text).split())
        try:
            return self.tokenizer([t], context_length=self.context_length, truncate=True)
        except TypeError:
            pass
        curr = t
        while True:
            try:
                return self.tokenizer([curr])
            except RuntimeError:
                if len(curr) <= 8:
                    return self.tokenizer([curr[:8]])
                curr = curr[: int(len(curr) * 0.8)]

    @torch.no_grad()
    def compute_similarity(self, image_path: str, text: str) -> float:
        try:
            image = Image.open(image_path).convert("RGB")
        except (FileNotFoundError, UnidentifiedImageError) as e:
            self.logger.warning(f"Failed to load image: {image_path}. Reason: {e}")
            return 0.0
        if not text or text.strip() == "":
            return 0.0
        img_t = self.preprocess(image).unsqueeze(0).to(self.device)
        txt_t = self._tokenize_safe(text).to(self.device)
        img_feat = self.model.encode_image(img_t)
        txt_feat = self.model.encode_text(txt_t)
        sim = _cos_sim_unit(img_feat[0], txt_feat[0])
        if not (0.0 <= sim <= 1.0):
            sim = max(min(sim, 1.0), 0.0)
        return float(sim)

    def run(self, storage: DataFlowStorage, image_key: str = "image_path", text_key: str = "text", output_key: str = "longclip_score"):
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
