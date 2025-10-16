import torch
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from transformers import BlipProcessor, BlipForQuestionAnswering

from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY


@OPERATOR_REGISTRY.register()
class VQAScoreEvaluator(OperatorABC):
    def __init__(
        self,
        model_name: str = "/data0/happykeyan/DataFlow-MM/Dataflow-MM-Preview/ckpt/blip-vqa-base",
        device: str = None,
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
    def get_desc(lang="zh"):
        return "VQA Score（BLIP-Yes/No 概率）" if lang == "zh" else "VQA Score via BLIP Yes/No probability."

    @torch.no_grad()
    def compute_yes_prob(self, image_path: str, text: str) -> float:
        try:
            image = Image.open(image_path).convert("RGB")
        except (FileNotFoundError, UnidentifiedImageError) as e:
            self.logger.warning(f"Failed to load image: {image_path}. Reason: {e}")
            return 0.0
        if not text or text.strip() == "":
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
        if not (0.0 <= p <= 1.0):
            p = max(min(p, 1.0), 0.0)
        return p

    def run(self, storage: DataFlowStorage, image_key: str = "image_path", text_key: str = "text", output_key: str = "vqa_score"):
        df = storage.read("dataframe")
        scores = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Implementing {self.__class__.__name__}"):
            img_path = row[image_key]
            text = str(row[text_key])
            score = self.compute_yes_prob(img_path, text)
            scores.append(score)
        df[output_key] = scores
        storage.write(df)
        return [output_key]
