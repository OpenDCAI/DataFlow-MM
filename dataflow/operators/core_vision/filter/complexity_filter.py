from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY

@OPERATOR_REGISTRY.register()
class ComplexityFilter(OperatorABC):
    CAPS = [
        "color",
        "shape",
        "object recognition",
        "action recognition",
        "text recognition",
        "spatial recognition",
        "counting",
        "spatial relationship",
        "object interaction",
        "scene understanding"
    ]
    TEMPLATE = "The following text describes {}."

    def __init__(self,
                 model_name: str = "facebook/bart-large-mnli",
                 threshold: float = 0.4,
                 min_k: int = 3,
                 device: str = None):
        self.logger = get_logger()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.thresh = threshold
        self.min_k = min_k

    @staticmethod
    def get_desc(self, lang):
        return "NLI能力属性过滤（caption能力丰富度）" if lang == "zh" else "NLI capability filter (caption capability diversity)."

    def detect_caps(self, caption: str) -> list:
        if not caption or len(caption.strip()) < 5:
            return []
        detected = []
        try:
            for cap in self.CAPS:
                hyp = self.TEMPLATE.format(cap)
                inputs = self.tokenizer(caption, hyp, return_tensors="pt", truncation=True).to(self.device)
                with torch.no_grad():
                    logits = self.model(**inputs).logits[0]
                    probs  = torch.softmax(logits, dim=-1)
                if probs[2].item() >= self.thresh:
                    detected.append(cap)
        except Exception as e:
            self.logger.warning(f"[CompositionalComplexityRefiner] Error: {e}")
        return detected

    def is_valid(self, image_path: str, caption: str) -> bool:
        caps = self.detect_caps(caption)
        return len(caps) >= self.min_k

    def run(self, storage: DataFlowStorage, caption_key: str):
        self.caption_key = caption_key
        dataframe = storage.read("dataframe")
        refined_mask = []
        for i, row in enumerate(tqdm(dataframe.itertuples(), desc=f"Implementing {self.__class__.__name__}")):
            caption = getattr(row, self.caption_key)
            ok = self.is_valid("", caption)
            refined_mask.append(ok)
            if not ok:
                self.logger.debug(f"Filtered weak caption at row {i}: {caption[:40]}")
        dataframe = dataframe[refined_mask].reset_index(drop=True)
        output_file = storage.write(dataframe)
        return [self.caption_key]
