from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY

@OPERATOR_REGISTRY.register()
class ConsistencyFilter(OperatorABC):
    def __init__(
        self,
        model_name: str = "/data0/happykeyan/workspace/ckpt/bart-large-mnli",
        threshold: float = 0.35,
        device: str = None
    ):
        self.logger = get_logger()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            local_files_only=True,
            use_safetensors=True
        ).to(self.device).eval()
        self.threshold = threshold

    @staticmethod
    def get_desc(lang="zh"):
        return "Caption-Question-Answer三字段连贯性过滤" if lang == "zh" else "Caption-Question-Answer Consistency Filter (NLI)."

    def entailment_score(self, caption: str, question: str, answer: str) -> float:
        premise = (caption or "").strip() + " " + (question or "").strip()
        hypothesis = (answer or "").strip()
        if len(hypothesis) == 0:
            return 0.0
        inputs = self.tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits[0]
            probs = torch.softmax(logits, dim=-1)
        return probs[2].item()

    def is_consistent(self, caption: str, question: str, answer: str):
        p = self.entailment_score(caption, question, answer)
        return p >= self.threshold, p

    def run(self, storage: DataFlowStorage, caption_key: str, question_key: str, answer_key: str):
        self.caption_key = caption_key
        self.question_key = question_key
        self.answer_key = answer_key
        dataframe = storage.read("dataframe")
        refined_mask = []
        for i, row in enumerate(tqdm(dataframe.itertuples(), desc=f"Implementing {self.__class__.__name__}")):
            caption = getattr(row, self.caption_key)
            question = getattr(row, self.question_key)
            answer = getattr(row, self.answer_key)
            ok, entail_score = self.is_consistent(caption, question, answer)
            refined_mask.append(ok)
            if not ok:
                self.logger.debug(f"Filtered at row {i}: entail_score={entail_score:.3f}, c={str(caption)[:30]}, q={str(question)[:30]}, a={str(answer)[:30]}")
        dataframe = dataframe[refined_mask].reset_index(drop=True)
        storage.write(dataframe)
        return [self.caption_key, self.question_key, self.answer_key]
