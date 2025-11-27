import os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY


@OPERATOR_REGISTRY.register()
class ImageSensitiveFilter(OperatorABC):
    LABEL_DESCRIPTIONS = {
        "sexual_content": "The text describes sexual content, nudity or pornography.",
        "violence": "The text describes or encourages physical violence, injury, or killing.",
        "self_harm": "The text mentions suicide, self-harm or wanting to die.",
        "hate": "The text attacks or insults a group based on race, religion, gender or similar traits.",
        "harassment": "The text insults, bullies or harasses a person.",
        "threat": "The text threatens to harm a person or a group.",
    }

    def __init__(
        self,
        model_name: str = "facebook/bart-large-mnli",
        threshold: float = 0.5,
        device: str | None = None,
    ):
        self.logger = get_logger()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=True,
            use_fast=True,
            weights_only=False,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            local_files_only=True,
            use_safetensors=True,
            weights_only=False,
        ).to(self.device).eval()
        self.threshold = threshold

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "ImageSensitiveFilter 算子：基于 BART Large MNLI 零样本文本推理模型，对与图像相关的多列文本进行多标签安全检测，"
                "识别涉黄、暴力、自残、仇恨、骚扰、威胁等高风险内容，并据此过滤样本。\n"
                "输入：通过 run(storage, input_image_key, input_text_keys) 调用，其中 input_image_key 指定图像路径所在列，"
                "input_text_keys 为若干文本列名列表（如 caption、question、answer 等），算子会逐行读取这些字段；\n"
                "输出：对每一行样本，首先检查图像路径是否存在，其次使用 MNLI 模型分别评估每个文本与多种风险描述（如“文本是否描述暴力或杀戮”）之间的蕴含概率，"
                "当任意文本在任意风险类别上的蕴含概率高于阈值 threshold 时，将该样本视为不安全并剔除；过滤后的 DataFrame 写回存储，"
                "并返回 [input_image_key] + input_text_keys 作为后续算子的输入列名；\n"
                "功能：无需人工维护关键词黑名单，通过自然语言风险描述 + 零样本 NLI 模型实现对多种敏感内容的统一检测，"
                "在保持较强灵活性的同时，适配不同任务下对图文数据安全合规清洗的需求。"
            )
        else:
            return (
                "ImageSensitiveFilter operator: uses a BART Large MNLI zero-shot NLI model to perform multi-label safety "
                "detection on text fields associated with images, identifying sexual content, violence, self-harm, hate, "
                "harassment and threats, and filtering unsafe samples.\n"
                "Inputs: called via run(storage, input_image_key, input_text_keys), where input_image_key is the column "
                "containing image paths and input_text_keys is a list of text column names (e.g., caption, question, answer). "
                "For each row, the operator loads the image path (for existence checking) and all specified text fields;\n"
                "Output: for each sample, it first checks that the image path exists, then uses the MNLI model to estimate, "
                "for each text, the entailment probability of several natural-language risk descriptions (such as “the text "
                "describes physical violence or killing”). If any text strongly entails any risk description with probability "
                "above the threshold, the sample is marked unsafe and removed. The filtered DataFrame is written back to storage, "
                "and [input_image_key] + input_text_keys is returned as the list of column names for downstream operators;\n"
                "Function: replaces rigid keyword blacklists with flexible zero-shot NLI over risk descriptions, providing a "
                "unified and extensible way to detect various sensitive content types when cleaning multimodal datasets."
            )

    @torch.no_grad()
    def score_text(self, text: str) -> dict:
        if not text or not text.strip():
            return {k: 0.0 for k in self.LABEL_DESCRIPTIONS}
        scores = {}
        for label, desc in self.LABEL_DESCRIPTIONS.items():
            inputs = self.tokenizer(
                text,
                desc,
                return_tensors="pt",
                truncation=True,
                padding=True,
            ).to(self.device)
            logits = self.model(**inputs).logits[0]
            probs = torch.softmax(logits, dim=-1)
            entail_score = float(probs[2].item())
            scores[label] = entail_score
        return scores

    def is_safe_text(self, text: str) -> bool:
        scores = self.score_text(text)
        max_risk = max(scores.values()) if scores else 0.0
        return max_risk < self.threshold

    def is_safe_image(self, image_path: str) -> bool:
        if not image_path:
            return False
        if not os.path.exists(image_path):
            self.logger.warning(f"Image not found: {image_path}")
            return False
        return True

    def is_safe(self, image_path: str, *texts) -> bool:
        if not self.is_safe_image(image_path):
            return False
        for t in texts:
            if not self.is_safe_text(t):
                return False
        return True

    def run(self, storage: DataFlowStorage, input_image_key: str, input_text_keys: list):
        df = storage.read("dataframe")
        refined_mask = []
        for i, row in enumerate(tqdm(df.itertuples(), desc=f"Implementing {self.__class__.__name__}")):
            img_path = getattr(row, input_image_key)
            texts = [getattr(row, k) for k in input_text_keys]
            safe = self.is_safe(img_path, *texts)
            refined_mask.append(safe)
            if not safe:
                self.logger.debug(
                    f"Sensitive content detected at row {i}: {img_path}, "
                    f"{[str(t)[:80] for t in texts]}"
                )
        df = df[refined_mask].reset_index(drop=True)
        storage.write(df)
        return [input_image_key] + input_text_keys
