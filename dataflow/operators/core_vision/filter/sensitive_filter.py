import os
from PIL import Image
from transformers import pipeline
from tqdm import tqdm
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY

@OPERATOR_REGISTRY.register()
class SensitiveFilter(OperatorABC):
    def __init__(self,
                 img_model_name="Falconsai/nsfw_image_detection",
                 txt_model_name="unitary/toxic-bert",
                 img_thresh=0.5,
                 txt_thresh=0.5):
        self.logger = get_logger()
        self.img_pipe = pipeline("image-classification", model=img_model_name)
        self.txt_pipe = pipeline("text-classification", model=txt_model_name)
        self.img_thresh = img_thresh
        self.txt_thresh = txt_thresh
        self.img_sensitive_labels = {"porn", "hentai", "sexy", "nsfw"}
        self.txt_sensitive_labels = {"toxic", "offensive", "hate", "obscene", "threat", "sexual_explicit", "identity_attack"}

    @staticmethod
    def get_desc(self, lang):
        return "过滤图片与文本中的敏感内容（涉黄、暴力、歧视等）" if lang == "zh" else "Filter sensitive content in images and text (porn, violence, hate, etc)."

    def is_safe_image(self, image_path: str) -> bool:
        if not os.path.exists(image_path):
            self.logger.warning(f"Image not found: {image_path}")
            return False
        image = Image.open(image_path).convert("RGB")
        results = self.img_pipe(image)
        for out in results:
            label = out["label"].lower()
            score = out["score"]
            if label in self.img_sensitive_labels and score >= self.img_thresh:
                return False
        return True

    def is_safe_text(self, text: str) -> bool:
        results = self.txt_pipe(text)
        for out in results:
            label = out["label"].lower()
            score = out["score"]
            if label in self.txt_sensitive_labels and score >= self.txt_thresh:
                return False
        return True

    def is_safe(self, image_path: str, *texts) -> bool:
        if not self.is_safe_image(image_path):
            return False
        for text in texts:
            if not self.is_safe_text(text):
                return False
        return True

    def run(self, storage: DataFlowStorage, image_key: str, text_keys: list):
        self.image_key = image_key
        self.text_keys = text_keys
        dataframe = storage.read("dataframe")
        refined_mask = []
        for i, row in enumerate(tqdm(dataframe.itertuples(), desc=f"Implementing {self.__class__.__name__}")):
            img_path = getattr(row, self.image_key)
            texts = [getattr(row, k) for k in self.text_keys]
            safe = self.is_safe(img_path, *texts)
            refined_mask.append(safe)
            if not safe:
                self.logger.debug(f"Sensitive content detected at row {i}: {img_path}, {[t[:30] for t in texts]}")
        dataframe = dataframe[refined_mask].reset_index(drop=True)
        output_file = storage.write(dataframe)
        return [self.image_key] + self.text_keys
