import re
import sys
import subprocess

import pytesseract
from PIL import Image
from tqdm import tqdm

from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY


# def _load_stanza_pipeline():
#     import torch
#     import numpy as np

#     # 兼容 PyTorch 2.6+ 权重加载安全限制
#     torch.serialization.add_safe_globals([np.core.multiarray._reconstruct, np.ndarray])
#     try:
#         import stanza
#     except ImportError:
#         subprocess.run([sys.executable, "-m", "pip", "install", "stanza==1.8.2"], check=True)
#         import stanza
#     try:
#         return stanza.Pipeline(lang="en", processors="tokenize,pos,lemma,depparse", use_gpu=False, verbose=False, weights_only=False)
#     except Exception:
#         stanza.download("en")
#         return stanza.Pipeline(lang="en", processors="tokenize,pos,lemma,depparse", use_gpu=False, verbose=False, weights_only=False)


# _nlp = _load_stanza_pipeline()


@OPERATOR_REGISTRY.register()
class CatFilter(OperatorABC):
    def __init__(self, min_triples: int = 2, ocr_overlap_threshold: float = 0.2):
        self.logger = get_logger()
        self.min_triples = min_triples
        self.ocr_thresh = ocr_overlap_threshold

    @staticmethod
    def get_desc(lang: str = "zh"):
        return "CaT 复杂度与OCR过滤" if lang == "zh" else "Caption-as-Teacher (complexity + OCR filter)"

    def _triples_and_has_verb(self, doc):
        triples = set()
        has_verb = False
        for sent in doc.sentences:
            words = sent.words
            children = {}
            for w in words:
                children.setdefault(w.head, []).append(w)
                if w.upos == "VERB":
                    has_verb = True
            for w in words:
                if w.deprel in ("nsubj", "nsubj:pass"):
                    head_idx = w.head
                    if head_idx == 0:
                        continue
                    head = words[head_idx - 1]
                    if head.upos != "VERB":
                        continue
                    for ch in children.get(head.id, []):
                        if ch.deprel in ("obj", "iobj", "xcomp", "obl", "attr", "ccomp"):
                            triples.add((w.text, head.lemma or head.text, ch.text))
        return len(triples), has_verb

    def is_complex_caption(self, doc) -> bool:
        triple_cnt, _ = self._triples_and_has_verb(doc)
        return triple_cnt >= self.min_triples

    def has_action_verb(self, doc) -> bool:
        _, has_verb = self._triples_and_has_verb(doc)
        return has_verb

    def is_not_ocr_only(self, image_path: str, caption: str) -> bool:
        img = Image.open(image_path).convert("RGB")
        ocr_text = pytesseract.image_to_string(img, lang="eng+chi_sim")
        ocr_tokens = set(re.findall(r"[A-Za-z']+", ocr_text.lower()))
        cap_tokens = set(re.findall(r"[A-Za-z']+", caption.lower()))
        if not ocr_tokens:
            return True
        jaccard = len(ocr_tokens & cap_tokens) / len(ocr_tokens | cap_tokens)
        return jaccard < self.ocr_thresh

    def is_consistent(self, image_path: str, caption: str) -> bool:
        if not caption or not caption.strip():
            return False
        doc = _nlp(caption)
        return self.is_complex_caption(doc) and self.has_action_verb(doc) and self.is_not_ocr_only(image_path, caption)

    def run(self, storage: DataFlowStorage, image_key: str, caption_key: str):
        dataframe = storage.read("dataframe")
        refined_mask = []
        for i, row in enumerate(tqdm(dataframe.itertuples(), desc=f"Implementing {self.__class__.__name__}")):
            img_path = getattr(row, image_key)
            caption = getattr(row, caption_key)
            ok = False
            try:
                ok = self.is_consistent(img_path, caption)
            except Exception as e:
                self.logger.debug(f"Filter error at row {i}: {e}")
                ok = False
            refined_mask.append(ok)
            if not ok:
                self.logger.debug(f"Filtered at row {i}: {img_path}, {str(caption)[:30]}")
        dataframe = dataframe[refined_mask].reset_index(drop=True)
        storage.write(dataframe)
        return [image_key, caption_key]
