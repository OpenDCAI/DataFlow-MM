import re

import torch
import pytesseract
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY


@OPERATOR_REGISTRY.register()
class ImageCatFilter(OperatorABC):
    CAPS_HYPOTHESES = [
        "The caption describes what people or objects are doing.",
        "The caption describes interactions between multiple people or objects.",
        "The caption provides rich details about the scene.",
        "The caption mentions spatial relationships or positions of objects in the scene.",
        "The caption describes multiple aspects of the image rather than a single short fact."
    ]

    ACTION_HYPOTHESIS = "The caption clearly describes an action happening in the scene."
    OCR_ONLY_HYPOTHESIS = (
        "The caption mainly transcribes the visible text in the image instead of describing the visual scene."
    )

    def __init__(
        self,
        model_name: str = "facebook/bart-large-mnli",
        complexity_thresh: float = 0.4,
        min_caps: int = 2,
        action_thresh: float = 0.4,
        ocr_overlap_threshold: float = 0.2,
        ocr_nli_thresh: float = 0.6,
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

        self.complexity_thresh = complexity_thresh
        self.min_caps = min_caps
        self.action_thresh = action_thresh
        self.ocr_thresh = ocr_overlap_threshold
        self.ocr_nli_thresh = ocr_nli_thresh

        self.ocr_available = True
        try:
            _ = pytesseract.get_tesseract_version()
        except (AttributeError, pytesseract.TesseractNotFoundError, OSError) as e:
            self.ocr_available = False
            self.logger.warning(
                f"Tesseract not available, OCR-based filtering will be skipped. Reason: {e}"
            )

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "ImageCatFilter 算子：基于 Caption-as-Teacher 思想，结合 BART-large-MNLI 的自然语言推理能力与可选的 OCR 文本重叠度，"
                "对图像-文本对进行“语义复杂度 + 动作性 + OCR 抄写”三重过滤。\n"
                "输入：通过 run(storage, input_image_key, input_caption_key) 中的 input_image_key 和 input_caption_key 指定，"
                "分别从 DataFlowStorage 中读取图像路径列与对应的英文描述列；\n"
                "输出：对每一行图文对进行判定，仅保留满足以下条件的样本：\n"
                "  1）caption 对多条“能力假设句”（如：描述动作、实体交互、场景细节等）的 NLI 蕴含概率不低于 complexity_thresh，"
                "且被蕴含的能力条数不少于 min_caps；\n"
                "  2）caption 对“该文本清晰描述了正在发生的动作”这一假设的蕴含概率不低于 action_thresh；\n"
                "  3）若本机安装了 Tesseract，则进一步计算图像 OCR 文本与 caption token 的 Jaccard 重叠度，"
                "并结合 NLI 对“caption 主要是抄写图像文字”的假设打分；当重叠度过高且该假设的蕴含概率大于 ocr_nli_thresh 时过滤，"
                "否则保留；若未安装 Tesseract，则自动跳过 OCR 相关过滤，仅使用前两条 NLI 规则；\n"
                "过滤后的 DataFrame 写回存储，并返回 [input_image_key, input_caption_key] 作为后续算子的输入列名。"
            )
        else:
            return (
                "ImageCatFilter operator: implements a Caption-as-Teacher style filter using BART-large-MNLI and optional OCR "
                "overlap to enforce caption complexity, action description, and non-OCR-only constraints.\n"
                "Inputs: via run(storage, input_image_key, input_caption_key), where input_image_key is the image path column "
                "and input_caption_key is the English caption column.\n"
                "Output: keeps only samples that\n"
                "  1) entail at least min_caps capability hypotheses (actions, interactions, scene details, etc.) with probability "
                "≥ complexity_thresh;\n"
                "  2) entail the action hypothesis with probability ≥ action_thresh;\n"
                "  3) if Tesseract OCR is available, also pass an OCR-based check: high token overlap combined with high entailment "
                "for the \"caption mainly transcribes on-image text\" hypothesis leads to filtering. If Tesseract is not installed, "
                "the OCR-based step is automatically skipped and only NLI rules are applied.\n"
                "The filtered DataFrame is written back to storage and [input_image_key, input_caption_key] is returned for "
                "downstream operators."
            )

    def _entail_prob(self, premise: str, hypothesis: str) -> float:
        premise = (premise or "").strip()
        hypothesis = (hypothesis or "").strip()
        if not premise or not hypothesis:
            return 0.0
        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            padding=True,
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits[0]
            probs = torch.softmax(logits, dim=-1)
        return float(probs[2].item())

    def _count_capabilities(self, caption: str) -> int:
        if not caption or len(caption.strip()) < 5:
            return 0
        cnt = 0
        for hyp in self.CAPS_HYPOTHESES:
            p = self._entail_prob(caption, hyp)
            if p >= self.complexity_thresh:
                cnt += 1
        return cnt

    def is_complex_caption(self, caption: str) -> bool:
        cap_num = self._count_capabilities(caption)
        return cap_num >= self.min_caps

    def has_action_verb(self, caption: str) -> bool:
        p = self._entail_prob(caption, self.ACTION_HYPOTHESIS)
        return p >= self.action_thresh

    def is_not_ocr_only(self, image_path: str, caption: str) -> bool:
        if not self.ocr_available or self.ocr_thresh <= 0:
            return True
        try:
            img = Image.open(image_path).convert("RGB")
        except (FileNotFoundError, UnidentifiedImageError) as e:
            self.logger.warning(f"Failed to load image for OCR: {image_path}. Reason: {e}")
            return True
        try:
            ocr_text = pytesseract.image_to_string(img, lang="eng+chi_sim")
        except Exception as e:
            self.logger.debug(f"OCR failed on image: {image_path}. Reason: {e}")
            return True
        ocr_tokens = set(re.findall(r"[A-Za-z']+", (ocr_text or "").lower()))
        cap_tokens = set(re.findall(r"[A-Za-z']+", (caption or "").lower()))
        if not ocr_tokens:
            return True
        jaccard = len(ocr_tokens & cap_tokens) / len(ocr_tokens | cap_tokens)
        if jaccard >= self.ocr_thresh:
            p_ocr_only = self._entail_prob(caption, self.OCR_ONLY_HYPOTHESIS)
            if p_ocr_only >= self.ocr_nli_thresh:
                return False
        return True

    def is_consistent(self, image_path: str, caption: str) -> bool:
        if not caption or not caption.strip():
            return False
        return (
            self.is_complex_caption(caption)
            and self.has_action_verb(caption)
            and self.is_not_ocr_only(image_path, caption)
        )

    def run(
        self,
        storage: DataFlowStorage,
        input_image_key: str = "image",
        input_caption_key: str = "caption",
    ):
        dataframe = storage.read("dataframe")
        refined_mask = []
        for i, row in enumerate(
            tqdm(
                dataframe.itertuples(),
                desc=f"Implementing {self.__class__.__name__}",
            )
        ):
            img_path = getattr(row, input_image_key)
            caption = getattr(row, input_caption_key)
            ok = False
            try:
                ok = self.is_consistent(img_path, caption)
            except Exception as e:
                self.logger.debug(f"Filter error at row {i}: {e}")
                ok = False
            refined_mask.append(ok)
            if not ok:
                self.logger.debug(
                    f"Filtered at row {i}: {img_path}, {str(caption)[:60]}"
                )
        dataframe = dataframe[refined_mask].reset_index(drop=True)
        storage.write(dataframe)
        return [input_image_key, input_caption_key]
