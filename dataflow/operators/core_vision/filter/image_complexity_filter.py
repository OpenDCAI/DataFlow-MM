from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY

@OPERATOR_REGISTRY.register()
class ImageComplexityFilter(OperatorABC):
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

    def __init__(
        self,
        model_name: str = "facebook/bart-large-mnli",
        threshold: float = 0.4,
        min_k: int = 2,
        device: str = None
    ):
        self.logger = get_logger()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True, use_fast=True, weights_only=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            local_files_only=True,
            use_safetensors=True,
            weights_only=False
        ).to(self.device).eval()
        self.thresh = threshold
        self.min_k = min_k

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "ImageComplexityFilter 算子：基于 NLI（自然语言推理）模型评估文本描述所覆盖的视觉能力属性数量，"
                "对能力维度过少的 caption 进行过滤，从而保留语义更丰富、更具多样性的描述。\n"
                "输入：通过 run(storage, input_caption_key) 中的 input_caption_key 指定，从 DataFlowStorage 中读取包含文本描述的列；\n"
                "输出：过滤掉不符合条件的样本后，将剩余数据写回 DataFrame，caption 列名保持不变，并返回 [input_caption_key] 作为后续算子的输入列名；\n"
                "功能：预先定义若干视觉能力标签（如颜色、形状、物体识别、动作识别、文本识别、空间关系、计数、场景理解等），"
                "对每条 caption 与“该文本描述某种能力”的模板句组成 NLI 前提-假设对，"
                "利用序列分类模型判断每个能力标签是否被文本支持，当被支持的能力数量不少于 min_k 且单个标签的置信度超过阈值 threshold 时，"
                "认为该 caption 复杂度合格，否则将其过滤，可用于构建高质量、能力多样的多模态数据集。"
            )
        else:
            return (
                "ComplexityFilter operator: uses an NLI (natural language inference) model to estimate how many visual "
                "capabilities a caption expresses and filters out captions with too few capability dimensions, keeping "
                "semantically richer and more diverse descriptions.\n"
                "Inputs: specified via run(storage, input_caption_key), where input_caption_key refers to the column in "
                "DataFlowStorage that contains text captions;\n"
                "Output: writes back a filtered DataFrame that only keeps rows whose captions pass the complexity check, "
                "preserves the caption column name, and returns [input_caption_key] as the input column list for downstream operators;\n"
                "Function: defines a set of visual capability labels (e.g., color, shape, object recognition, action recognition, "
                "text recognition, spatial relationship, counting, scene understanding), forms NLI premise–hypothesis pairs between "
                "each caption and a template sentence expressing one capability, and uses a sequence classification model to decide "
                "whether the caption entails that capability. If the number of supported capabilities is at least min_k and each "
                "selected capability has probability above threshold, the caption is kept; otherwise it is filtered out, which is "
                "useful for building high-quality multimodal datasets with diverse capabilities."
            )
        
    def detect_caps(self, caption: str) -> list:
        if not caption or len(caption.strip()) < 5:
            return []
        detected = []
        try:
            for cap in self.CAPS:
                hyp = self.TEMPLATE.format(cap)
                inputs = self.tokenizer(caption, hyp, return_tensors="pt", truncation=True, padding=True).to(self.device)
                with torch.no_grad():
                    logits = self.model(**inputs).logits[0]
                    probs = torch.softmax(logits, dim=-1)
                if probs[2].item() >= self.thresh:
                    detected.append(cap)
        except Exception as e:
            self.logger.warning(f"[CompositionalComplexityRefiner] Error: {e}")
        return detected

    def is_valid(self, image_path: str, caption: str) -> bool:
        caps = self.detect_caps(caption)
        return len(caps) >= self.min_k

    def run(self, storage: DataFlowStorage, input_caption_key: str = "caption"):
        dataframe = storage.read("dataframe")
        refined_mask = []
        for i, row in enumerate(tqdm(dataframe.itertuples(), desc=f"Implementing {self.__class__.__name__}")):
            caption = getattr(row, input_caption_key)
            ok = self.is_valid("", caption)
            refined_mask.append(ok)
            if not ok:
                self.logger.debug(f"Filtered weak caption at row {i}: {caption[:40]}")
        dataframe = dataframe[refined_mask].reset_index(drop=True)
        storage.write(dataframe)
        return [input_caption_key]