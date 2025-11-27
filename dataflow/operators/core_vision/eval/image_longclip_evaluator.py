import os
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
class ImageLongCLIPEvaluator(OperatorABC):
    def __init__(
        self,
        model_name: str = "BeichenZhang/LongCLIP-L-336px",
        device: str = None,
    ):
        self.logger = get_logger()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        from .model import longclip
        self.longclip = longclip

        if os.path.isdir(model_name):
            files = os.listdir(model_name)
            candidates = [f for f in files if f.endswith((".pt", ".bin", ".ckpt"))]
            if not candidates:
                raise FileNotFoundError(f"No checkpoint file found in directory: {model_name}")
            pref = [f for f in candidates if "longclip" in f.lower()]
            target = sorted(pref or candidates)[0]
            ckpt_path = os.path.join(model_name, target)
            self.logger.info(f"Loading LongCLIP checkpoint from directory: {ckpt_path}")
        else:
            ckpt_path = model_name
            self.logger.info(f"Loading LongCLIP checkpoint from path: {ckpt_path}")

        self.model, self.preprocess = longclip.load(ckpt_path, device=self.device)
        self.tokenizer = longclip.tokenize
        self.model.eval()
        self.context_length = 248

    @staticmethod
    def get_desc(lang: str = "zh") -> str:
        if lang == "zh":
            return (
                "LongCLIPEvaluator 算子：基于 LongCLIP 模型计算长文本与图像之间的语义对齐分数。\n"
                "输入：通过 run(storage, input_image_key, input_text_key, output_key) 中的 input_image_key 和 input_text_key 指定，"
                "分别从 DataFlowStorage 中读取包含图像路径的列和对应长文本内容的列；\n"
                "输出：在同一 DataFrame 中新增由 output_key 指定的列（默认名为 longclip_score），"
                "每一行对应一个图文对的相似度分数，数值范围为 [0, 1]，数值越大表示图像与长文本语义越接近；\n"
                "功能：对 DataFrame 中逐行遍历图文对，利用 LongCLIP 模型对图像和长文本进行编码，"
                "通过归一化余弦相似度计算图文匹配分数，可用于长文本图文检索、重排序以及生成结果的自动化评估。"
            )
        else:
            return (
                "LongCLIPEvaluator operator: computes the semantic alignment score between images and long texts using the LongCLIP model.\n"
                "Inputs: specified via run(storage, input_image_key, input_text_key, output_key); "
                "input_image_key refers to the column containing image file paths, and input_text_key refers to the column containing long-form text; \n"
                "Output: a new column in the same DataFrame, named by output_key (default 'longclip_score'), "
                "where each row stores the similarity score of an image–long-text pair in the range [0, 1], "
                "with larger values indicating stronger semantic alignment; \n"
                "Function: iterates over the DataFrame row by row, encodes images and long texts with the LongCLIP model, "
                "and computes a normalized cosine similarity, which can be used for long-text image–text retrieval, reranking, "
                "and automatic evaluation of multimodal generation results."
            )

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

    def run(
        self,
        storage: DataFlowStorage,
        input_image_key: str = "image_path",
        input_text_key: str = "text",
        output_key: str = "longclip_score",
    ):
        df = storage.read("dataframe")
        scores = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Implementing {self.__class__.__name__}"):
            img_path = row[input_image_key]
            text = str(row[input_text_key])
            score = self.compute_similarity(img_path, text)
            scores.append(score)
        df[output_key] = scores
        storage.write(df)
        return [output_key]
