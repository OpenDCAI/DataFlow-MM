import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

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
class ImageCLIPEvaluator(OperatorABC):
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str = None
    ):
        self.logger = get_logger()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = CLIPProcessor.from_pretrained(model_name, use_safetensors=True)
        self.model = CLIPModel.from_pretrained(model_name, use_safetensors=True).to(self.device).eval()

    @staticmethod
    def get_desc(lang: str = "zh") -> str:
        if lang == "zh":
            return (
                "CLIPEvaluator 算子：基于预训练 CLIP 模型计算图像与文本之间的语义对齐分数。\n"
                "输入：通过 run(storage, input_image_key, output_text_key, output_key) 中的 input_image_key 和 output_text_key 指定，"
                "分别从 DataFlowStorage 中读取包含图像路径的列和对应文本内容的列；\n"
                "输出：在同一 DataFrame 中新增由 output_key 指定的列（默认名为 clip_score），"
                "每一行对应一个图文对的相似度分数，数值范围为 [0, 1]，数值越大表示图像与文本语义越接近；\n"
                "功能：对 DataFrame 中逐行遍历图文对，调用 CLIP 模型编码图像与文本，计算归一化余弦相似度，"
                "用于图文匹配、召回重排序、生成结果评估等场景的自动化评分。"
            )
        else:
            return (
                "CLIPEvaluator operator: computes the semantic alignment score between images and texts using a pretrained CLIP model.\n"
                "Inputs: specified via run(storage, input_image_key, output_text_key, output_key); "
                "input_image_key points to the column containing image file paths, and output_text_key points to the column containing text; \n"
                "Output: a new column in the same DataFrame, named by output_key (default 'clip_score'), "
                "where each row stores the similarity score of an image-text pair in the range [0, 1], "
                "with larger values indicating stronger semantic alignment; \n"
                "Function: iterates over the DataFrame row by row, encodes images and texts with the CLIP model, "
                "and computes a normalized cosine similarity, which can be used for image-text matching, "
                "reranking, or automatic evaluation of generation results."
            )
        
    @torch.no_grad()
    def compute_similarity(self, image_path: str, text: str) -> float:
        try:
            image = Image.open(image_path).convert("RGB")
        except (FileNotFoundError, UnidentifiedImageError) as e:
            self.logger.warning(f"Failed to load image: {image_path}. Reason: {e}")
            return 0.0
        if not text or text.strip() == "":
            return 0.0
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=77
        ).to(self.device)
        outputs = self.model(**inputs)
        sim = _cos_sim_unit(outputs.image_embeds[0], outputs.text_embeds[0])
        if not (0.0 <= sim <= 1.0):
            sim = max(min(sim, 1.0), 0.0)
        return float(sim)

    def run(self, storage: DataFlowStorage, input_image_key: str = "image_path", input_text_key: str = "text", output_key: str = "clip_score"):
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
