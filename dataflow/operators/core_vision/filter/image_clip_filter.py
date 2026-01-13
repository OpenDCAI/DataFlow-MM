import torch
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY

@OPERATOR_REGISTRY.register()
class ImageClipFilter(OperatorABC):
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str = None
    ):
        self.logger = get_logger()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = CLIPProcessor.from_pretrained(model_name, use_safetensors=True, weights_only=False)
        self.model = CLIPModel.from_pretrained(model_name, use_safetensors=True, weights_only=False).to(self.device).eval()

    @staticmethod
    def get_desc(lang: str = "zh") -> str:
        if lang == "zh":
            return (
                "ImageClipFilter 算子：基于预训练 CLIP 模型计算图像与文本描述的语义相似度，并按照给定阈值过滤不一致的图文对。\n"
                "输入：通过 run(storage, input_image_key, input_caption_key, threshold) 中的 input_image_key 和 input_caption_key 指定，"
                "分别从 DataFlowStorage 中读取图像路径列和对应文本描述列；阈值参数 threshold 控制保留样本的最低相似度要求（默认 0.25）；\n"
                "输出：对 DataFrame 中的每一行图文对计算 CLIP 相似度，仅保留相似度大于等于 threshold 的行，将过滤后的 DataFrame 写回存储，"
                "并返回 [input_image_key, input_caption_key] 作为后续算子的输入列名；\n"
                "功能：使用 CLIPProcessor 对图像与文本进行预处理，由 CLIPModel 提取图像与文本嵌入，归一化后计算余弦相似度并裁剪到 [0, 1] 区间，"
                "依据设定阈值实现图文一致性过滤，可用于多模态训练数据清洗、数据集构建或评测集筛选等场景。"
            )
        else:
            return (
                "ImageClipFilter operator: uses a pretrained CLIP model to compute semantic similarity between images and text "
                "and filters out image–text pairs whose similarity is below a given threshold.\n"
                "Inputs: specified via run(storage, input_image_key, input_caption_key, threshold), where input_image_key points "
                "to the column containing image file paths, input_caption_key points to the column containing text descriptions, "
                "and threshold controls the minimum similarity required for a pair to be kept (default 0.25);\n"
                "Output: for each row in the DataFrame, computes the CLIP similarity of the image–text pair, keeps only rows with "
                "similarity greater than or equal to threshold, writes the filtered DataFrame back to storage, and returns "
                "[input_image_key, input_caption_key] as the column names for downstream operators;\n"
                "Function: uses CLIPProcessor to preprocess images and text, CLIPModel to encode them into embeddings, normalizes "
                "the embeddings, computes cosine similarity and clips it into the [0, 1] range, then applies the threshold to "
                "perform image–text consistency filtering for multimodal data cleaning or evaluation set construction."
            )
        
    def compute_similarity(self, image_path: str, text: str) -> float:
        try:
            image = Image.open(image_path).convert("RGB")
        except (FileNotFoundError, UnidentifiedImageError):
            self.logger.warning(f"Failed to load image: {image_path}")
            return 0.0
        if not text or text.strip() == "":
            self.logger.warning(f"Empty text for image: {image_path}")
            return 0.0
        inputs = self.processor(text=[text], images=[image], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            img_emb = outputs.image_embeds
            txt_emb = outputs.text_embeds
        img_norm = img_emb / img_emb.norm(p=2, dim=-1, keepdim=True)
        txt_norm = txt_emb / txt_emb.norm(p=2, dim=-1, keepdim=True)
        sim = (img_norm @ txt_norm.T).cpu().item()
        if not (0.0 <= sim <= 1.0):
            sim = max(min(sim, 1.0), 0.0)
        return sim

    def is_consistent(self, image_path: str, caption: str, threshold: float = 0.25) -> bool:
        return self.compute_similarity(image_path, caption) >= threshold

    def run(self, storage: DataFlowStorage, input_image_key: str = "image", input_caption_key: str = "caption", threshold: float = 0.25):
        dataframe = storage.read("dataframe")
        refined_mask = []
        for i, row in enumerate(tqdm(dataframe.itertuples(), desc=f"Implementing {self.__class__.__name__}")):
            img_path = getattr(row, input_image_key)
            cap = getattr(row, input_caption_key)
            sim = self.compute_similarity(img_path, cap)
            ok = sim >= threshold
            refined_mask.append(ok)
            if not ok:
                self.logger.debug(f"CLIP failed at row {i}: sim={sim:.3f}, img={img_path}, cap={cap}")
        dataframe = dataframe[refined_mask].reset_index(drop=True)
        storage.write(dataframe)
        return [input_image_key, input_caption_key]
