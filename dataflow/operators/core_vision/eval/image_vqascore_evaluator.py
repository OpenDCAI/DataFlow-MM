import torch
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from transformers import BlipProcessor, BlipForQuestionAnswering

from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY


@OPERATOR_REGISTRY.register()
class ImageVQAScoreEvaluator(OperatorABC):
    def __init__(
        self,
        model_name: str = "Salesforce/blip-vqa-base",
        device: str = None,
        local_only: bool = True,
    ):
        self.logger = get_logger()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = BlipProcessor.from_pretrained(model_name, local_files_only=local_only)
        self.model = BlipForQuestionAnswering.from_pretrained(model_name, local_files_only=local_only).to(self.device).eval()
        tok = self.processor.tokenizer
        self.yes_ids = tok("yes", add_special_tokens=True, return_tensors="pt").input_ids.to(self.device)
        self.no_ids = tok("no", add_special_tokens=True, return_tensors="pt").input_ids.to(self.device)

    @staticmethod
    def get_desc(lang: str = "zh") -> str:
        if lang == "zh":
            return (
                "VQAScoreEvaluator 算子：基于 BLIP 视觉问答模型计算图像与文本描述是否匹配的“是/否”概率分数。\n"
                "输入：通过 run(storage, input_image_key, output_text_key, output_key) 中的 input_image_key 和 output_text_key 指定，"
                "分别从 DataFlowStorage 中读取包含图像路径的列和对应文本描述的列；\n"
                "输出：在同一 DataFrame 中新增由 output_key 指定的列（默认名为 vqa_score），"
                "每一行存储一个介于 [0, 1] 的概率值，表示在问题“该图像是否匹配该文本描述？”下，模型回答“Yes”的置信度，数值越大表示越倾向匹配；\n"
                "功能：对 DataFrame 中逐行遍历图文对，将文本描述包装成英文问句输入 BLIP VQA 模型，同时分别以“Yes”和“No”作为候选答案，"
                "通过损失值构造相对概率，得到“Yes”的归一化概率，用于图文一致性评估、结果过滤、重排序以及多模态生成质量打分等场景。"
            )
        else:
            return (
                "VQAScoreEvaluator operator: computes a Yes/No probability score measuring whether an image matches a given text using the BLIP VQA model.\n"
                "Inputs: specified via run(storage, input_image_key, output_text_key, output_key); "
                "input_image_key points to the column containing image file paths, and output_text_key points to the column containing text descriptions; \n"
                "Output: a new column in the same DataFrame, named by output_key (default 'vqa_score'), "
                "where each row stores a probability in [0, 1] representing the model's confidence that the answer to "
                "“Does this image match the description?” is Yes, with larger values indicating stronger agreement; \n"
                "Function: iterates over the DataFrame row by row, wraps the text description into an English question, "
                "feeds it together with the image into the BLIP VQA model with “yes” and “no” as candidate labels, "
                "converts their losses into a normalized probability for “yes”, and uses this as an automatic multimodal alignment score "
                "for filtering, reranking, and evaluating image–text pairs."
            )
        
    @torch.no_grad()
    def compute_yes_prob(self, image_path: str, text: str) -> float:
        try:
            image = Image.open(image_path).convert("RGB")
        except (FileNotFoundError, UnidentifiedImageError) as e:
            self.logger.warning(f"Failed to load image: {image_path}. Reason: {e}")
            return 0.0
        if not text or text.strip() == "":
            return 0.0
        question = f"Does this image match the description: {text}? Answer yes or no."
        inputs = self.processor(images=image, text=question, return_tensors="pt").to(self.device)
        out_yes = self.model(**inputs, labels=self.yes_ids)
        out_no = self.model(**inputs, labels=self.no_ids)
        ly = float(out_yes.loss.item())
        ln = float(out_no.loss.item())
        py = torch.exp(torch.tensor(-ly))
        pn = torch.exp(torch.tensor(-ln))
        p = float((py / (py + pn + 1e-8)).item())
        if not (0.0 <= p <= 1.0):
            p = max(min(p, 1.0), 0.0)
        return p
    
    def run(self, storage: DataFlowStorage, input_image_key: str = "image_path", input_text_key: str = "text", output_key: str = "vqa_score"):
        df = storage.read("dataframe")
        scores = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Implementing {self.__class__.__name__}"):
            img_path = row[input_image_key]
            text = str(row[input_text_key])
            score = self.compute_yes_prob(img_path, text)
            scores.append(score)
        df[output_key] = scores
        storage.write(df)
        return [output_key]
