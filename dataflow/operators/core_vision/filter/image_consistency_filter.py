from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY

@OPERATOR_REGISTRY.register()
class ImageConsistencyFilter(OperatorABC):
    def __init__(
        self,
        model_name: str = "facebook/bart-large-mnli",
        threshold: float = 0.35,
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
        self.threshold = threshold

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "ImageConsistencyFilter 算子：利用 NLI（自然语言推理）模型，对 caption、question、answer 三个文本字段之间的语义连贯性进行判定，"
                "过滤不满足“答案由图像描述与问题共同蕴含”的样本。\n"
                "输入：通过 run(storage, input_caption_key, input_question_key, input_answer_key) 中的 input_caption_key、input_question_key、"
                "input_answer_key 指定，从 DataFlowStorage 中读取对应的文本列，分别表示图像描述（caption）、问题（question）和回答（answer）；\n"
                "输出：对每一行样本计算由 caption+question（作为前提）到 answer（作为假设）的蕴含概率，仅保留蕴含概率不低于阈值 threshold "
                f"（在 __init__ 中配置，默认 {0.35:.2f}）的行，将过滤后的 DataFrame 写回存储，并返回 "
                "[input_caption_key, input_question_key, input_answer_key] 作为后续算子的输入列名；\n"
                "功能：通过将 caption 与 question 拼接为前提，将 answer 作为假设输入 NLI 模型，读取“entailment”类别的概率作为连贯性得分，"
                "丢弃答案与图像描述或问题语义不一致的样本，可用于构建高质量的图文问答数据集或对现有标注进行自动清洗。"
            )
        else:
            return (
                "ImageConsistencyFilter operator: uses an NLI (natural language inference) model to assess semantic consistency "
                "among caption, question, and answer, and filters out samples where the answer is not entailed by the caption–question pair.\n"
                "Inputs: specified via run(storage, input_caption_key, input_question_key, input_answer_key), where "
                "input_caption_key, input_question_key, and input_answer_key refer to the columns in DataFlowStorage that store "
                "the caption, question, and answer texts respectively;\n"
                "Output: for each row, computes the entailment probability from the premise (caption + question) to the hypothesis "
                "(answer), keeps only rows whose entailment probability is greater than or equal to the configured threshold "
                "in __init__ (default 0.35), writes the filtered DataFrame back to storage, and returns "
                "[input_caption_key, input_question_key, input_answer_key] as the input column names for downstream operators;\n"
                "Function: concatenates caption and question as the NLI premise, takes answer as the hypothesis, feeds them into "
                "the sequence classification model, reads the entailment-class probability as a coherence score, and discards "
                "triples where the answer is not sufficiently supported by the caption–question context, which is useful for "
                "cleaning and constructing high-quality image-based QA datasets."
            )
        
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

    def run(
        self,
        storage: DataFlowStorage,
        input_caption_key: str = "caption",
        input_question_key: str = "question",
        input_answer_key: str = "answer",
    ):
        dataframe = storage.read("dataframe")
        refined_mask = []
        for i, row in enumerate(tqdm(dataframe.itertuples(), desc=f"Implementing {self.__class__.__name__}")):
            caption = getattr(row, input_caption_key)
            question = getattr(row, input_question_key)
            answer = getattr(row, input_answer_key)
            ok, entail_score = self.is_consistent(caption, question, answer)
            refined_mask.append(ok)
            if not ok:
                self.logger.debug(
                    f"Filtered at row {i}: entail_score={entail_score:.3f}, "
                    f"c={str(caption)[:30]}, q={str(question)[:30]}, a={str(answer)[:30]}"
                )
        dataframe = dataframe[refined_mask].reset_index(drop=True)
        storage.write(dataframe)
        return [input_caption_key, input_question_key, input_answer_key]






