import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger

from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.operators.core_audio import CTCForcedAlignmentSampleEvaluator

from typing import Union, List, Dict, Any

@OPERATOR_REGISTRY.register()
class CTCForcedAlignmentFilter(OperatorABC):
    def __init__(
        self, 
        model_path: str = "MahmoudAshraf/mms-300m-1130-forced-aligner", 
        device: Union[str, List[str]] = "cuda", 
        num_workers: int = 1,
        sampling_rate: int = 16000,
        language: str = "en",
        micro_batch_size: int = 16,
        chinese_to_pinyin: bool = False,
        threshold: float = 0.8,
        threshold_mode: str = "min",
        romanize: bool = True,
    ):
        self.logger = get_logger(__name__)
        self.evaluator = CTCForcedAlignmentSampleEvaluator(
            model_path=model_path, 
            device=device,
            num_workers=num_workers,
            sampling_rate=sampling_rate,
            language=language,
            micro_batch_size=micro_batch_size,
            chinese_to_pinyin=chinese_to_pinyin,
            romanize=romanize,
        )
        self.sampling_rate = sampling_rate
        self.language = language
        self.micro_batch_size = micro_batch_size
        self.chinese_to_pinyin = chinese_to_pinyin
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.romanize = romanize
        

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "CTCForcedAlignmentFilter（CTC 强制对齐过滤算子）\n"
                "----------------------------------------------\n"
                "功能简介：\n"
                "该算子内部调用 CTCForcedAlignmentSampleEvaluator 对音频与转录文本进行 CTC 强制对齐，\n"
                "并根据对齐结果中的逐词/片段得分（alignment[*]['score']）计算样本整体质量分数，\n"
                "再按阈值过滤，保留对齐质量较高的样本。\n\n"

                "一、__init__ 初始化接口\n"
                "def __init__(\n"
                "    self,\n"
                "    model_path: str = \"MahmoudAshraf/mms-300m-1130-forced-aligner\",\n"
                "    device: Union[str, List[str]] = \"cuda\",\n"
                "    num_workers: int = 1,\n"
                "    sampling_rate: int = 16000,\n"
                "    language: str = \"en\",\n"
                "    micro_batch_size: int = 16,\n"
                "    chinese_to_pinyin: bool = False,\n"
                "    threshold: float = 0.8,\n"
                "    threshold_mode: str = \"min\",\n"
                "    romanize: bool = True,\n"
                ")\n\n"
                "参数说明：\n"
                "- model_path：强制对齐模型路径（Hugging Face repo id 或本地路径），透传给内部 evaluator。\n"
                "- device：推理设备配置。\n"
                "  * 串行：\"cpu\" / \"cuda\" / \"cuda:0\" 等；\n"
                "  * 并行：设备列表 [\"cuda:0\", \"cuda:1\"] 等，由内部 evaluator 在多进程中分配使用。\n"
                "- num_workers：worker 进程数；>1 时内部 evaluator 启用多进程对齐。\n"
                "- sampling_rate：音频读取/重采样采样率（Hz），透传给内部 evaluator。\n"
                "- language：文本语言代码（如 \"en\"、\"zh\"），用于文本归一化/罗马化等处理。\n"
                "- micro_batch_size：生成 emissions 的微批大小，透传给内部 evaluator。\n"
                "- chinese_to_pinyin：若为 True，对齐前将中文文本转为拼音（空格分隔），透传给内部 evaluator。\n"
                "- threshold：过滤阈值。计算得到的样本整体分数 val >= threshold 才会被保留。\n"
                "- threshold_mode：样本整体分数聚合方式：\n"
                "  * \"min\"：val = min(alignment[*]['score'])\n"
                "  * \"mean\"：val = mean(alignment[*]['score'])\n"
                "- romanize：若为 True，内部 evaluator 会对文本做 uroman 罗马化（对非拉丁文字更友好）。\n\n"
                "初始化行为：\n"
                "- 创建 logger；\n"
                "- 实例化一个 CTCForcedAlignmentSampleEvaluator（evaluator），用于后续对齐计算；\n"
                "- 保存阈值与聚合方式等过滤配置（threshold/threshold_mode）。\n\n"

                "二、run 运行接口\n"
                "def run(\n"
                "    self,\n"
                "    storage: DataFlowStorage,\n"
                "    input_audio_key: str = \"audio\",\n"
                "    input_conversation_key: str = \"conversation\",\n"
                "    output_answer_key: str = \"forced_alignment_results\",\n"
                ")\n\n"
                "输入说明：\n"
                "- storage：DataFlowStorage，要求能读出上游写入的 DataFrame。\n"
                "- input_audio_key：音频路径列名；每行可为字符串或单元素列表（[path]）。\n"
                "- input_conversation_key：文本/转录列名；文本解析规则由内部 evaluator 处理。\n"
                "- output_answer_key：对齐结果写入列名。\n\n"
                "输出说明：\n"
                "run 会先把对齐结果写入 output_answer_key 列，然后按阈值过滤样本，并把过滤后的 DataFrame 写回 storage。\n"
                "对齐结果（每行）为 dict：\n"
                "{\n"
                "  'alignment': List[Dict],  # 每个元素包含 start/end/text/score\n"
                "  'error': Optional[str],  # None 表示成功，否则为错误信息\n"
                "}\n\n"
                "过滤规则：\n"
                "- 若 record['error'] 非 None：该样本直接丢弃；\n"
                "- 否则取 record['alignment'] 中所有 score：\n"
                "  * threshold_mode == 'min'  -> val = 最小 score\n"
                "  * threshold_mode == 'mean' -> val = 平均 score\n"
                "- 当 val >= threshold 时保留该样本。\n"
            )
        else:
            return (
                "CTCForcedAlignmentFilter (CTC Forced-Alignment Quality Filter)\n"
                "------------------------------------------------------------\n"
                "Overview:\n"
                "This operator wraps CTCForcedAlignmentSampleEvaluator to run CTC-based forced alignment on\n"
                "audio–text pairs, then filters samples by an aggregated quality score derived from\n"
                "per-word/segment alignment scores (alignment[*]['score']). Only samples whose aggregated\n"
                "score meets or exceeds a configurable threshold are kept.\n\n"

                "1) __init__ interface\n"
                "def __init__(\n"
                "    self,\n"
                "    model_path: str = \"MahmoudAshraf/mms-300m-1130-forced-aligner\",\n"
                "    device: Union[str, List[str]] = \"cuda\",\n"
                "    num_workers: int = 1,\n"
                "    sampling_rate: int = 16000,\n"
                "    language: str = \"en\",\n"
                "    micro_batch_size: int = 16,\n"
                "    chinese_to_pinyin: bool = False,\n"
                "    threshold: float = 0.8,\n"
                "    threshold_mode: str = \"min\",\n"
                "    romanize: bool = True,\n"
                ")\n\n"
                "Parameters:\n"
                "- model_path: forced alignment model identifier/path (HF repo id or local path), passed to the evaluator.\n"
                "- device: inference device configuration.\n"
                "  * Serial: a single string such as \"cpu\", \"cuda\", \"cuda:0\";\n"
                "  * Parallel: a list such as [\"cuda:0\", \"cuda:1\"], used by the evaluator in multiprocessing mode.\n"
                "- num_workers: number of worker processes; >1 enables multiprocessing alignment in the evaluator.\n"
                "- sampling_rate: audio loading/resampling rate in Hz, passed to the evaluator.\n"
                "- language: language code (e.g., \"en\", \"zh\"), used for normalization/romanization.\n"
                "- micro_batch_size: micro-batch size for emission generation, passed to the evaluator.\n"
                "- chinese_to_pinyin: if True, Chinese text is converted to space-separated pinyin before alignment.\n"
                "- threshold: filtering threshold. A sample is kept only if aggregated score val >= threshold.\n"
                "- threshold_mode: aggregation strategy for per-item scores:\n"
                "  * \"min\" : val = min(alignment[*]['score'])\n"
                "  * \"mean\": val = mean(alignment[*]['score'])\n"
                "- romanize: if True, the evaluator romanizes text via uroman (useful for non-Latin scripts).\n\n"
                "Initialization behavior:\n"
                "- Creates a logger;\n"
                "- Instantiates an internal CTCForcedAlignmentSampleEvaluator (self.evaluator) with the given alignment settings;\n"
                "- Stores filtering configs (threshold / threshold_mode).\n\n"

                "2) run interface\n"
                "def run(\n"
                "    self,\n"
                "    storage: DataFlowStorage,\n"
                "    input_audio_key: str = \"audio\",\n"
                "    input_conversation_key: str = \"conversation\",\n"
                "    output_answer_key: str = \"forced_alignment_results\",\n"
                ")\n\n"
                "Inputs:\n"
                "- storage: DataFlowStorage that provides an upstream DataFrame.\n"
                "- input_audio_key: column name containing audio paths (string or single-element list).\n"
                "- input_conversation_key: column name containing transcripts; parsing is handled by the evaluator.\n"
                "- output_answer_key: column name to store alignment records.\n\n"
                "Outputs:\n"
                "The operator first writes alignment records into output_answer_key, then filters rows and writes the\n"
                "filtered DataFrame back to storage.\n"
                "Each per-row alignment record is a dict:\n"
                "{\n"
                "  'alignment': List[Dict],  # each item has start/end/text/score\n"
                "  'error': Optional[str],  # None on success, otherwise an error message\n"
                "}\n\n"
                "Filtering rule:\n"
                "- If record['error'] is not None: the sample is discarded.\n"
                "- Otherwise aggregate scores from record['alignment']:\n"
                "  * threshold_mode == 'min'  -> val = minimum score\n"
                "  * threshold_mode == 'mean' -> val = average score\n"
                "- Keep the sample if val >= threshold.\n"
            )

    def run(self, 
            storage: DataFlowStorage,
            input_audio_key: str = "audio",
            input_conversation_key: str = "conversation",
            output_answer_key: str = "forced_alignment_results",
        ):
        assert self.threshold_mode in ['mean', 'min'], f"threshold_mode must be 'mean' or 'min', got '{self.threshold_mode}'"

        dataframe = storage.read()
        records = self.evaluator.eval(
            dataframe=dataframe,
            input_audio_key=input_audio_key,
            input_conversation_key=input_conversation_key,
        )

        dataframe = dataframe.copy()
        dataframe.loc[:, output_answer_key] = records
        output_dataframe = []

        for idx, row in dataframe.iterrows():
            if row[output_answer_key]['error'] is not None:
                continue
            
            alignment_list = row[output_answer_key]['alignment']

            if self.threshold_mode == 'min':
                val = min(alignment_dict['score'] for alignment_dict in alignment_list)
            else:
                val = sum(alignment_dict['score'] for alignment_dict in alignment_list) / len(alignment_list)
                    
            if val >= self.threshold:
                output_dataframe.append(row.to_dict())

        if output_dataframe:
            output_dataframe = pd.DataFrame(output_dataframe)
            storage.write(output_dataframe)
        else:
            self.logger.info(f"All data has been filtered out!")

    def close(self):
        if self.evaluator.is_parallel:
            self.evaluator.pool.close()
            self.evaluator.pool.join()