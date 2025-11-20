import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger

from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.operators.core_audio import CTCForcedAlignmentSampleEvaluator

from typing import Union, List, Dict, Any

@OPERATOR_REGISTRY.register()
class CTCForcedAlignmentFilter(OperatorABC):
    def __init__(self, model_path: str = "MahmoudAshraf/mms-300m-1130-forced-aligner", device: Union[str, List[str]] = "cuda", num_workers: int = 1):
        self.logger = get_logger(__name__)
        self.evaluator = CTCForcedAlignmentSampleEvaluator(model_path=model_path, num_workers=num_workers,device=device)

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            desc = (
                "该算子基于 CTC 强制对齐结果，对语音-文本样本进行质量过滤。\n"
                "内部调用 CTCForcedAlignmentSampleEvaluator 进行强制对齐，然后根据 spans 的得分\n"
                "计算每条样本的整体对齐分数，并按阈值筛选保留样本，写回过滤后的 dataframe。\n\n"

                "输入参数（run）：\n"
                "- input_audio_key: 音频路径所在列名，默认 'audio'\n"
                "- input_conversation_key: 文本/对话所在列名，默认 'conversation'\n"
                "- sampling_rate: 采样率（Hz），用于音频加载/重采样，默认 16000\n"
                "- language: 文本语言（ISO 639-1/639-3），用于 romanize 和正则化，例如 'en'、'zh'、'ara'\n"
                "- micro_batch_size: 每次前向计算时并行处理的音频块数量\n"
                "- chinese_to_pinyin: 是否先将中文转为拼音再对齐\n"
                "- retain_word_level_alignment: 是否在对齐阶段计算词级时间戳（对当前过滤逻辑不是必需，默认 True）\n"
                "- threshold: 过滤阈值，样本得分低于该值会被丢弃，默认 0.8\n"
                "- threshold_mode: 计算样本得分的方式，仅支持:\n"
                "    * 'min': 使用当前样本所有 spans 中的最小 score\n"
                "    * 'mean': 使用 spans 的平均 score\n"
                "- romanize: 是否对文本进行罗马化（使用 uroman），通常对非拉丁文字有帮助\n\n"

                "运行行为：\n"
                "1. 从 storage 中读取 'dataframe'\n"
                "2. 调用 CTCForcedAlignmentSampleEvaluator.eval 获取每条样本的对齐结果 records\n"
                "3. 对于每行：\n"
                "   - 若 records['error'] 非空，则跳过该样本\n"
                "   - 从 records['spans'] 中取出所有 span_dict['score']，按 threshold_mode 计算整体得分 val\n"
                "   - 若 val >= threshold，则保留该样本\n"
                "4. 若最终有样本保留，则将过滤后的 dataframe 写回 storage（覆盖原 'dataframe'）\n"
                "5. 若全部被过滤，则仅打印日志“All data has been filtered out!” 而不写回\n\n"

                "输出：\n"
                "- 覆盖写回到 storage 的 'dataframe'，仅包含通过对齐质量筛选的样本行。\n"
            )
        else:
            desc = (
                "This operator filters audio-text samples based on CTC forced alignment quality.\n"
                "It internally calls CTCForcedAlignmentSampleEvaluator to obtain alignment results, then\n"
                "computes an overall score per sample from spans and keeps only those above a given threshold.\n"
                "The filtered dataframe is written back to storage.\n\n"

                "Input parameters (run):\n"
                "- input_audio_key: Column name containing audio paths, default 'audio'\n"
                "- input_conversation_key: Column name containing transcript/text, default 'conversation'\n"
                "- sampling_rate: Sampling rate in Hz for audio loading/resampling, default 16000\n"
                "- language: Text language (ISO 639-1/639-3), used for romanization and normalization,\n"
                "            e.g., 'en', 'zh', 'ara'\n"
                "- micro_batch_size: Number of audio chunks processed in parallel per forward pass\n"
                "- chinese_to_pinyin: Whether to convert Chinese text into pinyin before alignment\n"
                "- retain_word_level_alignment: Whether to compute word-level timestamps during alignment\n"
                "                               (not strictly required for the filtering logic, default True)\n"
                "- threshold: Filtering threshold; samples with scores below this value are discarded, default 0.8\n"
                "- threshold_mode: How to aggregate span scores into a single sample score, one of:\n"
                "    * 'min' : use the minimum score among all spans for that sample\n"
                "    * 'mean': use the average score of all spans\n"
                "- romanize: Whether to romanize text using uroman (useful for non-Latin scripts)\n\n"

                "Runtime behavior:\n"
                "1. Read the input dataframe from storage key 'dataframe'.\n"
                "2. Call CTCForcedAlignmentSampleEvaluator.eval to obtain per-row alignment records.\n"
                "3. For each row:\n"
                "   - If records['error'] is not None, skip this sample.\n"
                "   - Extract all span_dict['score'] from records['spans'] and compute a single value `val`\n"
                "     according to `threshold_mode` ('min' or 'mean').\n"
                "   - Keep the sample only if val >= threshold.\n"
                "4. If there are remaining samples, construct a new dataframe containing only them and\n"
                "   write it back to storage (overwriting 'dataframe').\n"
                "5. If all samples are filtered out, log “All data has been filtered out!” and do not write back.\n\n"

                "Output:\n"
                "- The 'dataframe' in storage is overwritten with a filtered dataframe that only contains\n"
                "  samples whose alignment scores satisfy the configured threshold.\n"
            )
        return desc

    def run(self, 
            storage: DataFlowStorage,
            input_audio_key: str = "audio",
            input_conversation_key: str = "conversation",
            sampling_rate: int = 16000,
            language: str = "en",
            micro_batch_size: int = 16,
            chinese_to_pinyin: bool = False,
            retain_word_level_alignment: bool = True,
            threshold: float = 0.8,
            threshold_mode: str = "min",
            romanize: bool = True,
            ):
        assert threshold_mode in ['mean', 'min'], f"threshold_mode must be 'mean' or 'min', got '{threshold_mode}'"

        dataframe = storage.read('dataframe')
        records = self.evaluator.eval(
            dataframe=dataframe,
            input_audio_key=input_audio_key,
            input_conversation_key=input_conversation_key,
            sampling_rate=sampling_rate,
            language=language,
            micro_batch_size=micro_batch_size,
            chinese_to_pinyin=chinese_to_pinyin,
            retain_word_level_alignment=retain_word_level_alignment,       # 帧级强制对齐
            romanize=romanize,
        )

        dataframe = dataframe.copy()
        dataframe.loc[:, 'records'] = records
        output_dataframe = []

        for idx, row in dataframe.iterrows():
            if row['records']['error'] is not None:
                continue
            
            spans_list = row['records']['spans']

            if threshold_mode == 'min':
                val = min(span_dict['score'] for span_dict in spans_list)
            else:
                val = sum(span_dict['score'] for span_dict in spans_list) / len(spans_list)
                    
            if val >= threshold:
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