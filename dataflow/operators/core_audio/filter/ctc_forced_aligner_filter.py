import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger

from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.operators.core_audio import CTCForcedAlignSampleEvaluator

from typing import Union, List, Dict, Any

@OPERATOR_REGISTRY.register()
class CTCForcedAlignFilter(OperatorABC):
    def __init__(self, model_path: str = "MahmoudAshraf/mms-300m-1130-forced-aligner", device: Union[str, List[str]] = "cuda", num_workers: int = 1):
        self.logger = get_logger(__name__)
        self.evaluator = CTCForcedAlignSampleEvaluator(model_path=model_path, num_workers=num_workers,device=device)
    
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
        # if word_timestamps_list:
        #     dataframe['word_timestamps'] = word_timestamps_list
        # dataframe['spans'] = spans_list_list

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