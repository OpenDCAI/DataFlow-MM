import re
import math
import torch
import librosa
import torchaudio
import numpy as np
import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger

from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC

from dataclasses import dataclass

from iso639 import Lang
from ctc_forced_aligner import (
    load_alignment_model,
    generate_emissions,
    preprocess_text,
    get_alignments,
    get_spans,
    postprocess_results,
    merge_repeats
)

from tqdm import tqdm
import multiprocessing
from typing import Union, List, Dict, Any

import uroman

# @OPERATOR_REGISTRY.register()
# class CTCForcedAlignSampleEvaluator(OperatorABC):
#     '''
#     CTCForcedAligner is a class that performs CTC forced aligner on audio data.
#     '''
#     def __init__(self, model_path: str = "MahmoudAshraf/mms-300m-1130-forced-aligner", device: str = "cuda"):
#         self.logger = get_logger()
#         self.load_alignment_model(model_path=model_path, device=device)
    
#     @staticmethod
#     def get_desc(lang: str = "zh"):
#         if lang == "zh":
#             return "使用CTC强制对齐计算语音和文本转录对齐分数"
#         else:
#             return "Using CTCForcedAligner to compute the audio-text transcription alignment score"
    
#     def load_alignment_model(
#         self,
#         model_path: str = "MahmoudAshraf/mms-300m-1130-forced-aligner",
#         device: str = "cuda",
#     ):
#         self.device = device
#         self.model, self.tokenizer = load_alignment_model(model_path=model_path, device=device)

#     def eval(self, 
#             dataframe: pd.DataFrame,
#             input_audio_key: str = "audio",
#             input_conversation_key: str = "conversation",
#             sampling_rate: int = 16000,
#             language: str = "en",
#             micro_batch_size: int = 16,
#             chinese_to_pinyin: bool = False,
#             retain_word_level_alignment: bool = False,
#             ):
#         if chinese_to_pinyin:
#             from pypinyin import lazy_pinyin
#             def convert_chinese_to_pinyin(text: str) -> str:
#                 pinyin = lazy_pinyin(text, iso_code="cmn")
#                 return " ".join(pinyin)
            
#         self.logger.info(f"Loading, number of rows: {len(dataframe)}")

#         audio_paths = dataframe.get(input_audio_key, pd.Series([])).tolist()
#         conversations = dataframe.get(input_conversation_key, pd.Series([])).tolist()

#         # audio_list = []
#         texts_normalized = []

#         # for audio_path in audio_paths:
#         #     if isinstance(audio_path, list):
#         #         audio_path = audio_path[0]
#         #     audio, sr = librosa.load(audio_path, sr=sampling_rate)
#         #     audio_list.append(torch.from_numpy(audio).to(self.device))

#         for conversation in conversations:
#             if isinstance(conversation, list) and isinstance(conversation[0], dict) and 'value' in conversation[0]:
#                 text = conversation[0]['value']
#             else:
#                 text = conversation
            
#             if chinese_to_pinyin:
#                 text = convert_chinese_to_pinyin(text)
#             texts_normalized.append(text)
        
#         # word_timestamps_list = []
#         # spans_list_list = []
#         records = []

#         for audio_path, text in tqdm(zip(audio_paths, texts_normalized), total=len(audio_paths), unit=" row", desc="CTC Forced Aligner Processing"):
#             if isinstance(audio_path, list):
#                 audio_path = audio_path[0]

#             try:
#                 audio, sr = librosa.load(audio_path, sr=sampling_rate)
#                 audio = torch.from_numpy(audio).to(self.device)

#                 spans_list = []

#                 emissions, stride = generate_emissions(
#                             self.model, audio, batch_size=micro_batch_size
#                         )
#                 tokens_starred, text_starred = preprocess_text(
#                             text,
#                             romanize=True,
#                             language=Lang(language).pt3,
#                         )

#                 segments, scores, blank_token = self.get_alignments(
#                             emissions,
#                             tokens_starred,
#                             self.tokenizer,
#                         )

#                 spans = get_spans(tokens_starred, segments, blank_token)

#                 j = 0
#                 for seg_list in spans:
#                     for seg in seg_list:
#                         spans_list.append({
#                                     'label': seg.label,
#                                     'start': seg.start,
#                                     'end': seg.end,
#                                     'score': math.exp(scores[seg.start: seg.end + 1][0]),
#                                 })
#                         j += 1

#                 if retain_word_level_alignment:
#                     word_timestamps = postprocess_results(text_starred, spans, stride, scores)
#                     for i in range(len(word_timestamps)):
#                         score = word_timestamps[i]['score']
#                         word_timestamps[i]['score'] = math.exp(score)
#                 else:
#                     word_timestamps = []
    
#                 records.append({
#                             'spans': spans_list,
#                             'word_timestamps': word_timestamps,
#                             'error': None,
#                         })
#             except Exception as e:
#                 records.append({
#                     'spans': [],
#                     'word_timestamps': [],
#                     'error': str(e),
#                 })
#                 self.logger.info(f"error: {str(e)}")

#         # return word_timestamps_list, spans_list_list
#         return records

#     def run(
#         self, 
#         storage: DataFlowStorage,
#         input_audio_key: str = "audio",
#         input_conversation_key: str = "conversation",
#         output_answer_key='forced_alignment_results',
#         sampling_rate: int = 16000,
#         language: str = "en",
#         micro_batch_size: int = 16,
#         chinese_to_pinyin: bool = False,
#         retain_word_level_alignment: bool = False,
#     ):
#         dataframe = storage.read('dataframe')
#         records = self.eval(
#             dataframe=dataframe,
#             input_audio_key=input_audio_key,
#             input_conversation_key=input_conversation_key,
#             sampling_rate=sampling_rate,
#             language=language,
#             micro_batch_size=micro_batch_size,
#             chinese_to_pinyin=chinese_to_pinyin,
#             retain_word_level_alignment=retain_word_level_alignment
#         )

#         # if word_timestamps_list:
#         #     dataframe['word_timestamps'] = word_timestamps_list
#         # dataframe['spans'] = spans_list_list
#         dataframe = dataframe.copy()
#         dataframe.loc[:, output_answer_key] = records
#         storage.write(dataframe)
    
#     def get_alignments(
#         self,
#         emissions: torch.Tensor,
#         tokens: list,
#         tokenizer,
#     ):
#         assert len(tokens) > 0, "Empty transcript"

#         dictionary = tokenizer.get_vocab()
#         dictionary = {k.lower(): v for k, v in dictionary.items()}
#         dictionary["<star>"] = len(dictionary)

#         # Force Alignment
#         token_indices = [
#             dictionary[c] for c in " ".join(tokens).split(" ") if c in dictionary
#         ]

#         blank_id = dictionary.get("<blank>", tokenizer.pad_token_id)

#         if not emissions.is_cpu:
#             emissions = emissions.cpu()
#         # targets = np.asarray([token_indices], dtype=np.int64)
#         targets = torch.tensor([token_indices], dtype=torch.int64)

#         path, scores = torchaudio.functional.forced_align(
#             log_probs=emissions.unsqueeze(0).float(),
#             targets=targets,
#             blank=blank_id,
#         )
#         path = path.squeeze().tolist()
#         scores = scores.view(-1).cpu().numpy()

#         idx_to_token_map = {v: k for k, v in dictionary.items()}
#         segments = merge_repeats(path, idx_to_token_map)
#         return segments, scores, idx_to_token_map[blank_id]


# 全局变量，用于在每个子进程中存储模型实例
_worker_model_processor = None

def _init_worker(devices, model_init_args):
    global _worker_model_processor
    # 取 worker 的编号（从 1 开始）
    rank = multiprocessing.current_process()._identity[0] - 1
    device = devices[rank % len(devices)]
    cfg = {**model_init_args, "device": device}
    _worker_model_processor = Aligner(**cfg)

def _parallel_worker(payload: Dict[str, Any]) -> List[List[Dict[str, float]]]:
    """子进程工作函数"""
    global _worker_model_processor

    audio_paths_chunk = payload['audio_paths_chunk']
    text_chunk = payload['text_chunk']
    ctc_params = payload['ctc_params']
    # sampling_rate = ctc_params['sampling_rate']
    # language = ctc_params['language']
    # micro_batch_size = ctc_params['micro_batch_size']
    # retain_word_level_alignment = ctc_params['retain_word_level_alignment']

    records = []
    # 使用已经存在于子进程中的 _worker_model_processor
    for audio_path, text in tqdm(zip(audio_paths_chunk, text_chunk), total=len(audio_paths_chunk), unit=" row", desc="CTC Forced Aligner..."):
        if isinstance(audio_path, list): 
            audio_path = audio_path[0]
        records.append(_worker_model_processor.process_audio_file(audio_path, text, **ctc_params))

    return records
    

@OPERATOR_REGISTRY.register()
class CTCForcedAlignSampleEvaluator(OperatorABC):
    '''
    CTCForcedAligner is a class that performs CTC forced aligner on audio data.
    '''
    def __init__(
        self, 
        model_path: str = "MahmoudAshraf/mms-300m-1130-forced-aligner",
        device: Union[str, List[str]] = "cuda", 
        num_workers: int = 1,
    ):
        self.logger = get_logger()
        self.model_init_args = {'model_path': model_path}
        self.num_workers = num_workers
        self.is_parallel = self.num_workers > 1

        if self.is_parallel:
            # --- 并行模式配置 ---
            self.logger.info(f"Running in multiprocessing mode: {self.num_workers}")
            # 主进程不加载模型，self.model 将为 None
            self.model = None
            self.device = None

            # 准备每个 worker 的静态配置
            self.devices = device if isinstance(device, list) else [device]
            # self.worker_configs = [
            #     {'device': devices[i % len(devices)], **self.model_init_args}
            #     for i in range(self.num_workers)
            # ]

            # 使用 initializer 在每个子进程启动时加载模型
            ctx = multiprocessing.get_context('spawn')
            self.pool = ctx.Pool(
                processes=self.num_workers,
                initializer=_init_worker,
                initargs=(self.devices, self.model_init_args),
            )
            self.logger.info("Worker initialized...")

        else:
            # --- 串行模式配置 ---
            single_device = device[0] if isinstance(device, list) else device
            self.logger.info(f"Running in serial mode: {single_device}")
            self.model_processor = Aligner(
                model_path=model_path, device=single_device
            )
    
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return "使用CTC强制对齐计算语音和文本转录对齐分数"
        else:
            return "Using CTCForcedAligner to compute the audio-text transcription alignment score"

    def eval(
        self, 
        dataframe: pd.DataFrame,
        input_audio_key: str = "audio",
        input_conversation_key: str = "conversation",
        sampling_rate: int = 16000,
        language: str = "en",
        micro_batch_size: int = 16,
        chinese_to_pinyin: bool = False,
        retain_word_level_alignment: bool = False,  
        romanize: bool = True,
    ):
        if chinese_to_pinyin:
            from pypinyin import lazy_pinyin
            def convert_chinese_to_pinyin(text: str) -> str:
                pinyin = lazy_pinyin(text, iso_code="cmn")
                return " ".join(pinyin)

        if romanize:
            import uroman
            ur = uroman.Uroman()

        ctc_params = {
            'sampling_rate': sampling_rate,
            'language': language,
            'micro_batch_size': micro_batch_size,
            'retain_word_level_alignment': retain_word_level_alignment,
        }

        self.logger.info("Running CTC Forced Aligner...")
        self.input_audio_key = input_audio_key
        audio_paths = dataframe.get(self.input_audio_key, pd.Series([])).tolist()
        conversations = dataframe.get(input_conversation_key, pd.Series([])).tolist()

        texts_normalized = []
        for conversation in conversations:
            if isinstance(conversation, list) and isinstance(conversation[0], dict) and 'value' in conversation[0]:
                text = conversation[0]['value']
            else:
                text = conversation
            
            if chinese_to_pinyin:
                text = convert_chinese_to_pinyin(text)
            
            if romanize:
                text = ur.romanize_string(text, lang=Lang(language).pt3)
            texts_normalized.append(text)

        if self.is_parallel:
            records = self._parallel_process(audio_paths, texts_normalized, ctc_params=ctc_params)
        else:
            records = self._serial_process(audio_paths, texts_normalized, ctc_params=ctc_params)
    
        return records
    
    def _serial_process(self, audio_paths: List[str], texts_normalized: List[str], ctc_params: Dict[str, Any]) -> List[dict]:
        self.logger.info("Start serial processing...")
        results = []
        for audio_path, text in tqdm(zip(audio_paths, texts_normalized), total=len(audio_paths), unit=" row", desc="CTC forced alignment"):
            if isinstance(audio_path, list): 
                audio_path = audio_path[0]
            results.append(self.model_processor.process_audio_file(audio_path, text, **ctc_params))
        return results

    def _parallel_process(self, audio_paths: List[str], texts_normalized: List[str], ctc_params: Dict[str, Any]) -> List[dict]:
        self.logger.info("Start parallel processing...")
        audio_chunks = np.array_split(audio_paths, self.num_workers)
        text_chunks = np.array_split(texts_normalized, self.num_workers)

        # 只需要准备每个任务的数据负载，不再需要配置信息
        worker_payloads = []
        for i, (audio_chunk, text_chunk) in enumerate(zip(audio_chunks, text_chunks)):
            if len(audio_chunk) > 0:
                # 【重点】每次分发任务时，都附带上模型配置信息
                # device = self.devices[i % len(self.devices)]
                # model_config = {'device': device, **self.model_init_args}
                
                payload = {
                    'audio_paths_chunk': audio_chunk.tolist(),
                    'text_chunk': text_chunk.tolist(),
                    'ctc_params': ctc_params,
                    # 'model_config': model_config  # 传入配置
                }
                worker_payloads.append(payload)
        
        # 直接使用已存在的 self.pool
        results_nested = list(tqdm(                             # List[List[dict]]
            self.pool.imap(_parallel_worker, worker_payloads),
            total=len(worker_payloads),
            desc="CTC Forced Aligner parallel processing..."
        ))

        
        return [item for sublist in results_nested for item in sublist]

    def run(
        self, 
        storage: DataFlowStorage,
        input_audio_key: str = "audio",
        input_conversation_key: str = "conversation",
        output_answer_key='forced_alignment_results',
        sampling_rate: int = 16000,
        language: str = "en",
        micro_batch_size: int = 16,
        chinese_to_pinyin: bool = False,
        retain_word_level_alignment: bool = False,
        romanize=True,
    ):
        dataframe = storage.read('dataframe')
        records = self.eval(
            dataframe=dataframe,
            input_audio_key=input_audio_key,
            input_conversation_key=input_conversation_key,
            sampling_rate=sampling_rate,
            language=language,
            micro_batch_size=micro_batch_size,
            chinese_to_pinyin=chinese_to_pinyin,
            retain_word_level_alignment=retain_word_level_alignment,
            romanize=romanize,
        )

        dataframe = dataframe.copy()
        dataframe.loc[:, output_answer_key] = records
        storage.write(dataframe)
    
    def close(self):
        if self.pool:
            self.pool.close()
            self.pool.join()

class Aligner:
    def __init__(self, model_path: str = "MahmoudAshraf/mms-300m-1130-forced-aligner", device: str = "cuda"):
        self.logger = get_logger()
        self.load_alignment_model(model_path=model_path, device=device)

    def load_alignment_model(
        self,
        model_path: str = "MahmoudAshraf/mms-300m-1130-forced-aligner",
        device: str = "cuda",
    ):
        self.device = device
        self.model, self.tokenizer = load_alignment_model(model_path=model_path, device=device)
    
    def process_audio_file(
        self, 
        audio_path: str,
        text: str,
        **kwargs,
    ):
        sampling_rate = kwargs.get('sampling_rate', 16000)
        language = kwargs.get('language', 'en')
        micro_batch_size = kwargs.get('micro_batch_size', 16)
        retain_word_level_alignment = kwargs.get('retain_word_level_alignment', False)
        try:
            audio, sr = librosa.load(audio_path, sr=sampling_rate)
            audio = torch.from_numpy(audio).to(self.device)

            spans_list = []

            emissions, stride = generate_emissions(self.model, audio, batch_size=micro_batch_size)
            tokens_starred, text_starred = preprocess_text(text, romanize=True, language=Lang(language).pt3)
            segments, scores, blank_token = self.get_alignments(emissions, tokens_starred, self.tokenizer)
            # segments, scores, blank_token = get_alignments(emissions, tokens_starred, self.tokenizer)
            spans = get_spans(tokens_starred, segments, blank_token)

            j = 0
            for seg_list in spans:
                for seg in seg_list:
                    spans_list.append({
                        'label': seg.label,
                        'start': seg.start,
                        'end': seg.end,
                        'score': math.exp(scores[seg.start: seg.end + 1][0]),
                    })
                    j += 1

            if retain_word_level_alignment:
                word_timestamps = postprocess_results(text_starred, spans, stride, scores)
                for i in range(len(word_timestamps)):
                            score = word_timestamps[i]['score']
                            word_timestamps[i]['score'] = math.exp(score)
            else:
                word_timestamps = []
        
            record = {
                'spans': spans_list,
                'word_timestamps': word_timestamps,
                'error': None,
            }
        except Exception as e:
            record = {
                'spans': [],
                'word_timestamps': [],
                'error': str(e),
            }
            self.logger.info(f"error: {str(e)}")

        return record
    
    def get_alignments(
        self,
        emissions: torch.Tensor,
        tokens: list,
        tokenizer,
    ):
        assert len(tokens) > 0, "Empty transcript"

        dictionary = tokenizer.get_vocab()
        dictionary = {k.lower(): v for k, v in dictionary.items()}
        dictionary["<star>"] = len(dictionary)

        # Force Alignment
        token_indices = [
            dictionary[c] for c in " ".join(tokens).split(" ") if c in dictionary
        ]

        blank_id = dictionary.get("<blank>", tokenizer.pad_token_id)

        if not emissions.is_cpu:
            emissions = emissions.cpu()
        # targets = np.asarray([token_indices], dtype=np.int64)
        targets = torch.tensor([token_indices], dtype=torch.int64)

        path, scores = torchaudio.functional.forced_align(
            log_probs=emissions.unsqueeze(0).float(),
            targets=targets,
            blank=blank_id,
        )
        path = path.squeeze().tolist()
        scores = scores.view(-1).cpu().numpy()

        idx_to_token_map = {v: k for k, v in dictionary.items()}
        segments = merge_repeats(path, idx_to_token_map)
        return segments, scores, idx_to_token_map[blank_id]