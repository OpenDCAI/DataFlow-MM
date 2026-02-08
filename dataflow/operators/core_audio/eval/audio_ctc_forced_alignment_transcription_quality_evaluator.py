import os
import re
import math
import torch
import librosa
import torchaudio
import unicodedata
import numpy as np
import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger

import numpy as np

from packaging import version
from transformers import AutoModelForCTC, AutoTokenizer
from transformers import __version__ as transformers_version
from transformers.utils import is_flash_attn_2_available

from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC

from dataclasses import dataclass

from iso639 import Lang

from tqdm import tqdm
import multiprocessing
from typing import Union, List, Dict, Any

from uroman import Uroman

import math

from dataflow.utils.audio import (
    _read_audio_remote,
    _read_audio_local,
)


SAMPLING_FREQ = 16000

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

    records = []
    # 使用已经存在于子进程中的 _worker_model_processor
    for audio_path, text in zip(audio_paths_chunk, text_chunk):
        if isinstance(audio_path, list): 
            audio_path = audio_path[0]
        records.append(_worker_model_processor.process_audio_file(audio_path, text, **ctc_params))

    return records
    

@OPERATOR_REGISTRY.register()
class CTCForcedAlignmentSampleEvaluator(OperatorABC):
    '''
    CTCForcedAlignmentSampleEvaluator is a class that performs CTC forced alignment on audio data.
    '''
    def __init__(
        self, 
        model_path: str = "MahmoudAshraf/mms-300m-1130-forced-aligner",
        device: Union[str, List[str]] = "cuda", 
        num_workers: int = 1,
        sampling_rate: int = 16000,
        language: str = "en",
        micro_batch_size: int = 16,
        chinese_to_pinyin: bool = False,
        romanize: bool = True,
    ):
        self.logger = get_logger()
        self.model_init_args = {'model_path': model_path}
        self.num_workers = num_workers
        self.is_parallel = self.num_workers > 1
        self.sampling_rate = sampling_rate
        self.language = language
        self.micro_batch_size = micro_batch_size
        self.chinese_to_pinyin = chinese_to_pinyin
        self.romanize = romanize

        if self.is_parallel:
            # --- 并行模式配置 ---
            self.logger.info(f"Running in multiprocessing mode: {self.num_workers}")
            # 主进程不加载模型，self.model 将为 None
            self.model = None
            self.device = None

            # 准备每个 worker 的静态配置
            self.devices = device if isinstance(device, list) else [device]

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
            desc = (
                "CTCForcedAlignmentSampleEvaluator 算子基于 CTC 强制对齐模型，对语音与其转录文本进行对齐打分，"
                "支持串行与多进程并行两种运行模式。\n\n"

                "一、__init__ 初始化参数\n"
                "- model_path: str = \"MahmoudAshraf/mms-300m-1130-forced-aligner\"\n"
                "  预训练 CTC 强制对齐模型在 Hugging Face 或本地的路径。\n\n"
                "- device: Union[str, List[str]] = \"cuda\"\n"
                "  用于推理的设备配置：\n"
                "  * 串行模式：可以是 \"cpu\" 或 \"cuda\" 等单个设备字符串；\n"
                "  * 并行模式：可以是设备字符串列表，例如 [\"cuda:0\", \"cuda:1\"]，"
                "    不同进程会轮流使用这些设备。\n\n"
                "- num_workers: int = 1\n"
                "  工作进程数：\n"
                "  * num_workers <= 1：串行模式，仅在主进程中加载并运行模型；\n"
                "  * num_workers  > 1：多进程并行模式，每个子进程各自持有一份对齐模型实例。\n\n"
                "- sampling_rate: int = 16000\n"
                "  加载与重采样音频时使用的采样率（Hz）。\n\n"
                "- language: str = \"en\"\n"
                "  文本语言代码（ISO 639-1 或 639-3），用于文本正则化与罗马化，例如 \"en\"、\"zh\"。\n\n"
                "- micro_batch_size: int = 16\n"
                "  进行模型前向推理时的微批大小，用于控制 generate_emissions 中的批处理大小。\n\n"
                "- chinese_to_pinyin: bool = False\n"
                "  若为 True，将在对齐前使用 pypinyin 将中文文本转换为拼音（以空格分隔）。\n\n"
                "- romanize: bool = True\n"
                "  若为 True，则使用 uroman 对文本进行罗马化（适用于非拉丁文字），并在对齐时使用该罗马化序列。\n\n"
                "初始化行为：\n"
                "- 若 num_workers <= 1：\n"
                "  * 在指定 device 上实例化 Aligner，并在当前进程中加载模型；\n"
                "  * 后续所有样本在同一进程、同一设备上顺序处理。\n"
                "- 若 num_workers  > 1：\n"
                "  * 使用 multiprocessing (spawn) 启动 num_workers 个子进程；\n"
                "  * 通过 _init_worker 在每个子进程中创建 Aligner 实例并加载模型；\n"
                "  * 主进程仅负责切分数据并调度任务，不在主进程中加载模型本体。\n\n"

                "二、run 接口参数\n"
                "def run(\n"
                "    self,\n"
                "    storage: DataFlowStorage,\n"
                "    input_audio_key: str = \"audio\",\n"
                "    input_conversation_key: str = \"conversation\",\n"
                "    output_answer_key: str = \"forced_alignment_results\",\n"
                "):\n\n"
                "- storage: DataFlowStorage\n"
                "  数据流存储对象，上游应已将包含样本信息的 DataFrame 写入其中。\n\n"
                "- input_audio_key: str = \"audio\"\n"
                "  DataFrame 中音频路径所在的列名；每行可以是字符串或仅含单个路径的列表。\n\n"
                "- input_conversation_key: str = \"conversation\"\n"
                "  DataFrame 中文本/转录所在的列名：\n"
                "  * 若为字符串，视为整行文本；\n"
                "  * 若为带有字典的列表且字典含 'value' 键，则使用 conversation[0]['value'] 作为文本内容。\n\n"
                "- output_answer_key: str = \"forced_alignment_results\"\n"
                "  运行结束后，对齐结果将写入 DataFrame 的该列，类型为逐行一个 dict。\n\n"

                "三、run 执行行为\n"
                "1）从 storage 中读取上游写入的 DataFrame 。\n\n"
                "2）调用 eval 函数进行批量对齐：\n"
                "   - 根据 input_audio_key 获取音频路径列表 audio_paths；\n"
                "   - 根据 input_conversation_key 获取文本/对话列表 conversations；\n"
                "   - 对每条文本执行归一化预处理：\n"
                "     * 若 chinese_to_pinyin 为 True，则使用 pypinyin 将中文转换为拼音；\n"
                "     * 若 romanize 为 True，则使用 uroman 按照 language 指定的 ISO 639 代码进行罗马化；\n"
                "     * 统一转换得到 texts_normalized 列表。\n\n"
                "3）根据是否并行运行，选择不同的处理路径：\n"
                "   - 串行模式（num_workers <= 1）：\n"
                "     * 遍历 (audio_path, text_normalized) 对；\n"
                "     * 调用 Aligner.process_audio_file 逐条完成：\n"
                "       - 读取本地或远程音频（支持 http/https）；\n"
                "       - 重采样至指定 sampling_rate，转换为 torch.Tensor 并迁移至对应 device；\n"
                "       - 调用 generate_emissions 生成 CTC 发射概率（log_probs）；\n"
                "       - 调用 preprocess_text 将文本拆分为 tokens_starred 与 text_starred；\n"
                "       - 调用 get_alignments 进行强制对齐，得到 segments（Segment 列表）及 scores；\n"
                "       - 调用 get_spans 将对齐结果整理成帧级片段 spans；\n"
                "       - 若 retain_word_level_alignment 为 True，则调用 postprocess_results 计算词/片段级时间戳；\n"
                "       - 将所有信息整合为一条 record（dict）。\n\n"
                "   - 并行模式（num_workers > 1）：\n"
                "     * 使用 numpy.array_split 将 audio_paths 和 texts_normalized 按 worker 数划分为多个子列表；\n"
                "     * 为每个子列表构造 payload（包含 audio_paths_chunk、text_chunk 与 ctc_params）；\n"
                "     * 通过 multiprocessing.Pool 的 imap 接口分发到各个子进程，由 _parallel_worker 调用 "
                "       进程内缓存的 _worker_model_processor（即 Aligner 实例）进行批量处理；\n"
                "     * 收集各子进程返回的结果列表，并展开为与输入行数一一对应的 records 列表。\n\n"
                "4）将 records 作为新列 output_answer_key 写回到原 DataFrame 中，并通过 storage.write 持久化。\n\n"

                "四、输出结果格式\n"
                "run 函数执行结束后，DataFrame 中每一行会新增一列 output_answer_key，其值为一个 dict，结构为：\n\n"
                "- 'spans': List[Dict]\n"
                "  帧级对齐片段列表，每个元素表示一个连续的 CTC 对齐跨度，字段包括：\n"
                "  * 'label': str\n"
                "    该片段对应的标签（通常是对齐后的字符或 <blank>）。\n"
                "  * 'start': int\n"
                "    片段在 CTC 时间步上的起始索引（包含）。\n"
                "  * 'end': int\n"
                "    片段在 CTC 时间步上的结束索引（包含）。\n"
                "  * 'score': float\n"
                "    从 CTC 对齐 scores 中对应帧段的得分，经过 math.exp 后的值，可视作置信度近似。\n\n"
                "- 'word_timestamps': List[Dict]\n"
                "  （仅在 retain_word_level_alignment=True 时非空）词/片段级对齐结果列表，字段包括：\n"
                "  * 'start': float\n"
                "    该文本片段在原音频中的起始时间（秒）。\n"
                "  * 'end': float\n"
                "    该文本片段在原音频中的结束时间（秒）。\n"
                "  * 'text': str\n"
                "    对应的原始文本片段（来自 text_starred）。\n"
                "  * 'score': float\n"
                "    该片段聚合对齐得分的指数值（math.exp 之后）。\n\n"
                "- 'error': Optional[str]\n"
                "  若当前样本在处理/对齐过程中发生异常，则为错误信息字符串；\n"
                "  若无异常，则为 None。\n\n"
                "总结：\n"
                "该算子适合作为质量评估或标注辅助工具，用于计算语音-文本对的对齐质量指标，"
                "支持高吞吐量的多进程推理，并可根据需求选择是否输出词级时间戳。"
            )
        else:
            desc = (
                "CTCForcedAlignmentSampleEvaluator performs CTC-based forced alignment between "
                "speech audio and its transcription, supporting both serial and multi-process "
                "parallel execution.\n\n"

                "1. __init__ parameters\n"
                "- model_path: str = \"MahmoudAshraf/mms-300m-1130-forced-aligner\"\n"
                "  Path to the pretrained CTC forced alignment model (Hugging Face hub or local path).\n\n"
                "- device: Union[str, List[str]] = \"cuda\"\n"
                "  Device configuration for inference:\n"
                "  * Serial mode: a single string like \"cpu\" or \"cuda\";\n"
                "  * Parallel mode: a list of device strings, e.g. [\"cuda:0\", \"cuda:1\"],\n"
                "    and workers will round-robin over these devices.\n\n"
                "- num_workers: int = 1\n"
                "  Number of worker processes:\n"
                "  * num_workers <= 1: serial mode, model is loaded and used only in the main process;\n"
                "  * num_workers  > 1: multi-processing mode, each worker holds its own Aligner instance.\n\n"
                "- sampling_rate: int = 16000\n"
                "  Sampling rate (Hz) used when loading and resampling audio.\n\n"
                "- language: str = \"en\"\n"
                "  Language code (ISO 639-1 or 639-3) for text normalization and romanization, "
                "  e.g. \"en\", \"zh\".\n\n"
                "- micro_batch_size: int = 16\n"
                "  Micro batch size used when generating emissions in generate_emissions.\n\n"
                "- chinese_to_pinyin: bool = False\n"
                "  If True, Chinese text is first converted into pinyin (space-separated) using pypinyin.\n\n"
                "- romanize: bool = True\n"
                "  If True, text is romanized using uroman given the specified language code, which is\n"
                "  useful for non-Latin scripts.\n\n"
                "Initialization behavior:\n"
                "- If num_workers <= 1:\n"
                "  * Create a single Aligner instance on the specified device;\n"
                "  * All samples are processed sequentially within the same process and device.\n"
                "- If num_workers  > 1:\n"
                "  * Spawn num_workers child processes via multiprocessing (spawn context);\n"
                "  * Use _init_worker to create an Aligner instance and load the model in each worker;\n"
                "  * The main process only splits data and dispatches tasks; it does not load the model itself.\n\n"

                "2. run interface\n"
                "def run(\n"
                "    self,\n"
                "    storage: DataFlowStorage,\n"
                "    input_audio_key: str = \"audio\",\n"
                "    input_conversation_key: str = \"conversation\",\n"
                "    output_answer_key: str = \"forced_alignment_results\",\n"
                "):\n\n"
                "- storage: DataFlowStorage\n"
                "  DataFlow storage object. \n\n"
                "- input_audio_key: str = \"audio\"\n"
                "  Column name in the DataFrame containing audio paths. Each row can be a string or a\n"
                "  single-element list of strings.\n\n"
                "- input_conversation_key: str = \"conversation\"\n"
                "  Column name containing text/transcripts:\n"
                "  * If the cell is a string, it is treated as the full transcript;\n"
                "  * If the cell is a list whose first element is a dict with key 'value', then\n"
                "    conversation[0]['value'] is used as the text.\n\n"
                "- output_answer_key: str = \"forced_alignment_results\"\n"
                "  Name of the column where alignment results (one dict per row) will be written.\n\n"

                "3. run execution behavior\n"
                "1) Read the input DataFrame from storage.\n\n"
                "2) Call eval to perform batch alignment:\n"
                "   - Collect audio_paths from column input_audio_key;\n"
                "   - Collect conversations from column input_conversation_key;\n"
                "   - Normalize text for each row:\n"
                "     * If chinese_to_pinyin is True, convert Chinese text into pinyin;\n"
                "     * If romanize is True, romanize the text via uroman using the ISO 639 language code;\n"
                "     * Store all processed strings in texts_normalized.\n\n"
                "3) Depending on the execution mode:\n"
                "   - Serial mode (num_workers <= 1):\n"
                "     * Iterate over (audio_path, text_normalized) pairs;\n"
                "     * For each pair, call Aligner.process_audio_file which:\n"
                "       - Loads local or remote audio (http/https supported);\n"
                "       - Resamples to sampling_rate and moves tensor to the configured device;\n"
                "       - Calls generate_emissions to obtain CTC emissions (log-probs);\n"
                "       - Calls preprocess_text to obtain tokens_starred and text_starred;\n"
                "       - Calls get_alignments to perform forced alignment and obtain segments and scores;\n"
                "       - Calls get_spans to convert alignment results into frame-level spans;\n"
                "       - Optionally (if retain_word_level_alignment=True) calls postprocess_results to\n"
                "         compute word/segment-level timestamps;\n"
                "       - Packages all information into a single record dict.\n\n"
                "   - Parallel mode (num_workers > 1):\n"
                "     * Use numpy.array_split to split audio_paths and texts_normalized into chunks\n"
                "       according to num_workers;\n"
                "     * For each chunk, build a payload containing audio_paths_chunk, text_chunk and\n"
                "       ctc_params;\n"
                "     * Dispatch these payloads via multiprocessing.Pool.imap to _parallel_worker, which\n"
                "       routes the work to the per-process _worker_model_processor (Aligner instance);\n"
                "     * Collect the nested lists returned by workers and flatten them into a records list\n"
                "       aligned with the input rows.\n\n"
                "4) Attach records as a new column output_answer_key to the DataFrame and write it back\n"
                "   into storage.\n\n"

                "4. Output format\n"
                "After run completes, each row in the DataFrame will have a dict stored under\n"
                "output_answer_key, with the following structure:\n\n"
                "- 'spans': List[Dict]\n"
                "  Frame-level alignment segments. Each element describes a contiguous CTC span:\n"
                "  * 'label': str\n"
                "    Label of this span (typically an aligned character or <blank>).\n"
                "  * 'start': int\n"
                "    Start index of the span in CTC time steps (inclusive).\n"
                "  * 'end': int\n"
                "    End index of the span in CTC time steps (inclusive).\n"
                "  * 'score': float\n"
                "    Exponential of the alignment score over this frame range (math.exp), serving as\n"
                "    a confidence-like value.\n\n"
                "- 'word_timestamps': List[Dict]\n"
                "  (Non-empty only if retain_word_level_alignment=True) Word/segment-level alignment\n"
                "  results, each with:\n"
                "  * 'start': float\n"
                "    Start time (in seconds) in the original audio.\n"
                "  * 'end': float\n"
                "    End time (in seconds) in the original audio.\n"
                "  * 'text': str\n"
                "    Corresponding text chunk from text_starred.\n"
                "  * 'score': float\n"
                "    Exponential of the aggregated alignment score (math.exp).\n\n"
                "- 'error': Optional[str]\n"
                "  If an exception occurs while processing this sample, this field contains the\n"
                "  error message; otherwise it is None.\n\n"
                "In summary, this operator is suitable for evaluating the consistency between speech\n"
                "and transcription, providing both low-level frame spans and optional word-level\n"
                "timestamps, and can be scaled up via multi-process parallel inference."
            )
        return desc

    def eval(
        self, 
        dataframe: pd.DataFrame,
        input_audio_key: str = "audio",
        input_conversation_key: str = "conversation",
    ):
        if self.chinese_to_pinyin:
            from pypinyin import lazy_pinyin
            def convert_chinese_to_pinyin(text: str) -> str:
                pinyin = lazy_pinyin(text)
                return " ".join(pinyin)

        ctc_params = {
            'sampling_rate': self.sampling_rate,
            'language': self.language,
            'micro_batch_size': self.micro_batch_size,
            'romanize': self.romanize,
        }

        self.logger.info("Running CTC Forced Aligner...")
        # self.input_audio_key = input_audio_key
        audio_paths = dataframe.get(input_audio_key, pd.Series([])).tolist()
        conversations = dataframe.get(input_conversation_key, pd.Series([])).tolist()

        texts_normalized = []
        for conversation in conversations:
            if isinstance(conversation, list) and isinstance(conversation[0], dict) and 'value' in conversation[0]:
                text = conversation[0]['value']
            else:
                text = conversation
            
            if self.chinese_to_pinyin:
                text = convert_chinese_to_pinyin(text)
            
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
    ):
        dataframe = storage.read('dataframe')
        records = self.eval(
            dataframe=dataframe,
            input_audio_key=input_audio_key,
            input_conversation_key=input_conversation_key,
        )

        dataframe = dataframe.copy()
        dataframe.loc[:, output_answer_key] = records
        storage.write(dataframe)
    
    def close(self):
        if self.is_parallel:
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
        romanize = kwargs.get('romanize', False)

        try:
            # audio, sr = librosa.load(audio_path, sr=sampling_rate)
            if audio_path.startswith("http://") or audio_path.startswith("https://"):
                audio, sr = _read_audio_remote(audio_path, sr=sampling_rate)
            else:
                audio, sr = _read_audio_local(audio_path, sr=sampling_rate)
            audio = torch.from_numpy(audio).to(self.device)

            emissions, stride = generate_emissions(self.model, audio, batch_size=micro_batch_size)
            tokens_starred, text_starred = preprocess_text(text, romanize=romanize, language=Lang(language).pt3)
            segments, scores, blank_token = self.get_alignments(emissions, tokens_starred, self.tokenizer)
            spans = get_spans(tokens_starred, segments, blank_token)

            word_timestamps = postprocess_results(text_starred, spans, stride, scores)
            for i in range(len(word_timestamps)):
                score = word_timestamps[i]['score']
                word_timestamps[i]['score'] = math.exp(score)
        
            record = {
                'alignment': word_timestamps,
                'error': None,
            }
        except Exception as e:
            record = {
                'alignment': [],
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


@dataclass
class Segment:
    label: str
    start: int
    end: int

    def __repr__(self):
        return f"{self.label}: [{self.start:5d}, {self.end:5d})"

    @property
    def length(self):
        return self.end - self.start


def merge_repeats(path, idx_to_token_map):
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        while i2 < len(path) and path[i1] == path[i2]:
            i2 += 1
        segments.append(Segment(idx_to_token_map[path[i1]], i1, i2 - 1))
        i1 = i2
    return segments


def time_to_frame(time):
    stride_msec = 20
    frames_per_sec = 1000 / stride_msec
    return int(time * frames_per_sec)


def get_spans(tokens, segments, blank):
    ltr_idx = 0
    tokens_idx = 0
    intervals = []
    start, end = (0, 0)
    for seg_idx, seg in enumerate(segments):
        if tokens_idx == len(tokens):
            assert seg_idx == len(segments) - 1
            assert seg.label == blank
            continue
        cur_token = tokens[tokens_idx].split(" ")
        ltr = cur_token[ltr_idx]
        if seg.label == blank:
            continue
        assert seg.label == ltr, f"{seg.label} != {ltr}"
        if (ltr_idx) == 0:
            start = seg_idx
        if ltr_idx == len(cur_token) - 1:
            ltr_idx = 0
            tokens_idx += 1
            intervals.append((start, seg_idx))
            while tokens_idx < len(tokens) and len(tokens[tokens_idx]) == 0:
                intervals.append((seg_idx, seg_idx))
                tokens_idx += 1
        else:
            ltr_idx += 1
    spans = []
    for idx, (start, end) in enumerate(intervals):
        span = segments[start : end + 1]
        if start > 0:
            prev_seg = segments[start - 1]
            if prev_seg.label == blank:
                pad_start = (
                    prev_seg.start
                    if (idx == 0)
                    else int((prev_seg.start + prev_seg.end) / 2)
                )
                span = [Segment(blank, pad_start, span[0].start)] + span
        if end + 1 < len(segments):
            next_seg = segments[end + 1]
            if next_seg.label == blank:
                pad_end = (
                    next_seg.end
                    if (idx == len(intervals) - 1)
                    else math.floor((next_seg.start + next_seg.end) / 2)
                )
                span = span + [Segment(blank, span[-1].end, pad_end)]
        spans.append(span)
    return spans


def load_audio(audio_file: str, dtype: torch.dtype, device: str):
    waveform, audio_sf = torchaudio.load(audio_file)  # waveform: channels X T
    waveform = torch.mean(waveform, dim=0)

    if audio_sf != SAMPLING_FREQ:
        waveform = torchaudio.functional.resample(
            waveform, orig_freq=audio_sf, new_freq=SAMPLING_FREQ
        )
    waveform = waveform.to(dtype).to(device)
    return waveform


def generate_emissions(
    model,
    audio_waveform: torch.Tensor,
    window_length=30,
    context_length=2,
    batch_size=4,
):
    batch_size = max(batch_size, 1)
    window = int(window_length * SAMPLING_FREQ)
    if audio_waveform.size(0) < window:
        extension = 0
        context = 0
        input_tensor = audio_waveform.unsqueeze(0)
    else:
        # batching the input tensor and including a context
        # before and after the input tensor
        context = int(context_length * SAMPLING_FREQ)
        extension = math.ceil(
            audio_waveform.size(0) / window
        ) * window - audio_waveform.size(0)
        padded_waveform = torch.nn.functional.pad(
            audio_waveform, (context, context + extension)
        )
        input_tensor = padded_waveform.unfold(0, window + 2 * context, window)

    # Batched Inference
    emissions_arr = []
    with torch.inference_mode():
        for i in range(0, input_tensor.size(0), batch_size):
            input_batch = input_tensor[i : i + batch_size]
            emissions_ = model(input_batch).logits
            emissions_arr.append(emissions_)

    emissions = torch.cat(emissions_arr, dim=0)
    if context > 0:
        emissions = emissions[
            :,
            time_to_frame(context_length) : -time_to_frame(context_length) + 1,
        ]  # removing the context
    emissions = emissions.flatten(0, 1)

    if time_to_frame(extension / SAMPLING_FREQ) > 0:
        emissions = emissions[: -time_to_frame(extension / SAMPLING_FREQ)]

    emissions = torch.log_softmax(emissions, dim=-1)
    emissions = torch.cat(
        [emissions, torch.zeros(emissions.size(0), 1).to(emissions.device)], dim=1
    )  # adding a star token dimension
    stride = float(audio_waveform.size(0) * 1000 / emissions.size(0) / SAMPLING_FREQ)

    return emissions, math.ceil(stride)


def load_alignment_model(
    device: str,
    model_path: str = "MahmoudAshraf/mms-300m-1130-forced-aligner",
    attn_implementation: str = None,
    dtype: torch.dtype = torch.float32,
):
    if attn_implementation is None:
        if version.parse(transformers_version) < version.parse("4.41.0"):
            attn_implementation = "eager"
        elif (
            is_flash_attn_2_available()
            and device.startswith("cuda")
            and dtype in [torch.float16, torch.bfloat16]
        ):
            attn_implementation = "flash_attention_2"
        else:
            attn_implementation = "sdpa"

    model = (
        AutoModelForCTC.from_pretrained(
            model_path,
            attn_implementation=attn_implementation,
            torch_dtype=dtype,
        )
        .to(device)
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    return model, tokenizer



colon = ":"
comma = ","
exclamation_mark = "!"
period = re.escape(".")
question_mark = re.escape("?")
semicolon = ";"

left_curly_bracket = "{"
right_curly_bracket = "}"
quotation_mark = '"'

basic_punc = (
    period
    + question_mark
    + comma
    + colon
    + exclamation_mark
    + left_curly_bracket
    + right_curly_bracket
)

# General punc unicode block (0x2000-0x206F)
zero_width_space = r"\u200B"
zero_width_nonjoiner = r"\u200C"
left_to_right_mark = r"\u200E"
right_to_left_mark = r"\u200F"
left_to_right_embedding = r"\u202A"
pop_directional_formatting = r"\u202C"

# Here are some commonly ill-typed versions of apostrophe
right_single_quotation_mark = r"\u2019"
left_single_quotation_mark = r"\u2018"

# Language specific definitions
# Spanish
inverted_exclamation_mark = r"\u00A1"
inverted_question_mark = r"\u00BF"


# Hindi
hindi_danda = "\u0964"

# Egyptian Arabic
# arabic_percent = r"\u066A"
arabic_comma = r"\u060C"
arabic_question_mark = r"\u061F"
arabic_semicolon = r"\u061B"
arabic_diacritics = r"\u064B-\u0652"


arabic_subscript_alef_and_inverted_damma = r"\u0656-\u0657"


# Chinese
full_stop = r"\u3002"
full_comma = r"\uFF0C"
full_exclamation_mark = r"\uFF01"
full_question_mark = r"\uFF1F"
full_semicolon = r"\uFF1B"
full_colon = r"\uFF1A"
full_parentheses = r"\uFF08\uFF09"
quotation_mark_horizontal = r"\u300C-\u300F"
quotation_mark_vertical = r"\uFF41-\uFF44"
title_marks = r"\u3008-\u300B"
wavy_low_line = r"\uFE4F"
ellipsis = r"\u22EF"
enumeration_comma = r"\u3001"
hyphenation_point = r"\u2027"
forward_slash = r"\uFF0F"
wavy_dash = r"\uFF5E"
box_drawings_light_horizontal = r"\u2500"
fullwidth_low_line = r"\uFF3F"
chinese_punc = (
    full_stop
    + full_comma
    + full_exclamation_mark
    + full_question_mark
    + full_semicolon
    + full_colon
    + full_parentheses
    + quotation_mark_horizontal
    + quotation_mark_vertical
    + title_marks
    + wavy_low_line
    + ellipsis
    + enumeration_comma
    + hyphenation_point
    + forward_slash
    + wavy_dash
    + box_drawings_light_horizontal
    + fullwidth_low_line
)

# Armenian
armenian_apostrophe = r"\u055A"
emphasis_mark = r"\u055B"
exclamation_mark = r"\u055C"
armenian_comma = r"\u055D"
armenian_question_mark = r"\u055E"
abbreviation_mark = r"\u055F"
armenian_full_stop = r"\u0589"
armenian_punc = (
    armenian_apostrophe
    + emphasis_mark
    + exclamation_mark
    + armenian_comma
    + armenian_question_mark
    + abbreviation_mark
    + armenian_full_stop
)

lesser_than_symbol = r"&lt;"
greater_than_symbol = r"&gt;"

lesser_than_sign = r"\u003c"
greater_than_sign = r"\u003e"

nbsp_written_form = r"&nbsp"

# Quotation marks
left_double_quotes = r"\u201c"
right_double_quotes = r"\u201d"
left_double_angle = r"\u00ab"
right_double_angle = r"\u00bb"
left_single_angle = r"\u2039"
right_single_angle = r"\u203a"
low_double_quotes = r"\u201e"
low_single_quotes = r"\u201a"
high_double_quotes = r"\u201f"
high_single_quotes = r"\u201b"

all_punct_quotes = (
    left_double_quotes
    + right_double_quotes
    + left_double_angle
    + right_double_angle
    + left_single_angle
    + right_single_angle
    + low_double_quotes
    + low_single_quotes
    + high_double_quotes
    + high_single_quotes
    + right_single_quotation_mark
    + left_single_quotation_mark
)
mapping_quotes = (
    "["
    + high_single_quotes
    + right_single_quotation_mark
    + left_single_quotation_mark
    + "]"
)


# Digits

english_digits = r"\u0030-\u0039"
bengali_digits = r"\u09e6-\u09ef"
khmer_digits = r"\u17e0-\u17e9"
devanagari_digits = r"\u0966-\u096f"
oriya_digits = r"\u0b66-\u0b6f"
extended_arabic_indic_digits = r"\u06f0-\u06f9"
kayah_li_digits = r"\ua900-\ua909"
fullwidth_digits = r"\uff10-\uff19"
malayam_digits = r"\u0d66-\u0d6f"
myanmar_digits = r"\u1040-\u1049"
roman_numeral = r"\u2170-\u2179"
nominal_digit_shapes = r"\u206f"

# Load punctuations from MMS-lab data
with open(
    f"{os.path.dirname(__file__)}/punctuations.lst", "r", encoding="utf-8-sig"
) as punc_f:
    punc_list = punc_f.readlines()

punct_pattern = r""
for punc in punc_list:
    # the first character in the tab separated line is the punc to be removed
    punct_pattern += re.escape(punc.split("\t")[0])

shared_digits = (
    english_digits
    + bengali_digits
    + khmer_digits
    + devanagari_digits
    + oriya_digits
    + extended_arabic_indic_digits
    + kayah_li_digits
    + fullwidth_digits
    + malayam_digits
    + myanmar_digits
    + roman_numeral
    + nominal_digit_shapes
)

shared_punc_list = (
    basic_punc
    + all_punct_quotes
    + greater_than_sign
    + lesser_than_sign
    + inverted_question_mark
    + full_stop
    + semicolon
    + armenian_punc
    + inverted_exclamation_mark
    + arabic_comma
    + enumeration_comma
    + hindi_danda
    + quotation_mark
    + arabic_semicolon
    + arabic_question_mark
    + chinese_punc
    + punct_pattern
)

shared_mappping = {
    lesser_than_symbol: "",
    greater_than_symbol: "",
    nbsp_written_form: "",
    r"(\S+)" + mapping_quotes + r"(\S+)": r"\1'\2",
}

shared_deletion_list = (
    left_to_right_mark
    + zero_width_nonjoiner
    + arabic_subscript_alef_and_inverted_damma
    + zero_width_space
    + arabic_diacritics
    + pop_directional_formatting
    + right_to_left_mark
    + left_to_right_embedding
)

norm_config = {
    "*": {
        "lower_case": True,
        "punc_set": shared_punc_list,
        "del_set": shared_deletion_list,
        "mapping": shared_mappping,
        "digit_set": shared_digits,
        "unicode_norm": "NFKC",
        "rm_diacritics": False,
    }
}

# =============== Mongolian =============== #

norm_config["mon"] = norm_config["*"].copy()
# add soft hyphen to punc list to match with fleurs
norm_config["mon"]["del_set"] += r"\u00AD"

norm_config["khk"] = norm_config["mon"].copy()

# =============== Hebrew =============== #

norm_config["heb"] = norm_config["*"].copy()
# add "HEBREW POINT" symbols to match with fleurs
norm_config["heb"]["del_set"] += r"\u05B0-\u05BF\u05C0-\u05CF"

# =============== Thai =============== #

norm_config["tha"] = norm_config["*"].copy()
# add "Zero width joiner" symbols to match with fleurs
norm_config["tha"]["punc_set"] += r"\u200D"

# =============== Arabic =============== #
norm_config["ara"] = norm_config["*"].copy()
norm_config["ara"]["mapping"]["ٱ"] = "ا"
norm_config["ara"]["mapping"]["ٰ"] = "ا"
norm_config["ara"]["mapping"]["ۥ"] = "و"
norm_config["ara"]["mapping"]["ۦ"] = "ي"
norm_config["ara"]["mapping"]["ـ"] = ""
norm_config["ara"]["mapping"]["ٓ"] = ""
norm_config["ara"]["mapping"]["ٔ"] = "ء"
norm_config["ara"]["mapping"]["ٕ"] = "ء"
norm_config["arb"] = norm_config["ara"].copy()

# =============== Javanese =============== #
norm_config["jav"] = norm_config["*"].copy()
norm_config["jav"]["rm_diacritics"] = True


uroman_instance = Uroman()


def text_normalize(
    text, iso_code, lower_case=True, remove_numbers=True, remove_brackets=False
):
    """Given a text, normalize it by changing to lower case, removing punctuations,
    removing words that only contain digits and removing extra spaces

    Args:
        text : The string to be normalized
        iso_code : ISO 639-3 code of the language
        remove_numbers : Boolean flag to specify if words containing only digits should be removed

    Returns:
        normalized_text : the string after all normalization

    """

    config = norm_config.get(iso_code, norm_config["*"])

    for field in [
        "lower_case",
        "punc_set",
        "del_set",
        "mapping",
        "digit_set",
        "unicode_norm",
    ]:
        if field not in config:
            config[field] = norm_config["*"][field]

    text = unicodedata.normalize(config["unicode_norm"], text)

    # Convert to lower case

    if config["lower_case"] and lower_case:
        text = text.lower()

    # brackets

    # always text inside brackets with numbers in them. Usually corresponds to "(Sam 23:17)"
    text = re.sub(r"\([^\)]*\d[^\)]*\)", " ", text)
    if remove_brackets:
        text = re.sub(r"\([^\)]*\)", " ", text)

    # Apply mappings

    for old, new in config["mapping"].items():
        text = re.sub(old, new, text)

    # Replace punctutations with space

    punct_pattern = r"[" + config["punc_set"]

    punct_pattern += r"]"

    normalized_text = re.sub(punct_pattern, " ", text)

    # remove characters in delete list

    delete_patten = r"[" + config["del_set"] + r"]"

    normalized_text = re.sub(delete_patten, "", normalized_text)

    # Remove words containing only digits
    # We check for 3 cases:
    #   a)text starts with a number
    #   b) a number is present somewhere in the middle of the text
    #   c) the text ends with a number
    # For each case we use lookaround regex pattern to see if the digit pattern in preceded
    # and followed by whitespaces, only then we replace the numbers with space
    # The lookaround enables overlapping pattern matches to be replaced

    if remove_numbers:
        digits_pattern = r"[" + config["digit_set"]

        digits_pattern += r"]+"

        complete_digit_pattern = (
            r"^"
            + digits_pattern
            + r"(?=\s)|(?<=\s)"
            + digits_pattern
            + r"(?=\s)|(?<=\s)"
            + digits_pattern
            + r"$"
        )

        normalized_text = re.sub(complete_digit_pattern, " ", normalized_text)

    if config["rm_diacritics"]:
        from unidecode import unidecode

        normalized_text = unidecode(normalized_text)

    # Remove extra spaces
    normalized_text = re.sub(r"\s+", " ", normalized_text).strip()

    return normalized_text


# iso codes with specialized rules in uroman
special_isos_uroman = [
    "ara",
    "bel",
    "bul",
    "deu",
    "ell",
    "eng",
    "fas",
    "grc",
    "ell",
    "eng",
    "heb",
    "kaz",
    "kir",
    "lav",
    "lit",
    "mkd",
    "mkd2",
    "oss",
    "pnt",
    "pus",
    "rus",
    "srp",
    "srp2",
    "tur",
    "uig",
    "ukr",
    "yid",
]


def normalize_uroman(text):
    text = text.lower()
    text = re.sub("([^a-z' ])", " ", text)
    text = re.sub(" +", " ", text)
    return text.strip()


def get_uroman_tokens(norm_transcripts: list[str], iso=None):
    outtexts = [
        uroman_instance.romanize_string(transcript, lcode=iso)
        for transcript in norm_transcripts
    ]

    uromans = []
    for ot in outtexts:
        ot = " ".join(ot.strip())
        ot = re.sub(r"\s+", " ", ot).strip()
        normalized = normalize_uroman(ot)
        uromans.append(normalized)

    assert len(uromans) == len(norm_transcripts)

    return uromans


def split_text(text: str, split_size: str = "word"):
    if split_size == "sentence":
        from nltk.tokenize import PunktSentenceTokenizer

        sentence_checker = PunktSentenceTokenizer()
        sentences = sentence_checker.sentences_from_text(text)
        return sentences

    elif split_size == "word":
        return text.split()
    elif split_size == "char":
        return list(text)


def preprocess_text(
    text, romanize, language, split_size="word", star_frequency="segment"
):
    assert split_size in [
        "sentence",
        "word",
        "char",
    ], "Split size must be sentence, word, or char"
    assert star_frequency in [
        "segment",
        "edges",
    ], "Star frequency must be segment or edges"

    text_split = split_text(text, split_size)
    norm_text = [text_normalize(line.strip(), language) for line in text_split]

    if romanize:
        tokens = get_uroman_tokens(norm_text, language)
    else:
        tokens = [" ".join(list(word)) for word in norm_text]

    # add <star> token to the tokens and text
    # it's used extensively here but I found that it produces more accurate results
    # and doesn't affect the runtime
    if star_frequency == "segment":
        tokens_starred = []
        [tokens_starred.extend(["<star>", token]) for token in tokens]

        text_starred = []
        [text_starred.extend(["<star>", chunk]) for chunk in text_split]

    elif star_frequency == "edges":
        tokens_starred = ["<star>"] + tokens + ["<star>"]
        text_starred = ["<star>"] + text_split + ["<star>"]

    return tokens_starred, text_starred


def merge_segments(segments, threshold=0.00):
    for i in range(len(segments) - 1):
        if segments[i + 1]["start"] - segments[i]["end"] < threshold:
            segments[i + 1]["start"] = segments[i]["end"]


def postprocess_results(
    text_starred: list,
    spans: list,
    stride: float,
    scores: np.ndarray,
    merge_threshold: float = 0.0,
):
    results = []

    for i, t in enumerate(text_starred):
        if t == "<star>":
            continue
        span = spans[i]
        seg_start_idx = span[0].start
        seg_end_idx = span[-1].end

        audio_start_sec = seg_start_idx * (stride) / 1000
        audio_end_sec = seg_end_idx * (stride) / 1000
        score = scores[seg_start_idx:seg_end_idx].mean()
        sample = {
            "start": audio_start_sec,
            "end": audio_end_sec,
            "text": t,
            "score": score.item(),
        }
        results.append(sample)

    merge_segments(results, merge_threshold)
    return results
