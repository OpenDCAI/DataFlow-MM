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

from typing import Optional, Tuple


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
    for audio_path, text in tqdm(zip(audio_paths_chunk, text_chunk), total=len(audio_paths_chunk), unit=" row", desc="CTC Forced Aligner..."):
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
            return "Using CTCForcedAlignmentSampleEvaluator to compute the audio-text transcription alignment score"

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
            and device == "cuda"
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
    if language in ["jpn", "chi"]:
        split_size = "char"
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
        score = scores[seg_start_idx:seg_end_idx].sum()
        sample = {
            "start": audio_start_sec,
            "end": audio_end_sec,
            "text": t,
            "score": score.item(),
        }
        results.append(sample)

    merge_segments(results, merge_threshold)
    return results
