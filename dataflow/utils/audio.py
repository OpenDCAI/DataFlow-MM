import os
import math
import warnings
import base64
from io import BytesIO
from typing import List, Optional, Union, Dict, Tuple

import librosa
import numpy as np

# 不重采样
DEFAULT_SR = None

def _read_audio_local(path: str, sr: Optional[int] = DEFAULT_SR) -> Tuple[np.ndarray, int]:
    return librosa.load(path, sr=sr, mono=True)

def _read_audio_bytes(data: bytes, sr: Optional[int] = DEFAULT_SR) -> Tuple[np.ndarray, int]:
    return librosa.load(BytesIO(data), sr=sr, mono=True)

def _read_audio_base64(b64: str, sr: Optional[int] = DEFAULT_SR) -> Tuple[np.ndarray, int]:
    header, b64data = b64.split(",", 1)
    data = base64.b64decode(b64data)
    return _read_audio_bytes(data, sr=sr)

def process_audio_info(
    conversations: List[dict] | List[List[dict]],   # 这个conversation对应的是vllm中的messages列表(对应的是conversation_to_message函数的message)
    sampling_rate: Optional[int]
) -> Tuple[
    Optional[List[np.ndarray]],
    Optional[List[int]],
    Optional[List[str]]
]:
    """
    类似于 vision 的 process_vision_info，从 message 列表中提取音频输入。
    支持三种格式输入：
      - 本地或 http(s) URL 路径（通过 librosa 接口处理）
      - base64 编码 (data:audio/…;base64,…)
      - 直接传入 bytes 对象
    返回二元组：
      - audio_arrays: 解码后的 waveform (List[np.ndarray])
      - sample_rates: 采样率列表 (List[int])
    """
    if isinstance(conversations, list) and conversations and isinstance(conversations[0], dict):
        # 单条 conversaion
        conversations = [conversations]     # conversations被统一为List[List[dict]]

    audio_arrays = []
    sampling_rates = []

    for conv in conversations:
        for msg in conv:
            if not isinstance(msg.get("content"), list):
                continue
            for ele in msg["content"]:
                if ele.get("type") != "audio":
                    continue
                aud = ele.get("audio")
                if isinstance(aud, str):
                    if aud.startswith("data:audio") and "base64," in aud:
                        arr, sr = _read_audio_base64(aud, sr=sampling_rate)
                        audio_arrays.append(arr)
                        sampling_rates.append(sr)
                    elif aud.startswith("http://") or aud.startswith("https://"):
                        # 使用 librosa 支持远程路径
                        arr, sr = _read_audio_local(aud, sr=sampling_rate)
                        audio_arrays.append(arr)
                        sampling_rates.append(sr)
                    else:
                        # 本地路径
                        arr, sr = _read_audio_local(aud, sr=sampling_rate)
                        audio_arrays.append(arr)
                        sampling_rates.append(sr)
                elif isinstance(aud, (bytes, bytearray)):
                    arr, sr = _read_audio_bytes(bytes(aud), sr=sampling_rate)
                    audio_arrays.append(arr)
                    sampling_rates.append(sr)
                else:
                    raise ValueError(f"Unsupported audio type: {type(aud)}")

    if not audio_arrays:
        return None, None
    return audio_arrays, sampling_rates