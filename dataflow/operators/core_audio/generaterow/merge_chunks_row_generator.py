import librosa
import soundfile as sf
import numpy as np
import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger

from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC

from pathlib import Path
from typing import Literal, Optional

from tqdm import tqdm
import multiprocessing
from functools import partial

from dataflow.utils.audio import (
    _read_audio_remote,
    _read_audio_local,
)

def chunk_list(data, num_chunks):
    k, m = divmod(len(data), num_chunks)
    return [data[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(num_chunks) if data[i * k + min(i, m):(i + 1) * k + min(i + 1, m)]]

@OPERATOR_REGISTRY.register()
class MergeChunksRowGenerator(OperatorABC):
    def __init__(self, num_workers=1):
        super().__init__()
        self.logger = get_logger(__name__)
        self.num_workers = num_workers
        self.is_parallel = self.num_workers > 1
        self.pool = None        # 持久化进程池的占位符
        
        if self.is_parallel:
            ctx = multiprocessing.get_context('spawn')
            self.pool = ctx.Pool(processes=self.num_workers)
            self.logger.info("Worker initialized...")

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            desc = (
                "该算子根据 VAD 或对齐生成的时间戳（frame 或 time），"
                "从原始音频中按顺序提取语音片段，并按最大时长限制进行合并，"
                "最终生成多个新的音频序列文件（chunk 合并结果）。\n\n"

                "初始化参数（__init__）：\n"
                "- num_workers: 并行进程数，大于 1 时启用多进程处理。\n\n"

                "运行参数（run）：\n"
                "- dst_folder: 输出音频保存目录，若为空则保存到原音频目录。\n"
                "- input_audio_key: 输入 audio 列名，默认 'audio'。\n"
                "- input_timestamps_key: 输入 timestamp 列名（如 VAD 结果），默认 'timestamps'。\n"
                "- timestamp_type: 时间戳类型，'frame'（根据 hop_size 转时间）或 'time'（直接秒），默认 'time'。\n"
                "- max_audio_duration: 单个合并后音频的最大时长（秒），超过后开始新序列。\n"
                "- hop_size_samples: 当 timestamp_type='frame' 时，用于帧转秒的 hop 大小（样本数），默认 512。\n"
                "- sampling_rate: 音频加载与输出采样率，默认 16000。\n\n"

                "运行行为：\n"
                "1. 从 storage 读取 dataframe，依次获取音频路径与对应 timestamps。\n"
                "2. 根据 timestamp_type，将帧或时间戳转换为秒级范围。\n"
                "3. 按时间顺序提取音频片段，并按 max_audio_duration 合并成多个序列。\n"
                "4. 每个序列写成新的音频文件，文件名格式如：<原名>_<序列号>.wav。\n"
                "5. 写回 dataframe，每行对应一个新生成的 chunk 。\n\n"

                "输出：\n"
                "- 覆盖写回 storage 的 'dataframe'，包含多行，每行为：\n"
                "  * audio: 合并后音频文件路径（列表形式）\n"
                "  * original_audio_path: 源音频路径\n"
                "  * conversation: 固定为 [{'from': 'human', 'value': '<audio>'}]\n"
                "  * sequence_num: 当前源音频对应的序列编号\n\n"
            )
        else:
            desc = (
                "This operator merges audio chunks based on timestamps (frame-based or time-based), "
                "extracting segments from the original audio and grouping them into sequences "
                "according to a maximum duration limit. Each resulting sequence becomes a new audio file.\n\n"

                "Initialization parameters (__init__):\n"
                "- num_workers: Number of parallel workers; >1 enables multiprocessing.\n\n"

                "Runtime parameters (run):\n"
                "- dst_folder: Folder to write output audio files; if None, saves next to source audio.\n"
                "- input_audio_key: Column name for input audio paths, default 'audio'.\n"
                "- input_timestamps_key: Column name for timestamps (e.g., VAD results), default 'timestamps'.\n"
                "- timestamp_type: 'frame' (converted using hop_size) or 'time' (seconds), default 'time'.\n"
                "- max_audio_duration: Maximum duration (seconds) of a merged sequence; "
                "a new sequence is started once exceeded.\n"
                "- hop_size_samples: Hop size (samples) used when timestamp_type='frame', default 512.\n"
                "- sampling_rate: Sampling rate for loading and output, default 16000.\n\n"

                "Runtime behavior:\n"
                "1. Read dataframe from storage and obtain audio paths and timestamps.\n"
                "2. Convert timestamps to seconds if needed.\n"
                "3. Extract audio segments in chronological order.\n"
                "4. Merge segments into sequences based on max_audio_duration.\n"
                "5. Write each merged sequence as a new WAV file named: <original_name>_<sequence_num>.wav\n"
                "6. Write back a new dataframe where each row corresponds to one generated audio sequence.\n\n"

                "Output:\n"
                "- A rewritten 'dataframe' stored in storage, where each row contains:\n"
                "  * audio: Path to the merged audio file (as a list)\n"
                "  * original_audio_path: Path of the source audio\n"
                "  * conversation: Fixed as [{'from': 'human', 'value': '<audio>'}]\n"
                "  * sequence_num: Sequence number for that source audio\n\n"
            )
        return desc


    def run(self,
        storage: DataFlowStorage,
        dst_folder: str,
        input_audio_key: str = "audio",
        input_timestamps_key: str = "timestamps",
        timestamp_type: Literal["frame", "time"] = "time",  # 手动指定类型
        max_audio_duration: float = float('inf'),
        hop_size_samples: int = 512,  # hop_size, 是样本点数量
        sampling_rate: int = 16000,
    ):
        # 参数验证
        if timestamp_type not in ["frame", "time"]:
            raise ValueError(f"timestamp_type must be 'frame' or 'time'")
        
        dataframe = storage.read('dataframe')
        audio_column = dataframe[input_audio_key]
        timestamps_column = dataframe[input_timestamps_key]
        output_dataframe = []

        args_iter = [
            (audio_path, dst_folder, timestamps, timestamp_type,
             max_audio_duration, hop_size_samples, sampling_rate)
            for audio_path, timestamps in zip(audio_column, timestamps_column)
        ]

        if self.is_parallel:
            args_chunks = chunk_list(args_iter, self.num_workers)
            # args_chunks = [chunk.tolist() for chunk in args_chunks if len(chunk) > 0]
            worker = partial(MergeChunksRowGenerator._process_audio_chunk_static)
            for chunk_result in tqdm(self.pool.imap(worker, args_chunks), total=len(args_chunks),
                            desc="Merging Chunks", unit=" chunk"):
                output_dataframe.extend(chunk_result)

            # output_dataframe = list(tqdm(self.pool.imap(worker, args_chunks), total=self.num_workers,
            #                 desc="Merging Chunks", unit=" row"))
        else:
            for args in tqdm(args_iter, desc="Merging", unit=" row", total=len(args_iter)):
                ret = MergeChunksRowGenerator._process_audio(args)
                output_dataframe.extend(ret)

        output_dataframe = pd.DataFrame(output_dataframe)
        storage.write(output_dataframe)

    def close(self):
        if self.pool:
            self.pool.close()
            self.pool.join()
            self.logger.info("Worker pool closed")

    @staticmethod
    def _process_audio_chunk_static(args_chunk):
        """静态方法封装，给多进程用"""
        results = []
        for args in tqdm(args_chunk, desc="Merging", unit=" row", total=len(args_chunk)):
            results.extend(MergeChunksRowGenerator._process_audio(args))
        return results

    @staticmethod
    def _process_audio(args):
        audio_path = args[0]
        dst_folder = args[1]
        timestamps = args[2]
        timestamp_type = args[3]
        max_audio_duration = args[4]
        hop_size_samples = args[5]
        sampling_rate = args[6]

        if isinstance(audio_path, list):
            audio_path = audio_path[0]

        try:
            # audio, sr = librosa.load(audio_path, sr=sampling_rate)
            if audio_path.startswith("http://") or audio_path.startswith("https://"):
                audio, sr = _read_audio_remote(audio_path, sr=sampling_rate)
            else:
                audio, sr = _read_audio_local(audio_path, sr=sampling_rate)
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return []
        total_duration = len(audio) / sr
        converted_timestamps = convert_to_timestamps(
                timestamp_type=timestamp_type,
                timestamps=timestamps,
                hop_size_samples=hop_size_samples,
                sampling_rate=sampling_rate
            )

        audio_sequences = []
        current_segments = []  # 当前序列的音频片段
        current_duration = 0   # 当前序列的累计时长
        current_timestamps = []  # 当前序列使用的时间戳
        sequence_num = 1       # 序列编号
                
        for i, ts in enumerate(converted_timestamps):
            start_time = max(0, ts["start"])
            end_time = min(total_duration, ts["end"])
                    
            if start_time >= end_time:  # 跳过无效片段
                continue

            segment_duration = end_time - start_time
                    
            # 检查最大时长限制
            if current_duration + segment_duration > max_audio_duration and current_segments:
                # 已达到最长限制
                audio_sequences.append({
                    "segments": current_segments.copy(),
                    "timestamps": current_timestamps.copy(),
                    "duration": current_duration,
                    "sequence_num": sequence_num,
                    "source_audio_path": audio_path
                })

                # 重置为新序列
                current_segments = []
                current_duration = 0
                current_timestamps = []
                sequence_num += 1
                
            # 添加当前片段到序列
            start_sample = int(start_time * sampling_rate)
            end_sample = int(end_time * sampling_rate)
            segment = audio[start_sample:end_sample]
                
            current_segments.append(segment)
            current_timestamps.append(timestamps[i])
            current_duration += segment_duration
                
        # 保存最后一个序列
        if current_segments:
            audio_sequences.append({
                "segments": current_segments,
                "timestamps": current_timestamps,
                "duration": current_duration,
                "sequence_num": sequence_num,
                "source_audio_path": audio_path,
            })

        if not audio_sequences:
            return []
            
        # 写入文件
        audio_name = Path(audio_path).stem
        if dst_folder:
            audio_dir = Path(dst_folder)
        else:
            audio_dir = Path(audio_path).parent

        ret = []
        for seq in audio_sequences:
            # 合并当前序列的所有片段
            merged_audio = np.concatenate(seq["segments"])
                        
            # 生成文件名
            if seq["timestamps"]:
                if len(seq["segments"]) == 1:
                    # 单个片段
                    output_filename = f"{audio_name}_{seq['sequence_num']}.wav"
                else:
                    # 多个片段，添加序列号
                    output_filename = f"{audio_name}_{seq['sequence_num']}.wav"
                    
            output_path = audio_dir / output_filename
            sf.write(output_path, merged_audio, sampling_rate)
            ret.append(
                {
                    'audio': [str(output_path)],
                    'original_audio_path': str(audio_path),
                    'conversation': [{"from": "human", "value": "<audio>"}],
                    'sequence_num': seq['sequence_num'],
                }
            )
        return ret

def convert_to_timestamps(
    timestamp_type,
    timestamps,
    hop_size_samples,
    sampling_rate
):
# 根据类型处理时间戳
    if timestamp_type == "frame":
        # 帧编号模式
        if hop_size_samples is None:
            raise ValueError("'hop_size_samples' is required!")
            
        # 将帧编号转换为时间（秒）
        def frame_to_time(frame_idx):
            return (frame_idx * hop_size_samples) / sampling_rate
            
        converted_timestamps = [
            {"start": frame_to_time(ts["start"]), "end": frame_to_time(ts["end"])}
            for ts in timestamps
        ]
            
    else:  # timestamp_type == "time"
        # 时间戳模式
        converted_timestamps = timestamps

    return converted_timestamps

