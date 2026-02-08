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
class TimestampChunkRowGenerator(OperatorABC):
    def __init__(
        self, 
        dst_folder: str,
        timestamp_unit: Literal["frame", "second"] = "second",  # 手动指定类型
        mode: Literal["merge", "split"] = "split",
        max_audio_duration: float = float('inf'),
        hop_size_samples: int = 512,  # hop_size, 是样本点数量
        sampling_rate: int = 16000,
        num_workers: int = 1,
    ):
        super().__init__()
        self.logger = get_logger(__name__)
        self.num_workers = num_workers
        self.is_parallel = self.num_workers > 1
        self.pool = None        # 持久化进程池的占位符
        
        self.dst_folder = dst_folder
        self.timestamp_unit = timestamp_unit
        self.mode = mode
        self.max_audio_duration = max_audio_duration
        self.hop_size_samples = hop_size_samples
        self.sampling_rate = sampling_rate

        if self.is_parallel:
            ctx = multiprocessing.get_context('spawn')
            self.pool = ctx.Pool(processes=self.num_workers)
            self.logger.info("Worker initialized...")

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "TimestampChunkRowGenerator（按时间戳切分/合并音频并展开为多行）\n"
                "--------------------------------------------------------\n"
                "功能简介：\n"
                "该算子根据输入的时间戳列表（支持帧 frame 或秒 second），从原始音频中裁剪出对应片段，\n"
                "并按 mode 选择“逐段拆分输出”或“在最大时长约束下合并输出”，最终将一个原始音频样本展开为多行，\n"
                "每行对应一个新生成的音频文件（chunk）。支持串行与多进程并行两种运行模式。\n\n"

                "一、__init__ 初始化接口\n"
                "def __init__(\n"
                "    self,\n"
                "    dst_folder: str,\n"
                "    timestamp_unit: Literal[\"frame\", \"second\"] = \"second\",\n"
                "    mode: Literal[\"merge\", \"split\"] = \"split\",\n"
                "    max_audio_duration: float = float('inf'),\n"
                "    hop_size_samples: int = 512,\n"
                "    sampling_rate: int = 16000,\n"
                "    num_workers: int = 1,\n"
                ")\n\n"
                "参数说明：\n"
                "- dst_folder：输出音频（chunk）保存目录。若为空/None，则回退到原始音频所在目录。\n"
                "- timestamp_unit：输入时间戳单位：\n"
                "  * \"frame\"：timestamps 里的 start/end 为帧索引，需要用 hop_size_samples 与 sampling_rate 转成秒；\n"
                "  * \"second\"：timestamps 里的 start/end 已是秒，直接使用。\n"
                "- mode：处理模式：\n"
                "  * \"split\"：每个 timestamp 输出一个 wav（一个 chunk 一行）；\n"
                "  * \"merge\"：按时间顺序将多个片段合并为一个或多个 wav；当累计时长超过 max_audio_duration 时开启新序列。\n"
                "- max_audio_duration：仅在 mode=\"merge\" 时生效，限制单个合并后 chunk 的最大时长（秒）。\n"
                "- hop_size_samples：仅在 timestamp_unit=\"frame\" 时生效，帧转秒公式：t = frame_idx * hop_size_samples / sampling_rate。\n"
                "- sampling_rate：读写音频使用的采样率（Hz）。\n"
                "- num_workers：进程数；>1 时创建 multiprocessing pool 并并行处理多行。\n\n"
                "初始化行为：\n"
                "- 保存参数并创建 logger；\n"
                "- 若 num_workers > 1，则使用 spawn 上下文创建进程池 self.pool；否则串行处理。\n\n"

                "二、run 运行接口\n"
                "def run(\n"
                "    self,\n"
                "    storage: DataFlowStorage,\n"
                "    input_audio_key: str = \"audio\",\n"
                "    input_timestamps_key: str = \"timestamps\",\n"
                ")\n\n"
                "输入说明：\n"
                "- storage：DataFlowStorage，要求其中 key='dataframe' 存在输入 DataFrame。\n"
                "- input_audio_key：音频路径列名；每行可为字符串路径或单元素列表（[path]）。支持本地与 http/https。\n"
                "- input_timestamps_key：时间戳列名；每行通常为 List[Dict]，每个 dict 至少包含 'start' 与 'end'。\n\n"
                "输出说明：\n"
                "run 会生成新的 DataFrame 并写回 storage（覆盖原 dataframe）。输出 DataFrame 中每行对应一个新音频 chunk，包含：\n"
                "- audio: List[str]               新生成 wav 的路径（单元素列表）\n"
                "- original_audio_path: str       原始音频路径\n"
                "- sequence_num: int              序号（同一原始音频下从 1 开始）\n\n"
                "执行流程（概述）：\n"
                "- 校验 timestamp_unit ∈ {\"frame\",\"second\"} 且 mode ∈ {\"merge\",\"split\"}；\n"
                "- 读取 dataframe，并为每行构造参数 (audio_path, dst_folder, timestamps, ...)；\n"
                "- 串行：逐行调用 _process_audio 裁剪并写出 wav，返回多条行记录并汇总；\n"
                "- 并行：将参数按 chunk_list 切分后分发到进程池，子进程批处理并返回行记录；\n"
                "- 将所有行记录汇总为新的 DataFrame 并 storage.write(output_dataframe)。\n"
            )
        else:
            return (
                "TimestampChunkRowGenerator (Generate audio chunks from timestamps and expand rows)\n"
                "-------------------------------------------------------------------------------\n"
                "Overview:\n"
                "This operator takes per-row timestamps (frame-based or second-based), slices the corresponding\n"
                "segments from the source audio, and either (a) outputs one WAV per timestamp (split) or\n"
                "(b) merges consecutive segments into longer sequences under a max duration constraint (merge).\n"
                "It expands one source-audio row into multiple output rows, each representing one generated chunk.\n"
                "Both serial and multiprocessing execution are supported.\n\n"

                "1) __init__ interface\n"
                "def __init__(\n"
                "    self,\n"
                "    dst_folder: str,\n"
                "    timestamp_unit: Literal[\"frame\", \"second\"] = \"second\",\n"
                "    mode: Literal[\"merge\", \"split\"] = \"split\",\n"
                "    max_audio_duration: float = float('inf'),\n"
                "    hop_size_samples: int = 512,\n"
                "    sampling_rate: int = 16000,\n"
                "    num_workers: int = 1,\n"
                ")\n\n"
                "Parameters:\n"
                "- dst_folder: output directory for generated WAV chunks. If empty/None, it falls back to the source audio directory.\n"
                "- timestamp_unit:\n"
                "  * \"frame\": start/end are frame indices; converted to seconds via hop_size_samples and sampling_rate.\n"
                "  * \"second\": start/end are already seconds.\n"
                "- mode:\n"
                "  * \"split\": one timestamp -> one WAV chunk -> one output row.\n"
                "  * \"merge\": merge multiple segments sequentially; start a new sequence when the accumulated duration exceeds max_audio_duration.\n"
                "- max_audio_duration: (merge mode only) maximum duration (seconds) for a merged sequence.\n"
                "- hop_size_samples: (frame mode only) conversion uses t = frame_idx * hop_size_samples / sampling_rate.\n"
                "- sampling_rate: sampling rate (Hz) for audio I/O.\n"
                "- num_workers: number of processes; >1 creates a multiprocessing pool for parallel row processing.\n\n"
                "Initialization behavior:\n"
                "- Stores configs and creates a logger.\n"
                "- If num_workers > 1, creates a multiprocessing pool using the 'spawn' context; otherwise runs in serial mode.\n\n"

                "2) run interface\n"
                "def run(\n"
                "    self,\n"
                "    storage: DataFlowStorage,\n"
                "    input_audio_key: str = \"audio\",\n"
                "    input_timestamps_key: str = \"timestamps\",\n"
                ")\n\n"
                "Inputs:\n"
                "- storage: DataFlowStorage containing an input DataFrame under key='dataframe'.\n"
                "- input_audio_key: column name for audio paths (string or single-element list). Local paths and http/https are supported.\n"
                "- input_timestamps_key: column name for timestamps. Each row is typically List[Dict] with at least 'start' and 'end'.\n\n"
                "Outputs:\n"
                "The operator overwrites the stored DataFrame with a new expanded DataFrame, where each row represents a generated chunk:\n"
                "- audio: List[str]               path to the generated WAV (single-element list)\n"
                "- original_audio_path: str       source audio path\n"
                "- sequence_num: int              sequence index per source audio (starting from 1)\n\n"
                "Execution summary:\n"
                "- Validate timestamp_unit in {\"frame\",\"second\"} and mode in {\"merge\",\"split\"}.\n"
                "- Read the input dataframe and build per-row processing args.\n"
                "- Serial: call _process_audio per row to slice/write WAV(s) and collect output row dicts.\n"
                "- Parallel: split args into chunks and dispatch to worker processes; each worker returns output row dicts.\n"
                "- Combine all outputs into a new DataFrame and write back via storage.write(output_dataframe).\n"
            )


    def run(self,
        storage: DataFlowStorage,
        input_audio_key: str = "audio",
        input_timestamps_key: str = "timestamps",
        # output_conversation_key: str = "conversation",
        # output_conversation_value: str = "",
    ):
        # 参数验证
        if self.timestamp_unit not in ["frame", "second"]:
            raise ValueError(f"timestamp_unit must be 'frame' or 'second'")
        
        if self.mode not in ["merge", "split"]:
            raise ValueError(f"mode must be 'merge' or 'split'")
        
        dataframe = storage.read('dataframe')
        audio_column = dataframe[input_audio_key]
        timestamps_column = dataframe[input_timestamps_key]
        output_dataframe = []

        args_iter = [
            # (audio_path, self.dst_folder, timestamps, self.timestamp_unit, self.mode,
            #  self.max_audio_duration, self.hop_size_samples, self.sampling_rate, output_conversation_key, output_conversation_value)
            (audio_path, self.dst_folder, timestamps, self.timestamp_unit, self.mode,
             self.max_audio_duration, self.hop_size_samples, self.sampling_rate)
            for audio_path, timestamps in zip(audio_column, timestamps_column)
        ]

        if self.is_parallel:
            args_chunks = chunk_list(args_iter, self.num_workers)
            worker = partial(TimestampChunkRowGenerator._process_audio_chunk_static)
            for chunk_result in tqdm(self.pool.imap(worker, args_chunks), total=len(args_chunks),
                            desc="Generating Chunks...", unit=" chunk"):
                output_dataframe.extend(chunk_result)

        else:
            for args in tqdm(args_iter, desc="Generating Chunks", unit=" row", total=len(args_iter)):
                ret = TimestampChunkRowGenerator._process_audio(args)
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
        for args in args_chunk:
            results.extend(TimestampChunkRowGenerator._process_audio(args))
        return results

    @staticmethod
    def _process_audio(args):
        audio_path = args[0]
        dst_folder = args[1]
        timestamps = args[2]
        timestamp_unit = args[3]
        mode = args[4]
        max_audio_duration = args[5]
        hop_size_samples = args[6]
        sampling_rate = args[7]
        # output_conversation_key = args[8]
        # output_conversation_value = args[9]

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
                timestamp_unit=timestamp_unit,
                timestamps=timestamps,
                hop_size_samples=hop_size_samples,
                sampling_rate=sr
            )

        audio_name = Path(audio_path).stem
        audio_dir = Path(dst_folder) if dst_folder else Path(audio_path).parent
        audio_dir.mkdir(parents=True, exist_ok=True)

        audio_sequences = []

        if mode == "split":
            # 一个 timestamp 输出一个 wav（一个 sequence）
            sequence_num = 1
            for i, ts in enumerate(converted_timestamps):
                start_time = max(0, ts["start"])
                end_time = min(total_duration, ts["end"])
                if start_time >= end_time:
                    continue

                start_sample = int(start_time * sampling_rate)
                end_sample = int(end_time * sampling_rate)
                segment = audio[start_sample:end_sample]

                audio_sequences.append({
                    "segments": [segment],
                    "timestamps": [timestamps[i]],
                    "duration": end_time - start_time,
                    "sequence_num": sequence_num,
                    "source_audio_path": audio_path
                })
                sequence_num += 1
        elif mode == "merge":
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

        ret = []
        for seq in audio_sequences:
            # 合并当前序列的所有片段
            merged_audio = np.concatenate(seq["segments"])
                        
            # 生成文件名
            output_filename = f"{audio_name}_{seq['sequence_num']}.wav"
                    
            output_path = audio_dir / output_filename
            sf.write(output_path, merged_audio, sampling_rate)
            ret.append(
                {
                    'audio': [str(output_path)],
                    'original_audio_path': str(audio_path),
                    # output_conversation_key: [{"from": "human", "value": output_conversation_value}],
                    'sequence_num': seq['sequence_num'],
                }
            )
        return ret

def convert_to_timestamps(
    timestamp_unit,
    timestamps,
    hop_size_samples,
    sampling_rate
):
# 根据类型处理时间戳
    if timestamp_unit == "frame":
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
            
    else:  # timestamp_unit == "second"
        # 时间戳模式
        converted_timestamps = timestamps

    return converted_timestamps

