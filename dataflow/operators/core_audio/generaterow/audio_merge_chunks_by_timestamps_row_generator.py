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
    def __init__(
        self, 
        dst_folder: str,
        timestamp_type: Literal["frame", "time"] = "time",  # 手动指定类型
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
        self.timestamp_type = timestamp_type
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
            desc = (
                "MergeChunksRowGenerator 算子根据输入的时间戳（支持帧级 frame 或秒级 time），\n"
                "从原始音频中顺序裁剪语音片段，并在不超过最大时长的前提下进行合并，\n"
                "最终为每个原始音频生成若干条新的音频文件（序列 chunk），并将结果展开为多行写回 dataframe。\n\n"

                "一、__init__ 初始化参数\n"
                "def __init__(\n"
                "    self,\n"
                "    dst_folder: str,\n"
                "    timestamp_type: Literal['frame', 'time'] = 'time',\n"
                "    max_audio_duration: float = float('inf'),\n"
                "    hop_size_samples: int = 512,\n"
                "    sampling_rate: int = 16000,\n"
                "    num_workers: int = 1,\n"
                "):\n\n"
                "- dst_folder: str\n"
                "  合并后音频文件的输出目录：\n"
                "  * 若传入一个有效路径，所有新生成的音频都会写入该目录；\n"
                "  * 若传 None 或空字符串，内部逻辑会退回到原始音频所在目录（通过 _process_audio 中的逻辑处理）。\n\n"
                "- timestamp_type: Literal['frame', 'time'] = 'time'\n"
                "  输入时间戳类型：\n"
                "  * 'frame'：时间戳为帧编号（start/end 是帧索引），会使用 hop_size_samples 和 sampling_rate\n"
                "    转换为秒；\n"
                "  * 'time' ：时间戳已为秒单位，直接使用。\n\n"
                "- max_audio_duration: float = float('inf')\n"
                "  单个“合并后音频序列”的最大时长（秒）：\n"
                "  * 当当前序列累计时长 current_duration + 新片段时长 > max_audio_duration 且当前序列非空时，\n"
                "    会先将当前序列收束为一个输出文件，再开启新序列；\n"
                "  * 默认为无限长（不切分，只要有片段就全部合并为一个序列）。\n\n"
                "- hop_size_samples: int = 512\n"
                "  帧移（hop size），单位为样本点数，仅在 timestamp_type='frame' 时用于将帧编号转换为秒：\n"
                "  t = frame_idx * hop_size_samples / sampling_rate。\n\n"
                "- sampling_rate: int = 16000\n"
                "  加载音频和写出新音频文件时使用的采样率（Hz），用于 librosa/soundfile 或内部读写函数。\n\n"
                "- num_workers: int = 1\n"
                "  多进程并行处理的进程数：\n"
                "  * num_workers <= 1：串行模式，不启用多进程；\n"
                "  * num_workers  > 1：使用 multiprocessing 的 'spawn' 上下文创建进程池 self.pool，\n"
                "    通过 _process_audio_chunk_static 在子进程中批量处理多行样本。\n\n"
                "初始化行为：\n"
                "- 创建 logger 并保存传入的所有配置参数；\n"
                "- 若 num_workers > 1，则创建 multiprocessing.Pool，并在日志中输出“Worker initialized...”；\n"
                "- 若 num_workers <= 1，则不创建进程池，所有处理在主进程串行完成。\n\n"

                "二、run 接口参数\n"
                "def run(\n"
                "    self,\n"
                "    storage: DataFlowStorage,\n"
                "    input_audio_key: str = 'audio',\n"
                "    input_timestamps_key: str = 'timestamps',\n"
                "):\n\n"
                "- storage: DataFlowStorage\n"
                "  数据流存储对象。\n\n"
                "- input_audio_key: str = 'audio'\n"
                "  DataFrame 中音频路径所在列名；每个单元格可以是字符串路径，或仅包含一个路径的 list。\n\n"
                "- input_timestamps_key: str = 'timestamps'\n"
                "  DataFrame 中时间戳所在列名；每行通常是一个 List[Dict]，其中每个 dict 至少包含：\n"
                "    * 当 timestamp_type='frame' 时：{'start': 帧索引, 'end': 帧索引}\n"
                "    * 当 timestamp_type='time'  时：{'start': 秒, 'end': 秒}\n\n"

                "三、运行行为\n"
                "1）参数校验：\n"
                "   - 若 self.timestamp_type 不在 ['frame', 'time'] 中，抛出 ValueError。\n\n"
                "2）从 storage 读取输入 DataFrame：\n"
                "   - dataframe = storage.read('dataframe')；\n"
                "   - 取出 audio 列和 timestamps 列，构造迭代参数列表 args_iter：\n"
                "     每一项为 (audio_path, dst_folder, timestamps, timestamp_type,\n"
                "              max_audio_duration, hop_size_samples, sampling_rate)。\n\n"
                "3）根据 is_parallel 选择执行模式：\n"
                "   - 串行模式（num_workers <= 1）：\n"
                "     * 使用 tqdm 对 args_iter 逐行遍历；\n"
                "     * 调用静态方法 _process_audio(args)：\n"
                "       - 加载原始音频（支持本地路径和 HTTP/HTTPS 远程音频，分别通过 _read_audio_local / _read_audio_remote）；\n"
                "       - 根据 timestamp_type 调用 convert_to_timestamps，将帧索引或秒转为统一的秒级区间；\n"
                "       - 按时间顺序遍历每个片段：\n"
                "         · 计算该片段在原音频中的起止时间（裁剪到 [0, total_duration] 范围内）；\n"
                "         · 计算片段时长并累加到当前序列 current_duration；\n"
                "         · 如果 current_duration + 片段时长 > max_audio_duration 且当前序列非空，\n"
                "           则先将当前序列收束为一个输出序列（保存 segments/timestamps/duration/sequence_num），\n"
                "           然后开启新的序列；\n"
                "         · 将该片段对应的音频样本（通过起止 sample 下标裁剪）追加到当前序列的 segments；\n"
                "         · 同时保留原始 timestamps[i] 作为该序列的 timestamps 元素。\n"
                "       - 所有片段处理完后，如果当前序列仍有内容，则再追加为最后一个序列；\n"
                "       - 根据 dst_folder / 原始音频路径决定输出目录，依次将每个序列的 segments 通过 np.concatenate\n"
                "         合并为一个完整的波形 merged_audio，并写出为 WAV 文件：\n"
                "           文件名格式：<原始文件名>_<sequence_num>.wav；\n"
                "       - 为每个输出文件构造一个行记录：\n"
                "           {\n"
                "             'audio': [输出文件绝对路径],\n"
                "             'original_audio_path': 源音频路径,\n"
                "             'conversation': [{'from': 'human', 'value': '<audio>'}],\n"
                "             'sequence_num': 当前序列编号\n"
                "           }\n"
                "       - 返回该原始音频对应的所有行记录 list。\n"
                "     * run 中将这些记录累积到 output_dataframe 列表中。\n\n"
                "   - 并行模式（num_workers > 1）：\n"
                "     * 使用 chunk_list 将 args_iter 分成 num_workers 份 args_chunks，每份交给一个 worker 批处理；\n"
                "     * 通过 functools.partial 将 _process_audio_chunk_static 封装为 worker 函数，\n"
                "       然后使用 self.pool.imap(worker, args_chunks) 并配合 tqdm 展示进度；\n"
                "     * _process_audio_chunk_static 内部会遍历 chunk 中的每个 args，调用 _process_audio，\n"
                "       并将所有返回的行记录展平后返回；\n"
                "     * run 中逐个收集 chunk_result，将其 extend 到 output_dataframe 列表中。\n\n"
                "4）写回结果 DataFrame：\n"
                "   - 将 output_dataframe（list[dict]）转换为新的 pd.DataFrame；\n"
                "   - 通过 storage.write(output_dataframe) 覆盖写回到 DataFlowStorage。\n\n"
                "5）资源释放：\n"
                "   - 在 close() 中，如果已创建进程池 self.pool，则执行 self.pool.close() 和 self.pool.join()，\n"
                "     并输出日志 “Worker pool closed”。\n\n"

                "四、输出结果\n" 
                "run 执行结束后，storage 中的 DataFrame 不再是“每行一个原始音频”，\n"
                "而是“每行一个合并后的音频序列（chunk）”，包含以下字段：\n\n"
                "- audio: List[str]\n"
                "  新生成合并音频文件的路径列表。\n\n"
                "- original_audio_path: str\n"
                "  对应的原始音频文件路径。\n\n"
                "- conversation: List[Dict]\n"
                "  固定结构：[{ 'from': 'human', 'value': '<audio>' }]，便于下游按统一格式处理音频对话输入。\n\n"
                "- sequence_num: int\n"
                "  当前原始音频拆分-合并后生成的序列编号（从 1 开始递增）。\n\n"
                "该算子适用于：在已有 VAD / 对齐时间戳的基础上，对音频进行“裁剪 + 合并”，\n"
                "构造长度适配下游模型或标注流程的音频片段数据集。\n"
            )
        else:
            desc = (
                "MergeChunksRowGenerator takes timestamps (either frame-based or time-based),\n"
                "extracts the corresponding segments from the original audio, and merges them into\n"
                "new audio sequences under a maximum duration constraint. Each resulting sequence\n"
                "is saved as a new audio file and one row in the output dataframe.\n\n"

                "1. __init__ parameters\n"
                "def __init__(\n"
                "    self,\n"
                "    dst_folder: str,\n"
                "    timestamp_type: Literal['frame', 'time'] = 'time',\n"
                "    max_audio_duration: float = float('inf'),\n"
                "    hop_size_samples: int = 512,\n"
                "    sampling_rate: int = 16000,\n"
                "    num_workers: int = 1,\n"
                "):\n\n"
                "- dst_folder: str\n"
                "  Target folder to save merged audio files:\n"
                "  * If a valid path is provided, all new WAV files are saved there;\n"
                "  * If None or empty, the operator falls back to the directory of the source audio\n"
                "    (handled inside _process_audio).\n\n"
                "- timestamp_type: Literal['frame', 'time'] = 'time'\n"
                "  Type of timestamps stored in the dataframe:\n"
                "  * 'frame' – timestamps are frame indices (start/end are frame numbers) and are converted\n"
                "    to seconds using hop_size_samples and sampling_rate;\n"
                "  * 'time'  – timestamps are already in seconds and used as-is.\n\n"
                "- max_audio_duration: float = float('inf')\n"
                "  Maximum duration (in seconds) of a single merged sequence:\n"
                "  * When current_duration + segment_duration exceeds this value and the current sequence\n"
                "    is non-empty, the current sequence is finalized as an output file, and a new sequence\n"
                "    is started;\n"
                "  * Default is infinite (all segments for a given source audio are merged into one sequence\n"
                "    as long as there is at least one segment).\n\n"
                "- hop_size_samples: int = 512\n"
                "  Hop size in samples, used only when timestamp_type='frame'. The conversion rule is:\n"
                "  t = frame_idx * hop_size_samples / sampling_rate.\n\n"
                "- sampling_rate: int = 16000\n"
                "  Sampling rate (Hz) used when loading audio and writing merged WAV files.\n\n"
                "- num_workers: int = 1\n"
                "  Number of processes for parallel execution:\n"
                "  * num_workers <= 1 – serial mode, no multiprocessing;\n"
                "  * num_workers  > 1 – a multiprocessing.Pool is created (spawn context) and used to\n"
                "    process rows in parallel via _process_audio_chunk_static.\n\n"
                "Initialization behavior:\n"
                "- A logger is created and all configuration parameters are stored on the instance;\n"
                "- If num_workers > 1, a process pool is instantiated and a log message “Worker initialized...”\n"
                "  is emitted; otherwise, processing is done entirely in the main process.\n\n"

                "2. run interface\n"
                "def run(\n"
                "    self,\n"
                "    storage: DataFlowStorage,\n"
                "    input_audio_key: str = 'audio',\n"
                "    input_timestamps_key: str = 'timestamps',\n"
                "):\n\n"
                "- storage: DataFlowStorage\n"
                "  DataFlow storage object. \n\n"
                "  that contains audio paths and timestamps.\n\n"
                "- input_audio_key: str = 'audio'\n"
                "  Name of the column containing audio paths. Each cell may be a string path or a single-element\n"
                "  list containing the path.\n\n"
                "- input_timestamps_key: str = 'timestamps'\n"
                "  Name of the column containing timestamps. Each row is typically a List[Dict], where each dict\n"
                "  at least contains:\n"
                "    * For timestamp_type='frame': {'start': frame_idx, 'end': frame_idx}\n"
                "    * For timestamp_type='time' : {'start': seconds, 'end': seconds}\n\n"

                "3. Runtime behavior\n"
                "1) Validate configuration:\n"
                "   - If self.timestamp_type is not 'frame' or 'time', a ValueError is raised.\n\n"
                "2) Read the input DataFrame:\n"
                "   - dataframe = storage.read('dataframe');\n"
                "   - Extract the audio and timestamps columns;\n"
                "   - Build args_iter, a list of tuples\n"
                "       (audio_path, dst_folder, timestamps, timestamp_type,\n"
                "        max_audio_duration, hop_size_samples, sampling_rate)\n"
                "     for each row.\n\n"
                "3) Choose execution path based on is_parallel:\n"
                "   - Serial mode (num_workers <= 1):\n"
                "     * Iterate over args_iter with tqdm;\n"
                "     * For each args, call the static method _process_audio(args):\n"
                "       - Load the original audio (HTTP/HTTPS via _read_audio_remote, local path via _read_audio_local);\n"
                "       - Compute total_duration = len(audio) / sr;\n"
                "       - Call convert_to_timestamps(...) to normalize timestamps:\n"
                "         · For 'frame', convert frame indices to seconds using hop_size_samples/sampling_rate;\n"
                "         · For 'time', reuse timestamps as-is.\n"
                "       - Iterate over converted timestamps in chronological order:\n"
                "         · Clip [start, end] to [0, total_duration]; skip invalid intervals where start >= end;\n"
                "         · Compute segment_duration = end_time - start_time;\n"
                "         · If current_duration + segment_duration > max_audio_duration and current sequence\n"
                "           already has segments, finalize the current sequence:\n"
                "           · Append a dict with segments, original timestamps, duration, sequence_num, and\n"
                "             source_audio_path to audio_sequences;\n"
                "           · Reset current_segments/current_timestamps/current_duration and increment sequence_num.\n"
                "         · Convert times to sample indices, slice the audio to obtain the segment, and append it to\n"
                "           current_segments; store the original timestamps[i] in current_timestamps; update current_duration.\n"
                "       - After the loop, if current_segments is non-empty, finalize the last sequence and append it.\n"
                "       - If audio_sequences is empty, return an empty list (no output rows for this audio).\n"
                "       - Determine audio_dir: dst_folder if provided, otherwise the directory of audio_path.\n"
                "       - For each sequence in audio_sequences:\n"
                "         · Concatenate all segments via np.concatenate to form merged_audio;\n"
                "         · Build an output filename '<source_stem>_<sequence_num>.wav';\n"
                "         · Write merged_audio to disk via soundfile.write(output_path, merged_audio, sampling_rate);\n"
                "         · Create a row dict:\n"
                "             {\n"
                "               'audio': [str(output_path)],\n"
                "               'original_audio_path': str(audio_path),\n"
                "               'conversation': [{'from': 'human', 'value': '<audio>'}],\n"
                "               'sequence_num': sequence_num,\n"
                "             }\n"
                "       - Return the list of row dicts for this source audio.\n"
                "     * In run, extend output_dataframe with all returned rows.\n\n"
                "   - Parallel mode (num_workers > 1):\n"
                "     * Use chunk_list(args_iter, num_workers) to split work into arg chunks;\n"
                "     * Wrap _process_audio_chunk_static with functools.partial as worker and call\n"
                "       self.pool.imap(worker, args_chunks) with tqdm to show progress;\n"
                "     * _process_audio_chunk_static iterates over each args in args_chunk, calls _process_audio(args),\n"
                "       and accumulates all rows into results; this effectively batches multiple rows per worker;\n"
                "     * In run, extend output_dataframe with each chunk_result.\n\n"
                "4) Write back results:\n"
                "   - Convert output_dataframe (a list of dicts) into a new pandas DataFrame;\n"
                "   - Write it back to storage via storage.write(output_dataframe).\n\n"
                "5) Resource cleanup:\n"
                "   - In close(), if a process pool exists, call self.pool.close() and self.pool.join(), then log\n"
                "     “Worker pool closed”.\n\n"

                "4. Output\n"
                "After run completes, the DataFrame stored under storage is replaced by a new one in which\n"
                "each row corresponds to a single merged audio sequence (chunk), containing:\n\n"
                "- audio: List[str]\n"
                "  A list with a single element: the path to the merged WAV file.\n\n"
                "- original_audio_path: str\n"
                "  The original source audio path from which this sequence was extracted.\n\n"
                "- conversation: List[Dict]\n"
                "  A standardized conversation structure: [{'from': 'human', 'value': '<audio>'}], suitable for\n"
                "  downstream components expecting an audio-based “conversation” input.\n\n"
                "- sequence_num: int\n"
                "  The sequence number for this source audio (starting from 1 and incrementing for each new sequence).\n\n"
                "This operator is particularly useful when you already have VAD/alignment timestamps and want to\n"
                "turn them into a dataset of reasonably sized audio chunks, e.g., for training downstream models\n"
                "or for manual annotation.\n"
            )
        return desc


    def run(self,
        storage: DataFlowStorage,
        input_audio_key: str = "audio",
        input_timestamps_key: str = "timestamps",
    ):
        # 参数验证
        if self.timestamp_type not in ["frame", "time"]:
            raise ValueError(f"timestamp_type must be 'frame' or 'time'")
        
        dataframe = storage.read('dataframe')
        audio_column = dataframe[input_audio_key]
        timestamps_column = dataframe[input_timestamps_key]
        output_dataframe = []

        args_iter = [
            (audio_path, self.dst_folder, timestamps, self.timestamp_type,
             self.max_audio_duration, self.hop_size_samples, self.sampling_rate)
            for audio_path, timestamps in zip(audio_column, timestamps_column)
        ]

        if self.is_parallel:
            args_chunks = chunk_list(args_iter, self.num_workers)
            worker = partial(MergeChunksRowGenerator._process_audio_chunk_static)
            for chunk_result in tqdm(self.pool.imap(worker, args_chunks), total=len(args_chunks),
                            desc="Merging Chunks", unit=" chunk"):
                output_dataframe.extend(chunk_result)

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

