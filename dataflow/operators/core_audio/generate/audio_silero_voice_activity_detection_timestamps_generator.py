import torch
import numpy as np
import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger

from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC

from typing import Union, List, Dict, Any, Optional
import warnings

from tqdm import tqdm
import multiprocessing

# 全局变量，用于在每个子进程中存储模型实例
_worker_model_processor = None

def _init_worker(devices, model_init_args):
    global _worker_model_processor
    # 取 worker 的编号（从 1 开始）
    rank = multiprocessing.current_process()._identity[0] - 1
    device = devices[rank % len(devices)]
    cfg = {**model_init_args, "device": device}
    _worker_model_processor = SileroVADModel(**cfg)


@OPERATOR_REGISTRY.register()
class SileroVADGenerator(OperatorABC):
    def __init__(self, 
        repo_or_dir: str = "snakers4/silero-vad", 
        source: str = "github",
        device: Union[str, List[str]] = "cuda",
        num_workers: int = 1,
        threshold: float = 0.5,
        sampling_rate: int = 16000,
        min_speech_duration_s: float = 0.25,
        max_speech_duration_s: float = float('inf'),
        min_silence_duration_s: float = 0.1,
        speech_pad_s: float = 0.03,
        return_seconds: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.logger = get_logger(__name__)
        self.model_init_args = {'repo_or_dir': repo_or_dir, 'source': source}
        self.num_workers = num_workers
        self.is_parallel = self.num_workers > 1
        self.pool = None        # 持久化进程池的占位符
        self.threshold = threshold
        self.sampling_rate = sampling_rate
        self.min_speech_duration_s = min_speech_duration_s
        self.max_speech_duration_s = max_speech_duration_s
        self.min_silence_duration_s = min_silence_duration_s
        self.speech_pad_s = speech_pad_s
        self.return_seconds = return_seconds

        self.vad_params = {
            "threshold": self.threshold,
            "sampling_rate": self.sampling_rate,
            "min_speech_duration_s": self.min_speech_duration_s,
            "max_speech_duration_s": self.max_speech_duration_s,
            "min_silence_duration_s": self.min_silence_duration_s,
            "speech_pad_s": self.speech_pad_s,
            "return_seconds": self.return_seconds,
            "time_resolution": kwargs.pop("time_resolution", 1),
            "neg_threshold": kwargs.pop("neg_threshold", None),
            "min_silence_at_max_speech": kwargs.pop("min_silence_at_max_speech", 0.098),
            "use_max_poss_sil_at_max_speech": kwargs.pop("use_max_poss_sil_at_max_speech", True),
        }

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
            self.logger.info(f"配置为串行模式，设备: {single_device}")
            self.model_processor = SileroVADModel(
                repo_or_dir=repo_or_dir, source=source, device=single_device
            )

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            desc = (
                "SileroVADGenerator 算子基于 Silero VAD 模型，对输入音频进行语音活动检测（VAD），"
                "为每条音频样本输出语音片段的起止时间戳，支持串行和多进程两种运行模式。\n\n"

                "一、__init__ 初始化参数\n"
                "def __init__(\n"
                "    self,\n"
                "    repo_or_dir: str = \"snakers4/silero-vad\",\n"
                "    source: str = \"github\",\n"
                "    device: Union[str, List[str]] = \"cuda\",\n"
                "    num_workers: int = 1,\n"
                "    threshold: float = 0.5,\n"
                "    sampling_rate: int = 16000,\n"
                "    min_speech_duration_s: float = 0.25,\n"
                "    max_speech_duration_s: float = float('inf'),\n"
                "    min_silence_duration_s: float = 0.1,\n"
                "    speech_pad_s: float = 0.03,\n"
                "    return_seconds: bool = False,\n"
                "    time_resolution: int = 1,\n"
                "    neg_threshold: float = None,\n"
                "    min_silence_at_max_speech: float = 0.098,\n"
                "    use_max_poss_sil_at_max_speech: bool = True,\n"
                "):\n\n"
                "- repo_or_dir: str\n"
                "  Silero VAD 模型所在的 repo 或本地目录，传给 torch.hub.load，默认 'snakers4/silero-vad'。\n\n"
                "- source: str\n"
                "  torch.hub.load 的 source 参数，默认 'github'。\n\n"
                "- device: Union[str, List[str]]\n"
                "  推理设备配置：\n"
                "  * 串行模式：可为 'cpu'、'cuda'、'cuda:0' 等单个设备字符串；\n"
                "  * 并行模式：可为设备列表，例如 ['cuda:0', 'cuda:1']，各子进程轮流分配设备。\n\n"
                "- num_workers: int\n"
                "  进程数：\n"
                "  * num_workers <= 1：串行模式，在主进程中加载并运行模型；\n"
                "  * num_workers  > 1：多进程模式，每个子进程通过 _init_worker 持有独立 SileroVADModel 实例。\n\n"
                "- threshold: float\n"
                "  语音概率阈值，VAD 输出概率大于该值视为“语音”，默认 0.5。\n\n"
                "- sampling_rate: int\n"
                "  音频采样率，当前 Silero VAD 支持 8000 和 16000（以及 16000 的整数倍，会自动降采样），默认 16000。\n\n"
                "- min_speech_duration_s: float\n"
                "  最短有效语音段时长（秒），短于该时长的片段会在后处理阶段被丢弃，默认 0.25。\n\n"
                "- max_speech_duration_s: float\n"
                "  最长语音段时长（秒），超过此时长会尝试在静音位置切分；若找不到合适静音，则根据配置进行硬切，默认无限长。\n\n"
                "- min_silence_duration_s: float\n"
                "  判定“一段语音结束”所需的最小静音时长（秒），默认 0.1。\n\n"
                "- speech_pad_s: float\n"
                "  为每段语音首尾补偿的 padding（秒），用于避免切得过“干净”，默认 0.03。\n\n"
                "- return_seconds: bool\n"
                "  若为 True，返回的时间戳为秒；若为 False，则返回采样点索引（sample index），默认 False。\n\n"
                "- time_resolution: int\n"
                "  当 return_seconds=True 时，秒级时间戳保留的小数位数，默认 1。\n\n"
                "- neg_threshold: float\n"
                "  负阈值（从“语音状态”退回“静音状态”使用的阈值）。若为 None，则自动设置为 max(threshold - 0.15, 0.01)。\n\n"
                "- min_silence_at_max_speech: float\n"
                "  当语音段接近 max_speech_duration_s 时，用于寻找切分点的最小静音时长（秒），默认 0.098。\n\n"
                "- use_max_poss_sil_at_max_speech: bool\n"
                "  在过长语音段中，是否使用“最长可用静音”作为切分点（True），否则使用“最后一次静音”作为切分点，默认 True。\n\n"
                "初始化行为：\n"
                "- 创建 logger，保存所有 VAD 配置参数到实例成员；\n"
                "- 若 num_workers <= 1：\n"
                "  * 直接在指定 device 上实例化 SileroVADModel，加载 Silero VAD 模型；\n"
                "- 若 num_workers  > 1：\n"
                "  * 使用 multiprocessing.get_context('spawn') 启动进程池 self.pool；\n"
                "  * 通过 initializer=_init_worker，在每个子进程中创建并缓存 SileroVADModel 实例；\n"
                "  * 主进程不直接加载模型本体，仅负责任务切分与调度。\n\n"

                "二、run 接口参数\n"
                "def run(\n"
                "    self,\n"
                "    storage: DataFlowStorage,\n"
                "    input_audio_key: str = \"audio\",\n"
                "    output_answer_key: str = \"timestamps\",\n"
                "):\n\n"
                "- storage: DataFlowStorage\n"
                "  数据流存储对象。\n\n"
                "- input_audio_key: str = \"audio\"\n"
                "  DataFrame 中音频路径所在列名；每行可以是字符串路径，或仅含一个路径的 List。\n\n"
                "- output_answer_key: str = \"timestamps\"\n"
                "  本算子运行后，写回 VAD 结果的列名；每行会是该音频对应的语音片段列表 List[Dict]。\n\n"

                "三、运行行为\n"
                "1）参数检查：\n"
                "   - 若 output_answer_key 为 None，直接抛出 ValueError。\n\n"
                "2）构造 vad_params：\n"
                "   - 将 __init__ 中配置的 threshold、sampling_rate 等参数打包为字典 vad_params，\n"
                "     并设置 window_size_samples=512（内部会根据采样率覆盖为 512 或 256）。\n\n"
                "3）从 storage 中读取 DataFrame：\n"
                "   - 通过 storage.read('dataframe') 读取上游写入的数据；\n"
                "   - 记录当前行数用于日志；\n"
                "   - 使用 dataframe.get(input_audio_key, pd.Series([])).tolist() 获取音频路径列表 audio_paths。\n\n"
                "4）根据模式选择处理逻辑：\n"
                "   - 串行模式（num_workers <= 1）：\n"
                "     * 调用 _serial_process(audio_paths, vad_params)，依次对每个音频运行 VAD：\n"
                "       - 若 audio_path 为 list，则取第一个元素作为真实路径；\n"
                "       - 调用 self.model_processor.process_audio_file(audio_path, **vad_params)：\n"
                "         · read_audio 读取音频并转换为单通道张量；\n"
                "         · get_speech_timestamps 完整执行 Silero VAD 推理与后处理，返回 speeches 列表；\n"
                "       - 收集每条音频的结果到 results 列表。 \n\n"
                "   - 并行模式（num_workers > 1）：\n"
                "     * 调用 _parallel_process(audio_paths, vad_params)：\n"
                "       - 使用 np.array_split 将 audio_paths 按 num_workers 均分为多个 chunk；\n"
                "       - 为每个 chunk 构造 payload：{'audio_paths_chunk': ..., 'vad_params': vad_params}；\n"
                "       - 利用 self.pool.imap(_parallel_worker, worker_payloads) 将任务分发到各子进程；\n"
                "       - _parallel_worker 在子进程中使用全局 _worker_model_processor（已由 _init_worker 初始化）\n"
                "         逐条调用 SileroVADModel.process_audio_file，返回该 chunk 的结果列表；\n"
                "       - 将所有子结果展平为与原 audio_paths 一一对应的 timestamps_list。\n\n"
                "5）写回结果：\n"
                "   - 在原 DataFrame 上新增/覆盖列 output_answer_key，将 timestamps_list 逐行填入；\n"
                "   - 通过 storage.write(dataframe) 写回到 DataFlowStorage；\n"
                "   - 返回 output_answer_key 作为本算子的输出键名。\n\n"
                "6）close 资源释放：\n"
                "   - 在 close() 中，如 is_parallel=True，则调用 self.pool.close() 和 self.pool.join() 关闭进程池。\n\n"

                "四、输出结果\n"
                "run 执行结束后：\n"
                "- storage 中键为 'dataframe' 的 DataFrame 被覆盖式写回，新增一列 output_answer_key（默认 'timestamps'）；\n"
                "- 该列的每一行是一个语音片段列表 speeches: List[Dict]，其中每个字典通常包含：\n"
                "  * 'start': 语音段起点，单位为“秒”或“采样点索引”（由 return_seconds 决定）；\n"
                "  * 'end'  : 语音段终点，同样以秒或采样点表示；\n"
                "  经过内部后处理，语音段之间的静音会被适当合并与扩展，以获得更自然的切分边界。\n\n"
                "本算子可用作 VAD 预处理模块，为后续 ASR、分段转写、静音剪辑等任务提供精确的语音片段边界。\n"
            )
        else:
            desc = (
                "SileroVADGenerator applies Silero VAD to input audio in order to perform voice activity\n"
                "detection (VAD). For each audio sample it produces a list of speech segments with start/end\n"
                "timestamps, and supports both single-process and multi-process execution.\n\n"

                "1. __init__ parameters\n"
                "def __init__(\n"
                "    self,\n"
                "    repo_or_dir: str = \"snakers4/silero-vad\",\n"
                "    source: str = \"github\",\n"
                "    device: Union[str, List[str]] = \"cuda\",\n"
                "    num_workers: int = 1,\n"
                "    threshold: float = 0.5,\n"
                "    sampling_rate: int = 16000,\n"
                "    min_speech_duration_s: float = 0.25,\n"
                "    max_speech_duration_s: float = float('inf'),\n"
                "    min_silence_duration_s: float = 0.1,\n"
                "    speech_pad_s: float = 0.03,\n"
                "    return_seconds: bool = False,\n"
                "    time_resolution: int = 1,\n"
                "    neg_threshold: float = None,\n"
                "    min_silence_at_max_speech: float = 0.098,\n"
                "    use_max_poss_sil_at_max_speech: bool = True,\n"
                "):\n\n"
                "- repo_or_dir: str\n"
                "  Repo or local directory passed to torch.hub.load for Silero VAD (default: 'snakers4/silero-vad').\n\n"
                "- source: str\n"
                "  The 'source' argument to torch.hub.load (default: 'github').\n\n"
                "- device: Union[str, List[str]]\n"
                "  Inference device configuration:\n"
                "  * Serial mode: a single device string, e.g. 'cpu', 'cuda', 'cuda:0';\n"
                "  * Parallel mode: a list of devices, e.g. ['cuda:0', 'cuda:1'], distributed across workers.\n\n"
                "- num_workers: int\n"
                "  Number of worker processes:\n"
                "  * num_workers <= 1: serial mode; a single SileroVADModel is created in the main process;\n"
                "  * num_workers  > 1: multi-process mode; each worker has its own SileroVADModel.\n\n"
                "- threshold: float\n"
                "  Speech probability threshold; VAD probabilities above this value are considered SPEECH (default 0.5).\n\n"
                "- sampling_rate: int\n"
                "  Audio sampling rate. Silero VAD supports 8000 and 16000 (and multiples of 16000 that are\n"
                "  downsampled internally). Default is 16000.\n\n"
                "- min_speech_duration_s: float\n"
                "  Minimum duration (in seconds) of a valid speech segment; shorter segments are discarded (default 0.25).\n\n"
                "- max_speech_duration_s: float\n"
                "  Maximum duration (in seconds) of a speech segment. Longer segments are split at suitable\n"
                "  silences or, if no silence is found, split forcibly around this limit (default: infinity).\n\n"
                "- min_silence_duration_s: float\n"
                "  Minimum silence duration (in seconds) required to finalize a speech segment (default 0.1).\n\n"
                "- speech_pad_s: float\n"
                "  Padding (in seconds) added to both sides of each final speech segment to avoid overly tight cuts (default 0.03).\n\n"
                "- return_seconds: bool\n"
                "  If True, timestamps are returned in seconds; if False, in sample indices (default False).\n\n"
                "- time_resolution: int\n"
                "  Number of decimal places to keep when return_seconds=True (default 1).\n\n"
                "- neg_threshold: float\n"
                "  Negative/exit threshold for switching from SPEECH back to NON-SPEECH. If None, it is set to\n"
                "  max(threshold - 0.15, 0.01).\n\n"
                "- min_silence_at_max_speech: float\n"
                "  Minimal silence duration (in seconds) used as a candidate cut point when max_speech_duration_s\n"
                "  is reached (default 0.098).\n\n"
                "- use_max_poss_sil_at_max_speech: bool\n"
                "  If True, when splitting an overly long segment, choose the longest possible silence as the\n"
                "  split point; if False, use the last detected silence (default True).\n\n"
                "Initialization behavior:\n"
                "- A logger is created and all VAD configuration parameters are stored.\n"
                "- If num_workers <= 1:\n"
                "  * A single SileroVADModel is instantiated on the specified device and the model is loaded.\n"
                "- If num_workers  > 1:\n"
                "  * A multiprocessing.Pool is created using the 'spawn' context;\n"
                "  * _init_worker initializes a SileroVADModel instance for each worker and stores it in a\n"
                "    process-local global variable _worker_model_processor;\n"
                "  * The main process does not load the model directly; it only splits and dispatches tasks.\n\n"

                "2. run interface\n"
                "def run(\n"
                "    self,\n"
                "    storage: DataFlowStorage,\n"
                "    input_audio_key: str = \"audio\",\n"
                "    output_answer_key: str = \"timestamps\",\n"
                "):\n\n"
                "- storage: DataFlowStorage\n"
                "  DataFlow storage object. \n\n"
                "- input_audio_key: str = \"audio\"\n"
                "  Column name containing audio paths. Each row may be a string path or a single-element list.\n\n"
                "- output_answer_key: str = \"timestamps\"\n"
                "  Name of the column used to store VAD results after this operator runs. Each row will hold a\n"
                "  List[Dict] of speech segments.\n\n"

                "3. Runtime behavior\n"
                "1) Validate parameters:\n"
                "   - If output_answer_key is None, a ValueError is raised.\n\n"
                "2) Build vad_params:\n"
                "   - Collect all VAD-related arguments from __init__ into a dict (threshold, sampling_rate,\n"
                "     min_speech_duration_s, etc.) and set window_size_samples=512 (internally adjusted to\n"
                "     512 or 256 based on sampling_rate).\n\n"
                "3) Read DataFrame from storage:\n"
                "   - dataframe = storage.read('dataframe');\n"
                "   - Log the number of rows; extract audio_paths via dataframe.get(input_audio_key, ...).tolist().\n\n"
                "4) Select execution path based on num_workers:\n"
                "   - Serial mode (num_workers <= 1):\n"
                "     * Call _serial_process(audio_paths, vad_params):\n"
                "       - For each audio_path (if it is a list, use its first element), call\n"
                "         self.model_processor.process_audio_file(audio_path, **vad_params);\n"
                "       - process_audio_file loads audio using read_audio, moves it to device, and calls\n"
                "         get_speech_timestamps to perform VAD and post-processing, returning a speeches list;\n"
                "       - Collect all per-file results into results.\n\n"
                "   - Parallel mode (num_workers > 1):\n"
                "     * Call _parallel_process(audio_paths, vad_params):\n"
                "       - Use np.array_split to divide audio_paths into chunks by num_workers;\n"
                "       - For each chunk, create a payload {'audio_paths_chunk': ..., 'vad_params': vad_params};\n"
                "       - Dispatch payloads via self.pool.imap(_parallel_worker, worker_payloads);\n"
                "       - In each worker, _parallel_worker iterates over audio_paths_chunk and uses the process-local\n"
                "         _worker_model_processor (SileroVADModel) to call process_audio_file for each audio;\n"
                "       - Collect all nested lists and flatten them into timestamps_list aligned with input rows.\n\n"
                "5) Write results back:\n"
                "   - Attach timestamps_list to the DataFrame under output_answer_key;\n"
                "   - Write the updated DataFrame back to storage via storage.write(dataframe);\n"
                "   - Return output_answer_key as the operator's output key.\n\n"
                "6) Resource cleanup (close):\n"
                "   - In close(), if is_parallel is True, the worker pool is closed and joined.\n\n"

                "4. Output\n"
                "After run completes:\n"
                "- The DataFrame stored in storage is overwritten with an updated version\n"
                "  that contains an additional column output_answer_key (default 'timestamps').\n"
                "- Each row in this column is a List[Dict] of speech segments, where each dict typically has:\n"
                "  * 'start': start position of the speech segment (in seconds or samples, depending on return_seconds);\n"
                "  * 'end'  : end position of the speech segment (in seconds or samples).\n"
                "  After the internal post-processing, segments are padded and adjacent segments are adjusted so\n"
                "  that boundaries fall at more natural positions.\n\n"
                "This operator is well-suited as a VAD preprocessing component, providing speech segment\n"
                "boundaries for downstream tasks such as ASR, segmentation-based transcription, or silence-based\n"
                "audio editing.\n"
            )
        return desc

    def run(
        self,
        storage: DataFlowStorage,
        input_audio_key: str = "audio",
        output_answer_key: str = "timestamps",         
    ):
        if output_answer_key is None:
            raise ValueError("At least one of output_answer_key must be provided.")

        self.logger.info("Running SileroVADGenerator...")
        self.input_audio_key = input_audio_key
        # self.input_conversation_key = input_conversation_key
        self.output_answer_key = output_answer_key
        
        # Load the raw dataframe from the input file
        dataframe = storage.read('dataframe')
        self.logger.info(f"Loading, number of rows: {len(dataframe)}")

        audio_paths = dataframe.get(self.input_audio_key, pd.Series([])).tolist()

        if self.is_parallel:
            timestamps_list = self._parallel_process(audio_paths, vad_params=self.vad_params)
        else:
            timestamps_list = self._serial_process(audio_paths, vad_params=self.vad_params)

        dataframe[self.output_answer_key] = timestamps_list
        storage.write(dataframe)
        return output_answer_key

    def _serial_process(self, audio_paths: List[str], vad_params: Dict[str, Any]) -> List[List[Dict]]:
        """串行处理器：在一个循环中处理所有音频。"""
        self.logger.info("Start serial processing...")
        results = []
        for audio_path in tqdm(audio_paths, unit=" row", desc="SileroVAD"):
            if isinstance(audio_path, list): 
                audio_path = audio_path[0]
            results.append(self.model_processor.process_audio_file(audio_path, **vad_params))
        return results
    
    def _parallel_process(self, audio_paths: List[str], vad_params: Dict[str, Any]) -> List[List[Dict]]:
        """直接使用 self.pool 分发任务，不再创建Pool。"""
        self.logger.info("Start parallel processing...")
        audio_chunks = np.array_split(audio_paths, self.num_workers)
        
        # 只需要准备每个任务的数据负载，不再需要配置信息
        worker_payloads = []
        for i, chunk in enumerate(audio_chunks):
            if len(chunk) > 0:                
                payload = {
                    'audio_paths_chunk': chunk.tolist(),
                    'vad_params': vad_params,
                }
                worker_payloads.append(payload)
        
        # 直接使用已存在的 self.pool
        results_nested = list(tqdm(
            self.pool.imap(_parallel_worker, worker_payloads),
            total=len(worker_payloads),
            desc="SileroVAD parallel processing..."
        ))
        
        return [item for sublist in results_nested for item in sublist]
    
    def close(self):
        if self.is_parallel:
            self.pool.close()
            self.pool.join()
    
def _parallel_worker(payload: Dict[str, Any]) -> List[List[Dict[str, float]]]:
    """子进程工作函数"""
    global _worker_model_processor

    # if _worker_model_processor is None:
    #     config = payload['model_config']
    #     # print(f"Initializing model lazily in worker {os.getpid()} on device {config['device']}...")
    #     _worker_model_processor = SileroVADModel(**config)

    audio_paths_chunk = payload['audio_paths_chunk']
    vad_params = payload['vad_params']
            
    results = []
    # 使用已经存在于子进程中的 _worker_model_processor
    for audio_path in audio_paths_chunk:
        if isinstance(audio_path, list): 
            audio_path = audio_path[0]
        results.append(_worker_model_processor.process_audio_file(audio_path, **vad_params))

    return results

    
class SileroVADModel:
    """
    封装了一个独立的 Silero VAD 模型实例。
    负责加载模型、读取音频，并执行VAD算法。
    """
    def __init__(self, repo_or_dir: str, source: str, device: str):
        self.device = torch.device(device)
        self.load_model(repo_or_dir, source, self.device)

    def load_model(self, repo_or_dir: str, source: str, device: str):
        """加载模型和相关的工具函数"""
        self.model, vad_utils = torch.hub.load(
            repo_or_dir=repo_or_dir,
            model='silero_vad',
            source=source,
            force_reload=False,
            onnx=False,
            trust_repo=True
        )
        (_, _, self.read_audio, _, _) = vad_utils
        self.model.to(device)

    def process_audio_file(self, audio_path: str, **vad_params) -> List[Dict]:
        """
        处理单个音频文件的完整流程。
        """
        audio_tensor = self.read_audio(audio_path).to(self.device)
        return self.get_speech_timestamps(audio=audio_tensor, model=self.model, **vad_params)
    
    @torch.no_grad()
    def get_speech_timestamps(
        self,
        audio: torch.Tensor,
        model,
        **kwargs,
    ):

        """
        This method is used for splitting long audios into speech chunks using silero VAD

        Parameters
        ----------
        audio: torch.Tensor, one dimensional
            One dimensional float torch.Tensor, other types are casted to torch if possible

        model: preloaded .jit/.onnx silero VAD model

        threshold: float (default - 0.5)
            Speech threshold. Silero VAD outputs speech probabilities for each audio chunk, probabilities ABOVE this value are considered as SPEECH.
            It is better to tune this parameter for each dataset separately, but "lazy" 0.5 is pretty good for most datasets.

        sampling_rate: int (default - 16000)
            Currently silero VAD models support 8000 and 16000 (or multiply of 16000) sample rates

        min_speech_duration_ms: int (default - 250 milliseconds)
            Final speech chunks shorter min_speech_duration_ms are thrown out

        max_speech_duration_s: int (default -  inf)
            Maximum duration of speech chunks in seconds
            Chunks longer than max_speech_duration_s will be split at the timestamp of the last silence that lasts more than 100ms (if any), to prevent agressive cutting.
            Otherwise, they will be split aggressively just before max_speech_duration_s.

        min_silence_duration_ms: int (default - 100 milliseconds)
            In the end of each speech chunk wait for min_silence_duration_ms before separating it

        speech_pad_ms: int (default - 30 milliseconds)
            Final speech chunks are padded by speech_pad_ms each side

        return_seconds: bool (default - False)
            whether return timestamps in seconds (default - samples)

        time_resolution: bool (default - 1)
            time resolution of speech coordinates when requested as seconds

        visualize_probs: bool (default - False)
            whether draw prob hist or not

        progress_tracking_callback: Callable[[float], None] (default - None)
            callback function taking progress in percents as an argument

        neg_threshold: float (default = threshold - 0.15)
            Negative threshold (noise or exit threshold). If model's current state is SPEECH, values BELOW this value are considered as NON-SPEECH.

        min_silence_at_max_speech: float (default - 98ms)
            Minimum silence duration in ms which is used to avoid abrupt cuts when max_speech_duration_s is reached

        use_max_poss_sil_at_max_speech: bool (default - True)
            Whether to use the maximum possible silence at max_speech_duration_s or not. If not, the last silence is used.

        window_size_samples: int (default - 512 samples)
            !!! DEPRECATED, DOES NOTHING !!!

        Returns
        ----------
        speeches: list of dicts
            list containing ends and beginnings of speech chunks (samples or seconds based on return_seconds)
        """

        threshold = kwargs.get('threshold', 0.5)
        sampling_rate = kwargs.get('sampling_rate', 16000)
        min_speech_duration_s = kwargs.get('min_speech_duration_s', 0.25)
        max_speech_duration_s = kwargs.get('max_speech_duration_s', float('inf'))
        min_silence_duration_s = kwargs.get('min_silence_duration_s', 0.1)
        speech_pad_s = kwargs.get('speech_pad_s', 0.03)
        return_seconds = kwargs.get('return_seconds', False)
        time_resolution = kwargs.get('time_resolution', 1)
        neg_threshold = kwargs.get('neg_threshold', None)
        min_silence_at_max_speech = kwargs.get('min_silence_at_max_speech', 0.098)
        use_max_poss_sil_at_max_speech = kwargs.get('use_max_poss_sil_at_max_speech', True)

        if not torch.is_tensor(audio):
            try:
                audio = torch.Tensor(audio)
            except:
                raise TypeError("Audio cannot be casted to tensor. Cast it manually")

        if len(audio.shape) > 1:
            for i in range(len(audio.shape)):  # trying to squeeze empty dimensions
                audio = audio.squeeze(0)
            if len(audio.shape) > 1:
                raise ValueError("More than one dimension in audio. Are you trying to process audio with 2 channels?")

        if sampling_rate > 16000 and (sampling_rate % 16000 == 0):
            step = sampling_rate // 16000
            sampling_rate = 16000
            audio = audio[::step]
            warnings.warn('Sampling rate is a multiply of 16000, casting to 16000 manually!')
        else:
            step = 1

        if sampling_rate not in [8000, 16000]:
            raise ValueError("Currently silero VAD models support 8000 and 16000 (or multiply of 16000) sample rates")

        window_size_samples = 512 if sampling_rate == 16000 else 256
        hop_size_samples = int(window_size_samples)

        model.reset_states()
        min_speech_samples = sampling_rate * min_speech_duration_s
        speech_pad_samples = sampling_rate * speech_pad_s
        max_speech_samples = sampling_rate * max_speech_duration_s - window_size_samples - 2 * speech_pad_samples
        min_silence_samples = sampling_rate * min_silence_duration_s
        min_silence_samples_at_max_speech = sampling_rate * min_silence_at_max_speech

        audio_length_samples = len(audio)

        speech_probs = []
        for current_start_sample in range(0, audio_length_samples, hop_size_samples):
            chunk = audio[current_start_sample: current_start_sample + window_size_samples]
            if len(chunk) < window_size_samples:
                chunk = torch.nn.functional.pad(chunk, (0, int(window_size_samples - len(chunk))))
            try:
                # speech_prob = model(chunk, sampling_rate).item()
                speech_prob = model(chunk, sampling_rate)
                speech_prob = speech_prob.item()
            except Exception as e:
                import ipdb; ipdb.set_trace()
            speech_probs.append(speech_prob)
            # caculate progress and seng it to callback function
            # progress = current_start_sample + hop_size_samples
            # if progress > audio_length_samples:
            #     progress = audio_length_samples
            # progress_percent = (progress / audio_length_samples) * 100
            # if progress_tracking_callback:
            #     progress_tracking_callback(progress_percent)

        triggered = False
        speeches = []
        current_speech = {}

        if neg_threshold is None:
            neg_threshold = max(threshold - 0.15, 0.01)
        
        # 当VAD模型检测到音频从“有语音”状态转为“可能是静音”时，它不会立即判断语音段结束。
        # 相反，它会在那个时间点放下 temp_end 这个标记，然后继续观察一小段时间，以确认这到底是一个真正的静音段，
        # 还是仅仅是说话人一个短暂的停顿（比如呼吸、换气）。
        temp_end = 0  # to save potential segment end (and tolerate some silence)
        prev_end = next_start = 0  # to save potential segment limits in case of maximum segment size reached
        possible_ends = []

        for i, speech_prob in enumerate(speech_probs):
            if (speech_prob >= threshold) and temp_end:
                if temp_end != 0:
                    sil_dur = (hop_size_samples * i) - temp_end
                    if sil_dur > min_silence_samples_at_max_speech:
                        possible_ends.append((temp_end, sil_dur))
                    temp_end = 0
                if next_start < prev_end:
                    next_start = hop_size_samples * i

            if (speech_prob >= threshold) and not triggered:
                triggered = True
                current_speech['start'] = hop_size_samples * i
                continue

            # 这段语音太长了，要做切分
            if triggered and (hop_size_samples * i) - current_speech['start'] > max_speech_samples:
                # 这段语音存在可能的切分点(人不是一直在说话)
                if possible_ends:
                    # 当语音段达到最大长度时，是否使用“最长的可能静音”作为分割点，而不是用最后一个静音
                    if use_max_poss_sil_at_max_speech:  
                        prev_end, dur = max(possible_ends, key=lambda x: x[1])  # use the longest possible silence segment in the current speech chunk
                    else:
                        prev_end, dur = possible_ends[-1]   # use the last possible silence segement
                    current_speech['end'] = prev_end
                    speeches.append(current_speech)
                    current_speech = {}
                    next_start = prev_end + dur             # 下一段开始位置 = 上一段结束位置 + 静音长度
                    # 从静音之后开始而不是从下次滑动窗口的位置开始
                    if next_start < prev_end + hop_size_samples * i:  # previously reached silence (< neg_thres) and is still not speech (< thres)
                        #triggered = False
                        current_speech['start'] = next_start
                    else:   # 如果 next_start 已经很靠后了，说明后面可能又是新语音。那就先关掉触发器，等模型再次检测到语音再重新开始
                        triggered = False
                        #current_speech['start'] = next_start
                    # 清空临时变量, 准备处理下一段
                    prev_end = next_start = temp_end = 0
                    possible_ends = []
                else:
                    # 正常滑动窗口
                    current_speech['end'] = hop_size_samples * i
                    speeches.append(current_speech)
                    current_speech = {}
                    prev_end = next_start = temp_end = 0
                    triggered = False
                    possible_ends = []
                    continue
            
            # triggered表示当前正在说话状态
            # 当语音概率低于负阈值时，触发静音状态
            # 语音段还没有超长, 但是出现了疑似静音
            if (speech_prob < neg_threshold) and triggered:
                if not temp_end:    # 第一次进入疑似静音：用当前帧结束位置打时间戳
                    temp_end = hop_size_samples * i
                # if ((hop_size_samples * i) - temp_end) > min_silence_samples_at_max_speech:  # condition to avoid cutting in very short silence
                #     prev_end = temp_end
                # 静音太短了
                # temp_end到当前帧结束位置太短了
                if (hop_size_samples * i) - temp_end < min_silence_samples:
                    continue
                else:
                    current_speech['end'] = temp_end
                    if (current_speech['end'] - current_speech['start']) > min_speech_samples:
                        speeches.append(current_speech)
                    current_speech = {}
                    prev_end = next_start = temp_end = 0
                    triggered = False
                    possible_ends = []
                    continue
        
        # 处理结尾部分
        if current_speech and (audio_length_samples - current_speech['start']) > min_speech_samples:
            current_speech['end'] = audio_length_samples
            speeches.append(current_speech)

        for i, speech in enumerate(speeches):
            if i == 0:  # 往前多切一点点, 人实际开始说话可能比这个点早几毫秒(气息、弱辅音等被 VAD 漏掉)
                speech['start'] = int(max(0, speech['start'] - speech_pad_samples))
            if i != len(speeches) - 1:
                silence_duration = speeches[i+1]['start'] - speech['end']
                if silence_duration < 2 * speech_pad_samples:       # 两段语音间的静音太短了, 各扩一半
                    speech['end'] += int(silence_duration // 2)
                    speeches[i+1]['start'] = int(max(0, speeches[i+1]['start'] - silence_duration // 2))
                else:                                               # 两段语音间的静音太长了, 各括一个padding长度
                    speech['end'] = int(min(audio_length_samples, speech['end'] + speech_pad_samples))
                    speeches[i+1]['start'] = int(max(0, speeches[i+1]['start'] - speech_pad_samples))
            else:
                # 结尾语音, 只扩最后结尾
                speech['end'] = int(min(audio_length_samples, speech['end'] + speech_pad_samples))

        # round(speech_dict['start'] / sampling_rate, time_resolution)保留time_resolution位小数
        if return_seconds:
            audio_length_seconds = audio_length_samples / sampling_rate
            for speech_dict in speeches:
                speech_dict['start'] = max(round(speech_dict['start'] / sampling_rate, time_resolution), 0)
                speech_dict['end'] = min(round(speech_dict['end'] / sampling_rate, time_resolution), audio_length_seconds)
        elif step > 1:
            for speech_dict in speeches:
                speech_dict['start'] *= step
                speech_dict['end'] *= step

        return speeches