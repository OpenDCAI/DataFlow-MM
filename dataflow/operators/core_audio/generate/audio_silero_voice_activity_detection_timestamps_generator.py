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
            return (
                "SileroVADGenerator（Silero 语音活动检测生成算子）\n"
                "------------------------------------------------\n"
                "功能简介：\n"
                "该算子基于 Silero VAD（Voice Activity Detection）模型，对输入音频做语音活动检测，\n"
                "为每条音频输出语音片段的起止时间戳列表（可选择返回采样点索引或秒）。支持串行与多进程并行两种运行模式：\n"
                "- 串行模式（num_workers<=1）：主进程加载一次模型并逐条处理。\n"
                "- 并行模式（num_workers>1）：使用 spawn 启动多个子进程，每个子进程各自加载一份模型，主进程负责切分音频列表并分发任务。\n\n"

                "一、__init__ 初始化接口\n"
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
                "    **kwargs,\n"
                ")\n\n"
                "参数说明：\n"
                "- repo_or_dir：Silero VAD 模型的 torch.hub repo 或本地目录（用于 torch.hub.load）。\n"
                "- source：torch.hub.load 的 source 参数，通常为 \"github\"。\n"
                "- device：推理设备。\n"
                "  * 串行模式：\"cpu\" / \"cuda\" / \"cuda:0\" 等单个字符串；\n"
                "  * 并行模式：设备列表 [\"cuda:0\", \"cuda:1\"] 等，worker 按 rank 轮询分配设备。\n"
                "- num_workers：worker 数量；>1 启用多进程并行。\n"
                "- threshold：语音概率阈值，>= threshold 视为语音。\n"
                "- sampling_rate：VAD 处理使用的采样率（默认 16000）。\n"
                "- min_speech_duration_s：最短语音段（秒），短于该时长会被丢弃。\n"
                "- max_speech_duration_s：最长语音段（秒），超过该时长会尝试按静音切分（否则进行硬切）。\n"
                "- min_silence_duration_s：判定语音段结束所需的最短静音时长（秒）。\n"
                "- speech_pad_s：对语音段首尾做 padding（秒），用于避免切分过紧。\n"
                "- return_seconds：\n"
                "  * True：输出时间戳单位为秒（浮点数）；\n"
                "  * False：输出为采样点索引（整数）。\n"
                "- kwargs：额外 VAD 参数（会写入 vad_params），包括：\n"
                "  * time_resolution（默认 1）：return_seconds=True 时保留的小数位；\n"
                "  * neg_threshold（默认 None）：从语音退出到静音的负阈值；\n"
                "  * min_silence_at_max_speech（默认 0.098）：接近最长语音段时用于寻找切分点的最小静音；\n"
                "  * use_max_poss_sil_at_max_speech（默认 True）：过长语音段切分时是否选“最长可用静音”。\n\n"
                "初始化行为：\n"
                "- 将所有 VAD 配置整理到 self.vad_params；\n"
                "- 串行模式：在主进程创建 SileroVADModel 并加载模型；\n"
                "- 并行模式：创建 multiprocessing pool，并在每个子进程通过 _init_worker 初始化并缓存 SileroVADModel。\n\n"

                "二、run 运行接口\n"
                "def run(\n"
                "    self,\n"
                "    storage: DataFlowStorage,\n"
                "    input_audio_key: str = \"audio\",\n"
                "    output_answer_key: str = \"timestamps\",\n"
                "):\n\n"
                "输入说明：\n"
                "- storage：DataFlowStorage，要求其中 key='dataframe' 存在一个 DataFrame。\n"
                "- input_audio_key：音频路径列名；每行可为字符串路径或单元素列表（[path]）。\n"
                "- output_answer_key：输出列名，默认 \"timestamps\"。\n\n"
                "输出说明：\n"
                "run 会在 DataFrame 中新增/覆盖 output_answer_key 列，并写回 storage。\n"
                "每行输出为一个语音片段列表 List[Dict]，每个 dict 形如：\n"
                "{\n"
                "  'start': float|int,  # 语音段起点（秒或采样点）\n"
                "  'end':   float|int,  # 语音段终点（秒或采样点）\n"
                "}\n\n"
                "执行流程（概述）：\n"
                "- 从 storage 读取 dataframe，并取出 audio_paths；\n"
                "- 串行模式：逐条调用 SileroVADModel.process_audio_file 生成 speeches；\n"
                "- 并行模式：将 audio_paths 按 worker 切分并分发到子进程，由子进程内缓存模型执行；\n"
                "- 将 timestamps_list 写入 dataframe[output_answer_key] 并 storage.write(dataframe)；\n"
                "- 返回 output_answer_key 作为算子输出键。\n"
            )
        else:
            return (
                "SileroVADGenerator (Silero Voice Activity Detection Generator)\n"
                "------------------------------------------------------------\n"
                "Overview:\n"
                "This operator runs Silero VAD (Voice Activity Detection) on input audio and produces a list\n"
                "of speech segments (start/end timestamps) for each sample. It supports both serial and\n"
                "multi-process execution:\n"
                "- Serial (num_workers<=1): the model is loaded once in the main process and audio files are processed sequentially.\n"
                "- Parallel (num_workers>1): workers are spawned using the 'spawn' context; each worker loads its own model instance,\n"
                "  while the main process splits audio paths and dispatches jobs.\n\n"

                "1) __init__ interface\n"
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
                "    **kwargs,\n"
                ")\n\n"
                "Parameters:\n"
                "- repo_or_dir: torch.hub repo id or local directory for Silero VAD (used by torch.hub.load).\n"
                "- source: torch.hub.load 'source' argument (typically \"github\").\n"
                "- device: inference device(s).\n"
                "  * Serial: a single string such as \"cpu\", \"cuda\", \"cuda:0\".\n"
                "  * Parallel: a list such as [\"cuda:0\", \"cuda:1\"], assigned to workers round-robin by worker rank.\n"
                "- num_workers: number of worker processes; >1 enables multiprocessing.\n"
                "- threshold: speech probability threshold; values >= threshold are considered SPEECH.\n"
                "- sampling_rate: sampling rate used for VAD (default 16000).\n"
                "- min_speech_duration_s: minimum speech segment duration in seconds; shorter segments are discarded.\n"
                "- max_speech_duration_s: maximum speech segment duration in seconds; longer segments are split at silences if possible.\n"
                "- min_silence_duration_s: minimum silence duration in seconds required to close a speech segment.\n"
                "- speech_pad_s: padding (seconds) applied to both ends of each segment.\n"
                "- return_seconds:\n"
                "  * True: return timestamps in seconds (floats);\n"
                "  * False: return sample indices (ints).\n"
                "- kwargs: additional VAD settings stored in vad_params, including:\n"
                "  * time_resolution (default 1): decimal places when return_seconds=True;\n"
                "  * neg_threshold (default None): exit threshold from speech to non-speech;\n"
                "  * min_silence_at_max_speech (default 0.098): minimal silence used to find split points near max length;\n"
                "  * use_max_poss_sil_at_max_speech (default True): whether to pick the longest possible silence for splitting.\n\n"
                "Initialization behavior:\n"
                "- Packs all VAD configs into self.vad_params.\n"
                "- Serial: creates a SileroVADModel instance in the main process and loads the model.\n"
                "- Parallel: creates a multiprocessing pool and initializes a SileroVADModel per worker via _init_worker.\n\n"

                "2) run interface\n"
                "def run(\n"
                "    self,\n"
                "    storage: DataFlowStorage,\n"
                "    input_audio_key: str = \"audio\",\n"
                "    output_answer_key: str = \"timestamps\",\n"
                ")\n\n"
                "Inputs:\n"
                "- storage: DataFlowStorage containing a DataFrame under key='dataframe'.\n"
                "- input_audio_key: column name containing audio paths (string or single-element list).\n"
                "- output_answer_key: output column name (default \"timestamps\").\n\n"
                "Outputs:\n"
                "The operator writes/overwrites dataframe[output_answer_key] and persists it via storage.write.\n"
                "Each row is a List[Dict] of speech segments, typically:\n"
                "{\n"
                "  'start': float|int,  # segment start (seconds or samples)\n"
                "  'end':   float|int,  # segment end (seconds or samples)\n"
                "}\n\n"
                "Execution summary:\n"
                "- Read the dataframe from storage and extract audio paths.\n"
                "- Serial: call SileroVADModel.process_audio_file for each audio.\n"
                "- Parallel: split audio paths into chunks and dispatch them to workers that hold process-local models.\n"
                "- Write the resulting list of segments into output_answer_key and store the updated dataframe.\n"
                "- Return output_answer_key as the operator output key.\n"
            )

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