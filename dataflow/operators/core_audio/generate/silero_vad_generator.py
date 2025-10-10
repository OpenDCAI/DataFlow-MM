import torch
import numpy as np
import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger

from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC

from typing import Union, Optional, Callable, Text, List, Dict, Any
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
    ):
        super().__init__()
        self.logger = get_logger(__name__)
        self.model_init_args = {'repo_or_dir': repo_or_dir, 'source': source}
        self.num_workers = num_workers
        self.is_parallel = self.num_workers > 1
        self.pool = None        # 持久化进程池的占位符

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

    def run(
        self,
        storage: DataFlowStorage,
        input_audio_key: str = "audio",
        output_answer_key: str = "timestamps",
        threshold: float = 0.5,
        use_min_cut: bool = False,
        sampling_rate: int = 16000,
        min_speech_duration_s: int = 0.25,
        max_speech_duration_s: float = float('inf'),
        min_silence_duration_s: int = 0.1,
        speech_pad_s: int = 0.03,
        return_seconds: bool = False,
        time_resolution: int = 1,
        neg_threshold: float = None,
        window_size_samples: int = 512,
        min_silence_at_max_speech: float = 98,
        use_max_poss_sil_at_max_speech: bool = True            
    ):
        if output_answer_key is None:
            raise ValueError("At least one of output_answer_key must be provided.")
        
        vad_params = {
            "use_min_cut": use_min_cut,
            "threshold": threshold,
            "sampling_rate": sampling_rate,
            "min_speech_duration_s": min_speech_duration_s,
            "max_speech_duration_s": max_speech_duration_s,
            "min_silence_duration_s": min_silence_duration_s,
            "speech_pad_s": speech_pad_s,
            "return_seconds": return_seconds,
            "time_resolution": time_resolution,
            "neg_threshold": neg_threshold,
            "window_size_samples": window_size_samples,
            "min_silence_at_max_speech": min_silence_at_max_speech,
            "use_max_poss_sil_at_max_speech": use_max_poss_sil_at_max_speech,
        }

        self.logger.info("Running SileroVADGenerator...")
        self.input_audio_key = input_audio_key
        # self.input_conversation_key = input_conversation_key
        self.output_answer_key = output_answer_key
        
        # Load the raw dataframe from the input file
        dataframe = storage.read('dataframe')
        self.logger.info(f"Loading, number of rows: {len(dataframe)}")

        audio_paths = dataframe.get(self.input_audio_key, pd.Series([])).tolist()

        if self.is_parallel:
            timestamps_list = self._parallel_process(audio_paths, vad_params=vad_params)
        else:
            timestamps_list = self._serial_process(audio_paths, vad_params=vad_params)

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
                # 【重点】每次分发任务时，都附带上模型配置信息
                # device = self.devices[i % len(self.devices)]
                # model_config = {'device': device, **self.model_init_args}
                
                payload = {
                    'audio_paths_chunk': chunk.tolist(),
                    'vad_params': vad_params,
                    # 'model_config': model_config  # 传入配置
                }
                worker_payloads.append(payload)
        
        # 直接使用已存在的 self.pool
        results_nested = list(tqdm(
            self.pool.imap(_parallel_worker, worker_payloads),
            total=len(worker_payloads),
            desc="SileroVAD parallel processing..."
        ))

        # results_nested = self.pool.imap(_parallel_worker, worker_payloads)
        # self.pool.close()
        # self.pool.join()
        
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
    for audio_path in tqdm(audio_paths_chunk, unit=" row", desc="SileroVAD"):
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

        use_min_cut = kwargs.get('use_min_cut', False)
        threshold = kwargs.get('threshold', 0.5)
        sampling_rate = kwargs.get('sampling_rate', 16000)
        min_speech_duration_s = kwargs.get('min_speech_duration_s', 0.25)
        max_speech_duration_s = kwargs.get('max_speech_duration_s', float('inf'))
        min_silence_duration_s = kwargs.get('min_silence_duration_s', 0.1)
        speech_pad_s = kwargs.get('speech_pad_s', 0.03)
        return_seconds = kwargs.get('return_seconds', False)
        time_resolution = kwargs.get('time_resolution', 1)
        neg_threshold = kwargs.get('neg_threshold', None)
        window_size_samples = kwargs.get('window_size_samples', 512)
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
                    if use_min_cut:     # 使用min-cut算法对过长的音频进行截断
                        start_samp = current_speech['start']
                        end_samp = hop_size_samples * i

                        # 帧序号
                        first_frame = start_samp // hop_size_samples
                        last_frame  = end_samp // hop_size_samples
                        seg_probs = speech_probs[first_frame: last_frame + 1]
                        n = len(seg_probs)

                        if n <= 1:
                            # 兜底：段太短，硬切
                            split_frame_global = i
                        else:
                            # 在“当前段”的后半段里找最小值
                            search_after = n // 2
                            if search_after >= n:       # 后半段可能为空
                                split_frame_global = i  # 兜底
                            else:
                                sub = seg_probs[search_after:]
                                # 找子区间最小值位置
                                off_in_sub, _ = min(enumerate(sub), key=lambda x: x[1])
                                split_frame_global = first_frame + search_after + off_in_sub

                        # 保护：避免切点等于段首或超过当前帧，回退为硬切
                        if split_frame_global <= first_frame or split_frame_global >= i:
                            split_frame_global = i

                        # if search_after >= seg_probs.size(0):
                        #     # 兜底：整段太短，硬切到最后一帧
                        #     split_frame = seg_probs.size(0) - 1
                        # else:
                        #     split_offset = torch.argmin(seg_probs[search_after:]).item()  # 相对子切片
                        #     split_frame  = search_after + split_offset                    # 相对 seg_probs

                        split_samp   = split_frame_global * hop_size_samples     # 全局采样点

                        # 切成两段
                        current_speech['end'] = split_samp
                        speeches.append(current_speech)
                        # 后半段继续检测
                        current_speech = {'start': split_samp}
                        # triggered 保持 True
                        # ----------  局部 torch-min-cut end   ----------
                        prev_end = next_start = temp_end = 0
                        possible_ends = []
                        continue
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
    
    def close(self):
        if self.pool:
            self.pool.close()
            self.pool.join()
            self.logger.info("Worker pool closed")

# 已废弃的
# def run(self, 
    #         storage: DataFlowStorage,
    #         input_audio_key: str = "audio",
    #         # input_conversation_key: str = "conversation",
    #         # 输出的conversation可能是none也可能是conversation，请类型检查
    #         output_answer_key: str = "timestamps",
    #         threshold: float = 0.5,
    #         use_min_cut: bool = False,
    #         sampling_rate: int = 16000,
    #         min_speech_duration_s: int = 0.25,
    #         max_speech_duration_s: float = float('inf'),
    #         min_silence_duration_s: int = 0.1,
    #         speech_pad_s: int = 0.03,
    #         return_seconds: bool = False,
    #         time_resolution: int = 1,
    #         # visualize_probs: bool = False,
    #         progress_tracking_callback: Callable[[float], None] = None,
    #         neg_threshold: float = None,
    #         window_size_samples: int = 512,
    #         min_silence_at_max_speech: float = 98,
    #         use_max_poss_sil_at_max_speech: bool = True
    # ):
    #     if output_answer_key is None:
    #         raise ValueError("At least one of output_answer_key must be provided.")

    #     self.logger.info("Running SileroVADGenerator...")
    #     self.input_audio_key = input_audio_key
    #     # self.input_conversation_key = input_conversation_key
    #     self.output_answer_key = output_answer_key
        
    #     # Load the raw dataframe from the input file
    #     dataframe = storage.read('dataframe')
    #     self.logger.info(f"Loading, number of rows: {len(dataframe)}")

    #     audio_paths = dataframe.get(self.input_audio_key, pd.Series([])).tolist()
    #     timestamps_list = []
    #     for audio_path in tqdm(audio_paths, unit="row", desc="SileroVAD Processing"):
    #         if isinstance(audio_path, list):
    #             audio_path = audio_path[0]
    #         audio = self.read_audio(audio_path).to(self.device)
    #         timestamps = self.get_speech_timestamps(
    #             audio=audio,
    #             model=self.model,
    #             threshold=threshold,
    #             use_min_cut=use_min_cut,
    #             sampling_rate=sampling_rate,
    #             min_speech_duration_s=min_speech_duration_s,
    #             max_speech_duration_s=max_speech_duration_s,
    #             min_silence_duration_s=min_silence_duration_s,
    #             speech_pad_s=speech_pad_s,
    #             return_seconds=return_seconds,
    #             time_resolution=time_resolution,
    #             progress_tracking_callback=progress_tracking_callback,
    #             neg_threshold=neg_threshold,
    #             window_size_samples=window_size_samples,
    #             min_silence_at_max_speech=min_silence_at_max_speech,
    #             use_max_poss_sil_at_max_speech=use_max_poss_sil_at_max_speech
    #         )
    #         timestamps_list.append(timestamps)
    #     dataframe[self.output_answer_key] = timestamps_list
    #     storage.write(dataframe)
    #     return output_answer_key