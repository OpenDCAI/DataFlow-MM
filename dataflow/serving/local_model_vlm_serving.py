import os
import torch
from dataflow import get_logger
from huggingface_hub import snapshot_download
from dataflow.core import VLMServingABC
from dataflow.utils.registry import IO_REGISTRY
from transformers import AutoProcessor
from typing import Optional, Union, List, Dict, Any

class LocalModelVLMServing_vllm(VLMServingABC):
    '''
    A class for generating text using vllm, with model from huggingface or local directory
    '''
    def __init__(self, 
                 hf_model_name_or_path: str = None,
                 hf_cache_dir: str = None,
                 hf_local_dir: str = None,
                 vllm_tensor_parallel_size: int = 1,
                 vllm_temperature: float = 0.7,
                 vllm_top_p: float = 0.9,
                 vllm_max_tokens: int = 1024,
                 vllm_top_k: int = 40,
                 vllm_repetition_penalty: float = 1.0,
                 vllm_seed: int = 42,
                 vllm_max_model_len: int = None,
                 vllm_gpu_memory_utilization: float=0.9,
                 ):

        self.load_model(
            hf_model_name_or_path=hf_model_name_or_path,
            hf_cache_dir=hf_cache_dir,
            hf_local_dir=hf_local_dir,
            vllm_tensor_parallel_size=vllm_tensor_parallel_size,
            vllm_temperature=vllm_temperature, 
            vllm_top_p=vllm_top_p,
            vllm_max_tokens=vllm_max_tokens,
            vllm_top_k=vllm_top_k,
            vllm_repetition_penalty=vllm_repetition_penalty,
            vllm_seed=vllm_seed,
            vllm_max_model_len=vllm_max_model_len,
            vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
        )

    def load_model(self, 
                 hf_model_name_or_path: str = None,
                 hf_cache_dir: str = None,
                 hf_local_dir: str = None,
                 vllm_tensor_parallel_size: int = 1,
                 vllm_temperature: float = 0.7,
                 vllm_top_p: float = 0.9,
                 vllm_max_tokens: int = 1024,
                 vllm_top_k: int = 40,
                 vllm_repetition_penalty: float = 1.0,
                 vllm_seed: int = 42,
                 vllm_max_model_len: int = None,
                 vllm_gpu_memory_utilization: float=0.9,
                 ):
        self.logger = get_logger()
        if hf_model_name_or_path is None:
            raise ValueError("hf_model_name_or_path is required") 
        elif os.path.exists(hf_model_name_or_path):
            self.logger.info(f"Using local model path: {hf_model_name_or_path}")
            self.real_model_path = hf_model_name_or_path
        else:
            self.logger.info(f"Downloading model from HuggingFace: {hf_model_name_or_path}")
            self.real_model_path = snapshot_download(
                repo_id=hf_model_name_or_path,
                cache_dir=hf_cache_dir,
                local_dir=hf_local_dir,
            )
        # get the model name from the real_model_path
        self.model_name = os.path.basename(self.real_model_path)
        self.processor = AutoProcessor.from_pretrained(self.real_model_path, cache_dir=hf_cache_dir)
        print(f"Model name: {self.model_name}")
        print(IO_REGISTRY)
        self.IO = IO_REGISTRY.get(self.model_name)(self.processor)
        print(f"IO: {self.IO}")


        # Import vLLM and set up the environment for multiprocessing
        # vLLM requires the multiprocessing method to be set to spawn
        try:
            from vllm import LLM,SamplingParams
        except:
            raise ImportError("please install vllm first like 'pip install open-dataflow[vllm]'")
        # Set the environment variable for vllm to use spawn method for multiprocessing
        # See https://docs.vllm.ai/en/v0.7.1/design/multiprocessing.html 
        os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = "spawn"
        
        self.sampling_params = SamplingParams(
            temperature=vllm_temperature,
            top_p=vllm_top_p,
            max_tokens=vllm_max_tokens,
            top_k=vllm_top_k,
            repetition_penalty=vllm_repetition_penalty,
            seed=vllm_seed
        )
        
        self.llm = LLM(
            model=self.real_model_path,
            tensor_parallel_size=vllm_tensor_parallel_size,
            max_model_len=vllm_max_model_len,
            gpu_memory_utilization=vllm_gpu_memory_utilization,
        )
        self.logger.success(f"Model loaded from {self.real_model_path} by vLLM backend")

    def generate_from_input(self, 
                            # TODO: 这里为了跑通，防止自己被误导就写成list[str]了，后面可以改成list of list of tokens       
                            user_inputs: list[str], 
                            system_prompt: str = "You are a helpful assistant",
                            image_inputs: list[list] = None,
                            video_inputs: list[list] = None,
                            audio_inputs: list[list] = None,
                        ) -> list[str]:
        print('user_inputs_len', len(user_inputs))
        if image_inputs is not None:
            print('image_inputs_len', len(image_inputs))
        if video_inputs is not None:
            print('video_inputs_len', len(video_inputs))
        if audio_inputs is not None:
            print('audio_inputs_len', len(audio_inputs))

        # 检查是否为纯文本模式
        if image_inputs is None and video_inputs is None and audio_inputs is None:
            # 纯文本 prompt
            full_prompts = [system_prompt + '\n' + question for question in user_inputs]
        else:
            # 多模态 prompt
            full_prompts = []   # 2个pair，每个pair是一个instruction-image pair. 同一条数据对应2个图.
            for i in range(len(user_inputs)):       # len(user_inputs) == 2
                for j in range(max(
                    len(image_inputs[i]) if image_inputs is not None and image_inputs[i] else 0,
                    len(video_inputs[i]) if video_inputs is not None and video_inputs[i] else 0,
                    len(audio_inputs[i]) if audio_inputs is not None and audio_inputs[i] else 0
                )):
                    multimodal_entry = {}
                    if image_inputs is not None and image_inputs[i] is not None:
                        multimodal_entry['image'] = image_inputs[i][j]
                    if video_inputs is not None and video_inputs[i] is not None:
                        multimodal_entry['video'] = video_inputs[i][j]
                    if audio_inputs is not None and audio_inputs[i] is not None:
                        multimodal_entry['audio'] = audio_inputs[i][j]

                full_prompts.append({
                    'prompt': user_inputs[i],
                    'multi_modal_data': multimodal_entry
                })

        responses = self.llm.generate(full_prompts, self.sampling_params)
        return [output.outputs[0].text for output in responses]
        
    def generate_from_input_messages(
        self,
        conversations: list[list[dict]],
        image_list: list[list[str]] = None,
        video_list: list[list[str]] = None,
        audio_list: list[list[str]] = None
    ) -> list[str]:

        messages = self.IO._conversation_to_message(
            conversations,
            image_list,
            video_list,
            audio_list
        ) 
        print(f"messages: {messages}")
        full_prompts = self.IO.build_full_prompts(messages)
        print(f"full_prompts: {full_prompts}")
        # 直接调用LLM生成
        outputs = self.llm.generate(full_prompts, self.sampling_params)
        print(outputs)
        return [output.outputs[0].text for output in outputs]

    def cleanup(self):
        del self.llm
        import gc;
        gc.collect()
        torch.cuda.empty_cache()
    

class LocalModelVLMServing_sglang(VLMServingABC):
    """
    A class for multimodal generation using sglang Engine,
    支持从 HuggingFace 或本地目录加载模型。
    """
    def __init__(
        self,
        hf_model_name_or_path: str = None,
        hf_cache_dir: str = None,
        hf_local_dir: str = None,
        # sglang 分布式参数
        sgl_tp_size: int = 1,         # tensor parallel size
        sgl_dp_size: int = 1,         # data parallel size
        sgl_mem_fraction_static: float = 0.9,
        # 生成控制参数
        sgl_max_new_tokens: int = 1024,
        sgl_stop: Optional[Union[str, List[str]]] = None,
        sgl_stop_token_ids: Optional[List[int]] = None,
        sgl_temperature: float = 1.0,
        sgl_top_p: float = 1.0,
        sgl_top_k: int = -1,
        sgl_min_p: float = 0.0,
        sgl_frequency_penalty: float = 0.0,
        sgl_presence_penalty: float = 0.0,
        sgl_repetition_penalty: float = 1.0,
        sgl_min_new_tokens: int = 0,
        sgl_n: int = 1,
        sgl_json_schema: Optional[str] = None,
        sgl_regex: Optional[str] = None,
        sgl_ebnf: Optional[str] = None,
        sgl_structural_tag: Optional[str] = None,
        sgl_ignore_eos: bool = False,
        sgl_skip_special_tokens: bool = True,
        sgl_spaces_between_special_tokens: bool = True,
        sgl_no_stop_trim: bool = False,
        sgl_custom_params: Optional[Dict[str, Any]] = None,
        sgl_stream_interval: Optional[int] = None,
        sgl_logit_bias: Optional[Dict[str, float]] = None,
        **kwargs
    ):
        self.load_model(
            hf_model_name_or_path=hf_model_name_or_path,
            hf_cache_dir=hf_cache_dir,
            hf_local_dir=hf_local_dir,
            sgl_tp_size=sgl_tp_size,
            sgl_dp_size=sgl_dp_size,
            sgl_mem_fraction_static=sgl_mem_fraction_static,
            sgl_max_new_tokens=sgl_max_new_tokens,
            sgl_stop=sgl_stop,
            sgl_stop_token_ids=sgl_stop_token_ids,
            sgl_temperature=sgl_temperature,
            sgl_top_p=sgl_top_p,
            sgl_top_k=sgl_top_k,
            sgl_min_p=sgl_min_p,
            sgl_frequency_penalty=sgl_frequency_penalty,
            sgl_presence_penalty=sgl_presence_penalty,
            sgl_repetition_penalty=sgl_repetition_penalty,
            sgl_min_new_tokens=sgl_min_new_tokens,
            sgl_n=sgl_n,
            sgl_json_schema=sgl_json_schema,
            sgl_regex=sgl_regex,
            sgl_ebnf=sgl_ebnf,
            sgl_structural_tag=sgl_structural_tag,
            sgl_ignore_eos=sgl_ignore_eos,
            sgl_skip_special_tokens=sgl_skip_special_tokens,
            sgl_spaces_between_special_tokens=sgl_spaces_between_special_tokens,
            sgl_no_stop_trim=sgl_no_stop_trim,
            sgl_custom_params=sgl_custom_params,
            sgl_stream_interval=sgl_stream_interval,
            sgl_logit_bias=sgl_logit_bias,
            **kwargs
        )

    def load_model(
        self,
        hf_model_name_or_path: str,
        hf_cache_dir: str,
        hf_local_dir: str,
        sgl_tp_size: int,
        sgl_dp_size: int,
        sgl_mem_fraction_static:float,
        sgl_max_new_tokens: int,
        sgl_stop: Optional[Union[str, List[str]]] = None,
        sgl_stop_token_ids: Optional[List[int]] = None,
        sgl_temperature: float = 1.0,
        sgl_top_p: float = 1.0,
        sgl_top_k: int = -1,
        sgl_min_p: float = 0.0,
        sgl_frequency_penalty: float = 0.0,
        sgl_presence_penalty: float = 0.0,
        sgl_repetition_penalty: float = 1.0,
        sgl_min_new_tokens: int = 0,
        sgl_n: int = 1,
        sgl_json_schema: Optional[str] = None,
        sgl_regex: Optional[str] = None,
        sgl_ebnf: Optional[str] = None,
        sgl_structural_tag: Optional[str] = None,
        sgl_ignore_eos: bool = False,
        sgl_skip_special_tokens: bool = True,
        sgl_spaces_between_special_tokens: bool = True,
        sgl_no_stop_trim: bool = False,
        sgl_custom_params: Optional[Dict[str, Any]] = None,
        sgl_stream_interval: Optional[int] = None,
        sgl_logit_bias: Optional[Dict[str, float]] = None,
        **kwargs
    ):
        self.logger = get_logger()

        # 1. 确定模型路径
        if hf_model_name_or_path is None:
            raise ValueError("hf_model_name_or_path is required")
        if os.path.exists(hf_model_name_or_path):
            self.logger.info(f"Using local model path: {hf_model_name_or_path}")
            self.real_model_path = hf_model_name_or_path
        else:
            self.logger.info(f"Downloading model from HuggingFace: {hf_model_name_or_path}")
            self.real_model_path = snapshot_download(
                repo_id=hf_model_name_or_path,
                cache_dir=hf_cache_dir,
                local_dir=hf_local_dir
            )

        # 2. 根据模型名选 IO 类
        self.model_name = os.path.basename(self.real_model_path)
        self.processor = AutoProcessor.from_pretrained(self.real_model_path, cache_dir=hf_cache_dir)
        self.IO = IO_REGISTRY.find_best_match_by_model_str(self.model_name)(self.processor)

        # 3. 导入 sglang 并创建 Engine
        try:
            import sglang as sgl
        except ImportError:
            raise ImportError("please install sglang first: pip install open-dataflow[sglang]")

        self.llm = sgl.Engine(
            model_path=self.real_model_path,
            tp_size=sgl_tp_size,
            dp_size=sgl_dp_size,
            mem_fraction_static=sgl_mem_fraction_static,
        )

        # 4. 读取 processor（图像预处理 & prompt 模板）
        self.processor = AutoProcessor.from_pretrained(self.real_model_path, cache_dir=hf_cache_dir)

        # 5. 构造生成参数 dict，并去掉 None
        params = {
            "max_new_tokens": sgl_max_new_tokens,
            "stop": sgl_stop,
            "stop_token_ids": sgl_stop_token_ids,
            "temperature": sgl_temperature,
            "top_p": sgl_top_p,
            "top_k": sgl_top_k,
            "min_p": sgl_min_p,
            "frequency_penalty": sgl_frequency_penalty,
            "presence_penalty": sgl_presence_penalty,
            "repetition_penalty": sgl_repetition_penalty,
            "min_new_tokens": sgl_min_new_tokens,
            "n": sgl_n,
            "json_schema": sgl_json_schema,
            "regex": sgl_regex,
            "ebnf": sgl_ebnf,
            "structural_tag": sgl_structural_tag,
            "ignore_eos": sgl_ignore_eos,
            "skip_special_tokens": sgl_skip_special_tokens,
            "spaces_between_special_tokens": sgl_spaces_between_special_tokens,
            "no_stop_trim": sgl_no_stop_trim,
            "custom_params": sgl_custom_params,
            "stream_interval": sgl_stream_interval,
            "logit_bias": sgl_logit_bias,
            **kwargs
        }
        # 去掉值为 None 的 key
        self.sampling_params = {k: v for k, v in params.items() if v is not None}

        self.logger.success(f"Model loaded from {self.real_model_path} by SGLang VLM backend")
    def generate_from_input(self):
        pass
    
    def generate_from_input_messages(
        self,
        conversations: list[list[dict]],
        image_list: list[list[str]] = None,
        video_list: list[list[str]] = None,
        audio_list: list[list[str]] = None
    ) -> list[str]:
        """
        messages: [
            [ {"type":"text","data":"..."},
              {"type":"image","data":"/path/to/img.jpg"},
              ... ],
            ...
        ]
        """
        messages = self.IO._conversation_to_message(
            conversations,
            image_list,
            video_list,
            audio_list
        ) 
        
        print(f"messages: {messages}")
        full_prompts = self.IO.build_full_prompts(messages)
        print(f"full_prompts: {full_prompts}")
        
        prompt_list = []
        image_data_list = []
        video_data_list = []
        audio_data_list = []
        # See here for the entrypoint format:
        # not support for video and audio yet
        # https://github.com/sgl-project/sglang/blob/42960214994461d93dec2fc3e00383e33c9f0401/python/sglang/srt/entrypoints/engine.py#L138
        for entry in full_prompts:
            prompt_list.append(entry['prompt'])
            image_data_list.append(entry.get('multi_modal_data', {}).get('image', None))
            video_data_list.append(entry.get('multi_modal_data', {}).get('video', None))
            audio_data_list.append(entry.get('multi_modal_data', {}).get('audio', None))

        # 调用 sglang Engine 生成
        outputs = self.llm.generate(
            prompt=prompt_list,
            image_data=image_data_list,
            # video_data=video_data_list,
            sampling_params=self.sampling_params
        )
        
        # 输出取 text 字段
        return [output['text'] for output in outputs]

    def cleanup(self):
        # 结束 engine
        try:
            self.llm.shutdown()
        except:
            pass
        del self.llm
        import gc
        gc.collect()
        torch.cuda.empty_cache()
