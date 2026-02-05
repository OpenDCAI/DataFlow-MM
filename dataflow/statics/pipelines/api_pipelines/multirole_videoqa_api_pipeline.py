import os
os.environ["DF_API_KEY"] = "sk-xxxx"

import argparse
import os 
from dataflow.serving import LocalModelVLMServing_vllm
from dataflow.utils.storage import FileStorage
from dataflow.operators.core_vision import MultiroleVideoQAInitialGenerator, MultiroleVideoQAMultiAgentGenerator, MultiroleVideoQAFinalGenerator
from dataflow.serving.api_vlm_serving_openai import APIVLMServing_openai
try:
    import torch
    if 'spawn' not in torch.multiprocessing.get_all_start_methods():
        torch.multiprocessing.set_start_method('spawn', force=True)
except ImportError:
    pass


class MultiRoleVideoQAPipeline():
    def __init__(
        self,
        first_entry_file: str = "/dataflow/example/ads_QA/adsQA.jsonl",
        cache_path: str = "./cache_local",
        file_name_prefix: str = "dataflow_cache_step",
        cache_type: str = "jsonl",
        Meta_key: str = "Meta",
        clips_key: str = "Clips", 
        output_key: str = "QA"
    ):
        self.storage = FileStorage(
            first_entry_file_name=first_entry_file,
            cache_path=cache_path,
            file_name_prefix=file_name_prefix,
            cache_type=cache_type,
        )
        
        os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = "spawn"

        # self.llm_serving = LocalModelVLMServing_vllm(
        #     hf_model_name_or_path=model_path,
        #     hf_cache_dir=hf_cache_dir,
        #     hf_local_dir=download_dir,
        #     vllm_tensor_parallel_size=1, 
        #     vllm_temperature=0.7,
        #     vllm_top_p=0.9,
        #     vllm_max_tokens=6000,
        # )
        self.vlm_serving = APIVLMServing_openai(
            api_url="https://dashscope.aliyuncs.com/compatible-mode/v1", # Any API platform compatible with OpenAI format
            model_name="gpt-4o-mini",
            image_io=None,
            send_request_stream=False,
            max_workers=10,
            timeout=1800
        )

        self.initial_QA_generation = MultiroleVideoQAInitialGenerator(llm_serving = self.vlm_serving)
        self.multiAgent_QA_generation = MultiroleVideoQAMultiAgentGenerator(llm_serving = self.vlm_serving, max_iterations = 3)
        self.final_QA_generation = MultiroleVideoQAFinalGenerator(llm_serving = self.vlm_serving)

        self.input_meta_key = Meta_key
        self.input_clips_key = clips_key
        self.output_key = output_key

    def forward(self):
        init_df = self.initial_QA_generation.run(
            storage = self.storage.step(),
            input_meta_key = self.input_meta_key, 
            input_clips_key = self.input_clips_key, 
            output_key = self.output_key
        )
        middle_df = self.multiAgent_QA_generation.run(
            df = init_df,
            input_meta_key = self.input_meta_key, 
            input_clips_key = self.input_clips_key, 
            output_key = self.output_key
        )
        self.final_QA_generation.run(
            storage = self.storage,
            df = middle_df,
            input_meta_key = self.input_meta_key, 
            input_clips_key = self.input_clips_key, 
            output_key = self.output_key
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch video QA generation with DataFlow (Single GPU)")
    
    parser.add_argument("--images_file", default="./dataflow/example/ads_QA/adsQA.jsonl",
                                 help="Path to the first entry file for DataFlow.")
    parser.add_argument("--cache_path", default="./cache_local",
                                 help="Directory for caching DataFlow steps.")
    parser.add_argument("--file_name_prefix", default="caption",
                                 help="Prefix for cache file names.")
    parser.add_argument("--cache_type", default="jsonl",
                                 help="Type of cache file (e.g., jsonl).")

    args = parser.parse_args()
    
    pipe = MultiRoleVideoQAPipeline(
        first_entry_file=args.images_file,
        cache_path=args.cache_path,
        file_name_prefix=args.file_name_prefix,
        cache_type=args.cache_type,
    )
    pipe.forward()