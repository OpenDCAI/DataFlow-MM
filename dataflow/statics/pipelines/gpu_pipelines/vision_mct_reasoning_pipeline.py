import argparse
from dataflow.utils.storage import FileStorage
from dataflow.serving.local_model_vlm_serving import LocalModelVLMServing_vllm

# 引入原子算子
from dataflow.operators.core_text import MCTSTreeRefiner
from dataflow.operators.core_vision import VisualReasoningGenerator

class VisionMCTSReasoningPipeline:
    def __init__(
        self,
        model_path: str,
        *,
        # Storage
        first_entry_file: str,
        cache_path: str = "./cache_mcts",
        file_name_prefix: str = "mcts_reason",
        # Config
        prompt_type: str = "spatial",
        max_samples_per_file: int = 10000,
        # Keys
        input_question_key: str = "question",
        input_image_key: str = "image",
        input_tree_key: str = "tree",
        output_key: str = "final_reasoning_chains",
        # VLLM
        vllm_max_tokens: int = 1024
    ):
        self.storage = FileStorage(
            first_entry_file_name=first_entry_file,
            cache_path=cache_path,
            file_name_prefix=file_name_prefix,
            cache_type="jsonl"
        )
        
        self.serving = LocalModelVLMServing_vllm(
            hf_model_name_or_path=model_path,
            vllm_tensor_parallel_size=1,
            vllm_temperature=0.7,
            vllm_max_tokens=vllm_max_tokens
        )
        
        self.keys = {
            "q": input_question_key,
            "img": input_image_key,
            "tree": input_tree_key,
            "mcts_chains": "mcts_extracted_chains",
            "final": output_key
        }

        # ================== Operators ==================
        
        # 1. Refiner: MCTS -> Chains
        self.op_mcts_refine = MCTSTreeRefiner(
            max_chains_per_sample=max_samples_per_file
        )
        
        # 2. Generator: VLM -> Chains (Fallback)
        self.op_vlm_gen = VisualReasoningGenerator(
            serving=self.serving,
            prompt_type=prompt_type
        )

    def forward(self):
        print(">>> [Pipeline] Step 1: Extracting Chains from MCTS Trees...")
        self.op_mcts_refine.run(
            self.storage.step(),
            input_tree_key=self.keys["tree"],
            output_key=self.keys["mcts_chains"]
        )
        
        print(">>> [Pipeline] Step 2: Generating Chains via VLM (Fallback)...")
        # 将 mcts_chains 作为 input_existing_chains_key 传入
        # 如果 MCTS 解析成功，则复用；否则调用 VLM 生成
        self.op_vlm_gen.run(
            self.storage.step(),
            input_question_key=self.keys["q"],
            input_image_key=self.keys["img"],
            input_existing_chains_key=self.keys["mcts_chains"],
            output_key=self.keys["final"]
        )
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default="dataflow/example/image_to_text_pipeline/mct_reasoning.jsonl")
    parser.add_argument("--model_path", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--prompt_type", default="spatial")
    args = parser.parse_args()
    
    pipe = VisionMCTSReasoningPipeline(
        model_path=args.model_path,
        first_entry_file=args.input_file,
        prompt_type=args.prompt_type
    )
    pipe.forward()