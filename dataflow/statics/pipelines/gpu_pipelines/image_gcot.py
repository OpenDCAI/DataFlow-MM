#!/usr/bin/env python3
"""
Image GCoT Pipeline - Generate Grounded Chain-of-Thought for VQA tasks
"""

import argparse
from dataflow.utils.storage import FileStorage
from dataflow.serving.local_model_vlm_serving import LocalModelVLMServing_vllm
from dataflow.operators.core_vision.generate.image_gcot_generator import ImageGCoTGenerate


class ImageGCoTPipeline:
    """Image + Question + Answer → Grounded Chain-of-Thought"""
    
    def __init__(
        self,
        qwen_model_path: str,
        ovis_model_path: str,
        *,
        first_entry_file: str,
        cache_path: str = "./cache_gcot",
        file_name_prefix: str = "dataflow_cache_step",
        cache_type: str = "jsonl",
        hf_cache_dir: str | None = None,
        vllm_tensor_parallel_size: int = 1,
        vllm_max_tokens: int = 512,
        question_key: str = "question",
        answer_key: str = "answer",
        image_key: str = "image",
        output_key: str = "gcot",
        save_intermediate: bool = True,
    ):
        """
        Initialize GCoT pipeline.
        
        Args:
            qwen_model_path: Path to Qwen model (for CoT generation)
            ovis_model_path: Path to Ovis model (for bbox detection)
            first_entry_file: Input data file (json/jsonl)
            cache_path: Output cache directory
            file_name_prefix: Prefix for output files
            cache_type: Output format ('json' or 'jsonl')
            hf_cache_dir: HuggingFace cache directory
            vllm_tensor_parallel_size: Number of GPUs for vLLM
            vllm_max_tokens: Max tokens to generate
            question_key: Field name for questions
            answer_key: Field name for answers
            image_key: Field name for image paths
            output_key: Field name for GCoT output
            save_intermediate: Whether to save debug info
        """
        
        # 1. Storage
        self.storage = FileStorage(
            first_entry_file_name=first_entry_file,
            cache_path=cache_path,
            file_name_prefix=file_name_prefix,
            cache_type=cache_type,
        )
        
        # 2. Serving (Qwen for CoT generation)
        self.qwen_serving = LocalModelVLMServing_vllm(
            hf_model_name_or_path=qwen_model_path,
            hf_cache_dir=hf_cache_dir,
            vllm_tensor_parallel_size=vllm_tensor_parallel_size,
            vllm_temperature=0.7,
            vllm_top_p=0.9,
            vllm_max_tokens=vllm_max_tokens,
        )
        
        # 3. Operator (GCoT Generator)
        self.gcot_generator = ImageGCoTGenerate(
            llm_serving=self.qwen_serving,
            model_path=ovis_model_path,
            device="cuda"
        )
        
        # Parameters
        self.question_key = question_key
        self.answer_key = answer_key
        self.image_key = image_key
        self.output_key = output_key
        self.save_intermediate = save_intermediate
    
    def _unload_qwen(self):
        """Callback to unload Qwen model and free GPU memory"""
        if self.qwen_serving is not None:
            self.qwen_serving.cleanup()
            del self.qwen_serving
            self.qwen_serving = None
    
    def forward(self):
        """Run GCoT generation pipeline"""
        self.gcot_generator.run(
            storage=self.storage.step(),
            input_question_key=self.question_key,
            input_answer_key=self.answer_key,
            input_image_key=self.image_key,
            output_key=self.output_key,
            save_intermediate=self.save_intermediate,
            qwen_unload_callback=self._unload_qwen
        )


# ------------------------------ CLI ------------------------------ #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Image GCoT Pipeline - Generate Grounded Chain-of-Thought",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model paths
    parser.add_argument(
        "--qwen_model_path",
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="Path to Qwen model (for CoT generation)"
    )
    parser.add_argument(
        "--ovis_model_path",
        default="AIDC-AI/Ovis2.5-9B",
        help="Path to Ovis model (for bbox detection)"
    )
    
    # Data
    parser.add_argument(
        "--input_file",
        default="./dataflow/example/image_to_text_pipeline/image_qa_result.jsonl",
        help="Input VQA data file (json/jsonl with 'image', 'question', 'answer' fields)"
    )
    parser.add_argument(
        "--cache_path",
        default="./cache_gcot",
        help="Output cache directory"
    )
    parser.add_argument(
        "--file_name_prefix",
        default="dataflow_cache_step",
        help="Output file prefix"
    )
    parser.add_argument(
        "--cache_type",
        default="jsonl",
        choices=["json", "jsonl"],
        help="Output file format"
    )
    
    # Model config
    parser.add_argument(
        "--hf_cache_dir",
        default=None,
        help="HuggingFace cache directory"
    )
    parser.add_argument(
        "--vllm_tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs for vLLM tensor parallelism"
    )
    parser.add_argument(
        "--vllm_max_tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate"
    )
    
    # Field names (for custom datasets)
    parser.add_argument(
        "--question_key",
        default="question",
        help="Field name for questions in input data"
    )
    parser.add_argument(
        "--answer_key",
        default="answer",
        help="Field name for answers in input data"
    )
    parser.add_argument(
        "--image_key",
        default="image",
        help="Field name for image paths in input data"
    )
    parser.add_argument(
        "--output_key",
        default="gcot",
        help="Field name for GCoT output"
    )
    
    # Options
    parser.add_argument(
        "--no_save_intermediate",
        action="store_true",
        help="Don't save intermediate results and debug info"
    )
    
    args = parser.parse_args()
    
    # Create pipeline
    pipe = ImageGCoTPipeline(
        qwen_model_path=args.qwen_model_path,
        ovis_model_path=args.ovis_model_path,
        first_entry_file=args.input_file,
        cache_path=args.cache_path,
        file_name_prefix=args.file_name_prefix,
        cache_type=args.cache_type,
        hf_cache_dir=args.hf_cache_dir,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        vllm_max_tokens=args.vllm_max_tokens,
        question_key=args.question_key,
        answer_key=args.answer_key,
        image_key=args.image_key,
        output_key=args.output_key,
        save_intermediate=not args.no_save_intermediate,
    )
    
    # Run
    pipe.forward()

"""
运行:

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="/data0/happykeyan/workspace/DataFlow-MM/ckpt"

python test/test_image_gcot.py \
    --qwen_model_path /data0/happykeyan/Models/Qwen2.5-VL-7B-Instruct \
    --ovis_model_path AIDC-AI/Ovis2.5-9B \
    --input_file ./data/samples.jsonl \
    --cache_path ./output \
    --cache_type jsonl \
    --vllm_max_tokens 512

数据输入格式：

jsonl

{"image": "path/to/image1.jpg", "question": "What is this?", "answer": "cat"}
{"image": "path/to/image2.jpg", "question": "How many dogs?", "answer": "two"}

json

[
  {"image": "path/to/image1.jpg", "question": "What is this?", "answer": "cat"},
  {"image": "path/to/image2.jpg", "question": "How many dogs?", "answer": "two"}
]

"""