"""
Long-RL Reasoning Data Reformatter

This script reformats parsed reasoning data using LLM to optimize the reasoning process.
It should be run AFTER test_video_longrl_reasoning.py has generated the parsed reasoning.

Input: video_longrl_reasoning_step4.json (parsed reasoning from previous script)
Output: video_longrl_reformat_step2.json (reformatted reasoning ready for training)
"""

import pandas as pd
from dataflow.core.Operator import OperatorABC
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage, FileStorage

from dataflow.operators.core_vision import PromptedVQAGenerator
from dataflow.serving import LocalModelVLMServing_vllm


def _remove_captions(text):
    """
    Remove caption-related references and replace with video-related terms.
    """
    output = text.replace("video captions", "video") \
                 .replace("the captions", "the video") \
                 .replace("The captions", "The video") \
                 .replace("the video's captions", "the video") \
                 .replace("The video's captions", "The video") \
                 .replace("captions", "video frames") \
                 .replace("caption", "video")
    return output


# Prompt template for reformating reasoning
REFORMAT_PROMPT_TEMPLATE = """
You are an advanced AI language model designed to refine logical reasoning while maintaining accuracy. Your task is to optimize the provided reasoning so that it is more natural, logically coherent, and easy to read. Ensure that the refined reasoning:

1. Maintains all key information without introducing errors, while keeping the explanation detailed and avoiding any loss of information.
2. Uses step-by-step formatting, and smooth logic.
3. Removes unnecessary words like "Step" and time references such as (0:00:20–0:00:30).
4. Incorporates a thoughtful and logical thinking process, especially when reasoning involves interpreting or searching within a video. Use phrases like "checking the video," "analyzing the scene," or "searching for specific actions or details in the video" to reflect a step-by-step exploration of the content.

Here is the given input:

"question": "{question}"

"answer": "{answer}"

"reason": "{reason}"

Please return only the optimized reasoning without any additional text or formatting. Ensure the output reflects a clear understanding of the video content and includes logical steps like reviewing or analyzing video details as necessary. The output should be in plain text, directly usable in a program.
"""


class LongVideoReasoningReformatter(OperatorABC):
    """
    Reformats parsed reasoning data using LLM to optimize the reasoning process.
    Takes parsed reasoning as input and generates training-ready format.
    """
    
    def __init__(
        self,
        # LLM Model parameters (for reformatting reasoning)
        llm_model_name_or_path: str = "Qwen/Qwen2.5-72B-Instruct",
        llm_model_cache_dir: str = "./dataflow_cache",
        llm_tensor_parallel_size: int = 2,
        llm_temperature: float = 0.7,
        llm_top_p: float = 0.9,
        llm_max_tokens: int = 2000,
        llm_max_model_len: int = 32768,
        llm_gpu_memory_utilization: float = 0.9,
    ):
        """
        Initialize the Reasoning Reformatter operator.
        
        Args:
            llm_model_name_or_path: LLM model name or path for reformatting
            llm_model_cache_dir: Directory to cache LLM model files
            llm_tensor_parallel_size: Tensor parallel size for LLM
            llm_temperature: Sampling temperature for LLM
            llm_top_p: Top-p sampling parameter for LLM
            llm_max_tokens: Maximum number of tokens for LLM to generate
            llm_max_model_len: Maximum LLM context length
            llm_gpu_memory_utilization: GPU memory utilization ratio for LLM
        """
        self.logger = get_logger()
        
        # Initialize LLM serving for reformatting (pure text mode)
        self.logger.info("Initializing LLM serving for reasoning reformatting...")
        self.llm_serving = LocalModelVLMServing_vllm(
            hf_model_name_or_path=llm_model_name_or_path,
            hf_cache_dir=llm_model_cache_dir,
            vllm_tensor_parallel_size=llm_tensor_parallel_size,
            vllm_temperature=llm_temperature,
            vllm_top_p=llm_top_p,
            vllm_max_tokens=llm_max_tokens,
            vllm_max_model_len=llm_max_model_len,
            vllm_gpu_memory_utilization=llm_gpu_memory_utilization,
        )
        
        # Initialize PromptedVQAGenerator for reformatting
        self.prompted_vqa_generator = PromptedVQAGenerator(
            serving=self.llm_serving,
        )
        self.logger.info("✓ LLM serving initialized")
    
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "该算子使用LLM优化和重新格式化推理问答数据。\n\n"
                "输入要求：\n"
                "  - parsed_reasoning: 解析后的推理数据\n"
                "输出：\n"
                "  - reformatted_data: 优化后的训练格式数据\n"
            )
        elif lang == "en":
            return (
                "This operator uses LLM to optimize and reformat reasoning QA data.\n\n"
                "Input Requirements:\n"
                "  - parsed_reasoning: Parsed reasoning data\n"
                "Output:\n"
                "  - reformatted_data: Optimized training-format data\n"
            )
        else:
            return "LongVideoReasoningReformatter optimizes reasoning data using LLM."

    def run(
        self,
        storage: DataFlowStorage,
    ):
        """
        Execute the reasoning reformatting pipeline.
        
        Args:
            storage: DataFlow storage object
            
        Returns:
            str: Output key name
        """
        self.logger.info("="*60)
        self.logger.info("Running Reasoning Data Reformatting...")
        self.logger.info("="*60)
        
        # Step 1: Load and prepare data
        self.logger.info("\n[Step 1/3] Loading parsed reasoning data...")
        df = storage.step().read("dataframe")
        self.logger.info(f"Loaded {len(df)} videos with parsed reasoning")
        
        # Check if parsed_reasoning exists
        if "parsed_reasoning" not in df.columns:
            raise ValueError("Input dataframe must contain 'parsed_reasoning' column")
        
        # Step 2: Construct prompts for each video
        self.logger.info("\n[Step 2/3] Constructing reformatting prompts...")
        prompts = []
        valid_indices = []
        
        for idx, row in df.iterrows():
            parsed = row.get("parsed_reasoning", {})
            if not parsed or not isinstance(parsed, dict):
                self.logger.warning(f"Skipping row {idx}: Invalid parsed_reasoning")
                continue
            
            # Extract components
            question = parsed.get("QUESTION", "")
            options = parsed.get("OPTIONS", {})
            answer = parsed.get("ANSWER", "")
            reasons = parsed.get("REASONS", {})
            
            if not question or not answer or not reasons:
                self.logger.warning(f"Skipping row {idx}: Missing required fields")
                continue
            
            # Format question with options
            question_text = _remove_captions(question)
            options_text = _remove_captions("\n".join([f"{key}. {value}" for key, value in options.items()]))
            full_question = f"{question_text}\n{options_text}"
            
            # Construct reasons string
            reason_text = "Start thinking.\n" + "\n".join(
                [
                    f"{step}: " + "\n".join(details["reasons"])
                    for step, details in reasons.items()
                ]
            )
            reason_text = _remove_captions(reason_text)
            
            # Create prompt
            prompt = REFORMAT_PROMPT_TEMPLATE.format(
                question=full_question,
                answer=answer,
                reason=reason_text
            )
            
            prompts.append(prompt)
            valid_indices.append(idx)
        
        self.logger.info(f"✓ Prepared {len(prompts)} prompts for reformatting")
        
        if len(prompts) == 0:
            self.logger.error("No valid data to reformat!")
            return None
        
        # Create a new dataframe with only valid rows
        valid_df = df.loc[valid_indices].copy().reset_index(drop=True)
        valid_df["prompt"] = prompts
        
        # Add empty conversation column for PromptedVQAGenerator
        valid_df["conversation"] = [[{"from": "human", "value": p}] for p in prompts]
        
        storage.write(valid_df)
        
        # Step 3: Generate reformatted reasoning using LLM
        self.logger.info("\n[Step 3/3] Generating reformatted reasoning with LLM...")
        self.prompted_vqa_generator.run(
            storage=storage.step(),
            input_conversation_key="conversation",
            output_answer_key="reformatted_reasoning",
        )
        self.logger.info("✓ Reasoning reformatting complete")
        
        self.logger.info("="*60)
        self.logger.info("✓ Pipeline complete!")
        self.logger.info("="*60)
        
        return "reformatted_reasoning"


if __name__ == "__main__":
    # Test the operator
    from dataflow.utils.storage import FileStorage
    
    # Input: video_longrl_reasoning_step4.json (output from test_video_longrl_reasoning.py)
    # Output: video_longrl_reformat_step2.json (reformatted reasoning ready for training)
    storage = FileStorage(
        first_entry_file_name="./cache/video_longrl_reasoning_step4.json",
        cache_path="./cache",
        file_name_prefix="video_longrl_reformat",
        cache_type="json",
    )
    
    reformatter = LongVideoReasoningReformatter(
        # LLM parameters for reformatting reasoning
        llm_model_name_or_path="/mnt/shared-storage-user/mineru2-shared/zqt/zqt2/models/Qwen/Qwen2.5-72B-Instruct",
        llm_model_cache_dir="./dataflow_cache",
        llm_tensor_parallel_size=2,
        llm_temperature=0.7,
        llm_top_p=0.9,
        llm_max_tokens=2000,
        llm_max_model_len=32768,
        llm_gpu_memory_utilization=0.9,
    )
    
    reformatter.run(storage=storage)

