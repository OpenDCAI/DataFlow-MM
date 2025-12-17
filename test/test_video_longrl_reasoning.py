"""
Long-RL Reasoning QA Generator

This script generates reasoning QA from merged video captions using LLM.
It should be run AFTER test_video_longrl.py has generated the merged captions.

Input: video_longrl_step7.json (merged captions from previous script)
Output: video_longrl_reasoning_step3.json (captions + parsed reasoning QA)
"""

import re
from dataflow.core.Operator import OperatorABC
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage, FileStorage

from dataflow.operators.core_vision import VideoCaptionToQAGenerator
from dataflow.serving import LocalModelVLMServing_vllm
from dataflow.prompts.video import DiyVideoPrompt

# Long-RL reasoning QA generation prompt template
REASONING_QA_PROMPT = (
    "Based on the following captions for a video, generate a challenging multiple-choice question that requires **multiple reasoning steps** and deep understanding to answer. "
    "The question should involve as many logical steps as possible, ensuring that the answer cannot be deduced without careful analysis of the captions. "
    "Provide the question with four options (A, B, C, D), clearly indicating the correct answer, and include detailed reasoning with timestamps.\n\n"
    "The question should be related to Goal and Intention Reasoning.\n"
    "Captions:\n{caption}\n\nOutput format:\n"
    "QUESTION: <Your question>\n"
    "OPTIONS:\n"
    "A. <Option A>\n"
    "B. <Option B>\n"
    "C. <Option C>\n"
    "D. <Option D>\n"
    "ANSWER: <Correct answer (e.g., A, B, C, or D)>\n"
    "REASONS:\n"
    "##### From [start to end]:\n"
    "- <Reason 1>\n"
    "- <Reason 2>\n"
    "- <Reason 3>\n"
    "##### From [start to end]:\n"
    "- <Reason 4>\n"
    "- <Reason 5>\n"
    "##### (Add as many steps as needed, grouping reasons under shared timestamps where applicable)"
)


def parse_reasoning(reasoning_text):
    """
    Parse reasoning text into structured format.
    Based on step5_parse_reasoning_data.py from Long-RL.
    """
    parsed_data = {}

    # Extract QUESTION
    question_match = re.search(r"QUESTION:\s*(.*)", reasoning_text)
    parsed_data["QUESTION"] = question_match.group(1) if question_match else ""

    # Extract OPTIONS
    options_match = re.findall(r"([A-D])\.\s*(.*)", reasoning_text)
    parsed_data["OPTIONS"] = {opt: text for opt, text in options_match}

    # Extract ANSWER
    answer_match = re.search(r"ANSWER:\s*([A-D])", reasoning_text)
    parsed_data["ANSWER"] = answer_match.group(1) if answer_match else ""

    # Extract REASONS
    reasons = {}
    if "##### From [" in reasoning_text:
        reason_blocks = re.split(r"##### From \[.*?\]", reasoning_text)[1:]
        reason_blocks_2 = re.split(r"##### From ", reasoning_text)[1:]
    else:
        reason_blocks = re.split(r"##### From .*?\n", reasoning_text)[1:]
        reason_blocks_2 = re.split(r"##### From ", reasoning_text)[1:]

    for i, block in enumerate(reason_blocks):
        if block and block[0] == ":":
            block = block[1:]
        step_reasons = [line.strip('- ') for line in block.strip().split('\n') if line.startswith('- ')]
        try:
            # 尝试提取时间戳，支持两种格式：
            # 1. ##### From [0 to 10]:
            # 2. ##### From 0 to 10:
            timestamp_raw = reason_blocks_2[i].split(block)[0].strip()
            if "[" in timestamp_raw and "]" in timestamp_raw:
                # 格式1: 有方括号
                timestamp = timestamp_raw.split("[")[1].split("]")[0]
            else:
                # 格式2: 没有方括号，直接提取冒号前的内容
                timestamp = timestamp_raw.rstrip(":").strip()
        except:
            timestamp = ""
        reasons[f"Step {i + 1}"] = {"timestamp": timestamp, "reasons": step_reasons}

    parsed_data["REASONS"] = reasons

    return parsed_data


class LongVideoReasoningGenerator(OperatorABC):
    """
    Reasoning QA generation operator using LLM.
    Takes merged captions as input and generates reasoning QA.
    """
    
    def __init__(
        self,
        # LLM Model parameters (for reasoning QA generation)
        llm_model_name_or_path: str = "Qwen/Qwen2.5-72B-Instruct",
        llm_model_cache_dir: str = "./dataflow_cache",
        llm_tensor_parallel_size: int = 2,
        llm_temperature: float = 0.6,
        llm_top_p: float = 0.9,
        llm_max_tokens: int = 32768,
        llm_max_model_len: int = 32768,
        llm_gpu_memory_utilization: float = 0.9,
    ):
        """
        Initialize the Reasoning QA Generator operator.
        
        Args:
            llm_model_name_or_path: LLM model name or path for reasoning QA generation
            llm_model_cache_dir: Directory to cache LLM model files
            llm_tensor_parallel_size: Tensor parallel size for LLM
            llm_temperature: Sampling temperature for LLM
            llm_top_p: Top-p sampling parameter for LLM
            llm_max_tokens: Maximum number of tokens for LLM to generate
            llm_max_model_len: Maximum LLM context length
            llm_gpu_memory_utilization: GPU memory utilization ratio for LLM
        """
        self.logger = get_logger()
        
        # Initialize LLM serving for reasoning QA generation (pure text mode)
        self.logger.info("Initializing LLM serving for reasoning QA generation...")
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
        
        # Initialize reasoning QA generator with custom prompt
        self.reasoning_qa_generator = VideoCaptionToQAGenerator(
            vlm_serving=self.llm_serving,
            prompt_template=DiyVideoPrompt(REASONING_QA_PROMPT),
            use_video_input=False,  # Use pure text mode for reasoning generation
        )
        self.logger.info("✓ LLM serving initialized")
    
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "该算子使用LLM基于合并的视频字幕生成推理问答。\n\n"
                "输入要求：\n"
                "  - captions: 合并后的视频字幕文本\n"
                "输出：\n"
                "  - reasoning: 生成的推理问答\n"
            )
        elif lang == "en":
            return (
                "This operator generates reasoning QA from merged video captions using LLM.\n\n"
                "Input Requirements:\n"
                "  - captions: Merged video caption text\n"
                "Output:\n"
                "  - reasoning: Generated reasoning QA\n"
            )
        else:
            return "LongVideoReasoningGenerator generates reasoning QA from captions using LLM."

    def run(
        self,
        storage: DataFlowStorage,
    ):
        """
        Execute the reasoning QA generation pipeline.
        
        Args:
            storage: DataFlow storage object
            
        Returns:
            str: Output key name
        """
        self.logger.info("="*60)
        self.logger.info("Running Reasoning QA Generation...")
        self.logger.info("="*60)
        
        # Step 1: Prepare data for reasoning QA generation
        self.logger.info("\n[Step 1/3] Preparing data for reasoning QA generation...")
        df = storage.step().read("dataframe")
        self.logger.info(f"Loaded {len(df)} videos with merged captions")
        
        # Rename "captions" to "caption" for VideoCaptionToQAGenerator
        if "captions" in df.columns:
            df["caption"] = df["captions"]
        
        # Add conversation field required by VideoCaptionToQAGenerator
        if "conversation" not in df.columns:
            df["conversation"] = [[{"from": "human", "value": ""}]] * len(df)
        
        storage.write(df)
        self.logger.info("✓ Data preparation complete")
        
        # Step 2: Generate reasoning QA with LLM
        self.logger.info("\n[Step 2/3] Generating reasoning QA with LLM...")
        self.reasoning_qa_generator.run(
            storage=storage.step(),
            input_conversation_key="conversation",
            output_key="reasoning",
        )
        self.logger.info("✓ Reasoning QA generation complete")
        
        # Step 3: Parse reasoning results
        self.logger.info("\n[Step 3/3] Parsing reasoning results...")
        df = storage.step().read("dataframe")
        
        parsed_results = []
        failed_count = 0
        for idx, row in df.iterrows():
            try:
                if "reasoning" in row and row["reasoning"]:
                    parsed = parse_reasoning(row["reasoning"])
                    parsed_results.append(parsed)
                else:
                    parsed_results.append({})
                    failed_count += 1
            except Exception as e:
                self.logger.warning(f"Failed to parse reasoning for row {idx}: {e}")
                parsed_results.append({})
                failed_count += 1
        
        # Add parsed results to dataframe
        df["parsed_reasoning"] = parsed_results
        
        # Remove temporary fields (keep only original "captions")
        temp_fields = ["caption", "conversation"]
        df = df.drop(columns=[col for col in temp_fields if col in df.columns])
        
        storage.write(df)
        
        self.logger.info(f"✓ Parsing complete")
        self.logger.info(f"  Successfully parsed: {len(parsed_results) - failed_count}/{len(parsed_results)}")
        if failed_count > 0:
            self.logger.warning(f"  Failed to parse: {failed_count}/{len(parsed_results)}")
        
        self.logger.info("="*60)
        self.logger.info("✓ Pipeline complete!")
        self.logger.info("="*60)
        
        return "parsed_reasoning"


if __name__ == "__main__":
    # Test the operator
    from dataflow.utils.storage import FileStorage
    
    # Input: video_longrl_step7.json (output from test_video_longrl.py)
    # Output: video_longrl_step8.json (with reasoning QA added)
    storage = FileStorage(
        first_entry_file_name="./cache/video_longrl_step7.json",
        cache_path="./cache",
        file_name_prefix="video_longrl_reasoning",
        cache_type="json",
    )
    
    generator = LongVideoReasoningGenerator(
        # LLM parameters for reasoning QA generation
        llm_model_name_or_path="/mnt/shared-storage-user/mineru2-shared/zqt/zqt2/models/Qwen/Qwen2.5-72B-Instruct",
        llm_model_cache_dir="./dataflow_cache",
        llm_tensor_parallel_size=2,
        llm_temperature=0.6,
        llm_top_p=0.9,
        llm_max_tokens=32768,
        llm_max_model_len=32768,
        llm_gpu_memory_utilization=0.9,
    )
    
    generator.run(storage=storage)

