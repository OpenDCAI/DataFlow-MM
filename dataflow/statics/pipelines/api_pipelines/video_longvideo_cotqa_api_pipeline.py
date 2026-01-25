"""
Long Video CoTQA Pipeline (API Version)

This script integrates three pipelines using API models:
1. Video caption generation 
2. Reasoning QA generation 
3. Reasoning data reformatting 

All operations use API models instead of local models.
"""

import os
import re

# 设置 API Key 环境变量
os.environ["DF_API_KEY"] = "sk-ba117c8c8dbd465c8aac8e3bba7fdec2"

from dataflow.core.Operator import OperatorABC
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage, FileStorage

from dataflow.operators.core_vision import VideoInfoFilter
from dataflow.operators.core_vision import VideoSceneFilter
from dataflow.operators.core_vision import VideoClipFilter
from dataflow.operators.core_vision import VideoClipGenerator
from dataflow.operators.core_vision import VideoMergedCaptionGenerator
from dataflow.operators.core_vision import VideoCaptionToQAGenerator
from dataflow.operators.core_vision import PromptedVQAGenerator

from dataflow.serving.api_vlm_serving_openai import APIVLMServing_openai
from dataflow.prompts.video import DiyVideoPrompt


VIDEO_CAPTION_PROMPT = (
    "Elaborate on the visual and narrative elements of the video in detail. "
)

# long-video reasoning QA generation prompt template
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


def parse_reasoning(reasoning_text):
    """
    Parse reasoning text into structured format.
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


class LongVideoPipelineAPI(OperatorABC):
    """
    Complete long video cotqa processing pipeline using API models.
    Integrates caption generation, reasoning QA generation, and reformatting.
    """
    
    def __init__(self):
        """
        Initialize the long video cotqa pipeline with default API model configurations.
        """
        self.logger = get_logger()
        
        # Initialize video processing operators with default parameters
        self.video_info_filter = VideoInfoFilter(
            backend="opencv",
            ext=False,
        )
        self.video_scene_filter = VideoSceneFilter(
            frame_skip=0,
            start_remove_sec=0.0,
            end_remove_sec=0.0,
            min_seconds=0.0,
            max_seconds=10.0,
            disable_parallel=True,
            use_adaptive_detector=False,
            overlap=False,
            use_fixed_interval=True,
        )
        
        # Initialize clip processing operators
        self.video_clip_filter = VideoClipFilter()
        self.video_clip_generator = VideoClipGenerator(
            video_save_dir="./cache/video_clips",
        )
        
        # Initialize VLM API serving for caption generation
        self.logger.info("Initializing VLM API serving for caption generation...")
        self.vlm_serving = APIVLMServing_openai(
            api_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            key_name_of_api_key="DF_API_KEY",
            model_name="qwen3-vl-8b-instruct",
            image_io=None,
            send_request_stream=False,
            max_workers=10,
            timeout=1800
        )
        
        # Initialize caption generator with prompt template
        self.caption_prompt_template = DiyVideoPrompt(VIDEO_CAPTION_PROMPT)
        
        self.caption_vqa_generator = PromptedVQAGenerator(
            serving=self.vlm_serving,
            system_prompt="You are a helpful assistant.",
            prompt_template=self.caption_prompt_template
        )
        
        # Initialize merged caption generator
        self.video_merged_caption_generator = VideoMergedCaptionGenerator(
            caption_key="caption",
        )
        
        # Initialize LLM API serving for reasoning QA generation
        self.logger.info("Initializing LLM API serving for reasoning generation...")
        self.llm_serving = APIVLMServing_openai(
            api_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            key_name_of_api_key="DF_API_KEY",
            model_name="qwen2.5-72b-instruct",
            image_io=None,
            send_request_stream=False,
            max_workers=10,
            timeout=1800
        )
        
        # Initialize reasoning QA generator with custom prompt
        self.reasoning_qa_generator = VideoCaptionToQAGenerator(
            vlm_serving=self.llm_serving,
            prompt_template=DiyVideoPrompt(REASONING_QA_PROMPT),
            use_video_input=False,
        )
        
        # Initialize separate LLM API serving for reasoning reformatting
        self.logger.info("Initializing LLM API serving for reasoning reformatting...")
        self.reformat_serving = APIVLMServing_openai(
            api_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            key_name_of_api_key="DF_API_KEY",
            model_name="qwen2.5-72b-instruct",
            image_io=None,
            send_request_stream=False,
            max_workers=10,
            timeout=1800
        )
        
        # Initialize PromptedVQAGenerator for reformatting with separate LLM service
        self.prompted_vqa_generator = PromptedVQAGenerator(
            serving=self.reformat_serving,
        )
        
        self.logger.info("✓ All API services initialized")
    
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "完整的长视频CoTQA处理流水线（API 版本），集成了字幕生成、推理问答生成和格式化。\n\n"
                "功能特点：\n"
                "  - 自动提取视频信息并进行场景检测\n"
                "  - 使用 API 模型生成视频字幕\n"
                "  - 基于字幕生成推理问答\n"
                "  - 优化和重新格式化推理数据\n"
            )
        elif lang == "en":
            return (
                "Complete long video cotqa pipeline (API version) with caption generation, "
                "reasoning QA generation, and reformatting.\n\n"
                "Features:\n"
                "  - Automatic video info extraction and scene detection\n"
                "  - API-based video caption generation\n"
                "  - Caption-based reasoning QA generation\n"
                "  - Reasoning data optimization and reformatting\n"
            )
        else:
            return "long video cotqa pipeline using API models."
    
    def _build_reformatting_prompts(self, df):
        """
        Build reformatting prompts for each video with valid parsed reasoning.
        
        Args:
            df: DataFrame containing parsed_reasoning column
            
        Returns:
            tuple: (prompts, full_questions, valid_indices)
        """
        prompts = []
        full_questions = []
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
            full_questions.append(full_question)
            valid_indices.append(idx)
        
        return prompts, full_questions, valid_indices

    def run(
        self,
        storage: DataFlowStorage,
        input_video_key: str = "video",
        input_conversation_key: str = "conversation",
        output_key: str = "caption",
    ):
        """
        Execute the complete long video cotqa pipeline.
        
        Args:
            storage: DataFlow storage object
            input_video_key: Input video path field name (default: 'video')
            input_conversation_key: Input conversation field name (default: 'conversation')
            output_key: Output caption field name (default: 'caption')
            
        Returns:
            str: Output key name
        """
        
        # ============================================================
        # STAGE 1: Video Caption Generation
        # ============================================================
        self.logger.info("\n" + "="*80)
        self.logger.info("STAGE 1: VIDEO CAPTION GENERATION")
        self.logger.info("="*80)
        
        # Step 1: Extract video info
        self.logger.info("\n[Step 1/6] Extracting video info...")
        self.video_info_filter.run(
            storage=storage.step(),
            input_video_key=input_video_key,
            output_key="video_info",
        )

        # Step 2: Detect video scenes
        self.logger.info("\n[Step 2/6] Detecting video scenes...")
        self.video_scene_filter.run(
            storage=storage.step(),
            input_video_key=input_video_key,
            video_info_key="video_info",
            output_key="video_scene",
        )
        
        # Step 3: Generate clip metadata
        self.logger.info("\n[Step 3/6] Generating clip metadata...")
        self.video_clip_filter.run(
            storage=storage.step(),
            input_video_key=input_video_key,
            video_info_key="video_info",
            video_scene_key="video_scene",
            output_key="video_clip",
        )

        # Step 4: Cut and save video clips
        self.logger.info("\n[Step 4/6] Cutting and saving video clips...")
        self.video_clip_generator.run(
            storage=storage.step(),
            video_clips_key="video_clip",
            output_key="video",
        )

        # Step 5: Generate captions for each clip using API
        self.logger.info("\n[Step 5/6] Generating captions for each clip using API...")
        # Call PromptedVQAGenerator to generate captions
        self.caption_vqa_generator.run(
            storage=storage.step(),
            input_image_key="image",
            input_video_key="video",
            input_conversation_key=input_conversation_key,
            output_answer_key=output_key,
        )
        
        # Step 6: Generate merged captions by original video
        self.logger.info("\n[Step 6/6] Generating merged captions by original video...")
        self.video_merged_caption_generator.run(
            storage=storage.step(),
            caption_key=output_key,
        )
        
        # ============================================================
        # STAGE 2: Reasoning QA Generation
        # ============================================================
        self.logger.info("\n" + "="*80)
        self.logger.info("STAGE 2: REASONING QA GENERATION")
        self.logger.info("="*80)
        
        # Step 1: Generate reasoning QA with API LLM
        self.logger.info("\n[Step 1/2] Generating reasoning QA with API LLM...")
        self.reasoning_qa_generator.run(
            storage=storage.step(),
            input_conversation_key="conversation",
            input_caption_key="captions",
            output_key="reasoning",
        )
        
        # Step 2: Parse reasoning results
        self.logger.info("\n[Step 2/2] Parsing reasoning results...")
        df = storage.step().read("dataframe")
        
        parsed_results = []
        for idx, row in df.iterrows():
            parsed = parse_reasoning(row["reasoning"])
            parsed_results.append(parsed)

        
        # Add parsed results to dataframe
        df["parsed_reasoning"] = parsed_results
        storage.write(df)
        
        self.logger.info(f"✓ Parsing complete")
        
        # ============================================================
        # STAGE 3: Reasoning Data Reformatting
        # ============================================================
        self.logger.info("\n" + "="*80)
        self.logger.info("STAGE 3: REASONING DATA REFORMATTING")
        self.logger.info("="*80)
        
        # Step 1: Load and prepare data
        self.logger.info("\n[Step 1/3] Loading parsed reasoning data...")
        df = storage.step().read("dataframe")
        self.logger.info(f"Loaded {len(df)} videos with parsed reasoning")
        
        # Step 2: Construct prompts for each video
        self.logger.info("\n[Step 2/3] Constructing reformatting prompts...")
        prompts, full_questions, valid_indices = self._build_reformatting_prompts(df)
        self.logger.info(f"✓ Prepared {len(prompts)} prompts for reformatting")
        
        # Create a new dataframe with only valid rows
        valid_df = df.loc[valid_indices].copy().reset_index(drop=True)
        valid_df["prompt"] = prompts
        valid_df["full_question"] = full_questions
        
        storage.write(valid_df)
        
        # Step 3: Generate reformatted reasoning using API LLM
        self.logger.info("\n[Step 3/3] Generating reformatted reasoning with API LLM...")
        self.prompted_vqa_generator.run(
            storage=storage.step(),
            input_prompt_key="prompt",
            output_answer_key="reformatted_reasoning",
        )
        self.logger.info("✓ Reasoning reformatting complete")
        
        self.logger.info("\n" + "="*80)
        self.logger.info("✓ STAGE 3 COMPLETE: Reasoning data reformatted")
        self.logger.info("="*80)
        
        # ============================================================
        # Pipeline Complete
        # ============================================================
        self.logger.info("\n" + "="*80)
        self.logger.info("✓✓✓ COMPLETE PIPELINE FINISHED SUCCESSFULLY! ✓✓✓")
        self.logger.info("="*80)
        self.logger.info("\nOutput includes:")
        self.logger.info("  - captions: Merged video captions")
        self.logger.info("  - reasoning: Raw reasoning QA text")
        self.logger.info("  - parsed_reasoning: Structured reasoning data")
        self.logger.info("  - full_question: Complete question with options (cleaned)")
        self.logger.info("  - reformatted_reasoning: Optimized reasoning for training")
        self.logger.info("="*80)
        
        return "reformatted_reasoning"


if __name__ == "__main__":
    storage = FileStorage(
        first_entry_file_name="./dataflow/example/video_split/sample_data.json",
        cache_path="./cache",
        file_name_prefix="video_longvideo_cotqa_api",
        cache_type="json",
    )
    
    pipeline = LongVideoPipelineAPI()
    
    pipeline.run(
        storage=storage,
        input_video_key="video",
        input_conversation_key="conversation",
        output_key="caption",
    )

