from __future__ import annotations

import re
from typing import List, Optional

import pandas as pd

from dataflow import get_logger
from dataflow.core import OperatorABC, VLMServingABC
from dataflow.prompts.video import DiyVideoPrompt, VideoCOTQAGeneratorPrompt
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage
from dataflow.operators.core_vision.generate.prompted_vqa_generator import PromptedVQAGenerator


@OPERATOR_REGISTRY.register()
class VideoCOTQAGenerator(OperatorABC):
    """Generate Chain-of-Thought QA responses from video/image problems via a prompt template.

    This operator takes problem data (with problem_type, problem, options, data_type, video/image path, solution),
    generates CoT reasoning and answers using a VLM, computes rewards, and outputs structured results.
    """

    # ----------------------------- Lifecycle ---------------------------------
    def __init__(
        self,
        vlm_serving: VLMServingABC,
        prompt_template: Optional[VideoCOTQAGeneratorPrompt | DiyVideoPrompt | str] = None,
    ) -> None:
        self.logger = get_logger()
        self.vlm_serving = vlm_serving
        
        # Initialize prompt template
        if prompt_template is None:
            self.prompt_template: VideoCOTQAGeneratorPrompt | DiyVideoPrompt = VideoCOTQAGeneratorPrompt()
        elif isinstance(prompt_template, str):
            self.prompt_template = DiyVideoPrompt(prompt_template)
        else:
            self.prompt_template = prompt_template

    # ----------------------------- Metadata ----------------------------------
    @staticmethod
    def get_desc(lang: str = "zh") -> str:
        return "基于 CoT prompt 生成视频/图像 QA 数据" if lang == "zh" else "Generate video/image QA data with CoT reasoning."

    # ----------------------------- Helpers -----------------------------------
    def _build_prompts(self, df: pd.DataFrame) -> List[str]:
        """Build one prompt per row using the template and problem information."""
        prompts = []
        for _, row in df.iterrows():
            problem_type = row.get('problem_type', '')
            problem = row.get('problem', '')
            options = row.get('options', [])
            
            # Format question with options if multiple choice
            if problem_type == 'multiple choice' and options:
                question = problem + "Options:\n"
                for op in options:
                    question += op + "\n"
            else:
                question = problem
            
            # Build prompt with type-specific suffix
            type_template = getattr(self.prompt_template, 'type_template', {})
            type_suffix = type_template.get(problem_type, "")
            prompt = self.prompt_template.build_prompt(Question=question) + type_suffix
            prompts.append(prompt)
        
        return prompts

    @staticmethod
    def _set_first_user_message(conversation: object, value: str) -> object:
        """Safely set the first user message's 'value' in a conversation."""
        try:
            if isinstance(conversation, list) and conversation:
                first = conversation[0]
                if isinstance(first, dict) and "value" in first:
                    first["value"] = value
        except Exception:
            pass
        return conversation

    @staticmethod
    def extract_think(output_str: str) -> str:
        """Extract content between <think> and </think> tags."""
        pattern = r'<think>\s*(.*?)\s*</think>'
        match = re.search(pattern, output_str, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    @staticmethod
    def extract_answer(text: str) -> str:
        """Extract content between <answer> and </answer> tags."""
        pattern = r'<answer>\s*(.*?)\s*</answer>'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    # ----------------------------- Execution ---------------------------------
    def run(
        self,
        storage: DataFlowStorage,
        input_video_key: str = "video",
        input_image_key: str = "image",
        input_conversation_key: str = "conversation",
        output_answer_key: str = "answer",
        output_process_key: str = "process",
        output_full_response_key: str = "full_response",
    ) -> str:
        """
        Process dataframe with video/image CoT QA generation.
        
        This operator only generates responses and extracts answers/processes.
        For evaluation and filtering, use GeneralTextAnswerEvaluator and ScoreFilter separately.
        
        Expected input columns:
        - problem_type: str (multiple choice, numerical, OCR, free-form, regression)
        - problem: str
        - options: list
        - data_type: str (video or image)
        - video: list[str] (video path)
        - solution: str (ground truth with <answer> tags)
        """
        self.logger.info("Running VideoCOTQAGenerator ...")
        df: pd.DataFrame = storage.read("dataframe")
        self.logger.info("Loaded dataframe with %d rows", len(df))

        # Build prompts
        prompts = self._build_prompts(df)

        # Create or update conversations
        if input_conversation_key not in df.columns or df[input_conversation_key].isna().all():
            # Create default conversations
            df[input_conversation_key] = [
                [{"from": "human", "value": prompt}] for prompt in prompts
            ]
        else:
            # Update existing conversations
            df[input_conversation_key] = [
                self._set_first_user_message(conv, prompt)
                for conv, prompt in zip(df[input_conversation_key].tolist(), prompts)
            ]

        # Write the modified dataframe back to storage
        storage.write(df)

        # Use PromptedVQAGenerator to generate responses
        self.logger.info("Generating CoT QA responses...")
        prompted_vqa_generator = PromptedVQAGenerator(
            serving=self.vlm_serving,
            system_prompt="You are a helpful assistant."
        )
        
        # Call PromptedVQAGenerator to generate responses
        temp_response_key = "_temp_cotqa_response"
        prompted_vqa_generator.run(
            storage=storage.step(),
            input_image_key=input_image_key,
            input_video_key=input_video_key,
            input_conversation_key=input_conversation_key,
            output_answer_key=temp_response_key,
        )
        storage.step()
        # Read back the results with responses
        df = storage.read("dataframe")
        responses = df[temp_response_key].tolist()

        # Process responses - extract think chain and answer
        answers = []
        processes = []

        for response in responses:
            # Extract think chain and answer
            think_chain = self.extract_think(response)
            final_ans = self.extract_answer(response)
            
            answers.append(final_ans)
            processes.append(f"<think>{think_chain}</think>" if think_chain else "")

        # Attach extracted answers and processes
        df[output_answer_key] = answers
        df[output_process_key] = processes
        df[output_full_response_key] = responses
        
        # Clean up temporary column
        df = df.drop(columns=[temp_response_key])
        storage.write(df)

        self.logger.info("Generation finished for %d rows", len(df))
        
        return output_answer_key

