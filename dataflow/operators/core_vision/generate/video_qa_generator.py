from __future__ import annotations

from typing import List, Optional

import pandas as pd

from dataflow import get_logger
from dataflow.core import OperatorABC, VLMServingABC
from dataflow.prompts.video import DiyVideoPrompt, VideoQAGeneratorPrompt
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage
from dataflow.operators.core_vision.generate.prompted_vqa_generator import PromptedVQAGenerator


@OPERATOR_REGISTRY.register()
class VideoCaptionToQAGenerator(OperatorABC):
    """Generate QA conversations from video captions via a prompt template.

    This operator rewrites the first user message in each conversation to a
    prompt synthesized from the row's ``caption`` using a configurable template,
    then calls the provided VLM serving to produce responses.
    """

    # ----------------------------- Lifecycle ---------------------------------
    def __init__(
        self,
        vlm_serving: VLMServingABC,
        prompt_template: Optional[VideoQAGeneratorPrompt | DiyVideoPrompt | str] = None,
        use_video_input: bool = True,
    ) -> None:
        self.logger = get_logger()
        self.vlm_serving = vlm_serving
        self.use_video_input = use_video_input
        # Initialize prompt template
        if prompt_template is None:
            self.prompt_template: VideoQAGeneratorPrompt | DiyVideoPrompt = VideoQAGeneratorPrompt()
        elif isinstance(prompt_template, str):
            self.prompt_template = DiyVideoPrompt(prompt_template)
        else:
            self.prompt_template = prompt_template

    # ----------------------------- Metadata ----------------------------------
    @staticmethod
    def get_desc(lang: str = "zh") -> str:
        return "基于 prompt 生成数据" if lang == "zh" else "Generate data from a prompt."

    # ----------------------------- Helpers -----------------------------------
    def _build_prompts(self, df: pd.DataFrame) -> List[str]:
        """Build one prompt per row using the template and the ``caption`` column.

        Raises:
            KeyError: if the required ``caption`` column is missing.
        """
        if "caption" not in df.columns:
            raise KeyError("Input dataframe must contain a 'caption' column.")

        # Using .apply keeps the code concise and readable.
        prompts = df["caption"].apply(lambda c: self.prompt_template.build_prompt(caption=c))
        return prompts.tolist()

    @staticmethod
    def _set_first_user_message(conversation: object, value: str) -> object:
        """Safely set the first user message's 'value' in a conversation.

        Expected format: a list of messages (dicts), where the first item
        represents the user's message and has a 'value' field.
        """
        try:
            if isinstance(conversation, list) and conversation:
                first = conversation[0]
                if isinstance(first, dict) and "value" in first:
                    first["value"] = value
        except Exception:  # Be defensive but don't fail the whole run
            pass
        return conversation

    # ----------------------------- Execution ---------------------------------
    def run(
        self,
        storage: DataFlowStorage,
        input_image_key: str = "image",
        input_video_key: str = "video",
        input_conversation_key: str = "conversation",
        # 输出的 conversation 可能是 None 也可能是 conversation，请类型检查
        output_key: str = "answer",
    ) -> str:
        if not output_key:
            raise ValueError("'output_key' must be a non-empty string.")

        self.logger.info("Running VideoCaptionToQAGenerator ...")

        df: pd.DataFrame = storage.read("dataframe")
        self.logger.info("Loaded dataframe with %d rows", len(df))
        
        prompts = self._build_prompts(df)

        # Rewrite the first user message per conversation to the built prompt.
        if input_conversation_key not in df.columns:
            raise KeyError("Input dataframe must contain a 'conversation' column.")

        df[input_conversation_key] = [
            self._set_first_user_message(conv, prompt)
            for conv, prompt in zip(df[input_conversation_key].tolist(), prompts)
        ]

        # If use_video_input is False, temporarily clear the video column
        if not self.use_video_input and input_video_key in df.columns:
            df[input_video_key] = [None] * len(df)
        # Write the modified dataframe back to storage
        storage.write(df)

        # Use PromptedVQAGenerator to generate QA
        prompted_vqa_generator = PromptedVQAGenerator(
            serving=self.vlm_serving,
            system_prompt="You are a helpful assistant."
        )
        
        # Call PromptedVQAGenerator to generate responses
        output_key_result = prompted_vqa_generator.run(
            storage=storage.step(),
            input_image_key=input_image_key,
            input_video_key=input_video_key,
            input_conversation_key=input_conversation_key,
            output_answer_key=output_key,
        )

        self.logger.info("Generation finished for %d rows", len(df))
        return output_key_result

