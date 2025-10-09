from __future__ import annotations

from typing import List, Optional

import pandas as pd

from dataflow import get_logger
from dataflow.core import OperatorABC, VLMServingABC
from dataflow.prompts.video import DiyVideoPrompt, VideoQAGeneratorPrompt
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage


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
    ) -> None:
        self.logger = get_logger()
        self.vlm_serving = vlm_serving
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
        input_audio_key: str = "audio",
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

        # Prepare media columns (gracefully handle missing columns by filling None)
        def _safe_list(col: str) -> List[object]:
            return df[col].tolist() if col in df.columns else [None] * len(df)

        image_list = _safe_list(input_image_key)
        video_list = _safe_list(input_video_key)
        audio_list = _safe_list(input_audio_key)
        conversations = df[input_conversation_key].tolist()

        # Call serving
        responses = self.vlm_serving.generate_from_input_messages(
            conversations=conversations,
            image_list=image_list,
            video_list=video_list,
            audio_list=audio_list,
        )

        # Attach outputs
        df[output_key] = responses
        storage.write(df)

        self.logger.info("Generation finished for %d rows", len(df))
        return output_key

