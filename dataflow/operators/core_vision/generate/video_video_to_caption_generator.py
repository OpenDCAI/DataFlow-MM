from __future__ import annotations

from typing import List, Optional

import pandas as pd

from dataflow import get_logger
from dataflow.core import OperatorABC, VLMServingABC
from dataflow.prompts.video import DiyVideoPrompt, VideoCaptionGeneratorPrompt
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage


@OPERATOR_REGISTRY.register()
class VideoToCaptionGenerator(OperatorABC):
    """Generate captions from videos by prompting a VLM service.

    This operator rewrites the first user message in each conversation with a
    template-built prompt, then calls the provided VLM serving to generate
    captions. The generated captions are written to ``output_key`` (default:
    ``"caption"``).
    """

    def __init__(
        self,
        vlm_serving: VLMServingABC,
        prompt_template: Optional[VideoCaptionGeneratorPrompt | DiyVideoPrompt | str] = None,
    ) -> None:
        self.logger = get_logger()
        self.vlm_serving = vlm_serving
        # Initialize prompt template
        if prompt_template is None:
            self.prompt_template: VideoCaptionGeneratorPrompt | DiyVideoPrompt = VideoCaptionGeneratorPrompt()
        elif isinstance(prompt_template, str):
            self.prompt_template = DiyVideoPrompt(prompt_template)
        else:
            self.prompt_template = prompt_template

    @staticmethod
    def get_desc(lang: str = "zh") -> str:
        return "基于 prompt 生成数据" if lang == "zh" else "Generate data from a prompt."

    # ----------------------------- Helpers -----------------------------------
    def _build_prompts(self, df: pd.DataFrame) -> List[str]:
        """Build one prompt per row using the configured template.

        Unlike the QA variant, this template does not depend on row fields, so
        we simply create ``len(df)`` prompts by calling ``build_prompt()``.
        """
        return [self.prompt_template.build_prompt() for _ in range(len(df))]

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
        except Exception:
            # Be defensive but don't fail the whole run
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
        output_key: str = "caption",
    ) -> str:
        if not output_key:
            raise ValueError("'output_key' must be a non-empty string.")

        self.logger.info("Running VideoToCaptionGenerator ...")

        df: pd.DataFrame = storage.read("dataframe")
        self.logger.info("Loaded dataframe with %d rows", len(df))

        if input_conversation_key not in df.columns:
            raise KeyError("Input dataframe must contain a 'conversation' column.")

        prompts = self._build_prompts(df)

        # Rewrite the first user message per conversation to the built prompt.
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

        # Attach outputs and persist
        df[output_key] = responses
        storage.write(df)

        self.logger.info("Generation finished for %d rows", len(df))
        return output_key
