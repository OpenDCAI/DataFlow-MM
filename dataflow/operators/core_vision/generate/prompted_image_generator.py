import os
import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger

from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import VLMServingABC


@OPERATOR_REGISTRY.register()
class PromptedImageGenerator(OperatorABC):
    def __init__(
        self,
        t2i_serving: VLMServingABC,
        save_interval: int = 50,
    ):
        self.t2i_serving = t2i_serving
        self.save_interval = save_interval

    @staticmethod
    def get_desc(lang: str = "en") -> str:
        return (
            "Generate the required images based on the provided large set of textual prompts."
            if lang != "zh"
            else "基于给定的大量提示词，生成需要的图片"
        )

    def run(
        self,
        storage: DataFlowStorage,
        input_conversation_key: str = "conversation",
        output_image_key: str = "images",
    ):
        if output_image_key is None:
            raise ValueError("At least one of output_key must be provided.")

        # Read prompts into a DataFrame
        df = storage.read(output_type="dict")
        df = pd.DataFrame(df)
        if not isinstance(df, pd.DataFrame):
            raise ValueError("storage.read must return a pandas DataFrame")

        # Initialize the output column with empty lists
        df[output_image_key] = [[] for _ in range(len(df))]

        processed = 0
        total = len(df)
        ########## batch processing move to the serving ##########
        # for start in range(0, total, self.batch_size):
        #     batch_indices = list(range(start, min(start + self.batch_size, total)))
        #     batch_prompts = [
        #         df.at[idx, input_conversation_key][-1]["content"]
        #         for idx in batch_indices
        #     ]

        #     # Generate images for the batch
        #     generated = self.t2i_serving.generate_from_input(batch_prompts)

        #     # Assign generated images back to DataFrame and periodically save
        #     for idx, prompt in zip(batch_indices, batch_prompts):
        #         df.at[idx, output_image_key] = generated.get(prompt, [])
        #         processed += 1
        #         if processed % self.save_interval == 0:
        #             storage.media_key = output_image_key
        #             storage.write(df)
        prompts_and_idx = []
        for idx in range(total):
            conv = df.at[idx, input_conversation_key]
            if isinstance(conv, (list, tuple)):
                for msg in conv:
                    if isinstance(msg, dict) and isinstance(msg.get("content"), str) and msg["content"].strip():
                        prompts_and_idx.append((msg["content"], idx))

        if not prompts_and_idx:
            storage.media_key = output_image_key
            storage.write(df)
            return

        batch_prompts = [p for p, _ in prompts_and_idx]
        generated = self.t2i_serving.generate_from_input(batch_prompts)

        for prompt, idx in prompts_and_idx:
            imgs = generated.get(prompt, [])
            if imgs is None:
                imgs = []
            if not isinstance(imgs, list):
                imgs = [imgs]
            df.at[idx, output_image_key].extend(imgs)

        # Final flush of any remaining prompts
        storage.media_key = output_image_key
        storage.write(df)
