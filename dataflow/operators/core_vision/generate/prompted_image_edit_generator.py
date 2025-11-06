import os
import pandas as pd
import numpy as np
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger

from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import VLMServingABC


@OPERATOR_REGISTRY.register()
class PromptedImageEditGenerator(OperatorABC):
    def __init__(
        self,
        image_edit_serving: VLMServingABC,
        save_interval: int = 50,
    ):
        self.image_edit_serving = image_edit_serving
        self.save_interval = save_interval

    @staticmethod
    def get_desc(lang: str = "en") -> str:
        return (
            "Generate the corresponding edited results based on the given images and their associated editing instructions."
            if lang != "zh"
            else "基于给定的大量图片以及对应的编辑指令，生成对应的编辑结果"
        )

    def run(
        self,
        storage: DataFlowStorage,
        input_image_key: str = "images",
        input_conversation_key: str = "conversation",
        output_image_key: str = "edited_images",
        save_image_with_idx: bool = True,
    ):
        if output_image_key is None:
            raise ValueError("At least one of output_key must be provided.")

        # Read prompts into a DataFrame
        df = storage.read(output_type="dict")
        df = pd.DataFrame(df)
        if not isinstance(df, pd.DataFrame):
            raise ValueError("storage.read must return a pandas DataFrame")

        # Initialize the output column with empty lists
        if output_image_key not in df.columns:
            df[output_image_key] = [[] for _ in range(len(df))]

        processed = 0
        total = len(df)

        batch_prompts = []
        for idx, row in df.iterrows():
            if output_image_key in row.keys():
                if len(row[output_image_key]) > 0:
                    if row[output_image_key][0] != "":
                        continue
            if save_image_with_idx:
                batch_prompts.append({"idx": idx, "image_path": df.at[idx, input_image_key], "prompt": df.at[idx, input_conversation_key][-1]["content"]})
            else:
                batch_prompts.append((df.at[idx, input_image_key], df.at[idx, input_conversation_key][-1]["content"]))
        generated = self.image_edit_serving.generate_from_input(batch_prompts)
        for idx, prompt in enumerate(batch_prompts):
            if save_image_with_idx:
                df.at[prompt['idx'], output_image_key] = generated.get(f"sample_{prompt['idx']}", [])

            else:
                if isinstance(prompt, tuple):
                    prompt = prompt[1]
                df.at[idx, output_image_key] = generated[idx] if isinstance(generated, list) else generated.get(prompt, [])

        # Final flush of any remaining prompts
        storage.media_key = output_image_key
        storage.write(df)
