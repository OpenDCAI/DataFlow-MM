import os
import pandas as pd
import numpy as np
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
        save_image_with_idx: bool = True,
    ):
        logger = get_logger()
        
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

        prompts_and_idx = []
        save_id_list = []
        for idx, row in df.iterrows():
            if output_image_key in row.keys():
                if len(row[output_image_key]) > 0:
                    if row[output_image_key][0] != "":
                        continue

            conv = df.at[idx, input_conversation_key]
            if isinstance(conv, (list, tuple)):
                for c_idx, msg in enumerate(conv):
                    if isinstance(msg, dict) and isinstance(msg.get("content"), str) and msg["content"].strip():
                        save_id_list.append({"text_prompt": msg["content"],
                                             "sample_id": f"sample{idx}_condition{c_idx}"})
                        if save_image_with_idx:
                            prompts_and_idx.append((f"sample{idx}_condition{c_idx}", idx))
                        else:
                            prompts_and_idx.append((msg["content"], idx))

        if not prompts_and_idx:
            storage.media_key = output_image_key
            storage.write(df)
            return

        logger.info(f"Processing {len(prompts_and_idx)} prompts...")

        if save_image_with_idx:
            batch_prompts = save_id_list
        else:
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
