import os
import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger

from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import VLMServingABC


@OPERATOR_REGISTRY.register()
class Text2ImageGenerator(OperatorABC):
    def __init__(self,
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

        # 1. Read prompts into a DataFrame
        df = storage.read(output_type="dict")
        df = pd.DataFrame(df)
        if not isinstance(df, pd.DataFrame):
            raise ValueError("storage.read must return a pandas DataFrame")

        # Initialize the output column with empty lists
        df[output_image_key] = [[] for _ in range(len(df))]

        for idx, info in enumerate(df[input_conversation_key]):
            prompt = info[-1]["content"]  # the last modified prompt, and df[input_key] already a list

            generated = self.t2i_serving.generate_from_input([prompt])   ## 后续改进一次控制输入多少prompt

            df.at[idx, output_image_key] = generated[prompt]

            if (idx + 1) % self.save_interval == 0:
                storage.media_key = output_image_key
                storage.write(df)

        # Final flush of any remaining prompts
        storage.media_key = output_image_key
        storage.write(df)
