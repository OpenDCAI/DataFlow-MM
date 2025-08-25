import os
import pandas as pd
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
        #         (df.at[idx, input_image_key][0], df.at[idx, input_conversation_key][-1]["content"])
        #         for idx in batch_indices
        #     ]

        #     # Generate images for the batch
        #     generated = self.image_edit_serving.generate_from_input(batch_prompts)

        #     # Assign generated images back to DataFrame and periodically save
        #     for idx, prompt in zip(batch_indices, batch_prompts):
        #         if isinstance(prompt, tuple):
        #             prompt = prompt[1]
        #         df.at[idx, output_image_key] = generated.get(prompt, [])
        #         processed += 1
        #         if processed % self.save_interval == 0:
        #             storage.media_key = output_image_key
        #             storage.write(df)
        batch_prompts = [
            (df.at[idx, input_image_key][0], df.at[idx, input_conversation_key][-1]["content"])
            for idx in range(total)
        ]
        generated = self.image_edit_serving.generate_from_input(batch_prompts)
        for idx, prompt in zip(list(range(total)), batch_prompts):
            if isinstance(prompt, tuple):
                prompt = prompt[1]
            df.at[idx, output_image_key] = generated.get(prompt, [])

        # Final flush of any remaining prompts
        storage.media_key = output_image_key
        storage.write(df)
