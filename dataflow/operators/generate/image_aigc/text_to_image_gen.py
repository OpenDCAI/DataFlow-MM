import os
import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger

from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import VLMServingABC
from dataflow.io.diffuser.t2i_gen import ImageIO


@OPERATOR_REGISTRY.register()
class Text2ImageGenerator(OperatorABC):
    def __init__(self, pipe):
        self.pipe = pipe

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
        input_image_key: str = "images", 
        input_video_key: str = "videos",
        input_audio_key: str = "audios",
        input_conversation_key: str = "conversation",
        output_key: str = "gen_images",
        save_interval: int = 50,
    ):
        if output_key is None:
            raise ValueError("At least one of output_key must be provided.")

        # 1. Read prompts into a DataFrame
        df = storage.read(output_type="dict")
        df = pd.DataFrame(df)
        if not isinstance(df, pd.DataFrame):
            raise ValueError("storage.read must return a pandas DataFrame")
        
        self.image_io = ImageIO()

        # Initialize the output column with empty lists
        df[output_key] = [[] for _ in range(len(df))]

        for idx, info in enumerate(df[input_conversation_key]):
            prompt = info[-1]["content"]  # the last modified prompt, and df[input_key] already a list

            generated = self.pipe.generate_from_input([prompt])   ## 后续改进一次控制输入多少prompt
            images = generated.get(prompt, [])

            filenames = [f"{idx}_{img_idx}.png" for img_idx in range(len(images))]

            self.image_io.write(storage=storage, image_data=images, orig_paths=filenames)

            prefix = os.path.join(storage.cache_path, "images")
            full_paths = [os.path.join(prefix, fn) for fn in filenames]

            df.at[idx, output_key] = full_paths

            if (idx + 1) % save_interval == 0:
                storage.media_key = output_key
                storage.write(df)
                storage.step()

        # Final flush of any remaining prompts
        storage.media_key = output_key
        storage.write(df)
