import os
import pandas as pd

from dataflow.operators.core_vision import VisionSegCutoutRefiner

class InMemoryStorage:
    def __init__(self, df: pd.DataFrame):
        self._df = df.copy()
    def read(self, name: str):
        return self._df.copy()
    def write(self, df: pd.DataFrame):
        self._df = df.reset_index(drop=True)
        return self._df

img_path = "https://huggingface.co/datasets/OpenDCAI/dataflow-demo-image/resolve/main/seg_images/image1.png"
df = pd.DataFrame({"image_path": [img_path]})

storage = InMemoryStorage(df)

op = VisionSegCutoutRefiner(
    seg_model_path="../ckpt/yolo/yolo11l-seg.pt",   
    classes=None,           
    alpha_threshold=127,
    output_suffix="_seg",
    merge_instances=False
)

op.run(storage, image_key="image_path")

result_df = storage.read("dataframe")
print(result_df)

for p in result_df["image_path"].tolist():
    print(p, "-> exists:", os.path.exists(p))
