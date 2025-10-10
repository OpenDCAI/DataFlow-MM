import os
import pandas as pd

from dataflow.operators.core_vision import VisionSegCutoutRefine

class InMemoryStorage:
    def __init__(self, df: pd.DataFrame):
        self._df = df.copy()
    def read(self, name: str):
        return self._df.copy()
    def write(self, df: pd.DataFrame):
        self._df = df.reset_index(drop=True)
        return self._df

img_path = "/data0/happykeyan/workspace/DataFlow-MM/dataflow/example/test_image_editing/images/image1.png"
df = pd.DataFrame({"image_path": [img_path]})

storage = InMemoryStorage(df)

op = VisionSegCutoutRefine(
    seg_model_path="/data0/mt/.cache/ckpt/yolo/yolo11l-seg.pt",   
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
