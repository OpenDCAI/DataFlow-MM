import os
from typing import List, Optional
from tqdm import tqdm
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY

@OPERATOR_REGISTRY.register()
class VisionSegCutoutRefiner(OperatorABC):
    def __init__(
        self,
        seg_model_path: str = "/data0/mt/.cache/ckpt/yolo/yolo11l-seg.pt",
        conf: float = 0.25,
        classes: Optional[List[int]] = None,
        alpha_threshold: int = 127,
        output_suffix: str = "_seg",
        merge_instances: bool = False
    ):
        self.logger = get_logger()
        self.seg_model_path = seg_model_path
        self.conf = conf
        self.classes = classes
        self.alpha_threshold = alpha_threshold
        self.output_suffix = output_suffix
        self.merge_instances = merge_instances
        self.model = YOLO(self.seg_model_path)

    @staticmethod
    def get_desc(self, lang):
        return "实例分割抠图细化处理" if lang == "zh" else "Instance segmentation cutout refine operator."

    def _derive_output_path(self, img_path: str) -> str:
        base, _ = os.path.splitext(img_path)
        return f"{base}{self.output_suffix}.png"

    def _process_image(self, img_path: str) -> Optional[str]:
        if not os.path.isfile(img_path):
            return None
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            return None
        h, w = img_bgr.shape[:2]
        results = self.model.predict(source=img_path, save=False, verbose=False, conf=self.conf, classes=self.classes)
        r = results[0]
        if r.masks is None:
            return None
        output = np.zeros((h, w, 4), dtype=np.uint8)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        if self.merge_instances:
            masks = r.masks.data.cpu().numpy()
            cmask = np.zeros((h, w), dtype=np.uint8)
            for m in masks:
                m = cv2.resize((m * 255).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                cmask = np.maximum(cmask, m)
            sel = cmask > self.alpha_threshold
            output[:, :, :3][sel] = img_rgb[sel]
            output[:, :, 3][sel] = 255
        else:
            for m in r.masks.data.cpu().numpy():
                m = cv2.resize((m * 255).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                sel = m > self.alpha_threshold
                output[:, :, :3][sel] = img_rgb[sel]
                output[:, :, 3][sel] = 255
        out_path = self._derive_output_path(img_path)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        Image.fromarray(output).save(out_path)
        return out_path

    def run(self, storage: DataFlowStorage, image_key: str):
        dataframe = storage.read("dataframe")
        for i, row in enumerate(tqdm(dataframe.itertuples(), desc=f"Implementing {self.__class__.__name__}")):
            img_path = getattr(row, image_key)
            out_path = self._process_image(img_path)
            if out_path is not None:
                dataframe.at[i, image_key] = out_path
        storage.write(dataframe)
        return [image_key]
