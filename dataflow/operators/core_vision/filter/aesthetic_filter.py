import cv2
import numpy as np
from tqdm import tqdm
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY

@OPERATOR_REGISTRY.register()
class AestheticFilter(OperatorABC):
    def __init__(
        self,
        blur_thresh: float = 150.0,
        brightness_range: tuple[float, float] = (30, 230),
        contrast_thresh: float = 40.0,
        max_black_ratio: float = 0.90,
        max_white_ratio: float = 0.90
    ):
        self.logger = get_logger()
        self.blur_thresh = blur_thresh
        self.bright_min, self.bright_max = brightness_range
        self.contrast_thresh = contrast_thresh
        self.max_black_ratio = max_black_ratio
        self.max_white_ratio = max_white_ratio

    @staticmethod
    def get_desc(lang):
        return "图片基础美学过滤（清晰度、亮度、对比度、极端像素比例）" if lang == "zh" else \
               "Image aesthetic filter (sharpness, brightness, contrast, extreme pixel ratio)."

    def safe_read_gray(self, image_path: str):
        gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            self.logger.warning(f"Failed to load: {image_path}")
            return None
        return gray

    def variance_of_laplacian(self, img_gray: np.ndarray) -> float:
        return float(cv2.Laplacian(img_gray, cv2.CV_64F).var())

    def mean_brightness(self, img_gray: np.ndarray) -> float:
        return float(img_gray.mean())

    def contrast(self, img_gray: np.ndarray) -> float:
        return float(img_gray.std())

    def extreme_pixel_ratios(self, img_gray: np.ndarray):
        total = img_gray.size
        black = np.sum(img_gray < 10)
        white = np.sum(img_gray > 245)
        return black / total, white / total

    def is_clear(self, gray: np.ndarray) -> bool:
        return self.variance_of_laplacian(gray) >= self.blur_thresh

    def is_well_lit(self, gray: np.ndarray) -> bool:
        m = self.mean_brightness(gray)
        return self.bright_min <= m <= self.bright_max

    def has_contrast(self, gray: np.ndarray) -> bool:
        return self.contrast(gray) >= self.contrast_thresh

    def not_extreme(self, gray: np.ndarray) -> bool:
        black_ratio, white_ratio = self.extreme_pixel_ratios(gray)
        if black_ratio > self.max_black_ratio or white_ratio > self.max_white_ratio:
            self.logger.info(
                f"Extreme pixel ratio: black={black_ratio:.2f}, white={white_ratio:.2f}"
            )
            return False
        return True

    def is_quality(self, image_path: str) -> bool:
        gray = self.safe_read_gray(image_path)
        if gray is None:
            return False
        return (
            self.is_clear(gray)
            and self.is_well_lit(gray)
            and self.has_contrast(gray)
            and self.not_extreme(gray)
        )

    def run(self, storage: DataFlowStorage, image_key: str):
        dataframe = storage.read("dataframe")
        dataframe["quality"] = dataframe[image_key].apply(self.is_quality)
        refined_df = dataframe[dataframe["quality"]].reset_index(drop=True)
        storage.write(refined_df)
        return [image_key]