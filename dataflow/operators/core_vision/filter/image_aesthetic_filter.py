import cv2
import numpy as np
from tqdm import tqdm
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY

@OPERATOR_REGISTRY.register()
class ImageAestheticFilter(OperatorABC):
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
    def get_desc(lang: str = "zh") -> str:
        if lang == "zh":
            return (
                "ImageAestheticFilter 算子：对输入图像进行基础质量与美学过滤，综合评估清晰度、整体亮度、对比度以及极端黑/白像素比例，"
                "用于剔除模糊、过暗、过亮或大面积纯黑/纯白的低质量图片。\n"
                "输入：通过 run(storage, input_image_key) 中的 input_image_key 指定，从 DataFlowStorage 中读取包含图像路径的列；\n"
                "输出：在同一 DataFrame 中新增名为 quality 的布尔列，标记每张图像是否通过质量过滤，同时仅保留 quality 为 True 的行并写回存储；\n"
                "功能：逐个图像读取为灰度图，计算拉普拉斯方差衡量清晰度、像素均值衡量亮度、像素标准差衡量对比度，并统计接近纯黑与纯白像素的比例，"
                "满足清晰度阈值、亮度范围、对比度阈值且极端像素比例不超过上限的图像视为高质量，适合用于后续识别、检索或生成相关任务。"
            )
        else:
            return (
                "ImageAestheticFilter operator: performs basic image quality and aesthetic filtering by checking sharpness, "
                "overall brightness, contrast, and the ratio of extreme black/white pixels, in order to remove blurry, "
                "too dark, too bright, or nearly all-black/white low-quality images.\n"
                "Inputs: specified via run(storage, input_image_key), where input_image_key indicates the column in DataFlowStorage "
                "that contains image file paths;\n"
                "Output: adds a boolean column named quality in the same DataFrame to indicate whether each image passes the quality "
                "filter, and keeps only the rows with quality set to True when writing back to storage;\n"
                "Function: reads each image as a grayscale array, computes the variance of the Laplacian for sharpness, the mean "
                "intensity for brightness, the standard deviation for contrast, and the proportion of near-black and near-white pixels. "
                "An image is considered high quality if it meets the sharpness threshold, falls within the brightness range, exceeds "
                "the contrast threshold, and does not exceed the configured extreme pixel ratios, making it suitable for downstream "
                "recognition, retrieval, or generation tasks."
            )

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

    def run(self, storage: DataFlowStorage, input_image_key: str = "image_path"):
        dataframe = storage.read("dataframe")
        dataframe["quality"] = dataframe[input_image_key].apply(self.is_quality)
        refined_df = dataframe[dataframe["quality"]].reset_index(drop=True)
        storage.write(refined_df)
        return [input_image_key]