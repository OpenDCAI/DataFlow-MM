import sys
import cv2
import numpy as np
from pathlib import Path
from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage
from dataflow import get_logger

OP_NAME = "video_resolution_filter"


@OPERATOR_REGISTRY.register()
class VideoResolutionFilter(OperatorABC):
    """
    视频分辨率过滤器

    根据视频的宽度和高度过滤样本，保留分辨率在指定范围内的视频。
    """

    def __init__(
            self,
            min_width: int = 1,
            max_width: int = sys.maxsize,
            min_height: int = 1,
            max_height: int = sys.maxsize,
            any_or_all: str = "any",
    ):
        """
        初始化

        Args:
            min_width: 最小水平分辨率（宽度）
            max_width: 最大水平分辨率（宽度）
            min_height: 最小垂直分辨率（高度）
            max_height: 最大垂直分辨率（高度）
            any_or_all: 多视频保留策略 (any/all)
        """
        self.logger = get_logger()

        self.min_width = min_width
        self.max_width = max_width
        self.min_height = min_height
        self.max_height = max_height

        if any_or_all not in ["any", "all"]:
            raise ValueError(
                f"Keep strategy [{any_or_all}] is not supported. "
                f'Can only be one of ["any", "all"].'
            )
        self.any = any_or_all == "any"

    @staticmethod
    def get_desc(lang: str = "zh"):
        """获取算子描述"""
        if lang == "zh":
            return (
                "视频分辨率过滤器\n\n"
                "根据视频的宽度和高度过滤样本。\n\n"
                "参数说明：\n"
                "- min_width/max_width: 宽度范围\n"
                "- min_height/max_height: 高度范围\n"
                "- any_or_all: 多视频保留策略\n\n"
                "输出字段：\n"
                "- video_width: 视频宽度\n"
                "- video_height: 视频高度\n"
                "- passed_filter: 是否通过过滤\n"
            )
        elif lang == "en":
            return (
                "Video Resolution Filter\n\n"
                "Filters samples based on video width and height.\n\n"
                "Parameters:\n"
                "- min_width/max_width: Width range\n"
                "- min_height/max_height: Height range\n"
                "- any_or_all: Retention strategy for multiple videos\n\n"
                "Output Fields:\n"
                "- video_width: Video width\n"
                "- video_height: Video height\n"
                "- passed_filter: Whether passed the filter\n"
            )
        else:
            return "Video resolution filter"

    def _get_video_resolution(self, video_path: str) -> tuple:
        """
        获取视频分辨率

        Args:
            video_path: 视频文件路径

        Returns:
            tuple: (width, height)，如果失败返回 (-1, -1)
        """
        try:
            cap = cv2.VideoCapture(str(video_path))

            if not cap.isOpened():
                return (-1, -1)

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            cap.release()

            return (width, height)

        except Exception as e:
            self.logger.error(f"Error getting resolution for {video_path}: {e}")
            return (-1, -1)

    def run(self, storage: DataFlowStorage, video_key: str = 'video_path'):
        """
        运行算子

        Args:
            storage: DataFlow存储对象
            video_key: 视频路径字段名

        Returns:
            list: 输出字段名列表
        """
        dataframe = storage.read(output_type="dataframe")

        video_widths = []
        video_heights = []
        passed_filters = []

        for idx, row in dataframe.iterrows():
            try:
                width, height = self._get_video_resolution(row[video_key])

                if width > 0 and height > 0:
                    passed = (
                            self.min_width <= width <= self.max_width and
                            self.min_height <= height <= self.max_height
                    )
                else:
                    passed = False

                video_widths.append(width)
                video_heights.append(height)
                passed_filters.append(passed)

            except Exception as e:
                self.logger.error(f"Error processing sample {idx}: {e}")
                video_widths.append(-1)
                video_heights.append(-1)
                passed_filters.append(False)

        dataframe['video_width'] = video_widths
        dataframe['video_height'] = video_heights
        dataframe['passed_filter'] = passed_filters

        storage.write(dataframe)

        return ['video_width', 'video_height', 'passed_filter']