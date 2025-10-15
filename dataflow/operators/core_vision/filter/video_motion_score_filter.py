import sys
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union
from contextlib import contextmanager
from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage
from dataflow import get_logger

OP_NAME = "video_motion_score_filter"


@contextmanager
def VideoCapture(*args, **kwargs):
    """上下文管理器，确保视频资源正确释放"""
    cap = cv2.VideoCapture(*args, **kwargs)
    try:
        yield cap
    finally:
        cap.release()


@OPERATOR_REGISTRY.register()
class VideoMotionScoreFilter(OperatorABC):
    """
    视频运动分数过滤器
    
    使用 Farneback 光流算法计算视频的运动分数，保留运动分数在指定范围内的样本。
    运动分数计算为光流幅度的平均值，可以相对于帧对角线长度进行归一化。
    """
    
    _default_kwargs = {
        "pyr_scale": 0.5,
        "levels": 3,
        "winsize": 15,
        "iterations": 3,
        "poly_n": 5,
        "poly_sigma": 1.2,
        "flags": 0,
    }
    
    def __init__(
        self,
        min_score: float = 0.25,
        max_score: float = sys.float_info.max,
        sampling_fps: float = 2.0,
        size: Union[int, Tuple[int], Tuple[int, int], None] = None,
        max_size: Optional[int] = None,
        divisible: int = 1,
        relative: bool = False,
        any_or_all: str = "any",
        **kwargs
    ):
        """
        初始化
        
        Args:
            min_score: 保留样本的最小运动分数
            max_score: 保留样本的最大运动分数
            sampling_fps: 光流计算的采样帧率（帧/秒）
            size: 计算光流前调整帧大小
            max_size: 调整后较长边的最大允许值
            divisible: 尺寸必须能被该数整除
            relative: 是否归一化光流幅度
            any_or_all: 多视频保留策略 ('any'/'all')
            **kwargs: Farneback 算法的额外参数
        """
        self.logger = get_logger()
        
        self.min_score = min_score
        self.max_score = max_score
        self.sampling_fps = sampling_fps
        
        if isinstance(size, (list, tuple)):
            if len(size) not in [1, 2]:
                raise ValueError(
                    f"Size must be an int or a 1 or 2 element tuple/list, "
                    f"not a {len(size)} element tuple/list."
                )
        if isinstance(size, int):
            size = (size,)
        self.size = size
        self.max_size = max_size
        self.divisible = divisible
        self.relative = relative
        
        self.extra_kwargs = self._default_kwargs.copy()
        for key in kwargs:
            if key in self.extra_kwargs:
                self.extra_kwargs[key] = kwargs[key]
        
        if any_or_all not in ["any", "all"]:
            raise ValueError(
                f"Keep strategy [{any_or_all}] is not supported. "
                f'Can only be one of ["any", "all"].'
            )
        self.any = any_or_all == "any"
        
        self.model = cv2.calcOpticalFlowFarneback
    
    @staticmethod
    def get_desc(lang: str = "zh"):
        """获取算子描述"""
        if lang == "zh":
            return (
                "视频运动分数过滤器\n\n"
                "使用 Farneback 光流算法计算视频运动分数，保留分数在指定范围内的样本。\n\n"
                "参数说明：\n"
                "- min_score/max_score: 运动分数阈值范围\n"
                "- sampling_fps: 采样帧率（帧/秒）\n"
                "- size: 处理前调整帧大小\n"
                "- relative: 是否归一化分数\n"
                "- any_or_all: 多视频保留策略\n\n"
                "输出字段：\n"
                "- video_motion_score: 视频运动分数\n"
                "- passed_filter: 是否通过过滤\n"
            )
        elif lang == "en":
            return (
                "Video Motion Score Filter\n\n"
                "Calculates video motion scores using Farneback optical flow algorithm "
                "and retains samples with scores within specified range.\n\n"
                "Parameters:\n"
                "- min_score/max_score: Motion score threshold range\n"
                "- sampling_fps: Sampling frame rate (frames per second)\n"
                "- size: Frame size to resize before processing\n"
                "- relative: Whether to normalize the score\n"
                "- any_or_all: Retention strategy for multiple videos\n\n"
                "Output Fields:\n"
                "- video_motion_score: Video motion score\n"
                "- passed_filter: Whether the video passed the filter\n"
            )
        else:
            return "Video motion score filter using Farneback optical flow"
    
    def _calculate_resized_dimensions(
        self, 
        original_size: Tuple[int, int],
    ) -> Tuple[int, int]:
        """计算调整后的尺寸"""
        height, width = original_size
        
        if self.size is None:
            return (width, height)
        
        if len(self.size) == 1:
            size = self.size[0]
            if height > width:
                new_height = int(size * height / width)
                new_width = size
            else:
                new_width = int(size * width / height)
                new_height = size
            
            if self.max_size is not None:
                if max(new_height, new_width) > self.max_size:
                    if new_height > new_width:
                        new_height = self.max_size
                        new_width = int(self.max_size * width / height)
                    else:
                        new_width = self.max_size
                        new_height = int(self.max_size * height / width)
        else:
            new_height, new_width = self.size
        
        new_height = (new_height // self.divisible) * self.divisible
        new_width = (new_width // self.divisible) * self.divisible
        
        return (new_width, new_height)
    
    def _compute_flow(self, prev_frame, curr_frame):
        """计算两帧之间的光流"""
        curr_frame_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        if prev_frame is None:
            flow = None
        else:
            flow = self.model(
                prev_frame, 
                curr_frame_gray, 
                None, 
                **self.extra_kwargs
            )
        
        return flow, curr_frame_gray
    
    def _compute_video_motion_score(self, video_path: str) -> float:
        """计算单个视频的运动分数"""
        video_motion_scores = []
        
        with VideoCapture(str(video_path)) as cap:
            if not cap.isOpened():
                return -1.0
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            sampling_fps = min(self.sampling_fps, fps)
            sampling_step = round(fps / sampling_fps)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            sampling_step = max(min(sampling_step, total_frames - 1), 1)
            
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            new_size = self._calculate_resized_dimensions((height, width))
            
            prev_frame = None
            frame_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                if new_size != (width, height):
                    frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
                
                flow, prev_frame = self._compute_flow(prev_frame, frame)
                
                if flow is not None:
                    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    frame_motion_score = np.mean(mag)
                    
                    if self.relative:
                        frame_motion_score /= np.hypot(*frame.shape[:2])
                    
                    video_motion_scores.append(frame_motion_score)
                
                frame_count += sampling_step
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        
        if not video_motion_scores:
            return -1.0
        
        return float(np.mean(video_motion_scores))
    
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
        
        motion_scores = []
        passed_filters = []
        
        for idx, row in dataframe.iterrows():
            try:
                motion_score = self._compute_video_motion_score(row[video_key])
                passed = self.min_score <= motion_score <= self.max_score
                
                motion_scores.append(motion_score)
                passed_filters.append(passed)
                
            except Exception as e:
                self.logger.error(f"Error processing sample {idx}: {e}")
                motion_scores.append(-1.0)
                passed_filters.append(False)
        
        dataframe['video_motion_score'] = motion_scores
        dataframe['passed_filter'] = passed_filters
        
        storage.write(dataframe)
        
        return ['video_motion_score', 'passed_filter']