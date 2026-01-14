import pandas as pd
import numpy as np
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
import random
import numpy as np  # 如果还没导入numpy的话也需要添加

def filter_dataset(
    clips: list,
    frames_min: int = None,
    frames_max: int = None,
    fps_min: float = None,
    fps_max: float = None,
    resolution_max: int = None,
    aes_min: float = None,
    ocr_min: float = None,
    ocr_max: float = None,
    lum_min: float = None,
    lum_max: float = None,
    motion_min: float = None,
    motion_max: float = None,
    flow_min: float = None,
    flow_max: float = None,
    blur_max: float = None,
    strict_mode: bool = False
) -> list:
    """
    对视频的每个clip应用过滤条件，添加过滤标记。
    
    Args:
        clips: 视频片段列表，每个片段是一个字典，包含质量指标
        strict_mode: 是否严格模式，True则缺失列时报错，False则跳过该过滤条件
        
    Returns:
        list: 过滤后的clips列表，每个clip添加了`filtered`标记
    """
    filtered_clips = []

    # 定义列存在性检查辅助函数
    def check_column_exists(clip, column, condition_name):
        if column not in clip:
            msg = f"过滤条件 '{condition_name}' 需要列 '{column}'，但该clip中不存在"
            if strict_mode:
                raise ValueError(msg)
            else:
                get_logger().warning(msg + "，已跳过该过滤条件")
                return False
        return True

    for clip in clips:
        # 获取视频质量指标
        num_frames = clip.get("num_frames")
        fps = clip.get("fps")
        resolution = clip.get("resolution")
        aes_score = clip.get("aesthetic_score")
        ocr_score = clip.get("ocr_score")
        lum_mean = clip.get("luminance_mean")
        motion_score = clip.get("motion_score")
        flow_score = clip.get("flow_score")
        blur_score = clip.get("blur_score")

        # 检查是否已有 filtered 字段，如果有则使用已有值，否则默认为 True（通过）
        # 如果之前的过滤器已经设置为 False，则保持 False
        filtered = clip.get("filtered", True)  

        # 只有当当前 filtered 为 True 时，才进行后续的过滤检查
        # 如果已经是 False（被之前的过滤器标记），则跳过所有检查
        if filtered:
            # 应用过滤条件和列存在性检查
            if frames_min is not None:
                if not check_column_exists(clip, "num_frames", "最小帧数"):
                    filtered = False
                elif num_frames < frames_min:
                    filtered = False
            if frames_max is not None:
                if not check_column_exists(clip, "num_frames", "最大帧数"):
                    filtered = False
                elif num_frames > frames_max:
                    filtered = False
            if fps_min is not None:
                if not check_column_exists(clip, "fps", "最小帧率"):
                    filtered = False
                elif fps < fps_min:
                    filtered = False
            if fps_max is not None:
                if not check_column_exists(clip, "fps", "最大帧率"):
                    filtered = False
                elif fps > fps_max:
                    filtered = False
            if resolution_max is not None:
                if not check_column_exists(clip, "resolution", "最大分辨率"):
                    filtered = False
                elif resolution > resolution_max:
                    filtered = False
            if aes_min is not None:
                if not check_column_exists(clip, "aesthetic_score", "最小美学分数"):
                    filtered = False
                elif aes_score < aes_min:
                    filtered = False
            if ocr_min is not None:
                if not check_column_exists(clip, "ocr_score", "最小OCR分数"):
                    filtered = False
                elif ocr_score < ocr_min:
                    filtered = False
            if ocr_max is not None:
                if not check_column_exists(clip, "ocr_score", "最大OCR分数"):
                    filtered = False
                elif ocr_score > ocr_max:
                    filtered = False
            if lum_min is not None:
                if not check_column_exists(clip, "luminance_mean", "最小亮度分数"):
                    filtered = False
                elif lum_mean < lum_min:
                    filtered = False
            if lum_max is not None:
                if not check_column_exists(clip, "luminance_mean", "最大亮度分数"):
                    filtered = False
                elif lum_mean > lum_max:
                    filtered = False
            if motion_min is not None:
                if not check_column_exists(clip, "motion_score", "最小运动分数"):
                    filtered = False
                elif motion_score < motion_min:
                    filtered = False
            if motion_max is not None:
                if not check_column_exists(clip, "motion_score", "最大运动分数"):
                    filtered = False
                elif motion_score > motion_max:
                    filtered = False
            if flow_min is not None:
                if not check_column_exists(clip, "flow_score", "最小光流分数"):
                    filtered = False
                elif flow_score < flow_min:
                    filtered = False
            if flow_max is not None:
                if not check_column_exists(clip, "flow_score", "最大光流分数"):
                    filtered = False
                elif flow_score > flow_max:
                    filtered = False
            if blur_max is not None:
                if not check_column_exists(clip, "blur_score", "最大模糊分数"):
                    filtered = False
                elif blur_score > blur_max:
                    filtered = False

        # 给clip添加filtered标记
        clip["filtered"] = filtered

        # 如果该clip通过了过滤，则添加到结果列表
        filtered_clips.append(clip)

    return filtered_clips




@OPERATOR_REGISTRY.register()
class VideoScoreFilter(OperatorABC):
    """
    数据集过滤算子：根据各种质量指标过滤视频元数据
    支持多种过滤条件，如帧数、帧率、分辨率、美学分数等
    """

    def __init__(self,
                 frames_min: int = None,
                 frames_max: int = None,
                 fps_min: float = None,
                 fps_max: float = None,
                 resolution_max: int = None,
                 aes_min: float = 4,  
                 ocr_min: float = None,
                 ocr_max: float = 0.3,  
                 lum_min: float = 20,  
                 lum_max: float = 140,  
                 motion_min: float = 2,  
                 motion_max: float = 14,  
                 flow_min: float = None,
                 flow_max: float = None,
                 blur_max: float = None,
                 strict_mode: bool = False,  
                 seed: int = 42):
        """
        初始化数据集过滤算子
        
        Args:
            各种过滤条件参数，与filter_clips函数对应
            seed: 随机种子，用于保证可重复性
            strict_mode: 是否严格模式，True则缺失列时报错，False则跳过该过滤条件
        """
        self.logger = get_logger()
        self.strict_mode = strict_mode  
        self.frames_min = frames_min
        self.frames_max = frames_max
        self.fps_min = fps_min
        self.fps_max = fps_max
        self.resolution_max = resolution_max
        self.aes_min = aes_min
        self.ocr_min = ocr_min
        self.ocr_max = ocr_max
        self.lum_min = lum_min
        self.lum_max = lum_max
        self.motion_min = motion_min
        self.motion_max = motion_max
        self.flow_min = flow_min
        self.flow_max = flow_max
        self.blur_max = blur_max
        self.seed = seed
        
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return "数据集过滤：根据多种质量指标过滤视频元数据"
        else:
            return "Dataset Filter: Filter video metadata based on various quality metrics"

    def run(self,
            storage: DataFlowStorage,
            input_video_key: str = "video",
            video_clips_key: str = "video_clip",  
            output_key: str = "video_clips",  
            ):  
        """
        执行数据集过滤
        
        Args:
            storage: DataFlowStorage对象，用于数据读写
            input_video_key: 输入数据在storage中的键名
            output_key: 输出结果在storage中的键名
            
        Returns:
            str: 输出结果在storage中的键名
        """
        self.logger.info("Starting DatasetFilter...")
        
        # 从storage读取输入数据
        df = storage.read("dataframe")
        self.logger.info(f"Loaded dataframe: {len(df)} rows")
        
        # 获取视频和clips信息
        video_clips_data = df.get(video_clips_key, None)
        if video_clips_key not in df or df[video_clips_key].empty:
            raise ValueError(f"{video_clips_key} not found or is empty in the dataframe")
        

        # 假设 video_clips_data 是一个 Series 对象
        for idx, row in video_clips_data.items():  # 用 .items() 遍历 Series
            # import ipdb; ipdb.set_trace()
            # video_clips = row["clips"]  # 获取每个条目中的 "video_clip" 字段
            # 然后在这里处理 video_clips
            filtered_clips = filter_dataset(
                clips=row["clips"],
                strict_mode=self.strict_mode,
                frames_min=self.frames_min,
                frames_max=self.frames_max,
                fps_min=self.fps_min,
                fps_max=self.fps_max,
                resolution_max=self.resolution_max,
                aes_min=self.aes_min,
                ocr_min=self.ocr_min,
                ocr_max=self.ocr_max,
                lum_min=self.lum_min,
                lum_max=self.lum_max,
                motion_min=self.motion_min,
                motion_max=self.motion_max,
                flow_min=self.flow_min,
                flow_max=self.flow_max,
                blur_max=self.blur_max
            )
            # 更新 row 中的 video_clip
            row["clips"] = filtered_clips



        
        # 将过滤后的结果写回
        df[video_clips_key] = video_clips_data
        storage.write(df)
        self.logger.info(f"Filtered video clips and saved to {output_key}")

        return output_key
