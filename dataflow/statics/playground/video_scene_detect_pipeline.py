"""
Video Scene Clip Generator Pipeline

This pipeline integrates the basic video processing workflow:
- Video info extraction (VideoInfoFilter)
- Scene detection (VideoSceneFilter)
- Clip metadata generation (VideoClipFilter)
- Video cutting and saving (VideoClipGenerator)
"""

from dataflow.core.Operator import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow import get_logger

from dataflow.operators.core_vision import VideoInfoFilter
from dataflow.operators.core_vision import VideoSceneFilter
from dataflow.operators.core_vision import VideoClipFilter
from dataflow.operators.core_vision import VideoClipGenerator


class VideoSceneClipGenerator(OperatorABC):
    """
    Video scene-based clip generation pipeline without quality filtering.
    """
    
    def __init__(self):
        """
        Initialize the VideoSceneClipGenerator operator with default parameters.
        """
        self.logger = get_logger()
        
        # Initialize sub-operators
        self.video_info_filter = VideoInfoFilter(
            backend="opencv",
            ext=False,
        )
        self.video_scene_filter = VideoSceneFilter(
            frame_skip=0,
            start_remove_sec=0.0,
            end_remove_sec=0.0,
            min_seconds=2.0,
            max_seconds=15.0,
            disable_parallel=True,
        )
        self.video_clip_filter = VideoClipFilter(
            frames_min=None,
            frames_max=None,
            fps_min=None,
            fps_max=None,
            resolution_max=None,
        )
        self.video_clip_generator = VideoClipGenerator(
            video_save_dir="./cache/video_clips",
        )
    
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "该算子基于场景检测进行视频切分和保存，不包含质量过滤功能。\n\n"
                "输入参数：\n"
                "  - input_video_key: 输入视频路径字段名 (默认: 'video')\n"
                "  - output_key: 输出视频路径字段名 (默认: 'video')\n"
                "输出参数：\n"
                "  - output_key: 切割后的视频片段路径\n"
                "功能特点：\n"
                "  - 自动提取视频信息（帧率、分辨率等）\n"
                "  - 基于场景检测智能分割视频\n"
                "  - 自动切割并保存视频片段\n"
            )
        elif lang == "en":
            return (
                "This operator performs video scene-based segmentation and saving without quality filtering.\n\n"
                "Input Parameters:\n"
                "  - input_video_key: Input video path field name (default: 'video')\n"
                "  - output_key: Output video path field name (default: 'video')\n"
                "Output Parameters:\n"
                "  - output_key: Path to cut video clips\n"
                "Features:\n"
                "  - Automatic video info extraction (FPS, resolution, etc.)\n"
                "  - Intelligent video segmentation based on scene detection\n"
                "  - Automatic cutting and saving of video clips\n"
            )
        else:
            return "VideoSceneClipGenerator processes videos through scene detection and clip generation."

    def run(
        self,
        storage: DataFlowStorage,
        input_video_key: str = "video",
        output_key: str = "video",
    ):
        """
        Execute the video scene clip generation pipeline.
        
        Args:
            storage: DataFlow storage object
            input_video_key: Input video path field name (default: 'video')
            output_key: Output video path field name (default: 'video')
            
        Returns:
            str: Output key name
        """
        
        # Step 1: Extract video info
        self.logger.info("\n[Step 1/4] Extracting video info...")
        self.video_info_filter.run(
            storage=storage.step(),
            input_video_key=input_video_key,
            output_key="video_info",
        )

        # Step 2: Detect video scenes
        self.logger.info("\n[Step 2/4] Detecting video scenes...")
        self.video_scene_filter.run(
            storage=storage.step(),
            input_video_key=input_video_key,
            video_info_key="video_info",
            output_key="video_scene",
        )

        # Step 3: Generate clip metadata
        self.logger.info("\n[Step 3/4] Generating clip metadata...")
        self.video_clip_filter.run(
            storage=storage.step(),
            input_video_key=input_video_key,
            video_info_key="video_info",
            video_scene_key="video_scene",
            output_key="video_clip",
        )

        # Step 4: Cut and save video clips
        self.logger.info("\n[Step 4/4] Cutting and saving video clips...")
        self.video_clip_generator.run(
            storage=storage.step(),
            video_clips_key="video_clip",
            output_key=output_key,
        )
        
        return output_key


if __name__ == "__main__":
    # Test the operator
    from dataflow.utils.storage import FileStorage
    
    storage = FileStorage(
        first_entry_file_name="./dataflow/example/video_split/sample_data.json",
        cache_path="./cache",
        file_name_prefix="video_scene_detect",
        cache_type="json",
    )
    
    generator = VideoSceneClipGenerator()
    
    generator.run(
        storage=storage,
        input_video_key="video",
        output_key="video",
    )

