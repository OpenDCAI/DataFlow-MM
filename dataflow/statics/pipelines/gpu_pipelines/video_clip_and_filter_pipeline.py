"""
Video Filtered Clip Generator Operator

This operator integrates the complete video processing pipeline:
- Video info extraction (VideoInfoFilter)
- Scene detection (VideoSceneFilter)
- Clip metadata generation and basic filtering (VideoClipFilter)
- Frame extraction (VideoFrameFilter)
- Aesthetic scoring and filtering (VideoAestheticFilter)
- Luminance analysis and filtering (VideoLuminanceFilter)
- OCR analysis and filtering (VideoOCRFilter)
- Video cutting and saving (VideoClipGenerator)
"""

from dataflow.core.Operator import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage, FileStorage

from dataflow.operators.core_vision import VideoInfoFilter
from dataflow.operators.core_vision import VideoSceneFilter
from dataflow.operators.core_vision import VideoClipFilter
from dataflow.operators.core_vision import VideoFrameFilter
from dataflow.operators.core_vision import VideoAestheticFilter
from dataflow.operators.core_vision import VideoLuminanceFilter
from dataflow.operators.core_vision import VideoOCRFilter
from dataflow.operators.core_vision import VideoClipGenerator



class VideoFilteredClipGenerator(OperatorABC):
    """
    Complete video processing pipeline operator that integrates all filtering and generation steps.
    """
    
    def __init__(self):
        """
        Initialize the VideoFilteredClipGenerator operator with default parameters.
        """
        self.logger = get_logger()
        
        # Initialize all sub-operators with default parameters
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
        self.video_frame_filter = VideoFrameFilter(
            output_dir="./cache/extract_frames",
        )
        self.video_aesthetic_filter = VideoAestheticFilter(
            figure_root="./cache/extract_frames",
            clip_model="/path/to/ViT-L-14.pt",  # from https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt
            mlp_checkpoint="/path/to/sac+logos+ava1-l14-linearMSE.pth",  # from https://github.com/christophschuhmann/improved-aesthetic-predictor
            aes_min=4,
        )
        self.video_luminance_filter = VideoLuminanceFilter(
            figure_root="./cache/extract_frames",
            lum_min=20,
            lum_max=140,
        )
        self.video_ocr_filter = VideoOCRFilter(
            figure_root="./cache/extract_frames",
            det_model_dir="/path/to/PP-OCRv5_server_det",
            rec_model_dir="/path/to/PP-OCRv5_server_rec",
            ocr_min=None,
            ocr_max=0.3,
        )
        self.video_clip_generator = VideoClipGenerator(
            video_save_dir="./cache/video_clips",
        )
    
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "该算子整合了完整的视频处理流水线，包括信息提取、场景检测、片段生成、"
                "关键帧抽取、美学评分和过滤、亮度评估和过滤、OCR分析和过滤和视频切割保存。\n\n"
                "输入参数：\n"
                "  - input_video_key: 输入视频路径字段名 (默认: 'video')\n"
                "  - output_key: 输出视频路径字段名 (默认: 'video')\n"
                "输出参数：\n"
                "  - output_key: 切割后的视频片段路径\n"
                "功能特点：\n"
                "  - 自动提取视频信息（帧率、分辨率等）\n"
                "  - 基于场景检测智能分割视频\n"
                "  - 基础属性提前过滤（帧数、FPS、分辨率）\n"
                "  - 多维度质量评估和过滤（美学、亮度、OCR）\n"
                "  - 可配置的质量过滤条件\n"
                "  - 自动切割并保存高质量片段\n"
            )
        elif lang == "en":
            return (
                "This operator integrates the complete video processing pipeline, including "
                "info extraction, scene detection, clip generation, frame extraction, "
                "aesthetic scoring, luminance evaluation, OCR analysis, score filtering, and video cutting.\n\n"
                "Input Parameters:\n"
                "  - input_video_key: Input video path field name (default: 'video')\n"
                "  - output_key: Output video path field name (default: 'video')\n"
                "Output Parameters:\n"
                "  - output_key: Path to cut video clips\n"
                "Features:\n"
                "  - Automatic video info extraction (FPS, resolution, etc.)\n"
                "  - Intelligent video segmentation based on scene detection\n"
                "  - Multi-dimensional quality assessment (aesthetic, luminance, OCR)\n"
                "  - Configurable quality filtering criteria\n"
                "  - Automatic cutting and saving of high-quality clips\n"
            )
        else:
            return "VideoFilteredClipGenerator processes videos through a complete pipeline with quality filtering."

    def run(
        self,
        storage: DataFlowStorage,
        input_video_key: str = "video",
        output_key: str = "video",
    ):
        """
        Execute the complete video processing pipeline.
        
        Args:
            storage: DataFlow storage object
            input_video_key: Input video path field name (default: 'video')
            output_key: Output video path field name (default: 'video')
            
        Returns:
            str: Output key name
        """
        
        # Step 1: Extract video info
        self.logger.info("\n[Step 1/8] Extracting video info...")
        self.video_info_filter.run(
            storage=storage.step(),
            input_video_key=input_video_key,
            output_key="video_info",
        )

        # Step 2: Detect video scenes
        self.logger.info("\n[Step 2/8] Detecting video scenes...")
        self.video_scene_filter.run(
            storage=storage.step(),
            input_video_key=input_video_key,
            video_info_key="video_info",
            output_key="video_scene",
        )

        # Step 3: Generate clip metadata
        self.logger.info("\n[Step 3/8] Generating clip metadata...")
        self.video_clip_filter.run(
            storage=storage.step(),
            input_video_key=input_video_key,
            video_info_key="video_info",
            video_scene_key="video_scene",
            output_key="video_clip",
        )

        # Step 4: Extract frames from clips
        self.logger.info("\n[Step 4/8] Extracting frames from clips...")
        self.video_frame_filter.run(
            storage=storage.step(),
            input_video_key=input_video_key,
            video_info_key="video_info",
            video_clips_key="video_clip",
            output_key="video_frame_export",
        )

        # Step 5: Compute aesthetic scores and filter
        self.logger.info("\n[Step 5/8] Computing aesthetic scores and filtering...")
        self.video_aesthetic_filter.run(
            storage=storage.step(),
            input_video_key=input_video_key,
            video_clips_key="video_clip",
            output_key="video_clip",
        )

        # Step 6: Compute luminance statistics and filter
        self.logger.info("\n[Step 6/8] Computing luminance statistics and filtering...")
        self.video_luminance_filter.run(
            storage=storage.step(),
            input_video_key=input_video_key,
            video_clips_key="video_clip",
            output_key="video_clip",
        )

        # Step 7: Compute OCR scores and filter
        self.logger.info("\n[Step 7/8] Computing OCR scores and filtering...")
        self.video_ocr_filter.run(
            storage=storage.step(),
            input_video_key=input_video_key,
            video_clips_key="video_clip",
            output_key="video_clip",
        )

        # Step 8: Cut and save video clips
        self.logger.info("\n[Step 8/8] Cutting and saving video clips...")
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
        file_name_prefix="video_filter",
        cache_type="json",
    )
    
    generator = VideoFilteredClipGenerator()
    
    generator.run(
        storage=storage,
        input_video_key="video",
        output_key="video",
    )