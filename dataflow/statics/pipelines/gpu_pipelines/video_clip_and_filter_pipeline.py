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
    
    def __init__(
        self,
        # VideoInfoFilter parameters
        backend: str = "opencv",
        ext: bool = False,
        
        # VideoSceneFilter parameters
        frame_skip: int = 0,
        start_remove_sec: float = 0.0,
        end_remove_sec: float = 0.0,
        min_seconds: float = 2.0,
        max_seconds: float = 15.0,
        
        # VideoFrameFilter parameters
        frame_output_dir: str = "./cache/extract_frames",
        
        # VideoAestheticEvaluator parameters
        clip_model: str = "/path/to/ViT-L-14.pt",#from https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt
        mlp_checkpoint: str = "/path/to/sac+logos+ava1-l14-linearMSE.pth",#from https://github.com/christophschuhmann/improved-aesthetic-predictor
        
        # VideoOCREvaluator parameters
        det_model_dir: str = None,
        rec_model_dir: str = None,
        
        # Filter parameters
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
        
        # VideoClipGenerator parameters
        video_save_dir: str = "./cache/video_clips",
    ):
        """
        Initialize the VideoFilteredClipGenerator operator.
        
        Args:
            backend: Video backend for info extraction (opencv, torchvision, av)
            ext: Whether to filter non-existent files
            frame_skip: Frame skip for scene detection
            start_remove_sec: Seconds to remove from start of each scene
            end_remove_sec: Seconds to remove from end of each scene
            min_seconds: Minimum scene duration
            max_seconds: Maximum scene duration
            frame_output_dir: Directory to save extracted frames
            clip_model: Path to CLIP model for aesthetic scoring
            mlp_checkpoint: Path to MLP checkpoint for aesthetic scoring
            det_model_dir: Path to PaddleOCR detection model directory
            rec_model_dir: Path to PaddleOCR recognition model directory
            frames_min: Minimum number of frames filter (applied in VideoClipFilter)
            frames_max: Maximum number of frames filter (applied in VideoClipFilter)
            fps_min: Minimum FPS filter (applied in VideoClipFilter)
            fps_max: Maximum FPS filter (applied in VideoClipFilter)
            resolution_max: Maximum resolution in pixels (width*height) filter (applied in VideoClipFilter)
            aes_min: Minimum aesthetic score filter (applied in VideoAestheticFilter)
            ocr_min: Minimum OCR score filter (applied in VideoOCRFilter)
            ocr_max: Maximum OCR score filter (applied in VideoOCRFilter)
            lum_min: Minimum luminance filter (applied in VideoLuminanceFilter)
            lum_max: Maximum luminance filter (applied in VideoLuminanceFilter)
            video_save_dir: Directory to save cut video clips
        """
        self.logger = get_logger()
        
        # Initialize all sub-operators
        self.video_info_filter = VideoInfoFilter(
            backend=backend,
            ext=ext,
        )
        self.video_scene_filter = VideoSceneFilter(
            frame_skip=frame_skip,
            start_remove_sec=start_remove_sec,
            end_remove_sec=end_remove_sec,
            min_seconds=min_seconds,
            max_seconds=max_seconds,
            disable_parallel=True,
        )
        self.video_clip_filter = VideoClipFilter(
            frames_min=frames_min,
            frames_max=frames_max,
            fps_min=fps_min,
            fps_max=fps_max,
            resolution_max=resolution_max,
        )
        self.video_frame_filter = VideoFrameFilter(
            output_dir=frame_output_dir,
        )
        self.video_aesthetic_filter = VideoAestheticFilter(
            figure_root=frame_output_dir,
            clip_model=clip_model,
            mlp_checkpoint=mlp_checkpoint,
            aes_min=aes_min,
        )
        self.video_luminance_filter = VideoLuminanceFilter(
            figure_root=frame_output_dir,
            lum_min=lum_min,
            lum_max=lum_max,
        )
        self.video_ocr_filter = VideoOCRFilter(
            figure_root=frame_output_dir,
            det_model_dir=det_model_dir,
            rec_model_dir=rec_model_dir,
            ocr_min=ocr_min,
            ocr_max=ocr_max,
        )
        self.video_clip_generator = VideoClipGenerator(
            video_save_dir=video_save_dir,
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
        self.logger.info("="*60)
        self.logger.info("Running VideoFilteredClipGenerator Pipeline...")
        self.logger.info("="*60)
        
        # Step 1: Extract video info
        self.logger.info("\n[Step 1/8] Extracting video info...")
        self.video_info_filter.run(
            storage=storage.step(),
            input_video_key=input_video_key,
            output_key="video_info",
        )
        self.logger.info("✓ Video info extracted")

        # Step 2: Detect video scenes
        self.logger.info("\n[Step 2/8] Detecting video scenes...")
        self.video_scene_filter.run(
            storage=storage.step(),
            input_video_key=input_video_key,
            video_info_key="video_info",
            output_key="video_scene",
        )
        self.logger.info("✓ Scene detection complete")

        # Step 3: Generate clip metadata
        self.logger.info("\n[Step 3/8] Generating clip metadata...")
        self.video_clip_filter.run(
            storage=storage.step(),
            input_video_key=input_video_key,
            video_info_key="video_info",
            video_scene_key="video_scene",
            output_key="video_clip",
        )
        self.logger.info("✓ Clip metadata generated")

        # Step 4: Extract frames from clips
        self.logger.info("\n[Step 4/8] Extracting frames from clips...")
        self.video_frame_filter.run(
            storage=storage.step(),
            input_video_key=input_video_key,
            video_info_key="video_info",
            video_clips_key="video_clip",
            output_key="video_frame_export",
        )
        self.logger.info("✓ Frame extraction complete")

        # Step 5: Compute aesthetic scores and filter
        self.logger.info("\n[Step 5/8] Computing aesthetic scores and filtering...")
        self.video_aesthetic_filter.run(
            storage=storage.step(),
            input_video_key=input_video_key,
            video_clips_key="video_clip",
            output_key="video_clip",
        )
        self.logger.info("✓ Aesthetic scoring and filtering complete")

        # Step 6: Compute luminance statistics and filter
        self.logger.info("\n[Step 6/8] Computing luminance statistics and filtering...")
        self.video_luminance_filter.run(
            storage=storage.step(),
            input_video_key=input_video_key,
            video_clips_key="video_clip",
            output_key="video_clip",
        )
        self.logger.info("✓ Luminance evaluation and filtering complete")

        # Step 7: Compute OCR scores and filter
        self.logger.info("\n[Step 7/8] Computing OCR scores and filtering...")
        self.video_ocr_filter.run(
            storage=storage.step(),
            input_video_key=input_video_key,
            video_clips_key="video_clip",
            output_key="video_clip",
        )
        self.logger.info("✓ OCR analysis and filtering complete")

        # Step 8: Cut and save video clips
        self.logger.info("\n[Step 8/8] Cutting and saving video clips...")
        self.video_clip_generator.run(
            storage=storage.step(),
            video_clips_key="video_clip",
            output_key=output_key,
        )
        
        self.logger.info("="*60)
        self.logger.info("✓ Pipeline complete!")
        self.logger.info("="*60)
        
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
    
    generator = VideoFilteredClipGenerator(
        clip_model="ViT-L-14.pt",
        mlp_checkpoint="sac+logos+ava1-l14-linearMSE.pth",
        det_model_dir="PP-OCRv5_server_det",
        rec_model_dir="PP-OCRv5_server_rec",
        aes_min=4,
        ocr_max=0.3,
        lum_min=20,
        lum_max=140,
        video_save_dir="./cache/video_clips",
    )
    
    generator.run(
        storage=storage,
        input_video_key="video",
        output_key="video",
    )