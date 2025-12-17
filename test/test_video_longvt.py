"""
Video Caption Generator Operator

This operator integrates video processing pipeline with caption generation:
- Video info extraction (VideoInfoFilter)
- Scene detection (VideoSceneFilter)
- Video caption generation (VideoToCaptionGenerator)
"""

from dataflow.core.Operator import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage, FileStorage

from dataflow.operators.core_vision import VideoInfoFilter
from dataflow.operators.core_vision import VideoSceneFilter
from dataflow.operators.core_vision import VideoClipFilter
from dataflow.operators.core_vision import VideoClipGenerator
from dataflow.operators.core_vision import VideoToCaptionGenerator
from dataflow.serving import LocalModelVLMServing_vllm

# LongVT prompt template
VIDEO_CAPTION_PROMPT = (
    "You are a helpful assistant that summarizes the content of a video. "
    "Please provide a detailed description of the video. "
    "When describing the video, please include the following information: \n"
    "1. The main events in the video. \n"
    "2. The main characters in the video. \n"
    "3. The main locations in the video. \n"
    "4. The main objects in the video. \n"
    "5. The main actions in the video. \n"
    "6. The main emotions in the video. \n"
    "\nMake sure to describe the video in a way that is easy to understand and follow. "
    "Please include as much detail as possible and do not miss any information. "
)



class VideoCaptionGenerator(OperatorABC):
    """
    Video caption generation pipeline operator that integrates video processing and caption generation.
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
        use_adaptive_detector: bool = False,
        overlap: bool = False,
        
        # VLM Local Model parameters
        hf_model_name_or_path: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        model_cache_dir: str = "./dataflow_cache",
        vllm_tensor_parallel_size: int = 1,
        vllm_temperature: float = 0.7,
        vllm_top_p: float = 0.9,
        vllm_max_tokens: int = 2048,
        vllm_max_model_len: int = 51200,
        vllm_gpu_memory_utilization: float = 0.9,
        
        # VideoClipGenerator parameters
        video_save_dir: str = "./cache/video_clips",
    ):
        """
        Initialize the VideoCaptionGenerator operator.
        
        Args:
            backend: Video backend for info extraction (opencv, torchvision, av)
            ext: Whether to filter non-existent files
            frame_skip: Frame skip for scene detection
            start_remove_sec: Seconds to remove from start of each scene
            end_remove_sec: Seconds to remove from end of each scene
            min_seconds: Minimum scene duration
            max_seconds: Maximum scene duration (equivalent to 10s in clip_caption.py)
            use_adaptive_detector: Whether to use AdaptiveDetector in scene detection
            overlap: If True, use overlap splitting strategy similar to clip_caption.py
            hf_model_name_or_path: HuggingFace model name or path
            model_cache_dir: Directory to cache model files
            vllm_tensor_parallel_size: Tensor parallel size for vLLM
            vllm_temperature: Sampling temperature
            vllm_top_p: Top-p sampling parameter
            vllm_max_tokens: Maximum number of tokens to generate
            vllm_max_model_len: Maximum model context length
            vllm_gpu_memory_utilization: GPU memory utilization ratio
            video_save_dir: Directory to save cut video clips
        """
        self.logger = get_logger()
        
        # Initialize sub-operators
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
            use_adaptive_detector=use_adaptive_detector,
            overlap=overlap,
        )
        
        # Initialize clip processing operators
        self.video_clip_filter = VideoClipFilter()
        self.video_clip_generator = VideoClipGenerator(
            video_save_dir=video_save_dir,
        )
        
        # Initialize VLM serving with local model
        self.vlm_serving = LocalModelVLMServing_vllm(
            hf_model_name_or_path=hf_model_name_or_path,
            hf_cache_dir=model_cache_dir,
            vllm_tensor_parallel_size=vllm_tensor_parallel_size,
            vllm_temperature=vllm_temperature,
            vllm_top_p=vllm_top_p,
            vllm_max_tokens=vllm_max_tokens,
            vllm_max_model_len=vllm_max_model_len,
            vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
        )
        
        # Initialize caption generator with LongVT prompt template
        self.video_to_caption_generator = VideoToCaptionGenerator(
            vlm_serving=self.vlm_serving,
            prompt_template=VIDEO_CAPTION_PROMPT,
        )
    
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "该算子整合了视频处理和字幕生成流水线，包括信息提取、场景检测、片段切割和字幕生成。\n\n"
                "输入参数：\n"
                "  - input_video_key: 输入视频路径字段名 (默认: 'video')\n"
                "  - output_key: 输出字幕字段名 (默认: 'caption')\n"
                "输出参数：\n"
                "  - output_key: 生成的视频字幕（每个 clip 一个）\n"
                "功能特点：\n"
                "  - 自动提取视频信息（帧率、分辨率等）\n"
                "  - 基于场景检测智能分割视频\n"
                "  - 将视频切割成多个片段\n"
                "  - 使用视觉语言模型为每个片段生成字幕\n"
            )
        elif lang == "en":
            return (
                "This operator integrates video processing and caption generation pipeline, including "
                "info extraction, scene detection, clip cutting, and caption generation.\n\n"
                "Input Parameters:\n"
                "  - input_video_key: Input video path field name (default: 'video')\n"
                "  - output_key: Output caption field name (default: 'caption')\n"
                "Output Parameters:\n"
                "  - output_key: Generated video captions (one per clip)\n"
                "Features:\n"
                "  - Automatic video info extraction (FPS, resolution, etc.)\n"
                "  - Intelligent video segmentation based on scene detection\n"
                "  - Cut videos into multiple clips\n"
                "  - Vision-Language Model based caption generation for each clip\n"
            )
        else:
            return "VideoCaptionGenerator processes videos and generates captions using VLM."

    def run(
        self,
        storage: DataFlowStorage,
        input_video_key: str = "video",
        input_conversation_key: str = "conversation",
        output_key: str = "caption",
    ):
        """
        Execute the video caption generation pipeline.
        
        Args:
            storage: DataFlow storage object
            input_video_key: Input video path field name (default: 'video')
            input_conversation_key: Input conversation field name (default: 'conversation')
            output_key: Output caption field name (default: 'caption')
            
        Returns:
            str: Output key name
        """
        self.logger.info("="*60)
        self.logger.info("Running VideoCaptionGenerator Pipeline...")
        self.logger.info("="*60)
        
        # Step 1: Extract video info
        self.logger.info("\n[Step 1/5] Extracting video info...")
        self.video_info_filter.run(
            storage=storage.step(),
            input_video_key=input_video_key,
            output_key="video_info",
        )
        self.logger.info("✓ Video info extracted")

        # Step 2: Detect video scenes
        self.logger.info("\n[Step 2/5] Detecting video scenes...")
        
        self.video_scene_filter.run(
            storage=storage.step(),
            input_video_key=input_video_key,
            video_info_key="video_info",
            output_key="video_scene",
        )
        self.logger.info("✓ Scene detection complete")
        
        # Step 3: Generate clip metadata
        self.logger.info("\n[Step 3/5] Generating clip metadata...")
        self.video_clip_filter.run(
            storage=storage.step(),
            input_video_key=input_video_key,
            video_info_key="video_info",
            video_scene_key="video_scene",
            output_key="video_clip",
        )
        self.logger.info("✓ Clip metadata generated")

        # Step 4: Cut and save video clips
        self.logger.info("\n[Step 4/5] Cutting and saving video clips...")
        self.video_clip_generator.run(
            storage=storage.step(),
            video_clips_key="video_clip",
            output_key="video",
        )
        self.logger.info("✓ Video clips cut and saved")

        # Step 5: Generate captions for each clip
        self.logger.info("\n[Step 5/5] Generating captions for each clip...")
        self.video_to_caption_generator.run(
            storage=storage.step(),
            input_image_key="image",
            input_video_key="video",
            input_conversation_key=input_conversation_key,
            output_key=output_key,
        )
        self.logger.info("✓ Caption generation complete")
        
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
        file_name_prefix="video_longvt",
        cache_type="json",
    )
    
    generator = VideoCaptionGenerator(
        backend="opencv",
        ext=False,
        frame_skip=0,
        start_remove_sec=0.0,
        end_remove_sec=0.0,
        min_seconds=0.0,
        max_seconds=10.0,
        use_adaptive_detector=False,
        overlap=True,  # Set to True to enable overlap splitting strategy
        hf_model_name_or_path="/mnt/shared-storage-user/mineru2-shared/zqt/zqt2/models/Qwen/Qwen2.5-VL-7B-Instruct",
        model_cache_dir="./dataflow_cache",
        vllm_tensor_parallel_size=1,
        vllm_temperature=0.7,
        vllm_top_p=0.9,
        vllm_max_tokens=2048,
        vllm_max_model_len=102400,
        vllm_gpu_memory_utilization=0.9,
        video_save_dir="./cache/video_clips",
    )
    
    generator.run(
        storage=storage,
        input_video_key="video",
        input_conversation_key="conversation",
        output_key="caption",
    )