from dataflow.operators.core_vision import PromptedVQAGenerator
from dataflow.operators.conversations import Conversation2Message
from dataflow.serving import LocalModelVLMServing_vllm
from dataflow.utils.storage import FileStorage
from dataflow.operators.core_vision import VideoInfoFilter
from dataflow.operators.core_vision import VideoSceneFilter
from dataflow.operators.core_vision import VideoClipFilter
from dataflow.operators.core_vision import VideoFrameFilter
from dataflow.operators.core_vision import VideoAestheticEvaluator
from dataflow.operators.core_vision import VideoLuminanceEvaluator
from dataflow.operators.core_vision import VideoOCREvaluator
from dataflow.operators.core_vision import VideoScoreFilter
from dataflow.operators.core_vision import VideoClipGenerator

class VideoInfo_Filter():
    def __init__(self):
        self.storage = FileStorage(
            first_entry_file_name="./dataflow/example/video/sample_data.json",
            cache_path="./cache",
            file_name_prefix="video_filter",
            cache_type="json",
        )
        self.model_cache_dir = './dataflow_cache'

        self.video_info_filter = VideoInfoFilter()
        self.video_scene_filter = VideoSceneFilter()
        self.video_clip_filter = VideoClipFilter()
        self.video_frame_filter = VideoFrameFilter()
        self.video_aesthetic_evaluator = VideoAestheticEvaluator(
            figure_root = "./cache/extract_frames",
            clip_model = "ViT-L-14.pt",
            mlp_checkpoint = "aesthetic.pth",
        )
        self.video_luminance_evaluator = VideoLuminanceEvaluator(
            figure_root="./cache/extract_frames",
        )

        self.video_ocr_evaluator = VideoOCREvaluator(
            figure_root="./cache/extract_frames",
        )

        self.video_score_filter = VideoScoreFilter(
            frames_min=None,
            frames_max=None,
            fps_min=None,
            fps_max=None,
            resolution_max=None,
            aes_min=4,
            ocr_min=None,
            ocr_max=0.3,
            lum_min=20,
            lum_max=140,
            motion_min=2,
            motion_max=14,
            flow_min=None,
            flow_max=None,
            blur_max=None,
            strict_mode=False,
            seed=42
        )
        self.video_clip_generator = VideoClipGenerator(
            video_save_dir = "./cache/video_clips",
        )

    def forward(self):
        # Initial filters
        # self.format_converter.run(
        #     storage= self.storage.step(),
        #     input_conversation_key="conversation",
        #     output_message_key="messages",
        # )

        self.video_info_filter.run(
            storage = self.storage.step(),
            input_video_key="video",
            output_key="video_info",
        )

        self.video_scene_filter.run(
            storage = self.storage.step(),
            input_video_key="video",
            video_info_key="video_info",
            output_key="video_scene",
        )

        self.video_clip_filter.run(
            storage = self.storage.step(),
            input_video_key="video",
            video_info_key="video_info",
            video_scene_key="video_scene",
            output_key="video_clip",
        )

        self.video_frame_filter.run(
            storage = self.storage.step(),
            input_video_key="video",
            video_info_key="video_info",
            video_clips_key="video_clip",
            output_dir="./cache/extract_frames",
            output_key="video_frame_export",
        )

        self.video_aesthetic_evaluator.run(
            storage = self.storage.step(),
            input_video_key="video",
            video_clips_key="video_clip",
            output_key="video_clip",
        )

        self.video_luminance_evaluator.run(
            storage = self.storage.step(),
            input_video_key="video",
            video_clips_key="video_clip",
            output_key="video_clip",
        )

        self.video_ocr_evaluator.run(
            storage = self.storage.step(),
            input_video_key="video",
            video_clips_key="video_clip",
            output_key="video_clip",
        )

        self.video_score_filter.run(
            storage = self.storage.step(),
            input_video_key="video",
            video_clips_key="video_clip",
            output_key="video_clip",
        )

        self.video_clip_generator.run(
            storage = self.storage.step(),
            video_clips_key="video_clip",
            output_key="video_cut",
        )

if __name__ == "__main__":
    # This is the entry point for the pipeline
    model = VideoInfo_Filter()
    model.forward()
