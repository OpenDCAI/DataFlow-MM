"""
Video Luminance Pipeline (Whole Video)

This pipeline separates luminance evaluation from the clip-based workflow:
- Directly computes luminance statistics on the input video (no clip splitting required)
- Samples a few frames from the video and computes luminance_mean/min/max
"""

from dataflow.core.Operator import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow import get_logger

from dataflow.operators.core_vision import VideoLuminanceFilter


class VideoLuminancePipeline(OperatorABC):
    """
    Direct luminance evaluation for input videos (no clip metadata needed).
    """

    def __init__(self):
        self.logger = get_logger()
        self.luminance = VideoLuminanceFilter(
            # video_clips_key defaults to None (whole-video mode)
            output_key="video_luminance",
            load_num=3,
            batch_size=16,
            num_workers=2,
            lum_min=20,
            lum_max=140,
        )

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "该 pipeline 直接对输入视频进行亮度统计评估（不依赖 clip 切分/抽帧目录），"
                "默认采样少量帧并输出行级亮度统计到 `video_luminance` 字段。"
            )
        return "Directly compute luminance statistics on input videos (no clip splitting required)."

    def run(
        self,
        storage: DataFlowStorage,
        input_video_key: str = "video",
        output_key: str = "video_luminance",
    ):
        self.logger.info("\n[Step 1/1] Luminance evaluation on whole video...")
        self.luminance.run(
            storage=storage.step(),
            input_video_key=input_video_key,
            output_key=output_key,
        )
        return output_key


if __name__ == "__main__":
    from dataflow.utils.storage import FileStorage

    storage = FileStorage(
        first_entry_file_name="./dataflow/example/video_luminance/sample_data.json",
        cache_path="./cache",
        file_name_prefix="video_luminance_filter",
        cache_type="json",
    )

    pipe = VideoLuminancePipeline()
    pipe.run(storage=storage, input_video_key="video", output_key="video_luminance")


