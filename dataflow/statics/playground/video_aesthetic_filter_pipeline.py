"""
Video Aesthetic Pipeline (Whole Video)

This pipeline separates aesthetic scoring from the clip-based workflow:
- Directly runs aesthetic scoring on the input video (no clip splitting required)
- Samples a few frames from the video and predicts an aesthetic score
"""

from dataflow.core.Operator import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow import get_logger

from dataflow.operators.core_vision import VideoAestheticFilter


class VideoAestheticPipeline(OperatorABC):
    """
    Direct aesthetic scoring for input videos (no clip metadata needed).
    """

    def __init__(self):
        self.logger = get_logger()
        self.aesthetic = VideoAestheticFilter(
            # video_clips_key defaults to None (whole-video mode)
            output_key="video_aesthetic",
            load_num=3,
            batch_size=16,
            num_workers=2,
            clip_model="/mnt/shared-storage-user/mineru2-shared/zqt/zqt2/DataFlow-MM/models/ViT-L-14.pt",
            mlp_checkpoint="/mnt/shared-storage-user/mineru2-shared/zqt/zqt2/DataFlow-MM/models/sac+logos+ava1-l14-linearMSE.pth",
            aes_min=0.1,  # set threshold if you want {"filtered": ...}
        )

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "该 pipeline 直接对输入视频进行美学评估（不依赖 clip 切分/抽帧目录），"
                "默认采样少量帧并输出行级美学分数到 `video_aesthetic` 字段。"
            )
        return "Directly run aesthetic scoring on input videos (no clip splitting required)."

    def run(
        self,
        storage: DataFlowStorage,
        input_video_key: str = "video",
        output_key: str = "video_aesthetic",
    ):
        self.logger.info("\n[Step 1/1] Aesthetic scoring on whole video...")
        self.aesthetic.run(
            storage=storage.step(),
            input_video_key=input_video_key,
            output_key=output_key,
        )
        return output_key


if __name__ == "__main__":
    from dataflow.utils.storage import FileStorage

    storage = FileStorage(
        first_entry_file_name="./dataflow/example/video_aesthetic/sample_data.json",
        cache_path="./cache",
        file_name_prefix="video_aesthetic_filter",
        cache_type="json",
    )

    pipe = VideoAestheticPipeline()
    pipe.run(storage=storage, input_video_key="video", output_key="video_aesthetic")


