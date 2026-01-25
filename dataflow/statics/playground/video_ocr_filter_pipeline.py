"""
Video OCR Pipeline (Whole Video)

This pipeline separates OCR detection from the clip-based workflow:
- Directly runs OCR on the input video (no clip splitting required)
- Samples a few frames from the video and computes an OCR text-area ratio
"""

from dataflow.core.Operator import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow import get_logger

from dataflow.operators.core_vision import VideoOCRFilter


class VideoOCRPipeline(OperatorABC):
    """
    Direct OCR detection for input videos (no clip metadata needed).
    """

    def __init__(self):
        self.logger = get_logger()
        self.video_ocr = VideoOCRFilter(
            output_key="video_ocr",
            load_num=3,                    # sample 3 frames by default
            batch_size=8,
            num_workers=2,
            det_model_dir="/mnt/shared-storage-user/mineru2-shared/zqt/zqt2/DataFlow-MM/models/PP-OCRv5_server_det",
            rec_model_dir="/mnt/shared-storage-user/mineru2-shared/zqt/zqt2/DataFlow-MM/models/PP-OCRv5_server_rec",
            ocr_min=None,
            ocr_max=0.3,
        )

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "该 pipeline 直接对输入视频进行 OCR 检测（不依赖 clip 切分/抽帧目录），"
                "默认采样少量帧并输出行级 OCR 分数到 `video_ocr` 字段。"
            )
        return "Directly run OCR on input videos (no clip splitting required)."

    def run(
        self,
        storage: DataFlowStorage,
        input_video_key: str = "video",
        output_key: str = "video_ocr",
    ):
        self.logger.info("\n[Step 1/1] OCR detecting on whole video...")
        self.video_ocr.run(
            storage=storage.step(),
            input_video_key=input_video_key,
            video_clips_key=None,
            output_key=output_key,
        )
        return output_key


if __name__ == "__main__":
    from dataflow.utils.storage import FileStorage

    storage = FileStorage(
        first_entry_file_name="./dataflow/example/video_ocr/sample_data.json",
        cache_path="./cache",
        file_name_prefix="video_ocr_filter",
        cache_type="json",
    )

    pipe = VideoOCRPipeline()
    pipe.run(storage=storage, input_video_key="video", output_key="video_ocr")




