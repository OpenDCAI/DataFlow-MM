from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # === Generate ===
    from .generate.prompted_image_edit_generator import PromptedImageEditGenerator
    from .generate.prompted_image_generator import PromptedImageGenerator
    from .generate.prompted_vqa_generator import PromptedVQAGenerator
    from .generate.video_clip_generator import VideoClipGenerator
    from .generate.video_caption_to_qa_generator import VideoCaptionToQAGenerator
    from .generate.video_video_to_caption_generator import VideoToCaptionGenerator

    # === Filter ===
    from .filter.video_clip_filter import VideoClipFilter
    from .filter.video_frame_filter import VideoFrameFilter
    from .filter.video_info_filter import VideoInfoFilter
    from .filter.video_scene_filter import VideoSceneFilter
    from .filter.video_score_filter import VideoScoreFilter

    # === Eval ===
    from .eval.video_aesthetic_evaluator import VideoAestheticEvaluator
    from .eval.video_luminance_evaluator import VideoLuminanceEvaluator
    from .eval.video_ocr_evaluator import VideoOCREvaluator

else:
    import sys
    from dataflow.utils.registry import LazyLoader, generate_import_structure_from_type_checking
    cur_path = "dataflow/operators/core_vision/"
    _import_structure = generate_import_structure_from_type_checking(__file__, cur_path)
    sys.modules[__name__] = LazyLoader(__name__, "dataflow/operators/core_vision/", _import_structure)
