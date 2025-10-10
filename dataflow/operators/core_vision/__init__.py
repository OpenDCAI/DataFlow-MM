from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # === Generate ===
    from .generate.prompted_image_edit_generator import PromptedImageEditGenerator
    from .generate.image_caption_generator import ImageCaptionGenerate
    from .generate.image_qa_generator import ImageQAGenerate
    from .generate.multimodal_math_generator import MultimodalMathGenerate
    from .generate.personalized_qa_generator import PersQAGenerate
    from .generate.vision_mct_reasoning_sft_generator import VisionMCTSReasoningSFTGenerate
    from .generate.image_scale_caption_generator import ImageScaleCaptionGenerate, ImageScaleCaptionGenerateConfig
    from .filter.aesthetic_filter import AestheticFilter
    from .filter.cat_filter import CatFilter
    from .filter.clip_filter import ClipFilter
    from .filter.complexity_filter import ComplexityFilter
    from .filter.consistency_filter import ConsistencyFilter
    from .filter.diversity_filter import DiversityFilter
    from .filter.sensitive_filter import SensitiveFilter
    from .refine.vision_seg_cutout_refine import VisionSegCutoutRefine
    from .eval.code.code_evaluator import CodeStabilityEvaluator, CodeCorrectnessEvaluator, CodeMaintainabilityEvaluator
    from .eval.image.image_evaluator import EvalImageGenerationGenerator
    from .eval.image_text.image_text_evaluator import VQAScoreEvaluator, LongCLIPEvaluator, CLIPEvaluator

    from .filter.rule_base_filter import RuleBaseFilter
    from .filter.deduplication_filter import DeduplicateFilter
    from .filter.knn_similarity_filter import KNNSimilarityFilter
    from .filter.clipscore_filter import CLIPScoreFilter
    from .filter.datatailor_filter import DataTailorFilter
    from .filter.vision_dependent_filter import VisionDependentFilter
    from .filter.failrate_filter import FailRateFilter

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
    from .filter.video_motion_score_filter import VideoMotionScoreFilter
    from .filter.video_resolution_filter import VideoResolutionFilter

    # === Eval ===
    from .eval.video_aesthetic_evaluator import VideoAestheticEvaluator
    from .eval.video_luminance_evaluator import VideoLuminanceEvaluator
    from .eval.video_ocr_evaluator import VideoOCREvaluator
    from .eval.emscore_evaluator import EMScoreEval

else:
    import sys
    from dataflow.utils.registry import LazyLoader, generate_import_structure_from_type_checking
    cur_path = "dataflow/operators/core_vision/"
    _import_structure = generate_import_structure_from_type_checking(__file__, cur_path)
    sys.modules[__name__] = LazyLoader(__name__, "dataflow/operators/core_vision/", _import_structure)
