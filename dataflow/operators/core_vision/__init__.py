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
    from .generate.prompted_image_generator import PromptedImageGenerator
    from .generate.prompted_vqa_generator import PromptedVQAGenerator
    from .generate.video_clip_generator import VideoClipGenerator
    from .generate.video_caption_to_qa_generator import VideoCaptionToQAGenerator
    from .generate.video_video_to_caption_generator import VideoToCaptionGenerator
    from .generate.sk_vqa_generator import ImageSKVQAGenerate
    from .generate.image_caprl_mcq_generator import CapRLMCQGenerate, CapRLMCQConfig

    # === Filter ===
    from .filter.video_clip_filter import VideoClipFilter
    from .filter.video_frame_filter import VideoFrameFilter
    from .filter.video_info_filter import VideoInfoFilter
    from .filter.video_scene_filter import VideoSceneFilter
    from .filter.video_score_filter import VideoScoreFilter
    from .filter.video_motion_score_filter import VideoMotionScoreFilter
    from .filter.video_resolution_filter import VideoResolutionFilter
    from .filter.image_aesthetic_filter import ImageAestheticFilter
    from .filter.image_cat_filter import ImageCatFilter
    from .filter.image_clip_filter import ImageClipFilter
    from .filter.image_complexity_filter import ImageComplexityFilter
    from .filter.image_consistency_filter import ImageConsistencyFilter
    from .filter.image_diversity_filter import ImageDiversityFilter
    from .filter.image_sensitive_filter import ImageSensitiveFilter
    from .refine.vision_seg_cutout_refine import VisionSegCutoutRefine
    from .filter.rule_base_filter import RuleBaseFilter
    from .filter.image_deduplication_filter import ImageDeduplicateFilter
    from .filter.knn_similarity_filter import KNNSimilarityFilter
    from .filter.clipscore_filter import CLIPScoreFilter
    from .filter.datatailor_filter import DataTailorFilter
    from .filter.vision_dependent_filter import VisionDependentFilter
    from .filter.failrate_filter import FailRateFilter

    # === Eval ===
    from .eval.video_aesthetic_evaluator import VideoAestheticEvaluator
    from .eval.video_luminance_evaluator import VideoLuminanceEvaluator
    from .eval.video_ocr_evaluator import VideoOCREvaluator
    from .eval.emscore_evaluator import EMScoreEval
    from .eval.image.image_evaluator import EvalImageGenerationGenerator
    from .eval.image_clip_evaluator import ImageCLIPEvaluator
    from .eval.image_longclip_evaluator import ImageLongCLIPEvaluator
    from .eval.image_vqascore_evaluator import ImageVQAScoreEvaluator

else:
    import sys
    from dataflow.utils.registry import LazyLoader, generate_import_structure_from_type_checking
    cur_path = "dataflow/operators/core_vision/"
    _import_structure = generate_import_structure_from_type_checking(__file__, cur_path)
    sys.modules[__name__] = LazyLoader(__name__, "dataflow/operators/core_vision/", _import_structure)
