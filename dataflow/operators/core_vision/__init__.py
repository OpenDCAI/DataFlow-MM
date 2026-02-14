from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # === Generate ===
    from .generate.image_qa_generator import ImageQAGenerator
    from .generate.multimodal_math_generator import MultimodalMathGenerator
    from .generate.personalized_qa_generator import PersQAGenerator
    from .generate.prompted_image_generator import PromptedImageGenerator
    from .generate.prompted_vqa_generator import PromptedVQAGenerator
    from .generate.prompt_templated_vqa_generator import PromptTemplatedVQAGenerator
    from .generate.fix_prompted_vqa_generator import FixPromptedVQAGenerator
    from .generate.video_clip_generator import VideoClipGenerator
    from .generate.video_qa_generator import VideoCaptionToQAGenerator
    from .generate.video_caption_generator import VideoToCaptionGenerator
    from .generate.video_merged_caption_generator import VideoMergedCaptionGenerator
    from .generate.video_cotqa_generator import VideoCOTQAGenerator
    from .generate.multirole_videoqa_generator import MultiroleVideoQAInitialGenerator, MultiroleVideoQAMultiAgentGenerator, MultiroleVideoQAFinalGenerator
    from .generate.batch_vqa_generator import BatchVQAGenerator
    from .generate.vlm_bbox_generator import VLMBBoxGenerator

    # === Filter ===
    from .filter.video_clip_filter import VideoClipFilter
    from .filter.video_frame_filter import VideoFrameFilter
    from .filter.video_info_filter import VideoInfoFilter
    from .filter.video_scene_filter import VideoSceneFilter
    from .filter.video_score_filter import VideoScoreFilter
    from .filter.video_aesthetic_filter import VideoAestheticFilter
    from .filter.video_luminance_filter import VideoLuminanceFilter
    from .filter.video_ocr_filter import VideoOCRFilter
    from .filter.video_motion_score_filter import VideoMotionScoreFilter
    from .filter.video_resolution_filter import VideoResolutionFilter
    from .filter.image_aesthetic_filter import ImageAestheticFilter
    from .filter.image_cat_filter import ImageCatFilter
    from .filter.image_clip_filter import ImageClipFilter
    from .filter.image_complexity_filter import ImageComplexityFilter
    from .filter.image_consistency_filter import ImageConsistencyFilter
    from .filter.image_diversity_filter import ImageDiversityFilter
    from .filter.image_sensitive_filter import ImageSensitiveFilter
    from .refine.vision_seg_cutout_refiner import VisionSegCutoutRefiner
    from .filter.rule_base_filter import RuleBaseFilter
    from .filter.image_deduplication_filter import ImageDeduplicateFilter
    from .filter.knn_similarity_filter import KNNSimilarityFilter
    from .filter.clipscore_filter import CLIPScoreFilter
    from .filter.datatailor_filter import DataTailorFilter
    from .filter.vision_dependent_filter import VisionDependentFilter
    from .filter.failrate_filter import FailRateFilter
    from .filter.score_filter import ScoreFilter

    # === Eval ===
    from .eval.video_aesthetic_evaluator import VideoAestheticEvaluator
    from .eval.video_luminance_evaluator import VideoLuminanceEvaluator
    from .eval.video_ocr_evaluator import VideoOCREvaluator
    from .eval.general_text_answer_evaluator import GeneralTextAnswerEvaluator
    from .eval.image.image_evaluator import EvalImageGenerationGenerator
    from .eval.image_clip_evaluator import ImageCLIPEvaluator
    from .eval.image_longclip_evaluator import ImageLongCLIPEvaluator
    from .eval.image_vqascore_evaluator import ImageVQAScoreEvaluator

    # === Refine ===
    from .refine.wiki_qa_refiner import WikiQARefiner
    from .refine.visual_grounding_refiner import VisualGroundingRefiner
    from .refine.vision_seg_cutout_refiner import VisionSegCutoutRefiner
    from .refine.visual_dependency_refiner import VisualDependencyRefiner

else:
    import sys
    from pathlib import Path
    from dataflow.utils.registry import LazyLoader, generate_import_structure_from_type_checking
    cur_path = "dataflow/operators/core_vision/"
    _import_structure = generate_import_structure_from_type_checking(__file__, cur_path)
    _loader = LazyLoader(__name__, "dataflow/operators/core_vision/", _import_structure)
    _loader.__path__ = [str(Path(__file__).parent)]
    sys.modules[__name__] = _loader
    # sys.modules[__name__] = LazyLoader(__name__, "dataflow/operators/core_vision/", _import_structure)
