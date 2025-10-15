from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .generate.prompted_vqa_generator import PromptedVQAGenerator
    from .generate.prompted_image_generator import PromptedImageGenerator
    from .generate.prompted_image_edit_generator import PromptedImageEditGenerator
    from .generate.image_caption_generator import ImageCaptionGenerate
    from .generate.image_qa_generator import ImageQAGenerate
    from .generate.multimodal_math_generator import MultimodalMathGenerate
    from .generate.personalized_qa_generator import PersQAGenerate
    from .generate.vision_mct_reasoning_sft_generator import VisionMCTSReasoningSFTGenerate
    from .generate.image_scale_caption_generator import ImageScaleCaptionGenerate, ImageScaleCaptionGenerateConfig
    from .filter.image_aesthetic_filter import ImageAestheticFilter
    from .filter.cat_filter import CatFilter
    from .filter.clip_filter import ClipFilter
    from .filter.complexity_filter import ComplexityFilter
    from .filter.consistency_filter import ConsistencyFilter
    from .filter.text_image_diversity_filter import TextImageDiversityFilter
    from .filter.sensitive_filter import SensitiveFilter
    from .refine.vision_seg_cutout_refine import VisionSegCutoutRefine
    from .eval.image.image_evaluator import EvalImageGenerationGenerator
    from .eval.image_text.clip_image_text_evaluator import CLIPEvaluator
    from .eval.image_text.long_clip_image_text_evaluator import LongCLIPEvaluator
    from .eval.image_text.vqa_score_image_text_evaluator import VQAScoreEvaluator

else:
    import sys
    from dataflow.utils.registry import LazyLoader, generate_import_structure_from_type_checking
    cur_path = "dataflow/operators/core_vision/"
    _import_structure = generate_import_structure_from_type_checking(__file__, cur_path)
    sys.modules[__name__] = LazyLoader(__name__, "dataflow/operators/core_vision/", _import_structure)
