from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .generate.prompted_vqa_generator import PromptedVQAGenerator
    from .generate.prompted_image_generator import PromptedImageGenerator
    from .generate.prompted_image_edit_generator import PromptedImageEditGenerator
    from .filter.rule_base_filter import RuleBaseFilter
    from .filter.deduplication_filter import DeduplicateFilter
    from .filter.knn_similarity_filter import KNNSimilarityFilter
    from .filter.clipscore_filter import CLIPScoreFilter
    from .filter.datatailor_filter import DataTailorFilter
    from .filter.vision_dependent_filter import VisionDependentFilter
    from .filter.failrate_filter import FailRateFilter

else:
    import sys
    from dataflow.utils.registry import LazyLoader, generate_import_structure_from_type_checking
    cur_path = "dataflow/operators/core_vision/"
    _import_structure = generate_import_structure_from_type_checking(__file__, cur_path)
    sys.modules[__name__] = LazyLoader(__name__, "dataflow/operators/core_vision/", _import_structure)
