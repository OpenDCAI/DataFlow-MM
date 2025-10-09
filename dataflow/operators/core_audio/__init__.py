from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # === generate ===
    from .generate.prompted_aqa_generator import PromptedAQAGenerator
    from .generate.silero_vad_generator import SileroVADGenerator
    
    from .generaterow.cut_and_merge import MergeChunksByTimestamps

    # === Filter ===
    from .filter.ctc_forced_aligner_filter import CTCForcedAlignFilter

    # === Eval ===
    from .eval.video_audio_similarity_evaluator import VideoResolutionFilter
    from .eval.ctc_forced_aligner import CTCForcedAlignSampleEvaluator

else:
    import sys
    from dataflow.utils.registry import LazyLoader, generate_import_structure_from_type_checking
    cur_path = "dataflow/operators/core_audio/"
    _import_structure = generate_import_structure_from_type_checking(__file__, cur_path)
    sys.modules[__name__] = LazyLoader(__name__, "dataflow/operators/core_audio/", _import_structure)
