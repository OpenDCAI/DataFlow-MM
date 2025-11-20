from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # === generate ===
    from .generate.prompted_aqa_generator import PromptedAQAGenerator
    from .generate.silero_vad_generator import SileroVADGenerator
    
    from .generaterow.merge_chunks_row_generator import MergeChunksRowGenerator

    # === Filter ===
    from .filter.ctc_forced_alignment_filter import CTCForcedAlignmentFilter

    # === Eval ===
    from .eval.video_audio_similarity_evaluator import VideoAudioSimilarity
    from .eval.ctc_forced_alignment_evaluator import CTCForcedAlignmentSampleEvaluator

else:
    import sys
    from dataflow.utils.registry import LazyLoader, generate_import_structure_from_type_checking
    cur_path = "dataflow/operators/core_audio/"
    _import_structure = generate_import_structure_from_type_checking(__file__, cur_path)
    sys.modules[__name__] = LazyLoader(__name__, "dataflow/operators/core_audio/", _import_structure)
