from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # === generate ===
    from .generate.prompted_aqa_generator import PromptedAQAGenerator
    from .generate.audio_silero_voice_activity_detection_timestamps_generator import SileroVADGenerator
    
    from .generaterow.audio_merge_chunks_by_timestamps_row_generator import MergeChunksRowGenerator

    # === Filter ===
    from .filter.audio_ctc_forced_alignment_transcription_quality_filter import CTCForcedAlignmentFilter

    # === Eval ===
    from .eval.video_audio_similarity_evaluator import VideoAudioSimilarity
    from .eval.audio_ctc_forced_alignment_transcription_quality_evaluator import CTCForcedAlignmentSampleEvaluator

else:
    import sys
    from dataflow.utils.registry import LazyLoader, generate_import_structure_from_type_checking
    cur_path = "dataflow/operators/core_audio/"
    _import_structure = generate_import_structure_from_type_checking(__file__, cur_path)
    sys.modules[__name__] = LazyLoader(__name__, "dataflow/operators/core_audio/", _import_structure)
