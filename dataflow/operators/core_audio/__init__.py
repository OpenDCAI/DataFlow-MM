from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .generate.prompted_aqa_generator import PromptedAQAGenerator
    from .generate.silero_vad_generator import SileroVADGenerator
    
    from .generaterow.cut_and_merge import MergeChunksByTimestamps

    from .eval.ctc_forced_aligner import CTCForcedAlignSampleEvaluator

    from .filter.ctc_forced_aligner_filter import CTCForcedAlignFilter

else:
    import sys
    from dataflow.utils.registry import LazyLoader, generate_import_structure_from_type_checking
    cur_path = "dataflow/operators/core_audio/"
    _import_structure = generate_import_structure_from_type_checking(__file__, cur_path)
    sys.modules[__name__] = LazyLoader(__name__, "dataflow/operators/core_audio/", _import_structure)
