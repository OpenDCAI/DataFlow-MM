from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .generate.prompted_qa_generator import PromptedQAGenerator
    from .generate.prompt_templated_qa_generator import PromptTemplatedQAGenerator

else:
    import sys
    from dataflow.utils.registry import LazyLoader, generate_import_structure_from_type_checking
    cur_path = "dataflow/operators/core_text/"
    _import_structure = generate_import_structure_from_type_checking(__file__, cur_path)
    sys.modules[__name__] = LazyLoader(__name__, "dataflow/operators/core_text/", _import_structure)
