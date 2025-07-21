from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .vqa.prompted_vqa import PromptedVQA
    from .image_aigc.text_to_image_gen import Text2ImageGenerator
    from .image_aigc.image_edit import ImageEditor
else:
    import sys
    from dataflow.utils.registry import LazyLoader, generate_import_structure_from_type_checking
    cur_path = "dataflow/operators/generate/"
    _import_structure = generate_import_structure_from_type_checking(__file__, cur_path)
    sys.modules[__name__] = LazyLoader(__name__, "dataflow/operators/generate/", _import_structure)
