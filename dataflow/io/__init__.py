from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .qwen_vl.qwen_vl_2_5 import Qwen2_5VLIO
    from .qwen_audio.qwen_audio_2 import Qwen2_AudioIO
    from .diffuser.image_gen import ImageIO
    from .whisper.whisper import WhisperIO
else:
    import sys
    from dataflow.utils.registry import LazyLoader, generate_import_structure_from_type_checking
    cur_path = "dataflow/io/"

    _import_structure = generate_import_structure_from_type_checking(__file__, cur_path)
    sys.modules[__name__] = LazyLoader(__name__, "dataflow/io/", _import_structure, if_fuzzy_key_matching=True)