from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .format.conversation_to_message import Conversation2Message
else:
    import sys
    from dataflow.utils.registry import LazyLoader, generate_import_structure_from_type_checking
    cur_path = "dataflow/operators/conversations/"
    _import_structure = generate_import_structure_from_type_checking(__file__, cur_path)
    sys.modules[__name__] = LazyLoader(__name__, "dataflow/operators/conversations/", _import_structure)
