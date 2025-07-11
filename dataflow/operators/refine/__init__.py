import sys
from dataflow.utils.registry import LazyLoader
from .GeneralText import *
cur_path = "dataflow/operators/refine/"
_import_structure = {
    "HtmlUrlRemoverRefiner": (cur_path + "GeneralText" + "html_remove_refiner.py", "HtmlUrlRemoverRefiner")
}

sys.modules[__name__] = LazyLoader(__name__, "dataflow/operators/refine/", _import_structure)
