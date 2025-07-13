from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.logger import get_logger

@OPERATOR_REGISTRY.register()
class EvalOperator(OperatorABC):
    def __init__(self):
        self.logger = get_logger(self.name)

    def run(self):
        pass