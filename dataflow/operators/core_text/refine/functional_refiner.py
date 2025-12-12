import pandas as pd
from typing import Callable, Any
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow import get_logger

@OPERATOR_REGISTRY.register()
class FunctionalRefiner(OperatorABC):
    """
    [Refine] 通用 Python 逻辑精炼器。
    用于：分句、正则解析、字符串拼接等不涉及模型调用的数据加工。
    """
    def __init__(self, func: Callable[..., Any]):
        self.func = func
        self.logger = get_logger()

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "通用逻辑精炼算子 (FunctionalRefiner)。\n"
                "该算子不调用模型，仅对 DataFrame 执行纯 Python 函数逻辑。\n\n"
                "输入参数：\n"
                "  - func: 需要执行的 Python 函数（如分句、正则解析、拼接等）\n"
                "  - **input_keys: 传递给函数的参数映射 (函数参数名=列名)\n"
                "输出参数：\n"
                "  - output_key: 函数执行后的结果列\n"
                "功能特点：\n"
                "  - 极其灵活，作为 Pipeline 中的胶水层\n"
                "  - 支持行级 (Row-wise) 数据转换\n"
            )
        else:
            return (
                "Functional Logic Refiner (FunctionalRefiner).\n"
                "This operator executes pure Python logic on the DataFrame without model calls.\n\n"
                "Input Parameters:\n"
                "  - func: The Python function to execute (e.g., split, parse, join)\n"
                "  - **input_keys: Argument mapping for the function (arg_name=column_name)\n"
                "Output Parameters:\n"
                "  - output_key: The column storing the function result\n"
                "Features:\n"
                "  - Highly flexible glue layer for pipelines\n"
                "  - Supports row-wise data transformation\n"
            )
            
    def run(self, storage: DataFlowStorage, output_key: str, **input_keys):
        df = storage.read("dataframe")
        
        def apply_wrapper(row):
            kwargs = {k: row[v] for k, v in input_keys.items()}
            return self.func(**kwargs)

        df[output_key] = df.apply(apply_wrapper, axis=1)
        storage.write(df)
        return [output_key]