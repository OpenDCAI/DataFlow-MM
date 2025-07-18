from typing import ParamSpec, TypeVar, Generic, Protocol, List
from functools import wraps
import inspect
from dataflow.logger import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage, DummyStorage
from tqdm import tqdm
import pandas as pd
P = ParamSpec("P")
R = TypeVar("R")

class HasRun(Protocol[P, R]):
    def run(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """
        这里写一份通用的 run 文档也可以，
        不写的话会被下面动态拷贝原 operator.run 的 __doc__。
        """
        ...

class BatchWrapper(Generic[P, R]):
    """
    通用的批处理 Wrapper。

    在静态检查/IDE 里，BatchWrapper.run 的签名和 operator.run 完全一致。
    运行时，会把 operator.run 的 __doc__ 和 __signature__ 也拷过来，
    这样 help(bw.run) 时能看到原 operator 的文档。
    """
    def __init__(self, operator: HasRun[P, R], batch_size: int = 32) -> None:
        self._operator = operator
        self._logger = get_logger()
        self._batch_size = batch_size
        # 准备一个复用的 dummy storage，用来逐批写入
        self._dummy_storage = DummyStorage()

        # 动态拷贝 operator.run 的 __doc__ 和 inspect.signature
        orig = operator.run
        sig = inspect.signature(orig)
        # wrapped = wraps(orig)(self.run)        # 先 wrap docstring, __name__…
        # wrapped.__signature__ = sig            # 再贴上准确的 signature
        # 把它绑到实例上覆盖掉 class method，这样 help(instance.run) 能看到原文档
        # object.__setattr__(self, "run", wrapped)

    def run(self, *args: P.args, **kwargs: P.kwargs) -> List[R]:
        # —— 1. 提取 storage —— 
        if args:
            storage: DataFlowStorage = args[0]    # type: ignore[assignment]
            rest_args = args[1:]
            rest_kwargs = kwargs
        else:
            storage = kwargs.get("storage")      # type: ignore[assignment]
            if storage is None:
                raise ValueError(
                    f"A DataFlowStorage is required for {self._operator!r}.run()"
                )
            rest_kwargs = {k: v for k, v in kwargs.items() if k != "storage"}
            rest_args = ()

        # —— 2. 读出全量数据并按 batch_size 切分 —— 
        whole_dataframe = storage.read()
        batches = [
            whole_dataframe[i : i + self._batch_size] 
            for i in range(0, len(whole_dataframe), self._batch_size)
        ]

        # —— 3. 对每个 batch 写入 dummy_storage 并调用 operator.run —— 
        for batch_df in tqdm(batches):
            self._logger.info(f"Processing batch of size {len(batch_df)}")
            # 清空并写入当前批
            # self._dummy_storage.clear()
            self._dummy_storage.write(batch_df)
            # 这里把 dummy_storage 作为第一个参数传给 operator.run
            print(f"Running batch with {len(batch_df)} items...")
            self._operator.run(self._dummy_storage, *rest_args, **rest_kwargs)
            
            res: pd.DataFrame = self._dummy_storage.read()
    
            # 1) 找出 res 中但不在 whole_dataframe 中的列
            new_cols = [c for c in res.columns if c not in whole_dataframe.columns]
            # 2) 在 whole_dataframe 中先创建这些新列（填充 NaN）
            for c in new_cols:
                whole_dataframe[c] = pd.NA
            # 3) 按索引和列把 res 中的值写回去
            whole_dataframe.loc[res.index, res.columns] = res
        storage.write(whole_dataframe)
