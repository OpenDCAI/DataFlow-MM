import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage


@OPERATOR_REGISTRY.register()
class ScoreFilter(OperatorABC):
    """
    通用分数过滤算子：根据指定列的分数值进行过滤
    """

    def __init__(
        self,
        min_score: float = None,
        max_score: float = None,
    ):
        """
        初始化分数过滤算子
        
        Args:
            min_score: 最小分数阈值（包含），低于此分数的行将被标记为 False
            max_score: 最大分数阈值（包含），高于此分数的行将被标记为 False
        """
        self.logger = get_logger()
        self.min_score = min_score
        self.max_score = max_score

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return "通用分数过滤：根据指定列的分数值进行过滤"
        else:
            return "Score Filter: Filter rows based on score column values"

    def run(
        self,
        storage: DataFlowStorage,
        input_score_key: str = "score",
        output_select_key: str = "select",
    ) -> str:
        """
        执行分数过滤
        
        Args:
            storage: DataFlowStorage对象，用于数据读写
            input_score_key: 要过滤的分数列名（如 "reward", "accuracy" 等）
            output_select_key: 输出的筛选标记列名，默认为 "select"
            
        Returns:
            str: 输出列名
        """
        self.logger.info(f"Starting ScoreFilter on column '{input_score_key}'...")
        
        # 从storage读取输入数据
        df = storage.read("dataframe")
        self.logger.info(f"Loaded dataframe: {len(df)} rows")
        
        # 检查分数列是否存在
        if input_score_key not in df.columns:
            raise ValueError(f"Score column '{input_score_key}' not found in dataframe")
        
        # 应用过滤条件
        select_mask = pd.Series([True] * len(df), index=df.index)
        
        if self.min_score is not None:
            select_mask &= df[input_score_key] >= self.min_score
        
        if self.max_score is not None:
            select_mask &= df[input_score_key] <= self.max_score
        
        # 添加筛选标记列
        df[output_select_key] = select_mask
        
        # 写回数据
        storage.write(df)
        
        # 统计结果
        selected_count = select_mask.sum()
        total_count = len(df)
        self.logger.info(f"Filtered: {selected_count}/{total_count} rows passed the filter")
        
        return output_select_key

