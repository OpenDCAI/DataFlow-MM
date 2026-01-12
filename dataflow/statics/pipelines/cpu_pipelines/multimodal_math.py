import argparse
import os
from typing import List

from dataflow.utils.storage import FileStorage
from dataflow.operators.core_vision import MultimodalMathGenerator


class MultimodalMathPipeline:
    """
    一行命令即可完成多模态数学题目的批量生成。
    生成包含函数图像和对应问答对的数据集。
    """

    def __init__(
        self,
        *,
        image_dir: str = "./cache_math",
        seed: int | None = None,
        first_entry_file: str = "dataflow/example/image_to_text_pipeline/math_qa.jsonl",
        cache_path: str = "./cache_math",
        file_name_prefix: str = "math_data_step",
        cache_type: str = "jsonl",
    ):
        # ---------- 1. Storage ----------
        self.storage = FileStorage(
            first_entry_file_name=first_entry_file,
            cache_path=cache_path,
            file_name_prefix=file_name_prefix,
            cache_type=cache_type,
        )

        # ---------- 2. Operator ----------
        self.math_generator = MultimodalMathGenerator(
            image_dir=image_dir,
            seed=seed
        )

    # ------------------------------------------------------------------ #
    def forward(self):
        """
        一键生成多模态数学题目数据集：
            生成函数图像 + 对应问答对
        """
        self.math_generator.run(
            storage=self.storage.step(),       # Pipeline 负责推进 step
            input_key="mode"
        )


# ---------------------------- CLI 入口 -------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate multimodal math QA dataset with DataFlow")

    # 生成参数
    parser.add_argument("--seed", type=int, default=None, help="随机种子，用于结果可复现")
    
    # 存储参数
    parser.add_argument("--image_dir", default="./cache_local", 
                       help="生成的函数图像保存目录")
    parser.add_argument("--first_entry_file", default="dataflow/example/image_to_text_pipeline/math_qa.jsonl")
    parser.add_argument("--cache_path", default="./cache_local")
    parser.add_argument("--file_name_prefix", default="math_data")
    parser.add_argument("--cache_type", default="jsonl")

    args = parser.parse_args()

    pipe = MultimodalMathPipeline(
        image_dir=args.image_dir,
        seed=args.seed,
        first_entry_file=args.first_entry_file,
        cache_path=args.cache_path,
        file_name_prefix=args.file_name_prefix,
        cache_type=args.cache_type,
    )
    
    print(f"开始生成多模态数学题目数据集...")
    print(f"图像保存路径: {args.image_dir}")
    
    pipe.forward()
    print("多模态数学题目生成完成！")