import argparse
import os
from typing import List

from dataflow.utils.storage import FileStorage
from dataflow.operators.core_vision import MultimodalMathGenerate


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
        first_entry_file: str = "dataflow/example/multimodal_math/math_qa_dataset.jsonl",
        cache_path: str = "./cache_math",
        file_name_prefix: str = "math_data_step",
        cache_type: str = "jsonl",
        output_key: str = "multimodal_math",
    ):
        # ---------- 1. Storage ----------
        self.storage = FileStorage(
            first_entry_file_name=first_entry_file,
            cache_path=cache_path,
            file_name_prefix=file_name_prefix,
            cache_type=cache_type,
        )

        # ---------- 2. Operator ----------
        self.math_generator = MultimodalMathGenerate(
            image_dir=image_dir,
            seed=seed
        )
        
        self.output_key = output_key

    # ------------------------------------------------------------------ #
    def forward(self, n: int = 200, mode: str = "complex"):
        """
        一键生成多模态数学题目数据集：
            生成函数图像 + 对应问答对
        """
        self.math_generator.run(
            storage=self.storage.step(),       # Pipeline 负责推进 step
            n=n,
            mode=mode,
            output_key=self.output_key,
        )


# ---------------------------- CLI 入口 -------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate multimodal math QA dataset with DataFlow")

    # 生成参数
    parser.add_argument("--n", type=int, default=3, help="生成样本数量")
    parser.add_argument("--mode", choices=["simple", "complex"], default="complex", 
                       help="生成模式：simple-简单数值计算，complex-复杂数学概念")
    parser.add_argument("--seed", type=int, default=None, help="随机种子，用于结果可复现")
    
    # 存储参数
    parser.add_argument("--image_dir", default="./cache_local", 
                       help="生成的函数图像保存目录")
    parser.add_argument("--output_file", default="./cache_local/math_qa_dataset.jsonl",
                       help="输出数据集文件路径")
    parser.add_argument("--cache_path", default="./cache_local")
    parser.add_argument("--file_name_prefix", default="math_data")
    parser.add_argument("--cache_type", default="jsonl")
    parser.add_argument("--output_key", default="multimodal_math")

    args = parser.parse_args()

    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    os.makedirs(args.image_dir, exist_ok=True)

    pipe = MultimodalMathPipeline(
        image_dir=args.image_dir,
        seed=args.seed,
        first_entry_file=args.output_file,
        cache_path=args.cache_path,
        file_name_prefix=args.file_name_prefix,
        cache_type=args.cache_type,
        output_key=args.output_key,
    )
    
    print(f"开始生成多模态数学题目数据集...")
    print(f"样本数量: {args.n}")
    print(f"生成模式: {args.mode}")
    print(f"图像保存路径: {args.image_dir}")
    print(f"数据集保存路径: {args.output_file}")
    
    pipe.forward(n=args.n, mode=args.mode)
    
    print("多模态数学题目生成完成！")