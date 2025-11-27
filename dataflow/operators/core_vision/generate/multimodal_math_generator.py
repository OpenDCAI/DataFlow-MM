import os
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY


@OPERATOR_REGISTRY.register()
class MultimodalMathGenerate(OperatorABC):
    def __init__(self, image_dir: str = "~/cache", seed: int | None = None):
        self.logger = get_logger()
        self.image_dir = image_dir
        os.makedirs(self.image_dir, exist_ok=True)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "该算子用于根据输入 dataframe 每一行的模式值（simple 或 complex），自动生成对应的多模态数学题目（函数图像 + QA），并将结果拼接回原始 dataframe。\n\n"
                "输入参数：\n"
                "  - dataframe（由 storage 自动读取）\n"
                "  - input_key: 模式列的字段名（默认: 'mode'）。\n"
                "      - 当该列的值为 'simple' 时生成简单模式题目\n"
                "      - 其他任何值均按 complex 模式处理\n\n"
                "模式说明：\n"
                "  ● simple 模式（简单题）\n"
                "      - 使用固定函数类型（一次、二次、三角、指数）\n"
                "      - 随机选取 x 值，计算 f(x)\n"
                "      - 题目形式为：给定函数表达式，求 x = a 时的函数值\n"
                "      - 更偏向数值代入和基础函数认知\n\n"
                "  ● complex 模式（复杂题）\n"
                "      - 随机选取函数类型（如二次、三角、指数）\n"
                "      - 随机生成三类复杂数学问题之一：\n"
                "          1) derivative：判断某点导数（变化率）正/负\n"
                "          2) extremum：求函数在区间内的极小值点\n"
                "          3) monotonicity：判断函数在某区间是否单调\n"
                "      - 更强调数学分析能力（导数、极值、单调性）\n\n"
                "输出字段：\n"
                "  - image_path: 函数图像路径\n"
                "  - question: 自动生成的问题文本\n"
                "  - answer: 问题答案\n"
                "  - solution: 步骤说明或解释\n\n"
                "处理流程：\n"
                "  1. 遍历 dataframe 的每一行，根据 mode 生成对应题目\n"
                "  2. 自动绘图库生成函数图像并保存\n"
                "  3. 将生成的四列（image_path, question, answer, solution）与原 dataframe 横向拼接\n"
                "  4. 将新 dataframe 写回 storage\n\n"
                "功能特点：\n"
                "  - 支持 simple/complex 两种清晰的题目生成模式\n"
                "  - 行级别生成，适用于大规模数据集构建\n"
                "  - 自动绘图与文本生成结合\n"
                "  - 输出结构稳定，方便后续训练或评估使用\n"
            )

        elif lang == "en":
            return (
                "This operator generates multimodal math questions (function plots + QA) for each row\n"
                "in the input dataframe, based on a mode column that specifies either 'simple' or 'complex'.\n\n"
                "Input Parameters:\n"
                "  - dataframe (automatically read from storage)\n"
                "  - input_key: Name of the mode column (default: 'mode')\n"
                "      - If a row's value is 'simple', a simple question is generated\n"
                "      - Any other value is treated as complex mode\n\n"
                "Modes:\n"
                "  ● simple mode\n"
                "      - Uses predefined function types (linear, quadratic, sine, exponential)\n"
                "      - Randomly selects an x-value and asks for f(x)\n"
                "      - Question type: direct value substitution\n"
                "      - Focuses on basic numerical comprehension\n\n"
                "  ● complex mode\n"
                "      - Uses quadratic, sine, or exponential functions\n"
                "      - Randomly generates one of the following advanced question types:\n"
                "          1) derivative sign at a point (positive / negative / zero)\n"
                "          2) extremum location within the domain\n"
                "          3) monotonicity of the function on an interval\n"
                "      - Emphasizes mathematical reasoning (derivatives, extrema, monotonicity)\n\n"
                "Output Fields:\n"
                "  - image_path: Path of the generated function plot\n"
                "  - question: Generated math question\n"
                "  - answer: Final answer\n"
                "  - solution: Explanation or reasoning steps\n\n"
                "Processing Steps:\n"
                "  1. Iterate over all rows of the dataframe\n"
                "  2. Generate the corresponding simple/complex question\n"
                "  3. Create and save the function plot\n"
                "  4. Append the four output columns to the original dataframe\n"
                "  5. Save the updated dataframe back to storage\n\n"
                "Features:\n"
                "  - Clear simple vs. complex modes\n"
                "  - Row-wise multimodal sample generation\n"
                "  - Automatic plot creation\n"
                "  - Suitable for dataset construction and VQA/ML training\n"
            )
        else:
            return "MultimodalMathGenerate produces math questions with function plots for multimodal learning."

    def create_function_plot(self, func, x_range, img_path):
        x = np.linspace(*x_range, 200)
        y = func(x)
        plt.figure(figsize=(5, 4))
        plt.plot(x, y, label="f(x)")
        plt.grid(True)
        plt.title("Function Plot")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(img_path)
        plt.close()

    def generate_sample(self, idx: int):
        func_types = [
            ("Linear function", lambda x: 2 * x + 1, "f(x) = 2x + 1"),
            ("Quadratic function", lambda x: x**2, "f(x) = x²"),
            ("Sine function", lambda x: np.sin(x), "f(x) = sin(x)"),
            ("Exponential function", lambda x: np.exp(x / 2), "f(x) = exp(x/2)"),
        ]
        label, f, expr = random.choice(func_types)
        img_path = os.path.join(self.image_dir, f"plot_{idx}.png")
        x_range = (0, 5)
        self.create_function_plot(f, x_range, img_path)
        x_val = round(random.uniform(1.0, 4.0), 1)
        y_val = round(float(f(x_val)), 3)
        question = f"The function plot represents {expr}. What is the function value at x={x_val}?"
        answer = str(y_val)
        solution = f"According to the function expression {expr}, substitute x={x_val} to get y={y_val}."
        return {
            "image_path": img_path,
            "question": question,
            "answer": answer,
            "solution": solution,
        }

    class MathFunction:
        def __init__(self, name, func, expr, domain=(0, 5), kind="quadratic"):
            self.name = name
            self.f = func
            self.expr = expr
            self.domain = domain
            self.kind = kind
        def y(self, x):
            return float(self.f(x))
        def derivative_sign(self, x, eps=0.01):
            return np.sign(self.f(x + eps) - self.f(x - eps))
        def min_arg(self, n=100):
            x = np.linspace(*self.domain, n)
            y = self.f(x)
            return float(x[np.argmin(y)]), float(np.min(y))
        def monotonicity(self, x_start, x_end, steps=50):
            x = np.linspace(x_start, x_end, steps)
            y = self.f(x)
            d = np.diff(y)
            if np.all(d > 0):
                return "increasing"
            if np.all(d < 0):
                return "decreasing"
            return "not monotonic"

    def generate_derivative_question(self, fn: "MultimodalMathGenerate.MathFunction"):
        x = round(random.uniform(*fn.domain), 1)
        sign = fn.derivative_sign(x)
        direction = "positive" if sign > 0 else "negative" if sign < 0 else "zero"
        return {
            "question": f"The function plot represents {fn.expr}. Is the rate of change (derivative) at x={x} positive or negative?",
            "answer": direction,
            "solution": f"By observing the slope of the plot near x={x}, the rate of change is {direction}.",
        }

    def generate_extremum_question(self, fn: "MultimodalMathGenerate.MathFunction"):
        x_min, y_min = fn.min_arg()
        x_min = round(x_min, 2)
        return {
            "question": f"The function plot represents {fn.expr}. At which x-value does the function reach its minimum value in the shown domain?",
            "answer": str(x_min),
            "solution": f"From the plot, the minimum occurs at x={x_min}, with y={round(y_min, 2)}",
        }

    def generate_monotonicity_question(self, fn: "MultimodalMathGenerate.MathFunction"):
        a, b = sorted([round(random.uniform(*fn.domain), 1) for _ in range(2)])
        mono = fn.monotonicity(a, b)
        return {
            "question": f"The function plot represents {fn.expr}. Is the function monotonically increasing or decreasing in the interval [{a}, {b}]?",
            "answer": mono,
            "solution": f"By observing the function value trend in the interval [{a}, {b}], the function is {mono}.",
        }

    def generate_complex_sample(self, idx: int):
        fn_list = [
            self.MathFunction("Quadratic", lambda x: x**2, "f(x) = x²", domain=(0, 5), kind="quadratic"),
            self.MathFunction("Sine", lambda x: np.sin(x), "f(x) = sin(x)", domain=(0, 6), kind="sin"),
            self.MathFunction("Exponential", lambda x: np.exp(x / 2), "f(x) = exp(x/2)", domain=(0, 5), kind="exp"),
        ]
        fn = random.choice(fn_list)
        img_path = os.path.join(self.image_dir, f"plot_{idx}.png")
        self.create_function_plot(fn.f, fn.domain, img_path)
        generators = [self.generate_derivative_question, self.generate_extremum_question, self.generate_monotonicity_question]
        qdict = random.choice(generators)(fn)
        return {"image_path": img_path, **qdict}

    def run(
        self,
        storage: DataFlowStorage,
        input_key: str = "mode",
    ):
        rows = []
        dataframe = storage.read("dataframe")
        modes = dataframe[input_key].tolist() # 很多条
        for i, mode in enumerate(modes):
            if mode == "simple":
                rows.append(self.generate_sample(i))
            else:
                rows.append(self.generate_complex_sample(i))

        # 将 rows 变成一个 dataframe
        df_new = pd.DataFrame(rows, columns=["image_path", "question", "answer", "solution"])

        # 按行合并（横向拼接）
        dataframe_merged = pd.concat([dataframe.reset_index(drop=True),
                                    df_new.reset_index(drop=True)],
                                    axis=1)

        # 保存
        storage.write(dataframe_merged)
        return ["image_path", "question", "answer", "solution"]