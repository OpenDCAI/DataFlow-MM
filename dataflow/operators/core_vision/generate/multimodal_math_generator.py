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
    def __init__(self, image_dir: str = "/data0/mt/Dataflow-MM-Preview/cache", seed: int | None = None):
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
                "该算子用于生成多模态数学题目（函数图像 + 问答对）。\n\n"
                "输入参数：\n"
                "  - n: 生成样本数量 (默认: 200)\n"
                "  - mode: 生成模式，'simple'简单模式或'complex'复杂模式 (默认: 'complex')\n"
                "  - output_key: 输出数据字段名前缀 (默认: 'multimodal_math')\n"
                "  - seed: 随机种子，用于结果可复现 (可选)\n"
                "输出参数：\n"
                "  - image_path: 生成的函数图像文件路径\n"
                "  - question: 基于函数图像的数学问题\n"
                "  - answer: 问题答案\n"
                "  - solution: 解题步骤和说明\n"
                "功能特点：\n"
                "  - 支持多种函数类型：一次函数、二次函数、三角函数、指数函数等\n"
                "  - 提供两种生成模式：简单数值计算和复杂数学概念\n"
                "  - 复杂模式包含导数判断、极值点分析、单调性分析等高级题目\n"
                "  - 自动生成对应的函数图像并保存\n"
                "  - 支持随机种子设置，确保结果可复现\n"
                "应用场景：\n"
                "  - 多模态数学教育数据集构建\n"
                "  - 视觉问答模型训练\n"
                "  - 数学推理能力评估\n"
            )
        elif lang == "en":
            return (
                "This operator generates multimodal math questions (function plots + QA pairs).\n\n"
                "Input Parameters:\n"
                "  - n: Number of samples to generate (default: 200)\n"
                "  - mode: Generation mode, 'simple' or 'complex' (default: 'complex')\n"
                "  - output_key: Output data field name prefix (default: 'multimodal_math')\n"
                "  - seed: Random seed for reproducibility (optional)\n"
                "Output Parameters:\n"
                "  - image_path: Path to generated function plot image\n"
                "  - question: Math question based on the function plot\n"
                "  - answer: Answer to the question\n"
                "  - solution: Step-by-step solution and explanation\n"
                "Features:\n"
                "  - Supports multiple function types: linear, quadratic, trigonometric, exponential, etc.\n"
                "  - Two generation modes: simple numerical computation and complex math concepts\n"
                "  - Complex mode includes advanced topics: derivative analysis, extremum points, monotonicity\n"
                "  - Automatically generates and saves corresponding function plots\n"
                "  - Supports random seed for reproducible results\n"
                "Applications:\n"
                "  - Multimodal math education dataset construction\n"
                "  - Visual question answering model training\n"
                "  - Mathematical reasoning ability evaluation\n"
            )
        else:
            return "MultimodalMathGenerate produces math questions with function plots for multimodal learning."

    def create_function_plot(self, func, x_range, img_path):
        x = np.linspace(*x_range, 200)
        y = func(x)
        plt.figure(figsize=(5, 4))
        plt.plot(x, y, label="f(x)")
        plt.grid(True)
        plt.title("函数图像")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(img_path)
        plt.close()

    def generate_sample(self, idx: int):
        func_types = [
            ("一次函数", lambda x: 2 * x + 1, "f(x) = 2x + 1"),
            ("二次函数", lambda x: x**2, "f(x) = x²"),
            ("正弦函数", lambda x: np.sin(x), "f(x) = sin(x)"),
            ("指数函数", lambda x: np.exp(x / 2), "f(x) = exp(x/2)"),
        ]
        label, f, expr = random.choice(func_types)
        img_path = os.path.join(self.image_dir, f"plot_{idx}.png")
        x_range = (0, 5)
        self.create_function_plot(f, x_range, img_path)
        x_val = round(random.uniform(1.0, 4.0), 1)
        y_val = round(float(f(x_val)), 3)
        question = f"函数图像表示 {expr}，请问在 x={x_val} 时，函数值是多少？"
        answer = str(y_val)
        solution = f"根据函数表达式 {expr}，代入 x={x_val}，计算得 y={y_val}。"
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
                return "递增"
            if np.all(d < 0):
                return "递减"
            return "不单调"

    def generate_derivative_question(self, fn: "MultimodalMathGenerate.MathFunction"):
        x = round(random.uniform(*fn.domain), 1)
        sign = fn.derivative_sign(x)
        direction = "正" if sign > 0 else "负" if sign < 0 else "为零"
        return {
            "question": f"函数图像表示 {fn.expr}，请判断在 x={x} 处函数的变化率是正的还是负的？",
            "answer": direction,
            "solution": f"观察图像在 x={x} 附近的斜率，可知变化率为{direction}。",
        }

    def generate_extremum_question(self, fn: "MultimodalMathGenerate.MathFunction"):
        x_min, y_min = fn.min_arg()
        x_min = round(x_min, 2)
        return {
            "question": f"函数图像表示 {fn.expr}，该函数在图像中取得最小值的位置是在哪个 x？",
            "answer": str(x_min),
            "solution": f"观察图像可知最小值出现在 x={x_min}，对应 y={round(y_min, 2)}",
        }

    def generate_monotonicity_question(self, fn: "MultimodalMathGenerate.MathFunction"):
        a, b = sorted([round(random.uniform(*fn.domain), 1) for _ in range(2)])
        mono = fn.monotonicity(a, b)
        return {
            "question": f"函数图像表示 {fn.expr}，请判断在区间 [{a}, {b}] 内函数是单调递增还是递减？",
            "answer": mono,
            "solution": f"在区间 [{a}, {b}] 内观察函数值变化趋势，可知函数是{mono}的。",
        }

    def generate_complex_sample(self, idx: int):
        fn_list = [
            self.MathFunction("平方", lambda x: x**2, "f(x) = x²", domain=(0, 5), kind="quadratic"),
            self.MathFunction("正弦", lambda x: np.sin(x), "f(x) = sin(x)", domain=(0, 6), kind="sin"),
            self.MathFunction("指数", lambda x: np.exp(x / 2), "f(x) = exp(x/2)", domain=(0, 5), kind="exp"),
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
        n: int = 200,
        mode: str = "complex",
        output_key: str = "multimodal_math",
    ):
        rows = []
        if mode == "simple":
            for i in range(n):
                rows.append(self.generate_sample(i))
        else:
            for i in range(n):
                rows.append(self.generate_complex_sample(i))
        df = pd.DataFrame(rows, columns=["image_path", "question", "answer", "solution"])
        storage.write(df)
        return ["image_path", "question", "answer", "solution"]
