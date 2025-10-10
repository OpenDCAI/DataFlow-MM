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
        return "生成多模态数学题（函数图像 + QA）" if lang == "zh" else "Generate multimodal math QA (function plots)."

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
