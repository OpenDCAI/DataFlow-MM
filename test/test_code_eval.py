import os
import json
import pandas as pd

from dataflow.utils.storage import FileStorage
from dataflow.operators.core_vision import (
    CodeCorrectnessEvaluator,
    CodeStabilityEvaluator,
    CodeMaintainabilityEvaluator,
)

DEF_IN_PATH = "/data0/happykeyan/workspace/DataFlow-MM/dataflow/example/test_code_eval/raw_code.jsonl"
DEF_CACHE_DIR = "/data0/happykeyan/workspace/DataFlow-MM/dataflow/example/test_code_eval/cache"
DEF_PREFIX = "code_eval_step"

SAMPLE_JSONL = [
    {
        "id": 1,
        "title": "well_structured_math_utils",
        "code": '"""Math utilities."""\nfrom typing import List\n\ndef mean(xs: List[float]) -> float:\n    """\n    Return arithmetic mean.\n    """\n    assert len(xs) > 0\n    s = 0.0\n    for x in xs:\n        s += x\n    return s / len(xs)\n\ndef variance(xs: List[float]) -> float:\n    """\n    Return sample variance.\n    """\n    m = mean(xs)\n    acc = 0.0\n    for x in xs:\n        acc += (x - m) ** 2\n    assert acc >= 0.0\n    return acc / (len(xs) - 1)\n'
    },
    {
        "id": 2,
        "title": "syntax_error_missing_colon",
        "code": "def broken(x)\n    return x+1\n"
    },
    {
        "id": 3,
        "title": "exception_handling_bare_except",
        "code": "def read_int(s: str) -> int:\n    try:\n        return int(s)\n    except:\n        return 0\n"
    },
    {
        "id": 4,
        "title": "dangerous_subprocess_usage",
        "code": 'import subprocess\n\ndef run_cmd(cmd: str) -> str:\n    out = subprocess.check_output(cmd, shell=True).decode("utf-8", "ignore")\n    return out\n'
    },
    {
        "id": 5,
        "title": "file_io_without_with",
        "code": 'def count_lines(path: str) -> int:\n    f = open(path, "r", encoding="utf-8")\n    n = 0\n    for _ in f:\n        n += 1\n    f.close()\n    return n\n'
    },
    {
        "id": 6,
        "title": "class_with_docs_and_types",
        "code": 'from typing import Iterable\n\nclass Accumulator:\n    """\n    Accumulate values and compute running statistics.\n    """\n    def __init__(self) -> None:\n        """Initialize with zero state."""\n        self.n: int = 0\n        self.s: float = 0.0\n\n    def add(self, xs: Iterable[float]) -> None:\n        """\n        Add values to accumulator.\n        """\n        for x in xs:\n            self.n += 1\n            self.s += float(x)\n\n    def mean(self) -> float:\n        """Return mean or 0.0 if empty."""\n        assert self.n >= 0\n        return self.s / self.n if self.n else 0.0\n'
    },
    {
        "id": 7,
        "title": "very_long_function_low_maintainability",
        "code": "def pipeline(a, b, c):\n    r = a + b + c\n    r = r * 2\n    r = r - a\n    r = r / 2\n    r = r + 1\n    r = r * 3\n    r = r - b\n    r = r + c\n    r = r * r\n    r = r - 10\n    r = r + 5\n    r = r * 0.5\n    r = r / 3\n    r = r + 7\n    r = r - 2\n    r = r * 4\n    r = r / 2\n    r = r + 9\n    r = r - 1\n    r = r * 2\n    r = r + 6\n    r = r - 3\n    r = r * 5\n    r = r / 2\n    r = r + 11\n    r = r - 8\n    r = r * 7\n    r = r / 3\n    r = r + 13\n    r = r - 21\n    r = r + 34\n    r = r - 55\n    return r\n"
    },
    {
        "id": 8,
        "title": "minimal_hello_world",
        "code": 'def main():\n    print("hello")\n'
    }
]

def ensure_input_jsonl(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            for rec in SAMPLE_JSONL:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def main():
    ensure_input_jsonl(DEF_IN_PATH)

    storage = FileStorage(
        first_entry_file_name=DEF_IN_PATH,
        cache_path=DEF_CACHE_DIR,
        file_name_prefix=DEF_PREFIX,
        cache_type="jsonl",
    )

    storage.step()                       # step = 0，对应读取初始 JSONL
    _ = storage.read(output_type="dataframe")

    CodeCorrectnessEvaluator().run(storage, code_key="code", output_key="code_correctness")
    storage.step()                       # 进入 step = 1，读取上一写出的缓存

    CodeStabilityEvaluator().run(storage, code_key="code", output_key="code_stability")
    storage.step()                       # 进入 step = 2

    CodeMaintainabilityEvaluator().run(storage, code_key="code", output_key="code_maintainability")
    storage.step()                       # 进入 step = 3，便于读取最终结果

    df_final = storage.read(output_type="dataframe")
    out_path = os.path.join(DEF_CACHE_DIR, f"{DEF_PREFIX}_final.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_final.to_json(out_path, orient="records", lines=True, force_ascii=False)

    cols = ["id", "title", "code_correctness", "code_stability", "code_maintainability"]
    print(df_final[cols].to_string(index=False))
    print(f"\nSaved to: {out_path}")

if __name__ == "__main__":
    main()
