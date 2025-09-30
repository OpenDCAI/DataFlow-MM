import ast
import logging
import math
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY


def _safe_str(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return ""
    return str(x)


def _count_lines(code: str) -> int:
    return len(code.splitlines())


def _iter_func_defs(tree: ast.AST):
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield n


def _iter_class_defs(tree: ast.AST):
    for n in ast.walk(tree):
        if isinstance(n, ast.ClassDef):
            yield n


def _has_docstring(node: ast.AST) -> bool:
    return ast.get_docstring(node) is not None


def _typehint_coverage(func: ast.AST) -> Tuple[int, int]:
    total = 0
    hit = 0
    if isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
        for a in func.args.args + func.args.kwonlyargs:
            total += 1
            if a.annotation is not None:
                hit += 1
        if func.args.vararg is not None:
            total += 1
            if func.args.vararg.annotation is not None:
                hit += 1
        if func.args.kwarg is not None:
            total += 1
            if func.args.kwarg.annotation is not None:
                hit += 1
        total += 1
        if func.returns is not None:
            hit += 1
    return hit, total


def _avg_func_length(tree: ast.AST) -> float:
    lengths = []
    for f in _iter_func_defs(tree):
        if hasattr(f, "body") and len(f.body) > 0:
            first = f.body[0].lineno
            last = f.body[-1].lineno
            lengths.append(max(1, last - first + 1))
    if not lengths:
        return 0.0
    return float(np.mean(lengths))


def _complexity_proxy(code: str) -> float:
    tokens = re.findall(r"\b(if|for|while|and|or|try|except|with|assert|lambda|elif|case|match)\b", code)
    n = len(tokens)
    L = max(1, _count_lines(code))
    return n / L


def _exception_stats(code: str) -> Tuple[int, int]:
    try_cnt = len(re.findall(r"\btry\b", code))
    except_cnt = len(re.findall(r"\bexcept\b", code))
    bare_cnt = len(re.findall(r"\bexcept\s*:\s*", code))
    return max(try_cnt, except_cnt), bare_cnt


def _assert_density(code: str) -> float:
    cnt = len(re.findall(r"\bassert\b", code))
    L = max(1, _count_lines(code))
    return cnt / L


def _dangerous_calls(code: str) -> int:
    pats = [
        r"\beval\s*\(",
        r"\bexec\s*\(",
        r"\bos\.system\s*\(",
        r"\bsubprocess\.(Popen|call|run|check_call|check_output)\s*\(",
        r"\bpickle\.loads\s*\(",
        r"\byaml\.load\s*\(",
        r"\bctypes\.",
        r"\bsocket\.",
    ]
    s = 0
    for p in pats:
        s += len(re.findall(p, code))
    return s


def _resource_safety(code: str) -> float:
    with_cnt = len(re.findall(r"\bwith\s+", code))
    open_cnt = len(re.findall(r"\bopen\s*\(", code))
    if open_cnt == 0:
        return 1.0
    return min(1.0, with_cnt / open_cnt)


def _syntax_ok(code: str) -> Tuple[bool, Optional[ast.AST], Optional[Exception]]:
    try:
        tree = ast.parse(code)
        return True, tree, None
    except Exception as e:
        return False, None, e


def _clip(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(max(lo, min(hi, v)))


def _normalize_ratio(hit: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return _clip(hit / total)


def _score_correctness(code: str, tree: Optional[ast.AST], syntax_ok: bool) -> float:
    if not syntax_ok:
        return 0.0
    dens = _assert_density(code)
    cx = _complexity_proxy(code)
    cx_pen = 1.0 - _clip(cx / 0.15)
    dens_boost = _clip(dens / 0.02)
    return _clip(0.5 * 1.0 + 0.3 * cx_pen + 0.2 * dens_boost)


def _score_maintainability(code: str, tree: Optional[ast.AST]) -> float:
    if tree is None:
        return 0.0
    funcs = list(_iter_func_defs(tree))
    classes = list(_iter_class_defs(tree))
    doc_hit = 0
    doc_total = 0
    for f in funcs:
        doc_total += 1
        if _has_docstring(f):
            doc_hit += 1
    for c in classes:
        doc_total += 1
        if _has_docstring(c):
            doc_hit += 1
    doc_cov = _normalize_ratio(doc_hit, doc_total)
    th_hit = 0
    th_total = 0
    for f in funcs:
        h, t = _typehint_coverage(f)
        th_hit += h
        th_total += t
    th_cov = _normalize_ratio(th_hit, th_total)
    avg_len = _avg_func_length(tree)
    len_pen = 1.0 - _clip((avg_len - 60) / 140.0) if avg_len > 60 else 1.0
    return _clip(0.4 * doc_cov + 0.4 * th_cov + 0.2 * len_pen)


def _score_stability(code: str, tree: Optional[ast.AST]) -> float:
    exc_cnt, bare_cnt = _exception_stats(code)
    exc_score = _clip(exc_cnt / 3.0)
    bare_pen = 1.0 - _clip(bare_cnt / 2.0)
    dang = _dangerous_calls(code)
    safe_dep = 1.0 - _clip(dang / 3.0)
    res_safe = _resource_safety(code)
    return _clip(0.35 * exc_score + 0.2 * bare_pen + 0.25 * safe_dep + 0.2 * res_safe)


@OPERATOR_REGISTRY.register()
class CodeCorrectnessEvaluator(OperatorABC):
    def __init__(self, device: Optional[str] = None):
        self.logger = get_logger()
        self.device = device or "cpu"

    @staticmethod
    def get_desc(lang: str = "zh"):
        return "代码功能正确性评估" if lang == "zh" else "Code functional correctness evaluation."

    def run(self, storage: DataFlowStorage, code_key: str = "code", output_key: str = "code_correctness"):
        df = storage.read("dataframe")
        scores: List[float] = []
        for _, row in df.iterrows():
            code = _safe_str(row.get(code_key, ""))
            ok, tree, _ = _syntax_ok(code)
            s = _score_correctness(code, tree, ok)
            scores.append(s)
        df[output_key] = scores
        storage.write(df)
        return [output_key]


@OPERATOR_REGISTRY.register()
class CodeStabilityEvaluator(OperatorABC):
    def __init__(self, device: Optional[str] = None):
        self.logger = get_logger()
        self.device = device or "cpu"

    @staticmethod
    def get_desc(lang: str = "zh"):
        return "代码稳定性与健壮性评估" if lang == "zh" else "Code stability and robustness evaluation."

    def run(self, storage: DataFlowStorage, code_key: str = "code", output_key: str = "code_stability"):
        df = storage.read("dataframe")
        scores: List[float] = []
        for _, row in df.iterrows():
            code = _safe_str(row.get(code_key, ""))
            ok, tree, _ = _syntax_ok(code)
            s = _score_stability(code, tree)
            scores.append(s)
        df[output_key] = scores
        storage.write(df)
        return [output_key]


@OPERATOR_REGISTRY.register()
class CodeMaintainabilityEvaluator(OperatorABC):
    def __init__(self, device: Optional[str] = None):
        self.logger = get_logger()
        self.device = device or "cpu"

    @staticmethod
    def get_desc(lang: str = "zh"):
        return "代码可维护性评估" if lang == "zh" else "Code maintainability evaluation."

    def run(self, storage: DataFlowStorage, code_key: str = "code", output_key: str = "code_maintainability"):
        df = storage.read("dataframe")
        scores: List[float] = []
        for _, row in df.iterrows():
            code = _safe_str(row.get(code_key, ""))
            ok, tree, _ = _syntax_ok(code)
            s = _score_maintainability(code, tree)
            scores.append(s)
        df[output_key] = scores
        storage.write(df)
        return [output_key]


__all__ = [
    "CodeCorrectnessEvaluator",
    "CodeStabilityEvaluator",
    "CodeMaintainabilityEvaluator",
]
