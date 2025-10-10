# -*- coding: utf-8 -*-
import os
import sys
import json
import types
import importlib.util
import pandas as pd
from dataflow.serving.local_model_vlm_serving import LocalModelVLMServing_vllm

class InMemoryStorage:
    def __init__(self, df: pd.DataFrame, first_entry_file_name: str):
        self._df = df.copy()
        self.first_entry_file_name = first_entry_file_name  # 供算子决定输出目录
    def read(self, name: str):
        return self._df.copy()
    def write(self, df: pd.DataFrame):
        self._df = df.reset_index(drop=True)
        return self._df


def _install_dummy_registry():
    if "dataflow.utils.registry" in sys.modules:
        return
    dummy = types.ModuleType("dataflow.utils.registry")
    class _DummyRegistry:
        def register(self, *a, **kw):
            def deco(cls): 
                return cls
            return deco
    dummy.OPERATOR_REGISTRY = _DummyRegistry()
    sys.modules["dataflow.utils.registry"] = dummy

def load_operator_from_file(py_path: str):
    _install_dummy_registry()
    spec = importlib.util.spec_from_file_location("vision_mcts_generate_mod", py_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.VisionMCTSReasoningSFTGenerate

def main():
    # 可选：走 hf-mirror 下载 72B
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

    img_path = "/data0/happykeyan/DataFlow-MM/Dataflow-MM-Preview/dataflow/example/test_image_editing/images/image3.png"
    df = pd.DataFrame({
        "question": ["请指出图片中项链吊坠的中心坐标。"],
        "image": [img_path],
        "true_answer": ["[420, 560]"]
    })

    # 仅用于算子确定输出目录 reasoning_chains/*
    jsonl_stub = "/data0/happykeyan/workspace/DataFlow-MM/dataflow/example/image_to_text_pipeline/test_mct.jsonl"
    os.makedirs(os.path.dirname(jsonl_stub), exist_ok=True)
    with open(jsonl_stub, "w", encoding="utf-8") as f:
        f.write(json.dumps(df.iloc[0].to_dict(), ensure_ascii=False) + "\n")

    storage = InMemoryStorage(df, first_entry_file_name=jsonl_stub)

    # 按文件路径加载算子（不会触发 generate 目录的懒加载）
    op_file = "/data0/happykeyan/DataFlow-MM/Dataflow-MM-Preview/dataflow/operators/generate/image_caption/vision_mct_reasoning_sft_generator.py"
    VisionMCTSReasoningSFTGenerate = load_operator_from_file(op_file)

    # vLLM 模型（HF 仓库 ID，可配合 hf-mirror）
    model = LocalModelVLMServing_vllm(
        hf_model_name_or_path="/data0/happykeyan/Models/Qwen2.5-VL-3B-Instruct",
        vllm_tensor_parallel_size=1,   # 按你的 GPU 数量改
        vllm_temperature=0.7,
        vllm_top_p=0.9,
        vllm_max_tokens=1024,
    )

    op = VisionMCTSReasoningSFTGenerate(
        llm_serving=model,
        prompt_type="web_grounding",
        val_size=0.0,
        log_to_wandb=False,
        max_samples_per_file=10000,
        draw_points=True,
    )

    op.run(
        storage=storage,
        question_key="question",
        image_key="image",
        tree_key="tree",               
        true_answer_key="true_answer",
        output_key="sft_entry",
    )

    result_df = storage.read("dataframe")
    print(result_df[["question", "image", "sft_entry"]])

    out_dir = os.path.join(os.path.dirname(jsonl_stub), "reasoning_chains")
    for fn in ["reasoning_chains_train.json", "reasoning_chains_val.json", "reasoning_chains.txt"]:
        p = os.path.join(out_dir, fn)
        print(p, "-> exists:", os.path.exists(p))

if __name__ == "__main__":
    main()
