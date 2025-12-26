import json
import random
from typing import List, Dict, Any, Tuple

from dataflow import get_logger
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC


def _process_text_chain(chain: List[str]) -> Tuple[str, str]:
    if not chain: 
        return "", ""
    if chain[0].startswith("<image>") or chain[0].endswith("<image>"):
        chain = chain[1:]
    if not chain:
        return "", ""
        
    final_answer = chain[-1].replace("<answer>", "").replace("</answer>", "").strip()
    chain_content = chain[:-1]
    cleaned = []
    for line in chain_content:
        line = line.replace("<think>", "").replace("</think>", "")
        line = line.replace("<answer>", "").replace("</answer>", "")
        cleaned.append(line.strip())
    joined_chain = " ".join(cleaned)
    return joined_chain, final_answer


def _build_reasoning_chains_from_rollouts(
    node_data: Dict[str, Any],
    backtrack_message: str = "Wait, this seems off. Let's try something else.",
) -> List[str]:
    rollouts = node_data.get("rollouts", [])
    correct_rollouts, wrong_rollouts = [], []
    for r in rollouts:
        (correct_rollouts if r.get("reward", 0.0) >= 1.0 else wrong_rollouts).append(r)
    
    child_nodes = node_data.get("children", [])
    is_terminal = node_data.get("is_terminal", False)
    all_chains = []
    
    # 构造 "错误尝试 -> 回溯 -> 正确尝试" 的链
    for wr in wrong_rollouts:
        wc, _ = _process_text_chain(wr.get("ephemeral_texts", []))
        if not wc: continue
        wc += f"\n{backtrack_message}"
        for cr in correct_rollouts:
            cc, ca = _process_text_chain(cr.get("ephemeral_texts", []))
            c = f"<think>\n{wc}\n{cc}\n</think>\n<answer> {ca} </answer>"
            all_chains.append(c)
            
    # 构造 "直接正确" 的链
    for cr in correct_rollouts:
        cc, ca = _process_text_chain(cr.get("ephemeral_texts", []))
        c = f"<think>\n{cc}\n</think>\n<answer> {ca} </answer>"
        all_chains.append(c)
        
    if not is_terminal:
        for child in child_nodes:
            all_chains.extend(_build_reasoning_chains_from_rollouts(child, backtrack_message))
            
    return all_chains


@OPERATOR_REGISTRY.register()
class MCTSTreeRefiner(OperatorABC):
    """
    [Refine] 解析 MCTS 树结构，提取推理链。
    """
    def __init__(self, max_chains_per_sample: int = 10000, seed: int = 42):
        self.max_chains = max_chains_per_sample
        self.rng = random.Random(seed)
        self.logger = get_logger()

    @staticmethod
    def get_desc(lang: str = "zh"):
        return "解析 MCTS 搜索树，提取 '错误-回溯-正确' 或 '直接正确' 的推理链。" if lang == "zh" else "Parses MCTS trees to extract reasoning chains."

    def run(self, storage: DataFlowStorage, input_tree_key: str, output_key: str):
        df = storage.read("dataframe")
        results = []
        
        for idx, row in df.iterrows():
            tree_obj = row.get(input_tree_key)
            chains = []
            
            if tree_obj:
                try:
                    if isinstance(tree_obj, str):
                        tree_obj = json.loads(tree_obj)
                    
                    raw_chains = _build_reasoning_chains_from_rollouts(tree_obj)
                    chains = list(set(raw_chains))
                    
                    if len(chains) > self.max_chains:
                        chains = self.rng.sample(chains, self.max_chains)
                except Exception as e:
                    self.logger.warning(f"Failed to parse tree at index {idx}: {e}")
                    chains = []
            
            results.append(chains)

        df[output_key] = results
        storage.write(df)
        return [output_key]
