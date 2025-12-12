import re
import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage

def normalize_whitespace(s: str) -> str:
    """Collapse whitespace to single spaces and trim."""
    return re.sub(r'\s+', ' ', s or '').strip()

def clean_markdown_markers(s: str) -> str:
    """
    清洗字符串中的 markdown 标记，比如 **bold** 或 *italic*。
    用于处理嵌套 bold (e.g. **Title **Sub** End**) 的情况。
    """
    if not s:
        return ""
    # 去掉连续的 * 号
    return re.sub(r'\*+', '', s).strip()

def parse_wiki_qa(text: str) -> dict:
    """
    鲁棒性增强版：
    1. 先物理切分 Article 和 QA 区域。
    2. QA 解析依赖 '行结构' 而非 'Markdown 格式'，解决嵌套加粗问题。
    """
    if not isinstance(text, str) or not text.strip():
        return {"context": "", "qas": []}

    # =======================================================
    # Step 1: 寻找 Context 和 QA 的分界线
    # =======================================================
    # 匹配常见的 QA 标题变体
    split_pattern = re.compile(
        r'(?:^|\n)\s*(?:###|\*\*)\s*(?:Question\s*Answer\s*Pairs|QA|Q&A)(?:\s*\*\*|:)?', 
        flags=re.IGNORECASE
    )
    
    match_split = split_pattern.search(text)
    
    if match_split:
        # 分界线左边是 Context，右边是 QAs
        raw_context = text[:match_split.start()]
        raw_qa_section = text[match_split.end():]
    else:
        # 没找到分界线，假设全是 Context (或者你可以根据业务逻辑改)
        raw_context = text
        raw_qa_section = ""

    # 清洗 Context (去掉开头可能的 ### Article 等标记)
    context_clean = re.sub(r'^\s*###\s*(?:Wikipedia\s+)?Article\s*:?', '', raw_context, flags=re.IGNORECASE)
    context_clean = normalize_whitespace(context_clean)

    # =======================================================
    # Step 2: 鲁棒地解析 QA 部分
    # =======================================================
    qas = []
    if raw_qa_section:
        # 核心逻辑：不依赖 **，而是依赖 "数字." -> "换行" -> "- 答案" 的结构
        # Explanation:
        # (?m)^\s*(\d+)\.\s* -> 多行模式，行首，数字，点，空白
        # (.+?)               -> 捕获问题文本（非贪婪，直到遇到换行）
        # \s*(?:\n|\r)\s* -> 必须换行，且可能有缩进
        # [-–—]\s* -> 破折号或连字符作为答案引导
        # (.*?)               -> 捕获答案文本
        # (?=\n\s*\d+\.|\Z)   -> 向前看：直到遇到下一个 "数字." 或者 字符串结束
        
        qa_structure_pattern = re.compile(
            r'(?m)^\s*(\d+)\.\s*(.+?)\s*(?:\n|\r)\s*[-–—]\s*(.*?)(?=\n\s*\d+\.|\Z)', 
            flags=re.DOTALL
        )
        
        matches = qa_structure_pattern.findall(raw_qa_section)
        
        for _, raw_q, raw_a in matches:
            # 在提取内容后，再清洗 markdown 符号
            cleaned_q = clean_markdown_markers(raw_q)
            cleaned_q = normalize_whitespace(cleaned_q)
            
            cleaned_a = clean_markdown_markers(raw_a)
            cleaned_a = normalize_whitespace(cleaned_a)
            
            if cleaned_q and cleaned_a:
                qas.append({
                    "question": cleaned_q,
                    "answer": cleaned_a
                })

    return {"context": context_clean, "qas": qas}


@OPERATOR_REGISTRY.register()
class WikiQARefiner(OperatorABC):
    """
    文本格式规范化 + WikiQA 格式解析算子。
    """

    def __init__(self):
        self.logger = get_logger()

    @staticmethod
    def get_desc(lang="zh"):
        if lang == "zh":
            return (
                "该算子用于对文本进行格式规范化并解析 WikiQA 结构（Wikipedia Article + QA）。\n\n"
                "输入参数：\n"
                "  - input_key: 输入文本列名（默认: 'text'）\n"
                "  - output_key: 输出结果列名（默认: 'parsed'）\n"
                "输出参数：\n"
                "  - output_key: JSON 格式解析结果 {context, qas}\n"
                "特点：\n"
                "  - 纯文本处理，不依赖 GPU\n"
                "  - 容错解析 WikiQA 文本结构\n"
                "  - 支持批处理 dataframe"
            )
        else:
            return (
                "This operator normalizes raw text and parses WikiQA structure.\n"
                "Pure CPU, no model required."
            )

    def _validate_dataframe(self, df: pd.DataFrame):
        required_keys = [self.input_key]
        conflict = [self.output_key]

        missing = [k for k in required_keys if k not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        exists = [k for k in conflict if k in df.columns]
        if exists:
            raise ValueError(f"Output key already exists: {exists}")

    def run(
        self,
        storage: DataFlowStorage,
        input_key="text",
        output_key="parsed"
    ):
        """
        Reads dataframe -> cleans text -> parses -> writes back -> returns output_key
        """
        self.input_key = input_key
        self.output_key = output_key

        df = storage.read("dataframe")
        self._validate_dataframe(df)

        results = []
        for t in df[input_key].tolist():
            results.append(parse_wiki_qa(t))

        df[output_key] = results

        output_file = storage.write(df)
        self.logger.info(f"[WikiQAParse] Results saved to {output_file}")

        return [output_key]


if __name__ == "__main__":
    # Example usage
    from dataflow.utils.storage import FileStorage

    storage = FileStorage(
        first_entry_file_name="cache_local/context_vqa_step1.jsonl",
        cache_path="./cache_local",
        file_name_prefix="wikiqaparse",
        cache_type="jsonl",
    )
    storage.step()

    op = WikiQARefiner()
    op.run(storage=storage, input_key="vqa", output_key="parsed")
