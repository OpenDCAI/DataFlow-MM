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
    if not isinstance(text, str) or not text.strip():
        return {"context": "", "qas": []}

    # 1. 寻找 Context 和 QA 的分界线
    # 兼容: Question Answer Pairs, Q&A, QA, 以及中英文标点
    split_pattern = re.compile(
        r'(?i)(?:\n|^)\s*(?:###|\*\*|---)?\s*(?:Question[-–—\s]*Answer\s*Pairs|Q&A|QA|Questions?)\s*(?::|\*\*|---)?',
    )
    
    match_split = split_pattern.search(text)
    if match_split:
        raw_context = text[:match_split.start()]
        raw_qa_section = text[match_split.end():]
    else:
        # Fallback: 寻找第一个出现 "Question:" 或 "Q:" 的地方
        fallback_match = re.search(r'(?i)(?:\n|^)\s*(?:-\s*)?(?:Question|Q)\s*[:：]', text)
        if fallback_match:
            raw_context = text[:fallback_match.start()]
            raw_qa_section = text[fallback_match.start():]
        else:
            return {"context": normalize_whitespace(text), "qas": []}

    # 清洗 Context (移除引导词和末尾分割线)
    context_clean = re.sub(r'(?i)^\s*(?:###\s*)?(?:Wikipedia\s+)?Article\s*:?', '', raw_context).strip()
    context_clean = re.sub(r'\s*---+\s*$', '', context_clean)
    context_clean = normalize_whitespace(context_clean)

    # 2. 解析 QA 列表 (贪婪块匹配)
    qas = []
    # 步骤 A: 将 QA 区域切分成一个个独立的 QA 块
    # 匹配模式：以 "数字." 或 "- Question" 或 "Q:" 开头的部分
    qa_blocks = re.split(r'(?m)^\s*(?:\d+[\.\)]|[-•*]\s*)?(?:Question|Q)\s*[:：]?', raw_qa_section)
    
    # 第一个切分出来的一般是空字符串或标题余项，删掉
    for block in qa_blocks:
        if not block.strip():
            continue
            
        # 步骤 B: 在每个块内部寻找 Answer
        # 兼容: "Answer: xxx", "- Answer: xxx", "\n- A: xxx"
        ans_match = re.search(r'(?i)(?:\n|^|\s+)(?:-\s*)?(?:Answer|A)\s*[:：]\s*(.*)', block, re.DOTALL)
        
        if ans_match:
            # Answer 之前的部分就是 Question
            question_part = block[:ans_match.start()]
            answer_part = ans_match.group(1)
            
            # 清洗
            q = normalize_whitespace(clean_markdown_markers(question_part))
            a = normalize_whitespace(clean_markdown_markers(answer_part))
            
            if q and a:
                qas.append({"question": q, "answer": a})

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
        first_entry_file_name="cache_local/context_vqa_step1.json",
        cache_path="./cache_local",
        file_name_prefix="wikiqaparse",
        cache_type="json",
    )
    storage.step()

    op = WikiQARefiner()
    op.run(storage=storage, input_key="vqa", output_key="context_vqa")
