from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger

from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC

from whisper_normalizer.english import EnglishTextNormalizer
from whisper_normalizer.basic import BasicTextNormalizer
from .cn_tn import TextNorm
import pandas as pd
import re

PUNCS = r"[，。！？；：、,.!?;:'\"“”‘’（）()\[\]{}《》<>【】…—\-·/\\|@#$%^&*_+=~]"
def clean_punctuations(s: str) -> str:
    s = "" if s is None else str(s)
    s = re.sub(PUNCS, "", s)
    return s

@OPERATOR_REGISTRY.register()
class TextNormalizer(OperatorABC):
    def __init__(self,
        language: str = "en",
        remove_puncs: bool = False,
        **kwargs,
    ):
        self.logger = get_logger(__name__)
        self.language = language
        self.remove_puncs = remove_puncs

        if language == "en":
            self.text_normalizer = EnglishTextNormalizer()
        elif language == "zh":
            self.text_normalizer = TextNorm(
                to_banjiao = kwargs.get("to_banjiao", False),
                to_upper = kwargs.get("to_upper", False),
                to_lower = kwargs.get("to_lower", False),
                remove_fillers = kwargs.get("remove_fillers", False),
                remove_erhua = kwargs.get("remove_erhua", False),
                check_chars = kwargs.get("check_chars", False),
                remove_space = kwargs.get("remove_space", False),
                cc_mode = kwargs.get("cc_mode", ''),
            )
        else:
            self.text_normalizer = BasicTextNormalizer()

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "TextNormalizer（文本归一化算子）\n"
                "----------------------------\n"
                "功能简介：\n"
                "该算子对 DataFrame 指定列的文本进行归一化处理。根据 language 选择不同的归一化器：\n"
                "- 英文（en）：使用 EnglishTextNormalizer；\n"
                "- 中文（zh）：使用自定义 TextNorm（支持多种可选规则）；\n"
                "- 其他语言：使用 BasicTextNormalizer。\n"
                "可选开启 remove_puncs 以额外删除常见中英文标点。\n\n"

                "一、__init__ 初始化接口\n"
                "def __init__(\n"
                "    self,\n"
                "    language: str = \"en\",\n"
                "    remove_puncs: bool = False,\n"
                "    **kwargs,\n"
                ")\n\n"
                "参数说明：\n"
                "- language：文本语言标识。\n"
                "  * \"en\"：使用 EnglishTextNormalizer（Whisper normalizer）；\n"
                "  * \"zh\"：使用 TextNorm（cn_tn），可通过 kwargs 配置具体规则；\n"
                "  * 其他：使用 BasicTextNormalizer。\n"
                "- remove_puncs：是否在归一化后额外移除标点（使用内置正则 PUNCS）。\n"
                "- kwargs：仅在 language=\"zh\" 时生效，传给 TextNorm 的可选配置项：\n"
                "  * to_banjiao / to_upper / to_lower\n"
                "  * remove_fillers / remove_erhua\n"
                "  * check_chars / remove_space\n"
                "  * cc_mode\n\n"
                "初始化行为：\n"
                "- 创建 logger；\n"
                "- 根据 language 选择并初始化对应的 text_normalizer；\n"
                "- 保存 remove_puncs 配置。\n\n"

                "二、run 运行接口\n"
                "def run(\n"
                "    self,\n"
                "    storage: DataFlowStorage,\n"
                "    input_text_key: str = \"text\",\n"
                ")\n\n"
                "输入说明：\n"
                "- storage：DataFlowStorage，要求其中 key='dataframe' 存在一个 DataFrame。\n"
                "- input_text_key：需要归一化的文本列名（默认 \"text\"）。\n\n"
                "输出说明：\n"
                "- 将归一化后的文本写回 dataframe[input_text_key] 并通过 storage.write 持久化；\n"
                "- 返回 input_text_key 作为算子输出键。\n"
            )
        else:
            return (
                "TextNormalizer (Text Normalization Operator)\n"
                "-------------------------------------------\n"
                "Overview:\n"
                "This operator normalizes text in a specified DataFrame column. It selects a normalizer\n"
                "based on the language:\n"
                "- English (\"en\"): EnglishTextNormalizer;\n"
                "- Chinese (\"zh\"): custom TextNorm (cn_tn) with configurable rules;\n"
                "- Others: BasicTextNormalizer.\n"
                "Optionally, it can remove common punctuation characters after normalization.\n\n"

                "1) __init__ interface\n"
                "def __init__(\n"
                "    self,\n"
                "    language: str = \"en\",\n"
                "    remove_puncs: bool = False,\n"
                "    **kwargs,\n"
                ")\n\n"
                "Parameters:\n"
                "- language: language identifier for choosing the normalizer.\n"
                "  * \"en\": EnglishTextNormalizer (Whisper normalizer);\n"
                "  * \"zh\": TextNorm (cn_tn), configurable via kwargs;\n"
                "  * otherwise: BasicTextNormalizer.\n"
                "- remove_puncs: if True, removes punctuation characters using the built-in regex PUNCS.\n"
                "- kwargs: only effective when language=\"zh\"; passed to TextNorm, including:\n"
                "  * to_banjiao / to_upper / to_lower\n"
                "  * remove_fillers / remove_erhua\n"
                "  * check_chars / remove_space\n"
                "  * cc_mode\n\n"
                "Initialization behavior:\n"
                "- Creates a logger.\n"
                "- Instantiates the appropriate normalizer based on language.\n"
                "- Stores remove_puncs.\n\n"

                "2) run interface\n"
                "def run(\n"
                "    self,\n"
                "    storage: DataFlowStorage,\n"
                "    input_text_key: str = \"text\",\n"
                ")\n\n"
                "Inputs:\n"
                "- storage: DataFlowStorage containing a DataFrame under key='dataframe'.\n"
                "- input_text_key: name of the column to normalize (default \"text\").\n\n"
                "Outputs:\n"
                "- Writes normalized strings back to dataframe[input_text_key] and persists via storage.write.\n"
                "- Returns input_text_key as the operator output key.\n"
            )


    def run(self, storage: DataFlowStorage, input_text_key: str = "text"):
        self.logger.info(f"Normalizing text...")
        dataframe = storage.read('dataframe')
        self.logger.info(f"Loading, number of rows: {len(dataframe)}")
        
        texts = dataframe.get(input_text_key, pd.Series([])).tolist()

        normalized_texts = []
        for text in texts:
            text = self.text_normalizer(text)
            if self.remove_puncs:
                text = clean_punctuations(text)
            normalized_texts.append(text)
        dataframe[input_text_key] = normalized_texts
        storage.write(dataframe)
        return input_text_key