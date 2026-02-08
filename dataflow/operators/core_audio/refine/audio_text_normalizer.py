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