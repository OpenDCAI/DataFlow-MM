import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger

from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import VLMServingABC

@OPERATOR_REGISTRY.register()
class PromptedAQAGenerator(OperatorABC):
    '''
    PromptedVQA is a class that generates answers for questions based on provided context.
    '''
    def __init__(self, vlm_serving: VLMServingABC, system_prompt: str = "You are a helpful assistant."):
        self.logger = get_logger()
        self.vlm_serving = vlm_serving
        self.system_prompt = system_prompt
    
    @staticmethod
    def get_desc(lang: str = "zh"):
        return "基于prompt生成数据" if lang == "zh" else "Generate data from prompt."

    def run(self, 
            storage: DataFlowStorage,
            input_audio_key: str = "audio",
            input_conversation_key: str = "conversation",
            # 输出的conversation可能是none也可能是conversation，请类型检查
            output_answer_key: str = "answer",
            ):
        if output_answer_key is None:
            raise ValueError("At least one of output_answer_key must be provided.")

        self.logger.info("Running PromptedAQAGenerator...")
        self.input_audio_key = input_audio_key
        self.input_conversation_key = input_conversation_key
        self.output_answer_key = output_answer_key


        # Load the raw dataframe from the input file
        dataframe = storage.read('dataframe')
        self.logger.info(f"Loading, number of rows: {len(dataframe)}")

        audio_column = dataframe.get(self.input_audio_key, pd.Series([])).tolist()

        conversations = dataframe.get(self.input_conversation_key, pd.Series([])).tolist()

        response = self.vlm_serving.generate_from_input_messages(
            conversations=conversations,
            audio_list=audio_column,
            system_prompt=self.system_prompt,
        )
        dataframe[self.output_answer_key] = response
        storage.write(dataframe)
        return output_answer_key
