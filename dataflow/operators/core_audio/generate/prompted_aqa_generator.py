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
        if lang == "zh":
            desc = (
                "该算子用于基于大模型（VLM）和给定的 system_prompt，"
                "对输入的语音与文本对话进行推理并生成回答。\n\n"

                "输入参数（run）：\n"
                "- input_audio_key: 音频路径所在列名，默认 'audio'\n"
                "- input_conversation_key: 文本/对话所在列名，默认 'conversation'\n"
                "- output_answer_key: 输出答案所在列名，默认 'answer'\n\n"

                "运行行为：\n"
                "1. 从 storage 中读取输入 dataframe\n"
                "2. 获取音频列与对话列，传入 VLMServing 的 generate_from_input_messages 接口\n"
                "3. 使用 system_prompt 作为基础提示生成回答\n"
                "4. 将生成的回答写入 dataframe 的 output_answer_key 列\n"
                "5. 覆盖写回 storage\n\n"

                "输出：\n"
                "- 覆盖写回到 storage 的 'dataframe'，新增包含模型回答的 output_answer_key 列。\n"
            )
        else:
            desc = (
                "This operator uses a VLM (Vision-Language/Audio-Language Model) together with a "
                "given system_prompt to generate answers based on input audio and text.\n\n"

                "Input parameters (run):\n"
                "- input_audio_key: Column name containing audio paths, default 'audio'\n"
                "- input_conversation_key: Column name containing conversation text, default 'conversation'\n"
                "- output_answer_key: Column name to store generated answers, default 'answer'\n\n"

                "Runtime behavior:\n"
                "1. Read the input dataframe from storage\n"
                "2. Extract audio paths and conversations, then pass them to the VLMServing's "
                "   generate_from_input_messages method\n"
                "3. Use the configured system_prompt to guide generation\n"
                "4. Store generated answers in the output_answer_key column\n"
                "5. Write the updated dataframe back to storage\n\n"

                "Output:\n"
                "- The 'dataframe' in storage is overwritten with a new column (output_answer_key) "
                "containing model-generated answers.\n"
            )
        return desc

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
