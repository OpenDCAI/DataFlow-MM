import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger

from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import VLMServingABC

@OPERATOR_REGISTRY.register()
class PromptedVQA(OperatorABC):
    '''
    PromptedVQA is a class that generates answers for questions based on provided context.
    '''
    def __init__(self, vlm_serving: VLMServingABC, system_prompt: str = "You are a helpful agent."):
        self.logger = get_logger()
        self.vlm_serving = vlm_serving
        self.system_prompt = system_prompt
    
    @staticmethod
    def get_desc(lang: str = "zh"):
        return "基于prompt生成数据" if lang == "zh" else "Generate data from prompt."

    def run(self, 
            storage: DataFlowStorage,
            # input_image_key: str = "image", 
            # input_video_key: str = "video",
            # input_audio_key: str = "audio",
            input_conversation_key: str = "conversation",
            # 输出的conversation可能是none也可能是conversation，请类型检查
            output_answer_key: str = "answer",
            ):
        if output_answer_key is None:
            raise ValueError("At least one of output_answer_key must be provided.")

        self.logger.info("Running PromptedVQA...")
        # self.input_image_key = input_image_key
        # self.input_video_key = input_video_key
        # self.input_audio_key = input_audio_key
        self.input_conversation_key = input_conversation_key
        self.output_answer_key = output_answer_key
        # self.output_conversation_key = output_conversation_key

        self.logger.info("Running PromptGenerator...")

        # Load the raw dataframe from the input file
        dataframe = storage.read('dataframe')
        self.logger.info(f"Loading, number of rows: {len(dataframe)}")


        # print(dataframe)
        # # print each line of the dataframe
        # for index, row in dataframe.iterrows():
        #     print(f"Row {index}: {row.to_dict()}")
        # # get each column with a list

        # image_column = dataframe.get(self.input_image_key, pd.Series([])).tolist()
        # video_column = dataframe.get(self.input_video_key, pd.Series([])).tolist()
        # audio_column = dataframe.get(self.input_audio_key, pd.Series([])).tolist()
        # conversations = []
        # for _, row in dataframe.iterrows():
        #     conversations.append({
        #         "conversation": row["conversation"],
        #         "image": row.get("image", []),
        #         "video": row.get("video", []),
        #         "audio": row.get("audio", [])
        #     })
        conversation_list = dataframe.to_dict(orient="records")
        #conversations = dataframe.get(self.input_conversation_key, pd.Series([])).tolist()

        # print(f"Image column: {image_column}")
        # print(f"Video column: {video_column}")
        # print(f"Audio column: {audio_column}")
        # print(f"Conversation column: {messages}")

        response = self.vlm_serving.generate_from_input_messages(
            conversations=conversation_list,
            # image_list=image_column,
            # video_list=video_column,
            # audio_list=audio_column
        )
        dataframe[self.output_answer_key] = response
        storage.write(dataframe)
        return output_answer_key
