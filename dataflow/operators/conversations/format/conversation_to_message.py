import re
import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger

from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC

@OPERATOR_REGISTRY.register()
class Conversation2Message(OperatorABC):
    '''
    Conversation2Message is a class that generates messages from conversations.
    '''
    def __init__(
            self,
            image_list_key: str = "image",
            video_list_key: str = "video",
            audio_list_key: str = "audio",
            system_prompt: str = "You are a helpful agent."
            ):
        self.logger = get_logger()
        self.image_list_key = image_list_key
        self.video_list_key = video_list_key
        self.audio_list_key = audio_list_key
        self.system_prompt = system_prompt

    @staticmethod
    def get_desc(lang: str = "zh"):
        return "基于prompt生成数据" if lang == "zh" else "Generate data from prompt."

    def _parse_multimodal_tokens(self, text):
        """
        解析文本中的多模态token，并返回它们的计数和去除token后的文本。
        """
        image_count = len(re.findall(r"<image>", text))
        video_count = len(re.findall(r"<video>", text))
        audio_count = len(re.findall(r"<audio>", text))

        cleaned_text = text.replace("<image>", "").replace("<video>", "").replace("<audio>", "").strip()
        # 移除可能因为token移除而产生的多余换行符
        cleaned_text = re.sub(r'\n+', '\n', cleaned_text).strip()

        return {
            "image": image_count,
            "video": video_count,
            "audio": audio_count
        }, cleaned_text

    def _conversation_to_message(self, conversation_list):
        """
        将格式1的数据转换为格式2。
        支持多轮对话和多模态（图片、视频、音频），并验证token数量。
        """
        message_list = []
        for item1 in conversation_list:
            # 收集所有模态路径
            all_modal_paths = {
                "image": item1.get(self.image_list_key, []),
                "video": item1.get(self.video_list_key, []),
                "audio": item1.get(self.audio_list_key, [])
            }
            
            conversation = item1[self.input_conversation_key]

            messages = [
                {
                    "role": "system",
                    "content": self.system_prompt # 这里的system content可能需要根据实际情况调整
                }
            ]

            # 用于跟踪已使用的模态文件索引
            used_modal_indices = {"image": 0, "video": 0, "audio": 0}

            for turn in conversation:
                role = "user" if turn["from"] == "human" else "assistant"
                content_list = []
                
                # 解析多模态token
                token_counts, cleaned_value = self._parse_multimodal_tokens(turn["value"])

                # 添加模态内容
                for modal_type in ["image", "video", "audio"]:
                    for _ in range(token_counts[modal_type]):
                        if used_modal_indices[modal_type] < len(all_modal_paths[modal_type]):
                            content_list.append({
                                "type": modal_type,
                                "image": all_modal_paths[modal_type][used_modal_indices[modal_type]] # 这里的key统一用image，因为格式2的例子是image_path
                            })
                            used_modal_indices[modal_type] += 1
                        else:
                            raise ValueError(f"模态类型 {modal_type} 的token数量与提供的文件数量不匹配！")
                
                # 添加文本内容
                if cleaned_value:
                    content_list.append({"type": "text", "text": cleaned_value})
                
                # 如果没有文本内容也没有模态内容，则跳过此轮（或根据需求处理）
                if not content_list:
                    continue

                messages.append({"role": role, "content": content_list})
            message_list.append(messages)
        return message_list
    
    def run(self, 
            storage: DataFlowStorage,
            input_conversation_key: str = "conversation", 
            output_message_key: str = "message",
            ):
        self.logger.info("Running Conversation2Message...")
        self.input_conversation_key = input_conversation_key
        self.output_message_key = output_message_key
        # Load the raw dataframe from the input file
        dataframe = storage.read('dataframe')

        # convert dataframe to list of dicts

        conversation_list = dataframe.to_dict(orient="records")
        print(conversation_list, "conversation_list")

        converted_messages_list = self._conversation_to_message(conversation_list)
        print(converted_messages_list, "converted_messages")

        # insert the converted messages into the dataframe
        dataframe[self.output_message_key] = pd.Series(converted_messages_list)
        storage.write(dataframe)

        self.logger.info(f"Loading, number of rows: {len(dataframe)}")


        # print(dataframe)
        # # print each line of the dataframe
        # for index, row in dataframe.iterrows():
        #     print(f"Row {index}: {row.to_dict()}")
        # # get each column with a list

        # image_column = dataframe.get(self.input_image_key, pd.Series([])).tolist()
        # video_column = dataframe.get(self.input_video_key, pd.Series([])).tolist()
        # audio_column = dataframe.get(self.input_audio_key, pd.Series([])).tolist()
        # conversation_column = dataframe.get(self.input_conversation_key, pd.Series([])).tolist()

        # print(f"Image column: {image_column}")
        # print(f"Video column: {video_column}")
        # print(f"Audio column: {audio_column}")
        # print(f"Conversation column: {conversation_column}")

        # resoponse = self.vlm_serving.generate_from_input_conversations(
        #     conversations=conversation_column,
        #     image_list=image_column,
        #     video_list=video_column,
        #     audio_list=audio_column
        # )

        # return response
