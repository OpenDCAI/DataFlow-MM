import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger

from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import VLMServingABC

def is_api_serving(serving):
    """判断 serving 是否为 API 类型"""
    from dataflow.serving.api_vlm_serving_openai import APIVLMServing_openai
    return isinstance(serving, APIVLMServing_openai)

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

        if input_audio_key is None or input_audio_key not in dataframe.columns:
            audio_column = None
        else:
            audio_column = dataframe.get(input_audio_key, pd.Series([])).tolist()
            audio_column = [path if isinstance(path, list) else [path] for path in audio_column]
            if len(audio_column) == 0 or all(p is None for p in audio_column):
                audio_column = None

        audio_inputs_list = audio_column
        if audio_inputs_list is not None and all(aud is None for aud in audio_inputs_list):
            audio_inputs_list = None

        conversations_raw = dataframe.get(self.input_conversation_key, pd.Series([])).tolist()

        use_api_mode = is_api_serving(self.serving)

        if use_api_mode:
            conversations_list = []
            for conv in conversations_raw:
                conversation = []
                if isinstance(conv, list):
                    for turn in conv:
                        if isinstance(turn, dict):
                            # Convert from/value format to role/content format
                            role = "user" if turn.get("from") == "human" else "assistant"
                            content = turn.get("value", "")
                            conversation.append({"role": role, "content": content})
                conversations_list.append(conversation)
            
            response= self.serving.generate_from_input_messages(
                conversations=conversations_list,
                audio_list=audio_inputs_list,
                system_prompt=self.system_prompt
            )
        else:
            # Local 模式：保持原始格式 {"from": "human/gpt", "value": "..."}
            # 但需要注入 <image> 和 <video> tokens 供 IO 层识别
            self.logger.info("Using local serving mode with generate_from_input_messages")
            
            # Inject multimodal tokens into the first user message if needed
            conversations_with_tokens = []
            for idx, conv_raw in enumerate(conversations_raw):
                conversation = []
                for turn_idx, turn in enumerate(conv_raw):
                    if isinstance(turn, dict):
                        # Check if this is the first user message
                        is_first_user = turn.get("from") == "human" and turn_idx == 0
                        
                        if is_first_user:
                            # Inject tokens before the text
                            value = turn.get("value", "")
                            tokens = []
                            
                            if audio_inputs_list and idx < len(audio_inputs_list) and audio_inputs_list[idx]:
                                # Filter out None values
                                valid_audios = [aud for aud in audio_inputs_list[idx] if aud is not None]
                                if valid_audios:
                                    tokens.extend(["<audio>"] * len(valid_audios))
                            
                            # Combine tokens with original value
                            if tokens:
                                new_value = "".join(tokens) + value
                                turn = {**turn, "value": new_value}
                        
                        conversation.append(turn)
                conversations_with_tokens.append(conversation)
            
            response = self.serving.generate_from_input_messages(
                conversations=conversations_with_tokens,
                audio_list=audio_inputs_list,
                system_prompt=self.system_prompt
            )
        
        dataframe[self.output_answer_key] = response
        storage.write(dataframe)
        return output_answer_key
