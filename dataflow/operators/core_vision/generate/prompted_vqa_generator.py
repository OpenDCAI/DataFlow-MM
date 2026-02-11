import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger

from dataflow.utils.storage import FileStorage, DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC
from dataflow.serving.local_model_vlm_serving import LocalModelVLMServing_vllm

# 判断是否为 API serving
def is_api_serving(serving):
    """判断 serving 是否为 API 类型"""
    from dataflow.serving.api_vlm_serving_openai import APIVLMServing_openai
    return isinstance(serving, APIVLMServing_openai)

@OPERATOR_REGISTRY.register()
class PromptedVQAGenerator(OperatorABC):
    '''
    PromptedVQAGenerator reads conversation history with optional image/video inputs to generate answers.
    Supports both multimodal mode (with images/videos) and pure text mode (without any visual inputs).
    Works with both API serving and local model serving.
    '''
    def __init__(self, 
                 serving: LLMServingABC, 
                 system_prompt: str = "You are a helpful assistant.",
                 prompt_template = None):
        self.logger = get_logger()
        self.serving = serving
        self.system_prompt = system_prompt
        self.prompt_template = prompt_template
            
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return "读取对话历史和 image/video（可选）生成答案，支持纯文本模式和多模态模式"
        else:
            return "Read conversation history with optional image/video to generate answers. Supports both pure text and multimodal modes."
    
    def run(self, 
            storage: DataFlowStorage,
            input_conversation_key: str = "conversation",
            input_prompt_key: str = None,  # If provided, create conversations from this column
            input_image_key: str = None,  # Can be None for pure text mode
            input_video_key: str = None,  # Can be None for pure text mode
            output_answer_key: str = "answer",
            ):
        if output_answer_key is None:
            raise ValueError("output_answer_key must be provided.")

        self.logger.info("Running PromptedVQA with conversation mode...")
        
        # Load the raw dataframe from the input file
        dataframe = storage.read('dataframe')
        self.logger.info(f"Loading, number of rows: {len(dataframe)}")

        # Handle conversation creation from prompt column if specified
        if input_prompt_key is not None and input_prompt_key in dataframe.columns:
            self.logger.info(f"Creating conversations from '{input_prompt_key}' column...")
            prompts = dataframe[input_prompt_key].tolist()
            conversations_raw = [[{"from": "human", "value": p}] for p in prompts]
        else:
            # Load conversations
            conversations_raw = dataframe.get(input_conversation_key, pd.Series([])).tolist()
            
            # If prompt_template is provided, automatically inject prompts into conversations
            if self.prompt_template is not None:
                self.logger.info("Auto-injecting prompts from template into conversations...")
                for conv in conversations_raw:
                    if isinstance(conv, list) and len(conv) > 0:
                        first = conv[0]
                        if isinstance(first, dict) and "value" in first:
                            # Generate prompt and set it as the first message
                            prompt = self.prompt_template.build_prompt()
                            first["value"] = prompt

        # Handle image column: if input_image_key is None or key not in dataframe, treat as no images
        if input_image_key is None or input_image_key not in dataframe.columns:
            image_column = None
        else:
            image_column = dataframe.get(input_image_key, pd.Series([])).tolist()
            image_column = [path if isinstance(path, list) else [path] for path in image_column]
            if len(image_column) == 0 or all(p is None for p in image_column):
                image_column = None

        # Handle video column: if input_video_key is None or key not in dataframe, treat as no videos
        if input_video_key is None or input_video_key not in dataframe.columns:
            video_column = None
        else:
            video_column = dataframe.get(input_video_key, pd.Series([])).tolist()
            video_column = [path if isinstance(path, list) else [path] for path in video_column]
            if len(video_column) == 0 or all(p is None for p in video_column):
                video_column = None
        
        # Prepare image and video inputs
        image_inputs_list = image_column
        video_inputs_list = video_column
        
        # If all image/video inputs are None, pass None to enable pure text mode
        if image_inputs_list is not None and all(img is None for img in image_inputs_list):
            image_inputs_list = None
        if video_inputs_list is not None and all(vid is None for vid in video_inputs_list):
            video_inputs_list = None
        
        # 判断 serving 类型并相应地准备输入
        use_api_mode = is_api_serving(self.serving)
        
        if use_api_mode:
            # API 模式：转换格式为 {"role": "user/assistant", "content": "..."}
            self.logger.info("Using API serving mode with generate_from_input_messages")
            
            conversations_list = []
            for conv_raw in conversations_raw:
                conversation = []
                if isinstance(conv_raw, list):
                    for turn in conv_raw:
                        if isinstance(turn, dict):
                            # Convert from/value format to role/content format
                            role = "user" if turn.get("from") == "human" else "assistant"
                            content = turn.get("value", "")
                            conversation.append({"role": role, "content": content})
                conversations_list.append(conversation)
            
            outputs = self.serving.generate_from_input_messages(
                conversations=conversations_list,
                image_list=image_inputs_list,
                video_list=video_inputs_list,
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
                            
                            # Add image tokens
                            if image_inputs_list and idx < len(image_inputs_list) and image_inputs_list[idx]:
                                # Filter out None values
                                valid_images = [img for img in image_inputs_list[idx] if img is not None]
                                if valid_images:
                                    tokens.extend(["<image>"] * len(valid_images))
                            
                            # Add video tokens
                            if video_inputs_list and idx < len(video_inputs_list) and video_inputs_list[idx]:
                                # Filter out None values
                                valid_videos = [vid for vid in video_inputs_list[idx] if vid is not None]
                                if valid_videos:
                                    tokens.extend(["<video>"] * len(valid_videos))
                            
                            # Combine tokens with original value
                            if tokens:
                                new_value = "".join(tokens) + value
                                turn = {**turn, "value": new_value}
                        
                        conversation.append(turn)
                conversations_with_tokens.append(conversation)
            
            outputs = self.serving.generate_from_input_messages(
                conversations=conversations_with_tokens,
                image_list=image_inputs_list,
                video_list=video_inputs_list,
                system_prompt=self.system_prompt
            )
        
        # Save outputs to dataframe
        dataframe[output_answer_key] = outputs
        output_file = storage.write(dataframe)
        self.logger.info(f"Results saved to {output_file}")

        return output_answer_key
    
if __name__ == "__main__":
    # Initialize model
    model = LocalModelVLMServing_vllm(
        hf_model_name_or_path="Qwen/Qwen2.5-VL-3B-Instruct",
        vllm_tensor_parallel_size=1,
        vllm_temperature=0.7,
        vllm_top_p=0.9,
        vllm_max_tokens=512,
    )

    generator = PromptedVQAGenerator(
        serving=model,
        system_prompt="You are a helpful assistant.",
    )

    # Prepare input
    storage = FileStorage(
        first_entry_file_name="./dataflow/example/video_caption/sample_data.json", 
        cache_path="./cache_prompted_vqa",
        file_name_prefix="prompted_vqa",
        cache_type="json",
    )
    storage.step()  # Load the data

    generator.run(
        storage=storage,
        input_conversation_key="conversation",
        input_image_key="image",
        input_video_key="video",
        output_answer_key="caption",
    )
