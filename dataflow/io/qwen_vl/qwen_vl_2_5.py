from qwen_vl_utils import process_vision_info
import re
from dataflow.utils.registry import IO_REGISTRY
from dataflow.logger import get_logger

@IO_REGISTRY.register()
class Qwen2_5VLIO(object):
    model_str = "qwen2.5-vl"  # 用于匹配模型

    def __init__(self, processor, 
            system_prompt: str = "You are a helpful agent."):
        """
        processor: 传入apply_chat_template等处理文本的工具
        """
        self.processor = processor
        self.logger = get_logger()
        self.system_prompt = system_prompt

    def read_media(self, message):
        """解析单条消息中的多模态信息"""
        image_inputs, video_inputs = process_vision_info(message)
        media_dict = {}
        if image_inputs:
            media_dict['image'] = image_inputs
        if video_inputs:
            media_dict['video'] = video_inputs
        return media_dict

    def build_full_prompts(self, messages):
        """
        输入: messages -> list of list (每个元素是聊天序列)
        输出: full_prompts -> list of dict {prompt, multi_modal_data}
        """
        if not isinstance(messages, list):
            raise ValueError(f"messages must be a list, got {type(messages)}")

        full_prompts = []
        for i, i_message in enumerate(messages):
            if not isinstance(i_message, list):
                raise ValueError(f"Message at index {i} is not a list: {i_message}")

            # 多模态数据解析
            multimodal_entry = self.read_media(i_message)

            # 生成 prompt
            prompt = self.processor.apply_chat_template(
                i_message,
                tokenize=False,
                add_generation_prompt=True
            )

            full_prompts.append({
                'prompt': prompt,
                'multi_modal_data': multimodal_entry
            })

        return full_prompts

    def write_media(self, media_dict):
        raise NotImplementedError("Qwen2_5VLIO does not support write_media operation.")

    def _conversation_to_message(
            self,
            conversations: list[list[dict]],
            image_list: list[list[str]] = None,
            video_list: list[list[str]] = None,
            audio_list: list[list[str]] = None
        ):
        """
        将格式1的数据转换为格式2。
        支持多轮对话和多模态（图片、视频、音频），并验证token数量。
        """
        message_list = []
        for i, conversation in enumerate(conversations):
            # 收集所有模态路径
            all_modal_paths = {
                "image": image_list[i] if image_list else [],
                "video": video_list[i] if video_list else [],
                "audio": audio_list[i] if audio_list else []
            }

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