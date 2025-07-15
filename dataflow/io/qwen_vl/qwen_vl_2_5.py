
from qwen_vl_utils import process_vision_info
from dataflow.utils.registry import IO_REGISTRY
# Qwen 2.5 VL这里只有read，所以都是fetch的形式
# 后续可能在Diffusion中有write的需求

@IO_REGISTRY.register()
class Qwen2_5VLIO(object):
    model_str = "qwen2.5-vl" # use to match models
    def __init__(
        self,
    ):
        pass
    def read_media(self, message):
        image_inputs, video_inputs = process_vision_info(message)
        media_dict = {}
        if image_inputs:
            media_dict['image'] = image_inputs
        if video_inputs:
            media_dict['video'] = video_inputs
        return media_dict
    def write_media(self, media_dict):
        # 目前没有写的需求
        raise NotImplementedError("Qwen2_5VLIO does not support write_media operation.")
