from dataflow.operators.core_vision import VideoToCaptionGenerator, VideoCaptionToQAGenerator
from dataflow.operators.conversations import Conversation2Message
from dataflow.serving import LocalModelVLMServing_vllm
from dataflow.utils.storage import FileStorage

class VideoVQAGenerator():
    def __init__(self):
        self.storage = FileStorage(
            first_entry_file_name="./dataflow/example/video/sample_data.json",
            cache_path="./cache",
            file_name_prefix="video_vqa",
            cache_type="json",
        )
        self.model_cache_dir = './dataflow_cache'

        self.vlm_serving = LocalModelVLMServing_vllm(
            # hf_model_name_or_path="/data0/models/Qwen2.5-VL-7B-Instruct",
            hf_model_name_or_path="Qwen/Qwen2.5-VL-7B-Instruct",
            hf_cache_dir=self.model_cache_dir,
            vllm_tensor_parallel_size=1,
            vllm_temperature=0.7,
            vllm_top_p=0.9, 
            vllm_max_tokens=512,
            vllm_gpu_memory_utilization=0.9
        )

        self.video_to_caption_generator = VideoToCaptionGenerator(
            vlm_serving = self.vlm_serving,
        )
        self.videocaption_to_qa_generator = VideoCaptionToQAGenerator(
            vlm_serving = self.vlm_serving,
        )

    def forward(self):
        # Initial filters
        # self.format_converter.run(
        #     storage= self.storage.step(),
        #     input_conversation_key="conversation",
        #     output_message_key="messages",
        # )

        self.video_to_caption_generator.run(
            storage = self.storage.step(),
            input_image_key="image",
            input_video_key="video",
            input_audio_key="audio",
            input_conversation_key="conversation",
            output_key="caption",
        )

        self.videocaption_to_qa_generator.run(
            storage = self.storage.step(),
            input_image_key="image",
            input_video_key="video",
            input_audio_key="audio",
            input_conversation_key="conversation",
            output_key="qa",
        )

if __name__ == "__main__":
    # This is the entry point for the pipeline
    model = VideoVQAGenerator()
    model.forward()
