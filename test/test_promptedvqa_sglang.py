from dataflow.operators.generate import PromptedVQA
from dataflow.operators.conversations import Conversation2Message
from dataflow.serving import LocalModelVLMServing_sglang
from dataflow.utils.storage import FileStorage

class VQAGenerator():
    def __init__(self):
        self.storage = FileStorage(
            first_entry_file_name="./dataflow/example/vqa/sample_data.json",
            cache_path="./cache",
            file_name_prefix="vqa",
            cache_type="json",
        )
        self.model_cache_dir = './dataflow_cache'

        self.vlm_serving = LocalModelVLMServing_sglang(
            hf_model_name_or_path="/data0/public_models/Qwen2.5-VL-7B-Instruct",
            hf_cache_dir=self.model_cache_dir,
            sgl_dp_size=2,  # data parallel size
            sgl_tp_size=4,  # tensor parallel size
        )
        # self.format_converter = Conversation2Message(
        #     image_list_key="image",
        #     video_list_key="video",
        #     audio_list_key="audio",
        #     system_prompt="你是个热心的智能助手，是Dataflow的一个组件，你的任务是回答用户的问题。",
        # )
        self.prompt_generator = PromptedVQA(
            vlm_serving = self.vlm_serving,
        )

    def forward(self):
        # Initial filters
        # self.format_converter.run(
        #     storage= self.storage.step(),
        #     input_conversation_key="conversation",
        #     output_message_key="messages",
        # )

        self.prompt_generator.run(
            storage = self.storage.step(),
            input_image_key="image",
            input_video_key="video",
            input_audio_key="audio",
            input_conversation_key="conversation",
            output_answer_key="answer",
        )

if __name__ == "__main__":
    # This is the entry point for the pipeline
    model = VQAGenerator()
    model.forward()  