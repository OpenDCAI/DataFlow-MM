from dataflow.operators.core_vision import PromptedVQAGenerator, VideoCaptionToQAGenerator
from dataflow.serving import LocalModelVLMServing_vllm
from dataflow.utils.storage import FileStorage
from dataflow.prompts.video import VideoCaptionGeneratorPrompt

class VideoVQAGenerator():
    def __init__(self):
        """
        Initialize VideoVQAGenerator with default parameters.
        """
        self.storage = FileStorage(
            first_entry_file_name="./dataflow/example/video_caption/sample_data.json",
            cache_path="./cache",
            file_name_prefix="video_vqa",
            cache_type="json",
        )

        self.vlm_serving = LocalModelVLMServing_vllm(
            hf_model_name_or_path="Qwen/Qwen2.5-VL-7B-Instruct",
            hf_cache_dir="./dataflow_cache",
            vllm_tensor_parallel_size=1,
            vllm_temperature=0.7,
            vllm_top_p=0.9, 
            vllm_max_tokens=2048,
            vllm_max_model_len=51200,  
            vllm_gpu_memory_utilization=0.9
        )

        self.prompt_template = VideoCaptionGeneratorPrompt()
        
        self.prompted_vqa_generator = PromptedVQAGenerator(
            serving=self.vlm_serving,
            system_prompt="You are a helpful assistant.",
            prompt_template=self.prompt_template
        )
        
        self.videocaption_to_qa_generator = VideoCaptionToQAGenerator(
            vlm_serving=self.vlm_serving,
            use_video_input=True,
        )

    def forward(self):
        # Step 1: Generate video captions using PromptedVQAGenerator
        self.prompted_vqa_generator.run(
            storage=self.storage.step(),
            input_image_key="image",
            input_video_key="video",
            input_conversation_key="conversation",
            output_answer_key="caption",
        )
        
        # Step 2: Generate QA from captions
        self.videocaption_to_qa_generator.run(
            storage = self.storage.step(),
            input_image_key="image",
            input_video_key="video",
            input_conversation_key="conversation",
            output_key="qa",
        )

if __name__ == "__main__":
    model = VideoVQAGenerator()
    model.forward()
