from dataflow.operators.core_vision import PromptedVQAGenerator
from dataflow.serving import LocalModelVLMServing_vllm
from dataflow.utils.storage import FileStorage
from dataflow.prompts.video import VideoCaptionGeneratorPrompt

class VideoCaptionGenerator():
    def __init__(self):
        self.storage = FileStorage(
            first_entry_file_name="./dataflow/example/video_caption/sample_data.json",
            cache_path="./cache",
            file_name_prefix="video_caption",
            cache_type="json",
        )
        self.model_cache_dir = './dataflow_cache'

        self.vlm_serving = LocalModelVLMServing_vllm(
            hf_model_name_or_path="Qwen/Qwen2.5-VL-7B-Instruct",
            hf_cache_dir=self.model_cache_dir,
            vllm_tensor_parallel_size=1,
            vllm_temperature=0.7,
            vllm_top_p=0.9, 
            vllm_max_tokens=2048,
            vllm_max_model_len=51200,  
            vllm_gpu_memory_utilization=0.9
        )

        self.prompted_vqa_generator = PromptedVQAGenerator(
            serving=self.vlm_serving,
            system_prompt="You are a helpful assistant."
        )

        self.prompt_template = VideoCaptionGeneratorPrompt()

    def forward(self):
        # Load data from storage
        storage = self.storage.step()
        df = storage.read("dataframe")
        
        # Build prompts using the template (same prompt for all rows)
        prompts = [self.prompt_template.build_prompt() for _ in range(len(df))]
        
        # Modify conversation column to set first user message to the prompt
        if "conversation" in df.columns:
            conversations = df["conversation"].tolist()
            for conv, prompt in zip(conversations, prompts):
                if isinstance(conv, list) and conv:
                    first = conv[0]
                    if isinstance(first, dict) and "value" in first:
                        first["value"] = prompt
            df["conversation"] = conversations
        
        # Write modified dataframe back to storage
        storage.write(df)
        
        # Call PromptedVQAGenerator to generate captions
        self.prompted_vqa_generator.run(
            storage=storage.step(),
            input_image_key="image",
            input_video_key="video",
            input_conversation_key="conversation",
            output_answer_key="caption",
        )

if __name__ == "__main__":
    # This is the entry point for the pipeline
    model = VideoCaptionGenerator()
    model.forward()