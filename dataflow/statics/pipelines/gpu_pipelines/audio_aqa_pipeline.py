from dataflow.utils.storage import FileStorage
from dataflow.operators.core_audio import (
    PromptedAQAGenerator,
)
from dataflow.serving import LocalModelVLMServing_vllm
from dataflow.prompts.audio import AudioCaptionGeneratorPrompt

class Pipeline:
    def __init__(self):
        self.storage = FileStorage(
            first_entry_file_name="../example_data/audio_aqa_pipeline/sample_data.jsonl",
            cache_path="./cache",
            file_name_prefix="audio_aqa_pipeline",
            cache_type="jsonl",
        )

        self.serving = LocalModelVLMServing_vllm(
            hf_model_name_or_path="Qwen/Qwen2-Audio-7B-Instruct",
            hf_cache_dir="./dataflow_cache",
            vllm_tensor_parallel_size=2,
            vllm_temperature=0.3,
            vllm_top_p=0.9,
            vllm_gpu_memory_utilization=0.9
        )
        self.prompted_generator = PromptedAQAGenerator(
            vlm_serving=self.serving,
            system_prompt="You are a helpful assistant."
        )

    def forward(self):
        self.prompted_generator.run(
            storage=self.storage.step(),
            input_audio_key="audio",
            output_answer_key="answer",
        )

if __name__ == "__main__":
    pipeline = Pipeline()
    pipeline.forward()