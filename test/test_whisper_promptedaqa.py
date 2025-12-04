from dataflow.operators.core_audio import PromptedAQAGenerator

from dataflow.serving import LocalModelVLMServing_vllm
from dataflow.utils.storage import FileStorage
from dataflow.prompts.audio import WhisperTranscriptionPrompt

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # 设置可见的GPU设备

class AQAGenerator():
    def __init__(self):
        self.storage = FileStorage(
            first_entry_file_name="./dataflow/example/whisper_transcription/sample_data_local.jsonl",
            cache_path="./cache",
            file_name_prefix="whisper_transcription_aqa",
            cache_type="jsonl",
        )

        self.vlm_serving = LocalModelVLMServing_vllm(
            hf_model_name_or_path="openai/whisper-large-v3",
            hf_cache_dir='./dataflow_cache',
            vllm_tensor_parallel_size=2,
            vllm_temperature=0.3,
            vllm_top_p=0.9,
            vllm_max_tokens=512,
            vllm_max_model_len=448,
            vllm_gpu_memory_utilization=0.9
        )

        self.prompt_generator = PromptedAQAGenerator(
            vlm_serving = self.vlm_serving,
            system_prompt=WhisperTranscriptionPrompt().generate_prompt(language="mandarin", task="transcribe", with_timestamps=False)  # 使用 Whisper 的转录提示
        )

    def forward(self):

        self.prompt_generator.run(
            storage = self.storage.step(),
            input_audio_key="audio",
            input_conversation_key="conversation",
            output_answer_key="answer",
        )

if __name__ == "__main__":
    # This is the entry point for the pipeline
    model = AQAGenerator()
    model.forward()
