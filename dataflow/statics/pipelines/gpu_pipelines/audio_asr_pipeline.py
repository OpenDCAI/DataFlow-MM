from dataflow.utils.storage import FileStorage
from dataflow.operators.core_audio import (
    PromptedAQAGenerator,
    TextNormalizer,
    CTCForcedAlignmentSampleEvaluator,
)
from dataflow.serving import LocalModelVLMServing_vllm, APIVLMServing_openai
from dataflow.prompts.audio import WhisperTranscriptionPrompt

class Pipeline:
    def __init__(self):
        self.storage = FileStorage(
            first_entry_file_name="../example_data/audio_asr_pipeline/sample_data.jsonl",
            cache_path="./cache",
            file_name_prefix="audio_asr_pipeline",
            cache_type="jsonl",
        )

        self.serving = LocalModelVLMServing_vllm(
            hf_model_name_or_path="openai/whisper-large-v3",
            hf_cache_dir="./dataflow_cache",
            vllm_tensor_parallel_size=2,
            vllm_temperature=0.3,
            vllm_top_p=0.9,
            vllm_max_model_len=448,
            vllm_gpu_memory_utilization=0.9
        )

        # self.serving = APIVLMServing_openai(
        #     api_url="http://127.0.0.1:8091/v1",
        #     max_workers=3,
        #     model_name="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        # )

        # 对于whisper模型, 使用WhisperTranscriptionPrompt生成prompt
        self.prompted_generator = PromptedAQAGenerator(
            vlm_serving=self.serving,
            system_prompt=WhisperTranscriptionPrompt().generate_prompt(language="english", task="transcribe", with_timestamps=False)
        )

        self.text_normalizer = TextNormalizer(
            language="en",
            remove_puncs=True,
        )

        # self.filter = CTCForcedAlignmentFilter(
        #     model_path="MahmoudAshraf/mms-300m-1130-forced-aligner",
        #     device=["cuda:3"],
        #     num_workers=1,
        #     sampling_rate=16000,
        #     language="en",
        #     micro_batch_size=16,
        #     chinese_to_pinyin=False,
        #     threshold=0.1,
        #     threshold_mode="min",
        #     romanize=True,
        # )


        self.evaluator = CTCForcedAlignmentSampleEvaluator(
            model_path="MahmoudAshraf/mms-300m-1130-forced-aligner",
            device=["cuda:3"],
            num_workers=1,
            sampling_rate=16000,
            language="en",
            micro_batch_size=16,
            chinese_to_pinyin=False,
            romanize=True,
        )
        
    def forward(self):
        self.prompted_generator.run(
            storage=self.storage.step(),
            input_audio_key="audio",
            input_conversation_key="conversation",
            output_answer_key="transcript"
        )

        self.text_normalizer.run(
            storage=self.storage.step(),
            input_text_key="transcript",
        )

        # self.filter.run(
        #     storage=self.storage.step(),
        #     input_audio_key="audio",
        #     input_conversation_key="transcript",
        # )
        # self.filter.close()

        self.evaluator.run(
            storage=self.storage.step(),
            input_audio_key="audio",
            input_conversation_key="transcript",
            output_answer_key="forced_alignment_results",
        )

        self.evaluator.close()

if __name__ == "__main__":
    pipeline = Pipeline()
    pipeline.forward()