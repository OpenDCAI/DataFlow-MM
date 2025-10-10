import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'  # 设置可见的GPU设备

from dataflow.utils.storage import FileStorage
from dataflow.operators.core_audio import (
    SileroVADGenerator,
    MergeChunksByTimestamps,
    PromptedAQAGenerator,
    CTCForcedAlignFilter,
    CTCForcedAlignSampleEvaluator,
)
from dataflow.serving import LocalModelVLMServing_vllm
from dataflow.prompts.whisper_prompt_generator import WhisperTranscriptionPrompt

class Pipeline:
    def __init__(self):
        self.storage = FileStorage(
            first_entry_file_name="./dataflow/example/audio_asr_pipeline/sample_data_local.jsonl",
            cache_path="./cache",
            file_name_prefix="audio_asr_pipeline",
            cache_type="jsonl",
        )

        self.serving = LocalModelVLMServing_vllm(
            hf_model_name_or_path="/mnt/public/data/guotianyu/Models/whisper-large-v3",
            hf_cache_dir="./dataflow_cache",
            vllm_tensor_parallel_size=2,
            vllm_temperature=0.3,
            vllm_top_p=0.9,
            vllm_max_tokens=512,
            vllm_gpu_memory_utilization=0.9
        )

        self.silero_vad_generator = SileroVADGenerator(
            repo_or_dir="/mnt/public/data/guotianyu/dataflow_project/silero-vad",
            source="local",
            device=['cuda:2'],
            num_workers=2,
        )
        
        self.merger = MergeChunksByTimestamps(num_workers=2)

        self.prompted_generator = PromptedAQAGenerator(
            vlm_serving=self.serving,
            system_prompt=WhisperTranscriptionPrompt().generate_prompt(language="german", task="transcribe", with_timestamps=False),
        )

        # self.filter = CTCForcedAlignFilter(
        #     model_path="/mnt/public/data/guotianyu/Models/mms-300m-1130-forced-aligner",
        #     device=["cuda:3"],
        #     num_workers=1,
        # )

        self.evaluator = CTCForcedAlignSampleEvaluator(
            model_path="/mnt/public/data/guotianyu/Models/mms-300m-1130-forced-aligner",
            device=["cuda:3"],
            num_workers=2,
        )

    def forward(self):
        self.silero_vad_generator.run(
            storage=self.storage.step(),
            input_audio_key='audio',
            output_answer_key='timestamps',
            threshold=0.5,
            use_min_cut=True,
            sampling_rate=16000,
            max_speech_duration_s=30.0,
            min_silence_duration_s=0.1,
            speech_pad_s=0.03,
            return_seconds=True,
            time_resolution=1,
            neg_threshold=0.35,
            window_size_samples=512,
            min_silence_at_max_speech=0.098,
            use_max_poss_sil_at_max_speech=True
        )

        self.silero_vad_generator.close()     # 关闭多进程

        self.merger.run(
            storage=self.storage.step(),
            dst_folder="./cache",
            input_audio_key="audio",
            input_timestamps_key="timestamps",
            timestamp_type="time",  # 手动指定类型
            max_audio_duration=30.0,
            hop_size_samples=512,  # hop_size, 是样本点数量
            sampling_rate=16000,
        )

        self.merger.close()

        self.prompted_generator.run(
            storage=self.storage.step(),
            input_audio_key="audio",
            input_conversation_key="conversation",
            output_answer_key="transcript"
        )

        # self.filter.run(
        #     storage=self.storage.step(),
        #     input_audio_key="audio",
        #     input_conversation_key="transcript",
        #     sampling_rate=16000,
        #     language="de",
        #     micro_batch_size=16,
        #     chinese_to_pinyin=False,
        #     retain_word_level_alignment=True,
        #     threshold=0.1,
        #     threshold_mode="min",
        #     romanize=True,
        # )
        # self.filter.close()

        self.evaluator.run(
            storage=self.storage.step(),
            input_audio_key="audio",
            input_conversation_key="transcript",
            sampling_rate=16000,
            language="de",
            micro_batch_size=16,
            chinese_to_pinyin=False,
            retain_word_level_alignment=True,
            romanize=True,
        )

        self.evaluator.close()

if __name__ == "__main__":
    pipeline = Pipeline()
    pipeline.forward()