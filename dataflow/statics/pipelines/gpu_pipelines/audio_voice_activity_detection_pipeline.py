from dataflow.utils.storage import FileStorage
from dataflow.operators.core_audio import (
    SileroVADGenerator,
    TimestampChunkRowGenerator,
)
from dataflow.serving import LocalModelVLMServing_vllm, APIVLMServing_openai

class Pipeline:
    def __init__(self):
        self.storage = FileStorage(
            first_entry_file_name="../example_data/audio_voice_activity_detection_pipeline/sample_data.jsonl",
            cache_path="./cache",
            file_name_prefix="audio_voice_activity_detection_pipeline",
            cache_type="jsonl",
        )

        self.silero_vad_generator = SileroVADGenerator(
            repo_or_dir="snakers4/silero-vad",
            source="github",
            device=['cuda:0'],
            num_workers=1,
            threshold=0.5,
            sampling_rate=16000,
            max_speech_duration_s=30.0,
            min_silence_duration_s=0.1,
            speech_pad_s=0.03,
            return_seconds=True,
        )
        self.timestamp_chunk_row_generator = TimestampChunkRowGenerator(
            dst_folder="./cache",
            timestamp_unit="second",
            mode="split",
            max_audio_duration=30.0,
            hop_size_samples=512,
            sampling_rate=16000,
            num_workers=1,
        )
    
    def forward(self):
        self.silero_vad_generator.run(
            storage=self.storage.step(),
            input_audio_key='audio',
            output_answer_key='timestamps',
        )
        self.silero_vad_generator.close()

        self.timestamp_chunk_row_generator.run(
            storage=self.storage.step(),
            input_audio_key="audio",
            input_timestamps_key="timestamps",
        )
        self.timestamp_chunk_row_generator.close()

if __name__ == "__main__":
    pipeline = Pipeline()
    pipeline.forward()
