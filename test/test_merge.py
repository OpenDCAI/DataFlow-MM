from dataflow.utils.storage import FileStorage
from dataflow.operators.core_audio import MergeChunksRowGenerator

class TestMergeChunksRowGenerator:
    def __init__(self):
        self.storage = FileStorage(
            first_entry_file_name="./dataflow/example/merge_chunks/sample_data_local.jsonl",
            cache_path="./cache",
            file_name_prefix="merge_chunks_by_timestamps",
            cache_type="jsonl",
        )

        self.merger = MergeChunksRowGenerator(
            num_workers=2,
            dst_folder="./cache",
            timestamp_type="time",  # 手动指定类型
            max_audio_duration=30,
            hop_size_samples=512,  # hop_size, 是样本点数量
            sampling_rate=16000,
        )

    def forward(self):
        self.merger.run(
            storage=self.storage.step(),
            input_audio_key="audio",
            input_timestamps_key="timestamps",
        )
        self.merger.close()

if __name__ == "__main__":
    pipeline = TestMergeChunksRowGenerator()
    pipeline.forward()