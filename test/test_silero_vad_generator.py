from dataflow.utils.storage import FileStorage
from dataflow.operators.core_audio import SileroVADGenerator
# from dataflow.operators.core_audio import MergeChunksByTimestamps

class SileroVADGeneratorEval:
    def __init__(self):
        self.storage = FileStorage(
            first_entry_file_name="./dataflow/example/silero_vad/sample_data_local.jsonl",
            cache_path="./cache",
            file_name_prefix="silero_vad_mp_test",
            cache_type="jsonl",
        )

        self.silero_vad_generator = SileroVADGenerator(
            repo_or_dir="/mnt/public/data/guotianyu/dataflow_project/silero-vad",
            source="local",
            device=['cuda:0'],
            #device=['cuda:1', 'cuda:0'],
            #device="cuda:0",
            num_workers=1,
        )
    
    def forward(self):
        self.silero_vad_generator.run(
            storage=self.storage.step(),
            input_audio_key='audio',
            output_answer_key='timestamps',
            threshold=0.5,
            use_min_cut=False,
            sampling_rate=16000,
            min_speech_duration_s=0.25,
            max_speech_duration_s=float('inf'),
            min_silence_duration_s=0.1,
            speech_pad_s=0.03,
            return_seconds=True,
            time_resolution=1,
            #visualize_probs=False,
            neg_threshold=None,
            window_size_samples=512,
            min_silence_at_max_speech=98,
            use_max_poss_sil_at_max_speech=True,
        )

    
if __name__ == "__main__":
    pipline = SileroVADGeneratorEval()
    pipline.forward()