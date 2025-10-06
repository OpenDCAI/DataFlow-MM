from dataflow.utils.storage import FileStorage
from dataflow.operators.core_audio import CTCForcedAlignFilter
from dataflow.wrapper import BatchWrapper

class testCTCForcedAlignFilter:
    def __init__(self):
        self.storage = FileStorage(
            first_entry_file_name="./dataflow/example/forced_alignment/sample_data_local.jsonl",
            cache_path="./cache",
            file_name_prefix="forced_alignment_filter",
            cache_type="jsonl",
        )
        
        self.filter = CTCForcedAlignFilter(
            model_path="/mnt/public/data/guotianyu/Models/mms-300m-1130-forced-aligner",
            device=["cuda:0", "cuda:1", "cuda:2", "cuda:3", "cuda:4", "cuda:5", "cuda:6", "cuda:7"],
            num_workers=16,
        )
    
    def forward(self):
        self.filter.run(
            storage=self.storage.step(),
            input_audio_key='audio',
            input_conversation_key='conversation',
            language="en",  
            micro_batch_size=16,
            chinese_to_pinyin=False,
            retain_word_level_alignment=True,
            threshold=0.000,
            threshold_mode="min"    
        )
        self.filter.close()

if __name__ == "__main__":
    pipline = testCTCForcedAlignFilter()
    pipline.forward()