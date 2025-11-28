from dataflow.utils.storage import FileStorage
from dataflow.operators.core_audio import CTCForcedAlignmentFilter

class testCTCForcedAlignmentFilter:
    def __init__(self):
        self.storage = FileStorage(
            first_entry_file_name="./dataflow/example/forced_alignment/sample_data_local.jsonl",
            cache_path="./cache",
            file_name_prefix="forced_alignment_filter",
            cache_type="jsonl",
        )
        
        self.filter = CTCForcedAlignmentFilter(
            model_path="/share/project/guotianyu/models/mms-300m-1130-forced-aligner",
            device=["cuda:0"],
            num_workers=1,
            language="en",  
            micro_batch_size=16,
            chinese_to_pinyin=False,
            retain_word_level_alignment=True,
            threshold=0.900,
            threshold_mode="min" 
        )
    
    def forward(self):
        self.filter.run(
            storage=self.storage.step(),
            input_audio_key='audio',
            input_conversation_key='conversation',
   
        )
        self.filter.close()

if __name__ == "__main__":
    pipline = testCTCForcedAlignmentFilter()
    pipline.forward()