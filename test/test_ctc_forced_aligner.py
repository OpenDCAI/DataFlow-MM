from dataflow.operators.core_audio import CTCForcedAlignSampleEvaluator
from dataflow.operators.conversations import Conversation2Message
from dataflow.serving import LocalModelVLMServing_vllm
from dataflow.utils.storage import FileStorage

class ForcedAlignEval():
    def __init__(self):
        self.storage = FileStorage(
            first_entry_file_name="./dataflow/example/forced_alignment/sample_data_local.jsonl",
            cache_path="./cache",
            file_name_prefix="forced_alignment",
            cache_type="jsonl",
        )

        self.aligner = CTCForcedAlignSampleEvaluator(
            model_path="/mnt/public/data/guotianyu/Models/mms-300m-1130-forced-aligner",
            device="cpu"
        )
    
    def forward(self):
        self.aligner.run(
            storage=self.storage.step(),
            input_audio_key='audio',
            input_conversation_key='conversation',
            output_answer_key="forced_alignment_results",
            language="en",      
            micro_batch_size=16,
            chinese_to_pinyin=False,
            retain_word_level_alignment=True,
        )

if __name__ == "__main__":
    eval = ForcedAlignEval()
    eval.forward()