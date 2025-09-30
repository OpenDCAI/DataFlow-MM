import os
from dataflow.operators.core_vision.filter.curator_filter import CuratorFilter
from dataflow.operators.core_vision.filter.deduplication_filter import DeduplicateFilter
from dataflow.operators.core_vision.filter.knn_similarity_filter import KNNSimilarityFilter
from dataflow.operators.core_vision.filter.clipscore_filter import CLIPScoreFilter
from dataflow.operators.core_vision.filter.datatailor_filter import DataTailorFilter
from dataflow.operators.core_vision.filter.vision_dependent_filter import VisionDependentFilter
from dataflow.operators.core_vision.filter.failrate_filter import FailRateFilter
from dataflow.serving import LocalModelVLMServing_vllm, LocalModelLLMServing_vllm
from dataflow.utils.storage import FileStorage


class DataSelectionPipeline():
    def __init__(self):
        self.storage = FileStorage(
            first_entry_file_name="./dataflow/example/eval/prompts.jsonl",
            cache_path="./cache_local",
            file_name_prefix="dataflow_cache_step",
            cache_type="jsonl"
        )

        # self.rollout = LocalModelVLMServing_vllm(
        #     hf_model_name_or_path="/mnt/dhwfile/liuzheng/petrelfs/DCVLR/VLMEvalKit/models/qwen2.5vl-7b",
        #     vllm_tensor_parallel_size=1,
        #     vllm_temperature=0.7,
        #     vllm_top_p=0.9, 
        #     vllm_max_tokens=512,
        #     vllm_gpu_memory_utilization=0.4
        # )
        
        # self.verifier = LocalModelLLMServing_vllm(
        #     hf_model_name_or_path="/mnt/dhwfile/liuzheng/petrelfs/DCVLR/fail_score/models/CompassVerifier-3B",
        #     vllm_tensor_parallel_size=1,
        #     vllm_temperature=0.7,
        #     vllm_top_p=0.9, 
        #     vllm_max_tokens=512,
        #     vllm_gpu_memory_utilization=0.4
        # )
        
        # self.curator_filter = CuratorFilter()
        
        # self.clipscore_filter = CLIPScoreFilter(
        #     model_name = '/mnt/dhwfile/liuzheng/hwfile/model/clip-vit-large-patch14-336',
        #     keep_ratio=0.8,
        # )
        
        # self.deduplication_filter = DeduplicateFilter(
        #     model_name = '/mnt/dhwfile/liuzheng/hwfile/model/clip-vit-large-patch14-336',
        #     threshold=0.90,
        # )
        
        # self.knn_similarity_filter = KNNSimilarityFilter(
        #     model_name = '/mnt/dhwfile/liuzheng/hwfile/model/clip-vit-large-patch14-336',
        #     k_neighbors=5,
        #     keep_ratio=0.8,
        # )
        
        self.datatailor_filter = DataTailorFilter(
            model_name = '/mnt/dhwfile/liuzheng/petrelfs/DCVLR/VLMEvalKit/models/qwen2.5vl-7b',
            keep_ratio=0.8
        )
        # self.vision_dependent_filter = VisionDependentFilter(
        #     rollout = self.rollout,
        #     verifier = self.verifier
        # )
        
        # self.failrate_filter = FailRateFilter(
        #     rollout = self.rollout,
        #     verifier = self.verifier
        # )
        
    
    def forward(self):
        # self.curator_filter.run(
        #     storage=self.storage.step(),
        #     input_image_key="image",
        #     input_question_key="question",
        #     input_answer_key="answer"
        # )
        
        # self.clipscore_filter.run(
        #     storage=self.storage.step(),
        #     input_image_key="image",
        #     input_question_key="question",
        #     input_answer_key="answer",
        #     output_score_key="clipscore",
        # )
        
        # self.deduplication_filter.run(
        #     storage=self.storage.step(),
        #     input_image_key="image",
        #     output_score_key="deduplicated_image",
        # )
        
        # self.knn_similarity_filter.run(
        #     storage=self.storage.step(),
        #     input_image_key="image",
        #     output_score_key="knn_similarity_score",
        # )
        
        self.datatailor_filter.run(
            storage=self.storage.step(),
            input_image_key="image",
            input_question_key="question"
        )
        
        # self.vision_dependent_filter.run(
        #     storage=self.storage.step(),
        #     input_image_key="image",
        #     input_question_key="question",
        #     input_answer_key="answer",
        #     output_answer_key="vision_independent_answer"
        # )
        
        # self.failrate_filter.run(
        #     storage=self.storage.step(),
        #     input_image_key="image",
        #     input_question_key="question",
        #     input_answer_key="answer",
        #     output_answer_key="vision_independent_answer"
        # )

if __name__ == "__main__":
    model = DataSelectionPipeline()
    model.forward()
        
        
    