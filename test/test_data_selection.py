import os
import ray
from dataflow.operators.core_vision import RuleBaseFilter,DeduplicateFilter,KNNSimilarityFilter,CLIPScoreFilter,DataTailorFilter,VisionDependentFilter,FailRateFilter
from dataflow.serving import LocalModelVLMServing_vllm, LocalModelLLMServing_vllm
from dataflow.utils.storage import FileStorage

ray.init()

@ray.remote(num_gpus=1)
class VLMModelActor:
    def __init__(self, model_path):
        self.model = LocalModelVLMServing_vllm(
            hf_model_name_or_path=model_path,
            vllm_tensor_parallel_size=1,
            vllm_temperature=0.7,
            vllm_top_p=0.9,
            vllm_max_tokens=4096,
            vllm_gpu_memory_utilization=0.85
        )
    
    def generate_from_input_with_message(self, *args, **kwargs):
        return self.model.generate_from_input_with_message(*args, **kwargs)

@ray.remote(num_gpus=1)
class LLMModelActor:
    def __init__(self, model_path):
        self.model = LocalModelLLMServing_vllm(
            hf_model_name_or_path=model_path,
            vllm_tensor_parallel_size=1,
            vllm_temperature=0.7,
            vllm_top_p=0.9,
            vllm_max_tokens=4096,
            vllm_gpu_memory_utilization=0.85
        )
    
    def generate_from_input(self, *args, **kwargs):
        return self.model.generate_from_input(*args, **kwargs)
    
class DataSelectionPipeline():
    def __init__(self):
        self.storage = FileStorage(
            first_entry_file_name="./dataflow/example/image_selection/prompts.jsonl",
            cache_path="./cache_local",
            file_name_prefix="dataflow_cache_step",
            cache_type="jsonl"
        )

        self.rollout = VLMModelActor.remote(
            "Qwen/Qwen2.5-VL-7B-Instruct"
        )
        
        self.verifier = LLMModelActor.remote(
            "opencompass/CompassVerifier-3B"
        )
        
        self.rule_base_filter = RuleBaseFilter()
        
        self.clipscore_filter = CLIPScoreFilter(
            model_name = 'openai/clip-vit-large-patch14-336',
            keep_ratio=0.9,
        )
        
        self.deduplication_filter = DeduplicateFilter(
            model_name = 'openai/clip-vit-large-patch14-336',
            threshold=0.95,
        )
        
        self.knn_similarity_filter = KNNSimilarityFilter(
            model_name = 'openai/clip-vit-large-patch14-336',
            k_neighbors=5,
            keep_ratio=0.90,
        )
        
        # self.datatailor_filter = DataTailorFilter(
        #     model_name = 'Qwen/Qwen2.5-VL-7B-Instruct',
        #     keep_ratio=0.90
        # )
        
        self.vision_dependent_filter = VisionDependentFilter(
            rollout = self.rollout,
            verifier = self.verifier
        )
        
        self.failrate_filter = FailRateFilter(
            rollout = self.rollout,
            verifier = self.verifier
        )
        
    
    def forward(self):
        self.rule_base_filter.run(
            storage=self.storage.step(),
            input_image_key="image",
            input_question_key="question",
            input_answer_key="answer"
        )
        
        self.clipscore_filter.run(
            storage=self.storage.step(),
            input_image_key="image",
            input_question_key="question",
            input_answer_key="answer",
            output_score_key="clipscore",
        )
        
        self.deduplication_filter.run(
            storage=self.storage.step(),
            input_image_key="image",
            output_score_key="deduplicated_image",
        )
        
        self.knn_similarity_filter.run(
            storage=self.storage.step(),
            input_image_key="image",
            output_score_key="knn_similarity_score",
        )
        
        # self.datatailor_filter.run(
        #     storage=self.storage.step(),
        #     input_image_key="image",
        #     input_question_key="question",
        #     input_answer_key="answer"
        # )
        
        self.vision_dependent_filter.run(
            storage=self.storage.step(),
            input_image_key="image",
            input_question_key="question",
            input_answer_key="answer",
            output_answer_key="vision_independent_answer"
        )
        
        self.failrate_filter.run(
            storage=self.storage.step(),
            input_image_key="image",
            input_question_key="question",
            input_answer_key="answer",
            output_answer_key="vision_independent_answer"
        )

if __name__ == "__main__":
    model = DataSelectionPipeline()
    model.forward()
        
        
    