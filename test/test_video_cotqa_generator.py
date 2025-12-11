
from dataflow.operators.core_vision import VideoCOTQAGenerator, GeneralTextAnswerEvaluator, ScoreFilter
from dataflow.serving import LocalModelVLMServing_vllm
from dataflow.utils.storage import FileStorage
import os

class VideoCOTQATest:
    def __init__(self):
        # Initialize storage
        self.storage = FileStorage(
            first_entry_file_name="./dataflow/example/video_cot_qa/sample_data.json",
            cache_path="./cache",
            file_name_prefix="video_cotqa_test",
            cache_type="json",
        )
        
        self.model_cache_dir = './dataflow_cache'
        
        # Initialize VLM Serving
        # Note: Adjust model path as needed for your environment
        model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
        
        self.vlm_serving = LocalModelVLMServing_vllm(
            hf_model_name_or_path=model_path,
            hf_cache_dir=self.model_cache_dir,
            vllm_tensor_parallel_size=1,  # Adjust based on available GPUs
            vllm_temperature=1.0,
            vllm_top_p=0.95,
            vllm_max_tokens=2048,
            vllm_max_model_len=51200,
            vllm_gpu_memory_utilization=0.9,
        )

        # Initialize Operators
        self.video_cotqa_generator = VideoCOTQAGenerator(
            vlm_serving=self.vlm_serving,
        )
        
        self.evaluator = GeneralTextAnswerEvaluator(
            use_stemmer=True
        )
        
        self.score_filter = ScoreFilter(
            score_key="reward",
            min_score=0.6,
            output_select_key="select"
        )

    def run(self):
        print("Running VideoCOTQAGenerator pipeline...")
        
        # Step 1: Generate CoT QA responses
        print("\n[Step 1/3] Generating CoT QA responses...")
        answer_key = self.video_cotqa_generator.run(
            storage=self.storage.step(),
            input_video_key="video",
            input_image_key="image",
            input_conversation_key="conversation",
            output_answer_key="answer",
            output_process_key="process",
            output_full_response_key="full_response",
        )
        print(f"Generation finished. Output key: {answer_key}")
        
        # Step 2: Evaluate answers and calculate rewards
        print("\n[Step 2/3] Evaluating answers and calculating rewards...")
        reward_key = self.evaluator.run(
            storage=self.storage.step(),
            model_output_key="full_response",
            gt_solution_key="solution",
            question_type_key="problem_type",
            output_reward_key="reward",
        )
        print(f"Evaluation finished. Output key: {reward_key}")
        
        # Step 3: Filter based on reward threshold
        print("\n[Step 3/3] Filtering based on reward threshold...")
        select_key = self.score_filter.run(
            storage=self.storage.step(),
        )
        print(f"Filtering finished. Output key: {select_key}")
        
        # Verify results
        print("\n" + "="*60)
        print("Final Results:")
        print("="*60)
        result_df = self.storage.step().read("dataframe")
        print(f"Results shape: {result_df.shape}")
        if not result_df.empty:
            print("\nColumns:", result_df.columns.tolist())
            
            # Calculate and display statistics
            if 'reward' in result_df.columns and 'select' in result_df.columns:
                rewards = result_df['reward'].tolist()
                selects = result_df['select'].tolist()
                print(f"\nAverage reward: {sum(rewards)/len(rewards):.4f}")
                print(f"Selected samples: {sum(selects)}/{len(selects)}")
            
            # Print first result samples if available
            print("\nSample results:")
            cols_to_show = ['answer', 'process', 'reward', 'select']
            available_cols = [col for col in cols_to_show if col in result_df.columns]
            print(result_df[available_cols].head())

if __name__ == "__main__":
    # Set visible GPUs if necessary
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    try:
        test = VideoCOTQATest()
        test.run()
    except Exception as e:
        print(f"Test failed with error: {e}")

