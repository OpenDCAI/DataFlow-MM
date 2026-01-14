
from dataflow.operators.core_vision import PromptedVQAGenerator, GeneralTextAnswerEvaluator, ScoreFilter
from dataflow.serving import LocalModelVLMServing_vllm
from dataflow.utils.storage import FileStorage
from dataflow.prompts.video import VideoCOTQAGeneratorPrompt
import os
import re

class VideoCOTQATest:
    def __init__(self):
        # Initialize storage
        self.storage = FileStorage(
            first_entry_file_name="./dataflow/example/video_cot_qa/sample_data.json",
            cache_path="./cache",
            file_name_prefix="video_cotqa",
            cache_type="json",
        )
        
        self.model_cache_dir = './dataflow_cache'
        
        self.vlm_serving = LocalModelVLMServing_vllm(
            hf_model_name_or_path="Qwen/Qwen2.5-VL-7B-Instruct",
            hf_cache_dir=self.model_cache_dir,
            vllm_tensor_parallel_size=1,  # Adjust based on available GPUs
            vllm_temperature=1.0,
            vllm_top_p=0.95,
            vllm_max_tokens=2048,
            vllm_max_model_len=51200,
            vllm_gpu_memory_utilization=0.9,
        )

        # Initialize Operators
        self.prompted_vqa_generator = PromptedVQAGenerator(
            serving=self.vlm_serving,
            system_prompt="You are a helpful assistant."
        )
        self.prompt_template = VideoCOTQAGeneratorPrompt()
        
        self.evaluator = GeneralTextAnswerEvaluator(
            use_stemmer=True
        )
        
        self.score_filter = ScoreFilter(
            min_score=0.6,
        )

    @staticmethod
    def _extract_think(output_str: str) -> str:
        """Extract content between <think> and </think> tags."""
        pattern = r'<think>\s*(.*?)\s*</think>'
        match = re.search(pattern, output_str, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    @staticmethod
    def _extract_answer(text: str) -> str:
        """Extract content between <answer> and </answer> tags."""
        pattern = r'<answer>\s*(.*?)\s*</answer>'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def _build_prompts(self, df):
        """Build one prompt per row using the template and problem information."""
        prompts = []
        for _, row in df.iterrows():
            problem_type = row.get('problem_type', '')
            problem = row.get('problem', '')
            options = row.get('options', [])
            
            # Format question with options if multiple choice
            if problem_type == 'multiple choice' and options:
                question = problem + "Options:\n"
                for op in options:
                    question += op + "\n"
            else:
                question = problem
            
            # Build prompt with type-specific suffix
            type_template = getattr(self.prompt_template, 'type_template', {})
            type_suffix = type_template.get(problem_type, "")
            prompt = self.prompt_template.build_prompt(Question=question) + type_suffix
            prompts.append(prompt)
        
        return prompts

    @staticmethod
    def _set_first_user_message(conversation, value: str):
        """Safely set the first user message's 'value' in a conversation."""
        try:
            if isinstance(conversation, list) and conversation:
                first = conversation[0]
                if isinstance(first, dict) and "value" in first:
                    first["value"] = value
        except Exception:
            pass
        return conversation

    def run(self):
        print("Running VideoCOTQAGenerator pipeline...")
        
        # Step 1: Generate CoT QA responses
        print("\n[Step 1/3] Generating CoT QA responses...")
        
        # Load data and build prompts
        storage = self.storage.step()
        df = storage.read("dataframe")
        
        # Build prompts
        prompts = self._build_prompts(df)
        
        # Create or update conversations
        if "conversation" not in df.columns or df["conversation"].isna().all():
            # Create default conversations
            df["conversation"] = [
                [{"from": "human", "value": prompt}] for prompt in prompts
            ]
        else:
            # Update existing conversations
            df["conversation"] = [
                self._set_first_user_message(conv, prompt)
                for conv, prompt in zip(df["conversation"].tolist(), prompts)
            ]
        # import ipdb;ipdb.set_trace()
        # Write the modified dataframe back to storage
        storage.write(df)
        
        # Use PromptedVQAGenerator to generate responses
        temp_response_key = "_temp_cotqa_response"
        self.prompted_vqa_generator.run(
            storage=storage.step(),
            input_image_key="image",
            input_video_key="video",
            input_conversation_key="conversation",
            output_answer_key=temp_response_key,
        )
        
        # Read back the results with responses
        storage.step()
        df = storage.read("dataframe")
        responses = df[temp_response_key].tolist()
        
        # Process responses - extract think chain and answer
        answers = []
        processes = []
        
        for response in responses:
            # Extract think chain and answer
            think_chain = self._extract_think(response)
            final_ans = self._extract_answer(response)
            
            answers.append(final_ans)
            processes.append(f"<think>{think_chain}</think>" if think_chain else "")
        
        # Attach extracted answers and processes
        df["answer"] = answers
        df["process"] = processes
        df["full_response"] = responses
        
        # Clean up temporary column
        df = df.drop(columns=[temp_response_key])
        storage.write(df)
        
        answer_key = "answer"
        print(f"Generation finished. Output key: {answer_key}")
        
        # Step 2: Evaluate answers and calculate rewards
        print("\n[Step 2/3] Evaluating answers and calculating rewards...")
        reward_key = self.evaluator.run(
            storage=self.storage.step(),
            input_model_output_key="full_response",
            input_gt_solution_key="solution",
            input_question_type_key="problem_type",
            output_reward_key="reward",
        )
        print(f"Evaluation finished. Output key: {reward_key}")
        
        # Step 3: Filter based on reward threshold
        print("\n[Step 3/3] Filtering based on reward threshold...")
        select_key = self.score_filter.run(
            storage=self.storage.step(),
            input_score_key="reward",
            output_select_key="select",
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

