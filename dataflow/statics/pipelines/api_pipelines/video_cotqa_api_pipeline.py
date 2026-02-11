import os
import re
from dataflow.operators.core_vision import PromptedVQAGenerator, GeneralTextAnswerEvaluator, ScoreFilter
from dataflow.serving.api_vlm_serving_openai import APIVLMServing_openai
from dataflow.utils.storage import FileStorage
from dataflow.prompts.video import VideoCOTQAGeneratorPrompt

class VideoCOTQATest:
    def __init__(self):
        """
        Initialize VideoCOTQATest with API model parameters.
        """
        # Initialize storage
        self.storage = FileStorage(
            first_entry_file_name="./dataflow/example/video_cot_qa/sample_data.json",
            cache_path="./cache",
            file_name_prefix="video_cotqa_api",
            cache_type="json",
        )
        
        # Initialize VLM API serving
        self.vlm_serving = APIVLMServing_openai(
            api_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            key_name_of_api_key="DF_API_KEY",
            model_name="qwen3-vl-8b-instruct",
            image_io=None,
            send_request_stream=False,
            max_workers=10,
            timeout=1800
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
    
    def _process_responses(self, responses):
        """Process CoT QA responses to extract answers and think chains."""
        answers = []
        processes = []
        
        for response in responses:
            # Extract think chain and answer
            think_chain = self._extract_think(response)
            final_ans = self._extract_answer(response)
            
            answers.append(final_ans)
            processes.append(f"<think>{think_chain}</think>" if think_chain else "")
        
        return answers, processes
    
    def _print_results_summary(self, result_df):
        """Print summary of final results."""
        print("\n" + "="*60)
        print("Final Results:")
        print("="*60)
        print(f"Results shape: {result_df.shape}")
        
        if result_df.empty:
            return
        
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

    def run(self):
        print("Running VideoCOTQAGenerator pipeline (API version)...")
        
        # Step 1: Generate CoT QA responses
        print("\n[Step 1/3] Generating CoT QA responses using API...")
        
        # Load data and build prompts
        storage = self.storage.step()
        df = storage.read("dataframe")
        
        # Build prompts and add to dataframe
        prompts = self._build_prompts(df)
        df["prompt"] = prompts
        storage.write(df)
        
        # Use PromptedVQAGenerator to generate responses
        self.prompted_vqa_generator.run(
            storage=storage.step(),
            input_image_key="image",
            input_video_key="video",
            input_prompt_key="prompt",
            output_answer_key="_temp_cotqa_response",
        )
        
        # Read back the results with responses
        storage.step()
        df = storage.read("dataframe")
        responses = df["_temp_cotqa_response"].tolist()
        
        # Process responses - extract think chain and answer
        answers, processes = self._process_responses(responses)
        
        # Attach extracted answers and processes
        df["answer"] = answers
        df["process"] = processes
        df["full_response"] = responses
        storage.write(df)
                
        # Step 2: Evaluate answers and calculate rewards
        print("\n[Step 2/3] Evaluating answers and calculating rewards...")
        self.evaluator.run(
            storage=self.storage.step(),
            input_model_output_key="full_response",
            input_gt_solution_key="solution",
            input_question_type_key="problem_type",
            output_reward_key="reward",
        )
        
        # Step 3: Filter based on reward threshold
        print("\n[Step 3/3] Filtering based on reward threshold...")
        self.score_filter.run(
            storage=self.storage.step(),
            input_score_key="reward",
            output_select_key="select",
        )
        
        # Print results summary
        result_df = self.storage.step().read("dataframe")
        self._print_results_summary(result_df)

if __name__ == "__main__":
    test = VideoCOTQATest()
    test.run()

