import pandas as pd
import ray
from typing import Optional, List, Any
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.opencompass_verifier import CV_PROMPT, process_judgment
from dataflow.core import OperatorABC, VLMServingABC
from dataflow.utils.storage import DataFlowStorage


@OPERATOR_REGISTRY.register()
class VisionDependentFilter(OperatorABC):
    """
    Filter to identify questions that genuinely require visual context.
    
    Tests if questions can be answered correctly without visual input,
    keeping only those that fail without images (vision-dependent questions).
    """
    
    def __init__(
        self, 
        rollout: VLMServingABC, 
        verifier: VLMServingABC, 
        system_prompt: str = "You are a helpful assistant."
    ):
        """
        Initialize the VisionDependentFilter operator.
        
        Args:
            rollout: Model for generating responses (will run without visual input)
            verifier: Model for verifying response quality against ground truth
            system_prompt: System prompt for the rollout model
        """
        self.logger = get_logger()
        self.rollout = rollout
        self.verifier = verifier
        self.system_prompt = system_prompt
        self.is_ray_actor = isinstance(verifier, ray.actor.ActorHandle)
    
    @staticmethod
    def get_desc(lang: str = "zh") -> str:
        if lang == "zh":
            return "过滤出真正需要视觉信息的问题"
        return "Filter for questions that truly require visual information"

    def run(
        self, 
        storage: DataFlowStorage,
        input_image_key: str = "image", 
        input_question_key: str = "question",
        input_answer_key: str = "answer",
        output_answer_key: str = "vision_independent_answer"
    ) -> None:
        """
        Execute the vision dependency filtering pipeline.
        
        Process flow:
        1. Load data from storage
        2. Generate responses WITHOUT visual input (text-only)
        3. Verify if text-only responses are correct
        4. Remove questions answered correctly without visual context
        5. Keep only vision-dependent questions (failed without images)
        
        Args:
            storage: DataFlow storage object for reading/writing data
            input_image_key: Column name for image data
            input_question_key: Column name for input questions/prompts
            input_answer_key: Column name for gold/reference answers
        """
        self.logger.info("Running VisionDependentFilter...")
        
        self.input_image_key = input_image_key
        self.input_question_key = input_question_key
        self.input_answer_key = input_answer_key
        self.output_answer_key = output_answer_key
        
        # Load dataframe
        dataframe = storage.read('dataframe')
        self.logger.info(f"Loaded {len(dataframe)} rows from storage")

        # Extract columns (visual data extracted but intentionally not used)
        user_inputs = dataframe.get(self.input_question_key, pd.Series([])).tolist()
        answers = dataframe.get(self.input_answer_key, pd.Series([])).tolist()
        
        # Generate responses WITHOUT visual input to test vision dependency
        self.logger.info("Generating text-only responses (no visual input)...")
        
        if self.is_ray_actor:
            responses = ray.get(self.rollout.generate_from_input_with_message.remote(
                user_inputs=user_inputs,
                system_prompt=self.system_prompt,
                # Intentionally omitting image_list, video_list, audio_list
            ))
        else:
            responses = self.rollout.generate_from_input_with_message(
                user_inputs=user_inputs,
                system_prompt=self.system_prompt,
                # Intentionally omitting image_list, video_list, audio_list
            )
        
        dataframe[self.output_answer_key] = responses
        
        # Prepare verification prompts
        verify_inputs = []
        for user_input, answer, response in zip(user_inputs, answers, responses):
            verify_input = CV_PROMPT.format(
                question=user_input, 
                gold_answer=answer, 
                llm_response=response
            )
            verify_inputs.append(verify_input)
        
        # Verify text-only responses
        self.logger.info("Verifying text-only responses...")
        
        if self.is_ray_actor:
            verify_responses = ray.get(self.verifier.generate_from_input.remote(
                user_inputs=verify_inputs,
                system_prompt=self.system_prompt,
            ))
        else:
            verify_responses = self.verifier.generate_from_input(
                user_inputs=verify_inputs,
                system_prompt=self.system_prompt,
            )
        
        # Process judgments: 0.0 = correct without vision, 1.0 = failed without vision
        judgment = [
            0.0 if process_judgment(verify_response) == "A" else 1.0 
            for verify_response in verify_responses
        ]
        
        dataframe['independent'] = judgment
        dataframe = dataframe[dataframe['independent'] == 1.0]  # Keep questions that need visual input
        dataframe = dataframe.drop(columns=['independent'])
        
        # Save filtered dataframe
        storage.write(dataframe)