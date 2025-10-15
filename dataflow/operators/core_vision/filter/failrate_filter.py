import pandas as pd
import ray
from typing import Optional, List, Any
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.opencompass_verifier import CV_PROMPT, process_judgment
from dataflow.core import OperatorABC, VLMServingABC
from dataflow.utils.storage import DataFlowStorage


@OPERATOR_REGISTRY.register()
class FailRateFilter(OperatorABC):
    """
    Filter operator that removes successful responses and keeps only failed ones.
    
    Uses a rollout model to generate responses and a verifier model to evaluate them,
    then filters out correct responses to retain only incorrect/failed cases.
    """
    
    def __init__(
        self, 
        rollout: VLMServingABC, 
        verifier: VLMServingABC, 
        system_prompt: str = "You are a helpful assistant."
    ):
        """
        Initialize the FailRateFilter operator.
        
        Args:
            rollout: Vision-language model for generating responses
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
            return "过滤掉正确响应，仅保留错误数据"
        return "Filter out correct responses, keep only failed ones"

    def run(
        self, 
        storage: DataFlowStorage,
        input_image_key: str = "image", 
        input_question_key: str = "question",
        input_answer_key: str = "answer",
        output_answer_key: str = "failrate_answer"
    ) -> None:
        """
        Execute the filtering pipeline.
        
        Process flow:
        1. Load data from storage
        2. Generate responses using rollout model
        3. Verify responses against gold answers
        4. Remove successful responses (judgment == 0.0)
        5. Keep only failed responses (judgment == 1.0)
        
        Args:
            storage: DataFlow storage object for reading/writing data
            input_image_key: Column name for image data
            input_video_key: Column name for video data
            input_audio_key: Column name for audio data
            input_key: Column name for input questions/prompts
            answer_key: Column name for gold/reference answers
        """
        self.logger.info("Running FailRateFilter...")
        
        self.input_image_key = input_image_key
        self.input_question_key = input_question_key
        self.input_answer_key = input_answer_key
        self.output_answer_key = output_answer_key

        # Load dataframe
        dataframe = storage.read('dataframe')
        self.logger.info(f"Loaded {len(dataframe)} rows from storage")
        
        # Extract columns with safe defaults
        image_column = dataframe.get(self.input_image_key, pd.Series([])).tolist()
        user_inputs = dataframe.get(self.input_question_key, pd.Series([])).tolist()
        answers = dataframe.get(self.input_answer_key, pd.Series([])).tolist()
            
        # Generate responses
        self.logger.info("Generating responses with rollout model...")
        if self.is_ray_actor:
            responses = ray.get(self.rollout.generate_from_input_with_message.remote(
                user_inputs=user_inputs,
                image_list=image_column,
                system_prompt=self.system_prompt,
            ))
        else:
            responses = self.rollout.generate_from_input_with_message(
                user_inputs=user_inputs,
                image_inputs=image_column,
                system_prompt=self.system_prompt,
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
        
        # Verify responses
        self.logger.info("Verifying responses...")
        
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
        
        dataframe['dependent'] = judgment
        dataframe = dataframe[dataframe['dependent'] == 1.0]  # Keep questions that need visual input
        dataframe = dataframe.drop(columns=['dependent'])
        
        # Save filtered dataframe
        storage.write(dataframe)