'''
A collection of prompts for the video operators.
'''

from typing import List, Dict, Any


class VideoCaptionGeneratorPrompt:
    '''
    The prompt for the video quality evaluator.
    '''
    def __init__(self):
        pass

    def build_prompt(self,) -> str:
        """
        Generate system prompt for video quality evaluation.
        """
        prompt = (
            "Please describe the video in detail."
        )
        return prompt
    

class VideoQAGeneratorPrompt:
    def __init__(self):
        pass

    def build_prompt(self, caption: str) -> str:
        return (
            "### Task:\n"
            "Given a detailed description that summarizes the content of a video, generate question-answer pairs "
            "based on the description to help humans better understand the video.\n"
            "The question-answer pairs should be faithful to the content of the video description and developed "
            "from different dimensions to promote comprehensive understanding of the video.\n\n"
            "#### Guidelines For Question-Answer Pairs Generation:\n"
            "- Read the provided video description carefully. Pay attention to the scene, main characters, "
            "their behaviors, and the development of events.\n"
            "- Generate appropriate question-answer pairs based on the description. The pairs should cover "
            "as many question dimensions as possible and not deviate from the content.\n"
            "- Generate 5 to 10 question-answer pairs across different dimensions.\n\n"
            "### Output Format:\n"
            "1. Your output should be formatted as a JSON list.\n"
            "2. Only provide the Python dictionary string.\n"
            "Your response should look like:\n"
            "[\n"
            "  {\"Dimension\": <dimension-1>, \"Question\": <question-1>, \"Answer\": <answer-1>},\n"
            "  {\"Dimension\": <dimension-2>, \"Question\": <question-2>, \"Answer\": <answer-2>},\n"
            "  ...\n"
            "]\n\n"
            "Please generate question-answer pairs for the following video description:\n"
            f"Description: {caption}"
        )



class VideoCOTQAGeneratorPrompt:
    '''
    The prompt for the video Chain-of-Thought QA generator.
    '''
    def __init__(self):
        # Type-specific answer format templates
        self.type_template = {
            "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
            "numerical": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
            "OCR": " Please transcribe text from the image/video clearly and provide your text answer within the <answer> </answer> tags.",
            "free-form": " Please provide your text answer within the <answer> </answer> tags.",
            "regression": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags."
        }
    
    def build_prompt(self, **kwargs) -> str:
        """
        Generate prompt for CoT QA generation with question parameter.
        """
        question = kwargs.get('Question', '')
        prompt = (
            f"{question}\n"
            "Please think about this question as if you were a human pondering deeply. "
            "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions "
            "It's encouraged to include self-reflection or verification in the reasoning process. "
            "Provide your detailed reasoning between the <think> and </think> tags, and then give your final answer between the <answer> and </answer> tags."
        )
        return prompt


class DiyVideoPrompt:
    '''
    The prompt for custom code operations.
    '''
    def __init__(self, prompt_template: str):
        self.prompt_template = prompt_template
    
    def build_prompt(self, **kwargs) -> str:
        """
        Generate prompt using custom template.
        """
        try:
            return self.prompt_template.format(**kwargs)
        except Exception as e:
            # If formatting fails, return the original template
            return self.prompt_template
