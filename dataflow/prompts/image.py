from typing import Literal


class CaptionGeneratorPrompt:
    '''
    The prompt for the AutoPromptGenerator.
    '''
    def __init__(self):
        pass

    def build_prompt(self) -> str:
        prompt = "Please provide a comprehensive description of the image."

        system_prompt = f'''You are a image caption generator. Your task is to generate a concise and informative caption for the given image content.'''

        return prompt, system_prompt

class QAGeneratorPrompt:
    '''
    The prompt for the AutoPromptGenerator.
    '''
    def __init__(self):
        pass

    def build_prompt(self) -> str:
        prompt = "<image>\nPlease provide a detailed and comprehensive description of the image, then extract a question and answer pair from it. Just return the question and answer pair in the format: Question: <question>, Answer: <answer>."

        system_prompt = f'''You are a image caption generator and question-answer pair extractor. Your task is to generate a concise and informative caption for the given image content, then extract a question and answer pair from it. The generated caption should contain the main content and details of the image. The question should be related to the image content, and the answer should be directly extracted from the generated caption.'''

        return prompt, system_prompt
    
class PersQAGeneratorPrompt:
    '''
    The prompt for the AutoPromptGenerator.
    '''
    def __init__(self):
        self.qa_template = {
            "obj_qs": [
                "What's <sks> general texture like?",
                "What color is <sks>?",
                "What size is <sks>?",
                "What shape does <sks> have?",
                "What type of object is <sks>?",
                "Does <sks> have any patterns or markings?",
                "What is the overall vibe of <sks>?",
                "How would you describe <sks> overall appearance?",
                "Does <sks> have any distinctive features or details?",
                "What material is <sks> made of?"
            ],
            "human_qs": [
                "What is <sks> hair color?",
                "What color are <sks> eyes?",
                "Would you describe <sks>'s physique as athletic, slim, or otherwise?",
                "What is <sks> skin tone?",
                "How would you describe <sks> hairstyle?",
                "Does <sks> wear glasses or any accessories?",
                "How would you describe <sks>'s attire?",
                "Does <sks> have any distinctive facial features?",
                "What is <sks> overall build or physique?",
                "What is <sks> general expression or demeanor?"
            ]
        }

    def build_prompt(self) -> str:
        # 把上面的prompts变成模版
        prompt_template = "The name of the main character in the image is <{sks}>. You need to answer a question about <{sks}>.\nQuestion: {query} Please answer starting with <{sks}>!\nAnswer: "

        system_prompt = f'''You are a personal question-answer generator. Your task is to generate a concise and informative answer for the given question about the main character in the image. The question should be related to the character's appearance or attributes, and the answer should be directly related to the character's features.'''

        return prompt_template, self.qa_template, system_prompt

class SKVQAGeneratorPrompt:
    '''
    The prompt for the SKVQAGeneratorPrompt.
    '''
    def __init__(self):
        pass

    def build_prompt(self) -> str:

        prompt = """
        <image>\nWrite a Wikipedia article related to this image without directly referring to the image. Then write question answer pairs. The question answer pairs should satisfy the following criteria.
        1: The question should refer to the image.
        2: The question should avoid mentioning the name of the object in the image.
        3: The question should be answered by reasoning over the Wikipedia article.
        4: The question should sound natural and concise.
        5: The answer should be extracted from the Wikipedia article.
        6: The answer should not be any objects in the image.
        7: The answer should be a single word or phrase and list all correct answers separated by commas.
        8: The answer should not contain 'and', 'or', rather you can split them into multiple answers.
        """
        
        return prompt
    
class MCTReasoningPrompt:
    def build_prompt(self):
        return {
            "web_grounding": (
                "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. "
                "The Assistant systematically reasons through the problem step by step, verifying each step and grounding every step to a specific point in the image.\n\n"
                "All reasoning processes must be enclosed within a single set of '<think>' tags, with each reasoning step explicitly referencing a coordinate:\n\n"
                "<think>\n[Reasoning text with grounded points inline] (x1, y1). [Further reasoning] (x2, y2), [Final refinement] (x3, y3).\n</think>\n\n"
                "The final answer should be enclosed in '<answer>' tags in the format:\n<answer> (xf, yf) </answer>\n\n"
                "Your task is to help the user identify the precise coordinates (x, y) of a specific area/element/object on the screen based on a description.\n"
                "- Aim to point to the center or a representative point within the described area/element/object as accurately as possible.\n"
                "- If the description is unclear or ambiguous, infer the most relevant area or element based on its likely context or purpose.\n"
                "- The final output should be the single most precise coordinate for the requested element.\n"
                "- The Assistant should verify each step and check multiple possible solutions before selecting the final answer."
            ),
            "spatial": (
                "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. "
                "The Assistant systematically reasons through the problem step by step by checking and verifying possible solutions and image regions, "
                "while grounding reasoning steps to specific objects and their relationships in the image using (x,y) coordinates. "
                "There may be one image or two images concatenated together.\n\n"
                "All reasoning processes must be enclosed within a single set of '<think>' tags.\n\n"
                "The final answer should be enclosed in '<answer>' tags in the format:\n<answer> {text of selected answer choice} </answer>\n"
                "- Your answer should be the exact text of the selected option."
            ),
            "web_action": (
                "You are a helpful Assistant tasked with navigating a web browser. "
                "Each reasoning step must be enclosed within '<think>' tags and reference exactly one specific coordinate (x, y). "
                "When ready, provide exactly one final action in <answer>...</answer>."
            ),
            "vstar": (
                "You are an assistant answering a visual question by reasoning through image regions. "
                "All reasoning in one <think>...</think>; final answer in <answer>...</answer>."
            ),
        }
    

class ImageScaleCaptionPrompt:
    def build_prompt(self):
        prompt = {}
        # VLM Prompt 1: Initial Caption
        prompt["VLM_PROMPT_1"] = (
            "Describe the fine-grained content of the image, including scenes, objects, "
            "relationships, instance location, and any text present."
        )
        
        # LLM Prompt 1: Question Generation
        prompt["LLM_PROMPT_1"] = '''Your task is to convert each Object mentioned in a given sentence into a corresponding instruction, and all the resulting instructions are output as "Describe more details about the [Object]". Ensure your instructions do not cover the raw question, options, or thought process of answering the instructions. You should ignore the Objects that appear in some inferences, such as the sentences that begins with 'it might be' or 'there are probably'.
        
        Sentence: 
        The image depicts a man in a suit and tie jumping in the air above a bed in a bedroom
        Instructions:
        Describe more details about the man.
        Describe more details about the suit.
        Describe more details about the tie.
        Describe more details about the bed.
        Describe more details about the bedroom.

        Sentence:
        {sentence}
        Instructions:
        '''

        # LLM Prompt 4: Integration
        prompt["LLM_PROMPT_4"] = '''Basic Context:
        {context}

        Object Information:
        {object_info}

        Position Information:
        {position_info}

        Following the logic of the above Basic Context, organize all details provided in Object Information and Position Information to give a very comprehensive description about the image. Do not include any analysis or your opinions.'''
        
        return prompt


class ImageCaprlPrompt:
    """
    Prompt definitions for CapRL/VisualOnlyMCQ pipeline.
    """
    def build_prompt(self):
        prompt = {}
        # 用于生成的 System Prompt
        prompt["SYS_PROMPT_MCQ"] = (
            "Your task is to generate five multiple-choice questions and their answers about the object "
            "based on the provided image. The questions should be challenging and focus on the image content.\n"
            "Format:\n"
            "#### 1. **Question?**\n"
            "   - A) Option\n"
            "   - B) Option\n"
            "   - C) Option\n"
            "   - D) Option\n\n"
            "**Answer:** A) Option\n"
            "------\n"
            "All questions must be answerable from the image alone."
        )
        # 用于生成的 User Prompt (FixPromptedVQAGenerator)
        prompt["USER_PROMPT_MCQ"] = "Generate 5 MCQs based on the image."
        
        # 用于验证的 Instruction (VisualDependencyRefiner)
        # 包含一个 {} 用于填入具体的问题文本
        prompt["ANSWER_INSTRUCTION"] = "{}\nAnswer the question with only the correct letter."
        
        return prompt
