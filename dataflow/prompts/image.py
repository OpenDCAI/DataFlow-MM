from typing import Literal


class CaptionGeneratorPrompt:
    '''
    The prompt for the AutoPromptGenerator.
    '''
    def __init__(self):
        pass

    def build_prompt(self) -> str:
        prompt = "Please provide a comprehensive description of the image." # 这里开头加不加<image>都可以

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

        prompt = ''
        system_prompt = f'''You are a personal question-answer generator. Your task is to generate a concise and informative answer for the given question about the main character in the image. The question should be related to the character's appearance or attributes, and the answer should be directly related to the character's features.'''

        return prompt, system_prompt

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
    '''
    The prompt for the VisionMCTSReasoningSFTGenerate.
    '''
    def __init__(self):
        pass
    
    def build_prompt(self):
        prompt = {
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
        return prompt
    

class ImageScaleCaptionPrompt:
    '''
    The prompt for the ImageScaleCaptionGenerate.
    '''
    def __init__(self):
        pass
    
    def build_prompt(self):
        prompt = {}
        
        prompt["VLM_PROMPT_1"] = (
            "Describe the fine-grained content of the image, including scenes, objects, "
            "relationships, instance location, and any text present."
        )
        
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
        The train appears to be the main subject of the image, showcasing its sleek design and modern appearance
        Instructions:
        Describe more details about the train.

        Sentence:
        The table has a few other items on it, including a camera, a jar of jam, and a spoon, suggesting that there might be some people ready to eat
        Instructions:
        Describe more details about the table.
        Describe more details about the camera.
        Describe more details about the jam.
        Describe more details about the spoon.

        Sentence:
        The text "You see the world as you are!" is a playful and thought-provoking statement, encouraging viewers to appreciate their unique qualities and perspectives
        Instructions:
        Describe more details about the text.

        Sentence:
        1. **Preheat the Oven**: Preheat your oven to 350\u00b0F (175\u00bC).
        Instructions:
        Describe more details about the oven.
        Describe more details about the preheat temperature.

        Sentence:
        {}
        Instructions:
        '''

        prompt["LLM_PROMPT_2"] = '''Descriptions:
        {}

        Collect all details about each object from the descriptions, including detailed appearance, structure, material, and special marks or logos. Do not include any analysis or your opinions.'''

        prompt["LLM_PROMPT_3"] = '''Descriptions:
        {}

        Extract and abstract only the position information about each object from the decriptions. Do not include any analysis or your opinions.'''

        prompt["LLM_PROMPT_4"] = '''Basic Context:
        {}

        Object Information:
        {}

        Position Information:
        {}

        Following the logic of the above Basic Context, organize all details provided in Object Information and Position Information to give a very comprehensive description about the image. Do not include any analysis or your opinions.'''


class ImageGCoTPrompt:
    """
    Prompt generator for ImageGCoTGenerate.
    """

    def __init__(self):
        pass

    def build_prompt(
        self,
        prompt_type: Literal["cot", "bbox"],
        **kwargs,
    ) -> str:
        """
        Args:
            prompt_type: "cot" or "bbox"
            **kwargs:
                - if prompt_type == "cot": 需要 question, answer
                - if prompt_type == "bbox": 需要 keyword

        Returns:
            A string prompt for the given type.
        """
        if prompt_type == "cot":
            question = kwargs["question"]
            answer = kwargs["answer"]

            prompt = (
                f"Question: {question}\n"
                f"Answer: {answer}\n\n"
                f"Task: Provide a detailed step-by-step reasoning (Chain-of-Thought) that explains "
                f"how to arrive at this answer based on the image.\n\n"
                f"Then, extract key nouns and objects mentioned in your reasoning that are "
                f"visible in the image and can be spatially located.\n\n"
                f"Requirements for keywords:\n"
                f"1. Include concrete objects (e.g., cat, table, person, glasses)\n"
                f"2. Include objects with attributes (e.g., red apple, wooden chair)\n"
                f"3. Exclude pronouns (it, this, that) and abstract concepts (step, answer, image)\n"
                f"4. Exclude pure spatial words (left, right) unless combined with objects\n"
                f"5. Only include objects that can be visually located in the image\n\n"
                f"Format:\n"
                f"Step 1: ...\n"
                f"Step 2: ...\n"
                f"Answer: {answer}\n"
                f"Keywords: object1, object2, object3\n"
            )
            return prompt

        elif prompt_type == "bbox":
            keyword = kwargs["keyword"]
            # 简单做一下转义，防止 keyword 里有引号
            safe_keyword = keyword.replace('"', '\\"')

            prompt = (
                f'Please locate all instances of <ref>"{safe_keyword}"</ref> in this image.\n\n'
                f'Instructions:\n'
                f'1. If you can see "{safe_keyword}", provide bounding boxes for all instances\n'
                f'2. If "{safe_keyword}" is not visible, respond with: "not found"\n'
                f'3. Only return boxes you are confident about\n\n'
                f'Response format:\n'
                f'- If found: <box>(x1,y1),(x2,y2)</box> <box>(x1,y1),(x2,y2)</box> ...\n'
                f'- If not found: not found\n'
            )
            return prompt

class ImageCaprlPrompt:
    """
    Docstring for CapRLMCQGenerate
    """
    def __init__(self):
        pass
    
    def build_prompt(self):
        prompt = {}
        prompt["SYS_PROMPT_MCQ"] = (
            "Your task is to generate five multiple-choice questions and their answers about the object "
            "based on the provided image. The questions should be challenging and focus on the image content.\n"
            "You must strictly follow the format below and must not output irrelevant sentences:\n"
            "#### 1. **Example question?**\n"
            "   - A) Option A\n"
            "   - B) Option B\n"
            "   - C) Option C\n"
            "   - D) Option D\n\n"
            "**Answer:** D) Option D\n"
            "------\n"
            "#### 2. **Another example?**\n"
            "   - A) ...\n"
            "   - B) ...\n"
            "   - C) ...\n"
            "   - D) ...\n\n"
            "**Answer:** B) ...\n"
            "------\n"
            "All questions must be answerable from the image alone."
        )
        prompt["USER_PROMPT_MCQ"] = "Here is the image"
        prompt["ANSWER_LETTER_INSTRUCTION"] = "{}. Answer the question with only the correct letter"
        return prompt
