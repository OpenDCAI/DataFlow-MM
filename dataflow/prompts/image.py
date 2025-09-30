import random

class CaptionGeneratorPrompt:
    '''
    The prompt for the AutoPromptGenerator.
    '''
    def __init__(self):
        pass

    def caption_generator_prompt(self) -> str:
        prompt = "<image>\nPlease provide a detailed and comprehensive description of the image."

        system_prompt = f'''You are a image caption generator. Your task is to generate a concise and informative caption for the given image content.'''

        return prompt, system_prompt

class QAGeneratorPrompt:
    '''
    The prompt for the AutoPromptGenerator.
    '''
    def __init__(self):
        pass

    def qa_generator_prompt(self) -> str:
        # 请对给定的图像内容生成一个简洁且信息丰富的字幕，然后从中提取出一个问题和答案对。
        # 生成的字幕应包含图像的主要内容和细节。
        # 问题应与图像内容相关，答案应直接从生成的字幕中提取。
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

    def pers_generator_prompt(self) -> str:

        prompt = ''
        system_prompt = f'''You are a personal question-answer generator. Your task is to generate a concise and informative answer for the given question about the main character in the image. The question should be related to the character's appearance or attributes, and the answer should be directly related to the character's features.'''

        return prompt, system_prompt