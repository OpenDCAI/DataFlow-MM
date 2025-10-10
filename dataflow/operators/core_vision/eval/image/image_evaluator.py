from dataflow.prompts.prompt_generator import PromptGeneratorABC
import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger

from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC

class EvalImageGenerationGenerator(OperatorABC):
    '''
    Video Generator is a class that generates contents for given videos.
    '''
    def __init__(self, llm_serving: LLMServingABC, prompt_geenrator: PromptGeneratorABC, vision_info_processor):
        self.logger = get_logger()
        self.prompt_generator = prompt_geenrator
        self.llm_serving = llm_serving
        self.processor = self.llm_serving.processor
        self.vision_info_processor = vision_info_processor

    @staticmethod           # 静态方法参数不用传self, 在类实例化之前就被调用
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "该算子用于评估图片生成任务的质量。\n\n"
                "输入参数：\n"
                "输出参数：\n"
            )
        elif lang == "en":
            return (
                "This operator is used to evaluate the quality of image generation task.\n\n"
                "Input Parameters:\n"
                "Output Parameters:\n"
            )
        else:
            return "EvalImageGenerationGenerator evaluates the quality of image generation task."

    def run(
        self,
        storage: DataFlowStorage,
        output_key: str = "output",
    ):
        """
        Runs the caption generation process, reading from the input file and saving results to output.
        """
        storage.step()
        data = storage.read(output_type='dataframe')

        image_inputs_list = []
        video_inputs_list = []
        audio_inputs_list = []
        prompt_list = []
        for idx, item in data.iterrows():
            messages = self.prompt_generator.generate_prompt(item)
            image_inputs, video_inputs = self.vision_info_processor(messages)
            prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            image_inputs_list.append(image_inputs)
            video_inputs_list.append(video_inputs)
            audio_inputs_list.append(None)
            prompt_list.append(prompt)

        outputs = self.llm_serving.generate_from_input(
            user_inputs= prompt_list,
            image_inputs=image_inputs_list,
            video_inputs=video_inputs_list,
            audio_inputs=audio_inputs_list,
        )

        data[output_key] = outputs
        output_file = storage.write(data)
        self.logger.info(f"Results saved to {output_file}")
        return [output_key]