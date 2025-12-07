import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger

from dataflow.utils.storage import FileStorage, DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC
from dataflow.serving.local_model_vlm_serving import LocalModelVLMServing_vllm
from qwen_vl_utils import process_vision_info

@OPERATOR_REGISTRY.register()
class PromptedVQAGenerator(OperatorABC):
    '''
    PromptedVQAGenerator read prompt and image/video to generate answers.
    '''
    def __init__(self, 
                 serving: LLMServingABC, 
                 system_prompt: str = "You are a helpful assistant."):
        self.logger = get_logger()
        self.serving = serving
        self.system_prompt = system_prompt
            
    @staticmethod
    def get_desc(lang: str = "zh"):
        return "读取 prompt 和 image/video 生成答案" if lang == "zh" else "Read prompt and image/video to generate answers."
    
    def _prepare_batch_inputs(self, prompts, input_media_paths, is_image: bool = True):
        """
        Construct batched prompts and multimodal inputs from media paths.
        """
        prompt_list = []
        media_paths = []
        type_media = "image" if is_image else "video"


        for paths, p in zip(input_media_paths, prompts):
            raw_prompt = [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": p},
                    ],
                },
            ]
            for path in paths:
                raw_prompt[1]["content"].append({"type": type_media, type_media: path})

            media_path, _ = process_vision_info(raw_prompt)
            prompt = self.serving.processor.apply_chat_template(
                raw_prompt, tokenize=False, add_generation_prompt=True
            )

            media_paths.append(media_path)
            prompt_list.append(prompt)

        return prompt_list, media_paths

    def run(self, 
            storage: DataFlowStorage,
            input_prompt_key: str = "prompt",
            input_image_key: str = "image", 
            input_video_key: str = "video",
            output_answer_key: str = "answer",
            ):
        if output_answer_key is None:
            raise ValueError("At least one of output_answer_key must be provided.")

        self.logger.info("Running PromptedVQA...")
        self.input_image_key = input_image_key
        self.input_video_key = input_video_key
        self.output_answer_key = output_answer_key

        # Load the raw dataframe from the input file
        dataframe = storage.read('dataframe')
        self.logger.info(f"Loading, number of rows: {len(dataframe)}")

        image_column = dataframe.get(self.input_image_key, pd.Series([])).tolist()
        video_column = dataframe.get(self.input_video_key, pd.Series([])).tolist()

        image_column = [path if isinstance(path, list) else [path] for path in image_column]
        video_column = [path if isinstance(path, list) else [path] for path in video_column]
                
        if len(image_column) == 0:
            image_column = None
        if len(video_column) == 0:
            video_column = None
        if image_column is None and video_column is None:
            raise ValueError("At least one of input_image_key or input_video_key must be provided.")
        if image_column is not None and video_column is not None:
            raise ValueError("Only one of input_image_key or input_video_key must be provided.")
        
        prompt_column = dataframe.get(input_prompt_key, pd.Series([])).tolist()

        if image_column is not None:
            prompt_list, image_inputs_list = self._prepare_batch_inputs(prompt_column, image_column)
            video_inputs_list = None
        elif video_column is not None:
            prompt_list, video_inputs_list = self._prepare_batch_inputs(prompt_column, video_column, is_image=False)
            image_inputs_list = None

        outputs = self.serving.generate_from_input(
            system_prompt=self.system_prompt,
            user_inputs=prompt_list,
            image_inputs=image_inputs_list,
            video_inputs=video_inputs_list
        )

        dataframe[self.output_answer_key] = outputs
        output_file = storage.write(dataframe)
        self.logger.info(f"Results saved to {output_file}")

        return output_answer_key
    
if __name__ == "__main__":
    # Initialize model
    model = LocalModelVLMServing_vllm(
        hf_model_name_or_path="Qwen/Qwen2.5-VL-3B-Instruct",
        vllm_tensor_parallel_size=1,
        vllm_temperature=0.7,
        vllm_top_p=0.9,
        vllm_max_tokens=512,
    )

    generator = PromptedVQAGenerator(
        serving=model,
        system_prompt="You are a helpful assistant.",
    )

    # Prepare input
    storage = FileStorage(
        first_entry_file_name="./dataflow/example/image_to_text_pipeline/prompted_vqa.jsonl", 
        cache_path="./cache_prompted_vqa",
        file_name_prefix="prompted_vqa",
        cache_type="jsonl",
    )
    storage.step()  # Load the data

    generator.run(
        storage=storage,
        input_prompt_key="prompt",
        input_image_key="image",
        input_video_key="video",
        output_answer_key="answer",
    )