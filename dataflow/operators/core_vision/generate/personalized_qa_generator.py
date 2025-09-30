import random
from dataflow.core.Operator import OperatorABC

from dataflow.prompts.image import PersQAGeneratorPrompt
import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger

from dataflow.utils.storage import FileStorage, DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC
from dataflow.serving.local_model_vlm_serving import LocalModelVLMServing_vllm

from qwen_vl_utils import process_vision_info

@OPERATOR_REGISTRY.register()
class PersQAGenerate(OperatorABC):
    '''
    Caption Generator is a class that generates captions for given images.
    '''
    def __init__(self, llm_serving: LLMServingABC):
        self.logger = get_logger()
        self.prompt_generator = PersQAGeneratorPrompt()
        self.llm_serving = llm_serving

    @staticmethod           # 静态方法参数不用传self, 在类实例化之前就被调用
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "该算子用于调用视觉语言大模型生成图像描述。\n\n"
                "输入参数：\n"
                "输出参数：\n"
            )
        elif lang == "en":
            return (
                "This operator is used to call the large vision language model to generate image description.\n\n"
                "Input Parameters:\n"
                "Output Parameters:\n"
            )
        else:
            return "CaptionGenerator produces captions for given image."
    
    def _validate_dataframe(self, dataframe: pd.DataFrame):
        required_keys = [self.multi_modal_key]
        forbidden_keys = [self.output_key]

        missing = [k for k in required_keys if k not in dataframe.columns]
        conflict = [k for k in forbidden_keys if k in dataframe.columns]

        if missing:
            raise ValueError(f"Missing required column(s): {missing}")
        if conflict:
            raise ValueError(f"The following column(s) already exist and would be overwritten: {conflict}")

    def _prepare_batch_inputs(self, media_paths, query_list, sks):
        """
        Construct batched prompts and image inputs from media paths.
        """
        _, system_prompt = self.prompt_generator.pers_generator_prompt()

        prompt_list = []
        image_inputs_list = []

        for paths, query in zip(media_paths, query_list):
            for p in paths:
                prompts = f"The name of the main character in the image is <{sks}>. You need to answer a question about <{sks}>.\nQuestion: " + query + f" Please answer starting with <{sks}>!\nAnswer: "

                raw_prompt = [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": p},
                            {"type": "text", "text": prompts},
                        ],
                    },
                ]
                # Get vision inputs
                image_inputs, _ = process_vision_info(raw_prompt)

                # Format prompt using LLM processor
                prompt = self.llm_serving.processor.apply_chat_template(
                    raw_prompt, tokenize=False, add_generation_prompt=True
                )

                image_inputs_list.append(image_inputs)
                prompt_list.append(prompt)

        return prompt_list, image_inputs_list

    def run(
        self,
        storage: DataFlowStorage,
        multi_modal_key: str = "image", 
        output_key: str = "output"
    ):
        """
        Runs the caption generation process in batch mode, reading from the input file and saving results to output.
        """
        self.multi_modal_key, self.output_key = multi_modal_key, output_key
        # storage.step()
        dataframe = storage.read("dataframe")
        self._validate_dataframe(dataframe)
        # media_paths = storage.media_paths
        media_paths = dataframe[self.multi_modal_key].tolist()
        # 将media_paths中的非list类型的路径转换为list
        media_paths = [path if isinstance(path, list) else [path] for path in media_paths]
        
        sks = 'mam'
        query_list = [random.choice(self.prompt_generator.qa_template["obj_qs"]).replace("<sks>", f"<{sks}>") for _ in range(len(media_paths))]

        prompt_list, image_inputs_list = self._prepare_batch_inputs(media_paths, query_list, sks)

        outputs = self.llm_serving.generate_from_input(
            user_inputs=prompt_list,
            image_inputs=image_inputs_list
        )

        outputs = [f"Question: {ques}, Answer: {ans}" for ques, ans in zip(query_list, outputs)]  # Add sks prefix to each output

        dataframe[self.output_key] = outputs
        output_file = storage.write(dataframe)
        self.logger.info(f"Results saved to {output_file}")

        return [output_key]

if __name__ == "__main__":
    model_path = "/data0/mt/.cache/huggingface/hub/Qwen2.5-VL-3B-Instruct"

    # Initialize model
    model = LocalModelVLMServing_vllm(
        hf_model_name_or_path=model_path,
        vllm_tensor_parallel_size=1,
        vllm_temperature=0.7,
        vllm_top_p=0.9,
        vllm_max_tokens=512,
    )

    caption_generator = PersQAGenerate(
        llm_serving=model
    )

    # Prepare input
    storage = FileStorage(
        first_entry_file_name="dataflow/example/Image2TextPipeline/test_image2caption.jsonl", 
        cache_type="jsonl", 
        media_key="image", 
        media_type="image"
    )
    storage.step()  # Load the data

    caption_generator.run(
        storage=storage,
        multi_modal_key="image",
        output_key="pers_qa"
    )