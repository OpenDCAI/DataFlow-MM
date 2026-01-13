import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger

from dataflow.utils.storage import FileStorage, DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC
from dataflow.serving.local_model_vlm_serving import LocalModelVLMServing_vllm
    
@OPERATOR_REGISTRY.register()
class PromptedQAGenerator(OperatorABC):
    '''
    PromptedQAGenerator read prompt and generate answers.
    '''
    def __init__(self, 
                 serving: LLMServingABC, 
                 system_prompt: str = "You are a helpful assistant."):
        self.logger = get_logger()
        self.serving = serving
        self.system_prompt = system_prompt
            
    @staticmethod
    def get_desc(lang: str = "zh"):
        return "读取 prompt 生成答案" if lang == "zh" else "Read prompt to generate answers."
    
    def _prepare_batch_inputs(self, prompts):
        """
        Construct batched prompts.
        """
        prompt_list = []

        for p in prompts:
            raw_prompt = [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": p},
                    ],
                },
            ]

            prompt = self.serving.processor.apply_chat_template(
                raw_prompt, tokenize=False, add_generation_prompt=True
            )

            prompt_list.append(prompt)

        return prompt_list

    def run(self, 
            storage: DataFlowStorage,
            input_prompt_key: str = "prompt",
            output_answer_key: str = "answer",
            ):
        if output_answer_key is None:
            raise ValueError("At least one of output_answer_key must be provided.")

        self.logger.info("Running PromptedQA...")

        self.output_answer_key = output_answer_key

        # Load the raw dataframe from the input file
        dataframe = storage.read('dataframe')
        self.logger.info(f"Loading, number of rows: {len(dataframe)}")

        prompt_column = dataframe.get(input_prompt_key, pd.Series([])).tolist()
        prompt_list = self._prepare_batch_inputs(prompt_column) 

        outputs = self.serving.generate_from_input(
            system_prompt=self.system_prompt,
            user_inputs=prompt_list,
        )

        dataframe[self.output_answer_key] = outputs
        output_file = storage.write(dataframe)
        self.logger.info(f"Results saved to {output_file}")

        return output_answer_key
    
if __name__ == "__main__":
    # Initialize model
    model = LocalModelVLMServing_vllm(
        hf_model_name_or_path="/data0/happykeyan/Models/Qwen2.5-VL-3B-Instruct",
        vllm_tensor_parallel_size=1,
        vllm_temperature=0.7,
        vllm_top_p=0.9,
        vllm_max_tokens=512,
    )

    generator = PromptedQAGenerator(
        serving=model,
        system_prompt="You are a helpful assistant. Return the value of the math expression in the user prompt.",
    )

    # Prepare input
    storage = FileStorage(
        first_entry_file_name="./dataflow/example/text_to_text/prompted_qa.jsonl", 
        cache_path="./cache_prompted_qa",
        file_name_prefix="prompted_qa",
        cache_type="jsonl",
    )
    storage.step()  # Load the data

    generator.run(
        storage=storage,
        input_prompt_key="prompt",
        output_answer_key="answer",
    )
