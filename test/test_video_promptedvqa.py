from dataflow.operators.core_vision import PromptedVQAGenerator
from dataflow.operators.conversations import Conversation2Message
from dataflow.serving import LocalModelVLMServing_vllm
from dataflow.utils.storage import FileStorage
from dataflow.operators.core_prompt import PromptUpdate

class VQAGenerator():
    def __init__(self):
        self.storage = FileStorage(
            first_entry_file_name="./dataflow/example/video/sample_data.json",
            cache_path="./cache",
            file_name_prefix="vqa",
            cache_type="json",
        )
        self.model_cache_dir = './dataflow_cache'

        self.vlm_serving = LocalModelVLMServing_vllm(
            hf_model_name_or_path="Qwen/Qwen2.5-VL-7B-Instruct",
            hf_cache_dir=self.model_cache_dir,
            vllm_tensor_parallel_size=1,
            vllm_temperature=0.7,
            vllm_top_p=0.9, 
            vllm_max_tokens=512,
            vllm_gpu_memory_utilization=0.9
        )

        def build_prompt(value: str, row) -> str:
            # Fixed template
            template = """
            ### Task:
            Given a detailed description that summarizes the content of a video, generate question-answer pairs based on the description to help humans better understand the video.
            The question-answer pairs should be faithful to the content of the video description and developed from different dimensions to promote comprehensive understanding of the video.

            #### Guidelines For Question-Answer Pairs Generation:
            - Carefully read the provided video description. Pay attention to the content, such as the scene where the video takes place, the main characters and their behaviors, and the development of the events.
            - Generate appropriate question-answer pairs based on the description. The pairs should cover as many dimensions as possible and must not deviate from the video description.
            - Generate 5 ~ 10 question-answer pairs with different dimensions.

            ### Output Format:
            1. Your output should be in JSON format.
            2. Only provide the Python dictionary string.
            Your response should look like: 
            [{"Dimension": <dimension-1>, "Question": <question-1>, "Answer": <answer-1>},
            {"Dimension": <dimension-2>, "Question": <question-2>, "Answer": <answer-2>},
            ...]
            """

            user_query_free_form = f"""
            Please generate question-answer pairs for the following video description:
            Description: {row.get("caption", "")}
            """

            # Combine the template with the original value (i.e., the message from conversation)
            return template.strip() + "\n\n" + user_query_free_form.strip()

        self.prompt_converter = PromptUpdate(update_fn=build_prompt, only_roles=["human"])
        
        self.prompt_generator = PromptedVQAGenerator(
            vlm_serving=self.vlm_serving,
        )

    def forward(self):

        # Step 1: Generate caption from video/image/audio + conversation
        self.prompt_generator.run(
            storage=self.storage.step(),
            input_image_key="image",
            input_video_key="video",
            input_audio_key="audio",
            input_conversation_key="conversation",
            output_answer_key="caption",
        )

        # Step 2: Update conversation with new prompt template
        self.prompt_converter.run(self.storage.step())

        # Step 3: Generate QA pairs from the updated conversation
        self.prompt_generator.run(
            storage=self.storage.step(),
            input_image_key="image",
            input_video_key="video",
            input_audio_key="audio",
            input_conversation_key="conversation",
            output_answer_key="QA",
        )

if __name__ == "__main__":
    # Entry point for the pipeline
    model = VQAGenerator()
    model.forward()
