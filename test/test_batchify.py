from dataflow.operators.generate.vqa.prompted_vqa import PromptedVQA
from dataflow.serving import LocalModelVLMServing_sglang
from dataflow.utils.storage import FileStorage
from dataflow.wrapper import BatchWrapper

if __name__ == "__main__":
    
    storage = FileStorage(
        first_entry_file_name="./dataflow/example/vqa/sample_data.json",
        cache_path="./cache",
        file_name_prefix="vqa",
        cache_type="json",
    )
    vlm_serving = LocalModelVLMServing_sglang(
            hf_model_name_or_path="/data0/public_models/Qwen2.5-VL-7B-Instruct",
            sgl_dp_size=1,  # data parallel size
            sgl_tp_size=1,  # tensor parallel size
    )
    op = PromptedVQA(vlm_serving=vlm_serving)

    # 这里 batch_op.run(...) 在 PyCharm / Pylance 就能自动补全 PromptedVQA.run 的签名了
    batched_op = BatchWrapper(op, batch_size=3)
    
    batched_op.run(
        # storage.step(),
        storage=storage.step(),
        input_image_key="image",
        input_video_key="video",
        input_audio_key="audio",
        input_conversation_key="conversation",
        output_answer_key="answer",
    )