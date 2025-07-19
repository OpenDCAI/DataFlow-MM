import os
import torch
from PIL import Image
from huggingface_hub import snapshot_download
from dataflow.core import VLMServingABC
from .utils.diffusers.flux_kontext_pipeline import FluxKontextPipeline
from dataflow import get_logger
from diffusers import FluxPipeline
from typing import Optional, Union, List, Dict, Any


class LocalImageGenServing(VLMServingABC):
    def __init__(
        self,
        image_io,
        hf_model_name_or_path: str = None,
        hf_cache_dir: str = None,
        hf_local_dir: str = "./ckpt/models/",
        device: str = "cuda",                # here need to be change into API
        Image_gen_task: str = "text2image",  # text2image, imageedit
        diffuser_model_name: str = "FLUX",
        diffuser_image_height: int = 1024,
        diffuser_image_width: int = 1024,
        diffuser_num_inference_steps: int = 50,     # for FLUX-kontext set to 28
        diffuser_guidance_scale: float = 2.5,       # for FLUX-kontext set to 3.5
        diffuser_num_images_per_prompt: int = 1,
    ):
        self.image_io = image_io
        self.hf_model_name_or_path = hf_model_name_or_path
        self.hf_cache_dir = hf_cache_dir
        self.hf_local_dir = hf_local_dir
        self.device = device
        self.image_gen_task = Image_gen_task
        self.diffuser_model_name = diffuser_model_name
        self.diffuser_image_height = diffuser_image_height
        self.diffuser_image_width = diffuser_image_width
        self.diffuser_num_inference_steps = diffuser_num_inference_steps
        self.diffuser_guidance_scale = diffuser_guidance_scale
        self.diffuser_num_images_per_prompt = diffuser_num_images_per_prompt

        # Load the model into the pipeline
        self.load()

    def generate(self):
        """
        Generate without explicit prompts.
        You can override this method to provide default behavior.
        """
        raise NotImplementedError("Please implement the default generate() method.")

    def generate_from_input(
        self,
        user_inputs: list[str],
    ):
        if self.image_gen_task == "text2image":
            output = self.pipe(
                prompt=user_inputs,
                height=self.diffuser_image_height,
                width=self.diffuser_image_width,
                num_inference_steps=self.diffuser_num_inference_steps,
                guidance_scale=self.diffuser_guidance_scale,
                num_images_per_prompt=self.diffuser_num_images_per_prompt,
            )
            all_images = output.images

            grouped: dict[str, list] = {}
            for idx, prompt in enumerate(user_inputs):
                start = idx * self.diffuser_num_images_per_prompt
                end = start + self.diffuser_num_images_per_prompt
                grouped[prompt] = all_images[start:end]
            return self.image_io(grouped)
        
        elif self.image_gen_task == "imageedit":
            # now image editing only support single image editing, maybe need some changes
            input_img, input_prompt = user_inputs[0]
            output = self.pipe(
                image=input_img,
                prompt=input_prompt, 
                height=self.diffuser_image_height,
                width=self.diffuser_image_width,
                num_inference_steps=self.diffuser_num_inference_steps,
                guidance_scale=self.diffuser_guidance_scale,
                num_images_per_prompt=self.diffuser_num_images_per_prompt,
            )
            all_images = output.images

            grouped: dict[str, list] = {}
            grouped[input_prompt] = all_images
            return self.image_io(grouped)

        return grouped

    def load(self):
        self.logger = get_logger()
        if not self.hf_model_name_or_path:
            raise ValueError("model_name_or_path is required")
        elif os.path.exists(self.hf_model_name_or_path):
            self.logger.info(f"Using local model path: {self.hf_model_name_or_path}")
            self.real_model_path = self.hf_model_name_or_path
        else:
            self.logger.info(f"Downloading model from HuggingFace: {self.hf_model_name_or_path}")
            self.real_model_path = snapshot_download(
                repo_id=self.hf_model_name_or_path,
                cache_dir=self.hf_cache_dir,
                local_dir=self.hf_local_dir,
            )

        # Load the pipeline with dtype and device, trusting remote code if allowed  ### HERE may need to change into a model map inference
        if self.diffuser_model_name == "FLUX":
            self.pipe = FluxPipeline.from_pretrained(
                self.real_model_path,
                torch_dtype=torch.float16,
            ).to(self.device)
        
        if self.diffuser_model_name == "FLUX-Kontext":
            self.pipe = FluxKontextPipeline.from_pretrained(
                self.real_model_path,
                torch_dtype=torch.bfloat16
            ).to(self.device)

        # Enable attention slicing if GPU memory is limited
        if hasattr(self.pipe, "enable_attention_slicing"):
            self.pipe.enable_attention_slicing()

        self.logger.success(f"✅ Model {self.diffuser_model_name} loaded on {self.device} from {self.real_model_path}")

    def cleanup(self):
        del self.pipe
        import gc;
        gc.collect()
        torch.cuda.empty_cache()
