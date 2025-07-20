import os
import torch
from PIL import Image
from huggingface_hub import snapshot_download
from dataflow.core import VLMServingABC
from .utils.diffusers.flux_kontext_pipeline import FluxKontextPipeline
from dataflow import get_logger
from diffusers import FluxPipeline
from typing import Optional, Union, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

class LocalImageGenServing(VLMServingABC):
    def __init__(
        self,
        image_io,
        hf_model_name_or_path: Union[str, List[str]] = None,
        hf_cache_dir: str = None,
        hf_local_dir: str = "./ckpt/models/",
        Image_gen_task: str = "text2image",  # text2image, imageedit
        diffuser_model_name: str = "FLUX",
        diffuser_image_height: int = 1024,
        diffuser_image_width: int = 1024,
        diffuser_num_inference_steps: int = 50,   # for FLUX-Kontext set to 28
        diffuser_guidance_scale: float = 2.5,     # for FLUX-Kontext set to 3.5
        diffuser_num_images_per_prompt: int = 1,
    ):
        self.image_io = image_io
        self.hf_model_name_or_path = hf_model_name_or_path
        self.hf_cache_dir = hf_cache_dir
        self.hf_local_dir = hf_local_dir
        self.image_gen_task = Image_gen_task
        self.diffuser_model_name = diffuser_model_name
        self.diffuser_image_height = diffuser_image_height
        self.diffuser_image_width = diffuser_image_width
        self.diffuser_num_inference_steps = diffuser_num_inference_steps
        self.diffuser_guidance_scale = diffuser_guidance_scale
        self.diffuser_num_images_per_prompt = diffuser_num_images_per_prompt

        # Detect available GPUs
        self.num_gpus = torch.cuda.device_count()
        self.logger = get_logger()

        self.load_model()

    def load_model(self):
        """
        Load one or multiple diffusion pipelines and assign each to a separate GPU.
        """
        # Prepare model paths list
        if isinstance(self.hf_model_name_or_path, list):
            model_ids = self.hf_model_name_or_path
        else:
            # replicate single model across GPUs if multiple gpus, else single
            model_ids = [self.hf_model_name_or_path] * max(1, self.num_gpus)

        self.pipes: List[Any] = []
        self.devices: List[torch.device] = []

        for idx, model_id in enumerate(model_ids):
            # determine device
            if self.num_gpus > 0:
                device = torch.device(f"cuda:{idx % self.num_gpus}")
            else:
                device = torch.device("cpu")
            self.devices.append(device)

            # obtain local or remote path
            if not model_id:
                raise ValueError("model_name_or_path is required")
            if os.path.exists(model_id):
                real_model_path = model_id
                self.logger.info(f"Using local model path: {model_id}")
            else:
                self.logger.info(f"Downloading model from HuggingFace: {model_id}")
                real_model_path = snapshot_download(
                    repo_id=model_id,
                    cache_dir=self.hf_cache_dir,
                    local_dir=self.hf_local_dir,
                )

            # load pipeline depending on model type
            if self.diffuser_model_name == "FLUX":
                pipe = FluxPipeline.from_pretrained(
                    real_model_path,
                    torch_dtype=torch.float16,
                ).to(device)
            elif self.diffuser_model_name == "FLUX-Kontext":
                pipe = FluxKontextPipeline.from_pretrained(
                    real_model_path,
                    torch_dtype=torch.bfloat16,
                ).to(device)
            else:
                raise ValueError(f"Unknown diffuser model: {self.diffuser_model_name}")

            # enable attention slicing for memory efficiency
            if hasattr(pipe, "enable_attention_slicing"):
                pipe.enable_attention_slicing()

            self.pipes.append(pipe)
            self.logger.success(f"✅ Loaded {self.diffuser_model_name} on {device} from {model_id}")

    def _generate_on_pipe(self, pipe: Any, prompts: List[Any]) -> Dict[str, List[Image.Image]]:
        """
        Helper to generate images for a batch of prompts on a single pipeline.
        """
        if self.image_gen_task == "text2image":
            output = pipe(
                prompt=prompts,
                height=self.diffuser_image_height,
                width=self.diffuser_image_width,
                num_inference_steps=self.diffuser_num_inference_steps,
                guidance_scale=self.diffuser_guidance_scale,
                num_images_per_prompt=self.diffuser_num_images_per_prompt,
            )
        elif self.image_gen_task == "imageedit":
            # prompts expected as list of tuples (image, prompt)
            images, texts = zip(*prompts)
            output = pipe(
                image=list(images),
                prompt=list(texts),
                height=self.diffuser_image_height,
                width=self.diffuser_image_width,
                num_inference_steps=self.diffuser_num_inference_steps,
                guidance_scale=self.diffuser_guidance_scale,
                num_images_per_prompt=self.diffuser_num_images_per_prompt,
            )
        else:
            raise ValueError(f"Unknown task: {self.image_gen_task}")

        all_images = output.images
        grouped: Dict[str, List[Image.Image]] = {}
        if self.image_gen_task == "text2image":
            for idx, prompt in enumerate(prompts):
                start = idx * self.diffuser_num_images_per_prompt
                end = start + self.diffuser_num_images_per_prompt
                grouped[prompt] = all_images[start:end]
        else:
            for idx, (_, text) in enumerate(prompts):
                start = idx * self.diffuser_num_images_per_prompt
                end = start + self.diffuser_num_images_per_prompt
                grouped[text] = all_images[start:end]
        return grouped

    def generate_from_input(self, user_inputs: List[Any]) -> Any:
        """
        Distribute inputs across pipelines and run in parallel, then save via image_io.
        """
        n_pipes = len(self.pipes)
        if n_pipes <= 1:
            results = self._generate_on_pipe(self.pipes[0], user_inputs)
        else:
            # chunk inputs
            chunks = [[] for _ in range(n_pipes)]
            for i, inp in enumerate(user_inputs):
                chunks[i % n_pipes].append(inp)

            results = {}
            with ThreadPoolExecutor(max_workers=n_pipes) as executor:
                futures = []
                for pipe, chunk in zip(self.pipes, chunks):
                    if not chunk:
                        continue
                    futures.append(executor.submit(self._generate_on_pipe, pipe, chunk))
                for fut in futures:
                    batch_res = fut.result()
                    results.update(batch_res)

        # save all images
        return self.image_io(results)

    def cleanup(self):
        """
        Clear pipelines and free GPU memory.
        """
        for pipe in getattr(self, 'pipes', []):
            del pipe
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
