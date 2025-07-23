# The I/O of different diffuser models for the same task is the same
import re
import os
from PIL import Image
from typing import Any, Literal, Dict, List
from dataflow.utils.registry import IO_REGISTRY
from dataflow.logger import get_logger

@IO_REGISTRY.register()
class ImageIO(object):
    def __init__(self, save_path: str):
        self.logger = get_logger()
        self.save_path = save_path

    def read(self, image_paths: List[str]) -> List[Image.Image]:
        images = []
        for path in image_paths:
            try:
                img = Image.open(path).convert("RGB")
                images.append(img)
            except Exception as e:
                self.logger.warning(f"Failed to open image {path}: {e}")
                images.append(None)
        return images

    def write(self, image_data: Dict[str, List[Image.Image]]) -> Dict[str, List[str]]:
        result: Dict[str, List[str]] = {}
        # Ensure base save directory exists
        os.makedirs(self.save_path, exist_ok=True)

        for prompt, imgs in image_data.items():
            # Sanitize prompt for filesystem
            prompt_safe = re.sub(r"[^0-9a-zA-Z]+", "_", prompt).strip("_")
            prompt_dir = os.path.join(self.save_path, prompt_safe)
            os.makedirs(prompt_dir, exist_ok=True)

            saved_paths: List[str] = []
            for idx, img in enumerate(imgs):
                # Generate filename and save
                filename = f"{prompt_safe}_{idx}.png"
                file_path = os.path.join(prompt_dir, filename)
                try:
                    img.save(file_path)
                except Exception as e:
                    self.logger.error(f"Failed to save image for prompt '{prompt}' at {file_path}: {e}")
                    continue
                saved_paths.append(file_path)
            result[prompt] = saved_paths
        return result
    
    def __call__(self, image_data: Dict[str, List[Image.Image]]):
        return self.write(image_data=image_data)
