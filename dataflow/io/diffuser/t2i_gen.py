# The I/O of different diffuser models for the same task is the same
import re
import os
from PIL import Image
from typing import Any, Literal
from dataflow.utils.registry import IO_REGISTRY
from dataflow.utils.storage import FileStorage
from dataflow.logger import get_logger

@IO_REGISTRY.register()
class ImageIO(object):
    def __init__(self):
        self.logger = get_logger()

    def read(self, image_paths: list[str]) -> list[Image.Image]:
        images = []
        for path in image_paths:
            try:
                img = Image.open(path).convert("RGB")
                images.append(img)
            except Exception as e:
                self.logger.warning(f"Failed to open image {path}: {e}")
                images.append(None)
        return images

    def write(self,
              storage: FileStorage,
              image_data: list[Image.Image],
              orig_paths: list[str],
              save_cache=False) -> Any:
        if len(image_data) != len(orig_paths):
            raise ValueError("image_data and orig_paths must have the same length.")
        media_cache_folder = os.path.join(storage.cache_path, "images")

        saved_paths = []
        for img, orig_path in zip(image_data, orig_paths):
            new_path = os.path.join(media_cache_folder, orig_path.split(':')[-1].strip("/"))
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            img.save(new_path)
            if save_cache: saved_paths.append(new_path)
            else: saved_paths.append(orig_path)

        return saved_paths
