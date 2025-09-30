from typing import Optional, List, Any, Tuple
import numpy as np
import subprocess
import io
import torch
import logging
import colorlog
from dataflow.logger import get_logger
from dataflow.core import get_operator
from PIL import Image

def _load_image(image_data: Any) -> Optional[Image.Image]:
    """
    Load image from various formats (path, bytes, PIL Image).
        
    Args:
        image_data: Image in various formats
            
    Returns:
        PIL Image object or None if loading fails
    """
    try:
        if isinstance(image_data, str):
            # Image path
            return Image.open(image_data).convert("RGB")
        elif isinstance(image_data, bytes):
            # Image bytes
            return Image.open(io.BytesIO(image_data)).convert("RGB")
        elif isinstance(image_data, Image.Image):
            # Already a PIL Image
            return image_data.convert("RGB")
        else:
            return None
    except Exception as e:
        return None