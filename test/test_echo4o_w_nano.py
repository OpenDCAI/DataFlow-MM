# provide a image editing dataset

# text-to-image compose multiple objects

# task_type = "image editing", "multi-turn editing", "multi-image conditions generation", "text-to-image with multiple objects"


# step 1: utilize a local model to geenrate base images

# step 2: follow different task settings to realize image generation

import os
import argparse
from dataflow.operators.core_vision import PromptedImageEditGenerator
from dataflow.serving.local_image_gen_serving import LocalImageGenServing
from dataflow.serving.api_vlm_serving_openai import APIVLMServing_openai
from dataflow.utils.storage import FileStorage
from dataflow.io import ImageIO
