import os
import argparse
from pathlib import Path
from dataflow.operators.core_vision import PromptedImageGenerator
from dataflow.serving.api_image_gen_serving import APIImageGenServing
from dataflow.utils.storage import FileStorage
from dataflow.io import ImageIO
from dataflow import get_logger


class ImageGenerationAPIPipeline():
    """
    Text to Image Generation API Pipeline
    Supported Models:
        OpenAI format (api_format="openai"): dall-e-2, dall-e-3, gpt-image-1
        Gemini format (api_format="gemini"): gemini-2.5-flash-image, gemini-3-pro-image-preview, etc.
    """
    def __init__(
        self, 
        api_format="gemini",
        model_name="gemini-3-pro-image-preview",
        batch_size=4,
        first_entry_file_name=None,
        cache_path="./cache_local/text2image_api",
    ):
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent.parent.parent
        
        if first_entry_file_name is None:
            data_file = project_root / "dataflow" / "example" / "image_gen" / "text2image" / "prompts.jsonl"
            first_entry_file_name = str(data_file)
        
        self.storage = FileStorage(
            first_entry_file_name=first_entry_file_name,
            cache_path=cache_path,
            file_name_prefix="dataflow_cache_step",
            cache_type="jsonl"
        )

        self.logger = get_logger()
        
        api_key = os.environ.get("DF_API_KEY")
        api_url = os.environ.get("DF_BASE_URL")
        
        if api_key is None:
            raise ValueError("API key is required. Please set it via environment variable DF_API_KEY")
        
        if api_url is None:
            if api_format == "gemini":
                api_url = "https://generativelanguage.googleapis.com"
            else:
                api_url = "https://api.openai.com/v1"
        
        image_save_path = str(project_root / "cache_local" / "text2image_api")
        
        self.serving = APIImageGenServing(
            api_url=api_url,
            image_io=ImageIO(save_path=image_save_path),
            Image_gen_task="text2image",
            batch_size=batch_size,
            api_format=api_format,
            model_name=model_name,
            api_key=api_key,
        )

        self.text_to_image_generator = PromptedImageGenerator(
            t2i_serving=self.serving,
            save_interval=10
        )
    
    def forward(self):
        try:
            self.text_to_image_generator.run(
                storage=self.storage.step(),
                input_conversation_key="conversations",
                output_image_key="images",
            )
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {str(e)}")
            raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cloud API Image Generation Pipeline")
    parser.add_argument(
        '--api_format',
        choices=['openai', 'gemini'],
        default='gemini',
        help='API format type: openai (OpenAI DALL-E) or gemini (Google Gemini)'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='gemini-3-pro-image-preview',
        help='Model name: for openai format use "dall-e-2", "dall-e-3", "gpt-image-1"; for gemini format use "gemini-2.5-flash-image" or "gemini-3-pro-image-preview", etc.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='Batch size'
    )
    parser.add_argument(
        '--first_entry_file_name',
        type=str,
        default=None,
        help='Input data file path (default uses example_data)'
    )
    parser.add_argument(
        '--cache_path',
        type=str,
        default="./cache_local/text2image_api",
        help='Cache path'
    )
    args = parser.parse_args()
    
    if not os.environ.get("DF_API_KEY"):
        parser.error("Environment variable DF_API_KEY is not set. Please use export DF_API_KEY=your_api_key to set it")
    
    model = ImageGenerationAPIPipeline(
        api_format=args.api_format,
        model_name=args.model_name,
        batch_size=args.batch_size,
        first_entry_file_name=args.first_entry_file_name,
        cache_path=args.cache_path,
    )
    model.forward()

