import pandas as pd
import numpy as np
from typing import Optional, List, Any
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.utils import _load_image
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import io


@OPERATOR_REGISTRY.register()
class CLIPScoreFilter(OperatorABC):
    """
    Evaluate CLIP scores between images and text, adding scores to dataframe.
    
    Computes the cosine similarity between CLIP embeddings of images and text,
    which measures how well images match their corresponding text descriptions.
    No filtering is performed - only adds scores for evaluation.
    """
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        batch_size: int = 32,
        keep_ratio: float = 0.8,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the CLIPScoreEval operator.
        
        Args:
            model_name: CLIP model name from HuggingFace
            batch_size: Batch size for CLIP inference
            device: Device for CLIP model ('cuda' or 'cpu')
        """
        self.logger = get_logger()
        self.batch_size = batch_size
        self.device = device
        self.keep_ratio = keep_ratio
        
        # Initialize CLIP model and processor
        self.logger.info(f"Loading CLIP model: {model_name}")
        self.model = CLIPModel.from_pretrained(model_name, weights_only=False).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name, weights_only=False)
        self.model.eval()
    
    @staticmethod
    def get_desc(lang: str = "zh") -> str:
        if lang == "zh":
            return "计算图像和文本的CLIP相似度分数"
        return "Compute CLIP similarity scores between images and text"
    
    def _compute_clip_scores(
        self, 
        images: List[Any], 
        questions: List[str],
        texts: List[str]
    ) -> np.ndarray:
        """
        Compute CLIP scores between images and their corresponding texts.
        
        Args:
            images: List of images in various formats
            texts: List of text descriptions
            
        Returns:
            Array of CLIP scores (cosine similarities)
        """
        scores = []
        
        # Process in batches
        for i in range(0, len(images), self.batch_size):
            batch_end = min(i + self.batch_size, len(images))
            batch_images = images[i:batch_end]
            batch_questions = questions[i:batch_end]
            batch_texts = texts[i:batch_end]
            
            # Load images and prepare valid pairs
            valid_images = []
            valid_texts = []
            valid_indices = []
            
            for idx, (img_data, question, text) in enumerate(zip(batch_images, batch_questions, batch_texts)):
                text = question + text
                if pd.isna(text) or text == "":
                    # Skip if text is missing
                    scores.append(np.nan)
                    continue
                    
                pil_img = _load_image(img_data)
                if pil_img is None:
                    # Skip if image loading fails
                    scores.append(np.nan)
                    continue
                
                valid_images.append(pil_img)
                valid_texts.append(text)
                valid_indices.append(idx)
            
            if not valid_images:
                continue
            
            # Compute CLIP embeddings and scores
            with torch.no_grad():
                # Process images and texts
                inputs = self.processor(
                    text=valid_texts,
                    images=valid_images,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(self.device)
                
                # Get embeddings
                outputs = self.model(**inputs)
                image_embeds = outputs.image_embeds
                text_embeds = outputs.text_embeds
                
                # Normalize embeddings
                image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
                text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
                
                # Compute cosine similarity (diagonal elements for paired scores)
                batch_scores = (image_embeds * text_embeds).sum(dim=-1)
                batch_scores = batch_scores.cpu().numpy()
            
            # Map scores back to original indices
            batch_all_scores = [np.nan] * (batch_end - i)
            for idx, score in zip(valid_indices, batch_scores):
                batch_all_scores[idx] = float(score)
            
            scores.extend(batch_all_scores)
        
        return np.array(scores)

    def run(
        self, 
        storage: DataFlowStorage,
        input_image_key: str = "image",
        input_question_key: str = "question",
        input_answer_key: str = "answer",
        output_score_key: str = "clipscore"
    ) -> None:
        """
        Execute the CLIP score evaluation pipeline.
        
        Process flow:
        1. Load data from storage
        2. Extract images and text
        3. Compute CLIP scores for image-text pairs
        4. Add scores to dataframe
        5. Save dataframe with scores (no filtering)
        
        Args:
            storage: DataFlow storage object for reading/writing data
            input_image_key: Column name for image data
            input_answer_key: Column name for text data (questions/captions)
            output_score_key: Column name for CLIP scores (default: 'clipscore')
        """
        self.logger.info("Running CLIPScoreEval...")
        
        # Load dataframe
        dataframe = storage.read('dataframe')
        total_rows = len(dataframe)
        self.logger.info(f"Loaded {total_rows} rows from storage")
        
        # Validate required columns
        if input_image_key not in dataframe.columns:
            self.logger.error(f"Image column '{input_image_key}' not found")
            raise ValueError(f"Missing required column: {input_image_key}")
        
        if input_answer_key not in dataframe.columns:
            self.logger.error(f"Text column '{input_answer_key}' not found")
            raise ValueError(f"Missing required column: {input_answer_key}")
        
        # Extract images and text
        images = dataframe[input_image_key].tolist()
        questions = dataframe[input_question_key].tolist()
        texts = dataframe[input_answer_key].tolist()
        
        # Compute CLIP scores
        self.logger.info("Computing CLIP scores for image-text pairs...")
        clip_scores = self._compute_clip_scores(images, questions, texts)
        
        # Add scores to dataframe
        dataframe[output_score_key] = clip_scores
        
        self.logger.info(f"Filtering to keep top {self.keep_ratio*100:.1f}% of samples based on CLIP scores...")
        
        valid_scores = clip_scores[~np.isnan(clip_scores)]
        
        if len(valid_scores) > 0:
            score_keep_ratio = np.percentile(valid_scores, (1 - self.keep_ratio) * 100)
            
            filtered_dataframe = dataframe[dataframe[output_score_key] >= score_keep_ratio]
        
        # Save filtered dataframe
        storage.write(filtered_dataframe)