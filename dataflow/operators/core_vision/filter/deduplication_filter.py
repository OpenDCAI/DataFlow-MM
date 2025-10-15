import pandas as pd
import numpy as np
from typing import Optional, List, Any
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.utils import _load_image
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import CLIPProcessor, CLIPModel


@OPERATOR_REGISTRY.register()
class DeduplicateFilter(OperatorABC):
    """
    Filter operator that removes duplicate images based on CLIP embedding similarity.
    
    Uses CLIP to generate image embeddings, then identifies and removes
    duplicate images above a similarity threshold.
    """
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        threshold: float = 0.90,
        batch_size: int = 32,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the DeduplicateFilter operator.
        
        Args:
            model_name: CLIP model name from HuggingFace
            threshold: Cosine similarity threshold for duplicate detection (0.0-1.0)
            batch_size: Batch size for CLIP inference
            device: Device for CLIP model ('cuda' or 'cpu')
        """
        self.logger = get_logger()
        self.threshold = threshold
        self.batch_size = batch_size
        self.device = device
        
        # Initialize CLIP model and processor
        self.logger.info(f"Loading CLIP model: {model_name}")
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
    
    @staticmethod
    def get_desc(lang: str = "zh") -> str:
        if lang == "zh":
            return "基于CLIP嵌入相似度过滤重复图像"
        return "Filter duplicate images based on CLIP embedding similarity"
    
    def _extract_embeddings(self, images: List[Any]) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract CLIP embeddings for a list of images.
        
        Args:
            images: List of images in various formats
            
        Returns:
            Tuple of (embeddings, valid_indices)
        """
        embeddings = []
        valid_indices = []
        
        # Process images in batches
        for i in range(0, len(images), self.batch_size):
            batch_images = images[i:i + self.batch_size]
            batch_pil_images = []
            batch_valid_indices = []
            
            # Load and validate images in batch
            for idx, img_data in enumerate(batch_images):
                pil_img = _load_image(img_data)
                if pil_img is not None:
                    batch_pil_images.append(pil_img)
                    batch_valid_indices.append(i + idx)
            
            if not batch_pil_images:
                continue
            
            # Process batch through CLIP
            with torch.no_grad():
                inputs = self.processor(
                    images=batch_pil_images, 
                    return_tensors="pt"
                ).to(self.device)
                
                image_features = self.model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                embeddings.append(image_features.cpu().numpy())
                valid_indices.extend(batch_valid_indices)
        
        if not embeddings:
            self.logger.warning("No valid images found for embedding extraction")
            return np.array([]), np.array([])
        
        embeddings = np.vstack(embeddings)
        self.logger.info(f"Extracted embeddings for {len(embeddings)} images")
        return embeddings, np.array(valid_indices)
    
    def _find_duplicates(self, embeddings: np.ndarray) -> tuple[set, list]:
        """
        Find duplicate images based on embedding similarity.
        
        Args:
            embeddings: Image embeddings of shape (n_images, embedding_dim)
            
        Returns:
            Tuple of (duplicate_indices, duplicate_pairs)
        """
        duplicate_indices = set()
        duplicate_pairs = []
        
        self.logger.info("Computing similarities for duplicate detection...")
        
        # Compute cosine similarities for all embeddings at once
        similarities = cosine_similarity(embeddings, embeddings)
        
        # Find duplicates above threshold
        indices = np.where(similarities >= self.threshold)
        
        for i, j in zip(indices[0], indices[1]):
            # Skip self-comparison and already processed pairs
            if i >= j:
                continue
            
            # Mark the second occurrence as duplicate
            duplicate_indices.add(j)
            
            duplicate_pairs.append({
                'kept_idx': i,
                'removed_idx': j,
                'similarity': float(similarities[i, j])
            })
        
        return duplicate_indices, duplicate_pairs

    def run(
        self, 
        storage: DataFlowStorage,
        input_image_key: str = "image",
        output_score_key: str = "max_similarity"
    ) -> None:
        """
        Execute the deduplication filtering pipeline.
        
        Process flow:
        1. Load data from storage
        2. Extract CLIP embeddings for all images
        3. Calculate cosine similarities between embeddings
        4. Identify and remove duplicate images above threshold
        5. Keep only unique images
        
        Args:
            storage: DataFlow storage object for reading/writing data
            input_image_key: Column name for image data
            output_score_key: Column name to store maximum similarity scores
        """
        self.logger.info("Running DeduplicateFilter...")
        
        # Load dataframe
        dataframe = storage.read('dataframe')
        original_count = len(dataframe)
        self.logger.info(f"Loaded {original_count} rows from storage")
        
        # Extract image column
        if input_image_key not in dataframe.columns:
            self.logger.error(f"Image column '{input_image_key}' not found in dataframe")
            raise ValueError(f"Missing required column: {input_image_key}")
        
        images = dataframe[input_image_key].tolist()
        
        # Extract CLIP embeddings
        self.logger.info("Extracting CLIP embeddings...")
        embeddings, valid_indices = self._extract_embeddings(images)
        
        if len(embeddings) == 0:
            self.logger.error("No valid embeddings extracted")
            storage.write(dataframe.iloc[0:0])  # Write empty dataframe
            return
        
        # Filter dataframe to only valid images
        dataframe_valid = dataframe.iloc[valid_indices].reset_index(drop=True)
        
        # Find duplicates
        duplicate_indices, duplicate_pairs = self._find_duplicates(embeddings)
        
        # Calculate max similarity for each image (for reference)
        max_similarities = np.zeros(len(embeddings))
        for pair in duplicate_pairs:
            max_similarities[pair['kept_idx']] = max(
                max_similarities[pair['kept_idx']], 
                pair['similarity']
            )
            max_similarities[pair['removed_idx']] = max(
                max_similarities[pair['removed_idx']], 
                pair['similarity']
            )
        
        dataframe_valid[output_score_key] = max_similarities
        
        # Create keep mask (True for non-duplicates)
        keep_mask = np.ones(len(dataframe_valid), dtype=bool)
        keep_mask[list(duplicate_indices)] = False
        
        # Apply filter
        dataframe_filtered = dataframe_valid[keep_mask].reset_index(drop=True)
        
        # Save filtered dataframe
        storage.write(dataframe_filtered)
        
        # delete model
        del self.model