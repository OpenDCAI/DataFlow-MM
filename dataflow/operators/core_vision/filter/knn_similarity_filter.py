import pandas as pd
import numpy as np
from typing import Optional, List, Any, Tuple
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.utils import _load_image
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import io


@OPERATOR_REGISTRY.register()
class KNNSimilarityFilter(OperatorABC):
    """
    Filter to keep only the most unique/dissimilar images based on CLIP embeddings.
    
    Uses CLIP to generate image embeddings, then identifies and keeps images
    that are least similar to their k-nearest neighbors.
    """
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        k_neighbors: int = 5,
        keep_ratio: float = 0.5,
        batch_size: int = 32,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the KNNSimilarityFilter operator.
        
        Args:
            model_name: CLIP model name from HuggingFace
            k_neighbors: Number of nearest neighbors to consider
            keep_ratio: Ratio of most unique images to keep (0.5 = keep 50% most unique)
            similarity_threshold: Optional threshold - keep images with avg similarity below this
            batch_size: Batch size for CLIP inference
            device: Device for CLIP model ('cuda' or 'cpu')
        """
        self.logger = get_logger()
        self.k_neighbors = k_neighbors
        self.keep_ratio = keep_ratio
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
            return "基于CLIP相似度过滤，保留最独特的图像"
        return "Filter based on CLIP similarity, keeping most unique images"
    
    def _extract_embeddings(self, images: List[Any]) -> np.ndarray:
        """
        Extract CLIP embeddings for a list of images.
        
        Args:
            images: List of images in various formats
            
        Returns:
            Numpy array of shape (n_images, embedding_dim)
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
    
    def _calculate_uniqueness_scores(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Calculate uniqueness scores based on k-nearest neighbor similarity.
        
        Args:
            embeddings: Image embeddings of shape (n_images, embedding_dim)
            
        Returns:
            Uniqueness scores (lower similarity = more unique)
        """
        n_samples = embeddings.shape[0]
        k = min(self.k_neighbors, n_samples - 1)
        
        if k == 0:
            return np.zeros(n_samples)
        
        # Use sklearn's NearestNeighbors for efficient KNN search
        nbrs = NearestNeighbors(
            n_neighbors=k + 1,
            metric='cosine'
        ).fit(embeddings)
        
        # Find k nearest neighbors for each image
        distances, _ = nbrs.kneighbors(embeddings)
        
        # Calculate average similarity to k nearest neighbors
        similarities = 1 - distances[:, 1:]  # Exclude self (index 0)
        avg_similarities = np.mean(similarities, axis=1)
        
        return avg_similarities

    def run(
        self, 
        storage: DataFlowStorage,
        input_image_key: str = "image",
        output_score_key: str = "knn_similarity_score"
    ) -> None:
        """
        Execute the similarity-based filtering pipeline.
        
        Process flow:
        1. Load data from storage
        2. Extract CLIP embeddings for all images
        3. Calculate k-nearest neighbor similarities
        4. Keep images with lowest average similarity (most unique)
        
        Args:
            storage: DataFlow storage object for reading/writing data
            input_image_key: Column name for image data
            output_score_key: Column name to store uniqueness scores
        """
        self.logger.info("Running KNNSimilarityFilter...")
        
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
        dataframe_valid = dataframe.iloc[valid_indices]
        
        # Calculate uniqueness scores
        self.logger.info("Calculating uniqueness scores...")
        uniqueness_scores = self._calculate_uniqueness_scores(embeddings)
        dataframe_valid[output_score_key] = uniqueness_scores
        
        score_keep_ratio = np.percentile(uniqueness_scores, self.keep_ratio * 100)
        
        filtered_dataframe = dataframe_valid[dataframe_valid[output_score_key] <= score_keep_ratio]
        
        # Save filtered dataframe
        storage.write(filtered_dataframe)