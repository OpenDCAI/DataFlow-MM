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
class ImageDeduplicateFilter(OperatorABC):
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
            return (
                "ImageDeduplicateFilter 算子：基于预训练 CLIP 模型提取图像特征，通过余弦相似度进行去重，"
                "删除相似度高于阈值的重复图像，仅保留每组中一张代表图像，并记录每张图像与其它图像的最大相似度。\n"
                "输入：通过 run(storage, input_image_key, output_score_key) 中的 input_image_key 指定，从 DataFlowStorage 中读取图像列；\n"
                "输出：对有效图像提取 CLIP 嵌入，计算两两相似度，删除相似度大于等于阈值 threshold 的重复样本，仅保留首个出现的图像；"
                "同时在过滤后的 DataFrame 中新增由 output_score_key 指定的列（默认名为 max_similarity），"
                "其中每行记录该图像与其它图像之间的最高相似度；最终将结果写回存储，并返回 [input_image_key, output_score_key] 作为后续算子的输入列名；\n"
                "功能：适用于多模态数据清洗场景，通过 CLIP 嵌入统一衡量图像语义相似度，实现近重复图像去重，"
                "减少训练或评测数据中的冗余样本。"
            )
        else:
            return (
                "ImageDeduplicateFilter operator: uses a pretrained CLIP model to extract image embeddings and performs "
                "duplicate removal based on cosine similarity, keeping only one representative image per highly similar group "
                "and recording each image’s maximum similarity to others.\n"
                "Inputs: specified via run(storage, input_image_key, output_score_key), where input_image_key indicates the "
                "column in DataFlowStorage that contains image data or image paths;\n"
                "Output: extracts CLIP embeddings for all valid images, computes pairwise cosine similarities, removes images "
                "whose similarity to an earlier image is greater than or equal to the configured threshold, and keeps only the "
                "first image in each duplicate cluster. A new column named by output_score_key (default 'max_similarity') is "
                "added to the filtered DataFrame to store each image’s maximum similarity to any other image. The filtered "
                "DataFrame is written back to storage and [input_image_key, output_score_key] is returned as the column list for "
                "downstream operators;\n"
                "Function: designed for multimodal dataset cleaning, it leverages CLIP embeddings as a unified semantic representation "
                "to detect and remove near-duplicate images, reducing redundancy in training or evaluation data."
            )

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
    ):
        self.logger.info("Running ImageDeduplicateFilter...")
        dataframe = storage.read("dataframe")
        original_count = len(dataframe)
        self.logger.info(f"Loaded {original_count} rows from storage")
        if input_image_key not in dataframe.columns:
            self.logger.error(f"Image column '{input_image_key}' not found in dataframe")
            raise ValueError(f"Missing required column: {input_image_key}")
        images = dataframe[input_image_key].tolist()
        self.logger.info("Extracting CLIP embeddings...")
        embeddings, valid_indices = self._extract_embeddings(images)
        if len(embeddings) == 0:
            self.logger.error("No valid embeddings extracted")
            storage.write(dataframe.iloc[0:0])
            return [input_image_key, output_score_key]
        dataframe_valid = dataframe.iloc[valid_indices].reset_index(drop=True)
        duplicate_indices, duplicate_pairs = self._find_duplicates(embeddings)
        max_similarities = np.zeros(len(embeddings))
        for pair in duplicate_pairs:
            max_similarities[pair["kept_idx"]] = max(
                max_similarities[pair["kept_idx"]],
                pair["similarity"]
            )
            max_similarities[pair["removed_idx"]] = max(
                max_similarities[pair["removed_idx"]],
                pair["similarity"]
            )
        dataframe_valid[output_score_key] = max_similarities
        keep_mask = np.ones(len(dataframe_valid), dtype=bool)
        keep_mask[list(duplicate_indices)] = False
        dataframe_filtered = dataframe_valid[keep_mask].reset_index(drop=True)
        storage.write(dataframe_filtered)
        del self.model
        return [input_image_key, output_score_key]






