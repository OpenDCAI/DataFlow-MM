import pandas as pd
import numpy as np
from typing import Optional, List, Any, Tuple, Union
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.utils import _load_image
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image
from tqdm import tqdm


@OPERATOR_REGISTRY.register()
class DataTailorFilter(OperatorABC):
    """
    Evaluation operator that computes DataTailor's three core metrics for data quality assessment.
    
    Computes three metrics from DataTailor paper:
    - Informativeness: Measures sample difficulty via singular value entropy of multi-modal token embeddings
    - Uniqueness: Captures deviation from local data density
    - Representativeness: Ensures samples align with overall distribution
    
    This operator only evaluates and adds metrics to dataframe without filtering.
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
        clustering_threshold: float = 0.1,
        extract_layer: int = -2,  # Penultimate layer
        keep_ratio: float = 0.8,
        batch_size: int = 4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the DataTailorEval operator.
        
        Args:
            model_name: Qwen2-VL model name from HuggingFace
            clustering_threshold: Threshold for hierarchical clustering
            extract_layer: Which layer to extract features from (-2 = penultimate)
            batch_size: Batch size for model inference
            device: Device for model ('cuda' or 'cpu')
        """
        self.logger = get_logger()
        self.clustering_threshold = clustering_threshold
        self.extract_layer = extract_layer
        self.keep_ratio = keep_ratio
        self.batch_size = batch_size
        self.device = device
        
        # Load Qwen2-VL model and processor
        self.logger.info(f"Loading Qwen2-VL model: {model_name}")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model.eval()
    
    @staticmethod
    def get_desc(lang: str = "zh") -> str:
        if lang == "zh":
            return "计算DataTailor三个质量评估指标"
        return "Compute DataTailor's three quality assessment metrics"
    
    def _prepare_qwen_messages(
        self, 
        image: Any,
        question: str,
        answer: str,
    ) -> List[dict]:
        """
        Prepare messages format for Qwen2-VL.
        """
        if image is None:
            return None
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question + answer}
                ]
            }
        ]
        
        return messages
    
    def _extract_multimodal_features(
        self, 
        images: List[Any],
        questions: List[str],
        answers: List[str]
    ) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
        """
        Extract multi-modal token embeddings and sample features from Qwen2-VL.
        
        Returns:
            Tuple of (token_features_list, sample_features, valid_indices)
        """
        token_features_list = []
        sample_features_list = []
        valid_indices = []
        
        # Process samples one by one (Qwen2-VL doesn't support batch well for feature extraction)
        for idx, (image, question, answer) in enumerate(zip(images, questions, answers)):
            try:
                # Prepare input
                messages = self._prepare_qwen_messages(image, question, answer)
                if messages is None:
                    continue
                
                # Process through Qwen2-VL
                text_input = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                # Tokenize and prepare inputs
                inputs = self.processor(
                    text=[text_input],
                    images=[_load_image(image)],
                    padding=True,
                    return_tensors="pt"
                ).to(self.device)
                
                # Forward pass with output_hidden_states
                with torch.no_grad():
                    outputs = self.model(
                        **inputs,
                        output_hidden_states=True,
                        return_dict=True
                    )
                
                # Extract token embeddings from specified layer
                hidden_states = outputs.hidden_states
                
                # Get token embeddings from the specified layer
                token_embeddings = hidden_states[self.extract_layer][0]  # Remove batch dimension
                token_embeddings_np = token_embeddings.cpu().numpy()
                
                # Store token features for SVE calculation
                token_features_list.append(token_embeddings_np)
                
                # Get sample-level representation (mean pooling of all tokens)
                sample_embedding = token_embeddings.mean(dim=0).cpu().numpy()
                sample_features_list.append(sample_embedding)
                
                valid_indices.append(idx)
                
            except Exception as e:
                self.logger.warning(f"Failed to extract features for sample {idx}: {e}")
                continue
        
        if not sample_features_list:
            return [], np.array([]), np.array([])
        
        sample_features = np.vstack(sample_features_list)
        return token_features_list, sample_features, np.array(valid_indices)
    
    def _calculate_informativeness(self, token_features_list: List[np.ndarray]) -> np.ndarray:
        """
        Calculate informativeness using singular value entropy (SVE) of multi-modal token embeddings.
        
        Based on DataTailor paper equation (2):
        V_i^Inf = -∑(σ_j/∑σ_k) * log(σ_j/∑σ_k)
        
        Args:
            token_features_list: List of token embedding matrices [n_tokens, hidden_dim] for each sample
                                These include both image and text tokens from Qwen2-VL
        """
        n_samples = len(token_features_list)
        informativeness = np.zeros(n_samples)
        
        for i, token_features in enumerate(tqdm(token_features_list)):
            # token_features shape: [n_tokens, hidden_dim]
            # This includes both image and text tokens from Qwen2-VL
            
            try:
                # Ensure we have enough tokens
                if token_features.shape[0] < 2:
                    informativeness[i] = 0.0
                    continue
                
                # Compute SVD on the multi-modal token feature matrix
                token_features = token_features.astype(np.float32)
                _, singular_values, _ = np.linalg.svd(token_features, full_matrices=False)
                
                # Filter near-zero singular values
                singular_values = singular_values[singular_values > 1e-8]
                
                if len(singular_values) == 0:
                    informativeness[i] = 0.0
                    continue
                
                # Normalize singular values
                normalized_sv = singular_values / singular_values.sum()
                
                # Compute entropy (higher entropy = more complex/informative)
                entropy = -np.sum(normalized_sv * np.log(normalized_sv + 1e-10))
                informativeness[i] = entropy
                
            except np.linalg.LinAlgError:
                self.logger.warning(f"SVD failed for sample {i}, assigning 0")
                informativeness[i] = 0.0
        
        return informativeness
    
    def _cluster_samples(self, sample_features: np.ndarray) -> np.ndarray:
        """
        Perform hierarchical clustering on sample embeddings.
        """
        n_samples = sample_features.shape[0]
        
        if n_samples <= 2:
            return np.zeros(n_samples, dtype=int)
        
        # Determine number of clusters
        n_clusters = max(2, int(n_samples * self.clustering_threshold))
        n_clusters = min(n_clusters, n_samples // 2)  # At least 2 samples per cluster on average
        
        # Perform agglomerative clustering
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='cosine',
            linkage='average'
        )
        
        cluster_labels = clustering.fit_predict(sample_features)
        return cluster_labels
    
    def _calculate_uniqueness(
        self, 
        sample_features: np.ndarray,
        cluster_labels: np.ndarray,
        informativeness: np.ndarray
    ) -> np.ndarray:
        """
        Calculate uniqueness within clusters using weighted distances.
        
        Based on DataTailor paper equation (4):
        V_i^Uni = ∑||p_j - p_i||_2 * (V_j^Inf/∑V_k^Inf)
        """
        n_samples = sample_features.shape[0]
        uniqueness = np.zeros(n_samples)
        
        unique_clusters = np.unique(cluster_labels)
        
        for cluster_id in unique_clusters:
            cluster_mask = cluster_labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) <= 1:
                uniqueness[cluster_indices] = 0
                continue
            
            cluster_features = sample_features[cluster_mask]
            cluster_informativeness = informativeness[cluster_mask]
            
            # Normalize informativeness weights
            if cluster_informativeness.sum() > 0:
                inf_weights = cluster_informativeness / cluster_informativeness.sum()
            else:
                inf_weights = np.ones(len(cluster_informativeness)) / len(cluster_informativeness)
            
            # Calculate weighted distances for each sample in cluster
            for i, global_idx in enumerate(cluster_indices):
                # Euclidean distances to all other samples in cluster
                distances = np.linalg.norm(
                    cluster_features - cluster_features[i:i+1],
                    axis=1
                )
                distances[i] = 0  # Exclude self
                
                # Weighted sum of distances
                weighted_distance = np.sum(distances * inf_weights)
                uniqueness[global_idx] = weighted_distance
        
        return uniqueness
    
    def _calculate_representativeness(
        self,
        sample_features: np.ndarray,
        cluster_labels: np.ndarray,
        informativeness: np.ndarray
    ) -> np.ndarray:
        """
        Calculate representativeness using inter-cluster similarities.
        
        Based on DataTailor paper equations (5) and (6):
        τ_i^c = (1/(K-1)) * ∑exp(sim(p_k, p_c))
        V_i^Rep = τ_i^c * (V_i^Inf/∑V_k^Inf)
        """
        n_samples = sample_features.shape[0]
        representativeness = np.zeros(n_samples)
        
        unique_clusters = np.unique(cluster_labels)
        n_clusters = len(unique_clusters)
        
        if n_clusters <= 1:
            return np.ones(n_samples)
        
        # Calculate cluster centroids
        cluster_centroids = []
        for cluster_id in unique_clusters:
            cluster_mask = cluster_labels == cluster_id
            cluster_features = sample_features[cluster_mask]
            centroid = cluster_features.mean(axis=0)
            cluster_centroids.append(centroid)
        cluster_centroids = np.array(cluster_centroids)
        
        # Normalize centroids for cosine similarity
        cluster_centroids = cluster_centroids / (np.linalg.norm(cluster_centroids, axis=1, keepdims=True) + 1e-8)
        
        # Calculate inter-cluster similarities
        inter_cluster_sims = cosine_similarity(cluster_centroids)
        
        # Calculate representativeness for each sample
        for cluster_idx, cluster_id in enumerate(unique_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            
            # Association coefficient for this cluster
            other_sims = inter_cluster_sims[cluster_idx].copy()
            other_sims[cluster_idx] = 0  # Exclude self
            
            tau_c = np.exp(other_sims).sum() / (n_clusters - 1)
            
            # Weighted by informativeness
            cluster_informativeness = informativeness[cluster_mask]
            
            if cluster_informativeness.sum() > 0:
                inf_weights = cluster_informativeness / cluster_informativeness.sum()
            else:
                inf_weights = np.ones(len(cluster_informativeness)) / len(cluster_informativeness)
            
            # Assign weighted representativeness
            for i, global_idx in enumerate(cluster_indices):
                representativeness[global_idx] = tau_c * inf_weights[i]
        
        return representativeness
    
    def _perform_selection(
        self,
        informativeness: np.ndarray,
        uniqueness: np.ndarray,
        representativeness: np.ndarray,
        keep_ratio: float = None
    ) -> np.ndarray:
        """
        V_i = (1/3) * V_i^Inf + (1/3) * (V_i^Uni + V_i^Rep)
        """        
        n_samples = len(informativeness)
        
        inf_norm = np.argsort(np.argsort(informativeness)) / n_samples
        uni_norm = np.argsort(np.argsort(uniqueness)) / n_samples
        rep_norm = np.argsort(np.argsort(representativeness)) / n_samples
        
        collaborative_scores = (1/3) * inf_norm + (2/3) * (uni_norm + rep_norm) / 2
        
        n_select = max(1, int(n_samples * keep_ratio))
        selected_indices = np.argsort(collaborative_scores)[-n_select:]
        
        return np.sort(selected_indices)

    def run(
        self, 
        storage: DataFlowStorage,
        input_image_key: str = "image",
        input_question_key: str = "question",
        input_answer_key: str = "answer",
    ) -> None:
        """
        Execute the DataTailor evaluation pipeline.
        
        Process flow:
        1. Load data from storage
        2. Extract multi-modal token embeddings from Qwen2-VL
        3. Calculate informativeness using SVE of all tokens (image + text)
        4. Perform clustering on sample embeddings
        5. Calculate uniqueness (intra-cluster distances)
        6. Calculate representativeness (inter-cluster similarities)
        7. Add metrics to dataframe and save
        
        Args:
            storage: DataFlow storage object
            input_image_key: Column name for image data
            input_text_key: Column name for text/answer data
            input_question_key: Optional column name for question/prompt
        """
        self.logger.info("Running DataTailorEval with Qwen2-VL...")
        
        # Load dataframe
        dataframe = storage.read('dataframe')
        total_count = len(dataframe)
        self.logger.info(f"Loaded {total_count} rows from storage")
        
        # Extract columns
        images = dataframe.get(input_image_key, pd.Series([])).tolist()
        questions = dataframe.get(input_question_key, pd.Series([])).tolist()
        answers = dataframe.get(input_answer_key, pd.Series([])).tolist()
        
        # Initialize metric columns
        dataframe['informativeness'] = np.nan
        dataframe['uniqueness'] = np.nan
        dataframe['representativeness'] = np.nan
        
        # Extract multi-modal features from Qwen2-VL
        self.logger.info("Extracting multi-modal token embeddings from Qwen2-VL...")
        token_features_list, sample_features, valid_indices = self._extract_multimodal_features(
            images, questions, answers
        )
        
        dataframe = dataframe.iloc[valid_indices].reset_index(drop=True)
        
        if len(sample_features) == 0:
            self.logger.warning("No valid features extracted, all metrics will be NaN")
            storage.write(dataframe)
            return
        
        # Calculate metrics
        self.logger.info("Calculating informativeness (SVE of Qwen2-VL multi-modal tokens)...")
        informativeness = self._calculate_informativeness(token_features_list)
        
        self.logger.info("Clustering samples...")
        cluster_labels = self._cluster_samples(sample_features)
        
        self.logger.info("Calculating uniqueness...")
        uniqueness = self._calculate_uniqueness(sample_features, cluster_labels, informativeness)
        
        self.logger.info("Calculating representativeness...")
        representativeness = self._calculate_representativeness(
            sample_features, cluster_labels, informativeness
        )
        
        # Add metrics to dataframe
        dataframe['informativeness'] = informativeness
        dataframe['uniqueness'] = uniqueness
        dataframe['representativeness'] = representativeness
        
        self.logger.info(f"Performing data selection (keep_ratio={self.keep_ratio})...")

        selected_idx = self._perform_selection(
            informativeness,
            uniqueness,
            representativeness,
            self.keep_ratio
        )

        dataframe = dataframe.iloc[selected_idx].reset_index(drop=True)

        storage.write(dataframe)