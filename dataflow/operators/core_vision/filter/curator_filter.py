import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.utils import _load_image
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage

# Correct NeMo Curator imports based on actual availability
from nemo_curator.filters import (
    WordCountFilter,
    RepeatingTopNGramsFilter,
    RepeatingDuplicateNGramsFilter,
    CommonEnglishWordsFilter,
    LongWordFilter,
    WordsWithoutAlphabetsFilter,
)

# Import from heuristic_filter submodule
from nemo_curator.filters.heuristic_filter import (
    NonAlphaNumericFilter,
    SymbolsToWordsFilter,
    NumbersFilter,
    UrlsFilter,
    WhiteSpaceFilter,
    ParenthesesFilter,
    BoilerPlateStringFilter,
    RepeatedLinesFilter,
)

# For code-specific filtering
from nemo_curator.filters import AlphaFilter

from PIL import Image
import cv2
import torch
from transformers import pipeline


@OPERATOR_REGISTRY.register()
class CuratorFilter(OperatorABC):
    """
    Multi-modal filter operator using NVIDIA NeMo Curator for text and custom filters for images.
    
    Applies comprehensive filtering for both text and image data:
    - Text: NeMo Curator heuristic filters for quality and safety
    - Image: Resolution, aspect ratio, quality, and NSFW detection
    """
    
    def __init__(
        self,
        # Text filters (NeMo Curator)
        min_words=1,
        max_words=50000,
        max_word_length=20000,
        max_symbol_to_word_ratio=0.5,
        max_non_alpha_numeric_ratio=0.7,
        max_number_ratio=0.7,
        max_url_ratio=0.8,
        max_white_space_ratio=0.7,
        max_parentheses_ratio=0.5,
        max_boilerplate_ratio=0.7,
        remove_boilerplate_at_edges=False,
        max_repeated_lines_ratio=0.95,
        max_repeating_duplicate_ngrams_ratio=0.8,
        max_repeating_top_ngrams_ratio=0.8,
        repeating_ngram_n=7,
        min_words_with_alphabets_ratio=0.5,
        min_common_words_ratio=0.05,
        min_common_words_count=2,

        min_image_width=2,
        min_image_height=2,
        max_image_width=8192,
        max_image_height=8192,
        min_aspect_ratio=0.0001,
        max_aspect_ratio=10000,

        min_brightness=1.0,
        max_brightness=254.0,
        min_contrast=1.0,
        blur_threshold=10.0,

        filter_nsfw_images=True,
        nsfw_threshold=0.7,
        
        # Processing
        batch_size: int = 32,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the Multi-Modal Curator Filter.
        """
        self.logger = get_logger()
        
        # Store text filter parameters
        self.min_words = min_words
        self.max_words = max_words
        self.max_word_length = max_word_length
        self.max_symbol_to_word_ratio = max_symbol_to_word_ratio
        self.max_non_alpha_numeric_ratio = max_non_alpha_numeric_ratio
        self.max_number_ratio = max_number_ratio
        self.max_url_ratio = max_url_ratio
        self.max_white_space_ratio = max_white_space_ratio
        self.max_parentheses_ratio = max_parentheses_ratio
        self.max_boilerplate_ratio = max_boilerplate_ratio
        self.remove_boilerplate_at_edges = remove_boilerplate_at_edges
        self.max_repeated_lines_ratio = max_repeated_lines_ratio
        self.max_repeating_duplicate_ngrams_ratio = max_repeating_duplicate_ngrams_ratio
        self.max_repeating_top_ngrams_ratio = max_repeating_top_ngrams_ratio
        self.repeating_ngram_n = repeating_ngram_n
        self.min_words_with_alphabets_ratio = min_words_with_alphabets_ratio
        self.min_common_words_ratio = min_common_words_ratio
        self.min_common_words_count = min_common_words_count
        
        # Store image filter parameters
        self.min_image_width = min_image_width
        self.min_image_height = min_image_height
        self.max_image_width = max_image_width
        self.max_image_height = max_image_height
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.allowed_formats =  ['JPEG', 'PNG', 'JPG', 'WEBP', 'BMP', 'GIF']
        
        # Image quality parameters
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.min_contrast = min_contrast
        self.blur_threshold = blur_threshold
        
        # Safety parameters
        self.filter_nsfw_images = filter_nsfw_images
        self.nsfw_threshold = nsfw_threshold
        
        # Processing parameters
        self.batch_size = batch_size
        self.device = device
        
        # Initialize text filters
        self.text_filters = []
        self._setup_text_filters()
        
        # Initialize NSFW detector if needed
        self.nsfw_detector = None
        if self.filter_nsfw_images:
            self._load_nsfw_detector()
    
    @staticmethod
    def get_desc(lang: str = "zh") -> str:
        """Get operator description in specified language."""
        if lang == "zh":
            return "使用NeMo Curator进行多模态数据过滤（文本+图像）"
        return "Multi-modal data filtering using NeMo Curator (text + image)"
    
    def _load_nsfw_detector(self):
        """Load NSFW detection model."""
        self.nsfw_detector = pipeline(
            "image-classification",
            model="Falconsai/nsfw_image_detection",
            device=0 if self.device == "cuda" else -1
        )
        self.logger.info("NSFW detector loaded successfully")
    
    def _setup_text_filters(self):
        """Setup NeMo Curator text filters."""
        
        # Length-based filters
        self.text_filters.append(
            WordCountFilter(min_words=self.min_words, max_words=self.max_words)
        )
        
        self.text_filters.append(
            LongWordFilter(max_word_length=self.max_word_length)
        )
        
        # Content quality filters from heuristic_filter module
        self.text_filters.append(
            NonAlphaNumericFilter(max_non_alpha_numeric_to_text_ratio=self.max_non_alpha_numeric_ratio)
        )
        
        self.text_filters.append(
            SymbolsToWordsFilter(max_symbol_to_word_ratio=self.max_symbol_to_word_ratio)
        )
        
        self.text_filters.append(
            NumbersFilter(max_number_to_text_ratio=self.max_number_ratio)
        )
        
        self.text_filters.append(
            UrlsFilter(max_url_to_text_ratio=self.max_url_ratio)
        )
        
        self.text_filters.append(
            WhiteSpaceFilter(max_white_space_ratio=self.max_white_space_ratio)
        )
        
        self.text_filters.append(
            ParenthesesFilter(max_parentheses_ratio=self.max_parentheses_ratio)
        )
        
        # Boilerplate filter - check the exact parameter names
        self.text_filters.append(
            BoilerPlateStringFilter(max_boilerplate_string_ratio=self.max_boilerplate_ratio)
        )
        
        # Repetition filters
        self.text_filters.append(
            RepeatedLinesFilter(max_repeated_line_fraction=self.max_repeated_lines_ratio)
        )
        
        # These filters use max_ratio parameter according to the documentation
        self.text_filters.append(
            RepeatingDuplicateNGramsFilter(
                n=self.repeating_ngram_n,
                max_repeating_duplicate_ngram_ratio=self.max_repeating_duplicate_ngrams_ratio
            )
        )
        
        self.text_filters.append(
            RepeatingTopNGramsFilter(
                n=self.repeating_ngram_n,
                max_repeating_ngram_ratio=self.max_repeating_top_ngrams_ratio
            )
        )
        
        # Language quality filters
        self.text_filters.append(
            WordsWithoutAlphabetsFilter(min_words_with_alphabets=self.min_words_with_alphabets_ratio)
        )
        
        # CommonEnglishWordsFilter might have different parameter names
        self.text_filters.append(
            CommonEnglishWordsFilter(min_num_common_words=self.min_common_words_count)
        )
        
        self.logger.info(f"Initialized {len(self.text_filters)} text filters")
    
    def _check_image_basic(self, image: Image.Image) -> bool:
        """Check basic image properties."""
        # Check format
        if image.format and image.format.upper() not in self.allowed_formats:
            return False
        
        # Check dimensions
        width, height = image.size
        if width < self.min_image_width or height < self.min_image_height:
            return False
        
        if width > self.max_image_width or height > self.max_image_height:
            return False
        
        # Check aspect ratio
        aspect_ratio = width / height
        if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
            return False
        
        return True
    
    def _check_image_quality(self, image: Image.Image) -> bool:
        """Check image quality metrics."""
        # Convert to numpy array
        img_array = np.array(image.convert('L'))  # Convert to grayscale
        
        # Check brightness
        mean_brightness = np.mean(img_array)
        if mean_brightness < self.min_brightness:
            return False
        if mean_brightness > self.max_brightness:
            return False
        
        # Check contrast
        contrast = np.std(img_array)
        if contrast < self.min_contrast:
            return False
        
        # Check blur using Laplacian variance
        laplacian = cv2.Laplacian(img_array, cv2.CV_64F)
        blur_score = laplacian.var()
        if blur_score < self.blur_threshold:
            return False
        
        return True
    
    def _check_nsfw_content(self, image: Image.Image) -> bool:
        """Check if image contains NSFW content. Returns True if safe, False if NSFW."""
        if not self.nsfw_detector:
            return True
        
        # Run NSFW detection
        results = self.nsfw_detector(image)
        
        # Parse results - looking for 'nsfw' or 'porn' labels
        for result in results:
            label = result.get('label', '').lower()
            score = result.get('score', 0)
            
            if ('nsfw' in label or 'porn' in label or 'explicit' in label) and score > self.nsfw_threshold:
                return False
        
        return True
    
    def _apply_text_filters(self, text: str) -> bool:
        """Apply NeMo Curator text filters. Returns True if passed, False if failed."""
        for filter_obj in self.text_filters:
            # Most filters expect the text directly, not a document dictionary
            score = filter_obj.score_document(text)
            if not filter_obj.keep_document(score):
                return False
                
        return True
    
    def _filter_sample(self, text: Optional[str], image: Optional[Any]) -> bool:
        """Filter a single multi-modal sample. Returns True if keep, False if filter out."""
        # Check text if provided
        if text and isinstance(text, str) and len(text.strip()) > 0:
            if not self._apply_text_filters(text):
                return False
        
        # Check image if provided
        if image is not None:
            # Load image
            pil_image = _load_image(image)
            if pil_image is None:
                return False
            
            # Basic checks
            if not self._check_image_basic(pil_image):
                return False
            
            # Quality checks
            if not self._check_image_quality(pil_image):
                return False
            
            # NSFW check
            if self.filter_nsfw_images:
                if not self._check_nsfw_content(pil_image):
                    return False
        
        return True
    
    def run(
        self,
        storage: DataFlowStorage,
        input_image_key: str = "text",
        input_question_key: str = "question",
        input_answer_key: str = "answer"
    ) -> None:
        """
        Execute the multi-modal filtering pipeline.
        
        Args:
            storage: DataFlow storage object
            input_image_key: Column name for image data
            input_question_key: Column name for question data 
            input_answer_key: Column name for answer data
        """
        self.logger.info("Starting Multi-Modal Curator Filter...")
        
        # Load dataframe
        dataframe = storage.read('dataframe')
        original_count = len(dataframe)
        self.logger.info(f"Loaded {original_count} rows from storage")
        
        # Extract data
        images = dataframe[input_image_key].tolist()
        questions = dataframe.get(input_question_key).tolist() if input_question_key in dataframe else [None] * len(dataframe)
        answers = dataframe.get(input_answer_key).tolist() if input_answer_key in dataframe else [None] * len(dataframe)
        
        # Apply filters
        keep_mask = []
        filtered_count = 0
        
        for i, (question, answer, image) in enumerate(zip(questions, answers, images)):
            # Combine question and answer for text filtering
            text = ""
            if question:
                text += str(question) + " "
            if answer:
                text += str(answer)
            
            keep = self._filter_sample(text if text else None, image)
            keep_mask.append(keep)
        
        # Apply filter mask
        keep_mask = np.array(keep_mask)
        filtered_df = dataframe[keep_mask].reset_index(drop=True)
        
        # Save filtered dataframe
        storage.write(filtered_df)