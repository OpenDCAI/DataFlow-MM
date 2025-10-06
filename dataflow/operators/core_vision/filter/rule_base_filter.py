import pandas as pd
import numpy as np
import re
from typing import Optional, List, Dict, Any, Tuple
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.utils import _load_image
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from PIL import Image
import torch
from transformers import pipeline


# Helper functions from the provided code
class TextSlice:
    def __init__(self, text: str, start: int, end: int):
        self.text = text
        self.start = start
        self.end = end

def split_paragraphs(text: str, normalizer, remove_empty: bool = True) -> Tuple[TextSlice]:
    """Split a string into paragraphs."""
    text_slices = tuple(
        TextSlice(normalizer(text[match.start():match.end()]), match.start(), match.end())
        for match in re.finditer(r"([^\n]*\n|[^\n]+$)", text)
    )
    
    if remove_empty:
        text_slices = tuple(
            text_slice for text_slice in text_slices if text_slice.text.strip()
        )
    
    return text_slices

def normalize(
    text: str,
    remove_punct: bool = True,
    lowercase: bool = True,
    nfd_unicode: bool = True,
    white_space: bool = True
) -> str:
    import string
    import unicodedata
    
    if remove_punct:
        text = text.translate(str.maketrans('', '', string.punctuation))
    if lowercase:
        text = text.lower()
    if white_space:
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
    if nfd_unicode:
        text = unicodedata.normalize('NFD', text)
    
    return text


@OPERATOR_REGISTRY.register()
class RuleBaseFilter(OperatorABC):
    """
    Multi-modal filter operator using custom filters for text and image data.
    
    Applies comprehensive filtering for both text and image data:
    - Text: Custom heuristic filters for quality and safety
    - Image: Resolution, aspect ratio, quality, and NSFW detection
    """
    
    def __init__(
        self,
        # Text filter parameters (matching pipeline)
        ellipsis_threshold=0.3,
        mean_word_length_min=3,
        mean_word_length_max=20,
        symbol_word_ratio_threshold=0.4,
        id_card_threshold=3,
        no_punc_threshold=112,
        curly_bracket_threshold=0.025,
        capital_words_threshold=0.2,
        lorem_ipsum_threshold=3e-8,
        unique_words_threshold=0.1,
        bulletpoint_threshold=0.9,
        javascript_threshold=3,
        watermarks=['Copyright', 'Watermark', 'Confidential'],
        
        # Image filter parameters
        min_image_width=16,
        min_image_height=16,
        max_image_width=8192,
        max_image_height=8192,
        min_aspect_ratio=0.001,
        max_aspect_ratio=1000,
        
        # Safety parameters
        filter_nsfw_images=True,
        nsfw_threshold=0.5,
        
        # Processing parameters
        batch_size: int = 32,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """Initialize the Multi-Modal Curator Filter."""
        self.logger = get_logger()
        
        # Text filter parameters
        self.ellipsis_threshold = ellipsis_threshold
        self.mean_word_length_min = mean_word_length_min
        self.mean_word_length_max = mean_word_length_max
        self.symbol_word_ratio_threshold = symbol_word_ratio_threshold
        self.id_card_threshold = id_card_threshold
        self.no_punc_threshold = no_punc_threshold
        self.curly_bracket_threshold = curly_bracket_threshold
        self.capital_words_threshold = capital_words_threshold
        self.lorem_ipsum_threshold = lorem_ipsum_threshold
        self.unique_words_threshold = unique_words_threshold
        self.bulletpoint_threshold = bulletpoint_threshold
        self.javascript_threshold = javascript_threshold
        self.watermarks = watermarks
        
        # Image filter parameters
        self.min_image_width = min_image_width
        self.min_image_height = min_image_height
        self.max_image_width = max_image_width
        self.max_image_height = max_image_height
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.allowed_formats = ['JPEG', 'PNG', 'JPG', 'WEBP', 'BMP', 'GIF']
        
        # Safety parameters
        self.filter_nsfw_images = filter_nsfw_images
        self.nsfw_threshold = nsfw_threshold
        
        # Processing parameters
        self.batch_size = batch_size
        self.device = device
        
        # Initialize NSFW detector if needed
        self.nsfw_detector = None
        if self.filter_nsfw_images:
            self._load_nsfw_detector()
        
        # Prepare regex patterns for text filtering
        self._prepare_text_patterns()
    
    @staticmethod
    def get_desc(lang: str = "zh") -> str:
        """Get operator description in specified language."""
        if lang == "zh":
            return "使用自定义过滤器进行多模态数据过滤（文本+图像）"
        return "Multi-modal data filtering using custom filters (text + image)"
    
    def _load_nsfw_detector(self):
        """Load NSFW detection model."""
        self.nsfw_detector = pipeline(
            "image-classification",
            model="Falconsai/nsfw_image_detection",
            device=0 if self.device == "cuda" else -1
        )
        self.logger.info("NSFW detector loaded successfully")
    
    def _prepare_text_patterns(self):
        """Prepare regex patterns for text filtering."""
        # Sentence pattern
        self.SENT_PATTERN = re.compile(r'\b[^.!?\n]+[.!?]*', flags=re.UNICODE)
        
        # ID card pattern
        self.ID_PATTERN = re.compile(
            r"(身\s{0,10}份|id\s{0,10}number\s{0,10}|identification|identity|\s{0,10}ID\s{0,10}No\s{0,10}|"
            r"id\s{0,10}card\s{0,10}|NRIC\s{0,10}number\s{0,10}|IC\s{0,10}number\s{0,10}|"
            r"resident\s{0,10}registration\s{0,10}|I.D.\s{0,10}Number\s{0,10})", 
            re.I
        )
        
        # Special character patterns
        self.SPECIAL_CHAR_PATTERNS = [
            r"u200e",
            r"&#247;|\? :",
            r"[�□]|\{\/U\}",
            r"U\+26[0-F][0-D]|U\+273[3-4]|U\+1F[3-6][0-4][0-F]|U\+1F6[8-F][0-F]"
        ]
        
        # HTML entity patterns
        html_entity = ["nbsp", "lt", "gt", "amp", "quot", "apos", "hellip", "ndash", 
                      "mdash", "lsquo", "rsquo", "ldquo", "rdquo"]
        self.HTML_ENTITIES = []
        for entity in html_entity:
            self.HTML_ENTITIES.extend([
                f"&{entity}；", f"&{entity};", f"＆{entity}；", f"＆{entity};",
                f"＆{entity}", f"&{entity}"
            ])
        
        # Bullet point patterns
        self.BULLET_POINTS = [
            "\u2022", "\u2023", "\u25B6", "\u25C0", "\u25E6", 
            "\u25A0", "\u25A1", "\u25AA", "\u25AB", "\u2013"
        ]
        
        # Lorem ipsum pattern
        self.LOREM_PATTERN = re.compile(r"lorem ipsum", re.IGNORECASE)
        
        # Symbol list
        self.SYMBOLS = ["#", "...", "…"]
    
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
        """Apply all text filters. Returns True if passed, False if failed."""
        if not text or not text.strip():
            return False
    
        words = text.split()
        num_words = len(words)
        
        # 1. Colon end filter
        if text.endswith(':'):
            return False
        
        # 2. Line end with ellipsis filter
        raw_lines = split_paragraphs(text=text, normalizer=lambda x: x, remove_empty=True)
        if raw_lines:
            num_lines = len(raw_lines)
            ellipsis_count = sum([line.text.rstrip().endswith(("...", "…")) for line in raw_lines])
            if num_lines > 0 and (ellipsis_count / num_lines) >= self.ellipsis_threshold:
                return False
        
        # 3. Mean word length filter
        if num_words > 0:
            num_chars = sum(len(word) for word in words)
            mean_length = num_chars / num_words
            if mean_length < self.mean_word_length_min or mean_length >= self.mean_word_length_max:
                return False
        
        # 4. Symbol word ratio filter
        if num_words > 0:
            num_symbols = float(sum(text.count(symbol) for symbol in self.SYMBOLS))
            ratio = num_symbols / num_words
            if ratio >= self.symbol_word_ratio_threshold:
                return False
        
        # 5. HTML entity filter
        if any(entity in text for entity in self.HTML_ENTITIES):
            return False
        
        # 6. ID card filter
        matches = self.ID_PATTERN.findall(text)
        if len(matches) >= self.id_card_threshold:
            return False
        
        # 7. No punctuation filter (max words in a sentence)
        paragraphs = text.split('\n')
        max_word_count = 0
        for paragraph in paragraphs:
            if len(paragraph.strip()) == 0:
                continue
            sentences = re.split("[–.!?,;•/|…]", paragraph)
            for sentence in sentences:
                word_count = len(sentence.split())
                if word_count > max_word_count:
                    max_word_count = word_count
        
        if max_word_count > self.no_punc_threshold:
            return False
        
        # 8. Special character filter
        if any(re.search(pattern, text) for pattern in self.SPECIAL_CHAR_PATTERNS):
            return False
        
        # 9. Watermark filter
        if self.watermarks and re.search('|'.join(self.watermarks), text):
            return False
        
        # 10. Curly bracket filter
        if len(text) > 0:
            bracket_ratio = (text.count('{') + text.count('}')) / len(text)
            if bracket_ratio >= self.curly_bracket_threshold:
                return False
        
        # 11. Capital words filter
        if num_words > 0:
            num_caps_words = sum(map(str.isupper, words))
            caps_ratio = num_caps_words / num_words
            if caps_ratio > self.capital_words_threshold:
                return False
        
        # 12. Lorem ipsum filter
        if len(text) > 0:
            lorem_count = len(self.LOREM_PATTERN.findall(text.lower()))
            lorem_ratio = lorem_count / len(text)
            if lorem_ratio > self.lorem_ipsum_threshold:
                return False
        
        # 13. Unique words filter
        if num_words > 0:
            normalized_words = tuple(text.lower().split())
            num_unique = len(set(normalized_words))
            unique_ratio = num_unique / len(normalized_words)
            if unique_ratio <= self.unique_words_threshold:
                return False
        
        # 14. Line start with bulletpoint filter
        if raw_lines and len(raw_lines) > 0:
            bullet_count = sum([line.text.lstrip().startswith(tuple(self.BULLET_POINTS)) 
                               for line in raw_lines])
            bullet_ratio = bullet_count / len(raw_lines)
            if bullet_ratio > self.bulletpoint_threshold:
                return False
        
        # 15. Line with javascript filter
        normalized_lines = split_paragraphs(text=text, normalizer=normalize, remove_empty=True)
        if normalized_lines:
            num_lines = len(normalized_lines)
            js_count = sum(['javascript' in line.text.lower() for line in normalized_lines])
            num_not_occur = num_lines - js_count
            if not (num_lines <= 3 or num_not_occur >= self.javascript_threshold):
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
        
        # Log results
        filtered_count = len(filtered_df)
        self.logger.info(f"Filtering complete: {filtered_count}/{original_count} samples passed")
        self.logger.info(f"Filtered out {original_count - filtered_count} samples")
        
        # Save filtered dataframe
        storage.write(filtered_df)