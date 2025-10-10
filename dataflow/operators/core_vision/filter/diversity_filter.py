import os
from typing import List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import imagehash
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY

class TextDuplicateFilter:
    def __init__(self, similarity_thresh: float = 0.8, max_corpus: int = 10000):
        self.sim_thresh = similarity_thresh
        self._texts: List[str] = []
        self.max_corpus = max_corpus

    def check_similarity(self, text: str) -> Tuple[bool, float]:
        if not text or len(text) < 3:
            return False, 0.0
        if not self._texts:
            self._texts.append(text)
            return True, 0.0
        corpus = self._texts[-self.max_corpus:] + [text]
        vectorizer = TfidfVectorizer().fit(corpus)
        tfidf = vectorizer.transform(corpus)
        sims = cosine_similarity(tfidf[-1], tfidf[:-1]).flatten()
        max_sim = float(sims.max())
        if max_sim < self.sim_thresh:
            self._texts.append(text)
            return True, max_sim
        return False, max_sim

class ImageDuplicateFilter:
    def __init__(self, hash_size: int = 8, distance_thresh: int = 5, max_imgs: int = 10000):
        self.hash_size = hash_size
        self.dist_thresh = distance_thresh
        self._hashes: List[imagehash.ImageHash] = []
        self.max_imgs = max_imgs

    def check_distance(self, image_path: str) -> Tuple[bool, Optional[int]]:
        if not os.path.isfile(image_path):
            print(f"[Warning] Image not found: {image_path}")
            return False, None
        try:
            img = Image.open(image_path).convert("RGB")
            h = imagehash.phash(img, hash_size=self.hash_size)
        except (UnidentifiedImageError, OSError) as e:
            print(f"[Warning] Failed to open image: {image_path}, {e}")
            return False, None
        if not self._hashes:
            self._hashes.append(h)
            return True, None
        dists = [h - prev for prev in self._hashes[-self.max_imgs:]]
        min_dist = min(dists)
        if min_dist > self.dist_thresh:
            self._hashes.append(h)
            return True, min_dist
        return False, min_dist

@OPERATOR_REGISTRY.register()
class DiversityFilter(OperatorABC):
    def __init__(self, text_thresh: float = 0.8, hash_size: int = 8, img_dist_thresh: int = 5):
        self.logger = get_logger()
        self.text_filter = TextDuplicateFilter(similarity_thresh=text_thresh)
        self.img_filter  = ImageDuplicateFilter(hash_size=hash_size, distance_thresh=img_dist_thresh)

    @staticmethod
    def get_desc(self, lang):
        return "文本/图像去重过滤" if lang == "zh" else "Text and image duplicate filter."

    def check_diversity(self, image_path: str, text: str) -> Tuple[bool, float, Optional[int]]:
        is_text_unique, text_sim = self.text_filter.check_similarity(text)
        is_img_unique, img_dist  = self.img_filter.check_distance(image_path)
        return is_text_unique and is_img_unique, text_sim, img_dist

    def run(self, storage: DataFlowStorage, image_key: str, text_key: str):
        self.image_key = image_key
        self.text_key = text_key
        dataframe = storage.read("dataframe")
        refined_mask = []
        for i, row in enumerate(tqdm(dataframe.itertuples(), desc=f"Implementing {self.__class__.__name__}")):
            img_path = getattr(row, self.image_key)
            text = getattr(row, self.text_key)
            ok, text_sim, img_dist = self.check_diversity(img_path, text)
            refined_mask.append(ok)
            if not ok:
                self.logger.debug(
                    f"Filtered at row {i}: {img_path}, text_sim={text_sim:.2f}, img_dist={img_dist if img_dist is not None else 'N/A'}"
                )
        dataframe = dataframe[refined_mask].reset_index(drop=True)
        output_file = storage.write(dataframe)
        return [self.image_key, self.text_key]
