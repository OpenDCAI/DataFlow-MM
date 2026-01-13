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

@OPERATOR_REGISTRY.register()
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

@OPERATOR_REGISTRY.register()
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
class ImageDiversityFilter(OperatorABC):
    def __init__(self, text_thresh: float = 0.8, hash_size: int = 8, img_dist_thresh: int = 5):
        self.logger = get_logger()
        self.text_filter = TextDuplicateFilter(similarity_thresh=text_thresh)
        self.img_filter  = ImageDuplicateFilter(hash_size=hash_size, distance_thresh=img_dist_thresh)

    @staticmethod
    def get_desc(lang: str = "zh") -> str:
        if lang == "zh":
            return (
                "ImageDiversityFilter 算子：同时基于文本相似度与图像感知哈希距离进行去重过滤，保留内容更加多样的图文样本。\n"
                "输入：通过 run(storage, input_image_key, input_text_key) 中的 input_image_key 与 input_text_key 指定，"
                "分别从 DataFlowStorage 中读取图像路径列和对应文本描述列；\n"
                "输出：对每一行样本计算其文本与已保留样本文本之间的 TF-IDF 余弦相似度，以及图像感知哈希的汉明距离，"
                "若文本相似度低于 text_thresh 且图像哈希距离大于 img_dist_thresh，则认为该样本是新颖的并保留，"
                "否则视为近重复样本被过滤；过滤后的 DataFrame 写回存储，并返回 [input_image_key, input_text_key] 作为后续算子的输入列名；\n"
                "功能：使用 TextDuplicateFilter 维护一个文本语料缓存，通过 TF-IDF + 余弦相似度衡量 caption 近似程度；"
                "同时使用 ImageDuplicateFilter 维护一组图像感知哈希，通过哈希距离衡量图像是否近似，"
                "仅在“文本不太相似”且“图像不太相似”这两个条件同时满足时保留样本，可用于构建去重后的高多样性图文数据集。"
            )
        else:
            return (
                "ImageDiversityFilter operator: performs joint deduplication on text and images to keep more diverse image–text pairs.\n"
                "Inputs: specified via run(storage, input_image_key, input_text_key), where input_image_key points to the column "
                "containing image paths and input_text_key points to the column containing text descriptions;\n"
                "Output: for each sample, computes TF-IDF cosine similarity between its text and the texts of previously kept samples, "
                "and computes perceptual hash Hamming distance between its image and previously kept images. A sample is kept only if "
                "its text similarity is below text_thresh and its image hash distance is greater than img_dist_thresh; otherwise it is "
                "treated as a near-duplicate and filtered out. The filtered DataFrame is written back to storage, and "
                "[input_image_key, input_text_key] is returned as the input column list for downstream operators;\n"
                "Function: uses TextDuplicateFilter to maintain a cache of texts and measure similarity with TF-IDF + cosine similarity, "
                "and ImageDuplicateFilter to maintain perceptual hashes and measure image-level redundancy. By requiring both textual "
                "and visual diversity, this operator helps build de-duplicated, high-diversity multimodal datasets."
            )
        
    def check_diversity(self, image_path: str, text: str) -> Tuple[bool, float, Optional[int]]:
        is_text_unique, text_sim = self.text_filter.check_similarity(text)
        is_img_unique, img_dist  = self.img_filter.check_distance(image_path)
        return is_text_unique and is_img_unique, text_sim, img_dist

    def run(self, storage: DataFlowStorage, input_image_key: str = "image_path", input_text_key: str = "text"):
        dataframe = storage.read("dataframe")
        refined_mask = []
        for i, row in enumerate(tqdm(dataframe.itertuples(), desc=f"Implementing {self.__class__.__name__}")):
            img_path = getattr(row, input_image_key)
            text = getattr(row, input_text_key)
            ok, text_sim, img_dist = self.check_diversity(img_path, text)
            refined_mask.append(ok)
            if not ok:
                self.logger.debug(
                    f"Filtered at row {i}: {img_path}, text_sim={text_sim:.2f}, "
                    f"img_dist={img_dist if img_dist is not None else 'N/A'}"
                )
        dataframe = dataframe[refined_mask].reset_index(drop=True)
        storage.write(dataframe)
        return [input_image_key, input_text_key]