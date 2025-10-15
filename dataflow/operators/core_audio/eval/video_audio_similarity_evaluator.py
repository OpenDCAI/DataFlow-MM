import torch
import clip
import numpy as np
import cv2
from pathlib import Path
from PIL import Image, ImageOps
from typing import Literal
from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage
from dataflow import get_logger


@OPERATOR_REGISTRY.register()
class VideoAudioSimilarity(OperatorABC):
    """视频帧与音频相似度评估算子"""
    
    def __init__(
        self,
        hf_clip: str = 'openai/clip-vit-base-patch32',
        trust_remote_code: bool = False,
        min_score: float = 0.0,
        max_score: float = 1.0,
        frame_sampling_method: Literal['all_keyframes', 'uniform'] = 'uniform',
        frame_num: int = 3,
        horizontal_flip: bool = False,
        vertical_flip: bool = False,
        any_or_all: Literal['any', 'all'] = 'any',
        reduce_mode: Literal['avg', 'max', 'min'] = 'avg',
    ):
        """
        初始化
        
        Args:
            hf_clip: CLIP模型名称
            trust_remote_code: 是否信任HF模型的远程代码
            min_score: 保留样本的最小相似度分数
            max_score: 保留样本的最大相似度分数
            frame_sampling_method: 帧采样方法 (all_keyframes/uniform)
            frame_num: 均匀采样的帧数
            horizontal_flip: 是否水平翻转帧图像
            vertical_flip: 是否垂直翻转帧图像
            any_or_all: 多视频的保留策略 (any/all)
            reduce_mode: 多帧相似度的聚合方式 (avg/max/min)
        """
        self.logger = get_logger()
        
        if not torch.cuda.is_available():
            raise RuntimeError("需要GPU环境!")
        
        self.hf_clip = hf_clip
        self.trust_remote_code = trust_remote_code
        self.min_score = min_score
        self.max_score = max_score
        self.frame_sampling_method = frame_sampling_method
        self.frame_num = frame_num
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.any_or_all = any_or_all
        self.reduce_mode = reduce_mode
        self.device = "cuda"
        
        self._load_model()
    
    def _load_model(self):
        """加载CLIP模型"""
        try:
            if self.hf_clip == 'clip' or 'clip-vit' in self.hf_clip.lower():
                self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
                self.tokenizer = clip.tokenize
                self.model_type = 'clip'
            else:
                from transformers import AutoModel, AutoProcessor
                self.model = AutoModel.from_pretrained(
                    self.hf_clip,
                    trust_remote_code=self.trust_remote_code
                ).to(self.device)
                self.processor = AutoProcessor.from_pretrained(
                    self.hf_clip,
                    trust_remote_code=self.trust_remote_code
                )
                self.model.eval()
                self.model_type = 'transformers'
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    @staticmethod
    def get_desc(lang: str = "zh"):
        """获取算子描述"""
        if lang == "zh":
            return (
                "视频帧与音频相似度评估算子\n\n"
                "参数说明：\n"
                "- hf_clip: CLIP模型名称\n"
                "- frame_sampling_method: 帧采样方法 (all_keyframes/uniform)\n"
                "- frame_num: 均匀采样的帧数\n"
                "- min_score/max_score: 相似度阈值范围\n"
                "- reduce_mode: 聚合方式 (avg/max/min)\n\n"
                "输出字段：\n"
                "- avg_similarity: 平均相似度\n"
                "- max_similarity: 最大相似度\n"
                "- min_similarity: 最小相似度\n"
                "- passed_filter: 是否通过过滤\n"
            )
        elif lang == "en":
            return (
                "Video Frame and Audio Similarity Evaluator\n\n"
                "Parameters:\n"
                "- hf_clip: CLIP model name\n"
                "- frame_sampling_method: Frame sampling method (all_keyframes/uniform)\n"
                "- frame_num: Number of frames for uniform sampling\n"
                "- min_score/max_score: Similarity threshold range\n"
                "- reduce_mode: Aggregation method (avg/max/min)\n\n"
                "Output Fields:\n"
                "- avg_similarity: Average similarity score\n"
                "- max_similarity: Maximum similarity score\n"
                "- min_similarity: Minimum similarity score\n"
                "- passed_filter: Whether passed the filter\n"
            )
        else:
            return "Video frame and audio similarity evaluator"
    
    def _extract_key_frames(self, video_path):
        """提取关键帧"""
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        prev_frame = None
        threshold = 30.0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if prev_frame is not None:
                diff = cv2.absdiff(gray, prev_frame)
                mean_diff = np.mean(diff)
                
                if mean_diff > threshold:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(frame_rgb))
            else:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
            
            prev_frame = gray
        
        cap.release()
        return frames
    
    def _extract_frames_uniform(self, video_path, num_frames):
        """均匀提取帧"""
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if num_frames == 1:
            indices = [total_frames // 2]
        elif num_frames == 2:
            indices = [0, total_frames - 1]
        else:
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
        
        cap.release()
        return frames
    
    def _apply_augmentation(self, frames):
        """应用图像增强"""
        augmented_frames = []
        
        for frame in frames:
            augmented_frames.append(frame)
            
            if self.horizontal_flip:
                augmented_frames.append(ImageOps.mirror(frame))
            
            if self.vertical_flip:
                augmented_frames.append(ImageOps.flip(frame))
        
        return augmented_frames
    
    def _load_audio(self, audio_path):
        """加载音频文件"""
        try:
            import librosa
            audio, sr = librosa.load(str(audio_path), sr=16000)
            return audio
        except ImportError:
            from scipy.io import wavfile
            sr, audio = wavfile.read(str(audio_path))
            
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            elif audio.dtype == np.int32:
                audio = audio.astype(np.float32) / 2147483648.0
            
            return audio
    
    def _compute_similarity(self, frames, audio):
        """计算视频帧和音频的相似度"""
        similarities = []
        
        try:
            with torch.no_grad():
                for frame in frames:
                    if self.model_type == 'clip':
                        image_input = self.preprocess(frame).unsqueeze(0).to(self.device)
                        image_features = self.model.encode_image(image_input)
                        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                        
                        text_input = self.tokenizer(["audio sound"]).to(self.device)
                        text_features = self.model.encode_text(text_input)
                        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                        
                        similarity = (image_features @ text_features.T).item()
                    else:
                        inputs = self.processor(
                            images=frame,
                            audios=audio,
                            return_tensors="pt",
                            padding=True
                        )
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                        outputs = self.model(**inputs)
                        
                        similarity = torch.nn.functional.cosine_similarity(
                            outputs.image_embeds,
                            outputs.audio_embeds
                        ).item()
                    
                    similarities.append(similarity)
        
        except Exception as e:
            self.logger.error(f"Error computing similarity: {e}")
            return []
        
        return similarities
    
    def _reduce_scores(self, scores):
        """根据reduce_mode聚合分数"""
        if not scores:
            return 0.0
        
        if self.reduce_mode == 'avg':
            return np.mean(scores)
        elif self.reduce_mode == 'max':
            return np.max(scores)
        elif self.reduce_mode == 'min':
            return np.min(scores)
        else:
            return np.mean(scores)
    
    def run(self, storage: DataFlowStorage, 
            video_key: str = 'video_path',
            audio_key: str = 'audio_path'):
        """
        运行算子
        
        Args:
            storage: DataFlow存储对象
            video_key: 视频路径字段名
            audio_key: 音频路径字段名
            
        Returns:
            list: 输出字段名列表
        """
        dataframe = storage.read(output_type="dataframe")
        
        avg_similarities = []
        max_similarities = []
        min_similarities = []
        passed_filters = []
        
        for idx, row in dataframe.iterrows():
            try:
                video_path = row[video_key]
                audio_path = row[audio_key]
                
                if self.frame_sampling_method == 'all_keyframes':
                    frames = self._extract_key_frames(video_path)
                else:
                    frames = self._extract_frames_uniform(video_path, self.frame_num)
                
                if not frames:
                    raise ValueError(f"No frames extracted from {video_path}")
                
                if self.horizontal_flip or self.vertical_flip:
                    frames = self._apply_augmentation(frames)
                
                audio = self._load_audio(audio_path)
                
                similarities = self._compute_similarity(frames, audio)
                
                if not similarities:
                    raise ValueError("No similarities computed")
                
                avg_sim = self._reduce_scores(similarities)
                max_sim = np.max(similarities)
                min_sim = np.min(similarities)
                
                passed = self.min_score <= avg_sim <= self.max_score
                
                avg_similarities.append(avg_sim)
                max_similarities.append(max_sim)
                min_similarities.append(min_sim)
                passed_filters.append(passed)
                
            except Exception as e:
                self.logger.error(f"Error processing sample {idx}: {e}")
                avg_similarities.append(np.nan)
                max_similarities.append(np.nan)
                min_similarities.append(np.nan)
                passed_filters.append(False)
        
        dataframe['avg_similarity'] = avg_similarities
        dataframe['max_similarity'] = max_similarities
        dataframe['min_similarity'] = min_similarities
        dataframe['passed_filter'] = passed_filters
        
        # 写入所有数据，不过滤
        storage.write(dataframe)
        
        return ['avg_similarity', 'max_similarity', 'min_similarity', 'passed_filter']