import torch
import clip
import numpy as np
import cv2
import json
from pathlib import Path
from PIL import Image
from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage
from dataflow import get_logger
from dataflow.utils.utils import em_cos_score, get_idf_dict


@OPERATOR_REGISTRY.register()
class EMScoreEval(OperatorABC):
    """
    视频帧级别的EMScore评估算子。
    从视频中按指定策略提取帧，使用EMScorer计算每帧的详细评分。
    """

    def __init__(self, every_n_seconds=None, every_n_frames=None,
                 return_all_frames=False, clip_model_path=None,
                 score_types=None, metrics=None):
        """
        初始化EMScoreEval算子

        Args:
            every_n_seconds (float, optional): 每N秒提取一帧，与every_n_frames互斥
            every_n_frames (int, optional): 每N帧提取一帧，与every_n_seconds互斥
            return_all_frames (bool): 是否在DataFrame中返回每帧详细分数，默认False
            clip_model_path (str, optional): CLIP模型路径，默认自动查找
            score_types (list, optional): 要计算的评分类型，默认全部
                可选: ['EMScore(X,X*)', 'EMScore(X,V)', 'EMScore(X,V,X*)']
            metrics (list, optional): 要输出的指标，默认全部
                可选: ['figr_P', 'figr_R', 'figr_F', 'cogr', 'full_P', 'full_R', 'full_F']
        """
        self.logger = get_logger()
        self.logger.info(f'Initializing {self.__class__.__name__}...')

        # 参数验证
        if sum([every_n_seconds is not None, every_n_frames is not None]) != 1:
            raise ValueError("Must specify exactly one: every_n_seconds or every_n_frames")

        if not torch.cuda.is_available():
            raise RuntimeError("EMScoreEval requires CUDA GPU to run!")

        # 保存配置参数
        self.every_n_seconds = every_n_seconds
        self.every_n_frames = every_n_frames
        self.return_all_frames = return_all_frames
        self.device = "cuda"

        # 设置评分类型和指标
        valid_score_types = ['EMScore(X,X*)', 'EMScore(X,V)', 'EMScore(X,V,X*)']
        valid_metrics = ['figr_P', 'figr_R', 'figr_F', 'cogr', 'full_P', 'full_R', 'full_F']

        if score_types is None:
            self.score_types = valid_score_types
        else:
            for st in score_types:
                if st not in valid_score_types:
                    raise ValueError(f"Invalid score_type: {st}. Valid options: {valid_score_types}")
            self.score_types = score_types

        if metrics is None:
            self.metrics = valid_metrics
        else:
            for m in metrics:
                if m not in valid_metrics:
                    raise ValueError(f"Invalid metric: {m}. Valid options: {valid_metrics}")
            self.metrics = metrics

        # 加载CLIP模型用于特征提取
        self._initialize_clip_model(clip_model_path)

        self.logger.info(f'Score types: {self.score_types}')
        self.logger.info(f'Metrics: {self.metrics}')
        self.logger.info(f'{self.__class__.__name__} initialized.')

    def _initialize_clip_model(self, clip_model_path):
        """初始化CLIP模型"""
        if clip_model_path is None:
            possible_paths = [
                Path.home() / ".cache" / "clip" / "ViT-B-32.pt",
            ]

            default_path = None
            for p in possible_paths:
                if p.exists():
                    default_path = p
                    break

            if default_path:
                self.logger.info(f"Loading CLIP from cache: {default_path}")
                self._model, self._preprocess = clip.load("ViT-B/32", device=self.device)
            else:
                self.logger.info("Loading CLIP ViT-B/32 from default location")
                self._model, self._preprocess = clip.load("ViT-B/32", device=self.device)

        elif Path(clip_model_path).exists():
            self.logger.info(f"Loading CLIP from file: {clip_model_path}")
            self._model, self._preprocess = clip.load(clip_model_path, device=self.device)
        else:
            self.logger.info(f"Loading CLIP model: {clip_model_path}")
            self._model, self._preprocess = clip.load(clip_model_path, device=self.device)

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "视频帧级别EMScore评估算子。从视频中按指定策略提取帧，计算每帧的详细评分。\n\n"
                "初始化参数：\n"
                "- every_n_seconds: 每N秒提取一帧（与every_n_frames二选一）\n"
                "- every_n_frames: 每N帧提取一帧（与every_n_seconds二选一）\n"
                "- return_all_frames: 是否返回每帧详细分数到DataFrame，默认False\n"
                "- clip_model_path: CLIP模型路径，可选\n"
                "- score_types: 评分类型列表，默认全部。可选：\n"
                "  * 'EMScore(X,X*)': 候选文本vs参考文本\n"
                "  * 'EMScore(X,V)': 候选文本vs视频帧\n"
                "  * 'EMScore(X,V,X*)': 综合评分\n"
                "- metrics: 输出指标列表，默认全部。可选：\n"
                "  * 'figr_P/R/F': 局部精确率/召回率/F1\n"
                "  * 'cogr': 全局一致性\n"
                "  * 'full_P/R/F': 综合精确率/召回率/F1\n\n"
                "运行参数：\n"
                "- video_key: 视频路径字段名，默认'video_path'\n"
                "- candidate_key: 候选文本字段名，默认'candidate'\n"
                "- reference_key: 参考文本字段名，默认'reference'\n\n"
                "输出参数：\n"
                "- 各类EMScore评分列（整体平均值）\n"
                "- frame_details列（当return_all_frames=True时，JSON字符串格式）\n"
            )
        elif lang == "en":
            return (
                "Video frame-level EMScore evaluation operator. Extracts frames from videos using specified strategy and computes detailed scores per frame.\n\n"
                "Initialization Parameters:\n"
                "- every_n_seconds: Extract one frame every N seconds (mutually exclusive with every_n_frames)\n"
                "- every_n_frames: Extract one frame every N frames (mutually exclusive with every_n_seconds)\n"
                "- return_all_frames: Whether to return per-frame detailed scores in DataFrame, default False\n"
                "- clip_model_path: Path to CLIP model, optional\n"
                "- score_types: List of score types to compute, default all. Options:\n"
                "  * 'EMScore(X,X*)': Candidate text vs reference text\n"
                "  * 'EMScore(X,V)': Candidate text vs video frames\n"
                "  * 'EMScore(X,V,X*)': Combined score\n"
                "- metrics: List of metrics to output, default all. Options:\n"
                "  * 'figr_P/R/F': Fine-grained precision/recall/F1\n"
                "  * 'cogr': Coarse-grained (global) consistency\n"
                "  * 'full_P/R/F': Full precision/recall/F1\n\n"
                "Run Parameters:\n"
                "- video_key: Video path field name, default 'video_path'\n"
                "- candidate_key: Candidate text field name, default 'candidate'\n"
                "- reference_key: Reference text field name, default 'reference'\n\n"
                "Output Parameters:\n"
                "- EMScore columns for each metric (overall average values)\n"
                "- frame_details column (JSON string format when return_all_frames=True)\n"
            )

    def _extract_frame_features(self, video_path, frame_indices):
        """提取视频帧特征"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_features = {}
        current_idx = 0

        for target_idx in frame_indices:
            while current_idx < target_idx:
                cap.grab()
                current_idx += 1

            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = self._preprocess(Image.fromarray(frame_rgb)).unsqueeze(0).to(self.device)

            with torch.no_grad():
                features = self._model.encode_image(image).float()
                features /= features.norm(dim=-1, keepdim=True)

            frame_key = f"{video_path}#frame{target_idx}"
            frame_features[frame_key] = features.cpu()  # 保持2D形状 (1, feature_dim)

            current_idx += 1

        cap.release()
        return frame_features, fps

    def _compute_frame_indices(self, video_path):
        """计算要提取的帧索引"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if self.every_n_seconds is not None:
            interval = max(1, int(fps * self.every_n_seconds))
            frame_indices = list(range(0, total_frames, interval))
        else:
            frame_indices = list(range(0, total_frames, self.every_n_frames))

        return frame_indices, fps, total_frames

    def _format_frame_results(self, results, frame_indices, fps):
        """格式化每帧的评分结果"""
        frame_results = []

        for i, frame_idx in enumerate(frame_indices):
            frame_score = {
                'frame_index': frame_idx,
                'timestamp': round(frame_idx / fps, 2) if fps > 0 else 0
            }

            for score_type in self.score_types:
                if score_type in results:
                    frame_score[score_type] = {
                        metric: round(float(results[score_type][metric][i].item()), 4)
                        for metric in self.metrics
                        if metric in results[score_type]
                    }

            frame_results.append(frame_score)

        return frame_results

    def _compute_average_scores(self, results):
        """计算所有帧的平均评分"""
        avg_scores = {}

        for score_type in self.score_types:
            if score_type in results:
                for metric in self.metrics:
                    if metric in results[score_type]:
                        score_key = f"{score_type}_{metric}"
                        avg_scores[score_key] = float(results[score_type][metric].mean().item())

        return avg_scores

    def run(self, storage: DataFlowStorage, video_key: str = 'video_path',
            candidate_key: str = 'candidate', reference_key: str = 'reference'):
        """
        运行EMScore评估

        Args:
            storage: DataFlow存储对象
            video_key: 视频路径字段名
            candidate_key: 候选文本字段名
            reference_key: 参考文本字段名

        Returns:
            list: 输出字段名列表
        """
        dataframe = storage.read("dataframe")
        self.logger.info(f"Running EMScoreEval on {len(dataframe)} samples...")

        all_scores = []
        all_frame_details = [] if self.return_all_frames else None

        for idx, row in dataframe.iterrows():
            video_path = row[video_key]
            candidate = row[candidate_key]
            reference = row.get(reference_key, None)

            try:
                # 计算帧索引
                frame_indices, fps, total_frames = self._compute_frame_indices(video_path)

                if not frame_indices:
                    raise ValueError(f"No frames extracted from video: {video_path}")

                self.logger.info(
                    f"Video {idx}: Extracting {len(frame_indices)} frames from {total_frames} total frames")

                # 提取帧特征
                vid_feat_cache, fps = self._extract_frame_features(video_path, frame_indices)

                # 准备EMScorer输入
                frame_keys = [f"{video_path}#frame{fi}" for fi in frame_indices]
                cands = [candidate] * len(frame_indices)
                refs = [reference] * len(frame_indices) if reference else None

                # 准备IDF字典
                idf_corpus = refs if refs else cands
                idf_dict = get_idf_dict(idf_corpus, clip.tokenize, nthreads=4)
                idf_dict[max(list(idf_dict.keys()))] = sum(list(idf_dict.values())) / len(list(idf_dict.values()))

                # 计算EMScore
                results = em_cos_score(
                    self._model,
                    refs,
                    cands,
                    cands,
                    refs,
                    frame_keys,
                    vid_feat_cache,
                    clip.tokenize,
                    idf_dict,
                    self._preprocess,
                    verbose=False,
                    device=self.device,
                    batch_size=64,
                    return_matched_idx=False
                )

                # 处理结果
                final_results = {}

                if refs:
                    refs_all_local_preds = results['refs_result']['figr']
                    refs_all_global_preds = results['refs_result']['cogr']
                    refs_P, refs_R, refs_F = refs_all_local_preds[..., 0], refs_all_local_preds[..., 1], refs_all_local_preds[..., 2]

                    refs_results = {
                        'figr_P': refs_P,
                        'figr_R': refs_R,
                        'figr_F': refs_F,
                        'cogr': refs_all_global_preds,
                        'full_P': (refs_P + refs_all_global_preds) / 2,
                        'full_R': (refs_R + refs_all_global_preds) / 2,
                        'full_F': (refs_F + refs_all_global_preds) / 2
                    }
                    final_results['EMScore(X,X*)'] = refs_results

                if frame_keys:
                    vid_all_local_preds = results['vid_result']['figr']
                    vid_all_global_preds = results['vid_result']['cogr']
                    vid_P, vid_R, vid_F = vid_all_local_preds[..., 0], vid_all_local_preds[..., 1], vid_all_local_preds[..., 2]

                    vid_results = {
                        'figr_P': vid_P,
                        'figr_R': vid_R,
                        'figr_F': vid_F,
                        'cogr': vid_all_global_preds,
                        'full_P': (vid_P + vid_all_global_preds) / 2,
                        'full_R': (vid_R + vid_all_global_preds) / 2,
                        'full_F': (vid_F + vid_all_global_preds) / 2
                    }
                    final_results['EMScore(X,V)'] = vid_results

                if refs and frame_keys:
                    vid_refs_result = {
                        'figr_P': (final_results['EMScore(X,V)']['figr_P'] + final_results['EMScore(X,X*)']['figr_P']) / 2,
                        'figr_R': (final_results['EMScore(X,V)']['figr_R'] + final_results['EMScore(X,X*)']['figr_R']) / 2,
                        'figr_F': (final_results['EMScore(X,V)']['figr_F'] + final_results['EMScore(X,X*)']['figr_F']) / 2,
                        'cogr': (final_results['EMScore(X,V)']['cogr'] + final_results['EMScore(X,X*)']['cogr']) / 2,
                    }
                    vid_refs_result['full_P'] = (vid_refs_result['figr_P'] + vid_refs_result['cogr']) / 2
                    vid_refs_result['full_R'] = (vid_refs_result['figr_R'] + vid_refs_result['cogr']) / 2
                    vid_refs_result['full_F'] = (vid_refs_result['figr_F'] + vid_refs_result['cogr']) / 2
                    final_results['EMScore(X,V,X*)'] = vid_refs_result

                # 格式化每帧结果（如果需要）
                if self.return_all_frames:
                    frame_results = self._format_frame_results(final_results, frame_indices, fps)
                    all_frame_details.append(frame_results)

                # 计算整体平均分
                avg_scores = self._compute_average_scores(final_results)
                all_scores.append(avg_scores)

                self.logger.info(f"Video {idx}: Completed evaluation")

            except Exception as e:
                self.logger.error(f"Error processing video {video_path}: {str(e)}")
                import traceback
                traceback.print_exc()
                all_scores.append({})
                if self.return_all_frames:
                    all_frame_details.append([])

        # 写入平均分到DataFrame
        for key in all_scores[0].keys() if all_scores else []:
            dataframe[key] = [s.get(key, np.nan) for s in all_scores]

        # 如果需要，将每帧详情以JSON字符串形式添加到DataFrame
        if self.return_all_frames:
            dataframe['frame_details'] = [json.dumps(fd, ensure_ascii=False) for fd in all_frame_details]

        # 写入存储
        storage.write(dataframe)
        self.logger.info("EMScoreEval evaluation complete!")

        # 返回输出字段列表
        output_keys = list(all_scores[0].keys()) if all_scores else []
        if self.return_all_frames:
            output_keys.append('frame_details')

        return output_keys