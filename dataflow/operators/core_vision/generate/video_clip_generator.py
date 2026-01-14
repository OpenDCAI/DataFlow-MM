"""
Video cutting operator that consumes `video_clips` (list of video->clips dicts),
cuts only clips with filtered == False, saves to `video_save_dir`,
and writes a flat DataFrame (success-only) to `csv_save_path` & DataFlowStorage.
"""

import os
import queue
import concurrent.futures
import pandas as pd
import subprocess
from tqdm import tqdm
from multiprocessing import Manager
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger

from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC

FFMPEG_PATH = "usr/bin/ffmpeg"


def get_ffmpeg_acceleration():
    try:
        output = subprocess.check_output(
            [FFMPEG_PATH, "-encoders"], stderr=subprocess.DEVNULL
        ).decode("utf-8")
        if "hevc_nvenc" in output or "h264_nvenc" in output:
            return "nvidia"
        return "cpu"
    except Exception as e:
        print(f"FFmpeg acceleration detection failed: {e}")
        return "cpu"


ACCELERATION_TYPE = get_ffmpeg_acceleration()
print(f"FFmpeg acceleration type: {ACCELERATION_TYPE}")


# =========================
# Core cutting primitives
# =========================
def _process_single_clip_row(row_dict: dict, task_params: dict, process_id: int):
    """
    对单条 clip 元信息执行切割。
    仅当 row_dict['filtered'] == False 时执行；否则直接跳过。
    如果 'filtered' 字段不存在，默认为 False（不过滤）。
    返回 (row_list, valid)；valid=True 表示已成功落盘且 row 被更新为新路径。
    """
    # 如果 filtered 字段不存在，默认为 False（不过滤）
    if bool(row_dict.get("filtered", False)) is True:
        return None, False  # 跳过被过滤的片段

    video_path = row_dict["video_path"]
    save_dir = task_params["video_save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    # 不上采样：仅当 shorter_size 更小才缩放
    shorter_size = task_params.get("shorter_size", None)
    if (shorter_size is not None) and ("height" in row_dict) and ("width" in row_dict):
        if min(row_dict["height"], row_dict["width"]) <= shorter_size:
            shorter_size = None

    # timestamp_start and timestamp_end are now integer seconds
    seg_start_sec = row_dict["timestamp_start"]
    seg_end_sec = row_dict["timestamp_end"]

    clip_id = row_dict["id"]
    save_path = os.path.join(save_dir, f"{clip_id}.mp4")

    # 保存原视频路径（在覆盖之前）
    if "original_video_path" not in row_dict:
        row_dict["original_video_path"] = video_path
    
    # 复用已存在文件
    if os.path.exists(save_path):
        row_dict["video_path"] = save_path
        return list(row_dict.values()), True

    try:
        cmd = [FFMPEG_PATH, "-nostdin", "-y"]

        # 输入与精准裁剪（把 -ss 放到 -i 后面）
        cmd += [
            "-i", video_path,
            "-ss", str(seg_start_sec),
            "-to", str(seg_end_sec),
        ]

        # 编码器（稳妥：libx264；若你要 NVENC，解开注释）
        # if ACCELERATION_TYPE == "nvidia":
        #     cmd += ["-c:v", "h264_nvenc", "-preset", "fast"]
        # else:
        cmd += ["-c:v", "libx264", "-preset", "fast", "-crf", "23"]

        # 帧率
        target_fps = task_params.get("target_fps", None)
        if target_fps is not None:
            cmd += ["-r", str(target_fps)]

        # 缩放
        if shorter_size is not None:
            cmd += [
                "-vf",
                f"scale='if(gt(iw,ih),-2,{shorter_size})':'if(gt(iw,ih),{shorter_size},-2)'",
            ]

        # 仅输出视频流
        cmd += ["-map", "0:v:0", save_path]

        subprocess.run(cmd, check=True, stderr=subprocess.PIPE)
        row_dict["video_path"] = save_path
        return list(row_dict.values()), True

    except subprocess.CalledProcessError as e:
        print(f"ffmpeg command failed for {clip_id}: {e.stderr.decode('utf-8')}")
        return None, False


def _worker(task_queue, results_queue, task_params, process_id):
    while True:
        try:
            index, row_dict, columns = task_queue.get(timeout=1)
        except queue.Empty:
            break
        out_row, valid = _process_single_clip_row(row_dict, task_params, process_id)
        if valid and out_row is not None:
            results_queue.put((index, out_row, columns))
        task_queue.task_done()


def _flatten_video_clips(video_clips: list) -> pd.DataFrame:
    """
    把你给的 video_clips（list，每元素含 {success, error, clips:[...] }）
    展平成 DataFrame（每条 clip 一行）。
    """
    rows = []
    for item in video_clips or []:
        if not item or not item.get("success", False):
            continue
        for clip in item.get("clips", []):
            rows.append(clip)
    if not rows:
        return pd.DataFrame()
    # import ipdb; ipdb.set_trace()   
    # 按出现字段并集构建 DataFrame，缺失补 NaN
    all_keys = set().union(*[r.keys() for r in rows])
    df = pd.DataFrame(rows, columns=list(all_keys))
    # 保证这些列存在（有利于后续 drop）
    # 如果 filtered 列不存在，默认设为 False（不过滤任何 clips）
    for need in ["timestamp_start", "timestamp_end", "frame_start", "frame_end", "filtered", "original_video_path"]:
        if need not in df.columns:
            if need == "filtered":
                df[need] = False
            else:
                df[need] = None
    return df


def process_video_cutting_from_list(
    video_clips: list,
    video_save_dir: str,
    drop_invalid_timestamps: bool = False,
    disable_parallel: bool = True,
    num_workers: int = None,
    target_fps: float = None,
    shorter_size: int = None,
) -> pd.DataFrame:
    """
    消费 video_clips 列表；仅切割 filtered == False 的片段；返回成功 DataFrame 并写 CSV。
    如果 filtered 列不存在，会自动添加并设为 False（不过滤）。
    """
    df_in = _flatten_video_clips(video_clips)

    # _flatten_video_clips 已经保证 filtered 列存在（不存在时默认为 False）
    # 所以这里可以安全地使用
    df_todo = df_in[df_in["filtered"] == False].copy()

    task_params = {
        "video_save_dir": video_save_dir,
        "target_fps": target_fps if target_fps is not None else None,
        "shorter_size": shorter_size if shorter_size is not None else None,
    }

    results = []
    logger = get_logger()

    # 为了稳定，明确列顺序
    columns = list(df_in.columns)

    if disable_parallel:
        for idx, row in tqdm(df_todo.iterrows(), total=len(df_todo), desc="Cutting clips (serial)"):
            out_row, valid = _process_single_clip_row(row.to_dict(), task_params, process_id=0)
            if valid and out_row is not None:
                results.append((idx, out_row, columns))
    else:
        manager = Manager()
        task_queue = manager.Queue()
        results_queue = manager.Queue()

        for idx, row in df_todo.iterrows():
            task_queue.put((idx, row.to_dict(), columns))

        num_workers = num_workers or os.cpu_count()
        num_workers = min(num_workers, os.cpu_count())

        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for worker_id in range(num_workers):
                futures.append(
                    executor.submit(
                        _worker, task_queue, results_queue, task_params, worker_id
                    )
                )
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Cutting clips (parallel)"):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Worker failed with error: {str(e)}")

        while not results_queue.empty():
            results.append(results_queue.get())

    # 整理输出
    results.sort(key=lambda x: x[0])

    out_rows = [x[1] for x in results]
    new_df = pd.DataFrame(out_rows, columns=columns)

    # 保留原视频路径和时间戳信息（修改：不再删除这些字段）
    # 添加 original_video_path 字段来明确区分原视频和切割后的视频
    if "video_path" in new_df.columns:
        new_df["original_video_path"] = new_df["video_path"]
    
    # 注释掉原来删除时间戳的代码，保留追溯信息
    # keep_cols = [c for c in new_df.columns if c not in ["timestamp_start", "timestamp_end", "frame_start", "frame_end"]]
    # new_df = new_df[keep_cols]
    
    return new_df


# =========================
# DataFlow Operator
# =========================
@OPERATOR_REGISTRY.register()
class VideoClipGenerator(OperatorABC):
    """
    DataFlow operator:
    - 从 storage 的 dataframe 中取 `video_clips_key` 列（list 的结构）
    - 展平并仅切割 filtered == False 的片段
    - 将成功结果 DataFrame 写入 `csv_save_path` 与 storage[output_key]
    """

    def __init__(
        self,
        video_save_dir: str = "./cache/video_clips",
        drop_invalid_timestamps: bool = False,
        disable_parallel: bool = True,
        num_workers: int = None,
        target_fps: float = None,
        shorter_size: int = None,
    ):
        self.logger = get_logger()
        self.video_save_dir = video_save_dir
        self.drop_invalid_timestamps = drop_invalid_timestamps
        self.disable_parallel = disable_parallel
        self.num_workers = num_workers
        self.target_fps = target_fps
        self.shorter_size = shorter_size

    @staticmethod
    def get_desc(lang: str = "zh") -> str:
        return "切割 video_clips 中未被过滤的片段并保存" if lang == "zh" else "Cut unfiltered clips from video_clips and save."

    def run(
        self,
        storage: DataFlowStorage,
        video_clips_key: str = "video_clip",   # 你 dataframe 里的列名
        output_key: str = "video_info",        # 输出到 storage 的 key
    ):
        if output_key is None:
            raise ValueError("Parameter 'output_key' must not be None.")

        self.logger.info("Running VideoCutGenerator...")
        df = storage.read("dataframe")
        self.logger.info(f"Loaded dataframe with columns: {list(df.columns)}")

        # 从 dataframe 中拿到 video_clips（list 结构）
        video_clips = df.get(video_clips_key, pd.Series([])).tolist()

        
        # 核心处理：只切 filtered == False 的片段
        processed = process_video_cutting_from_list(
            video_clips=video_clips,
            video_save_dir=self.video_save_dir,
            drop_invalid_timestamps=self.drop_invalid_timestamps,
            disable_parallel=self.disable_parallel,
            num_workers=self.num_workers if not self.disable_parallel else 1,
            target_fps=self.target_fps,
            shorter_size=self.shorter_size,
        )
        # df: 上游 dataframe（含原始视频级信息）
        # processed: 切割后的 DataFrame（含每个 clip）

        if "conversation" in df.columns and len(df["conversation"]) > 0:
            # 使用广播的方式将标量值赋给所有行
            processed["conversation"] = [df["conversation"].iloc[0]] * len(processed)
        else:
            processed["conversation"] = None
        processed.rename(columns={"video_path": "video"}, inplace=True)

        storage.write(processed)
        self.logger.info(f"Wrote {len(processed)} rows to storage key '{output_key}'.")