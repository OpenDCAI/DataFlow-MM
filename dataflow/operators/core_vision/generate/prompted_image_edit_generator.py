import os
import pandas as pd
import numpy as np
from pathlib import Path
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger

from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import VLMServingABC


@OPERATOR_REGISTRY.register()
class PromptedImageEditGenerator(OperatorABC):
    def __init__(
        self,
        image_edit_serving: VLMServingABC,
        save_interval: int = 50,
    ):
        self.image_edit_serving = image_edit_serving
        self.save_interval = save_interval

    @staticmethod
    def get_desc(lang: str = "en") -> str:
        return (
            "Generate the corresponding edited results based on the given images and their associated editing instructions."
            if lang != "zh"
            else "基于给定的大量图片以及对应的编辑指令，生成对应的编辑结果"
        )
    
    def _resolve_image_path(self, image_path: str, base_dir: str) -> str:
        """
        Resolve image path. If it's a relative path, resolve it to an absolute path relative to base_dir.
        If it's an absolute path, return it directly.
        """
        if os.path.isabs(image_path):
            return image_path
        else:
            return os.path.normpath(os.path.join(base_dir, image_path))
    
    def _resolve_image_paths(self, image_paths, base_dir: str):
        """
        Resolve image paths (supports single path or list of paths). If relative paths, resolve them relative to base_dir.
        """
        if isinstance(image_paths, str):
            return self._resolve_image_path(image_paths, base_dir)
        elif isinstance(image_paths, list):
            return [self._resolve_image_paths(item, base_dir) for item in image_paths]
        else:
            return image_paths

    def run(
        self,
        storage: DataFlowStorage,
        input_image_key: str = "images",
        input_conversation_key: str = "conversation",
        output_image_key: str = "edited_images",
        save_image_with_idx: bool = True,
    ):
        if output_image_key is None:
            raise ValueError("At least one of output_key must be provided.")

        self.logger = get_logger()
        
        # 获取 prompts.jsonl 文件所在目录，用于解析相对路径
        if hasattr(storage, 'first_entry_file_name') and storage.first_entry_file_name:
            jsonl_file_path = Path(storage.first_entry_file_name).resolve()
            base_dir = str(jsonl_file_path.parent)
            self.logger.info(f"Using base directory for relative paths: {base_dir}")
        else:
            # 如果没有 first_entry_file_name，使用当前工作目录
            base_dir = os.getcwd()
            self.logger.warning(f"Could not determine JSONL file location, using current working directory: {base_dir}")
        
        # 总是从原始输入文件读取所有记录
        df = storage.read(output_type="dict")
        df = pd.DataFrame(df)
        if not isinstance(df, pd.DataFrame):
            raise ValueError("storage.read must return a pandas DataFrame")
        
        total = len(df)
        self.logger.info(f"Read {total} records from input file")

        # 尝试从缓存文件读取已处理的结果（用于断点续跑）
        cache_file_path = os.path.join(storage.cache_path, f"{storage.file_name_prefix}_step1.{storage.cache_type}")
        cache_df = None
        if os.path.exists(cache_file_path):
            try:
                cache_df = pd.DataFrame(storage._load_local_file(cache_file_path, storage.cache_type))
                self.logger.info(f"Found cache file with {len(cache_df)} records, will merge for resume functionality")
            except Exception as e:
                self.logger.warning(f"Failed to load cache file: {e}, will process all records")

        if output_image_key not in df.columns:
            df[output_image_key] = [[] for _ in range(len(df))]
        
        # 如果有缓存，合并已处理的结果
        if cache_df is not None and output_image_key in cache_df.columns:
            for idx in df.index:
                if idx < len(cache_df):
                    cache_output = cache_df.at[idx, output_image_key]
                    if isinstance(cache_output, list) and len(cache_output) > 0 and cache_output[0] != "":
                        df.at[idx, output_image_key] = cache_output

        batch_prompts = []
        skipped_count = 0
        for idx, row in df.iterrows():
            if output_image_key in row.keys():
                output_value = row[output_image_key]
                if isinstance(output_value, list) and len(output_value) > 0:
                    if output_value[0] != "":
                        skipped_count += 1
                        self.logger.info(f"Skipping record {idx} (already processed: {output_value[0]})")
                        continue
            if save_image_with_idx:
                # 构建输入数据，支持多轮对话
                image_path = df.at[idx, input_image_key]
                resolved_image_path = self._resolve_image_paths(image_path, base_dir)
                
                conversations = df.at[idx, input_conversation_key]
                
                # 提取 prompt：对于多轮对话，使用最后一条消息；对于单轮对话，也使用最后一条消息
                if isinstance(conversations, list) and len(conversations) > 0:
                    # 获取最后一条消息的内容
                    last_message = conversations[-1]
                    if isinstance(last_message, dict) and "content" in last_message:
                        prompt_text = last_message["content"]
                    else:
                        prompt_text = str(last_message)
                else:
                    prompt_text = ""
                
                prompt_data = {
                    "idx": idx,
                    "image_path": resolved_image_path,
                    "prompt": prompt_text,
                }
                
                if isinstance(conversations, list) and len(conversations) > 1:
                    prompt_data["conversations"] = conversations
                
                batch_prompts.append(prompt_data)
            else:
                image_path = df.at[idx, input_image_key]
                resolved_image_path = self._resolve_image_paths(image_path, base_dir)
                batch_prompts.append((resolved_image_path, df.at[idx, input_conversation_key][-1]["content"]))
        
        self.logger.info(f"Processing {len(batch_prompts)} records (skipped {skipped_count} already processed)")
        if len(batch_prompts) == 0:
            self.logger.info("No records to process, all records are already completed")
            return
        
        try:
            generated = self.image_edit_serving.generate_from_input(batch_prompts)
        except Exception as e:
            self.logger.error(f"Error during image generation: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            self.logger.warning("Processing failed, skipping cache write")
            return
        
        for idx, prompt in enumerate(batch_prompts):
            if save_image_with_idx:
                result = generated.get(f"sample_{prompt['idx']}", [])
                df.at[prompt['idx'], output_image_key] = result
                if result:
                    self.logger.info(f"Record {prompt['idx']} processed successfully, got {len(result)} image(s)")
                else:
                    self.logger.warning(f"Record {prompt['idx']} processed but got empty result (may have failed)")
            else:
                if isinstance(prompt, tuple):
                    prompt = prompt[1]
                result = generated[idx] if isinstance(generated, list) else generated.get(prompt, [])
                df.at[idx, output_image_key] = result

        try:
            storage.media_key = output_image_key
            storage.write(df)
            self.logger.info(f"Saved {len(df)} records to cache")
        except Exception as e:
            self.logger.error(f"Failed to write cache: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
