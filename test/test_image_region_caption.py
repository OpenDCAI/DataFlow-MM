import argparse
import numpy as np  # 原代码中paint_text_box依赖numpy，需确保导入
from typing import Optional
import sys
sys.path.append("/data0/happykeyan/workspace/DataFlow-MM")
from dataflow.utils.storage import FileStorage
from dataflow.serving.local_model_vlm_serving import LocalModelVLMServing_vllm

# -------------------------- 请根据实际文件路径调整导入 --------------------------
# 替换为你的ImageRegionCaptionGenerate和ExistingBBoxDataGenConfig所在文件路径
from dataflow.operators.core_vision.generate.image_region_caption_generator import (  # 例如：from dataflow.operators.region_caption import ...
    ImageRegionCaptionGenerate,
    ExistingBBoxDataGenConfig,
)


class ImageRegionCaptionPipeline:
    """
    一键测试ImageRegionCaptionGenerate算子：
    输入含{"image": 图片路径, "bbox": 边界框列表}的jsonl → 生成带框可视化图 → VLM描述每个区域 → 输出结果到缓存
    """

    def __init__(
        self,
        vlm_model_path: str,
        *,
        hf_cache_dir: Optional[str] = None,
        download_dir: str = "./ckpt/models",
        device: str = "cuda",

        # 存储配置（输入/输出缓存）
        input_images_file: str = "/data0/happykeyan/workspace/DataFlow-MM/dataflow/example/image_to_text_pipeline/region_captions.jsonl",  # 含image和bbox的jsonl
        cache_path: str = "/data0/happykeyan/workspace/DataFlow-MM/dataflow/example/cache",  # 结果缓存路径
        file_name_prefix: str = "region_caption",
        cache_type: str = "jsonl",

        # 字段名配置（需与输入jsonl的key对应）
        image_key: str = "image",       # 输入中"图片路径"的key
        bbox_key: str = "bbox",         # 输入中"边界框列表"的key
        output_key: str = "mdvp_record",# 输出中"VLM描述结果"的key

        # 算子核心配置（对应ExistingBBoxDataGenConfig）
        max_boxes: int = 10,            # 单图最大边界框数量（与模型输入对齐）
        draw_visualization: bool = True,# 是否生成带数字标记的边界框可视化图
        output_jsonl_path: Optional[str] = None,  # 可选：单独指定最终结果输出路径

        # vLLM生成配置（控制VLM的输出风格）
        vllm_tensor_parallel_size: int = 1,  # GPU并行数（多卡时调整）
        vllm_temperature: float = 0.1,       # 随机性（0.0=确定性输出，0.1=低随机性）
        vllm_top_p: float = 0.9,            # 采样Top-P
        vllm_max_tokens: int = 1024,        # VLM最大输出 tokens（描述每个区域需足够长度）
    ):
        # ---------- 1. 初始化存储（读取输入jsonl + 缓存中间/最终结果） ----------
        self.storage = FileStorage(
            first_entry_file_name=input_images_file,
            cache_path=cache_path,
            file_name_prefix=file_name_prefix,
            cache_type=cache_type,
        )

        # ---------- 2. 初始化VLM服务（视觉语言模型，核心依赖） ----------
        self.vlm_serving = LocalModelVLMServing_vllm(
            hf_model_name_or_path=vlm_model_path,
            hf_cache_dir=hf_cache_dir,
            hf_local_dir=download_dir,
            vllm_tensor_parallel_size=vllm_tensor_parallel_size,
            vllm_temperature=vllm_temperature,
            vllm_top_p=vllm_top_p,
            vllm_max_tokens=vllm_max_tokens,
        )

        # ---------- 3. 初始化算子配置（与ExistingBBoxDataGenConfig对齐） ----------
        self.operator_cfg = ExistingBBoxDataGenConfig(
            max_boxes=max_boxes,
            input_jsonl_path='/data0/happykeyan/workspace/DataFlow-MM/dataflow/example/image_to_text_pipeline/region_captions.jsonl',  # 由storage驱动输入，无需单独指定
            output_jsonl_path=output_jsonl_path,
            draw_visualization=draw_visualization,
        )

        # ---------- 4. 初始化目标算子（ImageRegionCaptionGenerate） ----------
        # 【注意】修复原算子潜在bug：原代码__init__参数是vlm_serving，却赋值给self.llm_serving
        # 若未修复，需将下方self.vlm_serving改为self.llm_serving（否则运行会报属性错误）
        self.operator = ImageRegionCaptionGenerate(
            llm_serving=self.vlm_serving,  # 原代码此处可能误写为llm_serving，需检查修正
            config=self.operator_cfg,
        )

        # ---------- 5. 记录关键字段名（后续传给算子run方法） ----------
        self.image_key = image_key
        self.bbox_key = bbox_key
        self.output_key = output_key

    def forward(self):
        """
        一键运行完整流程：
        1. 从storage读取输入数据（图片路径 + 边界框）
        2. 算子处理：边界框归一化 → 生成可视化图 → 构造VLM Prompt → 调用VLM生成描述
        3. 结果写回storage缓存
        """
        self.operator.run(
            storage=self.storage.step(),  # Pipeline推进存储的"步骤"，确保数据流转
            image_key=self.image_key,     # 输入图片路径的key
            bbox_key=self.bbox_key,       # 输入边界框的key
            output_key=self.output_key,   # 输出VLM描述结果的key
        )
        print(f"[ImageRegionCaptionPipeline] 运行完成！")
        print(f"→ 图片缓存路径：{self.storage.cache_path}")
        if self.operator_cfg.output_jsonl_path:
            print(f"→ 最终结果输出路径：{self.operator_cfg.output_jsonl_path}")


# ---------------------------- 命令行入口（一键启动测试） ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="测试ImageRegionCaptionGenerate算子（视觉区域描述生成）")

    # 1. 模型配置（必填：VLM模型路径）
    parser.add_argument("--vlm_model_path",
                       default="/data0/happykeyan/Models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--hf_cache_dir", default="~/.cache/huggingface", 
                        help="HuggingFace模型缓存目录")
    parser.add_argument("--download_dir", default="./ckpt/models", 
                        help="模型下载目录（若本地无模型，自动下载到此处）")
    parser.add_argument("--device", choices=["cuda", "cpu", "mps"], default="cuda", 
                        help="运行设备（CPU速度极慢，建议用GPU）")

    # 2. 输入/存储配置（必填：含image和bbox的jsonl路径）
    parser.add_argument("--input_images_file",
                        default='/data0/happykeyan/workspace/DataFlow-MM/dataflow/example/image_to_text_pipeline/region_captions.jsonl')
    parser.add_argument("--cache_path", default="/data0/happykeyan/workspace/DataFlow-MM/dataflow/example/cache", 
                        help="结果缓存目录")
    parser.add_argument("--file_name_prefix", default="region_caption", 
                        help="缓存文件前缀")
    parser.add_argument("--output_jsonl_path",
                        default='/data0/happykeyan/workspace/DataFlow-MM/dataflow/example/image_to_text_pipeline/region_captions_results.jsonl')

    # 3. 字段名配置（与输入jsonl的key匹配）
    parser.add_argument("--image_key", default="image", 
                        help="输入jsonl中'图片路径'的key")
    parser.add_argument("--bbox_key", default="bbox", 
                        help="输入jsonl中'边界框列表'的key（格式：[[x0,y0,w,h], ...]）")
    parser.add_argument("--output_key", default="mdvp_record", 
                        help="输出结果中'VLM描述'的key")

    # 4. 算子核心配置
    parser.add_argument("--max_boxes", type=int, default=10, 
                        help="单图最大边界框数量（超过会截断，不足会补零）")
    parser.add_argument("--no_draw_visualization", action="store_false", dest="draw_visualization",
                        help="禁用边界框可视化图生成（默认生成）")

    # 5. vLLM生成配置（控制输出质量）
    parser.add_argument("--tp", type=int, default=1, 
                        help="vLLM张量并行数（多卡时设为GPU数量，如2）")
    parser.add_argument("--temperature", type=float, default=0.1, 
                        help="生成随机性（0.0=完全确定，0.5=中等随机）")
    parser.add_argument("--top_p", type=float, default=0.9, 
                        help="采样Top-P（0.9=兼顾多样性和准确性）")
    parser.add_argument("--max_tokens", type=int, default=1024, 
                        help="VLM最大输出tokens（描述N个区域需至少N*50 tokens）")

    args = parser.parse_args()

    # 创建Pipeline实例并运行
    pipe = ImageRegionCaptionPipeline(
        vlm_model_path=args.vlm_model_path,
        hf_cache_dir=args.hf_cache_dir,
        download_dir=args.download_dir,
        device=args.device,

        input_images_file=args.input_images_file,
        cache_path=args.cache_path,
        file_name_prefix=args.file_name_prefix,
        output_jsonl_path=args.output_jsonl_path,

        image_key=args.image_key,
        bbox_key=args.bbox_key,
        output_key=args.output_key,

        max_boxes=args.max_boxes,
        draw_visualization=args.draw_visualization,

        vllm_tensor_parallel_size=args.tp,
        vllm_temperature=args.temperature,
        vllm_top_p=args.top_p,
        vllm_max_tokens=args.max_tokens,
    )
    pipe.forward()