import re
import os
import json
import pandas as pd
from typing import List, Dict, Tuple, Optional
import torch
import gc
from PIL import Image

from qwen_vl_utils import process_vision_info

from dataflow.core.Operator import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import LLMServingABC
from dataflow.prompts.image import ImageGCoTPrompt


@OPERATOR_REGISTRY.register()
class ImageGCoTGenerate(OperatorABC):
    """
    Grounded Chain-of-Thought Generator for Visual Question Answering.
    
    This operator generates reasoning chains with spatial grounding by:
    1. Using Qwen2.5-VL to generate Chain-of-Thought and extract keywords
    2. Using Ovis2.5 to detect bounding boxes for keywords
    3. Injecting bounding boxes into the reasoning chain to create GCoT
    """
    
    def __init__(
        self,
        llm_serving: Optional[LLMServingABC] = None,
        model_path: str = "AIDC-AI/Ovis2.5-9B",
        device: str = "cuda",
        max_new_tokens: int = 512,
    ):
        """
        Initialize GCoT Generator.
        
        Args:
            llm_serving: Qwen model service for CoT generation
            model_path: Ovis model path for bbox detection
            device: Device to run Ovis model on
            max_new_tokens: Maximum tokens for generation
        """
        self.logger = get_logger()
        
        # Qwen serving (for CoT generation)
        self.llm_serving = llm_serving
        self.prompt_generator = ImageGCoTPrompt()
        
        # Ovis model config
        self.model_path = model_path
        self.device = device
        self.max_new_tokens = max_new_tokens
        
        # Ovis model (lazy loading)
        self.ovis_model = None
    
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "该算子用于生成带视觉定位信息的思维链(Grounded Chain-of-Thought)。\n\n"
                "输入参数：\n"
                "- question: 问题文本\n"
                "- answer: 答案文本\n"
                "- image: 图像路径\n\n"
                "输出参数：\n"
                "- cot: 原始思维链\n"
                "- gcot: 带定位信息的思维链\n"
                "- bboxes: 关键词到边界框的映射\n\n"
                "功能：\n"
                "1. 使用 Qwen2.5-VL 生成思维链并提取关键词\n"
                "2. 使用 Ovis2.5 对关键词进行视觉定位\n"
                "3. 将定位信息注入到思维链中生成 GCoT"
            )
        elif lang == "en":
            return (
                "This operator generates Grounded Chain-of-Thought (GCoT) with visual grounding.\n\n"
                "Input Parameters:\n"
                "- question: Question text\n"
                "- answer: Answer text\n"
                "- image: Image path\n\n"
                "Output Parameters:\n"
                "- cot: Original Chain-of-Thought\n"
                "- gcot: Grounded Chain-of-Thought with bounding boxes\n"
                "- bboxes: Mapping from keywords to bounding boxes\n\n"
                "Functionality:\n"
                "1. Use Qwen2.5-VL to generate CoT and extract keywords\n"
                "2. Use Ovis2.5 to detect bounding boxes for keywords\n"
                "3. Inject bounding boxes into CoT to generate GCoT"
            )
        else:
            return "Generate Grounded Chain-of-Thought with visual grounding"
    
    def _validate_dataframe(self, dataframe: pd.DataFrame):
        """Validate input dataframe"""
        required_keys = [self.input_question_key, self.input_answer_key, self.input_image_key]
        forbidden_keys = [self.output_key, 'cot', 'bboxes']
        
        missing = [k for k in required_keys if k not in dataframe.columns]
        conflict = [k for k in forbidden_keys if k in dataframe.columns]
        
        if missing:
            raise ValueError(f"Missing required column(s): {missing}")
        if conflict:
            raise ValueError(f"Column(s) already exist and would be overwritten: {conflict}")
    
    # ==================== Ovis Model ====================
    
    def _load_ovis_model(self):
        """Load Ovis2.5 model for bbox detection"""
        if self.ovis_model is not None:
            return
        
        from transformers import AutoModelForCausalLM
        
        self.ovis_model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        ).to(self.device)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    def _generate_with_ovis(
        self,
        prompts: List[str],
        image_paths: List[str]
    ) -> List[str]:
        """Generate with Ovis model"""
        if self.ovis_model is None:
            self._load_ovis_model()
        
        outputs = []
        
        for prompt, image_path in zip(prompts, image_paths):
            try:
                image = Image.open(image_path)
                
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }]
                
                input_ids, pixel_values, grid_thws = self.ovis_model.preprocess_inputs(
                    messages=messages,
                    add_generation_prompt=True,
                    enable_thinking=False
                )
                
                with torch.inference_mode():
                    output_ids = self.ovis_model.generate(
                        inputs=input_ids.to(self.device),
                        pixel_values=pixel_values.to(self.device) if pixel_values is not None else None,
                        grid_thws=grid_thws.to(self.device) if grid_thws is not None else None,
                        enable_thinking=False,
                        max_new_tokens=self.max_new_tokens
                    )
                
                output_text = self.ovis_model.text_tokenizer.decode(
                    output_ids[0],
                    skip_special_tokens=True
                )
                
                outputs.append(output_text.strip())
                
            except Exception as e:
                self.logger.error(f"Ovis generation failed: {e}")
                outputs.append("")
        
        return outputs
    
    @staticmethod
    def _parse_bboxes(text: str) -> List[List[float]]:
        """Parse bounding boxes from Ovis output"""
        if not text:
            return []
        
        bboxes = []
        pattern = r'<box>\(([0-9.]+),([0-9.]+)\),\(([0-9.]+),([0-9.]+)\)</box>'
        
        for match in re.finditer(pattern, text):
            x1, y1, x2, y2 = map(float, match.groups())
            
            if all(0 <= c <= 1 for c in [x1, y1, x2, y2]) and x1 < x2 and y1 < y2:
                bboxes.append([x1, y1, x2, y2])
        
        return bboxes
    
    # ==================== CoT Generation ====================
    
    def _prepare_cot_inputs(self, dataframe: pd.DataFrame) -> Tuple[List[str], List[List]]:
        """Prepare inputs for CoT generation"""
        prompt_list = []
        image_inputs_list = []
        
        for idx, row in dataframe.iterrows():
            image_path = row[self.input_image_key]
            question = row[self.input_question_key]
            answer = row[self.input_answer_key]
            
            cot_prompt = self.prompt_generator.build_prompt(
                "cot",
                question=question,
                answer=answer,
            )                      
            raw_prompt = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": cot_prompt},
                ],
            }]
            
            try:
                image_inputs, _ = process_vision_info(raw_prompt)
                prompt = self.llm_serving.processor.apply_chat_template(
                    raw_prompt,
                    tokenize=False,
                    add_generation_prompt=True
                )
                prompt_list.append(prompt)
                image_inputs_list.append([image_inputs])
            except Exception as e:
                self.logger.warning(f"Failed to process sample {idx}: {e}")
        
        return prompt_list, image_inputs_list
    
    def _parse_cot_and_keywords(self, output: str) -> Tuple[str, List[str]]:
        """Parse CoT and keywords from output"""
        if not output:
            return "", []
        
        lines = output.split('\n')
        cot_lines = []
        keywords = []
        
        for line in lines:
            if line.strip().lower().startswith('keywords:'):
                keyword_str = line.split(':', 1)[-1].strip()
                keywords = [kw.strip().strip('.,;:!?"\'') 
                           for kw in keyword_str.replace(';', ',').split(',')
                           if kw.strip()]
            else:
                cot_lines.append(line)
        
        cot = '\n'.join(cot_lines).strip()
        
        return cot, keywords
    
    def _merge_adjacent_keywords(self, keywords: List[str], cot_text: str) -> List[str]:
        """Merge keywords that appear adjacent in CoT"""
        if len(keywords) <= 1:
            return keywords
        
        cot_lower = cot_text.lower()
        merged = []
        skip_indices = set()
        
        for i in range(len(keywords)):
            if i in skip_indices:
                continue
            
            best_match = keywords[i]
            best_indices = [i]
            
            # Try forward combinations
            for j in range(i + 1, min(i + 4, len(keywords))):
                if j in skip_indices:
                    break
                combined = ' '.join(keywords[i:j+1])
                if combined.lower() in cot_lower:
                    best_match = combined
                    best_indices = list(range(i, j+1))
                else:
                    break
            
            merged.append(best_match)
            skip_indices.update(best_indices)
        
        return merged
    
    def _generate_cot(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Generate CoT and extract keywords"""
        if self.llm_serving is None:
            raise RuntimeError("llm_serving is required for CoT generation")
        
        self.logger.info(f"Generating CoT for {len(dataframe)} samples")
        
        prompt_list, image_inputs_list = self._prepare_cot_inputs(dataframe)
        
        outputs = self.llm_serving.generate_from_input(
            user_inputs=prompt_list,
            image_inputs=image_inputs_list
        )
        
        cot_list = []
        keywords_list = []
        
        for output in outputs:
            cot, keywords = self._parse_cot_and_keywords(output)
            
            if keywords:
                keywords = self._merge_adjacent_keywords(keywords, cot)
            
            cot_list.append(cot)
            keywords_list.append(keywords)
        
        dataframe['cot'] = cot_list
        dataframe['keywords'] = keywords_list
        
        return dataframe
    
    # ==================== Bbox Detection ====================
    
    def _detect_bboxes(self, sub_questions: List[Dict]) -> List[Dict]:
        """Detect bounding boxes for keywords"""
        if not sub_questions:
            return []
        
        prompts = [
            self.prompt_generator.build_prompt(
                "bbox",
                keyword=sq["keyword"],
            )
            for sq in sub_questions
        ]
        
        image_paths = [sq['image'] for sq in sub_questions]
        outputs = self._generate_with_ovis(prompts, image_paths)
        
        results = []
        for sq, output in zip(sub_questions, outputs):
            if 'not found' in output.lower():
                bboxes = []
                status = 'not_found'
            else:
                bboxes = self._parse_bboxes(output)
                status = 'found' if bboxes else 'parse_failed'
            
            results.append({
                **sq,
                'bboxes': bboxes,
                'status': status
            })
        
        return results
    
    def _filter_bboxes(
        self,
        bbox_results: List[Dict],
        max_boxes_per_keyword: int = 3
    ) -> Dict[str, List[str]]:
        """Filter and format bounding boxes"""
        bbox_map = {}
        
        for result in bbox_results:
            keyword = result['keyword']
            bboxes = result.get('bboxes', [])
            status = result.get('status', 'unknown')
            
            # Skip if not found or too many boxes
            if status == 'not_found' or len(bboxes) > max_boxes_per_keyword:
                continue
            
            if bboxes:
                bbox_strs = [
                    f"[{b[0]:.3f}, {b[1]:.3f}, {b[2]:.3f}, {b[3]:.3f}]"
                    for b in bboxes
                ]
                bbox_map[keyword] = bbox_strs
        
        return bbox_map
    
    # ==================== GCoT Generation ====================
    
    @staticmethod
    def _inject_bboxes(cot_text: str, bbox_map: Dict[str, List[str]]) -> str:
        """Inject bounding boxes into CoT"""
        # Sort keywords by length (longest first)
        sorted_keywords = sorted(bbox_map.keys(), key=lambda x: len(x.split()), reverse=True)
        
        result_text = cot_text
        replaced = set()
        
        for keyword in sorted_keywords:
            if keyword in replaced:
                continue
            
            # Find first occurrence before "Answer:"
            answer_pos = result_text.find('Answer:')
            search_text = result_text[:answer_pos] if answer_pos != -1 else result_text
            
            pos = search_text.lower().find(keyword.lower())
            if pos == -1:
                continue
            
            # Build replacement
            bbox_strs = bbox_map[keyword]
            if len(bbox_strs) == 1:
                replacement = f"{keyword} {bbox_strs[0]}"
            else:
                replacement = f"{keyword} " + "".join(bbox_strs)
            
            # Handle punctuation
            end_pos = pos + len(keyword)
            if end_pos < len(result_text) and result_text[end_pos] in '.,;:!?"\'':
                replacement += result_text[end_pos]
                end_pos += 1
            
            result_text = result_text[:pos] + replacement + result_text[end_pos:]
            replaced.add(keyword)
        
        return result_text
    
    # ==================== Main Pipeline ====================
    
    def run(
        self,
        storage: DataFlowStorage,
        input_question_key: str = "question",
        input_answer_key: str = "answer",
        input_image_key: str = "image",
        output_key: str = "gcot",
        save_intermediate: bool = False,
        qwen_unload_callback = None
    ):
        """
        Run GCoT generation pipeline.
        
        Args:
            storage: DataFlow storage
            input_question_key: Field name for questions
            input_answer_key: Field name for answers
            input_image_key: Field name for image paths
            output_key: Field name for GCoT output
            save_intermediate: Whether to save intermediate results
            qwen_unload_callback: Callback to unload Qwen model
        
        Returns:
            List of output field names
        """
        self.input_question_key = input_question_key
        self.input_answer_key = input_answer_key
        self.input_image_key = input_image_key
        self.output_key = output_key
        
        dataframe = storage.read("dataframe")
        self._validate_dataframe(dataframe)
        
        # Part 1: Generate CoT with Qwen
        dataframe = self._generate_cot(dataframe)
        
        # Generate sub-questions for bbox detection
        all_sub_questions = []
        row_id_map = {}
        
        for idx, row in dataframe.iterrows():
            row_id = row.get('id', idx)
            keywords = row.get('keywords', [])
            
            if keywords:
                start_idx = len(all_sub_questions)
                for keyword in keywords:
                    all_sub_questions.append({
                        'id': row_id,
                        'image': row[self.input_image_key],
                        'keyword': keyword
                    })
                row_id_map[row_id] = (start_idx, len(all_sub_questions))
            else:
                row_id_map[row_id] = (0, 0)
        
        # Unload Qwen model
        if qwen_unload_callback:
            qwen_unload_callback()
        
        # Part 2: Detect bboxes with Ovis
        bbox_results = self._detect_bboxes(all_sub_questions)
        
        # Part 3: Generate GCoT
        gcot_list = []
        bbox_list = []
        
        for idx, row in dataframe.iterrows():
            row_id = row.get('id', idx)
            cot_text = row['cot']
            
            start_idx, end_idx = row_id_map.get(row_id, (0, 0))
            
            if start_idx == end_idx:
                gcot_list.append(cot_text)
                bbox_list.append({})
            else:
                row_results = bbox_results[start_idx:end_idx]
                bbox_map = self._filter_bboxes(row_results)
                
                if bbox_map:
                    gcot_text = self._inject_bboxes(cot_text, bbox_map)
                else:
                    gcot_text = cot_text
                
                gcot_list.append(gcot_text)
                bbox_list.append(bbox_map)
        
        dataframe[self.output_key] = gcot_list
        dataframe['bboxes'] = bbox_list
        
        # Save results
        output_file = storage.write(dataframe)
        self.logger.info(f"Results saved to {output_file}")
        
        return ['cot', self.output_key, 'bboxes']