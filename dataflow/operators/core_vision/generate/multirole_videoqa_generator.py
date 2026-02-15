import os
import json
import pandas as pd
import re
from typing import List, Dict, Any, Union

from dataflow.core.Operator import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import VLMServingABC

from dataflow.serving.api_vlm_serving_openai import APIVLMServing_openai

# 引入提示词模板
from dataflow.prompts.video import (
    MultiroleQAInitialQAGenerationPrompt, 
    MultiroleQACallExpertAgentsPrompt, 
    MultiroleQAProfile4ExpertAgents, 
    MultiroleQAMasterAgentRevisionPrompt,
    MultiroleQADIYFinalQASynthesisPrompt, 
    MultiroleQAClassificationPrompt
)

# -----------------------------------------------------------------------------
# 辅助函数与基类 (消除重复代码，统一调用规范)
# -----------------------------------------------------------------------------

def is_api_serving(serving):
    return isinstance(serving, APIVLMServing_openai)

class MultiroleVideoQABase(OperatorABC):
    """
    多智能体视频问答算子基类，提供统一的视频信息提取和模型调用接口。
    """
    def __init__(self, llm_serving: VLMServingABC):
        self.logger = get_logger()
        self.llm_serving = llm_serving

    def _extract_video_info(self, v_input: Dict[str, Any]) -> tuple[Dict[str, Any], List[str]]:
        """
        提取视频 Meta 和 Clips 文本信息，并将所有有效图片路径展平为一个 List，
        移除了极度消耗内存的 PIL.Image 预加载逻辑。
        """
        v_content = {
            "Meta": v_input.get("Meta", ""),
            "Clips": []
        }
        flat_image_paths = []

        for clip in v_input.get("Clips", []):
            processed_clip = {
                "Audio_Text": clip.get("Audio_Text", ""),
                "Description": clip.get("Description", "")
            }

            paths = clip.get("Frames_Images", [])
            if isinstance(paths, str):
                paths = [paths]
            
            # 过滤并收集有效的图片路径
            valid_paths = [p for p in paths if isinstance(p, str) and p.strip()]
            flat_image_paths.extend(valid_paths)
            
            processed_clip["Frames_Images"] = valid_paths
            v_content["Clips"].append(processed_clip)

        return v_content, flat_image_paths

    def _generate_answer(self, prompt_text: str, image_paths: List[str]) -> str:
        """
        统一的模型调用接口。自动处理 API/Local 模式和 <image> 占位符。
        代替了原来臃肿且有逻辑缺陷的 Callvlm 类。
        """
        use_api_mode = is_api_serving(self.llm_serving)
        
        if use_api_mode:
            content = prompt_text
        else:
            img_tokens = "<image>" * len(image_paths)
            content = f"{img_tokens}\n{prompt_text}" if img_tokens else prompt_text

        conversation = [{"role": "user", "content": content}]

        outputs = self.llm_serving.generate_from_input_messages(
            conversations=[conversation],
            image_list=[image_paths] if image_paths else None,
            system_prompt=""  # 保持与原逻辑一致，不使用系统提示词
        )
        
        if outputs and len(outputs) > 0:
            return str(outputs[0]).strip()
        return ""


# -----------------------------------------------------------------------------
# Operator 1: Initial QA Generator (阶段一：初始问答生成)
# -----------------------------------------------------------------------------

@OPERATOR_REGISTRY.register()
class MultiroleVideoQAInitialGenerator(MultiroleVideoQABase):
    def __init__(self, llm_serving: VLMServingABC):
        super().__init__(llm_serving)
        self.initial_gen_prompt = MultiroleQAInitialQAGenerationPrompt()

    def run(
        self,
        storage: DataFlowStorage,
        input_meta_key: str = "Meta", 
        input_clips_key: str = "Clips", 
        output_key: str = "QA"
    ):
        df: pd.DataFrame = storage.read("dataframe")
        
        if input_meta_key not in df.columns or input_clips_key not in df.columns:
             raise ValueError(f"Columns '{input_meta_key}' or '{input_clips_key}' not found.")

        if output_key not in df.columns:
            df[output_key] = None

        self.logger.info(f"[InitialGenerator] Start processing {len(df)} videos...")

        for idx, row in df.iterrows():
            # 跳过已处理的数据
            if row.get(output_key) and isinstance(row.get(output_key), list) and len(row[output_key]) > 0:
                continue

            clips_val = row.get(input_clips_key, [])
            if not isinstance(clips_val, list):
                self.logger.warning(f"Row {idx}: 'Clips' is not a list. Skipping.")
                df.at[idx, output_key] = [] 
                continue

            v_input = {"Meta": row.get(input_meta_key, ""), "Clips": clips_val}

            try:
                v_content, all_image_paths = self._extract_video_info(v_input)
                prompt_s1 = self.initial_gen_prompt.build_prompt(v_content)
                
                initial_qa_str = self._generate_answer(prompt_s1, all_image_paths)
                df.at[idx, output_key] = initial_qa_str

            except Exception as e:
                self.logger.error(f"Error processing row {idx}: {str(e)}")
                df.at[idx, output_key] = [] 

        storage.write(df)
        return [output_key]


# -----------------------------------------------------------------------------
# Operator 2: Multi Agent Generator (阶段二：多智能体专家迭代)
# -----------------------------------------------------------------------------

@OPERATOR_REGISTRY.register()
class MultiroleVideoQAMultiAgentGenerator(MultiroleVideoQABase):
    def __init__(self, llm_serving: VLMServingABC, max_iterations: int = 3):
        super().__init__(llm_serving)
        self.max_iterations = max_iterations
        self.call_expert_prompt = MultiroleQACallExpertAgentsPrompt()
        self.expert_profile_prompt = MultiroleQAProfile4ExpertAgents()
        self.master_revision_prompt = MultiroleQAMasterAgentRevisionPrompt()

    def experts(self, call_for_experts_response: str) -> List[Dict[str, str]]:
        experts_list: List[Dict[str, str]] = []
        json_matches = re.findall(r'\{.*?\}', call_for_experts_response, re.DOTALL)

        for json_str in json_matches:
            try:
                expert_data = json.loads(json_str.strip())
                role = expert_data.get("Expert_Role", "").strip('<> ').strip()
                subtask = expert_data.get("Subtask", "").strip('<> ').strip()

                if role and subtask:
                    experts_list.append({"role": role, "subtask": subtask})
            except (json.JSONDecodeError, AttributeError):
                continue

        return experts_list

    def run(
        self,
        storage: DataFlowStorage,
        input_meta_key: str = "Meta", 
        input_clips_key: str = "Clips", 
        output_key: str = "QA"
    ):
        df: pd.DataFrame = storage.read("dataframe")
        self.logger.info(f"[MultiAgentGenerator] Start processing {len(df)} videos...")

        for idx, row in df.iterrows():
            clips_val = row.get(input_clips_key, [])
            init_qa = row.get(output_key, "")

            if not isinstance(clips_val, list):
                continue

            v_input = {"Meta": row.get(input_meta_key, ""), "Clips": clips_val}

            try:
                v_content, all_image_paths = self._extract_video_info(v_input)

                qa_history = [init_qa]
                current_qa_pool_str = str(init_qa)
                expert_history = []

                for i in range(self.max_iterations):
                    self.logger.info(f"Row {idx} - Iteration {i + 1}: Check for Experts")
                    prompt_s2 = self.call_expert_prompt.build_prompt(v_content, current_qa_pool_str, expert_history)
                    call_for_experts_response = self._generate_answer(prompt_s2, all_image_paths)

                    if "NO_EXPERTS" in call_for_experts_response:
                        self.logger.info("Master Agent decided to end iteration.")
                        break

                    experts_list = self.experts(call_for_experts_response)
                    expert_history.extend(experts_list)

                    for expert in experts_list:
                        prompt_s3 = self.expert_profile_prompt.build_prompt(expert["role"], v_content, expert["subtask"])
                        expert_qa_str = self._generate_answer(prompt_s3, all_image_paths)

                        prompt_s4 = self.master_revision_prompt.build_prompt(v_content, expert_qa_str, current_qa_pool_str)
                        revised_qa_str = self._generate_answer(prompt_s4, all_image_paths)
                        
                        current_qa_pool_str += f"\n{revised_qa_str}"
                        qa_history.append(revised_qa_str)

                df.at[idx, output_key] = qa_history

            except Exception as e:
                self.logger.error(f"Error processing row {idx}: {str(e)}")

        storage.write(df)
        return [output_key]


# -----------------------------------------------------------------------------
# Operator 3: Final Generator (阶段三：最终合成与分类)
# -----------------------------------------------------------------------------

@OPERATOR_REGISTRY.register()
class MultiroleVideoQAFinalGenerator(MultiroleVideoQABase):
    def __init__(self, llm_serving: VLMServingABC):
        super().__init__(llm_serving)
        self.final_synthesis_prompt = MultiroleQADIYFinalQASynthesisPrompt()
        self.classification_prompt = MultiroleQAClassificationPrompt()

    def extract(self, final_qa_json_str: str) -> Union[List[Dict[str, Any]], str]:
        JSON_ARRAY_REGEX = re.compile(r"(\[.*\])", re.DOTALL)
        match = JSON_ARRAY_REGEX.search(final_qa_json_str)
        
        if not match:
            self.logger.warning("Failed to find JSON array structure.")
            return final_qa_json_str 

        try:
            qa_list = json.loads(match.group(1))
            if not isinstance(qa_list, list):
                raise TypeError("Parsed result is not a list.")
            return qa_list
        except Exception as e:
            self.logger.warning(f"Failed to parse extracted JSON block: {e}")
            return final_qa_json_str 

    def run(
        self,
        storage: DataFlowStorage,
        input_meta_key: str = "Meta", 
        input_clips_key: str = "Clips", 
        output_key: str = "QA"
    ):
        df: pd.DataFrame = storage.read("dataframe")
        self.logger.info(f"[FinalGenerator] Start processing {len(df)} videos...")

        for idx, row in df.iterrows():
            clips_val = row.get(input_clips_key, [])
            qa_history = row.get(output_key, [])

            if not isinstance(clips_val, list):
                continue

            v_input = {"Meta": row.get(input_meta_key, ""), "Clips": clips_val}

            try:
                v_content, all_image_paths = self._extract_video_info(v_input)

                # Step 5: Final QA Synthesis
                self.logger.info(f"Row {idx} - Step 5: Final QA Synthesis")
                prompt_s5 = self.final_synthesis_prompt.build_prompt(qa_history)
                synthesized_qa_str = self._generate_answer(prompt_s5, all_image_paths)

                # Step 6: Question Classification
                self.logger.info(f"Row {idx} - Step 6: Question Classification")
                prompt_s6 = self.classification_prompt.build_prompt(synthesized_qa_str)
                final_qa_json_str = self._generate_answer(prompt_s6, all_image_paths)

                # Extract and Save
                qa_list = self.extract(final_qa_json_str)
                df.at[idx, output_key] = qa_list

            except Exception as e:
                self.logger.error(f"Error processing row {idx}: {str(e)}")
                df.at[idx, output_key] = [] 

        output_file = storage.write(df)
        self.logger.info(f"All processing done. Results saved to {output_file}")

        return [output_key]
