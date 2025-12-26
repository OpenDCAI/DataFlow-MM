from typing import List, Dict, Any, Union
import os
import json
import pandas as pd
import re
from typing import List, Dict, Any
from PIL import Image
from dataflow.core.Operator import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import FileStorage, DataFlowStorage
from dataflow.core import VLMServingABC
from dataflow.serving.local_model_llm_serving import LocalModelLLMServing_vllm
from qwen_vl_utils import process_vision_info
from dataflow.prompts.video import (MultiroleQAInitialQAGenerationPrompt, 
                                MultiroleQACallExpertAgentsPrompt, 
                                MultiroleQAProfile4ExpertAgents, 
                                MultiroleQAMasterAgentRevisionPrompt,
                                MultiroleQADIYFinalQASynthesisPrompt, 
                                MultiroleQAClassificationPrompt)

# -----------------------------------------------------------------------------
class Callvlm:
    def __init__(self, vlm_serving):
        self.llm_serving = vlm_serving
    def call(self, prompt_text: str, image_paths: List[str], system_prompt: str) -> str:
    
        image_inputs_list = []

        for path in image_paths:
            for p in path:
                raw_prompt = [
                {"role": "system", "content": system_prompt}
                ]
                user_content = []

                user_content.append({"type": "image", "image": p})
                
                user_content.append({"type": "text", "text": prompt_text})
            
                raw_prompt.append({
                    "role": "user", 
                    "content": user_content 
                })         

                image_inputs, _ = process_vision_info(raw_prompt)

                formatted_prompt = self.llm_serving.processor.apply_chat_template(
                    raw_prompt, tokenize=False, add_generation_prompt=True
                )
                image_inputs_list.append(image_inputs)

        outputs = self.llm_serving.generate_from_input(
            user_inputs=[formatted_prompt],
            image_inputs=image_inputs_list
        )
        
        if not outputs:
            return "" 
        
        final_output = outputs[0]
        
        if isinstance(final_output, list) and final_output:
            final_output = final_output[0]
            
        return str(final_output).strip()
# -----------------------------------------------------------------------------
@OPERATOR_REGISTRY.register()
class MultiroleVideoQAInitialGenerator(OperatorABC):
    def __init__(self, llm_serving: VLMServingABC):
        self.logger = get_logger()
        self.llm_serving = llm_serving
        self.initial_gen_prompt = MultiroleQAInitialQAGenerationPrompt()

    def _serialize_v_input(self, v_input: Dict[str, Any]) -> Dict[str, Any]:
        v_output = {
            "Meta": v_input.get("Meta", ""),
            "Clips": []
        }
        raw_clips = v_input.get("Clips", [])
        total_image_paths = []

        for clip in raw_clips:
            processed_clip = {
                "Audio_Text": clip.get("Audio_Text", ""),
                "Description": clip.get("Description", "")
            }

            image_paths = clip.get("Frames_Images", [])
            loaded_images = []
            
            if isinstance(image_paths, list):
                for path in image_paths:
                    try:
                        img = Image.open(path).convert("RGB")
                        loaded_images.append(img)
                    except Exception as e:
                        if hasattr(self, 'logger'):
                            self.logger.error(f"Failed to load image at {path}: {e}")
            
            processed_clip["Frames_Images"] = loaded_images
            
            v_output["Clips"].append(processed_clip)
            total_image_paths.append(image_paths)

        return v_output, total_image_paths

    def _process_single_video(self, v_input: Dict[str, Any]) -> Dict[str, Any]:

        v_content, all_image_paths = self._serialize_v_input(v_input)

        self.logger.info("Executing Step 1: Initial QA Generation")

        prompt_s1 = self.initial_gen_prompt.build_prompt(v_content)

        call_vlm = Callvlm(self.llm_serving)
        initial_qa_str = call_vlm.call(prompt_s1, all_image_paths, "")

        v_output = v_input.copy()
        v_output["QA"] = initial_qa_str
        return v_output

    def run(
        self,
        storage: DataFlowStorage,
        input_meta_key: str = "Meta", 
        input_clips_key: str = "Clips", 
        output_key: str = "QA"
    ):
        if output_key is None:
            raise ValueError("output_key must be provided.")

        data_list = storage.read(output_type="dict")
        df = pd.DataFrame(data_list)
        
        if not isinstance(df, pd.DataFrame):
            raise ValueError("storage.read must return a pandas DataFrame")

        if input_meta_key not in df.columns or input_clips_key not in df.columns:
             raise ValueError(f"Input columns {input_meta_key} or {input_clips_key} not found in DataFrame.")

        if output_key not in df.columns:
            df[output_key] = [None for _ in range(len(df))]

        self.logger.info(f"Start processing {len(df)} videos...")

        for idx, row in df.iterrows():
            current_output = row[output_key]
            if current_output is not None and isinstance(current_output, list) and len(current_output) > 0:
                continue

            meta_val = row.get(input_meta_key, "")
            clips_val = row.get(input_clips_key, [])

            if not isinstance(clips_val, list):
                self.logger.warning(f"Row {idx}: 'Clips' is not a list. Skipping.")
                df.at[idx, output_key] = [] 
                continue

            v_input = {
                "Meta": meta_val, 
                "Clips": clips_val 
            }

            try:
                processed_output = self._process_single_video(v_input)
                
                qa_result = processed_output.get("QA", [])
                
                df.at[idx, output_key] = qa_result
                
            except Exception as e:
                self.logger.error(f"Error processing row {idx}: {str(e)}")
                df.at[idx, output_key] = [] 

        return df

# -----------------------------------------------------------------------------
@OPERATOR_REGISTRY.register()
class MultiroleVideoQAMultiAgentGenerator(OperatorABC):
    def __init__(self, llm_serving: VLMServingABC, max_iterations: int):
        self.logger = get_logger()
        self.llm_serving = llm_serving
        self.max_iterations = max_iterations
        self.call_expert_prompt = MultiroleQACallExpertAgentsPrompt()
        self.expert_profile_prompt = MultiroleQAProfile4ExpertAgents()
        self.master_revision_prompt = MultiroleQAMasterAgentRevisionPrompt()


    def _serialize_v_input(self, v_input: Dict[str, Any]) -> Dict[str, Any]:
        v_output = {
            "Meta": v_input.get("Meta", ""),
            "Clips": []
        }
        raw_clips = v_input.get("Clips", [])
        total_image_paths = []

        for clip in raw_clips:
            processed_clip = {
                "Audio_Text": clip.get("Audio_Text", ""),
                "Description": clip.get("Description", "")
            }

            image_paths = clip.get("Frames_Images", [])
            loaded_images = []
            
            if isinstance(image_paths, list):
                for path in image_paths:
                    try:
                        img = Image.open(path).convert("RGB")
                        loaded_images.append(img)
                    except Exception as e:
                        if hasattr(self, 'logger'):
                            self.logger.error(f"Failed to load image at {path}: {e}")
            
            processed_clip["Frames_Images"] = loaded_images
            
            v_output["Clips"].append(processed_clip)
            total_image_paths.append(image_paths)

        return v_output, total_image_paths

    def experts(self, call_for_experts_response: str) -> List[Dict[str, str]]:
            """
            """
            experts_list: List[Dict[str, str]] = []

            json_matches = re.findall(r'\{.*?\}', call_for_experts_response, re.DOTALL)

            for json_str in json_matches:
                try:
                    expert_data: Dict[str, Any] = json.loads(json_str.strip())

                    role_raw = expert_data.get("Expert_Role", "")
                    subtask_raw = expert_data.get("Subtask", "")

                    role = role_raw.strip('<> ').strip()
                    subtask = subtask_raw.strip('<> ').strip()

                    if role and subtask:
                        experts_list.append({
                            "role": role,
                            "subtask": subtask
                        })

                except json.JSONDecodeError:
                    continue
                except AttributeError:
                    continue

            return experts_list

    def _process_single_video(self, v_input: Dict[str, Any], init_QA: str) -> Dict[str, Any]:

        v_content, all_image_paths = self._serialize_v_input(v_input)

        qa_history = [] 
        qa_history.append(init_QA)
        current_qa_pool_str = init_QA

        # ---------------- Loop: Expert Iteration ----------------
        iteration_count = 0
        expert_history = []
        while iteration_count < self.max_iterations:
            self.logger.info(f"Iteration {iteration_count + 1}: Check for Experts")

            prompt_s2 = self.call_expert_prompt.build_prompt(v_content, current_qa_pool_str, expert_history)

            call_vlm = Callvlm(self.llm_serving)
            call_for_experts_response = call_vlm.call(prompt_s2, all_image_paths, "")

            if isinstance(call_for_experts_response, str):
                if "NO_EXPERTS" in call_for_experts_response:
                    self.logger.info("Master Agent decided to end iteration.")
                    break

            experts_list = self.experts(call_for_experts_response)
            for exp in experts_list:
                expert_history.append(exp)

            for expert in experts_list:
                expert_profile = expert["role"]
                subtask = expert["subtask"]
                
                prompt_s3 = self.expert_profile_prompt.build_prompt(expert_profile, v_content, subtask)
                expert_qa_str = call_vlm.call(prompt_s3, all_image_paths, "")

                prompt_s4 = self.master_revision_prompt.build_prompt(v_content, expert_qa_str, current_qa_pool_str)
                revised_qa_str = call_vlm.call(prompt_s4, all_image_paths, "")
                
                current_qa_pool_str += f"\n{revised_qa_str}"
                qa_history.append(revised_qa_str)

            iteration_count += 1

        v_output = v_input.copy()
        v_output["QA"] = qa_history
        return v_output

    def run(
        self,
        df: pd.DataFrame,
        input_meta_key: str = "Meta", 
        input_clips_key: str = "Clips", 
        output_key: str = "QA"
    ):
        if output_key is None:
            raise ValueError("output_key must be provided.")
        
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df must be a pandas DataFrame")

        if input_meta_key not in df.columns or input_clips_key not in df.columns:
             raise ValueError(f"Input columns {input_meta_key} or {input_clips_key} not found in DataFrame.")

        self.logger.info(f"Start processing {len(df)} videos...")

        for idx, row in df.iterrows():
            current_output = row[output_key]

            meta_val = row.get(input_meta_key, "")
            clips_val = row.get(input_clips_key, [])
            init_QA = row.get(output_key, "")

            if not isinstance(clips_val, list):
                self.logger.warning(f"Row {idx}: 'Clips' is not a list. Skipping.")
                df.at[idx, output_key] = [] 
                continue

            v_input = {
                "Meta": meta_val, 
                "Clips": clips_val 
            }

            try:
                processed_output = self._process_single_video(v_input, init_QA)
                
                qa_result = processed_output.get("QA", [])
                
                df.at[idx, output_key] = qa_result
                
            except Exception as e:
                self.logger.error(f"Error processing row {idx}: {str(e)}")
                df.at[idx, output_key] = [] 

        return df

# -----------------------------------------------------------------------------
@OPERATOR_REGISTRY.register()
class MultiroleVideoQAFinalGenerator(OperatorABC):
    def __init__(self, llm_serving: VLMServingABC):
        self.logger = get_logger()
        self.llm_serving = llm_serving
        self.final_synthesis_prompt = MultiroleQADIYFinalQASynthesisPrompt()
        self.classification_prompt = MultiroleQAClassificationPrompt()


    def _serialize_v_input(self, v_input: Dict[str, Any]) -> Dict[str, Any]:
        v_output = {
            "Meta": v_input.get("Meta", ""),
            "Clips": []
        }
        raw_clips = v_input.get("Clips", [])
        total_image_paths = []

        for clip in raw_clips:
            processed_clip = {
                "Audio_Text": clip.get("Audio_Text", ""),
                "Description": clip.get("Description", "")
            }

            image_paths = clip.get("Frames_Images", [])
            loaded_images = []
            
            if isinstance(image_paths, list):
                for path in image_paths:
                    try:
                        img = Image.open(path).convert("RGB")
                        loaded_images.append(img)
                    except Exception as e:
                        if hasattr(self, 'logger'):
                            self.logger.error(f"Failed to load image at {path}: {e}")
            
            processed_clip["Frames_Images"] = loaded_images
            
            v_output["Clips"].append(processed_clip)
            total_image_paths.append(image_paths)

        return v_output, total_image_paths

    def extract(
            self,
            final_qa_json_str: str, 
            logger: Any = None
        ) -> Union[List[Dict[str, Any]], str]:

            JSON_ARRAY_REGEX = re.compile(r"(\[.*\])", re.DOTALL)

            match = JSON_ARRAY_REGEX.search(final_qa_json_str)
            
            if not match:
                if logger:
                    logger.warning("Failed to find JSON array structure (missing [ or ]).")
                return final_qa_json_str 

            json_block = match.group(1)
            
            qa_list: Union[List[Dict[str, Any]], str]
            try:
                qa_list = json.loads(json_block)
                
                if not isinstance(qa_list, list):
                    raise TypeError("Parsed result is not a list (e.g., VLM outputted a single object instead of an array).")

            except json.JSONDecodeError as e:
                if logger:
                    logger.warning(f"Failed to parse extracted JSON block (Decode Error: {e}).")
                qa_list = final_qa_json_str 
                
            except Exception as e:
                if logger:
                    logger.warning(f"Failed to parse extracted JSON block (General Error: {e}).")
                qa_list = final_qa_json_str 
                
            return qa_list

    
    def _process_single_video(self, v_input: Dict[str, Any], qa_history: List[str]) -> Dict[str, Any]:

        v_content, all_image_paths = self._serialize_v_input(v_input)
        
        # ---------------- Step 5: Final QA Synthesis ----------------
        self.logger.info("Executing Step 5: Final QA Synthesis")

        prompt_s5 = self.final_synthesis_prompt.build_prompt(qa_history)

        call_vlm = Callvlm(self.llm_serving)
        synthesized_qa_str = call_vlm.call(prompt_s5, all_image_paths, "")

        # ---------------- Step 6: Question Classification ----------------
        self.logger.info("Executing Step 6: Question Classification")
        prompt_s6 = self.classification_prompt.build_prompt(synthesized_qa_str)

        final_qa_json_str = call_vlm.call(prompt_s6, all_image_paths, "")

        # ---------------- Construct V_output ----------------
        qa_list = self.extract(final_qa_json_str, self.logger)
        
        if isinstance(qa_list, str):
            self.logger.warning("Failed to parse Final QA JSON, returning raw string.")
            qa_list = final_qa_json_str

        v_output = v_input.copy()
        v_output["QA"] = qa_list
        return v_output

    def run(
        self,
        storage: DataFlowStorage,
        df: pd.DataFrame,
        input_meta_key: str = "Meta", 
        input_clips_key: str = "Clips", 
        output_key: str = "QA"
    ):
        if output_key is None:
            raise ValueError("output_key must be provided.")
        
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df must be a pandas DataFrame")

        if input_meta_key not in df.columns or input_clips_key not in df.columns:
             raise ValueError(f"Input columns {input_meta_key} or {input_clips_key} not found in DataFrame.")

        self.logger.info(f"Start processing {len(df)} videos...")

        for idx, row in df.iterrows():
            current_output = row[output_key]

            meta_val = row.get(input_meta_key, "")
            clips_val = row.get(input_clips_key, [])
            qa_history = row.get(output_key, "")

            if not isinstance(clips_val, list):
                self.logger.warning(f"Row {idx}: 'Clips' is not a list. Skipping.")
                df.at[idx, output_key] = [] 
                continue

            v_input = {
                "Meta": meta_val, 
                "Clips": clips_val 
            }

            try:
                processed_output = self._process_single_video(v_input, qa_history)
                
                qa_result = processed_output.get("QA", [])
                
                df.at[idx, output_key] = qa_result
                
            except Exception as e:
                self.logger.error(f"Error processing row {idx}: {str(e)}")
                df.at[idx, output_key] = [] 

        output_file = storage.write(df)
        self.logger.info(f"All processing done. Results saved to {output_file}")

        return [output_key]


    

# -----------------------------------------------------------------------------
