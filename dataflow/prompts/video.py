'''
A collection of prompts for the video operators.
'''

from typing import List, Dict, Any


class VideoCaptionGeneratorPrompt:
    '''
    The prompt for the video quality evaluator.
    '''
    def __init__(self):
        pass

    def build_prompt(self,) -> str:
        """
        Generate system prompt for video quality evaluation.
        """
        prompt = (
            "Please describe the video in detail."
        )
        return prompt
    

class VideoQAGeneratorPrompt:
    def __init__(self):
        pass

    def build_prompt(self, caption: str) -> str:
        return (
            "### Task:\n"
            "Given a detailed description that summarizes the content of a video, generate question-answer pairs "
            "based on the description to help humans better understand the video.\n"
            "The question-answer pairs should be faithful to the content of the video description and developed "
            "from different dimensions to promote comprehensive understanding of the video.\n\n"
            "#### Guidelines For Question-Answer Pairs Generation:\n"
            "- Read the provided video description carefully. Pay attention to the scene, main characters, "
            "their behaviors, and the development of events.\n"
            "- Generate appropriate question-answer pairs based on the description. The pairs should cover "
            "as many question dimensions as possible and not deviate from the content.\n"
            "- Generate 5 to 10 question-answer pairs across different dimensions.\n\n"
            "### Output Format:\n"
            "1. Your output should be formatted as a JSON list.\n"
            "2. Only provide the Python dictionary string.\n"
            "Your response should look like:\n"
            "[\n"
            "  {\"Dimension\": <dimension-1>, \"Question\": <question-1>, \"Answer\": <answer-1>},\n"
            "  {\"Dimension\": <dimension-2>, \"Question\": <question-2>, \"Answer\": <answer-2>},\n"
            "  ...\n"
            "]\n\n"
            "Please generate question-answer pairs for the following video description:\n"
            f"Description: {caption}"
        )



class VideoCOTQAGeneratorPrompt:
    '''
    The prompt for the video Chain-of-Thought QA generator.
    '''
    def __init__(self):
        # Type-specific answer format templates
        self.type_template = {
            "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
            "numerical": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
            "OCR": " Please transcribe text from the image/video clearly and provide your text answer within the <answer> </answer> tags.",
            "free-form": " Please provide your text answer within the <answer> </answer> tags.",
            "regression": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags."
        }
    
    def build_prompt(self, **kwargs) -> str:
        """
        Generate prompt for CoT QA generation with question parameter.
        """
        question = kwargs.get('Question', '')
        prompt = (
            f"{question}\n"
            "Please think about this question as if you were a human pondering deeply. "
            "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions "
            "It's encouraged to include self-reflection or verification in the reasoning process. "
            "Provide your detailed reasoning between the <think> and </think> tags, and then give your final answer between the <answer> and </answer> tags."
        )
        return prompt


class DiyVideoPrompt:
    '''
    The prompt for custom code operations.
    '''
    def __init__(self, prompt_template: str):
        self.prompt_template = prompt_template
    
    def build_prompt(self, **kwargs) -> str:
        """
        Generate prompt using custom template.
        """
        try:
            return self.prompt_template.format(**kwargs)
        except Exception as e:
            # If formatting fails, return the original template
            return self.prompt_template


# ------------------------------------------------------------------------------------------


class MultiroleQAInitialQAGenerationPrompt:
    '''
    The prompt for the Master Agent to generate the initial set of QA pairs
    based on the full video advertisement synopsis.
    '''
    def __init__(self):
        pass

    def build_prompt(
        self, 
        processed_v_data: Dict[str, Any]
    ) -> str:
        """
        Generate the detailed initial QA generation prompt using the processed
        video data structure (output of _serialize_v_input).

        Args:
            processed_v_data (Dict[str, Any]): 经过处理后的视频数据字典，
                                                包含 "Meta" (str) 和 "Clips" (List[Dict])。

        Returns:
            str: 格式化后的提示词。
        """
        
        clips = processed_v_data.get("Clips", [])
        meta_str = processed_v_data.get("Meta", "No metadata provided.")
        
        scene_nums = len(clips)
        
        # 提取并格式化 Clips 的详细信息
        clip_details_parts = []
        full_voiceover_parts = []
        
        for idx, clip in enumerate(clips):
            # 提取文本信息
            audio_text = clip.get('Audio_Text', 'N/A')
            description = clip.get('Description', 'N/A')
            
            # 统计图片数量 (Frames_Images 此时是 Image 对象列表)
            frame_count = len(clip.get('Frames_Images', []))
            
            clip_details_parts.append(f"\nClip {idx+1}:")
            # 按照您的要求，循环输出 Description, Frames, Audio/Text
            # {scene_descriptions} 对应 Description
            # {scene_frames_image} 对应 Frames: [...]
            # {voiceover} 对应 Audio/Text
            clip_details_parts.append(f"  Description: {description}")
            clip_details_parts.append(f"  Frames: [Contains {frame_count} Image Frames]")
            clip_details_parts.append(f"  Audio/Text: {audio_text}")
            
            # 收集完整的语音内容
            if audio_text and audio_text != 'N/A':
                full_voiceover_parts.append(audio_text)

        scene_details_block = "\n".join(clip_details_parts)
        full_voiceover = " ".join(full_voiceover_parts).strip()
        
        # --- 提示词模板 ---
        prompt = f"""
            You are tasked with generating questions from advertisement descriptions. Use the description alone to craft the questions and avoid making assumptions. The correct answer should blend seamlessly with the wrong ones in terms of length and complexity. Do not use direct quotes, and keep terminology simple. Questions should relate only to the content of the advertisement and avoid any external or behind-the-scenes details.

            This advertisement contains the following {scene_nums} scenes:

            {scene_details_block}

            Voiceover: {full_voiceover}

            The advertisement synopsis is based on the following general metadata:
            Full Video Meta: {meta_str}

            Create No more than ten questions based on this synopsis. Use these aspects as a reference when asking questions:
            1. Theme and core message (compulsory)
            2. Conveyance method for Theme, Brand, and Product features (if applicable)
            3. Specific visual elements (object, person, scene, event, etc.) and their relation to the theme (No more than three questions)
            4. Specific detail’s connection to the overall theme (compulsory)
            5. Target audience characteristics (if applicable)
            6. Emotional impact and tactics used
            7. Storyline and narration (if applicable)
            8. Metaphors or humorous techniques (if present)
            9. Logical arguments, factual claims, or expert opinions (if present)
            10. Characters and their relevance to the theme and audience (if present)
            11. Creativity and the overall impression of the ad. (if applicable)

            For each question, provide only one correct answer. The answers must be unique, and unbiased. Print each correct answer exactly as 'Correct answer: [full answer]'.
            """
        return prompt.strip()

class MultiroleQACallExpertAgentsPrompt:
    '''
    The prompt for the Master Agent to determine the need for new expert agents,
    describe their role, and assign a specific sub-task based on the current
    QA pool and video context.
    '''
    def __init__(self):
        pass

    def build_prompt(
        self, 
        initial_prompt: str, 
        current_annotation: str, 
        expert_history: List[Dict[str, str]]
    ) -> str:
        """
        Generates the prompt for the Master Agent to call for new expert agents.

        Args:
            initial_prompt (str): 初始 QA 生成时使用的完整上下文（包括视频信息、Meta等）。
            current_annotation (str): 当前已有的 QA 标注列表（包括初始和已修订的）。
            expert_history (List[Dict[str, str]]): 历史已招募专家的描述和角色。
                e.g., [{"role": "Marketing Expert", "description": "You are a XXX..."}]

        Returns:
            str: 格式化后的提示词。
        """
        
        # 格式化已招募的专家历史
        history_str = ""
        if expert_history:
            history_list = [f"Expert Role: {exp['role']}\nDescription: {exp['subtask']}" 
                            for exp in expert_history]
            history_str = "\n".join(history_list)
        else:
            history_str = "None."
            
        prompt = f"""
            {initial_prompt}
            {current_annotation}
            Now, you can create and collaborate with multiple experts to improve your generated question-answer pairs. Therefore, please describe in as much detail as possible the different skills and focuses you need from multiple experts individually. We will provide each expert with the same information and query. However, please note that each profession has its own specialization, so you can assign each expert to just one sub-task to ensure a more refined response. We will relay their responses to you in turn, allowing you to reorganize them into a better generation. Please note that the description should be narrated in the second person, for example: You are a XXX.
            These are the descriptions of the experts you have created before for this task:
            {history_str}
            Therefore, please remember you should not repeatedly create the same experts as described above. Now, you can give the description for a new expert (Please note that only be one, do not give multiple at one time):
            Please follow this exact output format:
            <Decision> [RECRUIT] or [TERMINATE] </Decision>
            If [TERMINATE], stop here and only output "NO_EXPERTS".
            If [RECRUIT], provide the expert details in JSON format:
            {{
            "Expert_Role": "<New Expert Role, e.g., Consumer Psychologist>",
            "Expert_Description": "<Second person narration of the expert's profile>",
            "Subtask": "<Specific task/focus for this expert based on the video gaps>"
            }}
            """
        return prompt.strip()

class MultiroleQAProfile4ExpertAgents:
    '''
    A collection of detailed profiles for various Expert Agents.
    This class is primarily used in Step 3 to initialize the Expert Agent's persona 
    before assigning a subtask.
    '''
    def __init__(self):
        # 内部可以存储所有已定义的专家档案，方便查找
        self._profiles = {
            "Conservation Psychologist": self._get_conservation_psychologist_profile()
            # 可以添加更多专家，如 Marketing Expert, Visual Design Expert 等
        }

    def _get_conservation_psychologist_profile(self) -> str:
        """
        返回保护心理学家的详细描述，用于 Expert Agent 的初始化。
        """
        return (
            "You are a conservation psychologist, specializing in understanding and promoting the psychological and "
            "emotional connection people have with nature and wildlife. Your expertise includes analyzing how visual "
            "and textual messaging in media can influence individuals’ attitudes and behaviors towards conservation and "
            "environmental protection. Your focus lies in interpreting the emotional responses elicited by multimedia "
            "content and identifying the aspects of an advertisement that enhance the viewers’ sense of urgency or empathy "
            "towards the subject. You provide insights on the psychological impact of specific scenes, colors, narratives, "
            "and the use of statistics or facts in fostering a sense of environmental stewardship and activism. Your role "
            "is to evaluate the effectiveness of the environmental messages conveyed and suggest ways to strengthen the "
            "emotional appeal and call to action within the advertisement."
        )

    def build_prompt(
        self, 
        expert_role: str, 
        v_context: str, 
        subtask: str, 
        master_agent_invitation: str = "Please generate new, professional QA pairs that fulfill the specific subtask assigned to you, based on your expertise."
    ) -> str:
        """
        Generates the full prompt for a specific Expert Agent (Step 3).

        Args:
            expert_role (str): 专家的角色名称，如 "Conservation Psychologist"。
            v_context (str): 序列化后的 V (视频广告多模态序列) 上下文。
            subtask (str): Master Agent 分配给该专家的具体子任务。
            master_agent_invitation (str): Master Agent 的邀请（通用指令）。

        Returns:
            str: 格式化后的提示词。
        """
        
        # 1. 根据角色获取配置文件
        expert_profile = self._profiles.get(expert_role, f"You are a skilled {expert_role} expert.")
        
        # 2. 构建最终的专家 Prompt
        prompt = f"""
            {expert_profile}
            VIDEO ADVERTISEMENT CONTEXT:
            {v_context}
            Your Specific Subtask: {subtask}
            Master Agent Invitation: {master_agent_invitation}
            Please generate a high-quality Question-Answer pair that leverage your specialized expertise to address the subtask. Your output must be a valid JSON list of objects, where each object has "Question" and "Answer" keys.
            """
        return prompt.strip()


class MultiroleQAMasterAgentRevisionPrompt:
    '''
    The prompt for the Master Agent to revise the current QA pool using the 
    newly generated QA from an Expert Agent.
    '''
    def __init__(self):
        pass

    def build_prompt(
        self, 
        current_qa_list_str: str, 
        expert_profile: str, 
        expert_qa_list_str: str
    ) -> str:
        """
        Generates the prompt for the Master Agent to refine the QA set.

        Args:
            current_qa_list_str (str): 当前已有的 QA 标注列表（包括初始和之前修订的）。
            expert_profile (str): 被邀请的专家角色及其描述。
            expert_qa_list_str (str): 专家智能体新生成的 QA 标注。

        Returns:
            str: 格式化后的提示词。
        """
        
        prompt = f"""
            Current QA Annotations or Initial Annotations are:
            {current_qa_list_str}
            You invite an expert whose description is: 
            {expert_profile}
            QA Annotations Generated by the last Expert Agent are:
            {expert_qa_list_str}
            Now you can refine your question-answer pairs with his generation to create more professional and challenging question-answer pairs. Keep in mind that his generation may not be perfect, so critically decide whether to accept some parts of his response or stick with your original one. 
            CRITICAL REVISION GUIDELINES:
            1. Integration: Combine the Current QA Annotations and the Expert's QA into a single, cohesive, high-quality list.
            2. Quality Check: Eliminate any QA pairs that are redundant, factually incorrect based on the video context (provided in the initial prompt history, which is implicit here), or poorly phrased.
            3. Enhancement: Utilize the expert's insights to improve the complexity, depth, and relevance of the existing questions, especially those related to the expert's specialty.
            4. Format: Your output must be a valid JSON list of dictionaries, representing the complete, revised set of QA pairs.
            Revised Question-Answer Pairs (Output only the JSON list):
            """
        return prompt.strip()

class MultiroleQADIYFinalQASynthesisPrompt:
    '''
    The prompt for the Master Agent to synthesize the final, definitive QA list 
    from all iterative annotations.
    '''
    def __init__(self):
        pass

    def build_prompt(
        self, 
        qa_history_list_str: List[str]
    ) -> str:
        """
        Generates the prompt for the Master Agent to synthesize the final QA set.

        Args:
            qa_history_list_str (List[str]): 多次迭代中所有生成的 QA 列表（以字符串形式）。

        Returns:
            str: 格式化后的提示词。
        """
        
        # 将历史 QA 列表连接成一个块，供 LLM 处理
        history_block = "\n" + "\n--- ITERATION BREAK ---\n".join(qa_history_list_str) + "\n"
        
        prompt = f"""
            You are the Master Agent responsible for the final synthesis and quality check of the generated Question-Answer pairs for the video advertisement. You have received several batches of QA pairs from multiple iterations, including initial generation and revisions by various expert agents.

            Your task is to review, consolidate, and finalize these QA pairs based on the following rules:

            1. De-duplication: Eliminate all exact or semantic duplicate QA pairs.
            2. Quality Selection: Select only the highest quality, most professional, and most challenging QA pairs from the pool(about 10 QA pairs).
            3. Consistency: Ensure the Questions relate only to the content of the advertisement (Context implicit in the history) and that the Answers are concise, unique, and unbiased.
            4. Format Adherence: The final output must strictly be a JSON list of dictionaries, containing only "Question" (Q) and "Answer" (A) keys.

            The complete history of QA pairs generated through all iterations is provided below:
            All Iterated QA Pairs are:
            {history_block}
            Final Selected QA Pairs (Output only the JSON list):
            """
        return prompt.strip()

class MultiroleQAClassificationPrompt:
    '''
    The prompt for the Master Agent to classify a single QA pair 
    into one or two of the five predefined categories.
    '''
    def __init__(self):
        pass

    def build_prompt(
        self, 
        Q_A: str
    ) -> str:
        """
        Generates the classification prompt for a single QA pair.

        Args:
            question (str): 需要分类的问题。
            answer (str): 对应问题的答案。

        Returns:
            str: 格式化后的提示词。
        """
        
        prompt = f"""
            Classify each of the question-answer pair in the question-answer pairs into one of the following categories:

            Type 1: The question-answer pair that focuses on the visual concepts, such as video details, characters in videos, a certain object, a certain scene, slogans presented in video, events, plot, and their interaction.
            Type 2: The question-answer pair that focuses on the emotional content by ad videos and assesses the potential psychological impact of these emotions.
            Type 3: The question-answer pair that focuses on the brand value, goal, theme, underlying message, or central idea** that the ad explores and conveys.
            Type 4: The question-answer pair that focuses on persuasion strategies that ad videos convey their core messages. These messages may not be directly articulated but could instead engage viewers through humor and visual rhetoric. (e.g., Any questions about the symbols, metaphors, humor, exaggeration, and any questions that focus on the Logical arguments, factual claims, Statistical charts, or expert opinions. (Any question about presenting factual information and logical arguments to demonstrate the product's benefits and value)
            Type 5: The question-answer pairs focus on the engagement, call for action, target audience, the characteristics of the target demographic, and who will be engaged.

            If this question could belong to multiple categories, please choose the most relevant two.

            Question and Answer pairs arse as follows: 
            {Q_A}

            Your output Labels should be just one or two of VU, ER, TE, PS, AM, and nothing else. (e.g., VU or AM)
            Output Format Specification (JSON Array and do not contain other word):
            [
            {{
                "Label": "<Label for the question>",
                "Q": "<The question>",
                "A": "<The correct>"
            }},
            // ... Continue with additional QA pairs in the same format
            ]
            """
        return prompt.strip()

# ------------------------------------------------------------------------------------------

