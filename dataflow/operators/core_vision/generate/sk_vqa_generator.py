from dataflow.core.Operator import OperatorABC

from dataflow.prompts.image import SKVQAGeneratorPrompt
import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger

from dataflow.utils.storage import FileStorage, DataFlowStorage
from dataflow.core import LLMServingABC
from dataflow.serving.local_model_vlm_serving import LocalModelVLMServing_vllm

from qwen_vl_utils import process_vision_info

import re

def normalize_whitespace(s: str) -> str:
    """Collapse whitespace to single spaces and trim."""
    return re.sub(r'\s+', ' ', s or '').strip()

def parse_wiki_qa(text: str) -> dict:
    """
    解析包含 '### Wikipedia Article' 和 '### Question Answer Pairs' 的文本。
    返回格式:
    {
        "context": "文章内容",
        "qas": [
            {"question": "问题", "answer": "答案"},
            ...
        ]
    }
    """
    if not isinstance(text, str) or not text.strip():
        return {"context": "", "qas": []}

    try:
        # 去除多余的星号、空行
        text_clean = re.sub(r'(?<!\*)\*(?!\*)', '', text)
        text_clean = text_clean.strip()

        # 提取 Wikipedia Article 段落
        m_article = re.search(
            r'###\s*Wikipedia Article\s*(.*?)\n###\s*Question Answer Pairs',
            text_clean, flags=re.DOTALL | re.IGNORECASE
        )
        article = normalize_whitespace(m_article.group(1)) if m_article else ""

        # 提取 QA 段落
        qa_section_match = re.search(
            r'###\s*Question Answer Pairs\s*(.*)',
            text_clean, flags=re.DOTALL | re.IGNORECASE
        )
        qas = []

        if qa_section_match:
            qa_section = qa_section_match.group(1).strip()

            # 优先用正则批量匹配问答对
            pattern = re.compile(
                r'\d+\.\s*\*\*(.*?)\*\*\s*(?:\r?\n|\s)*-+\s*(.+?)(?=(?:\n\d+\.|\Z))',
                flags=re.DOTALL
            )
            matches = pattern.findall(qa_section)

            for q, a in matches:
                q_text = normalize_whitespace(q)
                a_text = normalize_whitespace(a.replace('\n', ' '))
                a_text = re.sub(r'^\-+\s*', '', a_text).replace('*', '')
                if q_text and a_text:
                    qas.append({"question": q_text, "answer": a_text})

            # 如果没匹配到，用简单行级匹配容错
            if not qas:
                lines = qa_section.splitlines()
                cur_q = None
                for line in lines:
                    line = line.strip()
                    if re.match(r'^\d+\.\s*\*\*(.+)\*\*$', line):
                        cur_q = re.sub(r'^\d+\.\s*\*\*(.+)\*\*$', r'\1', line).strip()
                    elif line.startswith('-') and cur_q:
                        ans = line.lstrip('-').strip()
                        if cur_q and ans:
                            qas.append({
                                "question": normalize_whitespace(cur_q),
                                "answer": normalize_whitespace(ans)
                            })
                        cur_q = None

        # 最终结果
        return {"context": article, "qas": qas}

    except Exception as e:
        # 任意异常时安全返回空结构
        return {"context": "", "qas": []}


@OPERATOR_REGISTRY.register()
class ImageSKVQAGenerate(OperatorABC):
    '''
    SKVQA Generator is a class that generates structured visual question–answer descriptions for given images.
    '''
    def __init__(self, llm_serving: LLMServingABC):
        self.logger = get_logger()
        self.prompt_generator = SKVQAGeneratorPrompt()
        self.llm_serving = llm_serving

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "该算子用于生成 Synthetic Knowledge VQA（SKVQA）结果。\n\n"
                "📘 什么是 SKVQA：\n"
                "  - SKVQA（合成知识视觉问答）是在普通 VQA 的基础上增加了“上下文 (context)”信息，\n"
                "    模型不仅根据图像内容回答问题，还需结合给定的背景知识或文本片段进行推理。\n"
                "  - 这样可以让模型在面对复杂或知识相关的问题时，更好地理解场景并生成合理答案。\n\n"
                "🧩 功能说明：\n"
                "  - 输入多模态数据（如图像）后，自动构造提示词并调用视觉语言大模型生成结构化问答输出。\n"
                "  - 输出格式为：\n"
                "    {\n"
                "      'context': '与图像相关的上下文',\n"
                "      'qas': [\n"
                "        {'question': '问题1', 'answer': '答案1'},\n"
                "        {'question': '问题2', 'answer': '答案2'}\n"
                "      ]\n"
                "    }\n\n"
                "🧠 与普通 VQA 的区别：\n"
                "  - 普通 VQA：仅根据图像本身回答。\n"
                "  - SKVQA：在回答时结合上下文内容，更贴近真实推理与知识理解。\n\n"
                "⚙️ 参数说明：\n"
                "  - multi_modal_key: 输入图像所在列名，默认 'image'。\n"
                "  - output_key: 输出结果列名，默认 'skvqa'。\n\n"
                "💡 典型应用场景：\n"
                "  - 图像 + 产品说明 → 自动生成产品问答。\n"
                "  - 图片 + 文档内容 → 视觉知识理解。\n"
                "  - 多模态知识融合训练或数据增强。"
            )
        else:
            return (
                "This operator generates Synthetic Knowledge VQA (SKVQA) outputs.\n\n"
                "📘 What is SKVQA:\n"
                "  - SKVQA (Synthetic Knowledge Visual Question Answering) extends normal VQA by adding a textual 'context'.\n"
                "  - The model answers questions not only from the image but also by reasoning with the provided background text.\n\n"
                "🧩 Function:\n"
                "  - Takes images as input, builds prompts automatically, and uses a vision-language model to generate structured Q&A.\n"
                "  - Output format:\n"
                "    {\n"
                "      'context': 'related background information',\n"
                "      'qas': [\n"
                "        {'question': 'Question 1', 'answer': 'Answer 1'},\n"
                "        {'question': 'Question 2', 'answer': 'Answer 2'}\n"
                "      ]\n"
                "    }\n\n"
                "🧠 Difference from normal VQA:\n"
                "  - Normal VQA: answers purely from the image.\n"
                "  - SKVQA: answers by combining visual evidence with external knowledge or text context.\n\n"
                "⚙️ Parameters:\n"
                "  - multi_modal_key: name of the image column (default 'image').\n"
                "  - output_key: name of the output column (default 'skvqa').\n\n"
                "💡 Typical use cases:\n"
                "  - Image + product description → auto-generate product Q&A.\n"
                "  - Image + document → visual knowledge reasoning.\n"
                "  - Multimodal data augmentation or reasoning tasks."
            )

    
    def _validate_dataframe(self, dataframe: pd.DataFrame):
        required_keys = [self.multi_modal_key]
        forbidden_keys = [self.output_key]

        missing = [k for k in required_keys if k not in dataframe.columns]
        conflict = [k for k in forbidden_keys if k in dataframe.columns]

        if missing:
            raise ValueError(f"Missing required column(s): {missing}")
        if conflict:
            raise ValueError(f"The following column(s) already exist and would be overwritten: {conflict}")

    def _prepare_batch_inputs(self, media_paths):
        """
        Construct batched prompts and image inputs from media paths.
        """
        prompts = self.prompt_generator.build_prompt()

        prompt_list = []
        image_inputs_list = []

        for paths in media_paths:
            for p in paths:
                raw_prompt = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": p},
                            {"type": "text", "text": prompts},
                        ],
                    },
                ]
                # Get vision inputs
                image_inputs, _ = process_vision_info(raw_prompt)

                # Format prompt using LLM processor
                prompt = self.llm_serving.processor.apply_chat_template(
                    raw_prompt, tokenize=False, add_generation_prompt=True
                )

                image_inputs_list.append(image_inputs)
                prompt_list.append(prompt)

        return prompt_list, image_inputs_list

    def run(
        self,
        storage: DataFlowStorage,
        input_modal_key: str = "image", 
        output_key: str = "skvqa"
    ):
        """
        Runs the SKVQA generation process in batch mode, reading from the input file and saving results to output.
        """
        self.multi_modal_key, self.output_key = input_modal_key, output_key
        dataframe = storage.read("dataframe")
        self._validate_dataframe(dataframe)
        
        media_paths = dataframe.get(self.multi_modal_key, pd.Series([])).tolist()
        media_paths = [path if isinstance(path, list) else [path] for path in media_paths]
        
        prompt_list, image_inputs_list = self._prepare_batch_inputs(media_paths)

        outputs = self.llm_serving.generate_from_input(
            user_inputs=prompt_list,
            image_inputs=image_inputs_list
        )

        # 提取context和qa，然后存到skvqa这个key下面
        # 批量解析每个输出
        skvqa_results = []
        for out in outputs:
            skvqa_results.append(parse_wiki_qa(out))
        dataframe[self.output_key] = skvqa_results

        output_file = storage.write(dataframe)
        self.logger.info(f"Results saved to {output_file}")

        return [output_key]


if __name__ == "__main__":
    # Initialize model
    model = LocalModelVLMServing_vllm(
        hf_model_name_or_path="Qwen/Qwen2.5-VL-3B-Instruct",
        vllm_tensor_parallel_size=1,
        vllm_temperature=0.7,
        vllm_top_p=0.9,
        vllm_max_tokens=512,
    )

    skvqa_generator = ImageSKVQAGenerate(
        llm_serving=model
    )

    # Prepare input
    storage = FileStorage(
        first_entry_file_name="dataflow/example/image_to_text_pipeline/capsbench_captions.jsonl", 
        cache_type="jsonl"
    )
    storage.step()  # Load the data

    skvqa_generator.run(
        storage=storage,
        input_modal_key="image",
        output_key="skvqa"
    )