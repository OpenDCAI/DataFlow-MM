from __future__ import annotations

import re
from typing import Optional

import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage


@OPERATOR_REGISTRY.register()
class GeneralTextAnswerEvaluator(OperatorABC):
    """Evaluate text-based answers across multiple question types.
    
    This evaluator supports multiple question types with appropriate scoring metrics:
    - multiple choice: Exact match (0 or 1)
    - numerical: Numerical comparison with decimal handling
    - OCR: Word Error Rate (WER) based scoring
    - free-form: ROUGE score based evaluation
    - regression: Relative difference based scoring
    """

    def __init__(
        self,
        use_stemmer: bool = True,
    ) -> None:
        """Initialize the evaluator.
        
        Args:
            use_stemmer: Whether to use stemmer in ROUGE score calculation
        """
        self.logger = get_logger()
        self.use_stemmer = use_stemmer

    @staticmethod
    def get_desc(lang: str = "zh") -> str:
        return "通用文本答案评估器" if lang == "zh" else "General text answer evaluator."

    # ----------------------------- Helper Methods ----------------------------
    @staticmethod
    def extract_answer(text: str) -> str:
        """Extract content between <answer> and </answer> tags."""
        pattern = r'<answer>\s*(.*?)\s*</answer>'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    @staticmethod
    def normalize_number(num_str: str) -> Optional[float]:
        """Convert string to float, handling commas."""
        try:
            num_str = num_str.replace(',', '')
            return float(num_str)
        except Exception:
            return None

    @staticmethod
    def wer(reference: str, hypothesis: str) -> float:
        """Calculate Word Error Rate."""
        ref_words = reference.split()
        hyp_words = hypothesis.split()
        m = len(ref_words)
        n = len(hyp_words)
        d = [[0]*(n+1) for _ in range(m+1)]
        for i in range(m+1):
            d[i][0] = i
        for j in range(n+1):
            d[0][j] = j
        for i in range(1, m+1):
            for j in range(1, n+1):
                if ref_words[i-1] == hyp_words[j-1]:
                    d[i][j] = d[i-1][j-1]
                else:
                    d[i][j] = 1 + min(d[i-1][j], d[i][j-1], d[i-1][j-1])
        return d[m][n] / max(1, m)

    @staticmethod
    def compute_bleu_score(reference: str, hypothesis: str) -> float:
        """Calculate BLEU score."""
        try:
            smoothing = SmoothingFunction().method1
            ref_tokens = reference.split()
            hyp_tokens = hypothesis.split()
            score = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothing)
            return score
        except Exception:
            return 0.0

    def compute_rouge_score(self, reference: str, hypothesis: str) -> float:
        """Calculate average ROUGE F-measure."""
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=self.use_stemmer)
        scores = scorer.score(reference, hypothesis)
        average_fmeasure = (scores['rouge1'].fmeasure + scores['rouge2'].fmeasure + scores['rougeL'].fmeasure) / 3
        return average_fmeasure

    def calculate_reward(self, gt_solution: str, model_output: str, question_type: str) -> float:
        """Calculate reward based on question type and model output.
        
        Args:
            gt_solution: Ground truth solution (with <answer> tags)
            model_output: Model generated output (with <answer> tags)
            question_type: Type of question (multiple choice, numerical, OCR, free-form, regression)
            
        Returns:
            Reward score between 0.0 and 1.0
        """
        try:
            output_ans = self.extract_answer(model_output)
            gt_ans = self.extract_answer(gt_solution)
            
            if question_type == "multiple choice":
                return 1.0 if output_ans.strip() == gt_ans.strip() else 0.0
            elif question_type == "numerical":
                gt_has_decimal = ("." in gt_ans) or ("," in gt_ans)
                out_has_decimal = ("." in output_ans) or ("," in output_ans)
                if gt_has_decimal != out_has_decimal:
                    return 0.0
                gt_number = self.normalize_number(gt_ans)
                out_number = self.normalize_number(output_ans)
                if gt_number is None or out_number is None:
                    return 0.0
                return 1.0 if round(gt_number, 2) == round(out_number, 2) else 0.0
            elif question_type == "OCR":
                error_rate = self.wer(gt_ans, output_ans)
                reward = 1 - error_rate
                return max(0.0, min(1.0, reward))
            elif question_type == "free-form":
                score = self.compute_rouge_score(gt_ans, output_ans)
                return max(0.0, min(1.0, score))
            elif question_type == "regression":
                gt_number = self.normalize_number(gt_ans)
                out_number = self.normalize_number(output_ans)
                if gt_number is None or out_number is None:
                    return 0.0
                rel_diff = (abs(out_number - gt_number) + 1e-9) / (abs(gt_number) + 1e-9)
                rel_diff = min(1.0, max(0.0, rel_diff))
                return 1 - rel_diff
            else:
                return 0.0
        except Exception as e:
            self.logger.error(f"Error in calculate_reward for question_type '{question_type}': {e}")
            return 0.0

    # ----------------------------- Execution ---------------------------------
    def run(
        self,
        storage: DataFlowStorage,
        model_output_key: str = "model_output",
        gt_solution_key: str = "solution",
        question_type_key: str = "problem_type",
        output_reward_key: str = "reward",
    ) -> str:
        """Evaluate text answers and compute rewards.
        
        Args:
            storage: DataFlowStorage object
            model_output_key: Column name for model outputs
            gt_solution_key: Column name for ground truth solutions
            question_type_key: Column name for question types
            output_reward_key: Column name for output rewards
            
        Returns:
            The output_reward_key
            
        Expected input columns:
            - model_output_key: Model generated text with <answer> tags
            - gt_solution_key: Ground truth with <answer> tags
            - question_type_key: Question type (multiple choice, numerical, OCR, free-form, regression)
        """
        self.logger.info("Running GeneralTextAnswerEvaluator...")
        
        df: pd.DataFrame = storage.read("dataframe")
        self.logger.info("Loaded dataframe with %d rows", len(df))

        # Validate required columns
        required_cols = [model_output_key, gt_solution_key, question_type_key]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise KeyError(f"Missing required columns: {missing_cols}")

        # Calculate rewards
        rewards = []
        
        for _, row in df.iterrows():
            model_output = row.get(model_output_key, '')
            gt_solution = row.get(gt_solution_key, '')
            question_type = row.get(question_type_key, '')
            
            reward = self.calculate_reward(gt_solution, model_output, question_type)
            rewards.append(reward)

        # Attach outputs
        df[output_reward_key] = rewards
        
        storage.write(df)

        self.logger.info("Evaluation finished for %d rows", len(df))
        self.logger.info(f"Average reward: {sum(rewards)/len(rewards):.4f}")
        
        return output_reward_key

