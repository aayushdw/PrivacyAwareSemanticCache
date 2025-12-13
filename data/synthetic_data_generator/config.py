"""Configuration management for synthetic data generation pipeline."""

import os
from typing import List
from dotenv import load_dotenv

load_dotenv()


class PipelineConfig:
    """Configuration for the synthetic data generation pipeline."""

    def __init__(
        self,
        input_file: str = "data/raw/small_final_questions.csv",
        output_file: str = "data/generated_data/synthetic_questions.csv",
        log_dir: str = "data/synthetic_data_generator/logs",
        model_list: List[str] = None,
        llm_rpm_limit: int = 15,
        similar_count: int = 2,
        different_count: int = 3,
        temperature: float = 0.7,
    ):
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")

        self.input_file = input_file
        self.output_file = output_file
        self.log_dir = log_dir
        self.model_list = model_list or ["gemma-3-4b-it"]
        self.llm_rpm_limit = llm_rpm_limit
        self.similar_count = similar_count
        self.different_count = different_count
        self.temperature = temperature

        # Calculate minimum interval between questions (2 calls per question)
        self.min_interval = 60 / self.llm_rpm_limit * 2

    def __repr__(self):
        return (
            f"PipelineConfig(models={self.model_list}, rpm_limit={self.llm_rpm_limit}, "
            f"min_interval={self.min_interval}s)"
        )
