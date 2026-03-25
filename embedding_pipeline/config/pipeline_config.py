"""Pipeline configuration for the embedding model evaluation pipeline."""

import os
from dataclasses import dataclass, field
from typing import Tuple, List, Optional
from pathlib import Path


@dataclass
class EvaluationConfig:
    """Configuration for the evaluation stage."""

    min_precision: float = 0.80
    threshold_range: Tuple[float, float] = (0.50, 0.99)
    threshold_step: float = 0.01
    train_data: str = "data/processed/train.csv"
    test_data: str = "data/processed/val.csv"
    max_samples: int = 5000
    batch_size: int = 64


@dataclass
class RankingConfig:
    """Configuration for the ranking stage."""

    top_n: int = 3
    metric: str = "f1_score"
    min_precision: float = 0.80


@dataclass
class LoRAConfig:
    """Configuration for LoRA fine-tuning."""

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: List[str] = field(
        default_factory=lambda: ["query", "key", "value", "dense"]
    )
    learning_rate: float = 2e-4
    batch_size: int = 8
    max_epochs: int = 2
    patience: int = 1
    min_delta: float = 0.0001
    gradient_accumulation_steps: int = 8
    max_seq_length: int = 128
    train_data: str = "data/processed/train.csv"
    val_data: str = "data/processed/val.csv"


@dataclass
class MLFlowConfig:
    """Configuration for MLFlow integration."""

    tracking_uri: str = "http://localhost:5001"
    experiment_prefix: str = "Semantic_Cache_Evaluation_v3"
    threshold_tuning_experiment: str = "01_threshold_tuning"
    ranking_experiment: str = "02_model_ranking"
    lora_training_experiment: str = "03_lora_training"


@dataclass
class PipelineConfig:
    """Main pipeline configuration."""

    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    ranking: RankingConfig = field(default_factory=RankingConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    mlflow: MLFlowConfig = field(default_factory=MLFlowConfig)

    # Pipeline control
    skip_evaluation: bool = False
    skip_lora_training: bool = False
    model_categories: List[str] = field(
        default_factory=lambda: ["fast", "balanced", "quality"]
    )

    # Parallelization
    max_parallel_evaluations: int = 2
    small_model_threshold_mb: float = 500.0

    # Output paths
    output_dir: str = "embedding_pipeline/outputs"
    reports_dir: str = "embedding_pipeline/outputs/reports"
    models_dir: str = "embedding_pipeline/outputs/models"

    # Model cache directory (for HuggingFace/SentenceTransformers downloads)
    model_cache_dir: Optional[str] = None  # None = use default ~/.cache/huggingface


# Default configuration instance
PIPELINE_CONFIG = PipelineConfig()


def get_config() -> PipelineConfig:
    """Get the pipeline configuration, applying environment variable overrides."""
    config = PipelineConfig()

    # Override MLFlow tracking URI from environment
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
    if mlflow_uri:
        config.mlflow.tracking_uri = mlflow_uri

    # Override precision constraint from environment
    min_precision = os.getenv("PIPELINE_MIN_PRECISION")
    if min_precision:
        config.evaluation.min_precision = float(min_precision)
        config.ranking.min_precision = float(min_precision)

    # Override top-N from environment
    top_n = os.getenv("PIPELINE_TOP_N")
    if top_n:
        config.ranking.top_n = int(top_n)

    # Override model cache directory from environment
    # Falls back to ~/.cache/huggingface if not set
    cache_dir = os.getenv("HF_HOME") or os.getenv("MODEL_CACHE_DIR")
    if cache_dir:
        config.model_cache_dir = cache_dir
    elif config.model_cache_dir is None:
        # Default to user's home cache directory
        config.model_cache_dir = os.path.expanduser("~/.cache/huggingface")

    return config


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


def get_data_path(relative_path: str) -> Path:
    """Get absolute path for a data file relative to project root."""
    return get_project_root() / relative_path


def get_experiment_name(stage: str) -> str:
    """Get full experiment name for a pipeline stage."""
    config = get_config()
    stage_map = {
        "threshold_tuning": config.mlflow.threshold_tuning_experiment,
        "ranking": config.mlflow.ranking_experiment,
        "lora_training": config.mlflow.lora_training_experiment,
    }
    stage_suffix = stage_map.get(stage, stage)
    return f"{config.mlflow.experiment_prefix}/{stage_suffix}"
