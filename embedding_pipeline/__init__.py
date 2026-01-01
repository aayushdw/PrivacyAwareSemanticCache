"""
Multi-stage embedding model evaluation pipeline.

This pipeline evaluates candidate embedding models, tunes thresholds,
ranks models by F1 score (with precision constraints), and produces
LoRA fine-tuned models for semantic caching.
"""

__version__ = "0.1.0"
