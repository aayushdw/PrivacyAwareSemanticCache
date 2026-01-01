"""Extended ModelInfo dataclass for embedding models."""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ModelInfo:
    """
    Extended information about an embedding model.

    Includes metadata for pipeline execution, MPS compatibility,
    and memory estimation for M4 MacBook Air.
    """

    # Core identification
    name: str
    model_id: str  # HuggingFace model ID
    description: str
    dimension: int
    category: str  # 'fast', 'balanced', 'quality', 'multilingual'

    # Extended metadata for pipeline
    requires_instruction: bool = False
    instruction_template: Optional[str] = None
    max_seq_length: int = 512

    # Memory and compatibility
    estimated_memory_mb: Optional[float] = None
    supports_mps: bool = True
    supports_lora: bool = True

    # LoRA-specific configuration
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["query", "key", "value", "dense"]
    )

    # Evaluation state (populated at runtime, not persisted)
    optimal_threshold: Optional[float] = None
    f1_score: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    meets_constraint: bool = False

    def __post_init__(self):
        """Estimate memory if not provided."""
        if self.estimated_memory_mb is None:
            # Rough estimation based on dimension and typical model architecture
            # Small models (384d): ~100MB, Medium (768d): ~400MB, Large (1024d): ~800MB
            if self.dimension <= 384:
                self.estimated_memory_mb = 100.0
            elif self.dimension <= 768:
                self.estimated_memory_mb = 400.0
            elif self.dimension <= 1024:
                self.estimated_memory_mb = 800.0
            else:
                self.estimated_memory_mb = 1500.0

    def is_small_model(self, threshold_mb: float = 500.0) -> bool:
        """Check if model is considered small (can be parallelized)."""
        return (self.estimated_memory_mb or 0) < threshold_mb

    def get_size_category(self) -> str:
        """Get model size category for logging."""
        mem = self.estimated_memory_mb or 0
        if mem < 200:
            return "small"
        elif mem < 600:
            return "medium"
        else:
            return "large"

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "model_id": self.model_id,
            "description": self.description,
            "dimension": self.dimension,
            "category": self.category,
            "requires_instruction": self.requires_instruction,
            "instruction_template": self.instruction_template,
            "max_seq_length": self.max_seq_length,
            "estimated_memory_mb": self.estimated_memory_mb,
            "supports_mps": self.supports_mps,
            "supports_lora": self.supports_lora,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ModelInfo":
        """Create ModelInfo from dictionary."""
        return cls(
            name=data["name"],
            model_id=data["model_id"],
            description=data["description"],
            dimension=data["dimension"],
            category=data["category"],
            requires_instruction=data.get("requires_instruction", False),
            instruction_template=data.get("instruction_template"),
            max_seq_length=data.get("max_seq_length", 512),
            estimated_memory_mb=data.get("estimated_memory_mb"),
            supports_mps=data.get("supports_mps", True),
            supports_lora=data.get("supports_lora", True),
        )
