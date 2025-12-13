"""Data models for synthetic data generation pipeline."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, TypedDict, Annotated
from operator import add


@dataclass
class QuestionTask:
    """A question task assigned to a specific model."""
    qid: int
    question: str
    assigned_model: str


@dataclass
class GeneratedQuestion:
    """A generated synthetic question."""
    id: str  # Format: "{qid}_{index}"
    original_question: str
    generated_question: str
    is_semantically_similar: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "original_question": self.original_question,
            "generated_question": self.generated_question,
            "is_semantically_similar": self.is_semantically_similar,
        }


@dataclass
class WorkerResult:
    """Result from a model worker."""
    model_name: str
    processed_count: int
    failed_qids: List[int]
    generated_count: int


class PipelineState(TypedDict):
    """LangGraph pipeline state."""
    # Input questions loaded from CSV
    questions: List[Dict[str, Any]]

    # Questions distributed to each model: {model_name: [questions]}
    model_assignments: Dict[str, List[QuestionTask]]

    # Results from workers (accumulated using reducer)
    worker_results: Annotated[List[WorkerResult], add]

    # Summary stats
    total_processed: int
    total_failed: int
    total_generated: int
