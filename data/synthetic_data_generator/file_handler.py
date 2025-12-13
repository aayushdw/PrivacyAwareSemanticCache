"""Thread-safe file I/O operations for synthetic data generation."""

import os
import csv
import threading
from typing import List, Dict, Any
from datetime import datetime

from .models import GeneratedQuestion

# Global file lock for thread-safe writes
_file_lock = threading.Lock()
_error_lock = threading.Lock()


def load_input_questions(file_path: str) -> List[Dict[str, Any]]:
    """Load questions from input CSV file."""
    questions = []
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            questions.append({"qid": int(row["qid"]), "question": row["question"]})
    return questions


def initialize_output_file(file_path: str) -> None:
    """Initialize output CSV file with headers if it doesn't exist."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if not os.path.exists(file_path):
        with open(file_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "original_question", "generated_question", "is_semantically_similar"])


def write_results_threadsafe(results: List[GeneratedQuestion], output_path: str) -> None:
    """Append results to output CSV file with thread safety."""
    with _file_lock:
        with open(output_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["id", "original_question", "generated_question", "is_semantically_similar"]
            )
            for result in results:
                writer.writerow(result.to_dict())


def initialize_log_dir(log_dir: str) -> None:
    """Create log directory if it doesn't exist."""
    os.makedirs(log_dir, exist_ok=True)


def log_failed_question(qid: int, error_message: str, log_dir: str) -> None:
    """Log a failed question ID to the error log file."""
    log_path = os.path.join(log_dir, "failed_questions.log")
    timestamp = datetime.now().isoformat()
    with _error_lock:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{timestamp},{qid},{error_message}\n")
