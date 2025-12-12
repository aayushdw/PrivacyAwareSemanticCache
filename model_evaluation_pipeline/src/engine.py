import torch
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import numpy as np

def get_model(model_name: str) -> SentenceTransformer:
    # Check for Apple Silicon GPU (MPS)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading model '{model_name}' on device: {device.upper()}")
    return SentenceTransformer(model_name, device=device)

def get_similarity_scores(model: SentenceTransformer, q1: List[str], q2: List[str], batch_size: int = 32) -> np.ndarray:
    """
    Encodes list pairs and calculates cosine similarity with batching.

    Args:
        model: SentenceTransformer model
        q1: List of first questions
        q2: List of second questions
        batch_size: Batch size for encoding (default: 32)

    Returns:
        Array of cosine similarity scores
    """
    # convert_to_tensor=True allows the model to stay on GPU during encode
    # batch_size prevents OOM on large datasets
    embeddings1 = model.encode(q1, convert_to_tensor=True, batch_size=batch_size, show_progress_bar=True)
    embeddings2 = model.encode(q2, convert_to_tensor=True, batch_size=batch_size, show_progress_bar=True)

    # Compute Cosine Similarity (Result is a Tensor on MPS)
    cosine_scores = torch.nn.functional.cosine_similarity(embeddings1, embeddings2)

    # Move back to CPU for numpy operations
    return cosine_scores.cpu().numpy()

def get_model_info(model: SentenceTransformer) -> Dict[str, float]:
    """
    Extract model metadata and memory footprint.

    Args:
        model: SentenceTransformer model

    Returns:
        Dictionary with model parameters and size information
    """
    num_parameters = sum(p.numel() for p in model.parameters())
    model_size_mb = (num_parameters * 4) / (1024 * 1024)  # Assuming float32 (4 bytes per param)

    return {
        "model_parameters_millions": num_parameters / 1e6,
        "model_size_mb": model_size_mb,
    }