"""
Embedding Engine using Sentence Transformers with MiniLM model.
Provides text encoding functionality for semantic similarity tasks.
"""

from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingEngine:
    """
    Embedding engine that uses Sentence Transformers with MiniLM model
    to encode text into vector representations.
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the embedding engine with a Sentence Transformer model.

        Args:
            model_name: Sentence Transformer model to use.
                       Default is 'all-MiniLM-L6-v2' which provides a good
                       balance between speed and quality.
        """
        self.model_name = model_name
        print(f"Loading Sentence Transformer model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print(f"Model loaded successfully. Embedding dimension: {self.get_embedding_dimension()}")

    def encode(self, text: Union[str, List[str]],
               normalize: bool = True,
               batch_size: int = 32,
               show_progress_bar: bool = False) -> np.ndarray:
        """
        Encode text into vector embeddings.

        Args:
            text: Single text string or list of text strings to encode
            normalize: Whether to normalize embeddings to unit length (recommended for cosine similarity)
            batch_size: Batch size for encoding multiple texts
            show_progress_bar: Whether to show progress bar for batch encoding

        Returns:
            numpy array of shape (embedding_dim,) for single text or
            (num_texts, embedding_dim) for multiple texts
        """
        embeddings = self.model.encode(
            text,
            normalize_embeddings=normalize,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True
        )
        return embeddings

    def get_embedding_dimension(self) -> int:
        """
        Get the dimensionality of the embeddings produced by this model.

        Returns:
            Integer representing the embedding dimension
        """
        return self.model.get_sentence_embedding_dimension()

    def encode_batch(self, texts: List[str],
                    normalize: bool = True,
                    batch_size: int = 32,
                    show_progress_bar: bool = True) -> np.ndarray:
        """
        Encode a batch of texts into vector embeddings.
        This is an alias for encode() with batch-optimized defaults.

        Args:
            texts: List of text strings to encode
            normalize: Whether to normalize embeddings to unit length
            batch_size: Batch size for encoding
            show_progress_bar: Whether to show progress bar

        Returns:
            numpy array of shape (num_texts, embedding_dim)
        """
        return self.encode(
            texts,
            normalize=normalize,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar
        )

    def __repr__(self) -> str:
        """String representation of the embedding engine."""
        return f"EmbeddingEngine(model='{self.model_name}', dim={self.get_embedding_dimension()})"
