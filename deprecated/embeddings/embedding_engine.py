"""
Embedding Engine using Sentence Transformers with MiniLM model.
Provides text encoding functionality for semantic similarity tasks.
"""

from typing import List, Union
import numpy as np
import re
from sentence_transformers import SentenceTransformer


class EmbeddingEngine:
    """
    Embedding engine that uses Sentence Transformers with MiniLM model
    to encode text into vector representations.
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', enable_preprocessing: bool = True):
        """
        Initialize the embedding engine with a Sentence Transformer model.

        Args:
            model_name: Sentence Transformer model to use.
                       Default is 'all-MiniLM-L6-v2' which provides a good
                       balance between speed and quality.
            enable_preprocessing: Whether to enable text preprocessing by default.
                                 Preprocessing improves semantic caching by normalizing text.
        """
        self.model_name = model_name
        self.enable_preprocessing = enable_preprocessing
        print(f"Loading Sentence Transformer model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print(f"Model loaded successfully. Embedding dimension: {self.get_embedding_dimension()}")
        if enable_preprocessing:
            print("Text preprocessing enabled for better semantic caching")

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for better semantic caching by normalizing variations.

        Preprocessing steps:
        1. Strip leading/trailing whitespace
        2. Normalize whitespace (multiple spaces/tabs/newlines to single space)
        3. Convert to lowercase for case-insensitive matching
        4. Remove special characters that don't affect semantics
        5. Normalize punctuation spacing

        Args:
            text: Input text string

        Returns:
            Preprocessed text string
        """
        if not text or not isinstance(text, str):
            return text

        # Strip leading/trailing whitespace
        text = text.strip()

        # Normalize whitespace (multiple spaces, tabs, newlines to single space)
        text = re.sub(r'\s+', ' ', text)

        # Convert to lowercase for case-insensitive semantic matching
        text = text.lower()

        # Normalize punctuation spacing (remove spaces before punctuation)
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)

        # Add space after punctuation if missing
        text = re.sub(r'([.,!?;:])([^\s\d])', r'\1 \2', text)

        # Remove excessive punctuation (keep single punctuation)
        text = re.sub(r'([.,!?;:]){2,}', r'\1', text)

        # Remove special characters that don't affect semantics (keep alphanumeric, spaces, basic punctuation)
        text = re.sub(r'[^\w\s.,!?;:\'\"-]', ' ', text)

        # Final whitespace normalization
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def encode(self, text: Union[str, List[str]],
               normalize: bool = True,
               batch_size: int = 32,
               show_progress_bar: bool = False,
               preprocess: bool = None) -> np.ndarray:
        """
        Encode text into vector embeddings.

        Args:
            text: Single text string or list of text strings to encode
            normalize: Whether to normalize embeddings to unit length (recommended for cosine similarity)
            batch_size: Batch size for encoding multiple texts
            show_progress_bar: Whether to show progress bar for batch encoding
            preprocess: Whether to preprocess text before encoding. If None, uses enable_preprocessing from __init__

        Returns:
            numpy array of shape (embedding_dim,) for single text or
            (num_texts, embedding_dim) for multiple texts
        """
        # Determine if preprocessing should be applied
        should_preprocess = preprocess if preprocess is not None else self.enable_preprocessing

        # Preprocess text if enabled
        if should_preprocess:
            if isinstance(text, str):
                text = self.preprocess_text(text)
            elif isinstance(text, list):
                text = [self.preprocess_text(t) if isinstance(t, str) else t for t in text]

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
                    show_progress_bar: bool = True,
                    preprocess: bool = None) -> np.ndarray:
        """
        Encode a batch of texts into vector embeddings.
        This is an alias for encode() with batch-optimized defaults.

        Args:
            texts: List of text strings to encode
            normalize: Whether to normalize embeddings to unit length
            batch_size: Batch size for encoding
            show_progress_bar: Whether to show progress bar
            preprocess: Whether to preprocess text before encoding. If None, uses enable_preprocessing from __init__

        Returns:
            numpy array of shape (num_texts, embedding_dim)
        """
        return self.encode(
            texts,
            normalize=normalize,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            preprocess=preprocess
        )

    def __repr__(self) -> str:
        """String representation of the embedding engine."""
        return f"EmbeddingEngine(model='{self.model_name}', dim={self.get_embedding_dimension()})"
