"""
Vector Database module for storing and retrieving embeddings using ChromaDB.
Provides semantic search capabilities for privacy-aware caching.
"""

from .chroma_store import ChromaVectorStore

__all__ = ['ChromaVectorStore']
