"""
Semantic Cache for LLM queries with privacy-preserving capabilities.

This module provides a semantic caching layer that uses embeddings and vector
similarity to detect semantically similar queries and return cached responses,
reducing LLM API calls and improving response times.

Note: ChromaDB uses cosine distance (not similarity):
- Cosine distance = 1 - cosine similarity
- Range: 0 to 2 (0 = identical, 1 = orthogonal, 2 = opposite)
- Lower distance = more similar

Converting from evaluation thresholds:
- If your model evaluation found optimal cosine similarity threshold (e.g., 0.78)
- Convert to distance: cosine_distance = 1 - cosine_similarity
- Example: similarity 0.78 → distance threshold = 1 - 0.78 = 0.22
"""

import os
from typing import Optional, Dict, Any, Tuple
from datetime import datetime
import sys

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from embeddings.embedding_engine import EmbeddingEngine
from vector_database.chroma_store import ChromaVectorStore
from llm_provider import GeminiProvider


def similarity_to_distance_threshold(similarity_threshold: float) -> float:
    """
    Convert cosine similarity threshold to cosine distance threshold.

    Use this function to convert thresholds found during model evaluation
    (which typically use cosine similarity) to the distance threshold
    required by ChromaDB.

    Args:
        similarity_threshold: Cosine similarity threshold (0 to 1)
                            Higher similarity = more similar
                            Example: 0.78 from model evaluation

    Returns:
        float: Cosine distance threshold (0 to 2)
               Lower distance = more similar
               Example: 0.22 for cache hits

    Example:
        >>> # MPNet-Base evaluation found optimal similarity threshold = 0.78
        >>> distance_threshold = similarity_to_distance_threshold(0.78)
        >>> print(distance_threshold)  # 0.22
        >>>
        >>> # Use this threshold with SemanticCache
        >>> cache = SemanticCache(
        ...     embedding_model='sentence-transformers/all-mpnet-base-v2',
        ...     similarity_threshold=distance_threshold
        ... )
    """
    if not 0.0 <= similarity_threshold <= 1.0:
        raise ValueError("Similarity threshold must be between 0.0 and 1.0")

    distance_threshold = 1.0 - similarity_threshold
    return distance_threshold


class SemanticCache:
    """
    Semantic cache that stores LLM query-response pairs and retrieves them
    based on semantic similarity rather than exact string matching.

    The cache uses:
    - Embedding engine to convert queries to vector representations
    - Vector database to store and search embeddings efficiently
    - LLM provider to generate responses for cache misses

    Recommended thresholds (from evaluation - use similarity_to_distance_threshold() to convert):
    - MPNet-Base: 0.78 similarity → 0.22 distance (best balanced model)
    - RoBERTa-Large: 0.77 similarity → 0.23 distance (highest F1)
    - MiniLM-L6: 0.74 similarity → 0.26 distance (fast baseline)
    See embeddings/eval/MODELS_OVERVIEW.md for full evaluation results.
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "gemini-2.5-flash",
        similarity_threshold: float = 0.3,
        collection_name: str = "semantic_cache",
        persist_directory: str = "./data/chroma_db",
        llm_mock_mode: bool = False,
        enable_preprocessing: bool = True,
        verbose: bool = True
    ):
        """
        Initialize the semantic cache.

        Args:
            embedding_model: Sentence transformer model name for embeddings
            llm_model: Gemini model name to use for generating responses
            similarity_threshold: Cosine distance threshold for cache hits
                                 (lower distance = more similar)
                                 Use similarity_to_distance_threshold() to convert from
                                 evaluated similarity thresholds.
                                 Examples:
                                 - 0.22 (from MPNet-Base eval: 1 - 0.78)
                                 - 0.23 (from RoBERTa-Large eval: 1 - 0.77)
                                 - 0.26 (from MiniLM-L6 eval: 1 - 0.74)
            collection_name: ChromaDB collection name
            persist_directory: Directory for persistent storage
            llm_mock_mode: If True, LLM returns "LLM called" instead of real responses
            enable_preprocessing: Whether to enable text preprocessing for better matching
            verbose: Whether to print detailed logs
        """
        self.similarity_threshold = similarity_threshold
        self.verbose = verbose
        self.llm_mock_mode = llm_mock_mode

        # Statistics tracking
        self.stats = {
            "total_queries": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "llm_calls": 0
        }

        if self.verbose:
            print("=" * 70)
            print("Initializing Semantic Cache")
            print("=" * 70)
            print(f"Similarity threshold: {similarity_threshold} (cosine distance)")
            print(f"  Note: Lower distance = more similar (0=identical, 2=opposite)")
            print(f"LLM mock mode: {llm_mock_mode}")
            print(f"Text preprocessing: {enable_preprocessing}\n")

        # Initialize embedding engine
        if self.verbose:
            print("1. Loading embedding engine...")
        self.embedder = EmbeddingEngine(
            model_name=embedding_model,
            enable_preprocessing=enable_preprocessing
        )

        # Initialize vector database
        if self.verbose:
            print("\n2. Initializing vector database...")
        self.vector_store = ChromaVectorStore(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_dimension=self.embedder.get_embedding_dimension()
        )

        # Initialize LLM provider
        if self.verbose:
            print("\n3. Initializing LLM provider...")
        self.llm_provider = GeminiProvider(
            model_name=llm_model,
            mock_mode=llm_mock_mode
        )

        if self.verbose:
            print("\n" + "=" * 70)
            print("Semantic Cache Ready!")
            print("=" * 70 + "\n")

    def get_response(
        self,
        query: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        force_llm: bool = False
    ) -> str:
        """
        Get response for a query, using cache if available or querying LLM if not.

        This is the main method for interacting with the semantic cache.

        Args:
            query: User query string
            temperature: LLM temperature parameter (only used on cache miss)
            max_tokens: Maximum tokens to generate (only used on cache miss)
            force_llm: If True, skip cache and always query LLM

        Returns:
            str: Response text (either from cache or LLM)
        """
        self.stats["total_queries"] += 1

        if self.verbose:
            print(f"\n{'=' * 70}")
            print(f"Query #{self.stats['total_queries']}: {query}")
            print(f"{'=' * 70}")

        # Check if we should skip cache
        if force_llm:
            if self.verbose:
                print("Force LLM mode enabled - skipping cache lookup")
            return self._query_llm_and_cache(query, temperature, max_tokens)

        # Try to find similar query in cache
        cache_result = self._check_cache(query)

        if cache_result is not None:
            # Cache hit
            response, similarity_distance, matched_query = cache_result
            self.stats["cache_hits"] += 1

            if self.verbose:
                print(f"✓ CACHE HIT (distance: {similarity_distance:.4f})")
                print(f"  Matched query: '{matched_query}'")
                print(f"  Returning cached response")

            return response
        else:
            # Cache miss
            self.stats["cache_misses"] += 1

            if self.verbose:
                print("✗ CACHE MISS - Querying LLM...")

            return self._query_llm_and_cache(query, temperature, max_tokens)

    def _check_cache(self, query: str) -> Optional[Tuple[str, float, str]]:
        """
        Check if a semantically similar query exists in the cache.

        Args:
            query: Query string to search for

        Returns:
            Tuple of (response, distance, matched_query) if cache hit, None otherwise
        """
        # Generate embedding for query
        query_embedding = self.embedder.encode(query)

        # Search vector database
        results = self.vector_store.query(
            query_embedding=query_embedding,
            n_results=1,
            update_access_time=True
        )

        # Check if we have results and if they meet the similarity threshold
        if results['ids'] and len(results['ids']) > 0:
            distance = results['distances'][0]
            metadata = results['metadatas'][0]
            matched_query = results['documents'][0]

            # Lower distance = more similar (cosine distance)
            if distance < self.similarity_threshold:
                # Cache hit
                llm_response = metadata.get('llm_response')

                if llm_response is not None:
                    return (llm_response, distance, matched_query)

        # No similar query found or response missing
        return None

    def _query_llm_and_cache(
        self,
        query: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Query the LLM and store the response in cache.

        Args:
            query: User query string
            temperature: LLM temperature parameter
            max_tokens: Maximum tokens to generate

        Returns:
            str: LLM response
        """
        # Query LLM
        try:
            response = self.llm_provider.generate(
                prompt=query,
                temperature=temperature,
                max_tokens=max_tokens
            )
            self.stats["llm_calls"] += 1

            if self.verbose:
                print(f"  LLM responded ({len(response)} chars)")

        except Exception as e:
            error_msg = f"Error querying LLM: {str(e)}"
            if self.verbose:
                print(f"  ✗ {error_msg}")
            raise Exception(error_msg)

        # Store in cache
        self._add_to_cache(query, response)

        return response

    def _add_to_cache(self, query: str, llm_response: str) -> str:
        """
        Add a query-response pair to the cache.

        Args:
            query: Query string
            llm_response: LLM response string

        Returns:
            str: Document ID of the cached entry
        """
        # Generate embedding
        embedding = self.embedder.encode(query)

        # Add to vector store
        doc_id = self.vector_store.add_embedding(
            embedding=embedding,
            text=query,
            llm_response=llm_response
        )

        if self.verbose:
            print(f"  Cached with ID: {doc_id}")

        return doc_id

    def clear_cache(self) -> None:
        """
        Clear all entries from the cache.
        """
        self.vector_store.clear_collection()

        if self.verbose:
            print("Cache cleared successfully")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary containing:
                - total_queries: Total number of queries processed
                - cache_hits: Number of cache hits
                - cache_misses: Number of cache misses
                - llm_calls: Number of LLM API calls made
                - hit_rate: Cache hit rate as a percentage
                - total_cached_entries: Total entries in cache
                - similarity_threshold: Current similarity threshold
        """
        hit_rate = (
            (self.stats["cache_hits"] / self.stats["total_queries"] * 100)
            if self.stats["total_queries"] > 0
            else 0.0
        )

        cache_stats = self.vector_store.get_collection_stats()

        return {
            **self.stats,
            "hit_rate": round(hit_rate, 2),
            "total_cached_entries": cache_stats["total_embeddings"],
            "similarity_threshold": self.similarity_threshold,
            "llm_mock_mode": self.llm_mock_mode
        }

    def print_stats(self) -> None:
        """
        Print formatted cache statistics.
        """
        stats = self.get_stats()

        print("\n" + "=" * 70)
        print("SEMANTIC CACHE STATISTICS")
        print("=" * 70)
        print(f"Total Queries:        {stats['total_queries']}")
        print(f"Cache Hits:           {stats['cache_hits']}")
        print(f"Cache Misses:         {stats['cache_misses']}")
        print(f"LLM Calls:            {stats['llm_calls']}")
        print(f"Hit Rate:             {stats['hit_rate']:.2f}%")
        print(f"Cached Entries:       {stats['total_cached_entries']}")
        print(f"Similarity Threshold: {stats['similarity_threshold']} (cosine distance)")
        print(f"LLM Mock Mode:        {stats['llm_mock_mode']}")
        print("=" * 70 + "\n")

    def update_similarity_threshold(self, new_threshold: float) -> None:
        """
        Update the similarity threshold for cache hits.

        Args:
            new_threshold: New cosine distance threshold (0.0 to 2.0)
                          Lower values require more similarity for cache hits
                          Typical values: 0.1 (strict) to 0.5 (lenient)
        """
        if not 0.0 <= new_threshold <= 2.0:
            raise ValueError("Similarity threshold must be between 0.0 and 2.0")

        self.similarity_threshold = new_threshold

        if self.verbose:
            print(f"Similarity threshold updated to: {new_threshold}")

    def __repr__(self) -> str:
        """String representation of the semantic cache."""
        stats = self.get_stats()
        return (
            f"SemanticCache("
            f"entries={stats['total_cached_entries']}, "
            f"threshold={self.similarity_threshold}, "
            f"hit_rate={stats['hit_rate']:.1f}%)"
        )


# Example usage and testing
if __name__ == "__main__":
    import shutil

    print("\n" + "=" * 70)
    print("SEMANTIC CACHE - Example Usage")
    print("=" * 70 + "\n")

    # Use test directory for examples (not production data/chroma_db)
    test_db_path = "./data/chroma_db_test"

    # Convert evaluated similarity threshold to distance threshold
    # MPNet-Base evaluation found optimal similarity threshold = 0.78
    # See: embeddings/eval/MODELS_OVERVIEW.md
    mpnet_similarity_threshold = 0.78
    distance_threshold = similarity_to_distance_threshold(mpnet_similarity_threshold)
    print(f"Using MPNet-Base with evaluated threshold:")
    print(f"  Cosine similarity: {mpnet_similarity_threshold}")
    print(f"  Cosine distance:   {distance_threshold}\n")

    # Initialize cache in mock mode (no real LLM calls)
    cache = SemanticCache(
        embedding_model="sentence-transformers/all-mpnet-base-v2",
        similarity_threshold=distance_threshold,
        llm_mock_mode=True,
        verbose=True,
        persist_directory=test_db_path
    )

    # Example 1: First query (cache miss)
    print("\n--- Example 1: First Query ---")
    query1 = "What is machine learning?"
    response1 = cache.get_response(query1)
    print(f"\nResponse: {response1}")

    # Example 2: Similar query (should be cache hit)
    print("\n--- Example 2: Similar Query ---")
    query2 = "Can you explain machine learning?"
    response2 = cache.get_response(query2)
    print(f"\nResponse: {response2}")

    # Example 3: Another similar query
    print("\n--- Example 3: Another Similar Query ---")
    query3 = "what is ML?"
    response3 = cache.get_response(query3)
    print(f"\nResponse: {response3}")

    # Example 4: Different query (cache miss)
    print("\n--- Example 4: Different Query ---")
    query4 = "What is the capital of France?"
    response4 = cache.get_response(query4)
    print(f"\nResponse: {response4}")

    # Example 5: Force LLM (skip cache)
    print("\n--- Example 5: Force LLM Call ---")
    query5 = "What is machine learning?"
    response5 = cache.get_response(query5, force_llm=True)
    print(f"\nResponse: {response5}")

    # Print final statistics
    cache.print_stats()

    # Example 6: Update threshold and test
    print("\n--- Example 6: Update Threshold (More Strict) ---")
    cache.update_similarity_threshold(0.1)  # More strict (lower threshold)
    query6 = "Tell me about ML"
    response6 = cache.get_response(query6)
    print(f"\nResponse: {response6}")

    # Final statistics
    cache.print_stats()

    print("=" * 70)
    print("Example usage completed!")
    print("=" * 70 + "\n")

    # Cleanup: Remove test database to avoid interference with subsequent runs
    print("Cleaning up test database...")
    if os.path.exists(test_db_path):
        shutil.rmtree(test_db_path)
        print(f"✓ Removed test database at: {test_db_path}\n")
    else:
        print(f"✓ Test database already clean\n")
