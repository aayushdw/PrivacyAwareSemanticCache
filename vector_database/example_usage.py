"""
Example usage of ChromaVectorStore for semantic similarity search.
Demonstrates integration with the EmbeddingEngine for a complete workflow.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embeddings.embedding_engine import EmbeddingEngine
from vector_database.chroma_store import ChromaVectorStore
import numpy as np


def example_single_embedding():
    """Example: Add and query a single embedding."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Single Embedding")
    print("="*70 + "\n")

    # Initialize embedding engine
    print("Initializing embedding engine...")
    embedder = EmbeddingEngine(model_name='all-MiniLM-L6-v2')

    # Initialize vector store
    print("\nInitializing ChromaDB vector store...")
    vector_store = ChromaVectorStore(
        collection_name="semantic_cache",
        persist_directory="./data/chroma_db",
        embedding_dimension=embedder.get_embedding_dimension()
    )

    # Sample query and response
    query_text = "What is the capital of France?"
    llm_response = "The capital of France is Paris."

    # Generate embedding
    print(f"\nGenerating embedding for: '{query_text}'")
    embedding = embedder.encode(query_text)
    print(f"Embedding shape: {embedding.shape}")

    # Add to vector store
    print("\nAdding embedding to vector store...")
    doc_id = vector_store.add_embedding(
        embedding=embedding,
        text=query_text,
        llm_response=llm_response
    )
    print(f"Document added with ID: {doc_id}")

    # Query with similar text
    similar_query = "What is the capital city of france?"
    print(f"\nQuerying with similar text: '{similar_query}'")
    query_embedding = embedder.encode(similar_query)

    results = vector_store.query(
        query_embedding=query_embedding,
        n_results=1
    )

    print("\nQuery Results:")
    print(f"  - Distance: {results['distances'][0]:.4f} (cosine)")
    print(f"  - Original text: {results['documents'][0]}")
    print(f"  - LLM response: {results['metadatas'][0].get('llm_response', 'N/A')}")
    print(f"  - Cache hits: {results['metadatas'][0].get('cache_hit_count', 0)}")


def example_batch_embeddings():
    """Example: Add and query multiple embeddings in batch."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Batch Embeddings")
    print("="*70 + "\n")

    # Initialize components
    embedder = EmbeddingEngine(model_name='all-MiniLM-L6-v2')
    vector_store = ChromaVectorStore(
        collection_name="batch_cache",
        persist_directory="./data/chroma_db",
        embedding_dimension=embedder.get_embedding_dimension()
    )

    # Sample queries and responses
    queries = [
        "What is machine learning?",
        "How does neural network work?",
        "Explain deep learning",
        "What is artificial intelligence?",
        "What is the weather like today?"
    ]

    responses = [
        "Machine learning is a subset of AI that enables systems to learn from data.",
        "Neural networks work by processing data through layers of interconnected nodes.",
        "Deep learning uses multi-layered neural networks to learn complex patterns.",
        "Artificial intelligence is the simulation of human intelligence by machines.",
        "I cannot provide real-time weather information."
    ]

    # Generate embeddings in batch
    print(f"Generating embeddings for {len(queries)} queries...")
    embeddings = embedder.encode_batch(queries, show_progress_bar=False)
    print(f"Embeddings shape: {embeddings.shape}")

    # Add to vector store in batch
    print("\nAdding embeddings to vector store...")
    doc_ids = vector_store.add_embeddings_batch(
        embeddings=embeddings,
        texts=queries,
        llm_responses=responses
    )
    print(f"Added {len(doc_ids)} documents")

    # Query with a related question
    test_query = "Tell me about AI and machine learning"
    print(f"\nQuerying with: '{test_query}'")
    query_embedding = embedder.encode(test_query)

    results = vector_store.query(
        query_embedding=query_embedding,
        n_results=3
    )

    print("\nTop 3 Results:")
    for i, (dist, doc, metadata) in enumerate(zip(
        results['distances'],
        results['documents'],
        results['metadatas']
    ), 1):
        print(f"\n{i}. Distance: {dist:.4f}")
        print(f"   Query: {doc}")
        print(f"   Response: {metadata.get('llm_response', 'N/A')}")
        print(f"   Stored: {metadata.get('timestamp_stored', 'N/A')}")


def example_cache_simulation():
    """Example: Simulate a semantic cache with hit tracking."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Semantic Cache Simulation")
    print("="*70 + "\n")

    # Initialize components
    embedder = EmbeddingEngine(model_name='all-MiniLM-L6-v2')
    vector_store = ChromaVectorStore(
        collection_name="cache_simulation",
        persist_directory="./data/chroma_db",
        embedding_dimension=embedder.get_embedding_dimension()
    )

    # Clear previous data
    vector_store.clear_collection()

    # Add some cached queries
    cached_queries = [
        "What is Python?",
        "How to use lists in Python?",
        "What are Python dictionaries?"
    ]

    cached_responses = [
        "Python is a high-level programming language.",
        "Lists in Python are ordered, mutable collections.",
        "Dictionaries are key-value pair collections in Python."
    ]

    print("Populating cache with 3 queries...")
    embeddings = embedder.encode_batch(cached_queries, show_progress_bar=False)
    vector_store.add_embeddings_batch(
        embeddings=embeddings,
        texts=cached_queries,
        llm_responses=cached_responses
    )

    # Simulate cache lookups
    test_queries = [
        "what is python programming language?",  # Similar to cached
        "tell me about python",                    # Similar to cached
        "explain python lists",                    # Similar to cached
        "what is javascript?",                     # Not similar
    ]

    similarity_threshold = 0.5  # Distance threshold for cache hit

    print("\nSimulating cache lookups:")
    print("-" * 70)

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        query_embedding = embedder.encode(query)

        results = vector_store.query(
            query_embedding=query_embedding,
            n_results=1,
            update_access_time=True
        )

        if results['ids'] and results['distances'][0] < similarity_threshold:
            print(f"  ✓ CACHE HIT (distance: {results['distances'][0]:.4f})")
            print(f"    Matched: '{results['documents'][0]}'")
            print(f"    Response: {results['metadatas'][0].get('llm_response', 'N/A')}")
            print(f"    Total hits: {results['metadatas'][0].get('cache_hit_count', 0)}")
        else:
            distance = results['distances'][0] if results['distances'] else float('inf')
            print(f"  ✗ CACHE MISS (distance: {distance:.4f})")
            print(f"    Would need to query LLM")

    # Show collection stats
    print("\n" + "="*70)
    stats = vector_store.get_collection_stats()
    print("\nCollection Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


def example_vector_operations():
    """Example: Various vector database operations."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Vector Database Operations")
    print("="*70 + "\n")

    embedder = EmbeddingEngine(model_name='all-MiniLM-L6-v2')
    vector_store = ChromaVectorStore(
        collection_name="operations_demo",
        persist_directory="./data/chroma_db",
        embedding_dimension=embedder.get_embedding_dimension()
    )

    # Clear collection
    print("Clearing collection...")
    vector_store.clear_collection()

    # Add some data
    texts = ["Hello world", "Machine learning", "Data science"]
    embeddings = embedder.encode_batch(texts, show_progress_bar=False)
    doc_ids = vector_store.add_embeddings_batch(
        embeddings=embeddings,
        texts=texts,
        llm_responses=["Response 1", "Response 2", "Response 3"]
    )

    print(f"Added {len(doc_ids)} documents")
    print(f"Document IDs: {doc_ids}")

    # Get by IDs
    print("\nRetrieving documents by ID...")
    retrieved = vector_store.get_by_ids([doc_ids[0], doc_ids[1]])
    print(f"Retrieved {len(retrieved['ids'])} documents")
    for doc_id, text in zip(retrieved['ids'], retrieved['documents']):
        print(f"  - {doc_id}: {text}")

    # Delete specific documents
    print(f"\nDeleting document: {doc_ids[0]}")
    vector_store.delete_by_ids([doc_ids[0]])

    # Check remaining count
    stats = vector_store.get_collection_stats()
    print(f"Remaining documents: {stats['total_embeddings']}")

    # Display vector store info
    print(f"\n{vector_store}")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("ChromaDB Vector Store - Example Usage")
    print("="*70)

    try:
        example_single_embedding()
        example_batch_embeddings()
        example_cache_simulation()
        example_vector_operations()

        print("\n" + "="*70)
        print("All examples completed successfully!")
        print("="*70 + "\n")

        

    except Exception as e:
        print(f"\nError occurred: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
