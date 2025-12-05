"""
Example usage of ChromaVectorStore for semantic similarity search.
Demonstrates integration with both pre-trained and LoRA fine-tuned models.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embeddings.embedding_engine import EmbeddingEngine
from vector_database.chroma_store import ChromaVectorStore
import numpy as np


class LoRAEmbeddingEngine:
    """
    Embedding engine using LoRA fine-tuned MiniLM model.
    Loads the model trained via lora_training_pipeline/train_minilm_lora.py
    """

    def __init__(self, model_path: str, device: str = None):
        """
        Initialize the LoRA embedding engine.

        Args:
            model_path: Path to the LoRA model directory (e.g., 'lora_training_pipeline/models/best_model')
            device: Device to use ('cuda', 'mps', 'cpu', or None for auto-detect)
        """
        # Import dependencies
        import json
        import torch
        from transformers import AutoTokenizer, AutoModel
        from peft import PeftModel

        self.model_path = model_path
        self.torch = torch
        self.nn = torch.nn

        # Setup device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        print(f"Loading LoRA model from: {model_path}")
        print(f"Using device: {self.device}")

        # Load metadata
        metadata_path = os.path.join(model_path, 'metadata.json')
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        print(f"Model trained on: {self.metadata['training_date']}")
        print(f"Optimal threshold: {self.metadata['optimal_threshold']:.4f}")
        print(f"F1 Score: {self.metadata['metrics']['f1']:.4f}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Load base model
        base_model = AutoModel.from_pretrained(self.metadata['model_name'])

        # Load LoRA adapter
        self.model = PeftModel.from_pretrained(base_model, model_path)
        self.model.to(self.device)
        self.model.eval()

        print(f"LoRA model loaded successfully. Embedding dimension: {self.get_embedding_dimension()}")

    def mean_pooling(self, model_output, attention_mask):
        """Mean pooling to get sentence embeddings."""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return self.torch.sum(token_embeddings * input_mask_expanded, 1) / self.torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def encode(self, text: str, max_length: int = 128) -> np.ndarray:
        """
        Encode a single text into embedding.

        Args:
            text: Text to encode
            max_length: Maximum sequence length

        Returns:
            numpy array of shape (embedding_dim,)
        """
        # Tokenize
        encoded = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )

        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)

        # Generate embedding
        with self.torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = self.mean_pooling(outputs, attention_mask)
            # Normalize embeddings
            embeddings = self.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy()[0]

    def encode_batch(self, texts: list, max_length: int = 128, batch_size: int = 32) -> np.ndarray:
        """
        Encode a batch of texts into embeddings.

        Args:
            texts: List of texts to encode
            max_length: Maximum sequence length
            batch_size: Batch size for processing

        Returns:
            numpy array of shape (num_texts, embedding_dim)
        """
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # Tokenize batch
            encoded = self.tokenizer(
                batch_texts,
                padding='max_length',
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )

            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)

            # Generate embeddings
            with self.torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                embeddings = self.mean_pooling(outputs, attention_mask)
                # Normalize embeddings
                embeddings = self.nn.functional.normalize(embeddings, p=2, dim=1)

            all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings)

    def get_embedding_dimension(self) -> int:
        """Get the dimensionality of the embeddings."""
        return self.model.config.hidden_size

    def get_optimal_threshold(self) -> float:
        """
        Get the optimal similarity threshold from training metadata.

        Returns:
            Cosine similarity threshold (for use with dot product of normalized embeddings)
        """
        return self.metadata['optimal_threshold']

    def get_optimal_distance_threshold(self) -> float:
        """
        Get the optimal distance threshold for ChromaDB queries.

        ChromaDB uses cosine distance (1 - cosine_similarity), so we need to convert
        the similarity threshold from training.

        Returns:
            Cosine distance threshold (for use with ChromaDB query results)
        """
        # Convert similarity threshold to distance threshold
        # distance = 1 - similarity
        return 1.0 - self.metadata['optimal_threshold']


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


def example_lora_single_embedding():
    """Example: Use LoRA-trained model for single embedding."""
    print("\n" + "="*70)
    print("EXAMPLE 5: LoRA Model - Single Embedding")
    print("="*70 + "\n")

    # Get path to LoRA model
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    lora_model_path = os.path.join(project_root, 'lora_training_pipeline', 'models', 'best_model')

    # Check if model exists
    if not os.path.exists(lora_model_path):
        print(f"LoRA model not found at {lora_model_path}")
        print("Please train the model first using lora_training_pipeline/train_minilm_lora.py")
        return

    # Initialize LoRA embedding engine
    print("Initializing LoRA embedding engine...")
    embedder = LoRAEmbeddingEngine(model_path=lora_model_path)

    # Initialize vector store
    print("\nInitializing ChromaDB vector store...")
    vector_store = ChromaVectorStore(
        collection_name="lora_semantic_cache",
        persist_directory="./data/chroma_db_lora",
        embedding_dimension=embedder.get_embedding_dimension()
    )

    # Sample question pairs (similar questions)
    query1 = "How do I reset my password?"
    query2 = "What is the process for password reset?"
    llm_response = "To reset your password, go to Settings > Security > Reset Password."

    # Generate embeddings
    print(f"\nGenerating embedding for: '{query1}'")
    embedding1 = embedder.encode(query1)
    print(f"Embedding shape: {embedding1.shape}")

    # Add to vector store
    print("\nAdding embedding to vector store...")
    doc_id = vector_store.add_embedding(
        embedding=embedding1,
        text=query1,
        llm_response=llm_response
    )
    print(f"Document added with ID: {doc_id}")

    # Query with similar question
    print(f"\nQuerying with similar question: '{query2}'")
    query_embedding = embedder.encode(query2)

    results = vector_store.query(
        query_embedding=query_embedding,
        n_results=1
    )

    optimal_similarity = embedder.get_optimal_threshold()
    optimal_distance = embedder.get_optimal_distance_threshold()
    print(f"\nOptimal similarity threshold (from training): {optimal_similarity:.4f}")
    print(f"Optimal distance threshold (for ChromaDB):    {optimal_distance:.4f}")
    print("\nQuery Results:")
    print(f"  - Distance: {results['distances'][0]:.4f} (cosine)")
    print(f"  - Original text: {results['documents'][0]}")
    print(f"  - LLM response: {results['metadatas'][0].get('llm_response', 'N/A')}")

    # Check if it's a cache hit using the optimal threshold
    if results['distances'][0] < optimal_distance:
        print(f"  - Status: ✓ CACHE HIT (distance {results['distances'][0]:.4f} < {optimal_distance:.4f})")
    else:
        print(f"  - Status: ✗ CACHE MISS (distance {results['distances'][0]:.4f} >= {optimal_distance:.4f})")


def example_lora_batch_embeddings():
    """Example: Use LoRA-trained model for batch embeddings."""
    print("\n" + "="*70)
    print("EXAMPLE 6: LoRA Model - Batch Embeddings")
    print("="*70 + "\n")

    # Get path to LoRA model
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    lora_model_path = os.path.join(project_root, 'lora_training_pipeline', 'models', 'best_model')

    # Check if model exists
    if not os.path.exists(lora_model_path):
        print(f"LoRA model not found at {lora_model_path}")
        print("Please train the model first using lora_training_pipeline/train_minilm_lora.py")
        return

    # Initialize components
    embedder = LoRAEmbeddingEngine(model_path=lora_model_path)
    vector_store = ChromaVectorStore(
        collection_name="lora_batch_cache",
        persist_directory="./data/chroma_db_lora",
        embedding_dimension=embedder.get_embedding_dimension()
    )

    # Sample question pairs (questions that might appear in semantic cache)
    queries = [
        "How do I change my email address?",
        "What is the procedure to update my email?",
        "How can I reset my account password?",
        "What are the steps to change password?",
        "How do I delete my account?",
    ]

    responses = [
        "To change your email, go to Settings > Profile > Email and enter your new email.",
        "To update your email, navigate to Settings > Profile > Email.",
        "To reset your password, go to Settings > Security > Reset Password.",
        "To change your password, visit Settings > Security and click Change Password.",
        "To delete your account, go to Settings > Account > Delete Account.",
    ]

    # Generate embeddings in batch
    print(f"Generating embeddings for {len(queries)} queries...")
    embeddings = embedder.encode_batch(queries)
    print(f"Embeddings shape: {embeddings.shape}")

    # Add to vector store in batch
    print("\nAdding embeddings to vector store...")
    doc_ids = vector_store.add_embeddings_batch(
        embeddings=embeddings,
        texts=queries,
        llm_responses=responses
    )
    print(f"Added {len(doc_ids)} documents")

    # Test queries (semantically similar to cached queries)
    test_queries = [
        "email change process",  # Similar to queries 0 and 1
        "password reset steps",  # Similar to queries 2 and 3
        "how to remove my account",  # Similar to query 4
    ]

    optimal_similarity = embedder.get_optimal_threshold()
    optimal_distance = embedder.get_optimal_distance_threshold()
    print(f"\nOptimal similarity threshold (from training): {optimal_similarity:.4f}")
    print(f"Optimal distance threshold (for ChromaDB):    {optimal_distance:.4f}")
    print("\nTesting semantic cache with similar queries:")
    print("-" * 70)

    for test_query in test_queries:
        print(f"\nQuery: '{test_query}'")
        query_embedding = embedder.encode(test_query)

        results = vector_store.query(
            query_embedding=query_embedding,
            n_results=1
        )

        if results['ids']:
            distance = results['distances'][0]
            is_hit = distance < optimal_distance

            print(f"  {'✓ CACHE HIT' if is_hit else '✗ CACHE MISS'} (distance: {distance:.4f}, threshold: {optimal_distance:.4f})")
            print(f"  Matched: '{results['documents'][0]}'")
            print(f"  Response: {results['metadatas'][0].get('llm_response', 'N/A')}")


def example_lora_comparison():
    """Example: Compare LoRA model with pre-trained model."""
    print("\n" + "="*70)
    print("EXAMPLE 7: Comparison - LoRA vs Pre-trained Model")
    print("="*70 + "\n")

    # Get path to LoRA model
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    lora_model_path = os.path.join(project_root, 'lora_training_pipeline', 'models', 'best_model')

    # Check if model exists
    if not os.path.exists(lora_model_path):
        print(f"LoRA model not found at {lora_model_path}")
        print("Please train the model first using lora_training_pipeline/train_minilm_lora.py")
        return

    # Initialize both embedding engines
    print("Initializing pre-trained model...")
    pretrained_embedder = EmbeddingEngine(model_name='all-MiniLM-L6-v2')

    print("\nInitializing LoRA fine-tuned model...")
    lora_embedder = LoRAEmbeddingEngine(model_path=lora_model_path)

    # Test question pairs
    question_pairs = [
        ("How do I reset my password?", "What is the password reset process?"),
        ("How can I change my email?", "What is the procedure to update email?"),
        ("How do I delete my account?", "What are the steps to remove my account?"),
    ]

    print("\nComparing similarity scores for question pairs:")
    print("-" * 70)

    for q1, q2 in question_pairs:
        # Pre-trained model
        emb1_pretrained = pretrained_embedder.encode(q1)
        emb2_pretrained = pretrained_embedder.encode(q2)
        similarity_pretrained = np.dot(emb1_pretrained, emb2_pretrained)

        # LoRA model
        emb1_lora = lora_embedder.encode(q1)
        emb2_lora = lora_embedder.encode(q2)
        similarity_lora = np.dot(emb1_lora, emb2_lora)

        print(f"\nQ1: '{q1}'")
        print(f"Q2: '{q2}'")
        print(f"  Pre-trained similarity: {similarity_pretrained:.4f}")
        print(f"  LoRA similarity:        {similarity_lora:.4f}")
        print(f"  Difference:             {similarity_lora - similarity_pretrained:+.4f}")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("ChromaDB Vector Store - Example Usage")
    print("="*70)

    try:
        # Pre-trained model examples
        example_single_embedding()
        example_batch_embeddings()
        example_cache_simulation()
        example_vector_operations()

        # LoRA model examples
        example_lora_single_embedding()
        example_lora_batch_embeddings()
        example_lora_comparison()

        print("\n" + "="*70)
        print("All examples completed successfully!")
        print("="*70 + "\n")

    except Exception as e:
        print(f"\nError occurred: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup: Remove test databases
        print("\n" + "="*70)
        print("Cleaning up test databases...")
        print("="*70 + "\n")

        import shutil

        # Remove pre-trained model database
        db_path = "./data/chroma_db"
        if os.path.exists(db_path):
            try:
                shutil.rmtree(db_path)
                print(f"✓ Removed pre-trained model database: {db_path}")
            except Exception as e:
                print(f"✗ Failed to remove {db_path}: {e}")

        # Remove LoRA model database
        lora_db_path = "./data/chroma_db_lora"
        if os.path.exists(lora_db_path):
            try:
                shutil.rmtree(lora_db_path)
                print(f"✓ Removed LoRA model database: {lora_db_path}")
            except Exception as e:
                print(f"✗ Failed to remove {lora_db_path}: {e}")

        print("\nCleanup complete!")


if __name__ == "__main__":
    main()
