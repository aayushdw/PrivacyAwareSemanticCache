"""
Example usage of LoRA-trained model with ChromaVectorStore.
This script demonstrates how to use the LoRA fine-tuned model
trained via lora_training_pipeline/train_minilm_lora.py
"""

import sys
import os
import json

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
            model_path: Path to the LoRA model directory
            device: Device to use ('cuda', 'mps', 'cpu', or None for auto-detect)
        """
        # Import dependencies
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
        encoded = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )

        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)

        with self.torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = self.mean_pooling(outputs, attention_mask)
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

            encoded = self.tokenizer(
                batch_texts,
                padding='max_length',
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )

            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)

            with self.torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                embeddings = self.mean_pooling(outputs, attention_mask)
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


def example_single_query():
    """Example: Single query with semantic caching."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Single Query - Semantic Cache Lookup")
    print("="*70 + "\n")

    # Get path to LoRA model
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    lora_model_path = os.path.join(project_root, 'lora_training_pipeline', 'models', 'best_model')

    # Initialize LoRA embedding engine
    print("Initializing LoRA embedding engine...")
    embedder = LoRAEmbeddingEngine(model_path=lora_model_path)

    # Initialize vector store
    print("\nInitializing ChromaDB vector store...")
    vector_store = ChromaVectorStore(
        collection_name="lora_demo",
        persist_directory="./data/chroma_db_lora",
        embedding_dimension=embedder.get_embedding_dimension()
    )

    # Clear existing data for clean demo
    vector_store.clear_collection()

    # Sample question pair
    original_query = "How do I reset my password?"
    similar_query = "What is the process for password reset?"
    llm_response = "To reset your password, go to Settings > Security > Reset Password."

    # Add original query to cache
    print(f"\nAdding to cache: '{original_query}'")
    embedding = embedder.encode(original_query)
    vector_store.add_embedding(
        embedding=embedding,
        text=original_query,
        llm_response=llm_response
    )

    # Query with similar question
    print(f"\nQuerying with: '{similar_query}'")
    query_embedding = embedder.encode(similar_query)
    results = vector_store.query(query_embedding=query_embedding, n_results=1)

    optimal_similarity = embedder.get_optimal_threshold()
    optimal_distance = embedder.get_optimal_distance_threshold()
    distance = results['distances'][0]

    print(f"\nResults:")
    print(f"  Optimal similarity threshold (from training): {optimal_similarity:.4f}")
    print(f"  Optimal distance threshold (for ChromaDB):    {optimal_distance:.4f}")
    print(f"  Actual cosine distance:                       {distance:.4f}")
    print(f"  Status: {'✓ CACHE HIT' if distance < optimal_distance else '✗ CACHE MISS'}")
    print(f"  Matched query: '{results['documents'][0]}'")
    print(f"  Cached response: {results['metadatas'][0].get('llm_response', 'N/A')}")


def example_batch_queries():
    """Example: Batch queries with semantic caching."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Batch Queries - Building a Semantic Cache")
    print("="*70 + "\n")

    # Get path to LoRA model
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    lora_model_path = os.path.join(project_root, 'lora_training_pipeline', 'models', 'best_model')

    # Initialize components
    embedder = LoRAEmbeddingEngine(model_path=lora_model_path)
    vector_store = ChromaVectorStore(
        collection_name="lora_batch_demo",
        persist_directory="./data/chroma_db_lora",
        embedding_dimension=embedder.get_embedding_dimension()
    )

    # Sample FAQ queries and responses
    faq_queries = [
        "How do I change my email address?",
        "How can I reset my account password?",
        "How do I delete my account?",
        "What are the payment methods accepted?",
        "How do I contact customer support?",
    ]

    faq_responses = [
        "To change your email, go to Settings > Profile > Email and enter your new email.",
        "To reset your password, go to Settings > Security > Reset Password.",
        "To delete your account, go to Settings > Account > Delete Account.",
        "We accept credit cards, debit cards, PayPal, and bank transfers.",
        "You can contact support at support@example.com or call 1-800-SUPPORT.",
    ]

    # Build cache
    print(f"Building semantic cache with {len(faq_queries)} FAQ entries...")
    embeddings = embedder.encode_batch(faq_queries)
    vector_store.add_embeddings_batch(
        embeddings=embeddings,
        texts=faq_queries,
        llm_responses=faq_responses
    )
    print(f"Cache built with {len(faq_queries)} entries\n")

    # Test with user queries (similar but not exact matches)
    test_queries = [
        "email change process",
        "password reset steps",
        "how to remove my account",
        "accepted payment options",
        "reach customer service",
        "how to cancel my subscription",  # Not in cache
    ]

    optimal_similarity = embedder.get_optimal_threshold()
    optimal_distance = embedder.get_optimal_distance_threshold()
    print(f"Optimal similarity threshold (from training): {optimal_similarity:.4f}")
    print(f"Optimal distance threshold (for ChromaDB):    {optimal_distance:.4f}")
    print("\nTesting cache lookups:")
    print("-" * 70)

    for query in test_queries:
        print(f"\nUser query: '{query}'")
        query_embedding = embedder.encode(query)
        results = vector_store.query(query_embedding=query_embedding, n_results=1)

        if results['ids']:
            distance = results['distances'][0]
            is_hit = distance < optimal_distance

            if is_hit:
                print(f"  ✓ CACHE HIT (distance: {distance:.4f} < threshold: {optimal_distance:.4f})")
                print(f"  Matched: '{results['documents'][0]}'")
                print(f"  Response: {results['metadatas'][0].get('llm_response', 'N/A')}")
            else:
                print(f"  ✗ CACHE MISS (distance: {distance:.4f} >= threshold: {optimal_distance:.4f})")
                print(f"  Closest match: '{results['documents'][0]}'")
                print(f"  Would need to query LLM for this question")


def example_embedding_similarity_comparison():
    """Example: Compare ChromaDB distance vs direct embedding similarity computation."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Embedding Similarity Comparison - Direct Similarity Computation")
    print("="*70 + "\n")

    # Get path to LoRA model
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    lora_model_path = os.path.join(project_root, 'lora_training_pipeline', 'models', 'best_model')

    # Initialize components
    embedder = LoRAEmbeddingEngine(model_path=lora_model_path)
    vector_store = ChromaVectorStore(
        collection_name="lora_similarity_demo",
        persist_directory="./data/chroma_db_lora",
        embedding_dimension=embedder.get_embedding_dimension()
    )

    # Sample FAQ queries and responses
    faq_queries = [
        "How do I change my email address?",
        "How can I reset my account password?",
        "How do I delete my account?",
        "What are the payment methods accepted?",
        "How do I contact customer support?",
    ]

    faq_responses = [
        "To change your email, go to Settings > Profile > Email and enter your new email.",
        "To reset your password, go to Settings > Security > Reset Password.",
        "To delete your account, go to Settings > Account > Delete Account.",
        "We accept credit cards, debit cards, PayPal, and bank transfers.",
        "You can contact support at support@example.com or call 1-800-SUPPORT.",
    ]

    # Build cache
    print(f"Building semantic cache with {len(faq_queries)} FAQ entries...")
    embeddings = embedder.encode_batch(faq_queries)
    vector_store.add_embeddings_batch(
        embeddings=embeddings,
        texts=faq_queries,
        llm_responses=faq_responses
    )
    print(f"Cache built with {len(faq_queries)} entries\n")

    # Test queries
    test_queries = [
        "email change process",
        "password reset steps",
        "how to cancel my subscription",  # Not in cache
    ]

    optimal_similarity = embedder.get_optimal_threshold()
    optimal_distance = embedder.get_optimal_distance_threshold()

    print(f"Optimal similarity threshold (from training): {optimal_similarity:.4f}")
    print(f"Optimal distance threshold (for ChromaDB):    {optimal_distance:.4f}")
    print("\nComparing ChromaDB distance vs direct embedding similarity:")
    print("-" * 70)

    for query in test_queries:
        print(f"\nUser query: '{query}'")

        # Get query embedding
        query_embedding = embedder.encode(query)

        # Get closest match from ChromaDB
        results = vector_store.query(query_embedding=query_embedding, n_results=1)

        if results['ids']:
            matched_text = results['documents'][0]
            chromadb_distance = results['distances'][0]

            # Compute similarity using embedding model directly
            # Get the embedding of the matched document from cache
            matched_embedding = embedder.encode(matched_text)

            # Compute cosine similarity (dot product of normalized embeddings)
            computed_similarity = np.dot(query_embedding, matched_embedding)

            # Convert to distance for comparison
            computed_distance = 1.0 - computed_similarity

            # Classify using computed similarity
            is_hit_computed = computed_similarity >= optimal_similarity
            is_hit_chromadb = chromadb_distance < optimal_distance

            print(f"\n  Matched document: '{matched_text}'")
            print(f"\n  ChromaDB Results:")
            print(f"    Distance: {chromadb_distance:.4f}")
            print(f"    Classification: {'✓ CACHE HIT' if is_hit_chromadb else '✗ CACHE MISS'}")

            print(f"\n  Direct Embedding Similarity:")
            print(f"    Cosine similarity: {computed_similarity:.4f}")
            print(f"    Equivalent distance: {computed_distance:.4f}")
            print(f"    Classification: {'✓ CACHE HIT' if is_hit_computed else '✗ CACHE MISS'}")
            print(f"    Above threshold? {computed_similarity:.4f} >= {optimal_similarity:.4f}: {is_hit_computed}")

            # Show response if it's a cache hit
            if is_hit_computed:
                print(f"\n  Cached response: {results['metadatas'][0].get('llm_response', 'N/A')}")
            else:
                print(f"\n  → Would need to query LLM for this question")

            # Verify consistency
            if is_hit_computed != is_hit_chromadb:
                print(f"\n  ⚠ WARNING: Classification mismatch between methods!")
            else:
                print(f"\n  ✓ Both methods agree on classification")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("LoRA Model - Semantic Cache Examples")
    print("Using model from: lora_training_pipeline/models/best_model")
    print("="*70)

    # Check if model exists
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    lora_model_path = os.path.join(project_root, 'lora_training_pipeline', 'models', 'best_model')

    if not os.path.exists(lora_model_path):
        print(f"\nError: LoRA model not found at {lora_model_path}")
        print("Please train the model first using:")
        print("  python3 lora_training_pipeline/train_minilm_lora.py")
        return

    try:
        example_single_query()
        example_batch_queries()
        example_embedding_similarity_comparison()

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

        # Remove LoRA test database
        lora_db_path = "./data/chroma_db_lora"
        if os.path.exists(lora_db_path):
            try:
                shutil.rmtree(lora_db_path)
                print(f"✓ Removed LoRA test database: {lora_db_path}")
            except Exception as e:
                print(f"✗ Failed to remove {lora_db_path}: {e}")

        print("\nCleanup complete!")


if __name__ == "__main__":
    main()
