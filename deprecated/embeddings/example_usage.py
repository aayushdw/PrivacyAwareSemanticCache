"""
Example usage of the EmbeddingEngine class.
Demonstrates encoding single text and batches of text.
"""

from embeddings.embedding_engine import EmbeddingEngine
import numpy as np


def main():
    engine = EmbeddingEngine()
    print(f"\n{engine}\n")

    # Example 1: Encode a single text
    print("=" * 60)
    print("Example 1: Encoding a single text")
    print("=" * 60)
    text = "How do I recover my Gmail password?"
    embedding = engine.encode(text)
    print(f"Text: {text}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding (first 10 dimensions): {embedding[:10]}")
    print(f"Embedding norm: {np.linalg.norm(embedding):.4f}")
    print()

    # Example 2: Encode multiple texts
    print("=" * 60)
    print("Example 2: Encoding multiple texts")
    print("=" * 60)
    texts = [
        "How do I recover my Gmail password?",
        "What steps should I take to reset my Gmail password?",
        "Why am I angry with myself?",
        "I am extremely angry right now."
    ]
    embeddings = engine.encode_batch(texts, show_progress_bar=False)
    print(f"Number of texts: {len(texts)}")
    print(f"Embeddings shape: {embeddings.shape}")
    print()

    # Example 3: Compute similarity between texts
    print("=" * 60)
    print("Example 3: Computing semantic similarity")
    print("=" * 60)

    # Compute cosine similarity (dot product for normalized vectors)
    similarity_matrix = np.dot(embeddings, embeddings.T)

    print("Similarity matrix:")
    print(f"{'':40} ", end="")
    for i in range(len(texts)):
        print(f"Text{i+1:1} ", end="")
    print()

    for i, text in enumerate(texts):
        print(f"{text[:40]:40} ", end="")
        for j in range(len(texts)):
            print(f"{similarity_matrix[i][j]:5.2f} ", end="")
        print()


if __name__ == "__main__":
    main()
