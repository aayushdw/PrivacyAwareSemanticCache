"""
ChromaDB Vector Store for semantic similarity search.
Provides efficient storage and retrieval of embedding vectors with metadata.
"""

import os
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import numpy as np
import chromadb
from chromadb.config import Settings


class ChromaVectorStore:
    """
    Vector database using ChromaDB for storing and retrieving embeddings.
    Supports cosine similarity-based nearest neighbor search.
    """

    def __init__(
        self,
        collection_name: str = "semantic_cache",
        persist_directory: str = "./data/chroma_db",
        embedding_dimension: Optional[int] = None
    ):
        """
        Initialize ChromaDB vector store.

        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory path for persistent storage
            embedding_dimension: Dimension of embeddings (optional, for validation)
        """
        self.collection_name = collection_name
        self.persist_directory = os.path.abspath(persist_directory)
        self.embedding_dimension = embedding_dimension

        # Create persist directory if it doesn't exist
        os.makedirs(self.persist_directory, exist_ok=True)

        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )

        print(f"ChromaDB initialized at: {self.persist_directory}")
        print(f"Collection: {collection_name}")
        print(f"Total embeddings in collection: {self.collection.count()}")

    def add_embedding(
        self,
        embedding: Union[np.ndarray, List[float]],
        text: str,
        document_id: Optional[str] = None,
        llm_response: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a single embedding to the vector store.

        Args:
            embedding: Embedding vector (numpy array or list)
            text: Original text/query
            document_id: Unique identifier (generated if not provided)
            llm_response: LLM response associated with this query
            metadata: Additional custom metadata

        Returns:
            Document ID of the added embedding
        """
        # Convert numpy array to list if needed
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()

        # Validate embedding dimension
        if self.embedding_dimension is not None:
            if len(embedding) != self.embedding_dimension:
                raise ValueError(
                    f"Embedding dimension mismatch: expected {self.embedding_dimension}, "
                    f"got {len(embedding)}"
                )

        # Generate document ID if not provided
        if document_id is None:
            timestamp = datetime.now().isoformat()
            document_id = f"doc_{timestamp}_{hash(text) % 10**8}"

        # Prepare metadata
        current_time = datetime.now().isoformat()
        doc_metadata = {
            "timestamp_stored": current_time,
            "timestamp_last_access": current_time,
            "cache_hit_count": 0,
            "original_text": text
        }

        # Add LLM response if provided
        if llm_response is not None:
            doc_metadata["llm_response"] = llm_response

        # Add custom metadata if provided
        if metadata is not None:
            doc_metadata.update(metadata)

        # Add to collection
        self.collection.add(
            embeddings=[embedding],
            documents=[text],
            metadatas=[doc_metadata],
            ids=[document_id]
        )

        return document_id

    def add_embeddings_batch(
        self,
        embeddings: Union[np.ndarray, List[List[float]]],
        texts: List[str],
        document_ids: Optional[List[str]] = None,
        llm_responses: Optional[List[str]] = None,
        metadata_list: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """
        Add multiple embeddings to the vector store in batch.

        Args:
            embeddings: List of embedding vectors or numpy array
            texts: List of original texts/queries
            document_ids: List of unique identifiers (generated if not provided)
            llm_responses: List of LLM responses
            metadata_list: List of additional metadata dicts

        Returns:
            List of document IDs
        """
        # Convert numpy array to list if needed
        if isinstance(embeddings, np.ndarray):
            embeddings = embeddings.tolist()

        num_embeddings = len(embeddings)

        # Validate inputs
        if len(texts) != num_embeddings:
            raise ValueError(
                f"Number of texts ({len(texts)}) must match number of embeddings ({num_embeddings})"
            )

        # Validate embedding dimensions
        if self.embedding_dimension is not None:
            for i, emb in enumerate(embeddings):
                if len(emb) != self.embedding_dimension:
                    raise ValueError(
                        f"Embedding {i} dimension mismatch: expected {self.embedding_dimension}, "
                        f"got {len(emb)}"
                    )

        # Generate document IDs if not provided
        if document_ids is None:
            timestamp = datetime.now().isoformat()
            document_ids = [
                f"doc_{timestamp}_{i}_{hash(text) % 10**8}"
                for i, text in enumerate(texts)
            ]

        # Prepare metadata for all documents
        current_time = datetime.now().isoformat()
        metadatas = []

        for i in range(num_embeddings):
            doc_metadata = {
                "timestamp_stored": current_time,
                "timestamp_last_access": current_time,
                "cache_hit_count": 0,
                "original_text": texts[i]
            }

            # Add LLM response if provided
            if llm_responses is not None and i < len(llm_responses):
                if llm_responses[i] is not None:
                    doc_metadata["llm_response"] = llm_responses[i]

            # Add custom metadata if provided
            if metadata_list is not None and i < len(metadata_list):
                if metadata_list[i] is not None:
                    doc_metadata.update(metadata_list[i])

            metadatas.append(doc_metadata)

        # Add to collection
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=document_ids
        )

        return document_ids

    def query(
        self,
        query_embedding: Union[np.ndarray, List[float]],
        n_results: int = 5,
        update_access_time: bool = True
    ) -> Dict[str, Any]:
        """
        Query the vector store for nearest neighbors using cosine similarity.

        Args:
            query_embedding: Query embedding vector
            n_results: Number of nearest neighbors to return
            update_access_time: Whether to update last access time and hit count

        Returns:
            Dictionary containing:
                - ids: List of document IDs
                - distances: List of cosine distances (lower is more similar)
                - documents: List of original texts
                - metadatas: List of metadata dictionaries
                - embeddings: List of embedding vectors
        """
        # Convert numpy array to list if needed
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()

        # Validate embedding dimension
        if self.embedding_dimension is not None:
            if len(query_embedding) != self.embedding_dimension:
                raise ValueError(
                    f"Query embedding dimension mismatch: expected {self.embedding_dimension}, "
                    f"got {len(query_embedding)}"
                )

        # Query the collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["embeddings", "documents", "metadatas", "distances"]
        )

        # Update access time and hit count if requested
        if update_access_time and results['ids'] and len(results['ids'][0]) > 0:
            self._update_access_metadata(results['ids'][0])

        # Format results
        formatted_results = {
            'ids': results['ids'][0] if results['ids'] else [],
            'distances': results['distances'][0] if results['distances'] else [],
            'documents': results['documents'][0] if results['documents'] else [],
            'metadatas': results['metadatas'][0] if results['metadatas'] else [],
            'embeddings': results['embeddings'][0] if results['embeddings'] else []
        }

        return formatted_results

    def _update_access_metadata(self, document_ids: List[str]) -> None:
        """
        Update last access time and increment cache hit count for documents.

        Args:
            document_ids: List of document IDs to update
        """
        if not document_ids:
            return

        # Get current metadata
        results = self.collection.get(
            ids=document_ids,
            include=["metadatas"]
        )

        if not results['metadatas']:
            return

        # Update metadata
        current_time = datetime.now().isoformat()
        updated_metadatas = []

        for metadata in results['metadatas']:
            metadata['timestamp_last_access'] = current_time
            metadata['cache_hit_count'] = metadata.get('cache_hit_count', 0) + 1
            updated_metadatas.append(metadata)

        # Update in collection
        self.collection.update(
            ids=document_ids,
            metadatas=updated_metadatas
        )

    def clear_collection(self) -> None:
        """
        Clear all embeddings from the collection.
        """
        # Delete the collection
        self.client.delete_collection(name=self.collection_name)

        # Recreate empty collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        print(f"Collection '{self.collection_name}' cleared successfully")

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.

        Returns:
            Dictionary containing collection statistics
        """
        count = self.collection.count()

        stats = {
            "collection_name": self.collection_name,
            "total_embeddings": count,
            "persist_directory": self.persist_directory,
            "embedding_dimension": self.embedding_dimension
        }

        if count > 0:
            # Get a sample to check embedding dimension
            sample = self.collection.peek(limit=1)
            if sample['embeddings'] is not None and len(sample['embeddings']) > 0:
                stats["actual_embedding_dimension"] = len(sample['embeddings'][0])

        return stats

    def delete_by_ids(self, document_ids: List[str]) -> None:
        """
        Delete embeddings by their document IDs.

        Args:
            document_ids: List of document IDs to delete
        """
        if not document_ids:
            return

        self.collection.delete(ids=document_ids)
        print(f"Deleted {len(document_ids)} embeddings from collection")

    def get_by_ids(self, document_ids: List[str]) -> Dict[str, Any]:
        """
        Retrieve embeddings by their document IDs.

        Args:
            document_ids: List of document IDs to retrieve

        Returns:
            Dictionary containing the requested documents and their metadata
        """
        results = self.collection.get(
            ids=document_ids,
            include=["embeddings", "documents", "metadatas"]
        )

        return {
            'ids': results['ids'],
            'documents': results['documents'],
            'metadatas': results['metadatas'],
            'embeddings': results['embeddings']
        }

    def __repr__(self) -> str:
        """String representation of the vector store."""
        return (
            f"ChromaVectorStore(collection='{self.collection_name}', "
            f"count={self.collection.count()}, "
            f"path='{self.persist_directory}')"
        )
