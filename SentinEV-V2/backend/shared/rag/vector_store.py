"""
ChromaDB Vector Store Setup for RAG
Manages document embeddings and similarity search.
"""

import os
from typing import List, Dict, Any, Optional
from pathlib import Path
import hashlib

# ChromaDB with sentence-transformers
try:
    import chromadb
    from chromadb.config import Settings

    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    chromadb = None


class VectorStore:
    """
    Manages ChromaDB vector store for document embeddings.
    Uses sentence-transformers for embedding generation.
    """

    # Default embedding model (small, fast, good quality)
    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    def __init__(
        self,
        persist_dir: str = "data/chroma",
        collection_name: str = "sentinev_knowledge",
    ):
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self.client = None
        self.collection = None

        if CHROMADB_AVAILABLE:
            self._init_client()

    def _init_client(self):
        """Initialize ChromaDB client."""
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir), settings=Settings(anonymized_telemetry=False)
        )

        # Get or create collection with embedding function
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},  # Use cosine similarity
        )

    def _generate_doc_id(self, content: str, source: str) -> str:
        """Generate unique document ID based on content hash."""
        hash_input = f"{source}:{content[:500]}"
        return hashlib.md5(hash_input.encode()).hexdigest()

    def add_documents(
        self,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
    ) -> int:
        """
        Add documents to the vector store.

        Args:
            documents: List of text chunks
            metadatas: List of metadata dicts (source, category, etc.)
            ids: Optional list of document IDs

        Returns:
            Number of documents added
        """
        if not CHROMADB_AVAILABLE or not self.collection:
            return 0

        if not ids:
            ids = [
                self._generate_doc_id(doc, meta.get("source", ""))
                for doc, meta in zip(documents, metadatas)
            ]

        # Check for existing documents (avoid duplicates)
        existing = self.collection.get(ids=ids)
        existing_ids = set(existing.get("ids", []))

        # Filter out existing
        new_docs = []
        new_metas = []
        new_ids = []

        for doc, meta, doc_id in zip(documents, metadatas, ids):
            if doc_id not in existing_ids:
                new_docs.append(doc)
                new_metas.append(meta)
                new_ids.append(doc_id)

        if new_docs:
            self.collection.add(documents=new_docs, metadatas=new_metas, ids=new_ids)

        return len(new_docs)

    def query(
        self, query_text: str, n_results: int = 5, category_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Query the vector store for similar documents.

        Args:
            query_text: The query string
            n_results: Maximum number of results
            category_filter: Optional category to filter by

        Returns:
            List of matching documents with metadata and scores
        """
        if not CHROMADB_AVAILABLE or not self.collection:
            return []

        where_filter = None
        if category_filter:
            where_filter = {"category": category_filter}

        results = self.collection.query(
            query_texts=[query_text], n_results=n_results, where=where_filter
        )

        # Format results
        formatted = []
        if results and results.get("documents"):
            docs = results["documents"][0]
            metas = (
                results["metadatas"][0]
                if results.get("metadatas")
                else [{}] * len(docs)
            )
            distances = (
                results["distances"][0] if results.get("distances") else [0] * len(docs)
            )

            for doc, meta, dist in zip(docs, metas, distances):
                formatted.append(
                    {
                        "content": doc,
                        "metadata": meta,
                        "similarity_score": 1 - dist,  # Convert distance to similarity
                    }
                )

        return formatted

    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        if not CHROMADB_AVAILABLE or not self.collection:
            return {"available": False}

        return {
            "available": True,
            "collection_name": self.collection_name,
            "document_count": self.collection.count(),
            "persist_dir": str(self.persist_dir),
        }

    def delete_collection(self):
        """Delete the entire collection."""
        if self.client and self.collection:
            self.client.delete_collection(self.collection_name)
            self.collection = None


# Singleton instance
vector_store = VectorStore()
