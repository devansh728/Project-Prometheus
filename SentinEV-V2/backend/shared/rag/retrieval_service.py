"""
RAG Retrieval Service
Provides intelligent context retrieval for agent prompts.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from shared.rag.vector_store import vector_store


@dataclass
class RetrievalResult:
    """A single retrieval result."""

    content: str
    source: str
    category: str
    section: str
    similarity_score: float


class RetrievalService:
    """
    Handles intelligent retrieval from the knowledge base.
    Supports category-based filtering and relevance ranking.
    """

    # Category priorities for different query types
    CATEGORY_PRIORITIES = {
        "diagnosis": ["repair_guide", "diagnostic", "capa"],
        "repair": ["repair_guide", "manual"],
        "pattern": ["capa", "repair_guide"],
        "general": ["repair_guide", "manual", "capa", "general"],
    }

    def __init__(self, store=None):
        self.store = store or vector_store

    def retrieve(
        self,
        query: str,
        query_type: str = "general",
        n_results: int = 5,
        min_score: float = 0.3,
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: The search query
            query_type: Type of query (diagnosis, repair, pattern, general)
            n_results: Maximum number of results
            min_score: Minimum similarity score threshold

        Returns:
            List of retrieval results, ranked by relevance
        """
        # Get category priorities
        priorities = self.CATEGORY_PRIORITIES.get(query_type, ["general"])

        all_results = []

        # Query each category in priority order
        for category in priorities:
            results = self.store.query(
                query_text=query, n_results=n_results, category_filter=category
            )

            for r in results:
                if r.get("similarity_score", 0) >= min_score:
                    all_results.append(
                        RetrievalResult(
                            content=r.get("content", ""),
                            source=r.get("metadata", {}).get("source", "unknown"),
                            category=r.get("metadata", {}).get("category", "general"),
                            section=r.get("metadata", {}).get("section", ""),
                            similarity_score=r.get("similarity_score", 0),
                        )
                    )

        # Sort by score and deduplicate
        seen_content = set()
        unique_results = []
        for r in sorted(all_results, key=lambda x: x.similarity_score, reverse=True):
            content_hash = hash(r.content[:200])
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_results.append(r)

        return unique_results[:n_results]

    def retrieve_for_failure_type(
        self, failure_type: str, vehicle_info: Optional[Dict] = None
    ) -> List[RetrievalResult]:
        """
        Retrieve documents relevant to a specific failure type.
        """
        # Build query from failure type
        query_parts = [failure_type.replace("_", " ")]

        if vehicle_info:
            if vehicle_info.get("make"):
                query_parts.append(vehicle_info["make"])
            if vehicle_info.get("model"):
                query_parts.append(vehicle_info["model"])

        query = " ".join(query_parts)

        return self.retrieve(query=query, query_type="diagnosis", n_results=5)

    def retrieve_capa_patterns(
        self, symptoms: List[str], vehicle_info: Optional[Dict] = None
    ) -> List[RetrievalResult]:
        """
        Retrieve CAPA patterns matching given symptoms.
        """
        query = " ".join(symptoms)

        return self.retrieve(query=query, query_type="pattern", n_results=3)


# Singleton
retrieval_service = RetrievalService()
