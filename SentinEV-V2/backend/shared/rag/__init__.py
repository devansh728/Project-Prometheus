"""Shared RAG (Retrieval Augmented Generation) Package"""

from .vector_store import VectorStore, vector_store
from .embedding_pipeline import (
    DocumentChunker,
    EmbeddingPipeline,
    create_embedding_pipeline,
)
from .retrieval_service import RetrievalService, retrieval_service, RetrievalResult
from .context_assembler import ContextAssembler, context_assembler, AssembledContext
from .capa_engine import CAPAEngine, capa_engine, CAPAPattern

__all__ = [
    # Vector Store
    "VectorStore",
    "vector_store",
    # Embedding Pipeline
    "DocumentChunker",
    "EmbeddingPipeline",
    "create_embedding_pipeline",
    # Retrieval
    "RetrievalService",
    "retrieval_service",
    "RetrievalResult",
    # Context Assembly
    "ContextAssembler",
    "context_assembler",
    "AssembledContext",
    # CAPA
    "CAPAEngine",
    "capa_engine",
    "CAPAPattern",
]
