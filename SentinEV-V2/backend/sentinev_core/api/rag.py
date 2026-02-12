"""
RAG API Endpoints
Provides endpoints for knowledge base management and retrieval.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from shared.rag import (
    vector_store,
    retrieval_service,
    context_assembler,
    capa_engine,
    create_embedding_pipeline,
)

router = APIRouter(prefix="/rag", tags=["RAG & Intelligence"])


# --- Request Models ---


class QueryRequest(BaseModel):
    """Request for RAG query."""

    query: str
    query_type: str = "general"
    n_results: int = 5


class DiagnosisContextRequest(BaseModel):
    """Request for diagnosis context."""

    failure_type: str
    symptoms: List[str]
    vehicle_make: Optional[str] = None
    vehicle_model: Optional[str] = None


class CAPADetectionRequest(BaseModel):
    """Request for CAPA pattern detection."""

    symptoms: List[str]
    dtc_codes: List[str] = []
    vehicle_make: Optional[str] = None
    vehicle_mileage: int = 0
    driving_profile: Optional[str] = None


class IndexRequest(BaseModel):
    """Request to index a directory."""

    directory: str


# --- Endpoints ---


@router.post("/query")
async def query_knowledge_base(request: QueryRequest):
    """Query the knowledge base."""
    results = retrieval_service.retrieve(
        query=request.query, query_type=request.query_type, n_results=request.n_results
    )

    return {
        "query": request.query,
        "count": len(results),
        "results": [
            {
                "content": (
                    r.content[:500] + "..." if len(r.content) > 500 else r.content
                ),
                "source": r.source,
                "section": r.section,
                "category": r.category,
                "similarity_score": r.similarity_score,
            }
            for r in results
        ],
    }


@router.post("/context/diagnosis")
async def get_diagnosis_context(request: DiagnosisContextRequest):
    """Get assembled context for a diagnosis."""
    vehicle_info = None
    if request.vehicle_make:
        vehicle_info = {"make": request.vehicle_make, "model": request.vehicle_model}

    context = context_assembler.assemble_for_diagnosis(
        failure_type=request.failure_type,
        symptoms=request.symptoms,
        vehicle_info=vehicle_info,
    )

    return {
        "context": context.text,
        "sources": context.sources,
        "categories": context.categories,
        "token_estimate": context.token_estimate,
    }


@router.post("/capa/detect")
async def detect_capa_patterns(request: CAPADetectionRequest):
    """Detect matching CAPA patterns."""
    patterns = capa_engine.detect_patterns(
        symptoms=request.symptoms,
        dtc_codes=request.dtc_codes,
        vehicle_make=request.vehicle_make,
        vehicle_mileage=request.vehicle_mileage,
        driving_profile=request.driving_profile,
    )

    return {
        "count": len(patterns),
        "patterns": [
            {
                "pattern_id": p.pattern_id,
                "pattern_name": p.pattern_name,
                "match_confidence": p.match_confidence,
                "severity": p.severity,
                "root_cause": p.root_cause,
                "corrective_action": p.corrective_action,
                "preventive_action": p.preventive_action,
            }
            for p in patterns
        ],
    }


@router.get("/capa/preventive/{vehicle_make}")
async def get_preventive_recommendations(vehicle_make: str, mileage: int = 0):
    """Get preventive recommendations for a vehicle."""
    recommendations = capa_engine.get_preventive_recommendations(
        vehicle_make=vehicle_make, vehicle_mileage=mileage
    )

    return {
        "vehicle_make": vehicle_make,
        "mileage": mileage,
        "recommendations": recommendations,
    }


@router.post("/index")
async def index_documents(request: IndexRequest):
    """Index documents from a directory."""
    pipeline = create_embedding_pipeline()

    result = pipeline.process_directory(request.directory)

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return {"status": "success", **result}


@router.get("/stats")
async def get_rag_stats():
    """Get RAG system statistics."""
    return vector_store.get_stats()
