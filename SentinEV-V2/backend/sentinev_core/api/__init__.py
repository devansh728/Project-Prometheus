"""SentinEV Core API Package"""

from fastapi import APIRouter

from .telemetry import router as telemetry_router
from .agent import router as agent_router
from .rag import router as rag_router
from .diagnosis import router as diagnosis_router

router = APIRouter(tags=["Core"])

# Include sub-routers
router.include_router(telemetry_router)
router.include_router(agent_router)
router.include_router(rag_router)
router.include_router(diagnosis_router)


@router.get("/status")
async def get_system_status():
    """Get overall system status."""
    from sentinev_core.agents import master_agent
    from sentinev_core.services.telemetry_gen import telemetry_generator
    from shared.rag import vector_store

    return {
        "agents": {
            "master": master_agent.state.value,
            "data_analysis": "ready",
            "diagnosis": "ready",
            "engagement": "ready",
            "scheduling": "ready",
        },
        "vehicles_monitored": len(telemetry_generator.vehicles),
        "active_workflows": 0,
        "rag": vector_store.get_stats(),
    }
