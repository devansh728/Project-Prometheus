"""
Agent Orchestration API Endpoints
Exposes the master agent pipeline via REST and WebSocket.
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
from datetime import datetime
import json

from sentinev_core.agents import master_agent, WorkflowResult
from sentinev_core.services.telemetry_gen import telemetry_generator
from sentinev_core.services.decision_engine import decision_engine

router = APIRouter(prefix="/agent", tags=["Agent Orchestration"])


class AnalyzeRequest(BaseModel):
    """Request body for vehicle analysis."""

    vehicle_id: str
    sensors: Optional[Dict[str, float]] = None  # If None, generate from simulator
    force_refresh: bool = False


class AnalyzeResponse(BaseModel):
    """Response from analysis."""

    vehicle_id: str
    state: str
    severity: Optional[str]
    primary_concern: Optional[str]
    recommended_action: Optional[str]
    engagement_action: Optional[str]
    actions_log: List[str]
    decision: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

@router.get("/analyze/quick/{vehicle_id}")
async def quick_analyze(vehicle_id: str):
    """Quick analysis endpoint using GET."""
    request = AnalyzeRequest(vehicle_id=vehicle_id)
    return await analyze_vehicle(request)

@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_vehicle(request: AnalyzeRequest):
    """
    Run the full agent analysis pipeline for a vehicle.
    """
    vehicle_id = request.vehicle_id

    # Get vehicle config from simulator
    vehicle = telemetry_generator.vehicles.get(vehicle_id)
    if not vehicle:
        raise HTTPException(status_code=404, detail=f"Vehicle {vehicle_id} not found")

    # Get sensors - either from request or generate fresh
    if request.sensors:
        sensors = request.sensors
    else:
        telemetry = telemetry_generator.generate_telemetry(vehicle_id)
        sensors = telemetry["sensors"] if telemetry else {}

    # Run master agent pipeline
    result: WorkflowResult = master_agent.run(
        vehicle_id=vehicle_id,
        sensors=sensors,
        baseline=vehicle.get("baseline_config", {}),
        degradation_config=vehicle.get("degradation_config", {}),
        customer_id=vehicle.get("owner_id"),
        mileage=vehicle.get("mileage", 0),
    )

    # Get decision from decision engine
    decision_result = None
    if result.severity:
        from sentinev_core.services.ml_pipeline import Severity

        decision = decision_engine.decide(
            severity=Severity(result.severity),
            failure_probability=(
                result.service_request.get("failure_probability", 0)
                if result.service_request
                else 0
            ),
            confidence=0.85,
        )
        decision_result = {
            "action": decision.action.value,
            "priority": decision.priority,
            "notify_customer": decision.should_notify_customer,
            "delay_seconds": decision.delay_seconds,
            "rationale": decision.rationale,
        }

    return AnalyzeResponse(
        vehicle_id=result.vehicle_id,
        state=result.state.value,
        severity=result.severity,
        primary_concern=result.primary_concern,
        recommended_action=result.recommended_action,
        engagement_action=result.engagement_action,
        actions_log=result.actions_log,
        decision=decision_result,
        error=result.error,
    )


@router.get("/analyze/quick/{vehicle_id}")
async def quick_analyze(vehicle_id: str):
    """Quick analysis endpoint using GET."""
    request = AnalyzeRequest(vehicle_id=vehicle_id)
    return await analyze_vehicle(request)


@router.websocket("/stream/{vehicle_id}")
async def stream_agent_workflow(websocket: WebSocket, vehicle_id: str):
    """
    WebSocket: Stream agent workflow progress in real-time.
    """
    await websocket.accept()

    vehicle = telemetry_generator.vehicles.get(vehicle_id)
    if not vehicle:
        await websocket.send_json({"error": f"Vehicle {vehicle_id} not found"})
        await websocket.close()
        return

    try:
        # Generate fresh telemetry
        telemetry = telemetry_generator.generate_telemetry(vehicle_id)
        sensors = telemetry["sensors"] if telemetry else {}

        # Stream workflow events
        async for event in master_agent.run_async(
            vehicle_id=vehicle_id,
            sensors=sensors,
            baseline=vehicle.get("baseline_config", {}),
            degradation_config=vehicle.get("degradation_config", {}),
            customer_id=vehicle.get("owner_id"),
            mileage=vehicle.get("mileage", 0),
        ):
            await websocket.send_json(event)

    except WebSocketDisconnect:
        pass
    finally:
        await websocket.close()


@router.get("/fleet/analyze")
async def analyze_fleet():
    """Analyze all vehicles in the fleet."""
    results = {}

    for vehicle_id in telemetry_generator.vehicles.keys():
        try:
            response = await quick_analyze(vehicle_id)
            results[vehicle_id] = {
                "severity": response.severity,
                "primary_concern": response.primary_concern,
                "engagement_action": response.engagement_action,
            }
        except Exception as e:
            results[vehicle_id] = {"error": str(e)}

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "vehicles_analyzed": len(results),
        "results": results,
    }
