"""
SentinEV - API v2 Endpoints
===========================
New API endpoints for Phase 9-13 features:
- Scheduling (priority queue, towing, forecast)
- CAPA (recurrence, supplier risk, drift, recommendations)
- Monitoring (UEBA, SLA, daily reports)
- Kafka WebSocket bridge
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Query
from pydantic import BaseModel, Field

# Import agents and utilities
from agents.scheduling_agent import get_scheduling_agent
from agents.capa_agent import get_capa_agent
from db.monitoring import get_monitoring_db

# ==================== Pydantic Models ====================


class QueueAppointmentRequest(BaseModel):
    """Request to add appointment to priority queue."""

    vehicle_id: str
    component: str
    severity: str = "medium"
    failure_probability: float = Field(default=0.5, ge=0, le=1)
    diagnosis_summary: str = ""
    estimated_cost: str = ""
    customer_tier: str = "standard"


class TowingRequest(BaseModel):
    """Emergency towing request."""

    vehicle_id: str
    pickup_location: str
    notes: str = ""


class FleetBatchRequest(BaseModel):
    """Fleet batch scheduling request."""

    tenant_id: str
    vehicles: List[Dict[str, Any]]


class UEBACheckRequest(BaseModel):
    """UEBA rule check request."""

    agent_name: str
    action_type: str
    target_resource: Optional[str] = None
    agent_tenant: str = "default"
    resource_tenant: str = "default"


class FingerprintRequest(BaseModel):
    """Agent fingerprint request."""

    agent_id: str
    ip_address: str = "127.0.0.1"
    user_agent: str = "SentinEV-Agent/1.0"
    session_id: Optional[str] = None


class SLACheckRequest(BaseModel):
    """SLA check request."""

    operation: str
    latency_ms: float
    vehicle_id: Optional[str] = None


# ==================== Router ====================

router_v2 = APIRouter(prefix="/v2", tags=["v2"])


# ==================== Scheduling Endpoints ====================


@router_v2.post("/scheduling/queue")
async def queue_appointment(request: QueueAppointmentRequest):
    """Add appointment to priority queue."""
    agent = get_scheduling_agent()
    result = agent.queue_appointment(
        vehicle_id=request.vehicle_id,
        component=request.component,
        severity=request.severity,
        failure_probability=request.failure_probability,
        diagnosis_summary=request.diagnosis_summary,
        estimated_cost=request.estimated_cost,
        customer_tier=request.customer_tier,
    )
    return result


@router_v2.post("/scheduling/queue/process")
async def process_queue(max_process: int = Query(default=5, ge=1, le=20)):
    """Process appointments from priority queue."""
    agent = get_scheduling_agent()
    result = agent.process_priority_queue(max_process=max_process)
    return result


@router_v2.get("/scheduling/queue/status")
async def get_queue_status():
    """Get priority queue status."""
    agent = get_scheduling_agent()
    return {
        "queue_size": agent.priority_queue.size(),
        "items": [
            {
                "vehicle_id": item.vehicle_id,
                "component": item.component,
                "urgency_score": item.urgency_score,
                "severity": item.severity,
            }
            for item in agent.priority_queue.get_all()[:10]
        ],
    }


@router_v2.post("/scheduling/towing")
async def initiate_towing(request: TowingRequest):
    """Initiate emergency towing."""
    agent = get_scheduling_agent()
    result = agent.initiate_emergency_towing(
        vehicle_id=request.vehicle_id,
        pickup_location=request.pickup_location,
        notes=request.notes,
    )
    return result


@router_v2.get("/scheduling/towing/{request_id}")
async def get_towing_status(request_id: str):
    """Get towing request status."""
    agent = get_scheduling_agent()
    result = agent.get_towing_status(request_id)
    if not result.get("success"):
        raise HTTPException(status_code=404, detail="Towing request not found")
    return result


@router_v2.get("/scheduling/forecast/{center_id}")
async def get_demand_forecast(center_id: str, date: Optional[str] = None):
    """Get demand forecast for service center."""
    agent = get_scheduling_agent()
    result = agent.get_demand_forecast(center_id, date)
    return result


@router_v2.get("/scheduling/optimal-center/{vehicle_id}")
async def get_optimal_center(vehicle_id: str, component: Optional[str] = None):
    """Get optimal service center for vehicle."""
    agent = get_scheduling_agent()
    result = agent.get_optimal_center(vehicle_id, component)
    return result


@router_v2.post("/scheduling/fleet/batch")
async def schedule_fleet_batch(request: FleetBatchRequest):
    """Schedule multiple vehicles from a fleet."""
    agent = get_scheduling_agent()
    result = agent.schedule_fleet_batch(request.tenant_id, request.vehicles)
    return result


# ==================== CAPA Endpoints ====================


@router_v2.get("/capa/recurrence/{component}")
async def get_recurrence_analysis(component: str, vehicle_id: Optional[str] = None):
    """Analyze failure recurrence for component."""
    agent = get_capa_agent()
    result = agent.analyze_recurrence(component, vehicle_id)
    return result


@router_v2.get("/capa/supplier/{supplier_id}")
async def get_supplier_risk(supplier_id: str):
    """Get supplier risk assessment."""
    agent = get_capa_agent()
    result = agent.get_supplier_risk(supplier_id=supplier_id)
    if not result.get("success"):
        raise HTTPException(status_code=404, detail="Supplier not found")
    return result


@router_v2.get("/capa/supplier")
async def get_all_supplier_risks():
    """Get all supplier risk assessments."""
    agent = get_capa_agent()
    result = agent.get_all_supplier_risks()
    return result


@router_v2.get("/capa/drift/{component}")
async def get_model_year_drift(component: str, target_year: Optional[int] = None):
    """Track model year quality drift."""
    agent = get_capa_agent()
    result = agent.track_model_year_drift(component, target_year)
    return result


@router_v2.get("/capa/recommendations/{component}")
async def get_engineering_recommendations(
    component: str, include_patterns: bool = True
):
    """Get LLM-generated engineering recommendations."""
    agent = get_capa_agent()
    result = agent.generate_engineering_recommendations(component, include_patterns)
    return result


@router_v2.get("/capa/insights")
async def get_manufacturing_insights():
    """Get comprehensive manufacturing insights."""
    agent = get_capa_agent()
    result = agent.get_insights_api_data()
    return result


# ==================== Monitoring Endpoints ====================


@router_v2.post("/monitoring/ueba/check")
async def check_ueba_rules(request: UEBACheckRequest):
    """Check UEBA security rules for an action."""
    db = get_monitoring_db()
    result = db.check_ueba_rules(
        agent_name=request.agent_name,
        action_type=request.action_type,
        target_resource=request.target_resource,
        agent_tenant=request.agent_tenant,
        resource_tenant=request.resource_tenant,
    )

    # Log the action with UEBA result
    db.log_agent_action(
        agent_name=request.agent_name,
        action_type=request.action_type,
        target_resource=request.target_resource,
        ueba_flagged=result["flagged"],
        ueba_reason=str(result["violations"]) if result["flagged"] else None,
        risk_score=result["total_risk_score"],
    )

    return result


@router_v2.get("/monitoring/ueba/alerts")
async def get_ueba_alerts(limit: int = Query(default=50, ge=1, le=500)):
    """Get recent UEBA flagged actions."""
    db = get_monitoring_db()
    alerts = db.get_ueba_alerts(limit=limit)
    return {"count": len(alerts), "alerts": alerts}


@router_v2.post("/monitoring/sla/check")
async def check_sla(request: SLACheckRequest):
    """Check SLA for an operation."""
    db = get_monitoring_db()
    result = db.log_sla_check(
        operation=request.operation,
        latency_ms=request.latency_ms,
        vehicle_id=request.vehicle_id,
    )
    return result


@router_v2.get("/monitoring/sla/summary")
async def get_sla_summary(hours: int = Query(default=24, ge=1, le=168)):
    """Get SLA compliance summary."""
    db = get_monitoring_db()
    result = db.get_sla_summary(hours=hours)
    return result


@router_v2.post("/monitoring/fingerprint")
async def log_fingerprint(request: FingerprintRequest):
    """Log agent fingerprint for identity tracking."""
    db = get_monitoring_db()
    result = db.log_fingerprint(
        agent_id=request.agent_id,
        ip_address=request.ip_address,
        user_agent=request.user_agent,
        session_id=request.session_id,
    )
    return result


@router_v2.get("/monitoring/report/daily")
async def get_daily_report(report_date: Optional[str] = None):
    """Get daily operational report."""
    db = get_monitoring_db()
    result = db.generate_daily_report(report_date)
    return result


@router_v2.get("/monitoring/inference/stats")
async def get_inference_stats(hours: int = Query(default=24, ge=1, le=168)):
    """Get inference statistics."""
    db = get_monitoring_db()
    result = db.get_inference_stats(hours=hours)
    return result


@router_v2.get("/monitoring/drift/summary")
async def get_drift_summary(hours: int = Query(default=24, ge=1, le=168)):
    """Get drift metrics summary."""
    db = get_monitoring_db()
    result = db.get_drift_summary(hours=hours)
    return {"features_with_drift": result, "count": len(result)}


# ==================== Kafka WebSocket Bridge ====================

# Store active Kafka connections
kafka_connections: Dict[str, List[WebSocket]] = {}


@router_v2.websocket("/ws/kafka/{topic}")
async def kafka_websocket_bridge(websocket: WebSocket, topic: str):
    """
    WebSocket bridge for Kafka topics.
    Subscribes to Kafka topic and streams messages to WebSocket client.
    """
    await websocket.accept()

    # Add to connections
    if topic not in kafka_connections:
        kafka_connections[topic] = []
    kafka_connections[topic].append(websocket)

    try:
        # Import Kafka consumer (simulated if Kafka not available)
        try:
            from streaming import SimulatedKafkaStream

            kafka_available = True
        except ImportError:
            kafka_available = False

        if kafka_available:
            # Start simulated Kafka stream
            stream = SimulatedKafkaStream()

            async def consume_and_forward():
                """Consume from Kafka and forward to WebSocket."""
                async for message in stream.stream_telemetry():
                    if websocket.client_state.value == 1:  # CONNECTED
                        await websocket.send_json(
                            {
                                "topic": topic,
                                "timestamp": datetime.now().isoformat(),
                                "data": message,
                            }
                        )
                    else:
                        break

            # Run consumer task
            consumer_task = asyncio.create_task(consume_and_forward())

            # Wait for client messages (for ping/control)
            while True:
                try:
                    data = await asyncio.wait_for(
                        websocket.receive_text(), timeout=30.0
                    )
                    # Handle control messages
                    if data == "ping":
                        await websocket.send_text("pong")
                except asyncio.TimeoutError:
                    # Send keepalive
                    await websocket.send_json({"type": "keepalive"})

        else:
            # Simulated mode - send periodic updates
            while True:
                await websocket.send_json(
                    {
                        "topic": topic,
                        "timestamp": datetime.now().isoformat(),
                        "type": "simulated",
                        "message": f"Kafka not available - simulated message for {topic}",
                    }
                )
                await asyncio.sleep(5)

    except WebSocketDisconnect:
        pass
    finally:
        # Remove from connections
        if topic in kafka_connections:
            kafka_connections[topic].remove(websocket)


# ==================== Health Check ====================


@router_v2.get("/health")
async def health_check():
    """v2 API health check."""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat(),
        "features": [
            "scheduling_priority_queue",
            "capa_recurrence_analysis",
            "supplier_risk_scoring",
            "ueba_security_rules",
            "sla_enforcement",
            "kafka_websocket_bridge",
        ],
    }
