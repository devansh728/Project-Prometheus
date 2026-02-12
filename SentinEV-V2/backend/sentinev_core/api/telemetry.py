"""
Telematics API Endpoints
- REST: Get latest telemetry
- WebSocket: Stream real-time telemetry
- History: Log and retrieve telemetry history
- Fault Injection: Demo mode for brake fade scenario
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime
import json

from sentinev_core.services.telemetry_gen import telemetry_generator

router = APIRouter(prefix="/telemetry", tags=["Telemetry"])

# In-memory telemetry history storage (for demo purposes)
# In production, this would be stored in a database
telemetry_history: Dict[str, List[Dict[str, Any]]] = {}
MAX_HISTORY_SIZE = 100  # Keep last 100 readings per vehicle

# Fault injection state (demo mode)
injected_faults: Dict[str, Dict[str, Any]] = {}


class FaultInjectionRequest(BaseModel):
    """Schema for injecting faults into vehicle telemetry"""

    fault_type: str = "brake_fade"  # brake_fade, battery_drain, overheat
    severity: float = 0.7  # 0.0 to 1.0
    duration_seconds: int = 300  # How long the fault persists


class TelemetryReading(BaseModel):
    """Schema for logging telemetry data"""

    vehicle_id: str
    battery_voltage: float
    engine_temp: float
    brake_pressure: float
    motor_rpm: float
    coolant_level: float
    timestamp: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None


@router.get("/{vehicle_id}")
async def get_latest_telemetry(vehicle_id: str):
    """Get a single telemetry reading for a vehicle."""
    telemetry = telemetry_generator.generate_telemetry(vehicle_id)
    if not telemetry:
        return {"error": f"Vehicle {vehicle_id} not found"}
    return telemetry


@router.get("/")
async def get_fleet_telemetry():
    """Get latest telemetry for all vehicles in fleet."""
    return {
        vid: telemetry_generator.generate_telemetry(vid)
        for vid in telemetry_generator.vehicles.keys()
    }


@router.websocket("/stream/{vehicle_id}")
async def stream_vehicle_telemetry(
    websocket: WebSocket,
    vehicle_id: str,
    interval: float = Query(default=1.0, ge=0.5, le=10.0),
):
    """WebSocket: Stream real-time telemetry for a specific vehicle."""
    await websocket.accept()

    try:
        async for telemetry in telemetry_generator.stream_telemetry(
            vehicle_id, interval
        ):
            await websocket.send_json(telemetry)
    except WebSocketDisconnect:
        telemetry_generator.stop()


@router.websocket("/stream")
async def stream_all_telemetry(
    websocket: WebSocket, interval: float = Query(default=2.0, ge=0.5, le=10.0)
):
    """WebSocket: Stream real-time telemetry from all vehicles."""
    await websocket.accept()

    try:
        async for telemetry in telemetry_generator.stream_all_vehicles(interval):
            await websocket.send_json(telemetry)
    except WebSocketDisconnect:
        telemetry_generator.stop()


# ============================================================
# TELEMETRY HISTORY ENDPOINTS (Demo purpose - in-memory storage)
# ============================================================


@router.post("/history/log")
async def log_telemetry_history(reading: TelemetryReading):
    """
    Log a telemetry reading to history.
    Used by mobile app to send telemetry data for trend analysis.
    """
    global telemetry_history

    vehicle_id = reading.vehicle_id

    # Initialize history for vehicle if not exists
    if vehicle_id not in telemetry_history:
        telemetry_history[vehicle_id] = []

    # Create history entry
    entry = {
        "battery_voltage": reading.battery_voltage,
        "engine_temp": reading.engine_temp,
        "brake_pressure": reading.brake_pressure,
        "motor_rpm": reading.motor_rpm,
        "coolant_level": reading.coolant_level,
        "timestamp": reading.timestamp or datetime.now().isoformat(),
        "metrics": reading.metrics or {},
    }

    # Append to history
    telemetry_history[vehicle_id].append(entry)

    # Trim to max size
    if len(telemetry_history[vehicle_id]) > MAX_HISTORY_SIZE:
        telemetry_history[vehicle_id] = telemetry_history[vehicle_id][
            -MAX_HISTORY_SIZE:
        ]

    return {
        "status": "logged",
        "vehicle_id": vehicle_id,
        "entries_count": len(telemetry_history[vehicle_id]),
        "timestamp": entry["timestamp"],
    }


@router.get("/history/{vehicle_id}")
async def get_telemetry_history(
    vehicle_id: str,
    limit: int = Query(default=50, ge=1, le=MAX_HISTORY_SIZE),
):
    """
    Get telemetry history for a vehicle.
    Returns the most recent readings up to the specified limit.
    """
    if vehicle_id not in telemetry_history:
        return {
            "vehicle_id": vehicle_id,
            "entries": [],
            "count": 0,
            "message": "No history available for this vehicle",
        }

    history = telemetry_history[vehicle_id][-limit:]

    # Calculate trend summary
    if len(history) >= 2:
        first = history[0]
        last = history[-1]
        trends = {
            "battery_voltage": (
                "stable"
                if abs(last["battery_voltage"] - first["battery_voltage"]) < 0.2
                else (
                    "up"
                    if last["battery_voltage"] > first["battery_voltage"]
                    else "down"
                )
            ),
            "engine_temp": (
                "stable"
                if abs(last["engine_temp"] - first["engine_temp"]) < 5
                else ("up" if last["engine_temp"] > first["engine_temp"] else "down")
            ),
            "brake_pressure": (
                "stable"
                if abs(last["brake_pressure"] - first["brake_pressure"]) < 3
                else (
                    "up" if last["brake_pressure"] > first["brake_pressure"] else "down"
                )
            ),
        }
    else:
        trends = None

    return {
        "vehicle_id": vehicle_id,
        "entries": history,
        "count": len(history),
        "trends": trends,
    }


@router.delete("/history/{vehicle_id}")
async def clear_telemetry_history(vehicle_id: str):
    """Clear telemetry history for a vehicle (for testing/demo purposes)."""
    global telemetry_history

    if vehicle_id in telemetry_history:
        del telemetry_history[vehicle_id]
        return {"status": "cleared", "vehicle_id": vehicle_id}

    return {"status": "not_found", "vehicle_id": vehicle_id}


@router.get("/history/summary/{vehicle_id}")
async def get_telemetry_summary(vehicle_id: str):
    """
    Get a summary of telemetry statistics for a vehicle.
    Useful for dashboard displays.
    """
    if vehicle_id not in telemetry_history or len(telemetry_history[vehicle_id]) == 0:
        return {
            "vehicle_id": vehicle_id,
            "has_data": False,
            "message": "No history available",
        }

    history = telemetry_history[vehicle_id]

    # Calculate statistics
    def calc_stats(key):
        values = [h[key] for h in history]
        return {
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "latest": values[-1],
        }

    return {
        "vehicle_id": vehicle_id,
        "has_data": True,
        "total_readings": len(history),
        "time_range": {
            "earliest": history[0]["timestamp"],
            "latest": history[-1]["timestamp"],
        },
        "statistics": {
            "battery_voltage": calc_stats("battery_voltage"),
            "engine_temp": calc_stats("engine_temp"),
            "brake_pressure": calc_stats("brake_pressure"),
            "motor_rpm": calc_stats("motor_rpm"),
            "coolant_level": calc_stats("coolant_level"),
        },
        "health_status": (
            "healthy"
            if all(
                h.get("metrics", {}).get("anomalyScore", 0) < 0.1 for h in history[-10:]
            )
            else "monitor"
        ),
    }


# ============================================================
# FAULT INJECTION ENDPOINTS (Demo mode for brake fade scenario)
# ============================================================


@router.post("/inject-fault/{vehicle_id}")
async def inject_fault(vehicle_id: str, request: FaultInjectionRequest):
    """
    Inject a fault into vehicle telemetry for demo purposes.
    Simulates brake fade, battery drain, or overheating scenarios.
    """
    global injected_faults

    # Validate vehicle exists
    if vehicle_id not in telemetry_generator.vehicles:
        raise HTTPException(status_code=404, detail=f"Vehicle {vehicle_id} not found")

    # Store fault injection state
    injected_faults[vehicle_id] = {
        "fault_type": request.fault_type,
        "severity": request.severity,
        "duration_seconds": request.duration_seconds,
        "injected_at": datetime.now().isoformat(),
        "active": True,
    }

    # Return degradation parameters based on fault type
    degradation_params = {}
    if request.fault_type == "brake_fade":
        degradation_params = {
            "brake_pressure_reduction": 0.15 * request.severity,
            "engine_temp_increase": 12 * request.severity,
            "failure_probability": 0.25 + (0.65 * request.severity),
            "vibration_increase": 0.1 * request.severity,
        }
    elif request.fault_type == "battery_drain":
        degradation_params = {
            "battery_voltage_drop": 0.8 * request.severity,
            "coolant_level_drop": 0.05 * request.severity,
            "failure_probability": 0.2 + (0.5 * request.severity),
        }
    elif request.fault_type == "overheat":
        degradation_params = {
            "engine_temp_increase": 25 * request.severity,
            "coolant_level_drop": 0.1 * request.severity,
            "failure_probability": 0.3 + (0.6 * request.severity),
        }

    return {
        "status": "injected",
        "vehicle_id": vehicle_id,
        "fault_type": request.fault_type,
        "severity": request.severity,
        "degradation_params": degradation_params,
        "message": f"Fault '{request.fault_type}' injected into {vehicle_id}",
    }


@router.delete("/inject-fault/{vehicle_id}")
async def reset_fault(vehicle_id: str):
    """Reset/clear injected fault for a vehicle."""
    global injected_faults

    if vehicle_id in injected_faults:
        del injected_faults[vehicle_id]
        return {
            "status": "reset",
            "vehicle_id": vehicle_id,
            "message": "Fault injection cleared",
        }

    return {
        "status": "not_found",
        "vehicle_id": vehicle_id,
        "message": "No active fault injection found",
    }


@router.get("/inject-fault/{vehicle_id}/status")
async def get_fault_status(vehicle_id: str):
    """Get current fault injection status for a vehicle."""
    if vehicle_id in injected_faults:
        fault = injected_faults[vehicle_id]
        return {
            "vehicle_id": vehicle_id,
            "has_fault": True,
            **fault,
        }

    return {
        "vehicle_id": vehicle_id,
        "has_fault": False,
        "message": "No active fault injection",
    }


@router.get("/inject-fault")
async def list_all_faults():
    """List all currently injected faults across fleet."""
    return {
        "total_active": len(injected_faults),
        "faults": injected_faults,
    }
