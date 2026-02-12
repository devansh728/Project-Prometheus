"""
Diagnosis API Endpoints
- Generate agent insights for vehicles
- Brake fade specific diagnostics
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime
import random

router = APIRouter(prefix="/diagnosis", tags=["Diagnosis"])


# Agent insight types and their messages
AGENT_INSIGHTS = {
    "normal": [
        {
            "agent": "data_analysis",
            "message": "Telemetry patterns within normal range",
            "severity": "info",
        },
        {
            "agent": "behavior",
            "message": "Driving behavior: Normal acceleration patterns",
            "severity": "info",
        },
        {
            "agent": "trend",
            "message": "No significant trends detected in brake metrics",
            "severity": "info",
        },
    ],
    "brake_fade": [
        {
            "agent": "data_analysis",
            "message": "Brake temperature trending +15% above baseline",
            "severity": "warning",
        },
        {
            "agent": "behavior",
            "message": "Aggressive braking frequency detected ↑ 23%",
            "severity": "warning",
        },
        {
            "agent": "trend",
            "message": "Brake efficiency declining: -8% over last 48 hours",
            "severity": "warning",
        },
        {
            "agent": "diagnosis",
            "message": "Cross-referencing with manufacturer degradation curves...",
            "severity": "info",
        },
        {
            "agent": "data_analysis",
            "message": "Vibration amplitude increased in braking zones",
            "severity": "warning",
        },
        {
            "agent": "diagnosis",
            "message": "Brake pad wear pattern: Front-left accelerated wear",
            "severity": "warning",
        },
        {
            "agent": "trend",
            "message": "Projected failure window: 6-7 days at current rate",
            "severity": "critical",
        },
        {
            "agent": "diagnosis",
            "message": "Failure probability: 65-75% within prediction window",
            "severity": "critical",
        },
        {
            "agent": "engagement",
            "message": "Initiating proactive customer contact protocol...",
            "severity": "info",
        },
        {
            "agent": "engagement",
            "message": "Voice agent scheduled for customer notification",
            "severity": "info",
        },
    ],
    "battery_drain": [
        {
            "agent": "data_analysis",
            "message": "Battery voltage below optimal threshold",
            "severity": "warning",
        },
        {
            "agent": "trend",
            "message": "Discharge rate 12% higher than baseline",
            "severity": "warning",
        },
        {
            "agent": "diagnosis",
            "message": "Cell imbalance detected in battery pack",
            "severity": "warning",
        },
    ],
    "overheat": [
        {
            "agent": "data_analysis",
            "message": "Coolant temperature exceeding safe limits",
            "severity": "critical",
        },
        {
            "agent": "trend",
            "message": "Temperature rising 2.5°C per hour",
            "severity": "warning",
        },
        {
            "agent": "diagnosis",
            "message": "Possible cooling system restriction",
            "severity": "critical",
        },
    ],
}


class DiagnosisRequest(BaseModel):
    """Request for generating diagnosis insights"""

    fault_type: Optional[str] = None
    include_recommendations: bool = True


@router.get("/{vehicle_id}/insights")
async def get_vehicle_insights(
    vehicle_id: str,
    fault_type: Optional[str] = Query(default=None, description="Filter by fault type"),
    limit: int = Query(default=10, ge=1, le=20),
):
    """
    Get AI agent insights for a vehicle.
    Returns observations from Data Analysis, Behavior, Trend, Diagnosis, and Engagement agents.
    """
    # Determine insight set based on fault type
    if fault_type and fault_type in AGENT_INSIGHTS:
        base_insights = AGENT_INSIGHTS[fault_type]
    else:
        base_insights = AGENT_INSIGHTS["normal"]

    # Generate timestamped insights
    insights = []
    base_time = datetime.now()
    for i, insight in enumerate(base_insights[:limit]):
        insights.append(
            {
                "id": f"insight-{vehicle_id}-{i}",
                "vehicle_id": vehicle_id,
                "agent_type": insight["agent"],
                "message": insight["message"],
                "severity": insight["severity"],
                "timestamp": (base_time.timestamp() - (len(base_insights) - i) * 3),
                "formatted_time": base_time.strftime("%H:%M:%S"),
            }
        )

    return {
        "vehicle_id": vehicle_id,
        "fault_type": fault_type or "normal",
        "insights": insights,
        "total_count": len(insights),
    }


@router.get("/{vehicle_id}/brake-analysis")
async def get_brake_analysis(vehicle_id: str):
    """
    Get detailed brake system analysis for a vehicle.
    Used for brake fade scenario dashboard display.
    """
    # Simulated brake analysis data
    return {
        "vehicle_id": vehicle_id,
        "analysis_time": datetime.now().isoformat(),
        "brake_system": {
            "front_left": {
                "pad_wear": random.uniform(0.6, 0.85),
                "rotor_condition": "good",
                "caliper_status": "normal",
                "temperature": random.uniform(180, 220),
            },
            "front_right": {
                "pad_wear": random.uniform(0.7, 0.92),
                "rotor_condition": "good",
                "caliper_status": "normal",
                "temperature": random.uniform(175, 210),
            },
            "rear_left": {
                "pad_wear": random.uniform(0.75, 0.95),
                "rotor_condition": "good",
                "caliper_status": "normal",
                "temperature": random.uniform(140, 170),
            },
            "rear_right": {
                "pad_wear": random.uniform(0.78, 0.96),
                "rotor_condition": "good",
                "caliper_status": "normal",
                "temperature": random.uniform(135, 165),
            },
        },
        "fluid_level": random.uniform(0.85, 0.98),
        "abs_status": "operational",
        "recommendations": [
            {
                "priority": "high",
                "component": "front_left_brake_pad",
                "action": "Replace brake pads",
                "estimated_cost": "₹4,500 - ₹6,000",
                "urgency_days": 7,
            }
        ],
        "failure_probability": random.uniform(0.55, 0.75),
        "remaining_useful_life_hours": random.randint(150, 400),
    }


@router.get("/{vehicle_id}/health-summary")
async def get_health_summary(vehicle_id: str):
    """
    Get overall vehicle health summary with component scores.
    """
    return {
        "vehicle_id": vehicle_id,
        "overall_score": random.uniform(75, 95),
        "last_updated": datetime.now().isoformat(),
        "components": {
            "battery": {"score": random.uniform(85, 98), "status": "good"},
            "motor": {"score": random.uniform(90, 99), "status": "excellent"},
            "brakes": {"score": random.uniform(65, 80), "status": "attention"},
            "suspension": {"score": random.uniform(88, 97), "status": "good"},
            "cooling": {"score": random.uniform(82, 95), "status": "good"},
            "electrical": {"score": random.uniform(90, 99), "status": "excellent"},
        },
        "alerts": [
            {
                "type": "brake_wear",
                "message": "Front brake pads showing accelerated wear",
                "severity": "warning",
            }
        ],
        "next_service": {
            "recommended_date": "2026-02-15",
            "type": "Brake Service",
            "estimated_duration": "2 hours",
        },
    }
