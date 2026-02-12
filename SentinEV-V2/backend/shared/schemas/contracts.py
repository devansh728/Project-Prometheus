"""
Shared Pydantic Schemas - API Contracts
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


# ================== ENUMS ==================


class Severity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"


class ServiceStatus(str, Enum):
    REQUESTED = "REQUESTED"
    BOOKED = "BOOKED"
    CONFIRMED = "CONFIRMED"
    CHECK_IN = "CHECK_IN"
    DIAGNOSIS = "DIAGNOSIS"
    PARTS_ALLOCATED = "PARTS_ALLOCATED"
    REPAIR_IN_PROGRESS = "REPAIR_IN_PROGRESS"
    QUALITY_CHECK = "QUALITY_CHECK"
    READY = "READY"
    COMPLETED = "COMPLETED"


class AgentType(str, Enum):
    MASTER = "master"
    DATA_ANALYSIS = "data_analysis"
    DIAGNOSIS = "diagnosis"
    ENGAGEMENT = "engagement"
    SCHEDULING = "scheduling"
    UEBA = "ueba"


# ================== TELEMATICS CONTRACT ==================


class TelemetryPayload(BaseModel):
    """Contract: Vehicle Telematics Ingestion"""

    vehicle_id: str
    timestamp: datetime
    sensors: Dict[str, float] = Field(
        ...,
        example={
            "engine_temp": 92.5,
            "battery_voltage": 12.8,
            "coolant_level": 0.85,
            "brake_pressure": 45.2,
            "vibration_amplitude": 0.12,
            "rpm": 2500,
        },
    )
    gps: Optional[Dict[str, float]] = Field(
        None, example={"lat": 40.7128, "lon": -74.0060}
    )


class TelemetryResponse(BaseModel):
    """Response after telemetry ingestion"""

    received: bool = True
    anomaly_score: Optional[float] = None
    immediate_action_required: bool = False


# ================== HEALTH STATUS CONTRACT ==================


class ComponentHealth(BaseModel):
    name: str
    health_score: float = Field(..., ge=0, le=100)
    status: Severity
    last_service: Optional[datetime] = None


class VehicleHealthStatus(BaseModel):
    """Contract: Vehicle Health Query Response"""

    vehicle_id: str
    overall_health: float = Field(..., ge=0, le=100)
    anomaly_score: float = Field(..., ge=0, le=1)
    failure_probability: float = Field(..., ge=0, le=1)
    predicted_rul_days: Optional[int] = None
    primary_concern: Optional[str] = None
    severity: Severity
    confidence_score: float = Field(..., ge=0, le=1)
    components: List[ComponentHealth] = []
    last_updated: datetime


# ================== SERVICE REQUEST CONTRACT ==================


class ServiceRequest(BaseModel):
    """Contract: SentinEV Core -> ServiceOpsAI"""

    vehicle_id: str
    customer_id: str
    failure_type: str
    severity: Severity
    failure_probability: float
    estimated_rul_days: Optional[int] = None
    preferred_datetime: Optional[datetime] = None
    customer_location: Optional[Dict[str, float]] = None  # lat, lon


class ServiceSlot(BaseModel):
    slot_id: str
    center_id: str
    center_name: str
    distance_km: float
    datetime: datetime
    estimated_duration_hours: float
    recommendation_score: float


class ServiceRequestResponse(BaseModel):
    """Contract: ServiceOpsAI -> SentinEV Core"""

    available_slots: List[ServiceSlot]
    parts_available: bool
    estimated_cost_range: Optional[str] = None


# ================== AGENT COMMUNICATION CONTRACT ==================


class AgentMessage(BaseModel):
    """Contract: Inter-Agent Communication"""

    source_agent: AgentType
    target_agent: AgentType
    message_type: str
    payload: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    correlation_id: str


class AgentAction(BaseModel):
    """Logged action for UEBA analysis"""

    agent_id: AgentType
    action: str
    target_resource: str
    parameters: Dict[str, Any] = {}
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    success: bool = True
    duration_ms: Optional[int] = None


# ================== VOICE AGENT CONTRACT ==================


class VoiceCallIntent(BaseModel):
    """Trigger a voice call to user"""

    user_id: str
    vehicle_id: str
    severity: Severity
    reason: str
    script_key: str = "default_warning"
    metadata: Dict[str, Any] = {}


class VoiceCallResult(BaseModel):
    """Result of voice interaction"""

    call_id: str
    user_response: str  # "accepted", "declined", "no_answer"
    booking_created: bool = False
    booking_id: Optional[str] = None
    transcript: Optional[str] = None
