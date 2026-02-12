"""
Priority Queue Service for ServiceOps AI
Manages global service request queue with urgency-based prioritization.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import uuid
import heapq


class UrgencyLevel(str, Enum):
    """Urgency levels for service requests."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class ServiceRequest:
    """A service request in the priority queue."""

    request_id: str
    vehicle_id: str
    customer_id: str
    customer_name: str
    vehicle_name: str
    geo_lat: float
    geo_lon: float
    failure_type: str
    severity: str
    urgency_score: float  # 0-100, higher = more urgent
    urgency_level: UrgencyLevel
    max_diagnosis_days: int
    preferred_dates: List[str]
    historical_center_id: Optional[str]
    user_tier: str  # "standard", "premium", "vip"

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)

    # Processing state
    status: str = "queued"  # queued, processing, assigned, completed
    assigned_center_id: Optional[str] = None

    def __lt__(self, other):
        """For heap comparison - higher urgency comes first."""
        return self.urgency_score > other.urgency_score


class PriorityQueue:
    """
    Global priority queue for service requests.
    Ranks vehicles by urgency score derived from severity, failure type, and user tier.
    """

    # Severity weights
    SEVERITY_WEIGHTS = {
        "critical": 40,
        "high": 30,
        "medium": 20,
        "low": 10,
    }

    # Failure type weights (some failures more urgent)
    FAILURE_WEIGHTS = {
        "brake_degradation": 25,
        "brake_fade": 25,
        "battery_degradation": 20,
        "cooling_degradation": 15,
        "electrical_fault": 15,
        "general_maintenance": 5,
    }

    # User tier bonuses
    TIER_BONUSES = {
        "vip": 20,
        "premium": 10,
        "standard": 0,
    }

    def __init__(self):
        self._queue: List[ServiceRequest] = []
        self._requests: Dict[str, ServiceRequest] = {}  # request_id -> request
        self._history: List[Dict] = []  # For audit trail

    def compute_urgency_score(
        self, severity: str, failure_type: str, user_tier: str, max_diagnosis_days: int
    ) -> float:
        """
        Compute urgency score (0-100) from multiple factors.
        """
        base_score = self.SEVERITY_WEIGHTS.get(severity.lower(), 15)
        failure_score = self.FAILURE_WEIGHTS.get(failure_type.lower(), 10)
        tier_bonus = self.TIER_BONUSES.get(user_tier.lower(), 0)

        # Days urgency: fewer days = more urgent
        days_factor = max(0, 15 - max_diagnosis_days)  # 0-15 points

        total = base_score + failure_score + tier_bonus + days_factor
        return min(100, max(0, total))

    def get_urgency_level(self, score: float) -> UrgencyLevel:
        """Map score to urgency level."""
        if score >= 70:
            return UrgencyLevel.CRITICAL
        elif score >= 50:
            return UrgencyLevel.HIGH
        elif score >= 30:
            return UrgencyLevel.MEDIUM
        else:
            return UrgencyLevel.LOW

    def enqueue(
        self,
        vehicle_id: str,
        customer_id: str,
        customer_name: str,
        vehicle_name: str,
        geo_lat: float,
        geo_lon: float,
        failure_type: str,
        severity: str,
        max_diagnosis_days: int = 7,
        preferred_dates: Optional[List[str]] = None,
        historical_center_id: Optional[str] = None,
        user_tier: str = "standard",
    ) -> ServiceRequest:
        """
        Add a new service request to the queue.
        """
        request_id = f"REQ-{uuid.uuid4().hex[:8].upper()}"

        urgency_score = self.compute_urgency_score(
            severity, failure_type, user_tier, max_diagnosis_days
        )
        urgency_level = self.get_urgency_level(urgency_score)

        request = ServiceRequest(
            request_id=request_id,
            vehicle_id=vehicle_id,
            customer_id=customer_id,
            customer_name=customer_name,
            vehicle_name=vehicle_name,
            geo_lat=geo_lat,
            geo_lon=geo_lon,
            failure_type=failure_type,
            severity=severity,
            urgency_score=urgency_score,
            urgency_level=urgency_level,
            max_diagnosis_days=max_diagnosis_days,
            preferred_dates=preferred_dates or [],
            historical_center_id=historical_center_id,
            user_tier=user_tier,
        )

        heapq.heappush(self._queue, request)
        self._requests[request_id] = request

        self._history.append(
            {
                "event": "QUEUE_ENTRY",
                "request_id": request_id,
                "vehicle_id": vehicle_id,
                "urgency_score": urgency_score,
                "urgency_level": urgency_level.value,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

        return request

    def dequeue(self) -> Optional[ServiceRequest]:
        """
        Remove and return the highest-priority request.
        """
        if not self._queue:
            return None

        request = heapq.heappop(self._queue)
        request.status = "processing"

        self._history.append(
            {
                "event": "QUEUE_DEQUEUE",
                "request_id": request.request_id,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

        return request

    def peek(self) -> Optional[ServiceRequest]:
        """View highest-priority request without removing."""
        return self._queue[0] if self._queue else None

    def preempt(self, request: ServiceRequest) -> None:
        """
        Insert a critical request, potentially bumping others.
        Used for sudden high-urgency cases.
        """
        # Boost the urgency score to ensure it's at the top
        request.urgency_score = min(100, request.urgency_score + 20)
        request.urgency_level = UrgencyLevel.CRITICAL

        heapq.heappush(self._queue, request)
        self._requests[request.request_id] = request

        self._history.append(
            {
                "event": "PREEMPTION",
                "request_id": request.request_id,
                "urgency_score": request.urgency_score,
                "timestamp": datetime.utcnow().isoformat(),
                "reasoning": "Critical request inserted with priority boost",
            }
        )

    def reorder(self) -> None:
        """
        Re-heapify the queue (called after external changes).
        """
        heapq.heapify(self._queue)

    def get_queue(self) -> List[Dict[str, Any]]:
        """
        Get current queue state, sorted by urgency (for dashboard).
        """
        # Sort by urgency score descending
        sorted_queue = sorted(self._queue, key=lambda x: x.urgency_score, reverse=True)

        return [
            {
                "request_id": r.request_id,
                "vehicle_id": r.vehicle_id,
                "vehicle_name": r.vehicle_name,
                "customer_id": r.customer_id,
                "customer_name": r.customer_name,
                "failure_type": r.failure_type,
                "severity": r.severity,
                "urgency_score": round(r.urgency_score, 1),
                "urgency_level": r.urgency_level.value,
                "user_tier": r.user_tier,
                "max_diagnosis_days": r.max_diagnosis_days,
                "status": r.status,
                "created_at": r.created_at.isoformat(),
            }
            for r in sorted_queue
        ]

    def get_request(self, request_id: str) -> Optional[ServiceRequest]:
        """Get a specific request by ID."""
        return self._requests.get(request_id)

    def update_status(
        self, request_id: str, status: str, center_id: Optional[str] = None
    ) -> bool:
        """Update request status."""
        request = self._requests.get(request_id)
        if not request:
            return False

        request.status = status
        if center_id:
            request.assigned_center_id = center_id

        self._history.append(
            {
                "event": "STATUS_UPDATE",
                "request_id": request_id,
                "new_status": status,
                "center_id": center_id,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

        return True

    def get_history(self, limit: int = 50) -> List[Dict]:
        """Get recent queue events."""
        return self._history[-limit:]

    def size(self) -> int:
        """Get queue size."""
        return len(self._queue)

    def clear(self) -> None:
        """Clear the queue (for demo reset)."""
        self._queue.clear()
        self._requests.clear()
        self._history.clear()


# Singleton
priority_queue = PriorityQueue()
