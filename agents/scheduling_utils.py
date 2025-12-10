"""
SentinEV - Scheduling Utilities
Support structures for autonomous scheduling with priority queue, urgency scoring, and demand forecasting.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum
import heapq
import random


class TowingStatus(Enum):
    """Towing request status."""

    PENDING = "pending"
    DISPATCHED = "dispatched"
    EN_ROUTE = "en_route"
    ARRIVED = "arrived"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass
class PriorityAppointment:
    """
    Appointment request with priority scoring.
    Used in priority queue for scheduling.
    """

    vehicle_id: str
    component: str
    severity: str
    failure_probability: float
    urgency_score: float
    requested_at: str = field(default_factory=lambda: datetime.now().isoformat())
    customer_tier: str = "standard"  # standard, premium, enterprise
    diagnosis_summary: str = ""
    estimated_cost: str = ""
    assigned_center: Optional[str] = None
    assigned_slot: Optional[str] = None

    def __lt__(self, other):
        """For heap comparison - higher urgency first."""
        return self.urgency_score > other.urgency_score


@dataclass
class TowingRequest:
    """Emergency towing request for immobile vehicles."""

    request_id: str
    vehicle_id: str
    pickup_location: str
    destination_center: str
    status: TowingStatus = TowingStatus.PENDING
    tow_company: str = ""
    estimated_eta_minutes: int = 30
    driver_name: str = ""
    driver_phone: str = ""
    requested_at: str = field(default_factory=lambda: datetime.now().isoformat())
    dispatched_at: Optional[str] = None
    completed_at: Optional[str] = None
    notes: str = ""


@dataclass
class DemandForecast:
    """Service demand forecast for capacity planning."""

    center_id: str
    date: str
    hour: int
    predicted_demand: float  # 0-1 normalized
    actual_bookings: int = 0
    available_capacity: int = 0
    recommendation: str = ""  # "high_demand", "normal", "low_demand"


@dataclass
class FleetScheduleResult:
    """Result of batch fleet scheduling."""

    tenant_id: str
    total_vehicles: int
    scheduled_count: int
    failed_count: int
    appointments: List[Dict] = field(default_factory=list)
    summary: str = ""


# Severity weights for urgency calculation
SEVERITY_WEIGHTS = {
    "critical": 1.0,
    "high": 0.75,
    "medium": 0.5,
    "low": 0.25,
}

# Customer tier boosts
TIER_BOOSTS = {
    "enterprise": 1.3,
    "premium": 1.15,
    "standard": 1.0,
}

# Simulated service center locations (lat, lon) for geo-optimization
SIMULATED_CENTER_LOCATIONS = {
    "SC-001": {
        "name": "Downtown EV Hub",
        "lat": 40.7128,
        "lon": -74.0060,
        "avg_wait": 15,
    },
    "SC-002": {
        "name": "Uptown Service Center",
        "lat": 40.7831,
        "lon": -73.9712,
        "avg_wait": 25,
    },
    "SC-003": {
        "name": "Airport Express EV",
        "lat": 40.6413,
        "lon": -73.7781,
        "avg_wait": 10,
    },
    "SC-004": {
        "name": "Suburban Care Plus",
        "lat": 40.7282,
        "lon": -73.7949,
        "avg_wait": 5,
    },
}

# Simulated towing companies
TOWING_COMPANIES = [
    {"name": "QuickTow EV", "base_eta": 20, "phone": "555-0101"},
    {"name": "RapidRescue", "base_eta": 25, "phone": "555-0102"},
    {"name": "SafeHaul Services", "base_eta": 30, "phone": "555-0103"},
]


def calculate_urgency_score(
    severity: str,
    failure_probability: float,
    customer_tier: str = "standard",
    hours_since_detection: float = 0,
) -> float:
    """
    Calculate urgency score for scheduling priority.

    Args:
        severity: Severity level (critical, high, medium, low)
        failure_probability: ML-predicted failure probability (0-1)
        customer_tier: Customer tier (enterprise, premium, standard)
        hours_since_detection: Hours since issue was detected

    Returns:
        Urgency score (0-10 scale)
    """
    # Base from severity
    base = SEVERITY_WEIGHTS.get(severity.lower(), 0.5)

    # Failure probability boost (0-1 → 1-2 multiplier)
    prob_multiplier = 1 + failure_probability

    # Time decay - urgency increases over time
    decay_factor = 1 + (hours_since_detection * 0.05)  # 5% increase per hour
    decay_factor = min(decay_factor, 2.0)  # Cap at 2x

    # Customer tier boost
    tier_boost = TIER_BOOSTS.get(customer_tier.lower(), 1.0)

    # Calculate final score (0-10 scale)
    score = base * prob_multiplier * decay_factor * tier_boost * 10

    return min(score, 10.0)  # Cap at 10


def simulate_geo_distance(
    vehicle_lat: float, vehicle_lon: float, center_id: str
) -> float:
    """
    Simulate distance between vehicle and service center.
    Uses simple Euclidean distance as approximation.

    Args:
        vehicle_lat: Vehicle latitude
        vehicle_lon: Vehicle longitude
        center_id: Service center ID

    Returns:
        Simulated distance in km
    """
    center = SIMULATED_CENTER_LOCATIONS.get(center_id)
    if not center:
        return 999.0  # Unknown center

    # Simple Euclidean as proxy (not actual geo distance)
    lat_diff = abs(vehicle_lat - center["lat"])
    lon_diff = abs(vehicle_lon - center["lon"])

    # Rough conversion: 1 degree ≈ 111km
    distance_km = ((lat_diff**2 + lon_diff**2) ** 0.5) * 111

    return round(distance_km, 2)


def select_optimal_center(
    vehicle_lat: float,
    vehicle_lon: float,
    available_centers: List[str],
    component: str = None,
    parts_availability: Dict[str, bool] = None,
) -> Dict[str, Any]:
    """
    Select optimal service center using weighted scoring.

    Weights:
    - Distance: 40%
    - Wait time: 30%
    - Rating: 20%
    - Parts availability: 10%

    Returns:
        Dict with selected center and scoring details
    """
    parts_availability = parts_availability or {}

    if not available_centers:
        return {"selected": None, "reason": "No available centers"}

    scores = []

    for center_id in available_centers:
        center_info = SIMULATED_CENTER_LOCATIONS.get(center_id, {})

        # Distance score (lower is better, normalize 0-1)
        distance = simulate_geo_distance(vehicle_lat, vehicle_lon, center_id)
        distance_score = max(0, 1 - (distance / 50))  # 50km = 0 score

        # Wait time score (lower is better)
        wait = center_info.get("avg_wait", 30)
        wait_score = max(0, 1 - (wait / 60))  # 60min = 0 score

        # Rating score (simulated 4.0-5.0)
        rating = 4.0 + random.random() * 1.0
        rating_score = (rating - 4.0) / 1.0  # Normalize to 0-1

        # Parts availability score
        has_parts = parts_availability.get(center_id, True)
        parts_score = 1.0 if has_parts else 0.0

        # Weighted total
        total_score = (
            distance_score * 0.4
            + wait_score * 0.3
            + rating_score * 0.2
            + parts_score * 0.1
        )

        scores.append(
            {
                "center_id": center_id,
                "center_name": center_info.get("name", center_id),
                "distance_km": distance,
                "wait_minutes": wait,
                "rating": round(rating, 1),
                "has_parts": has_parts,
                "total_score": round(total_score, 3),
            }
        )

    # Sort by score descending
    scores.sort(key=lambda x: x["total_score"], reverse=True)

    return {
        "selected": scores[0]["center_id"],
        "selected_name": scores[0]["center_name"],
        "score": scores[0]["total_score"],
        "all_options": scores,
    }


def dispatch_towing(
    vehicle_id: str,
    pickup_location: str,
    destination_center: str,
) -> TowingRequest:
    """
    Dispatch emergency towing service.

    Args:
        vehicle_id: Vehicle needing towing
        pickup_location: Current vehicle location
        destination_center: Service center ID

    Returns:
        TowingRequest with dispatch details
    """
    import uuid

    # Select random towing company (simulation)
    tow_company = random.choice(TOWING_COMPANIES)

    # Generate request
    request = TowingRequest(
        request_id=f"tow-{uuid.uuid4().hex[:8]}",
        vehicle_id=vehicle_id,
        pickup_location=pickup_location,
        destination_center=destination_center,
        status=TowingStatus.DISPATCHED,
        tow_company=tow_company["name"],
        estimated_eta_minutes=tow_company["base_eta"] + random.randint(-5, 10),
        driver_name=f"Driver {random.randint(100, 999)}",
        driver_phone=tow_company["phone"],
        dispatched_at=datetime.now().isoformat(),
    )

    return request


def forecast_demand(
    center_id: str,
    date: str,
    historical_bookings: List[Dict] = None,
) -> List[DemandForecast]:
    """
    Generate demand forecast for a service center.

    Args:
        center_id: Service center ID
        date: Target date (YYYY-MM-DD)
        historical_bookings: Optional historical data

    Returns:
        List of hourly demand forecasts
    """
    forecasts = []

    # Simulated demand pattern (typical day)
    # Peak hours: 9-11am, 2-4pm
    hourly_pattern = {
        8: 0.4,
        9: 0.7,
        10: 0.9,
        11: 0.8,
        12: 0.5,
        13: 0.4,
        14: 0.7,
        15: 0.8,
        16: 0.6,
        17: 0.3,
        18: 0.2,
    }

    for hour, base_demand in hourly_pattern.items():
        # Add some randomness
        demand = base_demand + random.uniform(-0.15, 0.15)
        demand = max(0, min(1, demand))

        # Recommendation based on demand
        if demand > 0.7:
            recommendation = "high_demand"
        elif demand > 0.4:
            recommendation = "normal"
        else:
            recommendation = "low_demand"

        forecasts.append(
            DemandForecast(
                center_id=center_id,
                date=date,
                hour=hour,
                predicted_demand=round(demand, 2),
                available_capacity=int((1 - demand) * 10),  # 10 slots max
                recommendation=recommendation,
            )
        )

    return forecasts


class SchedulingPriorityQueue:
    """
    Priority queue for managing appointment requests.
    Highest urgency requests are processed first.
    """

    def __init__(self):
        self._queue: List[PriorityAppointment] = []
        self._index = 0

    def push(self, appointment: PriorityAppointment):
        """Add appointment to queue."""
        heapq.heappush(self._queue, appointment)

    def pop(self) -> Optional[PriorityAppointment]:
        """Get and remove highest priority appointment."""
        if self._queue:
            return heapq.heappop(self._queue)
        return None

    def peek(self) -> Optional[PriorityAppointment]:
        """View highest priority without removing."""
        if self._queue:
            return self._queue[0]
        return None

    def size(self) -> int:
        """Get queue size."""
        return len(self._queue)

    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return len(self._queue) == 0

    def get_all(self) -> List[PriorityAppointment]:
        """Get all appointments sorted by priority."""
        return sorted(self._queue, key=lambda x: x.urgency_score, reverse=True)

    def clear(self):
        """Clear the queue."""
        self._queue = []
