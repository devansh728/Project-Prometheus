"""
Scheduling Engine for ServiceOps AI
Coordinates the full scheduling flow: geo-routing, workforce, inventory, and lifecycle.
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

from serviceops_ai.services.geo_router import geo_router, ServiceCenterInfo
from serviceops_ai.services.workforce import workforce_manager, MechanicInfo
from serviceops_ai.services.inventory import inventory_manager
from serviceops_ai.services.lifecycle import service_lifecycle, ServiceState, ServiceJob


@dataclass
class SchedulingSlot:
    """A proposed scheduling slot."""

    service_center: ServiceCenterInfo
    mechanic: MechanicInfo
    available_at: datetime
    estimated_duration_minutes: int
    parts_available: bool
    estimated_cost: float
    recommendation_score: float


@dataclass
class SchedulingResult:
    """Result of scheduling attempt."""

    success: bool
    job: Optional[ServiceJob] = None
    slot: Optional[SchedulingSlot] = None
    alternatives: List[SchedulingSlot] = None
    error: Optional[str] = None


class SchedulingEngine:
    """
    Orchestrates the complete scheduling process:
    1. Find nearby service centers (geo-routing)
    2. Check workforce availability (skill matching)
    3. Verify parts inventory
    4. Create service job
    """

    # Estimated repair duration by failure type (minutes)
    REPAIR_DURATIONS = {
        "brake_degradation": 120,
        "battery_degradation": 180,
        "cooling_degradation": 90,
        "general_maintenance": 60,
    }

    def find_slots(
        self,
        customer_lat: float,
        customer_lon: float,
        failure_type: str,
        severity: str,
        max_distance_km: float = 50.0,
        limit: int = 3,
    ) -> List[SchedulingSlot]:
        """
        Find available scheduling slots across service centers.
        """
        slots = []

        # Get required capability based on failure type
        capability_map = {
            "brake_degradation": "brake",
            "battery_degradation": "ev_battery",
            "cooling_degradation": "cooling",
        }
        required_capability = capability_map.get(failure_type, "general")

        # Find nearby service centers
        centers = geo_router.find_nearest_centers(
            customer_lat,
            customer_lon,
            required_capability=required_capability,
            max_distance_km=max_distance_km,
            limit=limit * 2,  # Get extra to filter
        )

        for center in centers:
            # Find available mechanics
            mechanics = workforce_manager.find_available_mechanics(
                center.id, failure_type=failure_type, limit=2
            )

            if not mechanics or not mechanics[0].is_available:
                continue

            best_mechanic = mechanics[0]

            # Check parts availability
            parts_check = inventory_manager.check_parts_availability(
                center.id, failure_type
            )

            # Calculate slot score
            parts_score = 1.0 if parts_check["all_parts_available"] else 0.5
            mechanic_score = best_mechanic.skill_match_score

            slot_score = (
                center.recommendation_score * 0.4
                + mechanic_score * 0.4
                + parts_score * 0.2
            )

            # Estimate available time (simulate current queue)
            queue_delay = int((1 - center.load_factor) * 120)  # 0-120 min based on load
            available_at = datetime.utcnow() + timedelta(minutes=max(30, queue_delay))

            slots.append(
                SchedulingSlot(
                    service_center=center,
                    mechanic=best_mechanic,
                    available_at=available_at,
                    estimated_duration_minutes=self.REPAIR_DURATIONS.get(
                        failure_type, 90
                    ),
                    parts_available=parts_check["all_parts_available"],
                    estimated_cost=parts_check["estimated_parts_cost"],
                    recommendation_score=round(slot_score, 3),
                )
            )

        # Sort by recommendation score
        slots.sort(key=lambda x: x.recommendation_score, reverse=True)
        return slots[:limit]

    def schedule(
        self,
        vehicle_id: str,
        customer_id: str,
        customer_lat: float,
        customer_lon: float,
        failure_type: str,
        severity: str,
        preferred_datetime: Optional[datetime] = None,
    ) -> SchedulingResult:
        """
        Complete scheduling flow: find best slot and create job.
        """
        # Find available slots
        slots = self.find_slots(
            customer_lat, customer_lon, failure_type, severity, limit=5
        )

        if not slots:
            return SchedulingResult(
                success=False,
                error="No available service centers found within range",
                alternatives=[],
            )

        # Select best slot
        best_slot = slots[0]

        # Verify parts can be allocated
        if not best_slot.parts_available:
            # Try to find alternative with parts
            for slot in slots[1:]:
                if slot.parts_available:
                    best_slot = slot
                    break

        # Create service job
        job = service_lifecycle.create_job(
            vehicle_id=vehicle_id,
            customer_id=customer_id,
            service_center_id=best_slot.service_center.id,
            failure_type=failure_type,
            severity=severity,
        )

        # Transition to BOOKED
        scheduled_at = preferred_datetime or best_slot.available_at
        success, message = service_lifecycle.transition(
            job.job_id,
            ServiceState.BOOKED,
            metadata={
                "scheduled_at": scheduled_at,
                "mechanic_id": best_slot.mechanic.id,
            },
        )

        if not success:
            return SchedulingResult(
                success=False, error=message, alternatives=slots[1:]
            )

        # Assign job to mechanic
        workforce_manager.assign_job(best_slot.mechanic.id, job.job_id)

        # Allocate parts if available
        if best_slot.parts_available:
            inventory_manager.allocate_parts(
                best_slot.service_center.id, failure_type, job.job_id
            )

        return SchedulingResult(
            success=True, job=job, slot=best_slot, alternatives=slots[1:]
        )

    def get_slot_details(self, slot: SchedulingSlot) -> Dict[str, Any]:
        """Convert slot to JSON-serializable dict."""
        return {
            "service_center": {
                "id": slot.service_center.id,
                "name": slot.service_center.name,
                "distance_km": slot.service_center.distance_km,
                "quality_rating": slot.service_center.quality_rating,
            },
            "mechanic": {
                "id": slot.mechanic.id,
                "name": slot.mechanic.name,
                "skill_match_score": slot.mechanic.skill_match_score,
            },
            "available_at": slot.available_at.isoformat(),
            "estimated_duration_minutes": slot.estimated_duration_minutes,
            "parts_available": slot.parts_available,
            "estimated_cost": slot.estimated_cost,
            "recommendation_score": slot.recommendation_score,
        }


# Singleton
scheduling_engine = SchedulingEngine()
