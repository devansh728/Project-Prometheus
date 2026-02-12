"""
Bidding Engine for ServiceOps AI
Implements internal center bidding for service request assignment.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

from serviceops_ai.services.geo_router import geo_router, ServiceCenterInfo
from serviceops_ai.services.workforce import workforce_manager
from serviceops_ai.services.inventory import inventory_manager
from serviceops_ai.services.priority_queue import ServiceRequest


@dataclass
class CenterBid:
    """A bid from a service center for a service request."""

    center_id: str
    center_name: str
    distance_km: float
    estimated_cost: float
    workload_score: float  # 0-1, lower is better (less busy)
    skill_score: float  # 0-1, higher is better
    inventory_score: float  # 0-1, higher is better
    overall_bid_score: float  # 0-100, higher is better
    est_days_to_complete: int
    is_historical: bool  # True if this was user's previous center
    reasoning: str  # Explanation for judges

    # Raw metrics for visualization
    available_mechanics: int
    load_percentage: float
    parts_available: bool


class BiddingEngine:
    """
    Implements internal bidding between eligible service centers.
    Each center computes a bid based on cost, skill, load, inventory.
    """

    # Weights for bid scoring
    WEIGHTS = {
        "distance": 0.15,
        "cost": 0.20,
        "workload": 0.25,
        "skill": 0.25,
        "inventory": 0.15,
    }

    # Historical center preference bonus
    HISTORICAL_BONUS = 10

    def __init__(self):
        self._bid_history: Dict[str, List[CenterBid]] = {}  # request_id -> bids

    def get_eligible_centers(
        self,
        request: ServiceRequest,
        max_distance_km: float = 50.0,
    ) -> List[ServiceCenterInfo]:
        """
        Find centers eligible to bid on this request.
        """
        # Map failure type to capability
        capability_map = {
            "brake_degradation": "brake",
            "brake_fade": "brake",
            "battery_degradation": "ev_battery",
            "cooling_degradation": "cooling",
            "electrical_fault": "electrical",
            "general_maintenance": "general",
        }
        required_capability = capability_map.get(request.failure_type, "general")

        # Find nearby centers with required capability
        centers = geo_router.find_nearest_centers(
            request.geo_lat,
            request.geo_lon,
            required_capability=required_capability,
            max_distance_km=max_distance_km,
            limit=10,
        )

        return centers

    def compute_bid(
        self,
        center: ServiceCenterInfo,
        request: ServiceRequest,
    ) -> CenterBid:
        """
        Compute a bid from a single center for a request.
        """
        # Get workforce data
        mechanics = workforce_manager.find_available_mechanics(
            center.id, failure_type=request.failure_type, limit=5
        )
        workload = workforce_manager.get_workload_summary(center.id)

        # Get inventory data
        parts_check = inventory_manager.check_parts_availability(
            center.id, request.failure_type
        )

        # Calculate sub-scores (all normalized to 0-1)
        # Distance score: closer is better
        max_dist = 50.0
        distance_score = max(0, 1 - (center.distance_km / max_dist))

        # Cost score: lower cost is better (invert and normalize)
        max_cost = 50000
        cost = parts_check.get("estimated_parts_cost", 0)
        cost_score = max(0, 1 - (cost / max_cost))

        # Workload score: lower utilization is better
        workload_score = 1 - workload.get("utilization", 0.5)

        # Skill score: from mechanic matching
        skill_score = mechanics[0].skill_match_score if mechanics else 0.5

        # Inventory score
        inventory_score = 1.0 if parts_check.get("all_parts_available") else 0.4

        # Compute weighted overall score
        overall = (
            distance_score * self.WEIGHTS["distance"]
            + cost_score * self.WEIGHTS["cost"]
            + workload_score * self.WEIGHTS["workload"]
            + skill_score * self.WEIGHTS["skill"]
            + inventory_score * self.WEIGHTS["inventory"]
        ) * 100

        # Historical center bonus
        is_historical = request.historical_center_id == center.id
        if is_historical:
            overall += self.HISTORICAL_BONUS

        overall = min(100, overall)

        # Estimate days to complete
        est_days = 1 if workload.get("utilization", 0.5) < 0.7 else 2
        if not parts_check.get("all_parts_available"):
            est_days += 1

        # Generate reasoning
        reasoning = self._generate_reasoning(
            center,
            distance_score,
            cost_score,
            workload_score,
            skill_score,
            inventory_score,
            is_historical,
        )

        return CenterBid(
            center_id=center.id,
            center_name=center.name,
            distance_km=center.distance_km,
            estimated_cost=cost,
            workload_score=round(workload_score, 2),
            skill_score=round(skill_score, 2),
            inventory_score=round(inventory_score, 2),
            overall_bid_score=round(overall, 1),
            est_days_to_complete=est_days,
            is_historical=is_historical,
            reasoning=reasoning,
            available_mechanics=len([m for m in mechanics if m.is_available]),
            load_percentage=round(workload.get("utilization", 0.5) * 100, 1),
            parts_available=parts_check.get("all_parts_available", False),
        )

    def _generate_reasoning(
        self,
        center: ServiceCenterInfo,
        distance_score: float,
        cost_score: float,
        workload_score: float,
        skill_score: float,
        inventory_score: float,
        is_historical: bool,
    ) -> str:
        """Generate human-readable reasoning for the bid."""
        factors = []

        if distance_score > 0.7:
            factors.append("close proximity")
        elif distance_score < 0.3:
            factors.append("distant location")

        if workload_score > 0.6:
            factors.append("low current load")
        elif workload_score < 0.3:
            factors.append("high workload")

        if skill_score > 0.8:
            factors.append("excellent skill match")
        elif skill_score < 0.5:
            factors.append("limited skill match")

        if inventory_score >= 1.0:
            factors.append("all parts in stock")
        else:
            factors.append("some parts need ordering")

        if is_historical:
            factors.append("customer's preferred center")

        if factors:
            return f"{center.name}: {', '.join(factors)}"
        return f"{center.name}: standard evaluation"

    def generate_bids(
        self,
        request: ServiceRequest,
        max_distance_km: float = 50.0,
    ) -> List[CenterBid]:
        """
        Generate bids from all eligible centers for a request.
        """
        centers = self.get_eligible_centers(request, max_distance_km)

        bids = []
        for center in centers:
            bid = self.compute_bid(center, request)
            bids.append(bid)

        # Sort by overall score descending
        bids.sort(key=lambda b: b.overall_bid_score, reverse=True)

        # Store for history
        self._bid_history[request.request_id] = bids

        return bids

    def select_winner(self, bids: List[CenterBid]) -> Optional[CenterBid]:
        """
        Select winning bid. Not necessarily cheapest - balances all factors.
        """
        if not bids:
            return None

        # Best overall score wins
        return bids[0]

    def get_bid_history(self, request_id: str) -> List[Dict[str, Any]]:
        """Get bid details for a specific request."""
        bids = self._bid_history.get(request_id, [])
        return [
            {
                "center_id": b.center_id,
                "center_name": b.center_name,
                "distance_km": b.distance_km,
                "estimated_cost": b.estimated_cost,
                "workload_score": b.workload_score,
                "skill_score": b.skill_score,
                "inventory_score": b.inventory_score,
                "overall_bid_score": b.overall_bid_score,
                "est_days_to_complete": b.est_days_to_complete,
                "is_historical": b.is_historical,
                "reasoning": b.reasoning,
                "available_mechanics": b.available_mechanics,
                "load_percentage": b.load_percentage,
                "parts_available": b.parts_available,
            }
            for b in bids
        ]

    def get_all_history(self) -> Dict[str, List[Dict]]:
        """Get all bid histories."""
        return {
            req_id: self.get_bid_history(req_id) for req_id in self._bid_history.keys()
        }


# Singleton
bidding_engine = BiddingEngine()
