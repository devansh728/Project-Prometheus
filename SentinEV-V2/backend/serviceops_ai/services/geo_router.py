"""
Geo-routing Service for ServiceOps AI
Implements Haversine distance calculation and service center selection.
"""

from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import math
import json
from pathlib import Path


@dataclass
class ServiceCenterInfo:
    """Service center with distance and capacity info."""

    id: str
    name: str
    lat: float
    lon: float
    distance_km: float
    capabilities: List[str]
    quality_rating: float
    available_bays: int
    total_bays: int
    load_factor: float  # 0-1, how busy
    recommendation_score: float


class GeoRouter:
    """
    Handles geo-routing and service center selection.
    Uses Haversine formula for distance calculation.
    """

    EARTH_RADIUS_KM = 6371.0

    def __init__(self, data_path: str = "data/fleet_seed.json"):
        self.data_path = Path(data_path)
        self.service_centers: Dict[str, Dict] = {}
        self._load_centers()

    def _load_centers(self):
        """Load service center data."""
        if self.data_path.exists():
            with open(self.data_path, "r") as f:
                data = json.load(f)
                for sc in data.get("service_centers", []):
                    self.service_centers[sc["id"]] = sc

    def haversine_distance(
        self, lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """
        Calculate distance between two coordinates using Haversine formula.
        Returns distance in kilometers.
        """
        # Convert to radians
        lat1_r = math.radians(lat1)
        lat2_r = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)

        # Haversine formula
        a = (
            math.sin(delta_lat / 2) ** 2
            + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(delta_lon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return self.EARTH_RADIUS_KM * c

    def find_nearest_centers(
        self,
        customer_lat: float,
        customer_lon: float,
        required_capability: Optional[str] = None,
        max_distance_km: float = 100.0,
        limit: int = 5,
    ) -> List[ServiceCenterInfo]:
        """
        Find nearest service centers sorted by recommendation score.
        """
        candidates = []

        for sc_id, sc in self.service_centers.items():
            # Check capability if required
            if required_capability:
                capabilities = sc.get("capabilities", [])
                if (
                    required_capability not in capabilities
                    and "general" not in capabilities
                ):
                    continue

            # Calculate distance
            distance = self.haversine_distance(
                customer_lat, customer_lon, sc.get("lat", 0), sc.get("lon", 0)
            )

            if distance > max_distance_km:
                continue

            # Simulate capacity (in production would query real-time data)
            total_bays = sc.get("num_bays", 4)
            # Simulate 40-80% load
            load_factor = 0.4 + (hash(sc_id) % 40) / 100
            available = int(total_bays * (1 - load_factor))

            # Calculate recommendation score
            # Lower distance = better, higher quality = better, lower load = better
            distance_score = max(0, 1 - distance / max_distance_km)
            quality_score = sc.get("quality_rating", 4.0) / 5.0
            capacity_score = 1 - load_factor

            recommendation_score = (
                distance_score * 0.4 + quality_score * 0.35 + capacity_score * 0.25
            )

            candidates.append(
                ServiceCenterInfo(
                    id=sc_id,
                    name=sc.get("name", ""),
                    lat=sc.get("lat", 0),
                    lon=sc.get("lon", 0),
                    distance_km=round(distance, 2),
                    capabilities=sc.get("capabilities", []),
                    quality_rating=sc.get("quality_rating", 4.0),
                    available_bays=available,
                    total_bays=total_bays,
                    load_factor=round(load_factor, 2),
                    recommendation_score=round(recommendation_score, 3),
                )
            )

        # Sort by recommendation score (highest first)
        candidates.sort(key=lambda x: x.recommendation_score, reverse=True)
        return candidates[:limit]

    def get_center(self, center_id: str) -> Optional[Dict]:
        """Get service center by ID."""
        return self.service_centers.get(center_id)


# Singleton
geo_router = GeoRouter()
