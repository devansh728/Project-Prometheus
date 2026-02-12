"""
Workforce Management for ServiceOps AI
Handles mechanic assignment, skill matching, and shift optimization.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json
from pathlib import Path
from datetime import datetime, time, timedelta


@dataclass
class MechanicInfo:
    """Mechanic with workload and skill info."""

    id: str
    name: str
    service_center_id: str
    certifications: List[str]
    experience_years: int
    efficiency_rating: float
    current_jobs: int
    max_jobs_per_day: int
    is_available: bool
    skill_match_score: float


class WorkforceManager:
    """
    Manages workforce assignment, skill matching, and capacity.
    """

    # Skill requirements by failure type
    SKILL_REQUIREMENTS = {
        "brake_degradation": ["brake", "general"],
        "battery_degradation": ["ev_battery", "electrical"],
        "cooling_degradation": ["cooling", "general"],
        "electrical_fault": ["electrical", "ev_battery"],
        "general_maintenance": ["general"],
    }

    # Max daily jobs based on efficiency
    BASE_MAX_JOBS = 4

    def __init__(self, data_path: str = "data/fleet_seed.json"):
        self.data_path = Path(data_path)
        self.mechanics: Dict[str, Dict] = {}
        self.assignments: Dict[str, List[str]] = {}  # mechanic_id -> [job_ids]
        self._load_mechanics()

    def _load_mechanics(self):
        """Load mechanic data."""
        if self.data_path.exists():
            with open(self.data_path, "r") as f:
                data = json.load(f)
                for mech in data.get("mechanics", []):
                    self.mechanics[mech["id"]] = mech
                    self.assignments[mech["id"]] = []

    def find_available_mechanics(
        self, service_center_id: str, failure_type: Optional[str] = None, limit: int = 5
    ) -> List[MechanicInfo]:
        """
        Find available mechanics at a service center, sorted by skill match.
        """
        required_skills = self.SKILL_REQUIREMENTS.get(failure_type, ["general"])

        candidates = []

        for mech_id, mech in self.mechanics.items():
            if mech.get("service_center_id") != service_center_id:
                continue

            certs = mech.get("certifications", [])
            efficiency = mech.get("efficiency_rating", 1.0)
            experience = mech.get("experience_years", 1)

            # Calculate max jobs based on efficiency
            max_jobs = int(self.BASE_MAX_JOBS * efficiency)
            current_jobs = len(self.assignments.get(mech_id, []))
            is_available = current_jobs < max_jobs

            # Calculate skill match score
            matched_skills = set(required_skills) & set(certs)
            if required_skills:
                skill_score = len(matched_skills) / len(required_skills)
            else:
                skill_score = 1.0

            # Boost score based on experience
            experience_bonus = min(experience / 10, 0.2)
            skill_score = min(skill_score + experience_bonus, 1.0)

            candidates.append(
                MechanicInfo(
                    id=mech_id,
                    name=mech.get("name", ""),
                    service_center_id=service_center_id,
                    certifications=certs,
                    experience_years=experience,
                    efficiency_rating=efficiency,
                    current_jobs=current_jobs,
                    max_jobs_per_day=max_jobs,
                    is_available=is_available,
                    skill_match_score=round(skill_score, 3),
                )
            )

        # Sort by: available first, then by skill match score
        candidates.sort(key=lambda x: (not x.is_available, -x.skill_match_score))
        return candidates[:limit]

    def assign_job(self, mechanic_id: str, job_id: str) -> bool:
        """Assign a job to a mechanic."""
        if mechanic_id not in self.assignments:
            self.assignments[mechanic_id] = []

        mech = self.mechanics.get(mechanic_id)
        if not mech:
            return False

        max_jobs = int(self.BASE_MAX_JOBS * mech.get("efficiency_rating", 1.0))
        if len(self.assignments[mechanic_id]) >= max_jobs:
            return False

        self.assignments[mechanic_id].append(job_id)
        return True

    def complete_job(self, mechanic_id: str, job_id: str) -> bool:
        """Mark a job as complete for a mechanic."""
        if mechanic_id in self.assignments:
            if job_id in self.assignments[mechanic_id]:
                self.assignments[mechanic_id].remove(job_id)
                return True
        return False

    def get_workload_summary(self, service_center_id: str) -> Dict[str, Any]:
        """Get workload summary for a service center."""
        total_capacity = 0
        total_assigned = 0
        mechanics_available = 0

        for mech_id, mech in self.mechanics.items():
            if mech.get("service_center_id") != service_center_id:
                continue

            efficiency = mech.get("efficiency_rating", 1.0)
            max_jobs = int(self.BASE_MAX_JOBS * efficiency)
            current = len(self.assignments.get(mech_id, []))

            total_capacity += max_jobs
            total_assigned += current
            if current < max_jobs:
                mechanics_available += 1

        return {
            "service_center_id": service_center_id,
            "total_capacity": total_capacity,
            "total_assigned": total_assigned,
            "utilization": round(total_assigned / max(total_capacity, 1), 2),
            "mechanics_available": mechanics_available,
        }


# Singleton
workforce_manager = WorkforceManager()
