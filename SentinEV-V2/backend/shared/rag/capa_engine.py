"""
CAPA Pattern Detection Engine
Detects Corrective and Preventive Action patterns from historical data.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class CAPAPattern:
    """A detected CAPA pattern."""

    pattern_id: str
    pattern_name: str
    description: str
    match_confidence: float
    affected_vehicles: List[str]
    root_cause: str
    corrective_action: str
    preventive_action: str
    severity: str


class CAPAEngine:
    """
    Detects CAPA patterns based on symptoms, DTC codes, and vehicle data.
    Uses pattern matching against known CAPA cases.
    """

    # Known CAPA patterns (in production, would load from database/RAG)
    KNOWN_PATTERNS = {
        "CAPA-2025-001": {
            "name": "Brake Pad Premature Wear",
            "description": "Brake pad wear rates 2.3x higher than specification in aggressive driving profiles",
            "triggers": {
                "dtc_codes": ["C0035", "C0040"],
                "symptoms": ["brake", "wear", "vibration", "pad"],
                "vehicle_makes": ["Mahindra", "MG"],
                "driving_profile": ["aggressive"],
            },
            "severity": "HIGH",
            "root_cause": "Brake pad compound had lower thermal resistance than specified",
            "corrective_action": "Replace brake pads with updated compound (Part# BP-2023-B)",
            "preventive_action": "Updated brake pad specification, added thermal cycling test",
        },
        "CAPA-2025-002": {
            "name": "Battery Thermal Runaway in Hot Climates",
            "description": "Thermal runaway incidents in vehicles operated in ambient temperatures > 42°C",
            "triggers": {
                "dtc_codes": ["P0A80", "P0A0D", "P0ABF"],
                "symptoms": [
                    "thermal",
                    "battery",
                    "temperature",
                    "overheating",
                    "charging",
                ],
                "vehicle_makes": ["Mahindra", "BYD"],
                "driving_profile": [],
            },
            "severity": "CRITICAL",
            "root_cause": "Cooling pump efficiency degradation at high temperatures",
            "corrective_action": "Software update: Limit DC fast charge power to 80% when ambient > 38°C",
            "preventive_action": "Proactive cooling system inspection campaign",
        },
        "CAPA-2025-003": {
            "name": "Coolant System Leak Pattern",
            "description": "Recurring coolant leaks in vehicles with 50k+ km mileage",
            "triggers": {
                "dtc_codes": ["P0128", "P0116"],
                "symptoms": ["coolant", "leak", "temperature", "overheating"],
                "vehicle_makes": [],
                "mileage_threshold": 50000,
            },
            "severity": "WARNING",
            "root_cause": "Expansion tank degradation at high mileage",
            "corrective_action": "Replace expansion tank and inspect radiator hoses",
            "preventive_action": "Proactive replacement at 45k km for affected vehicles",
        },
    }

    def __init__(self):
        self.patterns = self.KNOWN_PATTERNS

    def detect_patterns(
        self,
        symptoms: List[str],
        dtc_codes: List[str] = None,
        vehicle_make: str = None,
        vehicle_mileage: int = 0,
        driving_profile: str = None,
    ) -> List[CAPAPattern]:
        """
        Detect matching CAPA patterns based on input criteria.

        Returns:
            List of matching patterns with confidence scores
        """
        matches = []

        symptoms_lower = [s.lower() for s in symptoms] if symptoms else []
        dtc_codes = dtc_codes or []

        for pattern_id, pattern in self.patterns.items():
            confidence = self._calculate_match_confidence(
                pattern=pattern,
                symptoms=symptoms_lower,
                dtc_codes=dtc_codes,
                vehicle_make=vehicle_make,
                vehicle_mileage=vehicle_mileage,
                driving_profile=driving_profile,
            )

            if confidence >= 0.3:  # Minimum threshold
                matches.append(
                    CAPAPattern(
                        pattern_id=pattern_id,
                        pattern_name=pattern["name"],
                        description=pattern["description"],
                        match_confidence=confidence,
                        affected_vehicles=pattern["triggers"].get("vehicle_makes", []),
                        root_cause=pattern["root_cause"],
                        corrective_action=pattern["corrective_action"],
                        preventive_action=pattern["preventive_action"],
                        severity=pattern["severity"],
                    )
                )

        # Sort by confidence
        matches.sort(key=lambda x: x.match_confidence, reverse=True)
        return matches

    def _calculate_match_confidence(
        self,
        pattern: Dict,
        symptoms: List[str],
        dtc_codes: List[str],
        vehicle_make: str,
        vehicle_mileage: int,
        driving_profile: str,
    ) -> float:
        """Calculate match confidence for a pattern."""
        triggers = pattern.get("triggers", {})
        scores = []
        weights = []

        # DTC code matching (high weight)
        pattern_dtcs = set(triggers.get("dtc_codes", []))
        if pattern_dtcs:
            matching_dtcs = pattern_dtcs.intersection(set(dtc_codes))
            dtc_score = len(matching_dtcs) / len(pattern_dtcs)
            scores.append(dtc_score)
            weights.append(0.4)

        # Symptom matching
        pattern_symptoms = triggers.get("symptoms", [])
        if pattern_symptoms:
            matching_symptoms = sum(
                1 for s in pattern_symptoms if any(s in symptom for symptom in symptoms)
            )
            symptom_score = matching_symptoms / len(pattern_symptoms)
            scores.append(symptom_score)
            weights.append(0.3)

        # Vehicle make matching
        pattern_makes = triggers.get("vehicle_makes", [])
        if pattern_makes and vehicle_make:
            make_score = 1.0 if vehicle_make in pattern_makes else 0.0
            scores.append(make_score)
            weights.append(0.15)

        # Driving profile matching
        pattern_profiles = triggers.get("driving_profile", [])
        if pattern_profiles and driving_profile:
            profile_score = 1.0 if driving_profile in pattern_profiles else 0.0
            scores.append(profile_score)
            weights.append(0.1)

        # Mileage threshold
        mileage_threshold = triggers.get("mileage_threshold")
        if mileage_threshold and vehicle_mileage:
            mileage_score = 1.0 if vehicle_mileage >= mileage_threshold else 0.0
            scores.append(mileage_score)
            weights.append(0.05)

        # Calculate weighted average
        if not scores:
            return 0.0

        total_weight = sum(weights)
        weighted_sum = sum(s * w for s, w in zip(scores, weights))
        return round(weighted_sum / total_weight, 3) if total_weight > 0 else 0.0

    def get_preventive_recommendations(
        self, vehicle_make: str, vehicle_mileage: int
    ) -> List[Dict[str, Any]]:
        """
        Get proactive preventive recommendations based on vehicle profile.
        """
        recommendations = []

        for pattern_id, pattern in self.patterns.items():
            triggers = pattern.get("triggers", {})

            # Check if vehicle is in affected list
            makes = triggers.get("vehicle_makes", [])
            if makes and vehicle_make not in makes:
                continue

            # Check mileage threshold
            mileage_threshold = triggers.get("mileage_threshold")
            if mileage_threshold:
                if vehicle_mileage >= mileage_threshold * 0.9:  # 90% of threshold
                    recommendations.append(
                        {
                            "pattern_id": pattern_id,
                            "pattern_name": pattern["name"],
                            "recommendation": pattern["preventive_action"],
                            "urgency": (
                                "high"
                                if vehicle_mileage >= mileage_threshold
                                else "medium"
                            ),
                            "mileage_threshold": mileage_threshold,
                        }
                    )

        return recommendations


# Singleton
capa_engine = CAPAEngine()
