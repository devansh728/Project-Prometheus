"""
RAG-based Repair Recommendation Service
========================================
Retrieves relevant repair procedures based on detected faults and provides recommendations.

Usage:
    from rag_recommender import RecommendationEngine
    engine = RecommendationEngine()
    recommendations = engine.get_recommendations(alert, vehicle_profile)
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import numpy as np


BASE_DIR = Path(__file__).parent.parent
KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base"


@dataclass
class VehicleProfile:
    """Digital twin vehicle profile."""

    vehicle_id: str
    vehicle_type: str
    mileage_km: float = 0
    battery_health_pct: float = 100
    last_service_date: str = ""
    fault_history: List[str] = field(default_factory=list)
    preferred_service_center: str = ""
    owner_contact: str = ""


@dataclass
class Recommendation:
    """Repair recommendation."""

    severity: str
    title: str
    summary: str
    symptoms: List[str]
    repair_procedure: List[str]
    estimated_time: str
    cost_range: str
    parts_needed: List[str]
    dtc_codes: List[str]
    urgency_message: str


class KnowledgeBase:
    """Loads and indexes repair manual knowledge."""

    def __init__(self, manual_path: Optional[Path] = None):
        self.manual_path = manual_path or KNOWLEDGE_BASE_DIR / "ev_repair_manual.json"
        self.manual = {}
        self.topic_index: Dict[str, dict] = {}
        self.dtc_index: Dict[str, dict] = {}
        self.symptom_keywords: Dict[str, List[str]] = {}
        self._load_manual()

    def _load_manual(self):
        """Load and index the repair manual."""
        if not self.manual_path.exists():
            print(f"Warning: Manual not found at {self.manual_path}")
            return

        with open(self.manual_path, "r") as f:
            self.manual = json.load(f)

        # Index topics by ID and DTC codes
        for section in self.manual.get("sections", []):
            for topic in section.get("topics", []):
                topic_id = topic["id"]
                self.topic_index[topic_id] = topic

                # Index by DTC codes
                for dtc in topic.get("dtc_codes", []):
                    self.dtc_index[dtc] = topic

                # Index keywords from symptoms
                keywords = []
                for symptom in topic.get("symptoms", []):
                    keywords.extend(symptom.lower().split())
                self.symptom_keywords[topic_id] = keywords

        print(f"Loaded {len(self.topic_index)} repair topics")
        print(f"Indexed {len(self.dtc_index)} DTC codes")

    def search_by_dtc(self, dtc_codes: List[str]) -> List[dict]:
        """Find repair topics matching DTC codes."""
        results = []
        for dtc in dtc_codes:
            if dtc in self.dtc_index:
                topic = self.dtc_index[dtc]
                if topic not in results:
                    results.append(topic)
        return results

    def search_by_failure_type(self, failure_type: str) -> Optional[dict]:
        """Find repair topic by failure type ID."""
        return self.topic_index.get(failure_type)

    def search_by_keywords(self, keywords: List[str]) -> List[dict]:
        """Search topics by symptom keywords."""
        scores = {}
        for topic_id, topic_keywords in self.symptom_keywords.items():
            score = sum(1 for kw in keywords if kw.lower() in topic_keywords)
            if score > 0:
                scores[topic_id] = score

        # Return topics sorted by relevance
        sorted_topics = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [self.topic_index[tid] for tid, _ in sorted_topics[:3]]


class VehicleProfileManager:
    """Manages vehicle digital twin profiles."""

    def __init__(self):
        self.profiles: Dict[str, VehicleProfile] = {}

    def get_or_create(self, vehicle_id: str, **kwargs) -> VehicleProfile:
        """Get existing profile or create new one."""
        if vehicle_id not in self.profiles:
            self.profiles[vehicle_id] = VehicleProfile(
                vehicle_id=vehicle_id,
                vehicle_type=kwargs.get("vehicle_type", "sedan"),
                mileage_km=kwargs.get("mileage_km", 0),
            )
        return self.profiles[vehicle_id]

    def update_fault_history(self, vehicle_id: str, fault_type: str):
        """Add fault to vehicle history."""
        profile = self.get_or_create(vehicle_id)
        if fault_type not in profile.fault_history:
            profile.fault_history.append(fault_type)

    def get_personalized_priority(self, vehicle_id: str, fault_type: str) -> str:
        """Get priority adjustment based on vehicle history."""
        profile = self.get_or_create(vehicle_id)

        # Higher priority if fault has occurred before
        if fault_type in profile.fault_history:
            return "elevated"

        # Higher priority for older vehicles
        if profile.mileage_km > 100000:
            return "elevated"

        return "normal"


class RecommendationEngine:
    """Main engine for generating repair recommendations."""

    def __init__(self):
        self.knowledge_base = KnowledgeBase()
        self.profile_manager = VehicleProfileManager()

        # Failure type to topic mapping
        self.failure_type_map = {
            "battery_degradation": "battery_degradation",
            "motor_bearing_wear": "motor_bearing_wear",
            "inverter_overheating": "inverter_overheating",
            "thermal_runaway_risk": "thermal_runaway",
            "charging_fault": "charging_fault",
        }

    def get_recommendations(
        self,
        failure_type: str,
        severity: str,
        dtc_codes: List[str] = None,
        vehicle_id: str = None,
    ) -> Recommendation:
        """Generate recommendations based on detected fault."""

        dtc_codes = dtc_codes or []

        # Search knowledge base
        topic = None

        # First try by failure type
        topic_id = self.failure_type_map.get(failure_type)
        if topic_id:
            topic = self.knowledge_base.search_by_failure_type(topic_id)

        # Fallback to DTC search
        if not topic and dtc_codes:
            topics = self.knowledge_base.search_by_dtc(dtc_codes)
            if topics:
                topic = topics[0]

        # Default topic if nothing found
        if not topic:
            topic = self._get_default_topic(failure_type)

        # Get severity-specific message
        severity_levels = topic.get("severity_levels", {})
        severity_message = severity_levels.get(
            severity, severity_levels.get("medium", "Inspection recommended")
        )

        # Check vehicle history for personalization
        priority = "normal"
        if vehicle_id:
            priority = self.profile_manager.get_personalized_priority(
                vehicle_id, failure_type
            )
            self.profile_manager.update_fault_history(vehicle_id, failure_type)

        # Build urgency message
        urgency_map = {
            "low": "Schedule service at your convenience within the next 2 weeks.",
            "medium": "Service recommended within 1 week. Monitor symptoms closely.",
            "high": "Urgent: Service recommended within 48 hours. Limit driving.",
            "critical": "CRITICAL: Stop driving immediately. Contact emergency services if smoke/fire detected.",
        }
        urgency = urgency_map.get(severity, urgency_map["medium"])

        if priority == "elevated":
            urgency = f"⚠️ Recurring issue detected. {urgency}"

        return Recommendation(
            severity=severity,
            title=topic.get("title", failure_type),
            summary=severity_message,
            symptoms=topic.get("symptoms", []),
            repair_procedure=topic.get("repair_procedures", []),
            estimated_time=topic.get("estimated_time", "Unknown"),
            cost_range=topic.get("cost_range", "Varies"),
            parts_needed=topic.get("parts", []),
            dtc_codes=topic.get("dtc_codes", []),
            urgency_message=urgency,
        )

    def _get_default_topic(self, failure_type: str) -> dict:
        """Return default topic for unknown failure types."""
        return {
            "id": failure_type,
            "title": f"System Alert: {failure_type.replace('_', ' ').title()}",
            "symptoms": ["Abnormal sensor readings detected"],
            "dtc_codes": [],
            "severity_levels": {
                "low": "Minor deviation detected - monitor",
                "medium": "Moderate issue - schedule inspection",
                "high": "Significant issue - prompt service required",
            },
            "repair_procedures": [
                "1. Run full vehicle diagnostic scan",
                "2. Inspect flagged subsystem",
                "3. Consult service manual for specific codes",
                "4. Contact authorized service center",
            ],
            "parts": ["TBD based on diagnosis"],
            "estimated_time": "1-4 hours",
            "cost_range": "Varies based on diagnosis",
        }

    def format_recommendation(self, rec: Recommendation) -> str:
        """Format recommendation as readable text."""
        lines = [
            f"🔧 {rec.title}",
            f"━" * 40,
            f"⚠️  Severity: {rec.severity.upper()}",
            f"",
            f"📋 Summary: {rec.summary}",
            f"",
            f"🚨 {rec.urgency_message}",
            f"",
            f"🔍 Symptoms to verify:",
        ]
        for s in rec.symptoms[:3]:
            lines.append(f"   • {s}")

        lines.extend([f"", f"🛠️  Repair Steps:"])
        for step in rec.repair_procedure[:4]:
            lines.append(f"   {step}")

        lines.extend(
            [
                f"",
                f"⏱️  Estimated Time: {rec.estimated_time}",
                f"💰 Cost Range: {rec.cost_range}",
                f"",
                f"🔩 Parts Needed: {', '.join(rec.parts_needed[:3])}",
            ]
        )

        if rec.dtc_codes:
            lines.append(f"📟 Related DTCs: {', '.join(rec.dtc_codes)}")

        return "\n".join(lines)


def main():
    """Demo the recommendation engine."""
    print("=" * 60)
    print("RAG Recommendation Engine Demo")
    print("=" * 60)

    engine = RecommendationEngine()

    # Test cases
    test_cases = [
        ("battery_degradation", "medium", ["P0A80"], "EV_001"),
        ("inverter_overheating", "high", ["P0A78"], "EV_002"),
        ("motor_bearing_wear", "low", [], "EV_003"),
    ]

    for failure_type, severity, dtcs, vehicle_id in test_cases:
        print(f"\n{'='*60}")
        print(f"Test: {failure_type} | Severity: {severity} | Vehicle: {vehicle_id}")
        print("=" * 60)

        rec = engine.get_recommendations(
            failure_type=failure_type,
            severity=severity,
            dtc_codes=dtcs,
            vehicle_id=vehicle_id,
        )

        print(engine.format_recommendation(rec))


if __name__ == "__main__":
    main()
