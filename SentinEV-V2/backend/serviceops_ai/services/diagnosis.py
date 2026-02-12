"""
Diagnosis Feedback & RCA/CAPA for ServiceOps AI
Computes diagnosis similarity and generates insights for manufacturing.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


@dataclass
class DiagnosisRecord:
    """Record of predicted vs actual diagnosis."""

    job_id: str
    vehicle_id: str
    predicted_failure: str
    predicted_severity: str
    actual_failure: str
    actual_severity: str
    similarity_score: float  # 0.0 - 1.0
    diagnosis_notes: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


class DiagnosisFeedback:
    """
    Handles diagnosis comparison and generates RCA/CAPA insights.
    """

    # Failure type similarity matrix (1.0 = same, 0.0 = unrelated)
    SIMILARITY_MATRIX = {
        ("brake_degradation", "brake_fade"): 0.9,
        ("brake_fade", "brake_degradation"): 0.9,
        ("battery_degradation", "electrical_fault"): 0.6,
        ("electrical_fault", "battery_degradation"): 0.6,
        ("cooling_degradation", "battery_degradation"): 0.4,
    }

    # Severity weights
    SEVERITY_WEIGHTS = {
        "critical": 4,
        "high": 3,
        "medium": 2,
        "low": 1,
    }

    def __init__(self):
        self._records: List[DiagnosisRecord] = []
        self._patterns: Dict[str, int] = {}  # Pattern -> count

    def compute_similarity(
        self,
        predicted: Dict[str, Any],
        actual: Dict[str, Any],
    ) -> float:
        """
        Compute similarity score between predicted and actual diagnosis.
        Returns 0.0 - 1.0.
        """
        pred_failure = predicted.get("failure_type", "").lower()
        actual_failure = actual.get("failure_type", "").lower()
        pred_severity = predicted.get("severity", "medium").lower()
        actual_severity = actual.get("severity", "medium").lower()

        # Failure type match (70% weight)
        if pred_failure == actual_failure:
            failure_score = 1.0
        else:
            key = (pred_failure, actual_failure)
            failure_score = self.SIMILARITY_MATRIX.get(key, 0.2)

        # Severity match (30% weight)
        pred_weight = self.SEVERITY_WEIGHTS.get(pred_severity, 2)
        actual_weight = self.SEVERITY_WEIGHTS.get(actual_severity, 2)
        severity_diff = abs(pred_weight - actual_weight)
        severity_score = max(0, 1 - (severity_diff * 0.25))

        total = (failure_score * 0.7) + (severity_score * 0.3)
        return round(total, 2)

    def log_feedback(
        self,
        job_id: str,
        vehicle_id: str,
        predicted: Dict[str, Any],
        actual: Dict[str, Any],
        notes: str = "",
    ) -> DiagnosisRecord:
        """
        Log diagnosis feedback for learning.
        """
        score = self.compute_similarity(predicted, actual)

        record = DiagnosisRecord(
            job_id=job_id,
            vehicle_id=vehicle_id,
            predicted_failure=predicted.get("failure_type", "unknown"),
            predicted_severity=predicted.get("severity", "medium"),
            actual_failure=actual.get("failure_type", "unknown"),
            actual_severity=actual.get("severity", "medium"),
            similarity_score=score,
            diagnosis_notes=notes,
        )

        self._records.append(record)

        # Track patterns for RCA
        if score < 0.8:  # Mismatch
            pattern = f"{record.predicted_failure} -> {record.actual_failure}"
            self._patterns[pattern] = self._patterns.get(pattern, 0) + 1

        return record

    def get_recent_feedback(self, limit: int = 20) -> List[Dict]:
        """Get recent diagnosis records."""
        return [
            {
                "job_id": r.job_id,
                "vehicle_id": r.vehicle_id,
                "predicted_failure": r.predicted_failure,
                "actual_failure": r.actual_failure,
                "similarity_score": r.similarity_score,
                "timestamp": r.timestamp.isoformat(),
            }
            for r in self._records[-limit:]
        ]

    def get_rca_insights(self) -> List[Dict[str, Any]]:
        """
        Generate Root Cause Analysis insights from diagnosis patterns.
        """
        insights = []

        # Analyze mismatches
        total_records = len(self._records)
        if total_records == 0:
            return insights

        mismatch_count = sum(1 for r in self._records if r.similarity_score < 0.8)
        accuracy_rate = 1 - (mismatch_count / total_records)

        insights.append(
            {
                "type": "PREDICTION_ACCURACY",
                "title": "Overall Prediction Accuracy",
                "value": f"{accuracy_rate * 100:.1f}%",
                "trend": "stable" if accuracy_rate > 0.7 else "needs_attention",
                "description": f"Based on {total_records} diagnosis comparisons",
            }
        )

        # Pattern-based insights
        for pattern, count in sorted(self._patterns.items(), key=lambda x: -x[1])[:5]:
            predicted, actual = pattern.split(" -> ")
            insights.append(
                {
                    "type": "PATTERN",
                    "title": f"Recurring Mismatch: {predicted.replace('_', ' ').title()}",
                    "value": f"{count} occurrences",
                    "trend": "investigation_needed",
                    "description": f"Predicted {predicted} but found {actual}. Review prediction model for this failure mode.",
                }
            )

        # Severity-based insights
        under_predicted = sum(
            1
            for r in self._records
            if self.SEVERITY_WEIGHTS.get(r.actual_severity, 2)
            > self.SEVERITY_WEIGHTS.get(r.predicted_severity, 2)
        )
        if under_predicted > 2:
            insights.append(
                {
                    "type": "SEVERITY_GAP",
                    "title": "Under-Prediction of Severity",
                    "value": f"{under_predicted} cases",
                    "trend": "warning",
                    "description": "Actual severity was higher than predicted. Consider adjusting prediction thresholds.",
                }
            )

        return insights

    def get_capa_recommendations(self) -> List[Dict[str, Any]]:
        """
        Generate Corrective & Preventive Action recommendations.
        """
        recommendations = []

        # Analyze brake-related issues
        brake_issues = sum(
            1 for r in self._records if "brake" in r.actual_failure.lower()
        )
        if brake_issues > 3:
            recommendations.append(
                {
                    "id": "CAPA-001",
                    "category": "DESIGN",
                    "priority": "HIGH",
                    "title": "Brake Component Durability",
                    "finding": f"{brake_issues} brake-related service events recorded",
                    "recommendation": "Review brake pad material specification for urban driving conditions. Consider enhanced cooling for brake system.",
                    "target_team": "Manufacturing Engineering",
                }
            )

        # Analyze battery issues
        battery_issues = sum(
            1 for r in self._records if "battery" in r.actual_failure.lower()
        )
        if battery_issues > 2:
            recommendations.append(
                {
                    "id": "CAPA-002",
                    "category": "QUALITY",
                    "priority": "MEDIUM",
                    "title": "Battery Cell Quality Control",
                    "finding": f"{battery_issues} battery degradation events recorded",
                    "recommendation": "Increase incoming inspection sampling for battery cells. Add thermal cycling test to QC process.",
                    "target_team": "Quality Assurance",
                }
            )

        # General recommendation based on patterns
        for pattern, count in self._patterns.items():
            if count >= 3:
                predicted, actual = pattern.split(" -> ")
                recommendations.append(
                    {
                        "id": f"CAPA-P{len(recommendations) + 1:03d}",
                        "category": "PREDICTION",
                        "priority": "MEDIUM",
                        "title": f"Improve {predicted.replace('_', ' ').title()} Detection",
                        "finding": f"Prediction model confused {predicted} with {actual} in {count} cases",
                        "recommendation": f"Retrain prediction model with additional telemetry features to distinguish between {predicted} and {actual}.",
                        "target_team": "Data Science",
                    }
                )

        return recommendations

    def reset(self):
        """Reset all data (for demo)."""
        self._records.clear()
        self._patterns.clear()


# Singleton
diagnosis_feedback = DiagnosisFeedback()
