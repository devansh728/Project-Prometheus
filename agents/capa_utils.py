"""
SentinEV - CAPA Utilities
Support structures for RCA/CAPA manufacturing feedback.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
import random


class RiskLevel(Enum):
    """Supplier risk level."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class QualityTrend(Enum):
    """Quality trend direction."""

    IMPROVING = "improving"
    STABLE = "stable"
    DECLINING = "declining"


@dataclass
class SupplierRiskScore:
    """Supplier risk assessment."""

    supplier_id: str
    supplier_name: str
    defect_rate_ppm: float  # Parts per million
    response_time_hours: float  # Average response to issues
    quality_trend: QualityTrend
    risk_score: float  # 0-10 scale
    risk_level: RiskLevel
    components_supplied: List[str] = field(default_factory=list)
    total_parts_supplied: int = 0
    defects_reported: int = 0
    last_audit_date: Optional[str] = None
    certifications: List[str] = field(default_factory=list)


@dataclass
class RecurrenceAnalysis:
    """Failure recurrence analysis."""

    component: str
    vehicle_id: Optional[str]
    occurrence_count: int
    first_occurrence: str
    last_occurrence: str
    is_chronic: bool  # >3 occurrences
    days_between_avg: float
    related_capa_ids: List[str] = field(default_factory=list)
    pattern_detected: str = ""
    recommendation: str = ""


@dataclass
class ModelYearDrift:
    """Model year quality drift analysis."""

    model_year: int
    component: str
    failure_rate_ppm: float
    sample_size: int
    comparison_to_avg: float  # +/- percentage
    trend: str  # "elevated", "normal", "improved"
    alert: bool
    contributing_factors: List[str] = field(default_factory=list)


# Simulated supplier database
SUPPLIER_DATABASE = {
    "SUP-001": {
        "name": "BrakeTech Industries",
        "components": ["brakes", "brake_pads", "rotors"],
        "defect_rate_ppm": 1200,
        "response_time_hours": 24,
        "parts_supplied": 150000,
        "defects": 180,
        "trend": "declining",
        "certifications": ["ISO 9001", "IATF 16949"],
        "last_audit": "2024-06-15",
    },
    "SUP-002": {
        "name": "PowerCell Energy",
        "components": ["battery", "battery_cells", "BMS"],
        "defect_rate_ppm": 450,
        "response_time_hours": 12,
        "parts_supplied": 80000,
        "defects": 36,
        "trend": "improving",
        "certifications": ["ISO 9001", "ISO 14001", "IATF 16949"],
        "last_audit": "2024-08-20",
    },
    "SUP-003": {
        "name": "EMotion Drives",
        "components": ["motor", "inverter", "reducer"],
        "defect_rate_ppm": 800,
        "response_time_hours": 48,
        "parts_supplied": 60000,
        "defects": 48,
        "trend": "stable",
        "certifications": ["ISO 9001"],
        "last_audit": "2024-03-10",
    },
    "SUP-004": {
        "name": "CoolFlow Systems",
        "components": ["coolant", "radiator", "thermal_paste"],
        "defect_rate_ppm": 2000,
        "response_time_hours": 72,
        "parts_supplied": 100000,
        "defects": 200,
        "trend": "declining",
        "certifications": ["ISO 9001"],
        "last_audit": "2023-11-01",
    },
    "SUP-005": {
        "name": "TirePro Solutions",
        "components": ["tires", "tire_pressure_sensor", "valve"],
        "defect_rate_ppm": 300,
        "response_time_hours": 8,
        "parts_supplied": 200000,
        "defects": 60,
        "trend": "improving",
        "certifications": ["ISO 9001", "IATF 16949", "ISO 14001"],
        "last_audit": "2024-09-05",
    },
}

# Simulated model year data
MODEL_YEAR_DATA = {
    2022: {
        "production_volume": 25000,
        "components": {
            "brakes": {"failures": 125, "rate_ppm": 5000},
            "battery": {"failures": 50, "rate_ppm": 2000},
            "motor": {"failures": 25, "rate_ppm": 1000},
        },
    },
    2023: {
        "production_volume": 50000,
        "components": {
            "brakes": {"failures": 150, "rate_ppm": 3000},
            "battery": {"failures": 75, "rate_ppm": 1500},
            "motor": {"failures": 50, "rate_ppm": 1000},
        },
    },
    2024: {
        "production_volume": 75000,
        "components": {
            "brakes": {"failures": 300, "rate_ppm": 4000},
            "battery": {"failures": 60, "rate_ppm": 800},
            "motor": {"failures": 45, "rate_ppm": 600},
        },
    },
}


def calculate_supplier_risk(supplier_id: str) -> SupplierRiskScore:
    """
    Calculate risk score for a supplier.

    Scoring:
    - Defect rate: 40% weight (higher = more risk)
    - Response time: 30% weight (slower = more risk)
    - Quality trend: 30% weight (declining = more risk)
    """
    supplier = SUPPLIER_DATABASE.get(supplier_id)
    if not supplier:
        return SupplierRiskScore(
            supplier_id=supplier_id,
            supplier_name="Unknown",
            defect_rate_ppm=0,
            response_time_hours=0,
            quality_trend=QualityTrend.STABLE,
            risk_score=5.0,
            risk_level=RiskLevel.MEDIUM,
        )

    # Defect rate score (0-10, lower is better)
    defect_score = min(supplier["defect_rate_ppm"] / 500, 10)

    # Response time score (0-10, faster is better)
    response_score = min(supplier["response_time_hours"] / 10, 10)

    # Trend score
    trend_map = {"improving": 2, "stable": 5, "declining": 8}
    trend_score = trend_map.get(supplier["trend"], 5)

    # Weighted calculation
    risk_score = (defect_score * 0.4) + (response_score * 0.3) + (trend_score * 0.3)
    risk_score = round(min(risk_score, 10), 2)

    # Determine risk level
    if risk_score >= 7:
        risk_level = RiskLevel.CRITICAL
    elif risk_score >= 5:
        risk_level = RiskLevel.HIGH
    elif risk_score >= 3:
        risk_level = RiskLevel.MEDIUM
    else:
        risk_level = RiskLevel.LOW

    # Quality trend enum
    trend_enum = {
        "improving": QualityTrend.IMPROVING,
        "stable": QualityTrend.STABLE,
        "declining": QualityTrend.DECLINING,
    }.get(supplier["trend"], QualityTrend.STABLE)

    return SupplierRiskScore(
        supplier_id=supplier_id,
        supplier_name=supplier["name"],
        defect_rate_ppm=supplier["defect_rate_ppm"],
        response_time_hours=supplier["response_time_hours"],
        quality_trend=trend_enum,
        risk_score=risk_score,
        risk_level=risk_level,
        components_supplied=supplier["components"],
        total_parts_supplied=supplier["parts_supplied"],
        defects_reported=supplier["defects"],
        last_audit_date=supplier["last_audit"],
        certifications=supplier["certifications"],
    )


def get_supplier_for_component(component: str) -> Optional[str]:
    """Get supplier ID for a component."""
    component_lower = component.lower()
    for supplier_id, data in SUPPLIER_DATABASE.items():
        if component_lower in [c.lower() for c in data["components"]]:
            return supplier_id
    return None


def analyze_model_year_drift(
    component: str,
    target_year: int = None,
) -> List[ModelYearDrift]:
    """
    Analyze failure rate trends across model years.

    Args:
        component: Component to analyze
        target_year: Optional specific year to focus on

    Returns:
        List of ModelYearDrift analyses
    """
    component_lower = component.lower()
    results = []

    # Calculate average rate across all years
    total_failures = 0
    total_volume = 0

    for year, data in MODEL_YEAR_DATA.items():
        comp_data = data["components"].get(component_lower, {})
        total_failures += comp_data.get("failures", 0)
        total_volume += data["production_volume"]

    avg_rate = (total_failures / max(total_volume, 1)) * 1_000_000  # PPM

    # Analyze each year
    for year, data in MODEL_YEAR_DATA.items():
        if target_year and year != target_year:
            continue

        comp_data = data["components"].get(component_lower, {})
        rate_ppm = comp_data.get("rate_ppm", 0)

        # Compare to average
        if avg_rate > 0:
            comparison = ((rate_ppm - avg_rate) / avg_rate) * 100
        else:
            comparison = 0

        # Determine trend
        if comparison > 20:
            trend = "elevated"
            alert = True
        elif comparison < -20:
            trend = "improved"
            alert = False
        else:
            trend = "normal"
            alert = False

        # Contributing factors
        factors = []
        if trend == "elevated":
            factors = [
                f"Supplier change in {year}",
                f"Design revision for {year} model",
                f"New manufacturing process introduced",
            ]

        results.append(
            ModelYearDrift(
                model_year=year,
                component=component,
                failure_rate_ppm=rate_ppm,
                sample_size=data["production_volume"],
                comparison_to_avg=round(comparison, 1),
                trend=trend,
                alert=alert,
                contributing_factors=factors[:2] if factors else [],
            )
        )

    return sorted(results, key=lambda x: x.model_year)


def detect_recurrence(
    capa_records: List[Dict],
    component: str,
    vehicle_id: str = None,
) -> RecurrenceAnalysis:
    """
    Detect failure recurrence patterns.

    Args:
        capa_records: List of CAPA records
        component: Component to analyze
        vehicle_id: Optional vehicle to filter

    Returns:
        RecurrenceAnalysis
    """
    component_lower = component.lower()

    # Filter matching records
    matching = []
    for record in capa_records:
        affected = [c.lower() for c in record.get("affected_components", [])]
        if component_lower in affected:
            if vehicle_id:
                if record.get("source_vehicle") == vehicle_id:
                    matching.append(record)
            else:
                matching.append(record)

    if not matching:
        return RecurrenceAnalysis(
            component=component,
            vehicle_id=vehicle_id,
            occurrence_count=0,
            first_occurrence="N/A",
            last_occurrence="N/A",
            is_chronic=False,
            days_between_avg=0,
            recommendation="No issues detected",
        )

    # Sort by date
    matching.sort(key=lambda x: x.get("created_date", ""))

    occurrence_count = len(matching)
    first_occurrence = matching[0].get("created_date", "Unknown")
    last_occurrence = matching[-1].get("created_date", "Unknown")
    is_chronic = occurrence_count > 3

    # Calculate average days between
    days_between_avg = 30 * occurrence_count  # Simulated

    # Pattern detection
    if is_chronic:
        pattern = f"Chronic failure pattern: {occurrence_count} occurrences of {component} issues"
        recommendation = f"Escalate to engineering review. Consider {component} design revision or supplier change."
    elif occurrence_count > 1:
        pattern = f"Repeat failure: {occurrence_count} occurrences"
        recommendation = (
            f"Monitor closely. Schedule preventive maintenance for similar vehicles."
        )
    else:
        pattern = "Single occurrence"
        recommendation = "Continue monitoring"

    return RecurrenceAnalysis(
        component=component,
        vehicle_id=vehicle_id,
        occurrence_count=occurrence_count,
        first_occurrence=first_occurrence,
        last_occurrence=last_occurrence,
        is_chronic=is_chronic,
        days_between_avg=days_between_avg,
        related_capa_ids=[r.get("capa_id") for r in matching],
        pattern_detected=pattern,
        recommendation=recommendation,
    )
