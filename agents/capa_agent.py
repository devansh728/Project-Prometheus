"""
SentinEV - CAPA Agent
Corrective and Preventive Action Agent for manufacturing feedback loop.
Handles RCA (Root Cause Analysis) and CAPA report generation.
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

# Optional: Gemini for RCA analysis
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False


@dataclass
class CAPAReport:
    """Represents a CAPA report."""

    capa_id: str
    created_date: str
    status: str
    failure_mode: str
    root_cause: Dict[str, Any]
    affected_components: List[str]
    affected_models: List[str]
    detection_method: str
    occurrence: Dict[str, Any]
    corrective_actions: List[Dict]
    preventive_actions: List[Dict]
    verification: Dict[str, Any]
    lessons_learned: str
    design_feedback: str
    region: Optional[str] = None


class CAPAAgent:
    """
    CAPA Agent for RCA/CAPA analysis and manufacturing feedback.

    Features:
    - Load and query CAPA records
    - Pattern detection across vehicles/regions
    - Generate new CAPA reports from diagnosis data
    - LLM-powered root cause analysis
    - Manufacturing feedback recommendations
    """

    def __init__(self, capa_data_path: str = "data/datasets/capa_records.json"):
        """Initialize CAPA Agent with data path."""
        self.capa_data_path = Path(capa_data_path)
        self.capa_records: List[Dict] = []
        self.llm = None
        self.rca_chain = None

        self._load_capa_records()
        self._initialize_llm()

    def _load_capa_records(self):
        """Load CAPA records from JSON file."""
        if self.capa_data_path.exists():
            try:
                with open(self.capa_data_path, "r") as f:
                    self.capa_records = json.load(f)
                print(f"✓ Loaded {len(self.capa_records)} CAPA records")
            except Exception as e:
                print(f"⚠️ Failed to load CAPA records: {e}")
                self.capa_records = []

    def _initialize_llm(self):
        """Initialize LLM for RCA analysis."""
        if not LLM_AVAILABLE:
            return

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return

        try:
            self.llm = ChatGoogleGenerativeAI(
                model=os.getenv("LLM_MODEL", "gemini-2.0-flash"),
                google_api_key=api_key,
                temperature=0.3,
            )

            # RCA analysis prompt
            rca_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """You are an automotive quality engineer specializing in EV systems.
Analyze the failure data and generate a root cause analysis.

Be specific and technical. Focus on:
1. Primary root cause
2. Contributing factors
3. Root cause category (Manufacturing Defect, Software/Algorithm, Material Specification, Supplier Change, Design Margin, Production Process)
4. Recommended corrective actions
5. Recommended preventive actions
6. Lessons learned
7. Design feedback for manufacturing

Format your response as JSON with these exact keys:
{{
    "primary_cause": "...",
    "contributing_factors": ["..."],
    "root_cause_category": "...",
    "corrective_actions": [{{"action": "...", "responsible": "...", "priority": "high/medium/low"}}],
    "preventive_actions": [{{"action": "...", "responsible": "..."}}],
    "lessons_learned": "...",
    "design_feedback": "..."
}}""",
                    ),
                    (
                        "human",
                        """Failure Data:
Component: {component}
Failure Mode: {failure_mode}
Occurrence Rate: {occurrence_rate} PPM
Detection Method: {detection_method}
Affected Region: {region}
Vehicle Data: {vehicle_data}

Generate RCA report:""",
                    ),
                ]
            )

            self.rca_chain = rca_prompt | self.llm | StrOutputParser()
        except Exception as e:
            print(f"⚠️ CAPA Agent LLM init failed: {e}")

    def get_all_reports(
        self,
        status: Optional[str] = None,
        component: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict]:
        """
        Get CAPA reports with optional filtering.

        Args:
            status: Filter by status (open, closed)
            component: Filter by affected component
            limit: Maximum results

        Returns:
            List of CAPA reports
        """
        results = self.capa_records.copy()

        if status:
            results = [r for r in results if r.get("status") == status]

        if component:
            results = [
                r
                for r in results
                if component.lower()
                in [c.lower() for c in r.get("affected_components", [])]
            ]

        return results[:limit]

    def get_report_by_id(self, capa_id: str) -> Optional[Dict]:
        """Get a specific CAPA report by ID."""
        for record in self.capa_records:
            if record.get("capa_id") == capa_id:
                return record
        return None

    def find_pattern_analysis(
        self, component: str, region: Optional[str] = None, min_occurrences: int = 2
    ) -> Dict[str, Any]:
        """
        Find patterns across vehicles for a component failure.

        This is the "50 vehicles in mountainous regions" analysis.

        Args:
            component: Component to analyze
            region: Optional region filter
            min_occurrences: Minimum occurrence count

        Returns:
            Pattern analysis results
        """
        matching_records = [
            r
            for r in self.capa_records
            if component.lower()
            in [c.lower() for c in r.get("affected_components", [])]
        ]

        # Aggregate occurrence data
        total_production = sum(
            r.get("occurrence", {}).get("production_volume", 0)
            for r in matching_records
        )
        total_failures = sum(
            r.get("occurrence", {}).get("failures", 0) for r in matching_records
        )

        # Find common root causes
        root_cause_categories = {}
        for r in matching_records:
            category = r.get("root_cause", {}).get("root_cause_category", "Unknown")
            root_cause_categories[category] = root_cause_categories.get(category, 0) + 1

        # Find common failure modes
        failure_modes = {}
        for r in matching_records:
            mode = r.get("failure_mode", "Unknown")
            failure_modes[mode] = failure_modes.get(mode, 0) + 1

        # Generate pattern summary
        if component.lower() == "brakes":
            pattern_summary = (
                f"Found {len(matching_records)} CAPA records related to brake issues. "
                f"Analysis indicates {total_failures} failures across {total_production:,} vehicles "
                f"in mountainous/high-temperature regions."
            )
            recommendation = (
                "Root Cause: Brake Compound B-22 fails > 300°C. "
                "Recommendation: Issue Recall for Batch #99."
            )
        else:
            pattern_summary = (
                f"Found {len(matching_records)} CAPA records for {component}."
            )
            recommendation = "Further analysis recommended."

        return {
            "component": component,
            "matching_records": len(matching_records),
            "total_production_volume": total_production,
            "total_failures": total_failures,
            "failure_rate_ppm": int(
                total_failures / max(total_production, 1) * 1_000_000
            ),
            "root_cause_distribution": root_cause_categories,
            "failure_mode_distribution": failure_modes,
            "pattern_summary": pattern_summary,
            "recommendation": recommendation,
            "affected_capa_ids": [r.get("capa_id") for r in matching_records],
            "region_analysis": {
                "mountainous": 50,  # Simulated for demo
                "urban": 15,
                "highway": 10,
            },
        }

    def generate_capa_from_diagnosis(
        self,
        vehicle_id: str,
        component: str,
        diagnosis_summary: str,
        failure_mode: str,
        region: str = "Mountainous",
        vehicle_data: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Generate a new CAPA report from diagnosis data.

        This is called after a service is completed to close the feedback loop.

        Args:
            vehicle_id: Vehicle identifier
            component: Failed component
            diagnosis_summary: Summary from diagnosis agent
            failure_mode: Description of failure
            region: Operating region
            vehicle_data: Additional vehicle telemetry

        Returns:
            Generated CAPA report
        """
        capa_id = (
            f"CAPA-{datetime.now().strftime('%Y')}-{len(self.capa_records) + 1:03d}"
        )

        # Try to generate RCA using LLM
        rca_result = None
        if self.rca_chain:
            try:
                rca_response = self.rca_chain.invoke(
                    {
                        "component": component,
                        "failure_mode": failure_mode,
                        "occurrence_rate": "1500",  # Estimated
                        "detection_method": "Predictive Maintenance System",
                        "region": region,
                        "vehicle_data": json.dumps(vehicle_data or {}),
                    }
                )

                # Parse JSON response
                rca_result = json.loads(rca_response)
            except Exception as e:
                print(f"LLM RCA failed: {e}")

        # Fallback RCA based on component
        if not rca_result:
            rca_result = self._fallback_rca(component, failure_mode, region)

        # Create CAPA report
        new_capa = {
            "capa_id": capa_id,
            "created_date": datetime.now().strftime("%Y-%m-%d"),
            "status": "open",
            "failure_mode": failure_mode,
            "root_cause": {
                "primary": rca_result.get("primary_cause", "Under investigation"),
                "contributing_factors": rca_result.get("contributing_factors", []),
                "root_cause_category": rca_result.get(
                    "root_cause_category", "Under Investigation"
                ),
            },
            "affected_components": [component],
            "affected_models": [f"SentinEV-X1 {vehicle_id}"],
            "detection_method": "SentinEV Predictive Maintenance System",
            "occurrence": {
                "vehicles_affected": 1,
                "estimated_fleet_risk": 50,  # Simulated
                "rate_ppm": 1500,
            },
            "corrective_actions": rca_result.get(
                "corrective_actions",
                [
                    {
                        "action": f"Replace {component} components in affected vehicles",
                        "responsible": "Service Operations",
                        "due_date": (datetime.now()).strftime("%Y-%m-%d"),
                        "status": "planned",
                    }
                ],
            ),
            "preventive_actions": rca_result.get(
                "preventive_actions",
                [
                    {
                        "action": f"Enhanced monitoring for {component} in similar conditions",
                        "responsible": "Data Science",
                        "status": "planned",
                    }
                ],
            ),
            "verification": {
                "method": "Field monitoring and customer follow-up",
                "result": "Pending",
                "closure_date": None,
            },
            "lessons_learned": rca_result.get(
                "lessons_learned",
                f"Early detection of {component} issues through predictive maintenance prevents safety incidents",
            ),
            "design_feedback": rca_result.get(
                "design_feedback",
                f"Consider enhanced {component} specifications for extreme operating conditions",
            ),
            "source_vehicle": vehicle_id,
            "source_diagnosis": diagnosis_summary,
            "region": region,
        }

        # Add to records (in memory)
        self.capa_records.append(new_capa)

        return {
            "success": True,
            "capa_id": capa_id,
            "report": new_capa,
            "message": f"CAPA report {capa_id} generated for manufacturing review",
            "pattern_analysis": self.find_pattern_analysis(component),
        }

    def _fallback_rca(self, component: str, failure_mode: str, region: str) -> Dict:
        """Generate fallback RCA when LLM is not available."""

        component_rcas = {
            "brakes": {
                "primary_cause": "Brake compound degradation at high temperatures during mountain descent",
                "contributing_factors": [
                    "Extended brake application on steep grades",
                    "Insufficient thermal capacity in brake design",
                    "High ambient temperature conditions",
                ],
                "root_cause_category": "Design Margin",
                "corrective_actions": [
                    {
                        "action": "Replace with high-temperature ceramic brake pads",
                        "responsible": "Service Operations",
                        "priority": "high",
                    },
                    {
                        "action": "Update brake thermal monitoring thresholds",
                        "responsible": "Vehicle Software",
                        "priority": "medium",
                    },
                ],
                "preventive_actions": [
                    {
                        "action": "Add mountain driving mode with enhanced regen braking",
                        "responsible": "Powertrain Engineering",
                    },
                    {
                        "action": "Implement predictive brake temperature warnings",
                        "responsible": "Data Science",
                    },
                ],
                "lessons_learned": "Brake thermal management must account for worst-case mountain descent scenarios with fully loaded vehicles",
                "design_feedback": "Increase brake thermal margin by 30% for vehicles sold in mountainous regions. Consider ceramic compound as standard.",
            },
            "battery": {
                "primary_cause": "Thermal management system undersized for rapid charging in hot climates",
                "contributing_factors": [
                    "High ambient temperature",
                    "Repeated fast charging sessions",
                    "Coolant flow rate limiting",
                ],
                "root_cause_category": "Design Margin",
                "corrective_actions": [
                    {
                        "action": "Update charging profile for hot weather",
                        "responsible": "BMS Engineering",
                        "priority": "high",
                    }
                ],
                "preventive_actions": [
                    {
                        "action": "Enhanced thermal modeling for hot climate validation",
                        "responsible": "Validation",
                    }
                ],
                "lessons_learned": "Battery thermal systems must handle compound thermal loads from charging and ambient conditions",
                "design_feedback": "Increase cooling capacity by 25% for hot climate variants",
            },
            "motor": {
                "primary_cause": "Motor resolver drift from thermal expansion",
                "contributing_factors": [
                    "Aggressive driving patterns",
                    "High motor temperatures from repeated acceleration",
                ],
                "root_cause_category": "Material Specification",
                "corrective_actions": [
                    {
                        "action": "Recalibrate resolver during service",
                        "responsible": "Service Operations",
                        "priority": "high",
                    }
                ],
                "preventive_actions": [
                    {
                        "action": "Use temperature-compensated resolver readings",
                        "responsible": "Motor Control",
                    }
                ],
                "lessons_learned": "Resolver mounting must maintain accuracy across full thermal range",
                "design_feedback": "Consider mechanical resolver retention as primary mounting method",
            },
        }

        return component_rcas.get(
            component.lower(),
            {
                "primary_cause": f"Under investigation for {component} failure",
                "contributing_factors": [f"{region} operating conditions"],
                "root_cause_category": "Under Investigation",
                "corrective_actions": [
                    {
                        "action": f"Inspect {component}",
                        "responsible": "Service Operations",
                        "priority": "medium",
                    }
                ],
                "preventive_actions": [
                    {
                        "action": f"Monitor {component} telemetry",
                        "responsible": "Data Science",
                    }
                ],
                "lessons_learned": f"Early detection of {component} issues through predictive maintenance",
                "design_feedback": f"Review {component} specifications for {region} conditions",
            },
        )

    def get_manufacturing_summary(self) -> Dict[str, Any]:
        """Get summary for manufacturing dashboard."""
        open_capas = [r for r in self.capa_records if r.get("status") == "open"]
        closed_capas = [r for r in self.capa_records if r.get("status") == "closed"]

        # Component distribution
        component_counts = {}
        for r in self.capa_records:
            for comp in r.get("affected_components", []):
                component_counts[comp] = component_counts.get(comp, 0) + 1

        # Root cause distribution
        rca_categories = {}
        for r in self.capa_records:
            category = r.get("root_cause", {}).get("root_cause_category", "Unknown")
            rca_categories[category] = rca_categories.get(category, 0) + 1

        return {
            "total_capas": len(self.capa_records),
            "open_capas": len(open_capas),
            "closed_capas": len(closed_capas),
            "component_distribution": component_counts,
            "root_cause_distribution": rca_categories,
            "recent_open": open_capas[:5],
            "high_priority_actions": self._get_high_priority_actions(),
        }

    def _get_high_priority_actions(self) -> List[Dict]:
        """Get high priority corrective actions."""
        actions = []
        for r in self.capa_records:
            if r.get("status") == "open":
                for action in r.get("corrective_actions", []):
                    if action.get("status") in ["planned", "in_progress"]:
                        actions.append(
                            {
                                "capa_id": r.get("capa_id"),
                                "action": action.get("action"),
                                "responsible": action.get("responsible"),
                                "due_date": action.get("due_date"),
                                "component": r.get("affected_components", ["unknown"])[
                                    0
                                ],
                            }
                        )
        return actions[:10]


# Singleton instance
_capa_agent: Optional[CAPAAgent] = None


def get_capa_agent() -> CAPAAgent:
    """Get or create CAPAAgent singleton."""
    global _capa_agent
    if _capa_agent is None:
        _capa_agent = CAPAAgent()
    return _capa_agent
