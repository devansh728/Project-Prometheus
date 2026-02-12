"""
Worker Agents for SentinEV Core
Each agent has a specialized responsibility and interfaces with the ML pipeline.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import json

from sentinev_core.services.ml_pipeline import ml_pipeline, DiagnosisResult, Severity


@dataclass
class AgentContext:
    """Shared context passed between agents."""

    vehicle_id: str
    customer_id: Optional[str] = None
    sensors: Dict[str, float] = field(default_factory=dict)
    baseline: Dict[str, Dict] = field(default_factory=dict)
    degradation_config: Dict[str, Any] = field(default_factory=dict)
    mileage: int = 0
    diagnosis: Optional[DiagnosisResult] = None
    actions_taken: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class DataAnalysisAgent:
    """
    Responsible for processing raw telemetry and running anomaly detection.
    Access: Telemetry data, vehicle profiles
    """

    name = "data_analysis"

    def process(self, context: AgentContext) -> AgentContext:
        """Run ML pipeline on sensor data."""
        if not context.sensors:
            context.actions_taken.append(f"{self.name}: No sensor data available")
            return context

        # Run full diagnosis
        diagnosis = ml_pipeline.diagnose(
            sensors=context.sensors,
            baseline=context.baseline,
            degradation_config=context.degradation_config,
            mileage=context.mileage,
        )

        context.diagnosis = diagnosis
        context.actions_taken.append(
            f"{self.name}: Analyzed telemetry, anomaly_score={diagnosis.anomaly.anomaly_score:.3f}, "
            f"failure_prob={diagnosis.prediction.failure_probability:.3f}"
        )

        return context


class DiagnosisAgent:
    """
    Responsible for generating detailed diagnosis and explanations.
    Uses RAG to enrich explanations with repair procedures.
    """

    name = "diagnosis"

    def process(self, context: AgentContext) -> AgentContext:
        """Generate detailed diagnosis with RAG-enhanced explanations."""
        if not context.diagnosis:
            context.actions_taken.append(f"{self.name}: No diagnosis data available")
            return context

        diagnosis = context.diagnosis

        # Simulate RAG retrieval (would query ChromaDB in production)
        rag_context = self._get_rag_context(diagnosis)

        # Enrich context with RAG-retrieved information
        context.metadata["rag_retrieval"] = rag_context
        context.metadata["detailed_diagnosis"] = self._generate_detailed_report(
            diagnosis, rag_context
        )

        context.actions_taken.append(
            f"{self.name}: Generated diagnosis report for {diagnosis.primary_concern or 'general'}"
        )

        return context

    def _get_rag_context(self, diagnosis: DiagnosisResult) -> Dict[str, Any]:
        """Simulate RAG retrieval based on diagnosis."""
        # In production, this would query ChromaDB
        component_guides = {
            "brake_degradation": {
                "source": "repair_guides/brake_system.md",
                "excerpt": "Check brake fluid condition and level. Inspect brake pad thickness (minimum 3mm).",
                "dtc_reference": ["C0035", "C0040"],
                "estimated_labor_hours": 2.0,
            },
            "battery_degradation": {
                "source": "repair_guides/battery_system.md",
                "excerpt": "Run full charge capacity test. Compare current Wh capacity to original.",
                "dtc_reference": ["P0A80", "P0AA6"],
                "estimated_labor_hours": 3.0,
            },
            "cooling_degradation": {
                "source": "repair_guides/cooling_system.md",
                "excerpt": "Pressure test cooling system (15 psi, 10 min). Check expansion tank for cracks.",
                "dtc_reference": ["P0128"],
                "estimated_labor_hours": 1.5,
            },
        }

        failure_type = diagnosis.prediction.failure_type
        return component_guides.get(
            failure_type,
            {
                "source": "general_maintenance.md",
                "excerpt": "Perform standard diagnostic inspection.",
                "dtc_reference": [],
                "estimated_labor_hours": 1.0,
            },
        )

    def _generate_detailed_report(self, diagnosis: DiagnosisResult, rag: Dict) -> str:
        """Generate a detailed diagnosis report."""
        report_parts = [
            f"## Vehicle Diagnosis Report",
            f"**Severity:** {diagnosis.severity.value}",
            f"**Primary Concern:** {diagnosis.primary_concern or 'None'}",
            f"",
            f"### Anomaly Analysis",
            f"- Anomaly Score: {diagnosis.anomaly.anomaly_score:.2%}",
            f"- Contributing Sensors: {', '.join(diagnosis.anomaly.contributing_sensors) or 'None'}",
            f"",
            f"### Failure Prediction",
            f"- Failure Probability: {diagnosis.prediction.failure_probability:.1%}",
            f"- Estimated RUL: {diagnosis.prediction.estimated_rul_days or 'N/A'} days",
            f"- Confidence: {diagnosis.prediction.confidence:.1%}",
            f"",
            f"### Repair Reference",
            f"- Source: {rag.get('source', 'N/A')}",
            f"- Procedure: {rag.get('excerpt', 'N/A')}",
            f"- Estimated Labor: {rag.get('estimated_labor_hours', 'N/A')} hours",
            f"",
            f"### Recommended Action",
            diagnosis.recommended_action,
        ]
        return "\n".join(report_parts)


class EngagementAgent:
    """
    Responsible for customer communication decisions.
    Decides whether to notify, call, or wait.
    """

    name = "engagement"

    # Action thresholds
    ACTION_MATRIX = {
        Severity.EMERGENCY: {"action": "voice_call", "priority": 1, "delay_minutes": 0},
        Severity.CRITICAL: {"action": "voice_call", "priority": 2, "delay_minutes": 5},
        Severity.WARNING: {
            "action": "push_notification",
            "priority": 3,
            "delay_minutes": 30,
        },
        Severity.INFO: {"action": "none", "priority": 4, "delay_minutes": 0},
    }

    def process(self, context: AgentContext) -> AgentContext:
        """Determine communication action based on severity."""
        if not context.diagnosis:
            context.actions_taken.append(f"{self.name}: No diagnosis to act on")
            return context

        severity = context.diagnosis.severity
        action_config = self.ACTION_MATRIX.get(severity, {"action": "none"})

        context.metadata["engagement_action"] = action_config["action"]
        context.metadata["engagement_priority"] = action_config["priority"]

        # Generate voice script if needed
        if action_config["action"] == "voice_call":
            context.metadata["voice_script"] = self._generate_voice_script(context)

        context.actions_taken.append(
            f"{self.name}: Decided action='{action_config['action']}' for severity={severity.value}"
        )

        return context

    def _generate_voice_script(self, context: AgentContext) -> Dict[str, str]:
        """Generate voice call script based on diagnosis."""
        diagnosis = context.diagnosis
        severity = diagnosis.severity

        # Tone mapping
        tone_map = {
            Severity.EMERGENCY: {"tone": "urgent", "stability": 0.95, "speed": 0.85},
            Severity.CRITICAL: {"tone": "serious", "stability": 0.85, "speed": 0.9},
            Severity.WARNING: {"tone": "concerned", "stability": 0.75, "speed": 0.95},
            Severity.INFO: {"tone": "friendly", "stability": 0.7, "speed": 1.0},
        }

        tone = tone_map.get(severity, tone_map[Severity.INFO])

        return {
            "opening": f"Hello, this is SentinEV Service calling about your vehicle.",
            "concern": diagnosis.explanation,
            "recommendation": diagnosis.recommended_action,
            "cta": "Would you like me to schedule a service appointment for you?",
            "closing": "Thank you for your time. Drive safely!",
            "tone_config": tone,
        }


class SchedulingAgent:
    """
    Responsible for interfacing with ServiceOpsAI for appointment scheduling.
    """

    name = "scheduling"

    def process(self, context: AgentContext) -> AgentContext:
        """Prepare scheduling request if needed."""
        if context.metadata.get("engagement_action") not in [
            "voice_call",
            "push_notification",
        ]:
            context.actions_taken.append(f"{self.name}: No scheduling action needed")
            return context

        if not context.diagnosis:
            return context

        # Prepare service request payload
        service_request = {
            "vehicle_id": context.vehicle_id,
            "customer_id": context.customer_id,
            "failure_type": context.diagnosis.prediction.failure_type,
            "severity": context.diagnosis.severity.value,
            "failure_probability": context.diagnosis.prediction.failure_probability,
            "estimated_rul_days": context.diagnosis.prediction.estimated_rul_days,
            "customer_location": context.metadata.get("customer_location"),
            "requested_at": datetime.utcnow().isoformat(),
        }

        context.metadata["service_request"] = service_request
        context.actions_taken.append(
            f"{self.name}: Prepared service request for {service_request['failure_type']}"
        )

        return context


# Agent registry
WORKER_AGENTS = {
    "data_analysis": DataAnalysisAgent(),
    "diagnosis": DiagnosisAgent(),
    "engagement": EngagementAgent(),
    "scheduling": SchedulingAgent(),
}
