"""
SentinEV - Diagnosis Agent
Handles major issues requiring service center repair.
Uses RAG + vehicle manual for step-by-step diagnosis.
"""

import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

# Google Gemini
try:
    import google.generativeai as genai

    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class DiagnosisStatus(Enum):
    """Status of diagnosis process."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SERVICE_SCHEDULED = "service_scheduled"
    SERVICE_DECLINED = "service_declined"


@dataclass
class DiagnosisStep:
    """Single step in the diagnosis process."""

    step_number: int
    title: str
    description: str
    finding: str
    status: str  # ok, warning, critical


@dataclass
class DiagnosisResult:
    """Complete diagnosis result."""

    diagnosis_id: str
    vehicle_id: str
    component: str
    issue_type: str
    severity: str
    status: DiagnosisStatus
    steps: List[DiagnosisStep]
    root_cause: str
    repair_action: str
    estimated_cost_range: str
    urgency: str  # immediate, soon, scheduled
    service_required: bool
    rag_sources: List[str]
    gemini_summary: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class DiagnosisAgent:
    """
    Diagnosis Agent for handling major issues.

    Triggered when:
    - User REJECTS prediction and repeats behavior
    - Component damage detected
    - Time-to-failure < 7 days and requires service

    Actions:
    1. Query RAG + vehicle manual
    2. Generate step-by-step diagnosis
    3. Identify repair requirements
    4. Prepare service scheduling
    """

    # Diagnostic procedures by component
    DIAGNOSTIC_PROCEDURES = {
        "battery": [
            (
                "Cell Voltage Check",
                "Measure individual cell voltages",
                "cell_voltage_diff_v",
            ),
            (
                "Thermal Analysis",
                "Check battery temperature distribution",
                "battery_temp_c",
            ),
            ("SOC Calibration", "Verify state of charge accuracy", "soc_deviation_pct"),
            (
                "Isolation Test",
                "Measure HV isolation resistance",
                "isolation_resistance_mohm",
            ),
            (
                "Cooling System",
                "Inspect coolant flow and pump operation",
                "coolant_flow_lpm",
            ),
        ],
        "motor": [
            (
                "Winding Resistance",
                "Measure stator winding resistance",
                "motor_winding_resistance_ohm",
            ),
            ("Bearing Check", "Listen for bearing noise", "motor_bearing_noise_db"),
            ("Resolver Test", "Verify position sensor accuracy", "resolver_drift_deg"),
            ("Temperature Profile", "Check temperature under load", "motor_temp_c"),
            ("Efficiency Test", "Measure motor efficiency", "motor_efficiency_pct"),
        ],
        "inverter": [
            ("IGBT Check", "Test power transistor health", "inverter_fault_flag"),
            ("Thermal Paste", "Inspect thermal interface", "inverter_temp_c"),
            ("Capacitor Test", "Check DC link capacitors", "dc_link_ripple_pct"),
            ("Gate Driver", "Verify gate driver signals", "gate_driver_status"),
        ],
        "brakes": [
            (
                "Pad Thickness",
                "Measure brake pad remaining material",
                "brake_pad_thickness_mm",
            ),
            ("Rotor Condition", "Inspect rotor surface and runout", "rotor_runout_mm"),
            (
                "Caliper Function",
                "Check caliper slide pins and pistons",
                "caliper_drag_detected",
            ),
            (
                "Fluid Level",
                "Verify brake fluid level and condition",
                "brake_fluid_level_pct",
            ),
            ("Regen System", "Test regenerative braking function", "regen_efficiency"),
        ],
        "cooling": [
            ("Coolant Level", "Check coolant reservoir level", "coolant_level_pct"),
            ("Pump Operation", "Verify coolant pump flow rate", "coolant_flow_lpm"),
            (
                "Radiator Check",
                "Inspect radiator for blockages",
                "radiator_efficiency_pct",
            ),
            ("Thermostat", "Test thermostat operation", "thermostat_open_temp_c"),
            ("Fan Function", "Check cooling fan operation", "cooling_fan_rpm"),
        ],
    }

    # Cost estimates by repair type
    REPAIR_COSTS = {
        "battery_thermal": "$200 - $500 (cooling system service)",
        "battery_cell": "$1,000 - $5,000 (cell replacement)",
        "battery_bms": "$500 - $1,500 (BMS repair)",
        "motor_bearing": "$300 - $800 (bearing replacement)",
        "motor_winding": "$2,000 - $5,000 (motor replacement)",
        "inverter_igbt": "$1,500 - $4,000 (inverter replacement)",
        "brake_pads": "$150 - $400 (brake pad replacement)",
        "brake_rotor": "$300 - $700 (rotor replacement)",
        "brake_caliper": "$200 - $500 (caliper service)",
        "cooling_pump": "$200 - $600 (pump replacement)",
        "cooling_radiator": "$400 - $900 (radiator replacement)",
        "general_diagnostic": "$100 - $200 (diagnostic fee)",
    }

    def __init__(self, knowledge_base=None):
        """Initialize Diagnosis Agent."""
        self.knowledge_base = knowledge_base
        self.llm = None
        self.active_diagnoses: Dict[str, DiagnosisResult] = {}
        self._init_gemini()

    def _init_gemini(self):
        """Initialize Gemini LLM."""
        if not GEMINI_AVAILABLE:
            return

        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if api_key:
            try:
                genai.configure(api_key=api_key)
                self.llm = genai.GenerativeModel("gemini-2.0-flash")
                print("✓ Diagnosis Agent Gemini initialized")
            except Exception as e:
                print(f"⚠️ Diagnosis Agent Gemini init failed: {e}")

    def _get_rag_diagnosis(self, component: str, issue_type: str) -> List[Dict]:
        """Get diagnostic info from RAG knowledge base."""
        if not self.knowledge_base:
            return []

        try:
            query = f"{component} {issue_type} diagnosis repair service procedure"
            results = self.knowledge_base.semantic_search(query, k=5)
            return results
        except Exception as e:
            print(f"RAG diagnosis search error: {e}")
            return []

    def _determine_urgency(self, severity: str, days_to_failure: int) -> str:
        """Determine repair urgency."""
        if severity == "critical" or days_to_failure <= 2:
            return "immediate"
        elif severity == "high" or days_to_failure <= 5:
            return "soon"
        else:
            return "scheduled"

    def _get_repair_cost(self, component: str, issue_type: str) -> str:
        """Get estimated repair cost range."""
        key = f"{component}_{issue_type.split('_')[0]}".lower()
        return self.REPAIR_COSTS.get(key, self.REPAIR_COSTS["general_diagnostic"])

    def _generate_diagnostic_steps(
        self, component: str, telemetry_data: Dict[str, Any], rag_results: List[Dict]
    ) -> List[DiagnosisStep]:
        """Generate diagnostic steps based on component and data."""
        procedures = self.DIAGNOSTIC_PROCEDURES.get(
            component, self.DIAGNOSTIC_PROCEDURES.get("battery", [])
        )

        steps = []
        for i, (title, description, metric) in enumerate(procedures, 1):
            # Check telemetry for this metric
            value = telemetry_data.get(metric, None)

            if value is not None:
                # Determine finding status based on thresholds
                if "temp" in metric.lower():
                    if value > 60:
                        status = "critical"
                        finding = f"Temperature elevated: {value}°C (critical)"
                    elif value > 45:
                        status = "warning"
                        finding = f"Temperature elevated: {value}°C (warning)"
                    else:
                        status = "ok"
                        finding = f"Temperature normal: {value}°C"
                elif "efficiency" in metric.lower() or "pct" in metric.lower():
                    if value < 0.5:
                        status = "critical"
                        finding = f"Efficiency low: {value:.1%}"
                    elif value < 0.75:
                        status = "warning"
                        finding = f"Efficiency below optimal: {value:.1%}"
                    else:
                        status = "ok"
                        finding = f"Efficiency normal: {value:.1%}"
                else:
                    status = "ok"
                    finding = f"Measured value: {value}"
            else:
                status = "pending"
                finding = "Requires physical inspection"

            steps.append(
                DiagnosisStep(
                    step_number=i,
                    title=title,
                    description=description,
                    finding=finding,
                    status=status,
                )
            )

        return steps

    def _identify_root_cause(
        self, steps: List[DiagnosisStep], rag_results: List[Dict]
    ) -> str:
        """Identify root cause from diagnostic steps and RAG."""
        critical_findings = [s for s in steps if s.status == "critical"]
        warning_findings = [s for s in steps if s.status == "warning"]

        if critical_findings:
            root_cause = f"Critical issue detected in: {critical_findings[0].title}. "
            root_cause += critical_findings[0].finding
        elif warning_findings:
            root_cause = f"Warning condition in: {warning_findings[0].title}. "
            root_cause += warning_findings[0].finding
        else:
            root_cause = (
                "No immediate issues detected, but preventive service recommended."
            )

        # Enhance with RAG
        if rag_results:
            rag_content = rag_results[0].get("content", "")
            if "root_cause" in rag_content.lower() or "rca" in rag_content.lower():
                # Extract relevant info
                for line in rag_content.split("\n"):
                    if "cause" in line.lower():
                        root_cause += f" Industry data suggests: {line.strip()[:100]}"
                        break

        return root_cause

    def _generate_gemini_summary(
        self,
        component: str,
        steps: List[DiagnosisStep],
        root_cause: str,
        repair_action: str,
        rag_results: List[Dict],
    ) -> str:
        """Generate diagnosis summary using Gemini."""
        if not self.llm:
            return f"Diagnosis complete for {component}. {root_cause}"

        # Build context
        step_summary = "\n".join(
            [
                f"- {s.title}: {s.finding} ({s.status})"
                for s in steps
                if s.status != "ok"
            ]
        )

        rag_context = ""
        if rag_results:
            rag_context = rag_results[0].get("content", "")[:300]

        prompt = f"""You are an expert EV mechanic explaining a diagnosis to a customer.

Component: {component}
Diagnostic Findings:
{step_summary}

Root Cause: {root_cause}
Recommended Action: {repair_action}

Industry Reference:
{rag_context}

Write a clear, professional summary (3-4 sentences) explaining:
1. What the diagnosis found
2. Why this is a problem
3. What needs to be done

Use simple language that a non-technical person can understand."""

        try:
            response = self.llm.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"Gemini diagnosis summary error: {e}")
            return f"Diagnosis complete for {component}. {root_cause} Recommended action: {repair_action}"

    def start_diagnosis(
        self,
        diagnosis_id: str,
        vehicle_id: str,
        component: str,
        issue_type: str,
        severity: str,
        days_to_failure: int,
        telemetry_data: Dict[str, Any] = None,
    ) -> DiagnosisResult:
        """
        Start a diagnosis process for a component issue.

        Args:
            diagnosis_id: Unique diagnosis identifier
            vehicle_id: Vehicle identifier
            component: Affected component
            issue_type: Type of issue
            severity: Severity level
            days_to_failure: Estimated days until failure
            telemetry_data: Current telemetry readings

        Returns:
            DiagnosisResult with steps and recommendations
        """
        telemetry_data = telemetry_data or {}

        # Get RAG information
        rag_results = self._get_rag_diagnosis(component, issue_type)
        rag_sources = [
            r.get("metadata", {}).get("fault_id", "unknown") for r in rag_results
        ]

        # Generate diagnostic steps
        steps = self._generate_diagnostic_steps(component, telemetry_data, rag_results)

        # Identify root cause
        root_cause = self._identify_root_cause(steps, rag_results)

        # Determine repair action
        critical_steps = [s for s in steps if s.status == "critical"]
        if critical_steps:
            repair_action = f"Replace/repair {component} - {critical_steps[0].title}"
        else:
            repair_action = f"Preventive service recommended for {component}"

        # Get cost estimate
        cost_range = self._get_repair_cost(component, issue_type)

        # Determine urgency
        urgency = self._determine_urgency(severity, days_to_failure)

        # Generate Gemini summary
        gemini_summary = self._generate_gemini_summary(
            component, steps, root_cause, repair_action, rag_results
        )

        result = DiagnosisResult(
            diagnosis_id=diagnosis_id,
            vehicle_id=vehicle_id,
            component=component,
            issue_type=issue_type,
            severity=severity,
            status=DiagnosisStatus.COMPLETED,
            steps=steps,
            root_cause=root_cause,
            repair_action=repair_action,
            estimated_cost_range=cost_range,
            urgency=urgency,
            service_required=severity in ["high", "critical"],
            rag_sources=rag_sources,
            gemini_summary=gemini_summary,
        )

        # Store active diagnosis
        self.active_diagnoses[diagnosis_id] = result

        return result

    def confirm_service(self, diagnosis_id: str) -> Dict[str, Any]:
        """Confirm service scheduling for a diagnosis."""
        if diagnosis_id not in self.active_diagnoses:
            return {"error": "Diagnosis not found"}

        diagnosis = self.active_diagnoses[diagnosis_id]
        diagnosis.status = DiagnosisStatus.SERVICE_SCHEDULED

        return {
            "type": "service_scheduled",
            "diagnosis_id": diagnosis_id,
            "vehicle_id": diagnosis.vehicle_id,
            "component": diagnosis.component,
            "repair_action": diagnosis.repair_action,
            "estimated_cost": diagnosis.estimated_cost_range,
            "urgency": diagnosis.urgency,
            "message": f"Service scheduled for {diagnosis.component}. A service advisor will contact you shortly.",
            "timestamp": datetime.now().isoformat(),
        }

    def decline_service(self, diagnosis_id: str) -> Dict[str, Any]:
        """User declines service scheduling."""
        if diagnosis_id not in self.active_diagnoses:
            return {"error": "Diagnosis not found"}

        diagnosis = self.active_diagnoses[diagnosis_id]
        diagnosis.status = DiagnosisStatus.SERVICE_DECLINED

        return {
            "type": "service_declined",
            "diagnosis_id": diagnosis_id,
            "vehicle_id": diagnosis.vehicle_id,
            "warning": f"⚠️ Continuing to drive without repair may cause further damage to {diagnosis.component}.",
            "monitoring": "We will continue to monitor and alert you if the condition worsens.",
            "timestamp": datetime.now().isoformat(),
        }

    def to_chatbot_message(self, result: DiagnosisResult) -> Dict[str, Any]:
        """Convert DiagnosisResult to chatbot message format."""
        # Format steps for display
        steps_formatted = []
        for step in result.steps:
            icon = (
                "✅"
                if step.status == "ok"
                else "⚠️" if step.status == "warning" else "❌"
            )
            steps_formatted.append(
                {
                    "step": step.step_number,
                    "title": step.title,
                    "finding": step.finding,
                    "status": step.status,
                    "icon": icon,
                }
            )

        return {
            "type": "diagnosis",
            "diagnosis_id": result.diagnosis_id,
            "component": result.component,  # Added for voice call trigger detection
            "title": f"🔧 Diagnosis Complete - {result.component.replace('_', ' ').title()}",
            "summary": result.gemini_summary,
            "steps": steps_formatted,
            "root_cause": result.root_cause,
            "repair_action": result.repair_action,
            "estimated_cost": result.estimated_cost_range,
            "urgency": result.urgency,
            "service_required": result.service_required,
            "rag_sources": result.rag_sources,
            "actions": (
                ["confirm_service", "decline_service"]
                if result.service_required
                else []
            ),
            "timestamp": result.timestamp,
        }


# Singleton instance
_diagnosis_agent = None


def get_diagnosis_agent(knowledge_base=None) -> DiagnosisAgent:
    """Get or create Diagnosis Agent singleton."""
    global _diagnosis_agent
    if _diagnosis_agent is None:
        _diagnosis_agent = DiagnosisAgent(knowledge_base)
    elif knowledge_base and _diagnosis_agent.knowledge_base is None:
        _diagnosis_agent.knowledge_base = knowledge_base
    return _diagnosis_agent
