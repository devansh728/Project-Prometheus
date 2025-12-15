"""
SentinEV - Realistic Driving Scenarios
Five complex scenarios for testing the predictive maintenance system.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
import random
import math


class ScenarioPhase(Enum):
    """Phases within a scenario."""

    NORMAL = "normal"
    BUILDING = "building"  # Issue building up
    WARNING = "warning"  # System detects issue
    CRITICAL = "critical"  # Critical threshold
    RESOLVED = "resolved"  # After intervention


@dataclass
class ScenarioEvent:
    """Event within a scenario timeline."""

    time_offset_seconds: int
    event_type: str
    description: str
    telemetry_modifiers: Dict[str, Any]
    phase: ScenarioPhase
    triggers_prediction: bool = False
    prediction_message: Optional[str] = None
    days_to_failure: Optional[int] = None


@dataclass
class Scenario:
    """Complete scenario definition."""

    id: str
    name: str
    description: str
    duration_seconds: int
    component: str
    anomaly_type: str
    target_severity: str
    events: List[ScenarioEvent]
    requires_service: bool  # True = Diagnosis Agent, False = Safety Agent
    chatbot_intro: str


class ScenarioManager:
    """
    Manages realistic driving scenarios for testing.

    Each scenario simulates a progression from normal driving
    to anomaly detection with realistic telemetry patterns.
    """

    def __init__(self):
        self.scenarios = self._build_scenarios()
        self.active_scenarios: Dict[str, Dict] = {}  # vehicle_id -> scenario state

    def _build_scenarios(self) -> Dict[str, Scenario]:
        """Build all predefined scenarios."""
        return {
            "aggressive_driver": self._scenario_aggressive_driver(),
            "battery_stress": self._scenario_battery_stress(),
            "brake_fade": self._scenario_brake_fade(),
            "motor_strain": self._scenario_motor_strain(),
            "combined_stress": self._scenario_combined_stress(),
        }

    def _scenario_aggressive_driver(self) -> Scenario:
        """Scenario 1: Aggressive driving pattern leading to brake wear."""
        # FAST MODE: Reduced times by 10x for testing
        events = [
            # Phase 1: Normal driving (0-5s)
            ScenarioEvent(
                time_offset_seconds=0,
                event_type="start",
                description="Starting normal city driving",
                telemetry_modifiers={"speed_kmh": 50, "jerk_ms3": 1.0},
                phase=ScenarioPhase.NORMAL,
            ),
            # Phase 2: Aggressive behavior starts (5-15s)
            ScenarioEvent(
                time_offset_seconds=5,
                event_type="behavior_change",
                description="Driver becomes aggressive in traffic",
                telemetry_modifiers={
                    "jerk_ms3": lambda: random.uniform(4.5, 7.0)
                    * random.choice([1, -1]),
                    "acceleration_ms2": lambda: random.uniform(3.0, 5.0)
                    * random.choice([1, -1]),
                    "speed_kmh": lambda: random.uniform(80, 130),
                },
                phase=ScenarioPhase.BUILDING,
            ),
            # Phase 3: System should detect from behavior (15s)
            ScenarioEvent(
                time_offset_seconds=15,
                event_type="prediction",
                description="ML model should detect aggressive pattern",
                telemetry_modifiers={
                    # NO temp overrides - let physics/fault injection drive temps
                    "speed_kmh": lambda: random.uniform(90, 140),
                    "wear_index": lambda: random.uniform(0.4, 0.6),
                },
                phase=ScenarioPhase.WARNING,
                triggers_prediction=False,  # DISABLED: Now ML-driven only
                prediction_message="Based on your driving pattern, brake pads may wear excessively within 7 days. Harsh braking detected 15 times in the last 3 minutes.",
                days_to_failure=7,
            ),
            # Phase 4: Continued aggressive if ignored (25-45s)
            ScenarioEvent(
                time_offset_seconds=25,
                event_type="escalation",
                description="Aggressive driving continues",
                telemetry_modifiers={
                    # NO temp overrides - only behavioral changes
                    "speed_kmh": lambda: random.uniform(100, 150),
                    "wear_index": lambda: random.uniform(0.6, 0.8),
                    "jerk_ms3": lambda: random.uniform(5.0, 8.0)
                    * random.choice([1, -1]),
                },
                phase=ScenarioPhase.CRITICAL,
            ),
        ]

        return Scenario(
            id="aggressive_driver",
            name="Aggressive Driver Pattern",
            description="[FAST] Aggressive driving → brake wear. Warning at ~15s.",
            duration_seconds=45,
            component="brakes",
            anomaly_type="driving_behavior",
            target_severity="medium",
            events=events,
            requires_service=False,  # Can be fixed by behavior change → Safety Agent
            chatbot_intro="🚗 Starting Aggressive Driver scenario (FAST MODE). Warning in ~15 seconds.",
        )

    def _scenario_battery_stress(self) -> Scenario:
        """Scenario 2: Battery thermal stress from hot weather + fast charging."""
        # FAST MODE: Reduced times by 10x for testing
        events = [
            ScenarioEvent(
                time_offset_seconds=0,
                event_type="start",
                description="Hot summer day highway driving",
                telemetry_modifiers={
                    "speed_kmh": 110,
                    "battery_temp_c": 38,
                    "battery_soc_pct": 45,
                },
                phase=ScenarioPhase.NORMAL,
            ),
            ScenarioEvent(
                time_offset_seconds=8,
                event_type="temperature_rise",
                description="Battery temperature rising under load",
                telemetry_modifiers={
                    # NO temp overrides - let physics/fault injection drive temps
                    "power_draw_kw": lambda: random.uniform(60, 85),
                },
                phase=ScenarioPhase.BUILDING,
            ),
            ScenarioEvent(
                time_offset_seconds=20,
                event_type="thermal_warning",
                description="Battery approaching thermal limits",
                telemetry_modifiers={
                    # NO temp overrides - only behavioral changes
                    "power_draw_kw": lambda: random.uniform(40, 60),
                },
                phase=ScenarioPhase.WARNING,
                triggers_prediction=False,  # DISABLED: Now ML-driven only
                prediction_message="Battery temperature is elevated. Continued high-power driving in this heat may cause thermal degradation within 8 days.",
                days_to_failure=8,
            ),
            ScenarioEvent(
                time_offset_seconds=35,
                event_type="critical_temp",
                description="Battery temperature critical",
                telemetry_modifiers={
                    # NO temp overrides
                    "power_draw_kw": lambda: random.uniform(30, 50),
                },
                phase=ScenarioPhase.CRITICAL,
            ),
        ]

        return Scenario(
            id="battery_stress",
            name="Battery Thermal Stress",
            description="[FAST] Hot weather → battery overheating. Warning at ~20s.",
            duration_seconds=50,
            component="battery",
            anomaly_type="thermal_battery",
            target_severity="high",
            events=events,
            requires_service=False,  # Can cool down → Safety Agent
            chatbot_intro="☀️ Starting Battery Thermal Stress (FAST MODE). Warning in ~20 seconds.",
        )

    def _scenario_brake_fade(self) -> Scenario:
        """Scenario 3: Mountain driving with brake fade - REQUIRES SERVICE (Diagnosis Agent)."""
        # FAST MODE: Reduced times by 10x for testing
        events = [
            ScenarioEvent(
                time_offset_seconds=0,
                event_type="start",
                description="Starting mountain descent",
                telemetry_modifiers={
                    "speed_kmh": 60,
                    "brake_temp_c": 80,
                    "regen_efficiency": 0.75,
                    "battery_soc_pct": 85,
                },
                phase=ScenarioPhase.NORMAL,
            ),
            ScenarioEvent(
                time_offset_seconds=5,
                event_type="regen_limited",
                description="Regen limited due to high SOC",
                telemetry_modifiers={
                    "regen_efficiency": lambda: random.uniform(0.3, 0.5),
                    # NO brake_temp override - let physics/fault drive temps
                },
                phase=ScenarioPhase.BUILDING,
            ),
            ScenarioEvent(
                time_offset_seconds=12,
                event_type="brake_heating",
                description="Friction brakes taking heavy load",
                telemetry_modifiers={
                    # NO temp overrides - only behavioral changes
                    "regen_efficiency": lambda: random.uniform(0.2, 0.4),
                },
                phase=ScenarioPhase.WARNING,
                triggers_prediction=False,  # DISABLED: Now ML-driven only
                prediction_message="CRITICAL: Brake temperature elevated due to sustained downhill braking. Brake fade may occur. Service required.",
                days_to_failure=3,
            ),
            ScenarioEvent(
                time_offset_seconds=22,
                event_type="brake_critical",
                description="Brake fade imminent",
                telemetry_modifiers={
                    # NO temp overrides
                },
                phase=ScenarioPhase.CRITICAL,
            ),
        ]

        return Scenario(
            id="brake_fade",
            name="Mountain Brake Fade ⚠️",
            description="[FAST] CRITICAL: Brake overheating → Diagnosis Agent. Warning at ~12s.",
            duration_seconds=35,
            component="brakes",
            anomaly_type="thermal_brake",
            target_severity="critical",
            events=events,
            requires_service=True,  # → Diagnosis Agent (not Safety Agent)
            chatbot_intro="⛰️ Starting Mountain Brake Fade (FAST MODE). CRITICAL scenario - routes to Diagnosis Agent. Warning in ~12 seconds.",
        )

    def _scenario_motor_strain(self) -> Scenario:
        """Scenario 4: High-speed sustained driving straining motor/inverter."""
        # FAST MODE: Reduced times by 10x for testing
        events = [
            ScenarioEvent(
                time_offset_seconds=0,
                event_type="start",
                description="Entering highway, accelerating",
                telemetry_modifiers={
                    "speed_kmh": 80,
                    "motor_temp_c": 55,
                    "inverter_temp_c": 50,
                },
                phase=ScenarioPhase.NORMAL,
            ),
            ScenarioEvent(
                time_offset_seconds=6,
                event_type="high_speed",
                description="Sustained high-speed driving",
                telemetry_modifiers={
                    "speed_kmh": lambda: random.uniform(140, 160),
                    # NO temp overrides - let physics/fault drive temps
                    "power_draw_kw": lambda: random.uniform(80, 120),
                },
                phase=ScenarioPhase.BUILDING,
            ),
            ScenarioEvent(
                time_offset_seconds=18,
                event_type="thermal_buildup",
                description="Motor and inverter heating up",
                telemetry_modifiers={
                    # NO temp overrides - only behavioral changes
                    "power_draw_kw": lambda: random.uniform(60, 80),
                },
                phase=ScenarioPhase.WARNING,
                triggers_prediction=False,  # DISABLED: Now ML-driven only
                prediction_message="Motor and inverter temperatures elevated from sustained high-speed driving. Cooling period recommended to prevent inverter damage.",
                days_to_failure=7,
            ),
            ScenarioEvent(
                time_offset_seconds=30,
                event_type="power_derating",
                description="System derating power to protect components",
                telemetry_modifiers={
                    # NO temp overrides
                    "power_draw_kw": lambda: random.uniform(40, 60),
                },
                phase=ScenarioPhase.CRITICAL,
            ),
        ]

        return Scenario(
            id="motor_strain",
            name="Motor/Inverter Strain",
            description="[FAST] High-speed driving → motor overheating. Warning at ~18s.",
            duration_seconds=40,
            component="motor",
            anomaly_type="thermal_motor",
            target_severity="high",
            events=events,
            requires_service=False,  # Can cool down → Safety Agent
            chatbot_intro="🏎️ Starting Motor/Inverter Strain (FAST MODE). Warning in ~18 seconds.",
        )

    def _scenario_combined_stress(self) -> Scenario:
        """Scenario 5: Multiple systems under stress - REQUIRES SERVICE (Diagnosis Agent)."""
        # FAST MODE: Reduced times by 10x for testing
        events = [
            ScenarioEvent(
                time_offset_seconds=0,
                event_type="start",
                description="Hot day, aggressive driving start",
                telemetry_modifiers={
                    "speed_kmh": 60,
                    "battery_temp_c": 35,
                    "motor_temp_c": 50,
                    "brake_temp_c": 60,
                },
                phase=ScenarioPhase.NORMAL,
            ),
            ScenarioEvent(
                time_offset_seconds=8,
                event_type="multi_stress_begin",
                description="Aggressive driving in hot weather",
                telemetry_modifiers={
                    "jerk_ms3": lambda: random.uniform(4.0, 6.0)
                    * random.choice([1, -1]),
                    # NO temp overrides - let physics/fault drive temps
                    "speed_kmh": lambda: random.uniform(100, 140),
                },
                phase=ScenarioPhase.BUILDING,
            ),
            ScenarioEvent(
                time_offset_seconds=20,
                event_type="systems_warning",
                description="Multiple systems showing stress",
                telemetry_modifiers={
                    # NO temp overrides - only behavioral changes
                    "wear_index": lambda: random.uniform(0.5, 0.7),
                },
                phase=ScenarioPhase.WARNING,
                triggers_prediction=False,  # DISABLED: Now ML-driven only
                prediction_message="CRITICAL: Multiple systems under stress - battery thermal, motor thermal, brake wear. Comprehensive service required.",
                days_to_failure=5,
            ),
            ScenarioEvent(
                time_offset_seconds=35,
                event_type="multi_critical",
                description="Multiple systems critical",
                telemetry_modifiers={
                    # NO temp overrides
                    "wear_index": lambda: random.uniform(0.7, 0.9),
                },
                phase=ScenarioPhase.CRITICAL,
            ),
        ]

        return Scenario(
            id="combined_stress",
            name="Combined System Stress ⚠️",
            description="[FAST] CRITICAL: Multi-system failure → Diagnosis Agent. Warning at ~20s.",
            duration_seconds=50,
            component="multiple",
            anomaly_type="combined",
            target_severity="critical",
            events=events,
            requires_service=True,  # → Diagnosis Agent (not Safety Agent)
            chatbot_intro="⚡ Starting Combined Stress (FAST MODE). CRITICAL scenario - routes to Diagnosis Agent. Warning in ~20 seconds.",
        )

    def list_scenarios(self) -> List[Dict[str, Any]]:
        """List all available scenarios."""
        return [
            {
                "id": s.id,
                "name": s.name,
                "description": s.description,
                "duration_seconds": s.duration_seconds,
                "component": s.component,
                "severity": s.target_severity,
                "requires_service": s.requires_service,
            }
            for s in self.scenarios.values()
        ]

    def start_scenario(self, vehicle_id: str, scenario_id: str) -> Dict[str, Any]:
        """Start a scenario for a vehicle."""
        if scenario_id not in self.scenarios:
            return {"error": f"Scenario {scenario_id} not found"}

        scenario = self.scenarios[scenario_id]

        self.active_scenarios[vehicle_id] = {
            "scenario": scenario,
            "started_at": datetime.now(),
            "current_event_index": 0,
            "prediction_triggered": False,
            "user_response": None,  # accept/reject
        }

        return {
            "status": "started",
            "vehicle_id": vehicle_id,
            "scenario_id": scenario_id,
            "scenario_name": scenario.name,
            "duration_seconds": scenario.duration_seconds,
            "chatbot_message": {
                "type": "scenario_start",
                "message": scenario.chatbot_intro,
                "scenario": scenario.name,
                "timestamp": datetime.now().isoformat(),
            },
        }

    def get_current_modifiers(self, vehicle_id: str) -> Optional[Dict[str, Any]]:
        """Get current telemetry modifiers for active scenario."""
        if vehicle_id not in self.active_scenarios:
            return None

        state = self.active_scenarios[vehicle_id]
        scenario = state["scenario"]
        elapsed = (datetime.now() - state["started_at"]).total_seconds()

        # Find current event
        current_event = None
        for event in reversed(scenario.events):
            if elapsed >= event.time_offset_seconds:
                current_event = event
                break

        if not current_event:
            current_event = scenario.events[0]

        # Apply modifiers
        modifiers = {}
        for key, value in current_event.telemetry_modifiers.items():
            if callable(value):
                modifiers[key] = value()
            else:
                modifiers[key] = value

        # Check if prediction should trigger
        result = {
            "modifiers": modifiers,
            "phase": current_event.phase.value,
            "event_type": current_event.event_type,
            "description": current_event.description,
        }

        if current_event.triggers_prediction and not state["prediction_triggered"]:
            state["prediction_triggered"] = True
            result["trigger_prediction"] = {
                "message": current_event.prediction_message,
                "days_to_failure": current_event.days_to_failure,
                "component": scenario.component,
                "anomaly_type": scenario.anomaly_type,
                "severity": scenario.target_severity,
                "requires_service": scenario.requires_service,
            }

        # Check if scenario complete
        if elapsed >= scenario.duration_seconds:
            result["scenario_complete"] = True

        return result

    def get_scenario_status(self, vehicle_id: str) -> Optional[Dict[str, Any]]:
        """Get current scenario status for a vehicle."""
        if vehicle_id not in self.active_scenarios:
            return None

        state = self.active_scenarios[vehicle_id]
        scenario = state["scenario"]
        elapsed = (datetime.now() - state["started_at"]).total_seconds()

        return {
            "vehicle_id": vehicle_id,
            "scenario_id": scenario.id,
            "scenario_name": scenario.name,
            "elapsed_seconds": int(elapsed),
            "duration_seconds": scenario.duration_seconds,
            "progress_pct": min(100, int((elapsed / scenario.duration_seconds) * 100)),
            "prediction_triggered": state["prediction_triggered"],
            "user_response": state["user_response"],
            "active": elapsed < scenario.duration_seconds,
        }

    def stop_scenario(self, vehicle_id: str) -> Dict[str, Any]:
        """Stop active scenario for a vehicle."""
        if vehicle_id in self.active_scenarios:
            scenario = self.active_scenarios[vehicle_id]["scenario"]
            del self.active_scenarios[vehicle_id]
            return {
                "status": "stopped",
                "vehicle_id": vehicle_id,
                "scenario_id": scenario.id,
            }
        return {"status": "no_active_scenario"}

    def set_user_response(self, vehicle_id: str, response: str) -> bool:
        """Set user response (accept/reject) for active scenario."""
        if vehicle_id in self.active_scenarios:
            self.active_scenarios[vehicle_id]["user_response"] = response
            return True
        return False


# Singleton instance
_scenario_manager = None


def get_scenario_manager() -> ScenarioManager:
    """Get or create ScenarioManager singleton."""
    global _scenario_manager
    if _scenario_manager is None:
        _scenario_manager = ScenarioManager()
    return _scenario_manager
