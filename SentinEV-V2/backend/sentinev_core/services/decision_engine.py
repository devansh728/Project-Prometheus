"""
Decision Engine for SentinEV Core
Implements the action matrix and confidence-based decision making.
"""

from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum

from sentinev_core.services.ml_pipeline import Severity


class ActionType(str, Enum):
    """Types of actions the system can take."""

    NONE = "none"
    LOG_ONLY = "log_only"
    PUSH_NOTIFICATION = "push_notification"
    IN_APP_ALERT = "in_app_alert"
    VOICE_CALL = "voice_call"
    EMERGENCY_ALERT = "emergency_alert"
    AUTO_SCHEDULE = "auto_schedule"


@dataclass
class Decision:
    """A decision made by the engine."""

    action: ActionType
    priority: int  # 1 = highest
    confidence: float
    should_notify_customer: bool
    should_notify_service_center: bool
    delay_seconds: int
    rationale: str


class DecisionEngine:
    """
    Central decision-making engine that determines what actions to take
    based on diagnosis results and configurable business rules.
    """

    # Action matrix: Maps (severity, failure_probability) to action
    # Format: (action_type, priority, notify_customer, notify_service, delay_seconds)
    ACTION_MATRIX = {
        # Emergency conditions
        (Severity.EMERGENCY, "high"): (
            ActionType.EMERGENCY_ALERT,
            1,
            True,
            True,
            0,
            "Emergency condition requires immediate action",
        ),
        (Severity.EMERGENCY, "medium"): (
            ActionType.VOICE_CALL,
            1,
            True,
            True,
            0,
            "Emergency level alert - immediate voice contact",
        ),
        # Critical conditions
        (Severity.CRITICAL, "high"): (
            ActionType.VOICE_CALL,
            2,
            True,
            True,
            60,
            "Critical failure probability - proactive voice call",
        ),
        (Severity.CRITICAL, "medium"): (
            ActionType.VOICE_CALL,
            2,
            True,
            True,
            300,
            "Critical severity with moderate probability",
        ),
        (Severity.CRITICAL, "low"): (
            ActionType.PUSH_NOTIFICATION,
            3,
            True,
            False,
            600,
            "Critical severity but lower probability",
        ),
        # Warning conditions
        (Severity.WARNING, "high"): (
            ActionType.PUSH_NOTIFICATION,
            3,
            True,
            True,
            1800,
            "Warning with high failure probability",
        ),
        (Severity.WARNING, "medium"): (
            ActionType.PUSH_NOTIFICATION,
            4,
            True,
            False,
            3600,
            "Standard warning notification",
        ),
        (Severity.WARNING, "low"): (
            ActionType.IN_APP_ALERT,
            5,
            True,
            False,
            7200,
            "Low priority warning - in-app only",
        ),
        # Info conditions
        (Severity.INFO, "high"): (
            ActionType.IN_APP_ALERT,
            5,
            True,
            False,
            86400,
            "Informational with elevated probability",
        ),
        (Severity.INFO, "medium"): (
            ActionType.LOG_ONLY,
            6,
            False,
            False,
            0,
            "Standard monitoring - log only",
        ),
        (Severity.INFO, "low"): (
            ActionType.NONE,
            7,
            False,
            False,
            0,
            "No action required",
        ),
    }

    # Confidence thresholds for action escalation
    CONFIDENCE_THRESHOLDS = {
        "high": 0.85,
        "medium": 0.70,
        "low": 0.50,
    }

    def __init__(self, min_confidence: float = 0.5):
        self.min_confidence = min_confidence

    def decide(
        self,
        severity: Severity,
        failure_probability: float,
        confidence: float,
        previous_actions: Optional[List[str]] = None,
    ) -> Decision:
        """
        Make a decision based on inputs.

        Args:
            severity: Classified severity level
            failure_probability: Predicted failure probability (0-1)
            confidence: Model confidence (0-1)
            previous_actions: List of actions already taken for this vehicle
        """
        # Categorize failure probability
        prob_category = self._categorize_probability(failure_probability)

        # Lookup in action matrix
        key = (severity, prob_category)
        if key in self.ACTION_MATRIX:
            action_config = self.ACTION_MATRIX[key]
        else:
            # Default fallback
            action_config = (ActionType.LOG_ONLY, 6, False, False, 0, "Default action")

        action_type, priority, notify_cust, notify_svc, delay, rationale = action_config

        # Apply confidence adjustment
        if confidence < self.min_confidence:
            # Downgrade action if confidence is too low
            action_type = self._downgrade_action(action_type)
            priority += 1
            rationale += f" (downgraded due to low confidence: {confidence:.0%})"

        # Check for duplicate actions
        if previous_actions and action_type.value in previous_actions:
            # Don't repeat the same action within short period
            action_type = self._downgrade_action(action_type)
            rationale += " (downgraded to avoid duplicate action)"

        return Decision(
            action=action_type,
            priority=priority,
            confidence=confidence,
            should_notify_customer=notify_cust and confidence >= 0.6,
            should_notify_service_center=notify_svc and confidence >= 0.7,
            delay_seconds=delay,
            rationale=rationale,
        )

    def _categorize_probability(self, probability: float) -> str:
        """Categorize failure probability."""
        if probability >= 0.7:
            return "high"
        elif probability >= 0.4:
            return "medium"
        else:
            return "low"

    def _downgrade_action(self, action: ActionType) -> ActionType:
        """Downgrade an action to a less intrusive one."""
        downgrade_map = {
            ActionType.EMERGENCY_ALERT: ActionType.VOICE_CALL,
            ActionType.VOICE_CALL: ActionType.PUSH_NOTIFICATION,
            ActionType.PUSH_NOTIFICATION: ActionType.IN_APP_ALERT,
            ActionType.IN_APP_ALERT: ActionType.LOG_ONLY,
            ActionType.LOG_ONLY: ActionType.NONE,
            ActionType.NONE: ActionType.NONE,
        }
        return downgrade_map.get(action, ActionType.LOG_ONLY)

    def should_escalate(
        self,
        current_action: ActionType,
        hours_since_first_alert: float,
        customer_responded: bool,
    ) -> Optional[ActionType]:
        """
        Determine if we should escalate to a more urgent action.
        """
        if customer_responded:
            return None

        # Escalation rules
        escalation_schedule = {
            ActionType.IN_APP_ALERT: (24, ActionType.PUSH_NOTIFICATION),
            ActionType.PUSH_NOTIFICATION: (48, ActionType.VOICE_CALL),
        }

        if current_action in escalation_schedule:
            threshold_hours, escalated_action = escalation_schedule[current_action]
            if hours_since_first_alert >= threshold_hours:
                return escalated_action

        return None


# Singleton instance
decision_engine = DecisionEngine()
