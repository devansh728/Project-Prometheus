"""
SentinEV - Monitoring Utilities
Support structures for UEBA, SLA enforcement, and operational reporting.
"""

from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import hashlib


class UEBARuleType(Enum):
    """UEBA security rule types."""

    FREQUENCY_ANOMALY = "frequency_anomaly"
    OFF_HOURS_ACCESS = "off_hours_access"
    SENSITIVE_RESOURCE = "sensitive_resource"
    CROSS_TENANT = "cross_tenant"


class SLAStatus(Enum):
    """SLA compliance status."""

    COMPLIANT = "compliant"
    WARNING = "warning"
    VIOLATION = "violation"


@dataclass
class UEBARule:
    """UEBA security rule definition."""

    rule_id: str
    rule_type: UEBARuleType
    name: str
    description: str
    threshold: float
    risk_score: float  # Score added when triggered (0-10)
    enabled: bool = True

    def check(self, context: Dict) -> bool:
        """Check if rule is violated. Override in subclasses."""
        return False


@dataclass
class SLAThreshold:
    """Latency SLA threshold definition."""

    operation: str
    warning_ms: float
    violation_ms: float
    description: str = ""


@dataclass
class AgentFingerprint:
    """Agent identity fingerprint."""

    agent_id: str
    fingerprint_hash: str
    ip_hash: str
    user_agent: str
    session_id: str
    first_seen: str
    last_seen: str
    is_suspicious: bool = False


@dataclass
class DailyReport:
    """Daily operational report."""

    report_date: str
    generated_at: str
    inference_stats: Dict[str, Any]
    alert_summary: Dict[str, Any]
    ueba_events: List[Dict]
    drift_status: Dict[str, Any]
    sla_compliance: Dict[str, Any]
    top_vehicles: List[Dict]
    recommendations: List[str]


# ==================== UEBA Rule Definitions ====================

# Rule 1: Anomalous action frequency (>10 actions/min)
RULE_FREQUENCY_ANOMALY = UEBARule(
    rule_id="UEBA-001",
    rule_type=UEBARuleType.FREQUENCY_ANOMALY,
    name="Anomalous Action Frequency",
    description="Agent performing >10 actions per minute",
    threshold=10,  # actions per minute
    risk_score=6.0,
)

# Rule 2: Off-hours access (outside 6am-10pm)
RULE_OFF_HOURS_ACCESS = UEBARule(
    rule_id="UEBA-002",
    rule_type=UEBARuleType.OFF_HOURS_ACCESS,
    name="Off-Hours Access",
    description="Access outside business hours (6am-10pm)",
    threshold=0,  # N/A for time-based
    risk_score=4.0,
)

# Rule 3: Sensitive resource access pattern
RULE_SENSITIVE_RESOURCE = UEBARule(
    rule_id="UEBA-003",
    rule_type=UEBARuleType.SENSITIVE_RESOURCE,
    name="Sensitive Resource Access",
    description="Access to PII, credentials, or admin resources",
    threshold=0,  # N/A
    risk_score=8.0,
)

# Rule 4: Cross-tenant access attempts
RULE_CROSS_TENANT = UEBARule(
    rule_id="UEBA-004",
    rule_type=UEBARuleType.CROSS_TENANT,
    name="Cross-Tenant Access",
    description="Agent attempting to access resources from another tenant",
    threshold=0,  # N/A
    risk_score=9.0,
)

# All rules
UEBA_RULES = [
    RULE_FREQUENCY_ANOMALY,
    RULE_OFF_HOURS_ACCESS,
    RULE_SENSITIVE_RESOURCE,
    RULE_CROSS_TENANT,
]

# Sensitive resources list
SENSITIVE_RESOURCES = [
    "user_credentials",
    "api_keys",
    "pii_data",
    "admin_panel",
    "billing_info",
    "encryption_keys",
    "audit_logs",
    "security_config",
]

# ==================== SLA Thresholds ====================

SLA_THRESHOLDS = {
    "inference": SLAThreshold(
        operation="inference",
        warning_ms=500,
        violation_ms=1000,
        description="ML inference prediction",
    ),
    "anomaly_detection": SLAThreshold(
        operation="anomaly_detection",
        warning_ms=200,
        violation_ms=500,
        description="Anomaly detection run",
    ),
    "voice_response": SLAThreshold(
        operation="voice_response",
        warning_ms=1000,
        violation_ms=3000,
        description="Voice agent response time",
    ),
    "scheduling": SLAThreshold(
        operation="scheduling",
        warning_ms=300,
        violation_ms=800,
        description="Appointment scheduling",
    ),
    "database_query": SLAThreshold(
        operation="database_query",
        warning_ms=50,
        violation_ms=200,
        description="Database query execution",
    ),
}


# ==================== UEBA Rule Checkers ====================


def check_frequency_anomaly(
    agent_id: str,
    action_count: int,
    time_window_seconds: int = 60,
) -> tuple[bool, str]:
    """
    Check if agent is performing too many actions.

    Args:
        agent_id: Agent identifier
        action_count: Number of actions in time window
        time_window_seconds: Time window (default 60s)

    Returns:
        Tuple of (is_violation, reason)
    """
    threshold = RULE_FREQUENCY_ANOMALY.threshold * (time_window_seconds / 60)

    if action_count > threshold:
        return (
            True,
            f"Agent {agent_id} performed {action_count} actions in {time_window_seconds}s (threshold: {threshold})",
        )
    return False, ""


def check_off_hours_access(current_time: datetime = None) -> tuple[bool, str]:
    """
    Check if access is during off-hours.

    Business hours: 6am - 10pm
    """
    if current_time is None:
        current_time = datetime.now()

    hour = current_time.hour

    if hour < 6 or hour >= 22:
        return True, f"Off-hours access at {current_time.strftime('%H:%M')}"
    return False, ""


def check_sensitive_resource(resource: str) -> tuple[bool, str]:
    """
    Check if resource is sensitive.
    """
    resource_lower = resource.lower()

    for sensitive in SENSITIVE_RESOURCES:
        if sensitive in resource_lower:
            return True, f"Access to sensitive resource: {resource}"
    return False, ""


def check_cross_tenant(
    agent_tenant: str,
    resource_tenant: str,
) -> tuple[bool, str]:
    """
    Check for cross-tenant access attempt.
    """
    if agent_tenant != resource_tenant:
        return (
            True,
            f"Cross-tenant access: agent={agent_tenant}, resource={resource_tenant}",
        )
    return False, ""


# ==================== SLA Helpers ====================


def check_sla_status(
    operation: str,
    latency_ms: float,
) -> tuple[SLAStatus, str]:
    """
    Check SLA status for an operation.

    Returns:
        Tuple of (status, message)
    """
    threshold = SLA_THRESHOLDS.get(operation)

    if not threshold:
        return SLAStatus.COMPLIANT, "No SLA defined"

    if latency_ms >= threshold.violation_ms:
        return (
            SLAStatus.VIOLATION,
            f"{operation} latency {latency_ms}ms exceeds violation threshold {threshold.violation_ms}ms",
        )
    elif latency_ms >= threshold.warning_ms:
        return (
            SLAStatus.WARNING,
            f"{operation} latency {latency_ms}ms exceeds warning threshold {threshold.warning_ms}ms",
        )

    return SLAStatus.COMPLIANT, f"{operation} latency {latency_ms}ms within SLA"


# ==================== Fingerprinting ====================


def generate_fingerprint(
    agent_id: str,
    ip_address: str = "127.0.0.1",
    user_agent: str = "SentinEV-Agent/1.0",
    session_id: str = None,
) -> AgentFingerprint:
    """
    Generate agent fingerprint.
    """
    import uuid

    if not session_id:
        session_id = str(uuid.uuid4())[:8]

    # Create fingerprint hash
    fingerprint_data = f"{agent_id}|{ip_address}|{user_agent}"
    fingerprint_hash = hashlib.sha256(fingerprint_data.encode()).hexdigest()[:16]

    # Hash IP for privacy
    ip_hash = hashlib.md5(ip_address.encode()).hexdigest()[:8]

    now = datetime.now().isoformat()

    return AgentFingerprint(
        agent_id=agent_id,
        fingerprint_hash=fingerprint_hash,
        ip_hash=ip_hash,
        user_agent=user_agent,
        session_id=session_id,
        first_seen=now,
        last_seen=now,
    )


def compare_fingerprints(
    stored: AgentFingerprint,
    current: AgentFingerprint,
) -> tuple[bool, List[str]]:
    """
    Compare fingerprints to detect changes.

    Returns:
        Tuple of (is_different, changed_fields)
    """
    changes = []

    if stored.fingerprint_hash != current.fingerprint_hash:
        changes.append("fingerprint_hash")
    if stored.ip_hash != current.ip_hash:
        changes.append("ip_hash")
    if stored.user_agent != current.user_agent:
        changes.append("user_agent")

    return len(changes) > 0, changes
