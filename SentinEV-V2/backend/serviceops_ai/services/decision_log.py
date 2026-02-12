"""
Decision Log for ServiceOps AI
Tracks all agent decisions for transparency and visualization.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class DecisionType(str, Enum):
    """Types of decisions logged by the system."""

    QUEUE_ENTRY = "QUEUE_ENTRY"
    QUEUE_DEQUEUE = "QUEUE_DEQUEUE"
    BIDDING_START = "BIDDING_START"
    BIDDING_COMPLETE = "BIDDING_COMPLETE"
    CENTER_SELECTED = "CENTER_SELECTED"
    SLOT_RESERVED = "SLOT_RESERVED"
    TASKS_CREATED = "TASKS_CREATED"
    TECHNICIAN_ASSIGNED = "TECHNICIAN_ASSIGNED"
    PREEMPTION = "PREEMPTION"
    REORDER_TRIGGERED = "REORDER_TRIGGERED"
    DIAGNOSIS_FEEDBACK = "DIAGNOSIS_FEEDBACK"
    SCHEDULE_UPDATED = "SCHEDULE_UPDATED"
    OVERLOAD_ALERT = "OVERLOAD_ALERT"
    SYSTEM_INFO = "SYSTEM_INFO"


@dataclass
class DecisionEntry:
    """A single decision/event entry."""

    id: str
    event_type: DecisionType
    timestamp: datetime
    entity_id: Optional[str]  # e.g., request_id, job_id
    entity_type: Optional[str]  # e.g., "request", "job", "task"
    details: Dict[str, Any]
    reasoning: str
    impact: Optional[str] = None  # What changed as a result


class DecisionLog:
    """
    Singleton that logs all system decisions for:
    - Live dashboard feed
    - Audit trail
    - Explainability for judges
    """

    def __init__(self):
        self._entries: List[DecisionEntry] = []
        self._counter = 0

    def log(
        self,
        event_type: DecisionType,
        details: Dict[str, Any],
        reasoning: str,
        entity_id: Optional[str] = None,
        entity_type: Optional[str] = None,
        impact: Optional[str] = None,
    ) -> DecisionEntry:
        """
        Log a decision/event.
        """
        self._counter += 1
        entry_id = f"DEC-{self._counter:06d}"

        entry = DecisionEntry(
            id=entry_id,
            event_type=event_type,
            timestamp=datetime.utcnow(),
            entity_id=entity_id,
            entity_type=entity_type,
            details=details,
            reasoning=reasoning,
            impact=impact,
        )

        self._entries.append(entry)

        return entry

    def get_recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get recent decisions for dashboard feed.
        """
        entries = self._entries[-limit:]
        entries = list(reversed(entries))  # Most recent first

        return [
            {
                "id": e.id,
                "event_type": e.event_type.value,
                "timestamp": e.timestamp.isoformat(),
                "entity_id": e.entity_id,
                "entity_type": e.entity_type,
                "details": e.details,
                "reasoning": e.reasoning,
                "impact": e.impact,
            }
            for e in entries
        ]

    def get_by_entity(self, entity_id: str) -> List[Dict]:
        """Get all decisions related to an entity."""
        return [
            {
                "id": e.id,
                "event_type": e.event_type.value,
                "timestamp": e.timestamp.isoformat(),
                "reasoning": e.reasoning,
            }
            for e in self._entries
            if e.entity_id == entity_id
        ]

    def get_by_type(self, event_type: DecisionType, limit: int = 20) -> List[Dict]:
        """Get decisions of a specific type."""
        filtered = [e for e in self._entries if e.event_type == event_type]
        return [
            {
                "id": e.id,
                "timestamp": e.timestamp.isoformat(),
                "entity_id": e.entity_id,
                "reasoning": e.reasoning,
            }
            for e in filtered[-limit:]
        ]

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of decisions."""
        counts = {}
        for e in self._entries:
            key = e.event_type.value
            counts[key] = counts.get(key, 0) + 1

        return {
            "total_decisions": len(self._entries),
            "by_type": counts,
            "last_decision_at": (
                self._entries[-1].timestamp.isoformat() if self._entries else None
            ),
        }

    def clear(self):
        """Clear all entries (for demo reset)."""
        self._entries.clear()
        self._counter = 0

    # Convenience methods for common events
    def log_queue_entry(self, request_id: str, vehicle_name: str, urgency: float):
        return self.log(
            DecisionType.QUEUE_ENTRY,
            {"request_id": request_id, "vehicle": vehicle_name, "urgency": urgency},
            f"Vehicle {vehicle_name} added to queue with urgency score {urgency:.1f}",
            entity_id=request_id,
            entity_type="request",
        )

    def log_bidding_complete(self, request_id: str, num_bids: int, winner: str):
        return self.log(
            DecisionType.BIDDING_COMPLETE,
            {"request_id": request_id, "bid_count": num_bids, "winner": winner},
            f"Bidding complete: {num_bids} centers participated, {winner} selected based on best overall score",
            entity_id=request_id,
            entity_type="request",
            impact=f"Service request assigned to {winner}",
        )

    def log_center_selected(self, request_id: str, center_name: str, reasoning: str):
        return self.log(
            DecisionType.CENTER_SELECTED,
            {"request_id": request_id, "center": center_name},
            reasoning,
            entity_id=request_id,
            entity_type="request",
        )

    def log_tasks_created(self, job_id: str, task_count: int, failure_type: str):
        return self.log(
            DecisionType.TASKS_CREATED,
            {"job_id": job_id, "task_count": task_count, "failure_type": failure_type},
            f"Job decomposed into {task_count} tasks based on {failure_type.replace('_', ' ')}",
            entity_id=job_id,
            entity_type="job",
        )

    def log_technician_assigned(self, task_id: str, mechanic_name: str, task_name: str):
        return self.log(
            DecisionType.TECHNICIAN_ASSIGNED,
            {"task_id": task_id, "mechanic": mechanic_name, "task": task_name},
            f"Assigned {mechanic_name} to '{task_name}' based on skill match and availability",
            entity_id=task_id,
            entity_type="task",
        )

    def log_preemption(self, request_id: str, vehicle_name: str, reason: str):
        return self.log(
            DecisionType.PREEMPTION,
            {"request_id": request_id, "vehicle": vehicle_name},
            f"URGENT: {vehicle_name} preempted queue - {reason}",
            entity_id=request_id,
            entity_type="request",
            impact="Lower priority jobs delayed",
        )

    def log_diagnosis_feedback(self, job_id: str, similarity: float, insight: str):
        return self.log(
            DecisionType.DIAGNOSIS_FEEDBACK,
            {"job_id": job_id, "similarity_score": similarity},
            f"Diagnosis feedback received: {similarity*100:.0f}% match. {insight}",
            entity_id=job_id,
            entity_type="job",
        )

    def log_reorder(self, center_id: str, part: str, urgency: str):
        return self.log(
            DecisionType.REORDER_TRIGGERED,
            {"center_id": center_id, "part": part, "urgency": urgency},
            f"Proactive reorder triggered for {part} at {center_id} ({urgency} priority)",
            entity_id=center_id,
            entity_type="center",
        )


# Singleton
decision_log = DecisionLog()
