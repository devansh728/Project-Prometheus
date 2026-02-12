"""
Service Lifecycle State Machine for ServiceOps AI
Manages the complete lifecycle of a service request.
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum


class ServiceState(str, Enum):
    """Service lifecycle states."""

    REQUESTED = "REQUESTED"
    BOOKED = "BOOKED"
    CONFIRMED = "CONFIRMED"
    CHECK_IN = "CHECK_IN"
    DIAGNOSIS = "DIAGNOSIS"
    PARTS_ALLOCATED = "PARTS_ALLOCATED"
    REPAIR_IN_PROGRESS = "REPAIR_IN_PROGRESS"
    QUALITY_CHECK = "QUALITY_CHECK"
    READY = "READY"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"


# Valid state transitions
STATE_TRANSITIONS = {
    ServiceState.REQUESTED: [ServiceState.BOOKED, ServiceState.CANCELLED],
    ServiceState.BOOKED: [ServiceState.CONFIRMED, ServiceState.CANCELLED],
    ServiceState.CONFIRMED: [ServiceState.CHECK_IN, ServiceState.CANCELLED],
    ServiceState.CHECK_IN: [ServiceState.DIAGNOSIS],
    ServiceState.DIAGNOSIS: [
        ServiceState.PARTS_ALLOCATED,
        ServiceState.READY,
    ],  # READY if no parts needed
    ServiceState.PARTS_ALLOCATED: [ServiceState.REPAIR_IN_PROGRESS],
    ServiceState.REPAIR_IN_PROGRESS: [ServiceState.QUALITY_CHECK],
    ServiceState.QUALITY_CHECK: [
        ServiceState.READY,
        ServiceState.REPAIR_IN_PROGRESS,
    ],  # Back if failed
    ServiceState.READY: [ServiceState.COMPLETED],
    ServiceState.COMPLETED: [],
    ServiceState.CANCELLED: [],
}

# Estimated durations for each state (in minutes)
STATE_DURATIONS = {
    ServiceState.CHECK_IN: 15,
    ServiceState.DIAGNOSIS: 30,
    ServiceState.PARTS_ALLOCATED: 15,
    ServiceState.REPAIR_IN_PROGRESS: 90,
    ServiceState.QUALITY_CHECK: 20,
}


@dataclass
class ServiceJob:
    """Represents a service job in the lifecycle."""

    job_id: str
    vehicle_id: str
    customer_id: str
    service_center_id: str
    failure_type: str
    severity: str

    state: ServiceState = ServiceState.REQUESTED
    mechanic_id: Optional[str] = None
    scheduled_at: Optional[datetime] = None

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    booked_at: Optional[datetime] = None
    checked_in_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # State history
    state_history: List[Dict] = field(default_factory=list)

    def __post_init__(self):
        self._log_transition(None, self.state)

    def _log_transition(
        self, from_state: Optional[ServiceState], to_state: ServiceState
    ):
        """Log state transition."""
        self.state_history.append(
            {
                "from": from_state.value if from_state else None,
                "to": to_state.value,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )


class ServiceLifecycle:
    """
    Manages service job state transitions and lifecycle events.
    """

    def __init__(self):
        self.jobs: Dict[str, ServiceJob] = {}
        self._job_counter = 0

    def create_job(
        self,
        vehicle_id: str,
        customer_id: str,
        service_center_id: str,
        failure_type: str,
        severity: str,
    ) -> ServiceJob:
        """Create a new service job."""
        self._job_counter += 1
        job_id = f"JOB-{datetime.utcnow().strftime('%Y%m%d')}-{self._job_counter:04d}"

        job = ServiceJob(
            job_id=job_id,
            vehicle_id=vehicle_id,
            customer_id=customer_id,
            service_center_id=service_center_id,
            failure_type=failure_type,
            severity=severity,
        )

        self.jobs[job_id] = job
        return job

    def transition(
        self, job_id: str, to_state: ServiceState, metadata: Optional[Dict] = None
    ) -> Tuple[bool, str]:
        """
        Transition a job to a new state.
        Returns (success, message).
        """
        job = self.jobs.get(job_id)
        if not job:
            return False, f"Job {job_id} not found"

        # Check if transition is valid
        valid_transitions = STATE_TRANSITIONS.get(job.state, [])
        if to_state not in valid_transitions:
            return (
                False,
                f"Invalid transition from {job.state.value} to {to_state.value}",
            )

        # Perform transition
        from_state = job.state
        job.state = to_state
        job._log_transition(from_state, to_state)

        # Handle specific transitions
        if to_state == ServiceState.BOOKED:
            job.booked_at = datetime.utcnow()
            if metadata and "scheduled_at" in metadata:
                job.scheduled_at = metadata["scheduled_at"]
            if metadata and "mechanic_id" in metadata:
                job.mechanic_id = metadata["mechanic_id"]

        elif to_state == ServiceState.CHECK_IN:
            job.checked_in_at = datetime.utcnow()

        elif to_state == ServiceState.COMPLETED:
            job.completed_at = datetime.utcnow()

        return True, f"Transitioned to {to_state.value}"

    def get_job(self, job_id: str) -> Optional[ServiceJob]:
        """Get a job by ID."""
        return self.jobs.get(job_id)

    def get_jobs_by_center(
        self, service_center_id: str, state_filter: Optional[List[ServiceState]] = None
    ) -> List[ServiceJob]:
        """Get all jobs for a service center, optionally filtered by state."""
        jobs = [
            j for j in self.jobs.values() if j.service_center_id == service_center_id
        ]

        if state_filter:
            jobs = [j for j in jobs if j.state in state_filter]

        return sorted(jobs, key=lambda x: x.created_at, reverse=True)

    def get_active_jobs(self, service_center_id: str) -> List[ServiceJob]:
        """Get all active (non-completed, non-cancelled) jobs."""
        active_states = [
            ServiceState.BOOKED,
            ServiceState.CONFIRMED,
            ServiceState.CHECK_IN,
            ServiceState.DIAGNOSIS,
            ServiceState.PARTS_ALLOCATED,
            ServiceState.REPAIR_IN_PROGRESS,
            ServiceState.QUALITY_CHECK,
            ServiceState.READY,
        ]
        return self.get_jobs_by_center(service_center_id, active_states)

    def get_estimated_completion(self, job: ServiceJob) -> Optional[datetime]:
        """Estimate completion time based on current state."""
        if job.state in [ServiceState.COMPLETED, ServiceState.CANCELLED]:
            return job.completed_at

        remaining_minutes = 0
        started_counting = False

        for state in ServiceState:
            if state == job.state:
                started_counting = True

            if started_counting and state in STATE_DURATIONS:
                remaining_minutes += STATE_DURATIONS[state]

            if state == ServiceState.READY:
                break

        return datetime.utcnow() + timedelta(minutes=remaining_minutes)


# Singleton
service_lifecycle = ServiceLifecycle()
