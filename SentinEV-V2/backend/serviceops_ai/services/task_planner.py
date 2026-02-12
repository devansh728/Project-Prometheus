"""
Task Decomposition & Technician Planning for ServiceOps AI
Breaks down jobs into subtasks and assigns technicians with skill/fatigue awareness.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid

from serviceops_ai.services.workforce import workforce_manager, MechanicInfo


class TaskStatus(str, Enum):
    """Task execution status."""

    TODO = "TODO"
    IN_PROGRESS = "IN_PROGRESS"
    DONE = "DONE"
    BLOCKED = "BLOCKED"


@dataclass
class ServiceTask:
    """A subtask within a service job."""

    task_id: str
    job_id: str
    name: str
    description: str
    required_skill: str
    estimated_minutes: int
    dependency_order: int  # 0 = can start first, higher = must wait

    # Assignment
    assigned_mechanic_id: Optional[str] = None
    assigned_mechanic_name: Optional[str] = None

    # Status tracking
    status: TaskStatus = TaskStatus.TODO
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # For Gantt visualization
    scheduled_start: Optional[datetime] = None
    scheduled_end: Optional[datetime] = None


class TaskPlanner:
    """
    Decomposes jobs into subtasks and assigns technicians.
    """

    # Task templates by failure type
    TASK_TEMPLATES = {
        "brake_degradation": [
            {
                "name": "Initial Inspection",
                "skill": "general",
                "duration": 15,
                "order": 0,
            },
            {
                "name": "Brake Pad Measurement",
                "skill": "brake",
                "duration": 20,
                "order": 1,
            },
            {
                "name": "Brake Pad Replacement",
                "skill": "brake",
                "duration": 45,
                "order": 2,
            },
            {
                "name": "Brake Fluid Check & Top-up",
                "skill": "brake",
                "duration": 15,
                "order": 3,
            },
            {
                "name": "Calibration & Testing",
                "skill": "brake",
                "duration": 20,
                "order": 4,
            },
            {"name": "Quality Check", "skill": "general", "duration": 15, "order": 5},
        ],
        "brake_fade": [
            {
                "name": "Initial Inspection",
                "skill": "general",
                "duration": 15,
                "order": 0,
            },
            {
                "name": "Brake System Diagnosis",
                "skill": "brake",
                "duration": 30,
                "order": 1,
            },
            {
                "name": "Brake Pad Replacement",
                "skill": "brake",
                "duration": 45,
                "order": 2,
            },
            {
                "name": "Brake Rotor Inspection",
                "skill": "brake",
                "duration": 20,
                "order": 3,
            },
            {"name": "Brake Fluid Flush", "skill": "brake", "duration": 25, "order": 4},
            {"name": "Test Drive", "skill": "general", "duration": 15, "order": 5},
            {"name": "Quality Check", "skill": "general", "duration": 15, "order": 6},
        ],
        "battery_degradation": [
            {
                "name": "Initial Inspection",
                "skill": "general",
                "duration": 15,
                "order": 0,
            },
            {
                "name": "Battery Health Diagnostic",
                "skill": "ev_battery",
                "duration": 30,
                "order": 1,
            },
            {
                "name": "Cell Module Identification",
                "skill": "ev_battery",
                "duration": 20,
                "order": 2,
            },
            {
                "name": "Battery Module Replacement",
                "skill": "ev_battery",
                "duration": 90,
                "order": 3,
            },
            {
                "name": "BMS Recalibration",
                "skill": "ev_battery",
                "duration": 30,
                "order": 4,
            },
            {
                "name": "Charge Cycle Test",
                "skill": "ev_battery",
                "duration": 45,
                "order": 5,
            },
            {"name": "Quality Check", "skill": "general", "duration": 15, "order": 6},
        ],
        "cooling_degradation": [
            {
                "name": "Initial Inspection",
                "skill": "general",
                "duration": 15,
                "order": 0,
            },
            {
                "name": "Coolant Level Check",
                "skill": "cooling",
                "duration": 10,
                "order": 1,
            },
            {
                "name": "Thermostat Testing",
                "skill": "cooling",
                "duration": 20,
                "order": 2,
            },
            {"name": "Coolant Flush", "skill": "cooling", "duration": 30, "order": 3},
            {
                "name": "Thermostat Replacement",
                "skill": "cooling",
                "duration": 35,
                "order": 4,
            },
            {
                "name": "System Pressure Test",
                "skill": "cooling",
                "duration": 15,
                "order": 5,
            },
            {"name": "Quality Check", "skill": "general", "duration": 15, "order": 6},
        ],
        "general_maintenance": [
            {
                "name": "Initial Inspection",
                "skill": "general",
                "duration": 20,
                "order": 0,
            },
            {
                "name": "Fluid Levels Check",
                "skill": "general",
                "duration": 15,
                "order": 1,
            },
            {"name": "Tire Inspection", "skill": "general", "duration": 15, "order": 2},
            {"name": "Service Report", "skill": "general", "duration": 10, "order": 3},
        ],
    }

    def __init__(self):
        self._tasks: Dict[str, ServiceTask] = {}  # task_id -> task
        self._job_tasks: Dict[str, List[str]] = {}  # job_id -> [task_ids]
        self._mechanic_hours: Dict[str, float] = {}  # mechanic_id -> hours worked today

    def decompose(
        self,
        job_id: str,
        failure_type: str,
        severity: str = "medium",
    ) -> List[ServiceTask]:
        """
        Decompose a job into subtasks based on failure type.
        """
        template = self.TASK_TEMPLATES.get(
            failure_type, self.TASK_TEMPLATES["general_maintenance"]
        )

        tasks = []
        current_time = datetime.utcnow() + timedelta(hours=1)  # Start in 1 hour

        for t in template:
            task_id = f"TASK-{uuid.uuid4().hex[:6].upper()}"

            scheduled_end = current_time + timedelta(minutes=t["duration"])

            task = ServiceTask(
                task_id=task_id,
                job_id=job_id,
                name=t["name"],
                description=f"{t['name']} for {failure_type.replace('_', ' ')}",
                required_skill=t["skill"],
                estimated_minutes=t["duration"],
                dependency_order=t["order"],
                scheduled_start=current_time,
                scheduled_end=scheduled_end,
            )

            tasks.append(task)
            self._tasks[task_id] = task

            current_time = scheduled_end + timedelta(minutes=5)  # 5 min buffer

        self._job_tasks[job_id] = [t.task_id for t in tasks]

        return tasks

    def assign_technicians(
        self,
        tasks: List[ServiceTask],
        center_id: str,
    ) -> List[ServiceTask]:
        """
        Assign technicians to tasks based on skill, availability, and fatigue.
        """
        for task in tasks:
            # Find available mechanics with matching skill
            candidates = workforce_manager.find_available_mechanics(
                center_id, failure_type=None, limit=10  # We'll filter by skill
            )

            # Score candidates
            best_score = -1
            best_mechanic = None

            for mech in candidates:
                if not mech.is_available:
                    continue

                # Check skill match
                skill_match = 1.0 if task.required_skill in mech.certifications else 0.3
                if "general" in mech.certifications:
                    skill_match = max(skill_match, 0.6)

                # Fatigue score (hours worked today)
                hours_worked = self._mechanic_hours.get(mech.id, 0)
                fatigue_score = max(0, 1 - (hours_worked / 8))  # 8 hour shift

                # Fairness (fewer jobs = higher score)
                fairness_score = 1 - (mech.current_jobs / max(mech.max_jobs_per_day, 1))

                # Combined score
                score = (
                    (skill_match * 0.5) + (fatigue_score * 0.3) + (fairness_score * 0.2)
                )

                if score > best_score:
                    best_score = score
                    best_mechanic = mech

            if best_mechanic:
                task.assigned_mechanic_id = best_mechanic.id
                task.assigned_mechanic_name = best_mechanic.name

                # Update hours tracking
                hours = self._mechanic_hours.get(best_mechanic.id, 0)
                self._mechanic_hours[best_mechanic.id] = hours + (
                    task.estimated_minutes / 60
                )

        return tasks

    def update_task_status(
        self,
        task_id: str,
        status: TaskStatus,
    ) -> Optional[ServiceTask]:
        """Update task status."""
        task = self._tasks.get(task_id)
        if not task:
            return None

        old_status = task.status
        task.status = status

        if status == TaskStatus.IN_PROGRESS and not task.started_at:
            task.started_at = datetime.utcnow()
        elif status == TaskStatus.DONE and not task.completed_at:
            task.completed_at = datetime.utcnow()

        return task

    def get_job_tasks(self, job_id: str) -> List[Dict[str, Any]]:
        """Get all tasks for a job."""
        task_ids = self._job_tasks.get(job_id, [])
        tasks = [self._tasks.get(tid) for tid in task_ids if tid in self._tasks]

        return [
            {
                "task_id": t.task_id,
                "job_id": t.job_id,
                "name": t.name,
                "description": t.description,
                "required_skill": t.required_skill,
                "estimated_minutes": t.estimated_minutes,
                "dependency_order": t.dependency_order,
                "assigned_mechanic_id": t.assigned_mechanic_id,
                "assigned_mechanic_name": t.assigned_mechanic_name,
                "status": t.status.value,
                "started_at": t.started_at.isoformat() if t.started_at else None,
                "completed_at": t.completed_at.isoformat() if t.completed_at else None,
                "scheduled_start": (
                    t.scheduled_start.isoformat() if t.scheduled_start else None
                ),
                "scheduled_end": (
                    t.scheduled_end.isoformat() if t.scheduled_end else None
                ),
            }
            for t in tasks
        ]

    def get_gantt_data(self, center_id: str) -> List[Dict[str, Any]]:
        """
        Get Gantt-chart-ready data for a service center.
        Groups tasks by mechanic with time slots.
        """
        # Get all active tasks for this center's mechanics
        gantt_items = []

        for task in self._tasks.values():
            if task.assigned_mechanic_id:
                # Verify mechanic is at this center (simplified check)
                gantt_items.append(
                    {
                        "task_id": task.task_id,
                        "task_name": task.name,
                        "mechanic_id": task.assigned_mechanic_id,
                        "mechanic_name": task.assigned_mechanic_name or "Unknown",
                        "start": (
                            task.scheduled_start.isoformat()
                            if task.scheduled_start
                            else None
                        ),
                        "end": (
                            task.scheduled_end.isoformat()
                            if task.scheduled_end
                            else None
                        ),
                        "duration_minutes": task.estimated_minutes,
                        "status": task.status.value,
                        "skill": task.required_skill,
                    }
                )

        # Sort by mechanic and start time
        gantt_items.sort(key=lambda x: (x["mechanic_id"], x["start"] or ""))

        return gantt_items

    def replan(
        self,
        job_id: str,
        diagnosis_delta: Dict[str, Any],
    ) -> List[ServiceTask]:
        """
        Regenerate tasks if diagnosis differs from prediction.
        """
        # Clear existing tasks
        old_task_ids = self._job_tasks.get(job_id, [])
        for tid in old_task_ids:
            if tid in self._tasks:
                del self._tasks[tid]

        # Determine new failure type from diagnosis
        new_failure_type = diagnosis_delta.get(
            "actual_failure_type", "general_maintenance"
        )
        severity = diagnosis_delta.get("severity", "medium")

        # Decompose with new type
        new_tasks = self.decompose(job_id, new_failure_type, severity)

        return new_tasks

    def get_task(self, task_id: str) -> Optional[ServiceTask]:
        """Get a specific task."""
        return self._tasks.get(task_id)

    def reset(self):
        """Reset all tasks (for demo)."""
        self._tasks.clear()
        self._job_tasks.clear()
        self._mechanic_hours.clear()


# Singleton
task_planner = TaskPlanner()
