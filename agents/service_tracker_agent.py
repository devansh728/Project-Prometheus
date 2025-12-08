"""
Service Tracker Agent - Amazon-style repair lifecycle tracking

This agent manages the lifecycle of a service ticket from INTAKE to PICKED_UP,
with real-time status updates via webhooks and WebSocket notifications.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from enum import Enum

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from db.database import get_database

logger = logging.getLogger(__name__)


class ServiceStage(Enum):
    """Service lifecycle stages."""

    INTAKE = "INTAKE"  # Vehicle dropped off
    DIAGNOSIS = "DIAGNOSIS"  # Inspecting the issue
    WAITING_PARTS = "WAITING_PARTS"  # Parts ordered, awaiting delivery
    REPAIR = "REPAIR"  # Active repair in progress
    QUALITY_CHECK = "QUALITY_CHECK"  # Post-repair verification
    READY = "READY"  # Ready for pickup
    PICKED_UP = "PICKED_UP"  # Customer picked up vehicle


# Valid stage transitions
STAGE_TRANSITIONS = {
    ServiceStage.INTAKE: [ServiceStage.DIAGNOSIS],
    ServiceStage.DIAGNOSIS: [ServiceStage.REPAIR, ServiceStage.WAITING_PARTS],
    ServiceStage.WAITING_PARTS: [ServiceStage.REPAIR],
    ServiceStage.REPAIR: [ServiceStage.QUALITY_CHECK, ServiceStage.WAITING_PARTS],
    ServiceStage.QUALITY_CHECK: [ServiceStage.READY, ServiceStage.REPAIR],
    ServiceStage.READY: [ServiceStage.PICKED_UP],
    ServiceStage.PICKED_UP: [],  # Terminal state
}

# Estimated time for each stage (hours)
STAGE_DURATION = {
    ServiceStage.INTAKE: 0.5,
    ServiceStage.DIAGNOSIS: 1.0,
    ServiceStage.WAITING_PARTS: 24.0,  # Parts may take a day
    ServiceStage.REPAIR: 2.0,
    ServiceStage.QUALITY_CHECK: 0.5,
    ServiceStage.READY: 0.0,
}


class ServiceTrackerAgent:
    """
    Agent for tracking service lifecycle with Amazon-style updates.

    Features:
    - Manages ticket lifecycle (INTAKE → PICKED_UP)
    - Validates stage transitions
    - Calculates estimated completion times
    - Generates human-readable status messages
    - Supports webhook updates from external systems
    """

    def __init__(self):
        self.db = get_database()
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite-preview-06-17",
            temperature=0.7,
        )
        logger.info("ServiceTrackerAgent initialized")

    def create_ticket(
        self,
        appointment_id: str,
        vehicle_id: str,
        technician_id: str = None,
        service_type: str = None,
    ) -> Dict:
        """Create a new service ticket when appointment begins."""
        # Calculate estimated completion
        estimated_hours = sum(STAGE_DURATION.values())
        estimated_completion = (
            datetime.now() + timedelta(hours=estimated_hours)
        ).isoformat()

        ticket = self.db.create_service_ticket(
            appointment_id=appointment_id,
            vehicle_id=vehicle_id,
            technician_id=technician_id,
            estimated_completion=estimated_completion,
        )

        logger.info(f"Created service ticket: {ticket['ticket_id']}")

        return {
            **ticket,
            "message": self._generate_status_message("INTAKE", service_type),
            "estimated_completion": estimated_completion,
            "estimated_hours": estimated_hours,
        }

    def update_status(
        self,
        ticket_id: str,
        new_status: str,
        note: str = "",
        technician_notes: str = None,
    ) -> Dict:
        """
        Update ticket status (webhook handler).

        Validates transition and updates estimated completion.
        """
        # Get current ticket
        ticket = self.db.get_ticket_by_id(ticket_id)
        if not ticket:
            return {"error": "Ticket not found", "success": False}

        current_status = ticket.get("status")

        # Validate transition
        try:
            current_stage = ServiceStage(current_status)
            new_stage = ServiceStage(new_status)
        except ValueError:
            return {"error": f"Invalid status: {new_status}", "success": False}

        valid_next = STAGE_TRANSITIONS.get(current_stage, [])
        if new_stage not in valid_next:
            return {
                "error": f"Invalid transition: {current_status} → {new_status}",
                "valid_transitions": [s.value for s in valid_next],
                "success": False,
            }

        # Calculate new estimated completion
        remaining_stages = self._get_remaining_stages(new_stage)
        remaining_hours = sum(STAGE_DURATION.get(s, 0) for s in remaining_stages)
        estimated_completion = (
            datetime.now() + timedelta(hours=remaining_hours)
        ).isoformat()

        # Update ticket
        result = self.db.update_ticket_status(
            ticket_id=ticket_id,
            new_status=new_status,
            note=note,
            technician_notes=technician_notes,
            estimated_completion=estimated_completion,
        )

        if "error" in result:
            return result

        # Generate user-friendly message
        message = self._generate_status_message(new_status, note)

        logger.info(f"Ticket {ticket_id} updated: {current_status} → {new_status}")

        return {
            **result,
            "message": message,
            "estimated_completion": estimated_completion,
            "remaining_hours": remaining_hours,
            "success": True,
            "notification_type": self._get_notification_type(new_stage),
        }

    def get_ticket_status(self, ticket_id: str) -> Optional[Dict]:
        """Get full ticket status with progress info."""
        ticket = self.db.get_ticket_by_id(ticket_id)
        if not ticket:
            return None

        current_stage = ServiceStage(ticket["status"])

        # Calculate progress percentage
        all_stages = list(ServiceStage)
        current_idx = all_stages.index(current_stage)
        progress = int((current_idx / (len(all_stages) - 1)) * 100)

        # Get remaining stages
        remaining = self._get_remaining_stages(current_stage)

        return {
            **ticket,
            "progress_percent": progress,
            "remaining_stages": [s.value for s in remaining],
            "message": self._generate_status_message(ticket["status"]),
            "is_complete": current_stage == ServiceStage.PICKED_UP,
        }

    def get_vehicle_ticket(self, vehicle_id: str) -> Optional[Dict]:
        """Get active ticket for a vehicle."""
        ticket = self.db.get_ticket_by_vehicle(vehicle_id)
        if ticket:
            return self.get_ticket_status(ticket["id"])
        return None

    def get_ticket_timeline(self, ticket_id: str) -> List[Dict]:
        """Get visual timeline for ticket stages."""
        ticket = self.db.get_ticket_by_id(ticket_id)
        if not ticket:
            return []

        stage_log = ticket.get("stage_log", [])
        current_status = ticket.get("status")

        timeline = []
        all_stages = list(ServiceStage)

        for stage in all_stages:
            # Find if this stage was reached
            stage_entry = next(
                (s for s in stage_log if s.get("stage") == stage.value), None
            )

            if stage_entry:
                status = "completed" if stage.value != current_status else "current"
                timeline.append(
                    {
                        "stage": stage.value,
                        "label": self._get_stage_label(stage),
                        "status": status,
                        "timestamp": stage_entry.get("timestamp"),
                        "note": stage_entry.get("note", ""),
                    }
                )
            else:
                # Check if it's upcoming
                current_idx = all_stages.index(ServiceStage(current_status))
                stage_idx = all_stages.index(stage)

                if stage_idx > current_idx:
                    timeline.append(
                        {
                            "stage": stage.value,
                            "label": self._get_stage_label(stage),
                            "status": "upcoming",
                            "timestamp": None,
                            "note": "",
                        }
                    )

        return timeline

    def _get_remaining_stages(self, current: ServiceStage) -> List[ServiceStage]:
        """Get list of remaining stages after current."""
        all_stages = list(ServiceStage)
        current_idx = all_stages.index(current)
        return all_stages[current_idx + 1 :]

    def _get_stage_label(self, stage: ServiceStage) -> str:
        """Get human-readable label for stage."""
        labels = {
            ServiceStage.INTAKE: "🚗 Vehicle Received",
            ServiceStage.DIAGNOSIS: "🔍 Diagnosing Issue",
            ServiceStage.WAITING_PARTS: "📦 Waiting for Parts",
            ServiceStage.REPAIR: "🔧 Repair in Progress",
            ServiceStage.QUALITY_CHECK: "✅ Quality Check",
            ServiceStage.READY: "🎉 Ready for Pickup",
            ServiceStage.PICKED_UP: "👋 Picked Up",
        }
        return labels.get(stage, stage.value)

    def _generate_status_message(self, status: str, context: str = None) -> str:
        """Generate user-friendly status message."""
        messages = {
            "INTAKE": "Your vehicle has been received at our service center. We'll begin diagnosis shortly.",
            "DIAGNOSIS": "Our technician is currently inspecting your vehicle to identify the issue.",
            "WAITING_PARTS": "We've ordered the necessary parts for your repair. Expected arrival: 24-48 hours.",
            "REPAIR": "Repair work has started! Our technician is working on your vehicle.",
            "QUALITY_CHECK": "Repair complete! We're now performing a final quality check.",
            "READY": "Great news! Your vehicle is ready for pickup. See you soon!",
            "PICKED_UP": "Thank you for choosing our service! We hope to see you again.",
        }

        base_message = messages.get(status, f"Status updated to: {status}")

        if context:
            base_message += f" Note: {context}"

        return base_message

    def _get_notification_type(self, stage: ServiceStage) -> str:
        """Determine notification urgency type."""
        if stage == ServiceStage.READY:
            return "action_required"
        elif stage in [ServiceStage.WAITING_PARTS, ServiceStage.REPAIR]:
            return "info"
        elif stage == ServiceStage.PICKED_UP:
            return "complete"
        else:
            return "update"

    async def generate_personalized_update(
        self,
        ticket_id: str,
        vehicle_id: str,
    ) -> str:
        """Use LLM to generate personalized status update."""
        ticket = self.db.get_ticket_by_id(ticket_id)
        if not ticket:
            return "Unable to find your service ticket."

        # Get user profile for personalization
        profile = self.db.get_user_profile(vehicle_id)

        prompt = f"""Generate a friendly, personalized service status update.

Ticket Status: {ticket['status']}
Stage: {self._get_stage_label(ServiceStage(ticket['status']))}
Estimated Completion: {ticket.get('estimated_completion', 'Unknown')}
Technician Notes: {ticket.get('technician_notes', 'None')}

User Profile:
- Driving Style: {profile.get('driving_style', 'normal') if profile else 'normal'}
- Past Services: {profile.get('total_services', 0) if profile else 0}
- Satisfaction: {profile.get('avg_satisfaction', 'N/A') if profile else 'N/A'}

Write a 2-3 sentence update that:
1. Informs about current status
2. Sets clear expectations
3. Shows appreciation (especially if returning customer)

Keep it warm and professional."""

        try:
            response = await self.llm.ainvoke(
                [
                    SystemMessage(
                        content="You are a friendly service advisor for an EV service center."
                    ),
                    HumanMessage(content=prompt),
                ]
            )
            return response.content
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return self._generate_status_message(ticket["status"])


# Singleton instance
_tracker_agent: Optional[ServiceTrackerAgent] = None


def get_service_tracker() -> ServiceTrackerAgent:
    """Get or create ServiceTrackerAgent singleton."""
    global _tracker_agent
    if _tracker_agent is None:
        _tracker_agent = ServiceTrackerAgent()
    return _tracker_agent
