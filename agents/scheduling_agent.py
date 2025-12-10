"""
SentinEV - Scheduling Agent
Handles appointment booking, rescheduling, and service center coordination.
Uses collaborative conversation for negotiating appointment times.
"""

import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json

# LLM imports
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

from db.database import get_database, AppointmentStatus, ServiceUrgency

# Phase 11: Scheduling utilities
from agents.scheduling_utils import (
    PriorityAppointment,
    TowingRequest,
    DemandForecast,
    FleetScheduleResult,
    TowingStatus,
    SchedulingPriorityQueue,
    calculate_urgency_score,
    select_optimal_center,
    dispatch_towing,
    forecast_demand,
    simulate_geo_distance,
    SIMULATED_CENTER_LOCATIONS,
)


class SchedulingAgent:
    """
    Scheduling Agent for appointment management.

    Features:
    - Checks service center availability
    - Proposes negotiable time slots
    - Books and manages appointments
    - Collaborative conversation for scheduling
    """

    def __init__(self):
        self.db = get_database()
        self.llm = None
        self._initialize_llm()

        # Phase 11: Priority queue for scheduling requests
        self.priority_queue = SchedulingPriorityQueue()

        # Phase 11: Active towing requests
        self.active_towing: Dict[str, TowingRequest] = {}

        # Phase 11: Simulated vehicle locations (would come from telematics)
        self.vehicle_locations: Dict[str, Dict] = {
            "EV_001": {"lat": 40.7300, "lon": -73.9950},
            "EV_002": {"lat": 40.7580, "lon": -73.9855},
            "EV_003": {"lat": 40.6892, "lon": -74.0445},
        }

        # Conversation templates
        self.propose_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a friendly and helpful scheduling assistant for SentinEV vehicle maintenance.
Your job is to help customers book service appointments. Be conversational, helpful, and proactive.

Customer's vehicle: {vehicle_id}
Issue diagnosed: {diagnosis_summary}
Component: {component}
Urgency: {urgency}
Estimated cost: {estimated_cost}

Available slots:
{slots_info}

Recommend the best slot based on urgency, but offer alternatives. Be persuasive but not pushy.
If urgency is critical or high, emphasize the importance of prompt service.""",
                ),
                ("human", "Please help me schedule a service appointment."),
            ]
        )

        self.negotiate_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a scheduling assistant helping reschedule an appointment.
Be understanding and offer alternatives. The customer wants a different time.

Original appointment: {original_slot}
Customer's preference: {preference}

Available alternatives:
{alternatives}

Suggest the best alternative that matches their preference. Be helpful and accommodating.""",
                ),
                ("human", "{customer_message}"),
            ]
        )

        self.confirmation_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """Generate a friendly appointment confirmation message.
Include all details and remind them what to bring (vehicle, any warning lights info).
Be warm and professional.""",
                ),
                (
                    "human",
                    """Confirm this appointment:
Service Center: {center_name}
Date: {date}
Time: {time}
Service: {component} - {diagnosis_summary}
Estimated Cost: {estimated_cost}""",
                ),
            ]
        )

    def _initialize_llm(self):
        """Initialize the LLM for conversation generation."""
        if not LLM_AVAILABLE:
            return

        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if api_key:
            try:
                self.llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash-lite",
                    google_api_key=api_key,
                    temperature=0.7,
                )
            except Exception as e:
                print(f"Warning: Could not initialize scheduling LLM: {e}")

    def check_availability(
        self,
        component: str,
        urgency: str = "medium",
        preferred_date: str = None,
        limit: int = 5,
    ) -> Dict[str, Any]:
        """
        Check service center availability for a component.

        Args:
            component: Component needing service (brakes, battery, motor, etc.)
            urgency: Service urgency level
            preferred_date: Optional preferred date (YYYY-MM-DD)
            limit: Max number of slots to return

        Returns:
            Dict with available centers and slots
        """
        # Find centers that handle this component
        centers = self.db.get_center_by_specialty(component)

        if not centers:
            return {
                "available": False,
                "message": f"No service centers found for {component}",
                "centers": [],
                "slots": [],
            }

        # Get available slots
        all_slots = []
        for center in centers:
            slots = self.db.get_available_slots(
                center_id=center["id"],
                date=preferred_date,
                component=component,
                limit=limit,
            )
            for slot in slots:
                slot["center_name"] = center["name"]
                slot["center_location"] = center["location"]
                slot["center_rating"] = center["rating"]
            all_slots.extend(slots)

        # Sort by date and rating
        all_slots.sort(key=lambda s: (s["date"], -s["center_rating"]))

        # For urgent cases, prioritize earliest slots
        if urgency in ["high", "critical"]:
            all_slots = all_slots[:limit]

        return {
            "available": len(all_slots) > 0,
            "component": component,
            "urgency": urgency,
            "centers": centers,
            "slots": all_slots[:limit],
            "total_available": len(all_slots),
        }

    def propose_slots(
        self,
        vehicle_id: str,
        component: str,
        diagnosis_summary: str,
        estimated_cost: str,
        urgency: str = "medium",
    ) -> Dict[str, Any]:
        """
        Propose appointment slots with collaborative conversation.

        Returns 3 slot options with a conversational message.
        """
        availability = self.check_availability(component, urgency, limit=5)

        if not availability["available"]:
            return {
                "success": False,
                "message": "I apologize, but I couldn't find any available slots for this service right now. Would you like me to check again tomorrow?",
                "slots": [],
            }

        # Select top 3 slots
        top_slots = availability["slots"][:3]

        # Format slots for LLM
        slots_info = "\n".join(
            [
                f"{i+1}. {s['center_name']} - {s['date']} at {s['start_time']} (Rating: {s['center_rating']}⭐)"
                for i, s in enumerate(top_slots)
            ]
        )

        # Generate conversational proposal
        if self.llm:
            try:
                chain = self.propose_template | self.llm | StrOutputParser()
                message = chain.invoke(
                    {
                        "vehicle_id": vehicle_id,
                        "diagnosis_summary": diagnosis_summary,
                        "component": component,
                        "urgency": urgency,
                        "estimated_cost": estimated_cost,
                        "slots_info": slots_info,
                    }
                )
            except Exception as e:
                message = self._fallback_proposal_message(top_slots, urgency, component)
        else:
            message = self._fallback_proposal_message(top_slots, urgency, component)

        return {
            "success": True,
            "message": message,
            "slots": [
                {
                    "option": i + 1,
                    "slot_id": s["id"],
                    "center_id": s["center_id"],
                    "center_name": s["center_name"],
                    "location": s["center_location"],
                    "date": s["date"],
                    "time": s["start_time"],
                    "rating": s["center_rating"],
                }
                for i, s in enumerate(top_slots)
            ],
            "diagnosis_summary": diagnosis_summary,
            "estimated_cost": estimated_cost,
            "component": component,
            "urgency": urgency,
        }

    def _fallback_proposal_message(
        self, slots: List[Dict], urgency: str, component: str
    ) -> str:
        """Generate a proposal message without LLM."""
        urgency_text = {
            "critical": "⚠️ Based on our diagnosis, this requires immediate attention.",
            "high": "This is a priority service that should be addressed soon.",
            "medium": "We recommend scheduling this service at your earliest convenience.",
            "low": "This is routine maintenance that can be scheduled when convenient.",
        }

        msg = f"""I've found some available appointment slots for your {component} service.

{urgency_text.get(urgency, urgency_text['medium'])}

Here are your options:
"""
        for i, s in enumerate(slots):
            msg += f"\n**Option {i+1}:** {s['center_name']}\n"
            msg += f"   📅 {s['date']} at {s['start_time']}\n"
            msg += f"   📍 {s.get('center_location', 'Location TBD')}\n"
            msg += f"   ⭐ Rating: {s.get('center_rating', 4.5)}\n"

        msg += "\nWhich option works best for you? Or let me know if you'd prefer a different day/time."
        return msg

    def book_appointment(
        self,
        vehicle_id: str,
        slot_id: str,
        center_id: str,
        component: str,
        diagnosis_summary: str,
        estimated_cost: str,
        urgency: str = "medium",
        notes: str = "",
    ) -> Dict[str, Any]:
        """
        Book an appointment and generate confirmation.
        """
        # Create the appointment
        appointment = self.db.create_appointment(
            vehicle_id=vehicle_id,
            center_id=center_id,
            slot_id=slot_id,
            component=component,
            diagnosis_summary=diagnosis_summary,
            estimated_cost=estimated_cost,
            urgency=urgency,
            notes=notes,
        )

        if "error" in appointment:
            return {
                "success": False,
                "message": f"Sorry, I couldn't book that slot: {appointment['error']}",
                "appointment": None,
            }

        # Log demand for forecasting
        self.db.log_demand(center_id, component)

        # Get center info for confirmation
        centers = self.db.get_service_centers()
        center = next((c for c in centers if c["id"] == center_id), None)
        center_name = center["name"] if center else "Service Center"

        # Generate confirmation message
        if self.llm:
            try:
                chain = self.confirmation_template | self.llm | StrOutputParser()
                message = chain.invoke(
                    {
                        "center_name": center_name,
                        "date": appointment["scheduled_date"],
                        "time": appointment["scheduled_time"],
                        "component": component,
                        "diagnosis_summary": diagnosis_summary,
                        "estimated_cost": estimated_cost,
                    }
                )
            except Exception:
                message = self._fallback_confirmation(
                    appointment, center_name, component, diagnosis_summary
                )
        else:
            message = self._fallback_confirmation(
                appointment, center_name, component, diagnosis_summary
            )

        return {
            "success": True,
            "message": message,
            "appointment": appointment,
            "notification": {
                "type": "appointment_confirmed",
                "title": "✅ Appointment Booked!",
                "body": f"{center_name} on {appointment['scheduled_date']} at {appointment['scheduled_time']}",
            },
        }

    def _fallback_confirmation(
        self, appt: Dict, center_name: str, component: str, diagnosis: str
    ) -> str:
        """Generate confirmation without LLM."""
        return f"""✅ **Appointment Confirmed!**

Your service appointment has been booked:

📍 **Where:** {center_name}
📅 **When:** {appt['scheduled_date']} at {appt['scheduled_time']}
🔧 **Service:** {component.title()} - {diagnosis}

**What to bring:**
- Your vehicle
- Any relevant warning light information
- This confirmation (ID: {appt['id']})

We'll send you a reminder before your appointment. See you soon! 🚗"""

    def get_appointments(self, vehicle_id: str, status: str = None) -> Dict[str, Any]:
        """Get all appointments for a vehicle."""
        appointments = self.db.get_appointments(vehicle_id, status)

        if not appointments:
            return {
                "success": True,
                "message": "You don't have any scheduled appointments.",
                "appointments": [],
            }

        return {
            "success": True,
            "message": f"You have {len(appointments)} appointment(s).",
            "appointments": appointments,
        }

    def reschedule_appointment(
        self, appointment_id: str, new_slot_id: str, reason: str = ""
    ) -> Dict[str, Any]:
        """Reschedule an existing appointment."""
        # Get current appointment
        conn = self.db._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM appointments WHERE id = ?", (appointment_id,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            return {
                "success": False,
                "message": "I couldn't find that appointment. Please check the ID.",
            }

        old_slot_id = row["slot_id"]

        # Release old slot
        self.db.release_slot(old_slot_id)

        # Book new slot
        if not self.db.book_slot(new_slot_id):
            # Revert if new slot unavailable
            self.db.book_slot(old_slot_id)
            return {
                "success": False,
                "message": "That time slot is no longer available. Would you like me to suggest alternatives?",
            }

        # Update appointment
        conn = self.db._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT date, start_time FROM time_slots WHERE id = ?", (new_slot_id,)
        )
        new_slot = cursor.fetchone()

        cursor.execute(
            """
            UPDATE appointments 
            SET slot_id = ?, scheduled_date = ?, scheduled_time = ?, notes = notes || ?
            WHERE id = ?
        """,
            (
                new_slot_id,
                new_slot["date"],
                new_slot["start_time"],
                f" Rescheduled: {reason}",
                appointment_id,
            ),
        )

        conn.commit()
        conn.close()

        return {
            "success": True,
            "message": f"""✅ **Appointment Rescheduled!**

Your new appointment is on **{new_slot['date']}** at **{new_slot['start_time']}**.

We look forward to seeing you! 🚗""",
            "new_date": new_slot["date"],
            "new_time": new_slot["start_time"],
        }

    def cancel_appointment(
        self, appointment_id: str, reason: str = ""
    ) -> Dict[str, Any]:
        """Cancel an appointment."""
        success = self.db.cancel_appointment(appointment_id)

        if success:
            return {
                "success": True,
                "message": """Your appointment has been cancelled.

We're sorry to see you go! If you'd like to reschedule for a later date, just let me know.

Is there anything else I can help you with?""",
                "notification": {
                    "type": "appointment_cancelled",
                    "title": "Appointment Cancelled",
                    "body": f"Appointment {appointment_id} has been cancelled.",
                },
            }
        else:
            return {
                "success": False,
                "message": "I couldn't find that appointment. Please check the ID and try again.",
            }

    def negotiate_alternative(
        self, vehicle_id: str, preference: str, component: str
    ) -> Dict[str, Any]:
        """
        Handle customer preference for different time.

        Args:
            vehicle_id: Vehicle ID
            preference: Customer's stated preference (e.g., "morning", "next week", "weekend")
            component: Component being serviced
        """
        # Parse preference to find matching slots
        availability = self.check_availability(component, limit=10)

        if not availability["available"]:
            return {
                "success": False,
                "message": "I apologize, but there are no available slots at this time. Can I put you on a waitlist?",
                "slots": [],
            }

        # Filter slots based on preference keywords
        slots = availability["slots"]
        filtered_slots = []

        preference_lower = preference.lower()

        for slot in slots:
            hour = int(slot["start_time"].split(":")[0])

            # Morning preference
            if "morning" in preference_lower and hour < 12:
                filtered_slots.append(slot)
            # Afternoon preference
            elif "afternoon" in preference_lower and 12 <= hour < 17:
                filtered_slots.append(slot)
            # Weekend (would need actual day of week check)
            elif "later" in preference_lower or "next week" in preference_lower:
                # Get later slots
                filtered_slots = slots[-3:]
                break
            elif "soon" in preference_lower or "asap" in preference_lower:
                filtered_slots = slots[:3]
                break

        if not filtered_slots:
            filtered_slots = slots[:3]  # Default to first available

        message = f"""I understand you'd prefer {preference}. Here are some options that might work better:

"""
        for i, s in enumerate(filtered_slots[:3]):
            message += f"**Option {i+1}:** {s['center_name']} - {s['date']} at {s['start_time']}\n"

        message += "\nDo any of these work for you?"

        return {"success": True, "message": message, "slots": filtered_slots[:3]}

    def get_service_progress(self, appointment_id: str) -> Dict[str, Any]:
        """Get the progress status of a service appointment."""
        appointments = self.db.get_appointments()
        appt = next((a for a in appointments if a["id"] == appointment_id), None)

        if not appt:
            return {"success": False, "message": "I couldn't find that appointment."}

        status = appt["status"]
        status_messages = {
            "scheduled": "📅 Your appointment is scheduled and confirmed. We're looking forward to seeing you!",
            "confirmed": "✅ The service center has confirmed your appointment. See you soon!",
            "in_progress": "🔧 Your vehicle is currently being serviced. We'll notify you when it's ready!",
            "completed": "✅ Great news! Your service has been completed. Your vehicle is ready for pickup!",
            "cancelled": "❌ This appointment was cancelled.",
        }

        return {
            "success": True,
            "appointment_id": appointment_id,
            "status": status,
            "message": status_messages.get(status, f"Status: {status}"),
            "details": {
                "center": appt["center_name"],
                "date": appt["scheduled_date"],
                "time": appt["scheduled_time"],
                "component": appt["component"],
            },
        }

    def update_service_status(
        self, appointment_id: str, new_status: str
    ) -> Dict[str, Any]:
        """Update the status of a service appointment."""
        valid_statuses = [
            "scheduled",
            "confirmed",
            "in_progress",
            "completed",
            "cancelled",
        ]

        if new_status not in valid_statuses:
            return {
                "success": False,
                "message": f"Invalid status. Must be one of: {valid_statuses}",
            }

        success = self.db.update_appointment_status(appointment_id, new_status)

        if success:
            notifications = {
                "confirmed": (
                    "📩 Appointment Confirmed",
                    "The service center has confirmed your appointment.",
                ),
                "in_progress": (
                    "🔧 Service Started",
                    "Your vehicle is now being serviced.",
                ),
                "completed": (
                    "✅ Service Complete",
                    "Your vehicle is ready for pickup!",
                ),
            }

            return {
                "success": True,
                "status": new_status,
                "message": f"Appointment status updated to: {new_status}",
                "notification": (
                    {
                        "type": f"service_{new_status}",
                        "title": notifications.get(
                            new_status, ("Status Update", f"Status: {new_status}")
                        )[0],
                        "body": notifications.get(
                            new_status, ("Status Update", f"Status: {new_status}")
                        )[1],
                    }
                    if new_status in notifications
                    else None
                ),
            }
        else:
            return {"success": False, "message": "Failed to update status."}

    # ==================== Phase 11: Advanced Scheduling Methods ====================

    def queue_appointment(
        self,
        vehicle_id: str,
        component: str,
        severity: str,
        failure_probability: float = 0.5,
        diagnosis_summary: str = "",
        estimated_cost: str = "",
        customer_tier: str = "standard",
    ) -> Dict[str, Any]:
        """
        Add appointment request to priority queue.

        Args:
            vehicle_id: Vehicle ID
            component: Component needing service
            severity: Severity level
            failure_probability: ML-predicted failure probability
            diagnosis_summary: Diagnosis details
            estimated_cost: Cost estimate
            customer_tier: Customer tier for priority boost

        Returns:
            Queue confirmation with urgency score
        """
        urgency_score = calculate_urgency_score(
            severity, failure_probability, customer_tier
        )

        appointment = PriorityAppointment(
            vehicle_id=vehicle_id,
            component=component,
            severity=severity,
            failure_probability=failure_probability,
            urgency_score=urgency_score,
            customer_tier=customer_tier,
            diagnosis_summary=diagnosis_summary,
            estimated_cost=estimated_cost,
        )

        self.priority_queue.push(appointment)

        return {
            "success": True,
            "message": f"Request queued with urgency score {urgency_score:.1f}/10",
            "urgency_score": urgency_score,
            "queue_position": self._get_queue_position(vehicle_id),
            "queue_size": self.priority_queue.size(),
        }

    def _get_queue_position(self, vehicle_id: str) -> int:
        """Get position of vehicle in priority queue."""
        all_items = self.priority_queue.get_all()
        for i, item in enumerate(all_items):
            if item.vehicle_id == vehicle_id:
                return i + 1
        return -1

    def process_priority_queue(self, max_process: int = 5) -> Dict[str, Any]:
        """
        Process top priority appointments from queue.

        Args:
            max_process: Maximum number of appointments to process

        Returns:
            Processing results
        """
        processed = []
        failed = []

        for _ in range(min(max_process, self.priority_queue.size())):
            appointment = self.priority_queue.pop()
            if not appointment:
                break

            # Find optimal center
            center_result = self.get_optimal_center(
                appointment.vehicle_id, appointment.component
            )

            if not center_result.get("selected"):
                failed.append(
                    {
                        "vehicle_id": appointment.vehicle_id,
                        "reason": "No available center",
                    }
                )
                continue

            # Check availability and book
            availability = self.check_availability(
                appointment.component, appointment.severity, limit=1
            )

            if availability["available"] and availability["slots"]:
                slot = availability["slots"][0]
                result = self.book_appointment(
                    vehicle_id=appointment.vehicle_id,
                    slot_id=slot["id"],
                    center_id=center_result["selected"],
                    component=appointment.component,
                    diagnosis_summary=appointment.diagnosis_summary,
                    estimated_cost=appointment.estimated_cost,
                    urgency=appointment.severity,
                )

                if result["success"]:
                    processed.append(
                        {
                            "vehicle_id": appointment.vehicle_id,
                            "center": center_result["selected_name"],
                            "appointment": result.get("appointment", {}),
                        }
                    )
                else:
                    failed.append(
                        {
                            "vehicle_id": appointment.vehicle_id,
                            "reason": result.get("message", "Booking failed"),
                        }
                    )
            else:
                failed.append(
                    {
                        "vehicle_id": appointment.vehicle_id,
                        "reason": "No available slots",
                    }
                )

        return {
            "success": True,
            "processed_count": len(processed),
            "failed_count": len(failed),
            "processed": processed,
            "failed": failed,
            "remaining_in_queue": self.priority_queue.size(),
        }

    def get_optimal_center(
        self, vehicle_id: str, component: str = None
    ) -> Dict[str, Any]:
        """
        Get optimal service center for a vehicle using geo-optimization.

        Args:
            vehicle_id: Vehicle ID
            component: Optional component for parts check

        Returns:
            Optimal center selection with scoring
        """
        # Get vehicle location (simulated)
        location = self.vehicle_locations.get(
            vehicle_id, {"lat": 40.7128, "lon": -74.0060}  # Default NYC
        )

        # Get available centers
        centers = self.db.get_service_centers()
        available_center_ids = [c["id"] for c in centers]

        # Simulate parts availability
        parts_availability = {cid: True for cid in available_center_ids}

        # Select optimal
        result = select_optimal_center(
            vehicle_lat=location["lat"],
            vehicle_lon=location["lon"],
            available_centers=available_center_ids,
            component=component,
            parts_availability=parts_availability,
        )

        return result

    def initiate_emergency_towing(
        self,
        vehicle_id: str,
        pickup_location: str,
        notes: str = "",
    ) -> Dict[str, Any]:
        """
        Initiate emergency towing for immobile vehicle.

        Args:
            vehicle_id: Vehicle ID
            pickup_location: Current vehicle location description
            notes: Additional notes

        Returns:
            Towing request details
        """
        # Find nearest center
        center_result = self.get_optimal_center(vehicle_id)
        destination = center_result.get("selected", "SC-001")

        # Dispatch towing
        tow_request = dispatch_towing(
            vehicle_id=vehicle_id,
            pickup_location=pickup_location,
            destination_center=destination,
        )
        tow_request.notes = notes

        # Store active towing
        self.active_towing[tow_request.request_id] = tow_request

        return {
            "success": True,
            "request_id": tow_request.request_id,
            "status": tow_request.status.value,
            "tow_company": tow_request.tow_company,
            "driver_name": tow_request.driver_name,
            "driver_phone": tow_request.driver_phone,
            "eta_minutes": tow_request.estimated_eta_minutes,
            "destination": center_result.get("selected_name", destination),
            "message": f"🚗 Emergency towing dispatched! {tow_request.tow_company} will arrive in approximately {tow_request.estimated_eta_minutes} minutes.",
        }

    def get_towing_status(self, request_id: str) -> Dict[str, Any]:
        """Get status of a towing request."""
        tow_request = self.active_towing.get(request_id)

        if not tow_request:
            return {"success": False, "error": "Towing request not found"}

        return {
            "success": True,
            "request_id": request_id,
            "vehicle_id": tow_request.vehicle_id,
            "status": tow_request.status.value,
            "tow_company": tow_request.tow_company,
            "eta_minutes": tow_request.estimated_eta_minutes,
            "pickup_location": tow_request.pickup_location,
            "destination": tow_request.destination_center,
        }

    def get_demand_forecast(self, center_id: str, date: str = None) -> Dict[str, Any]:
        """
        Get service demand forecast for capacity planning.

        Args:
            center_id: Service center ID
            date: Target date (defaults to today)

        Returns:
            Hourly demand forecast
        """
        from datetime import datetime

        if not date:
            date = datetime.now().strftime("%Y-%m-%d")

        forecasts = forecast_demand(center_id, date)

        # Find peak hours
        peak_hours = [f for f in forecasts if f.recommendation == "high_demand"]
        low_hours = [f for f in forecasts if f.recommendation == "low_demand"]

        # Generate recommendation
        if peak_hours:
            peak_times = ", ".join([f"{f.hour}:00" for f in peak_hours[:3]])
            recommendation = f"Avoid scheduling during peak hours: {peak_times}"
        else:
            recommendation = "Normal demand expected throughout the day"

        return {
            "success": True,
            "center_id": center_id,
            "date": date,
            "hourly_forecast": [
                {
                    "hour": f.hour,
                    "demand": f.predicted_demand,
                    "available_capacity": f.available_capacity,
                    "status": f.recommendation,
                }
                for f in forecasts
            ],
            "peak_hours": [f.hour for f in peak_hours],
            "low_demand_hours": [f.hour for f in low_hours],
            "recommendation": recommendation,
        }

    def schedule_fleet_batch(
        self,
        tenant_id: str,
        vehicle_requests: List[Dict],
    ) -> Dict[str, Any]:
        """
        Schedule multiple vehicles from a fleet in batch.

        Args:
            tenant_id: Fleet tenant ID
            vehicle_requests: List of {vehicle_id, component, severity, ...}

        Returns:
            Batch scheduling results
        """
        # Sort by urgency
        sorted_requests = sorted(
            vehicle_requests,
            key=lambda r: calculate_urgency_score(
                r.get("severity", "medium"),
                r.get("failure_probability", 0.5),
            ),
            reverse=True,
        )

        scheduled = []
        failed = []

        for request in sorted_requests:
            vehicle_id = request.get("vehicle_id")
            component = request.get("component", "general")
            severity = request.get("severity", "medium")

            # Find optimal center
            center_result = self.get_optimal_center(vehicle_id, component)

            if not center_result.get("selected"):
                failed.append(
                    {"vehicle_id": vehicle_id, "reason": "No center available"}
                )
                continue

            # Check availability
            availability = self.check_availability(component, severity, limit=1)

            if availability["available"] and availability["slots"]:
                slot = availability["slots"][0]
                result = self.book_appointment(
                    vehicle_id=vehicle_id,
                    slot_id=slot["id"],
                    center_id=center_result["selected"],
                    component=component,
                    diagnosis_summary=request.get("diagnosis", f"{component} service"),
                    estimated_cost=request.get("cost", "TBD"),
                    urgency=severity,
                    notes=f"[FLEET:{tenant_id}]",
                )

                if result["success"]:
                    scheduled.append(
                        {
                            "vehicle_id": vehicle_id,
                            "appointment": result.get("appointment"),
                            "center": center_result["selected_name"],
                        }
                    )
                else:
                    failed.append(
                        {"vehicle_id": vehicle_id, "reason": result.get("message")}
                    )
            else:
                failed.append(
                    {"vehicle_id": vehicle_id, "reason": "No slots available"}
                )

        # Generate summary
        summary = f"Fleet batch scheduling complete: {len(scheduled)} scheduled, {len(failed)} failed"

        return {
            "success": True,
            "tenant_id": tenant_id,
            "total_requested": len(vehicle_requests),
            "scheduled_count": len(scheduled),
            "failed_count": len(failed),
            "scheduled": scheduled,
            "failed": failed,
            "summary": summary,
        }


# Singleton instance
_scheduling_agent: Optional[SchedulingAgent] = None


def get_scheduling_agent() -> SchedulingAgent:
    """Get or create SchedulingAgent singleton."""
    global _scheduling_agent
    if _scheduling_agent is None:
        _scheduling_agent = SchedulingAgent()
    return _scheduling_agent
