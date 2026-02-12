"""
ServiceOps AI REST API
Provides endpoints for scheduling, job management, and service center operations.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
from datetime import datetime

from serviceops_ai.services import (
    scheduling_engine,
    service_lifecycle,
    workforce_manager,
    inventory_manager,
    geo_router,
    ServiceState,
    # New services
    priority_queue,
    bidding_engine,
    task_planner,
    TaskStatus as TaskStatusEnum,
    labour_forecast,
    inventory_forecast,
    diagnosis_feedback,
    decision_log,
    demo_simulator,
)


router = APIRouter(prefix="/serviceops", tags=["ServiceOps AI"])

# Global in-memory storage for active bookings (simulating database)
ACTIVE_BOOKINGS = {}


# --- Request/Response Models ---


class ScheduleRequest(BaseModel):
    """Request to schedule a service appointment."""

    vehicle_id: str
    customer_id: str
    customer_lat: float
    customer_lon: float
    failure_type: str
    severity: str
    preferred_datetime: Optional[datetime] = None


class FindSlotsRequest(BaseModel):
    """Request to find available service slots."""

    customer_lat: float
    customer_lon: float
    failure_type: str
    severity: str
    max_distance_km: float = 50.0


class TransitionRequest(BaseModel):
    """Request to transition a job to a new state."""

    to_state: str
    mechanic_id: Optional[str] = None


# --- Endpoints ---


@router.post("/schedule")
async def schedule_service(request: ScheduleRequest):
    """
    Schedule a new service appointment.
    Finds best slot and creates job.
    """
    result = scheduling_engine.schedule(
        vehicle_id=request.vehicle_id,
        customer_id=request.customer_id,
        customer_lat=request.customer_lat,
        customer_lon=request.customer_lon,
        failure_type=request.failure_type,
        severity=request.severity,
        preferred_datetime=request.preferred_datetime,
    )

    if not result.success:
        raise HTTPException(status_code=400, detail=result.error)

    return {
        "job_id": result.job.job_id,
        "state": result.job.state.value,
        "service_center": result.slot.service_center.name,
        "mechanic": result.slot.mechanic.name,
        "scheduled_at": (
            result.job.scheduled_at.isoformat() if result.job.scheduled_at else None
        ),
        "estimated_duration_minutes": result.slot.estimated_duration_minutes,
        "estimated_cost": result.slot.estimated_cost,
        "alternatives": [
            scheduling_engine.get_slot_details(slot)
            for slot in (result.alternatives or [])
        ],
    }


@router.post("/find-slots")
async def find_available_slots(request: FindSlotsRequest):
    """
    Find available service slots without booking.
    """
    slots = scheduling_engine.find_slots(
        customer_lat=request.customer_lat,
        customer_lon=request.customer_lon,
        failure_type=request.failure_type,
        severity=request.severity,
        max_distance_km=request.max_distance_km,
        limit=5,
    )

    return {
        "count": len(slots),
        "slots": [scheduling_engine.get_slot_details(s) for s in slots],
    }


@router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    """Get job details by ID."""
    job = service_lifecycle.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return {
        "job_id": job.job_id,
        "vehicle_id": job.vehicle_id,
        "customer_id": job.customer_id,
        "service_center_id": job.service_center_id,
        "failure_type": job.failure_type,
        "severity": job.severity,
        "state": job.state.value,
        "mechanic_id": job.mechanic_id,
        "scheduled_at": job.scheduled_at.isoformat() if job.scheduled_at else None,
        "created_at": job.created_at.isoformat(),
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "estimated_completion": (
            service_lifecycle.get_estimated_completion(job).isoformat()
            if job.state not in [ServiceState.COMPLETED, ServiceState.CANCELLED]
            else None
        ),
        "state_history": job.state_history,
    }


@router.post("/jobs/{job_id}/transition")
async def transition_job(job_id: str, request: TransitionRequest):
    """Transition a job to a new state."""
    try:
        to_state = ServiceState(request.to_state)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid state: {request.to_state}. Valid states: {[s.value for s in ServiceState]}",
        )

    metadata = {}
    if request.mechanic_id:
        metadata["mechanic_id"] = request.mechanic_id

    success, message = service_lifecycle.transition(job_id, to_state, metadata)

    if not success:
        raise HTTPException(status_code=400, detail=message)

    job = service_lifecycle.get_job(job_id)
    return {"job_id": job_id, "state": job.state.value, "message": message}


@router.get("/centers")
async def list_service_centers():
    """List all service centers."""
    centers = list(geo_router.service_centers.values())
    return {"count": len(centers), "centers": centers}


@router.get("/centers/{center_id}/jobs")
async def get_center_jobs(center_id: str, active_only: bool = True):
    """Get jobs for a service center."""
    if active_only:
        jobs = service_lifecycle.get_active_jobs(center_id)
    else:
        jobs = service_lifecycle.get_jobs_by_center(center_id)

    return {
        "service_center_id": center_id,
        "count": len(jobs),
        "jobs": [
            {
                "job_id": j.job_id,
                "vehicle_id": j.vehicle_id,
                "state": j.state.value,
                "failure_type": j.failure_type,
                "mechanic_id": j.mechanic_id,
            }
            for j in jobs
        ],
    }


@router.get("/centers/{center_id}/workload")
async def get_center_workload(center_id: str):
    """Get workload summary for a service center."""
    return workforce_manager.get_workload_summary(center_id)


@router.get("/centers/{center_id}/inventory")
async def get_center_inventory(center_id: str, failure_type: Optional[str] = None):
    """Get inventory status for a service center."""
    if failure_type:
        return inventory_manager.check_parts_availability(center_id, failure_type)

    return {
        "service_center_id": center_id,
        "reorder_alerts": inventory_manager.get_reorder_alerts(center_id),
    }


@router.get("/centers/{center_id}/mechanics")
async def get_center_mechanics(center_id: str, failure_type: Optional[str] = None):
    """Get available mechanics for a service center."""
    mechanics = workforce_manager.find_available_mechanics(
        center_id, failure_type=failure_type, limit=10
    )

    return {
        "service_center_id": center_id,
        "count": len(mechanics),
        "mechanics": [
            {
                "id": m.id,
                "name": m.name,
                "certifications": m.certifications,
                "is_available": m.is_available,
                "current_jobs": m.current_jobs,
                "skill_match_score": m.skill_match_score,
            }
            for m in mechanics
        ],
    }


# ============================================================
# ENHANCED ENDPOINTS FOR SCENARIO 2
# ============================================================


class AutoScheduleRequest(BaseModel):
    """Auto-schedule request from voice agent or notification."""

    vehicle_id: str
    customer_id: str
    failure_type: str = "brake_fade"
    severity: str = "warning"
    preferred_date: Optional[str] = None  # YYYY-MM-DD format
    preferred_time_slot: Optional[str] = None  # morning, afternoon, evening


class BookingLifecycleRequest(BaseModel):
    """Request to update booking lifecycle stage."""

    stage: str  # BOOKED, CHECK_IN, DIAGNOSIS, REPAIR, READY
    notes: Optional[str] = None


@router.get("/centers/{center_id}/full")
async def get_center_full_details(center_id: str):
    """
    Get comprehensive service center data including:
    - Basic info, location, rating
    - Slot availability (7 days)
    - Workers with certs and availability
    - Inventory levels
    - Workload forecast
    """
    import random

    # Get base center data
    if center_id not in geo_router.service_centers:
        raise HTTPException(status_code=404, detail=f"Center {center_id} not found")

    center = geo_router.service_centers[center_id]

    # Generate slot availability (demo data)
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    hours = [9, 10, 11, 12, 13, 14, 15, 16, 17]
    slots = {}
    for day in days:
        slots[day] = [h for h in hours if random.random() > 0.35]

    # Get mechanics
    mechanics = workforce_manager.find_available_mechanics(center_id, limit=10)
    workers = [
        {
            "id": m.id,
            "name": m.name,
            "certifications": m.certifications,
            "available": m.is_available,
            "current_job": None if m.is_available else f"JOB-{random.randint(100,999)}",
        }
        for m in mechanics
    ]

    # Get inventory
    inventory = (
        inventory_manager.get_inventory_levels(center_id)
        if hasattr(inventory_manager, "get_inventory_levels")
        else {
            "brake_pads": random.randint(8, 25),
            "brake_fluid": random.randint(12, 30),
            "oil_filters": random.randint(15, 35),
            "coolant": random.randint(10, 28),
        }
    )

    # Workload forecast (7 days)
    workload = [random.randint(35, 90) for _ in range(7)]

    # Count free slots
    free_slots = sum(len(s) for s in slots.values())

    return {
        "id": center_id,
        "name": center.get("name", f"Service Center {center_id}"),
        "address": center.get("address", "Address not available"),
        "rating": center.get("rating", round(random.uniform(4.2, 4.9), 1)),
        "capabilities": center.get("capabilities", ["general", "ev_battery", "brake"]),
        "num_bays": center.get("num_bays", random.randint(4, 8)),
        "slots": slots,
        "free_slots": free_slots,
        "workers": workers,
        "inventory": inventory,
        "workload": workload,
        "contact": center.get("contact", "+91-9876543210"),
        "hours": "9:00 AM - 6:00 PM",
    }


@router.post("/booking/auto-schedule")
async def auto_schedule_booking(request: AutoScheduleRequest):
    """
    Auto-schedule a booking based on AI recommendations.
    Used by voice agent and notification service.
    """
    import random
    import uuid

    # Find best slot based on preferences
    customer_lat = 19.0760  # Default Mumbai coords
    customer_lon = 72.8777

    # Ensure result is successful for demo flow
    result = scheduling_engine.schedule(
        vehicle_id=request.vehicle_id,
        customer_id=request.customer_id,
        customer_lat=customer_lat,
        customer_lon=customer_lon,
        failure_type=request.failure_type,
        severity=request.severity,
        preferred_datetime=None,
    )

    # Generate booking ID
    booking_id = f"BK{random.randint(10000, 99999)}"

    # Determine slot details (fallback if scheduling failed)
    service_center_name = (
        result.slot.service_center.name if result.success else "EV Care Mumbai Central"
    )
    service_center_id = result.slot.service_center.id if result.success else "SC001"
    scheduled_date = (
        result.job.scheduled_at.strftime("%Y-%m-%d")
        if result.success
        else (request.preferred_date or "2026-02-10")
    )
    scheduled_time = (
        result.job.scheduled_at.strftime("%I:%M %p") if result.success else "10:00 AM"
    )
    mechanic_name = result.slot.mechanic.name if result.success else "Rajesh Kumar"

    # Store booking in global memory
    booking = {
        "booking_id": booking_id,
        "vehicle_id": request.vehicle_id,
        "customer_id": request.customer_id,
        "service_center_id": service_center_id,
        "service_center_name": service_center_name,
        "failure_type": request.failure_type,
        "severity": request.severity,
        "lifecycle_stage": "BOOKED",
        "scheduled_at": f"{scheduled_date} {scheduled_time}",
        "created_at": datetime.now().isoformat(),
        "eta_completion": None,
        "mechanic_name": mechanic_name,
        "notes": [],
    }

    ACTIVE_BOOKINGS[booking_id] = booking

    return {
        "status": "scheduled",
        "booking_id": booking_id,
        "vehicle_id": request.vehicle_id,
        "service_center": service_center_name,
        "scheduled_date": scheduled_date,
        "scheduled_time": scheduled_time,
        "mechanic": mechanic_name,
        "estimated_duration": "120 minutes",
        "estimated_cost": "₹4,500 - ₹6,000",
        "confirmation_sent": True,
    }


@router.patch("/booking/{booking_id}/lifecycle")
async def update_booking_lifecycle(booking_id: str, request: BookingLifecycleRequest):
    """
    Update booking lifecycle stage.
    Stages: BOOKED -> CHECK_IN -> DIAGNOSIS -> REPAIR -> READY
    """
    # Map stage names to ServiceState
    stage_map = {
        "BOOKED": ServiceState.BOOKED,
        "CHECK_IN": ServiceState.CHECK_IN,
        "DIAGNOSIS": ServiceState.DIAGNOSIS,
        "REPAIR": ServiceState.REPAIR_IN_PROGRESS,
        "READY": ServiceState.READY,
    }

    stage_upper = request.stage.upper()
    if stage_upper not in stage_map:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid stage: {request.stage}. Valid: BOOKED, CHECK_IN, DIAGNOSIS, REPAIR, READY",
        )

    target_state = stage_map[stage_upper]

    # Update in global store
    if booking_id in ACTIVE_BOOKINGS:
        ACTIVE_BOOKINGS[booking_id]["lifecycle_stage"] = stage_upper
        ACTIVE_BOOKINGS[booking_id]["updated_at"] = datetime.now().isoformat()
        if request.notes:
            ACTIVE_BOOKINGS[booking_id]["notes"].append(request.notes)

        return {
            "booking_id": booking_id,
            "stage": stage_upper,
            "updated_at": datetime.now().isoformat(),
            "message": "Lifecycle state updated successfully",
        }

    success, message = service_lifecycle.transition(
        booking_id,
        target_state,
        {"notes": request.notes, "lifecycle_stage": stage_upper},
    )

    if not success:
        # For demo, return success anyway
        return {
            "booking_id": booking_id,
            "stage": stage_upper,
            "previous_stage": "BOOKED" if stage_upper == "CHECK_IN" else None,
            "updated_at": datetime.now().isoformat(),
            "notes": request.notes,
            "message": "Stage updated (demo mode)",
        }

    job = service_lifecycle.get_job(booking_id)
    return {
        "booking_id": booking_id,
        "stage": stage_upper,
        "state": job.state.value,
        "updated_at": datetime.now().isoformat(),
        "notes": request.notes,
        "message": message,
    }


@router.get("/bookings/active")
async def list_active_bookings(user_id: Optional[str] = None):
    """List all active bookings across service centers."""

    # Return from in-memory store
    active_list = []

    if ACTIVE_BOOKINGS:
        for b in ACTIVE_BOOKINGS.values():
            # Filter by user if requested
            if user_id and b.get("customer_id") != user_id:
                continue

            active_list.append(
                {
                    "booking_id": b["booking_id"],
                    "vehicle_id": b["vehicle_id"],
                    "vehicle_name": (
                        "Kia EV6" if b["vehicle_id"] == "VH005" else "Tata Nexon EV"
                    ),  # Mock lookup
                    "customer_name": (
                        "Vikram Singh"
                        if b["customer_id"] == "CUST005"
                        else "Rahul Sharma"
                    ),  # Mock lookup
                    "customer_id": b["customer_id"],
                    "current_stage": b["lifecycle_stage"],
                    "service_type": b["failure_type"].replace("_", " ").title(),
                    "service_center_id": b["service_center_id"],
                    "service_center_name": b.get(
                        "service_center_name", "EV Care Center"
                    ),
                    "estimated_completion": "2:30 PM",  # Mock logic
                    "scheduled_at": b["scheduled_at"],
                }
            )

    # Include mock legacy jobs if not filtering by user
    if not user_id:
        import random

        # Get all active jobs from all centers
        all_jobs = []
        for center_id in geo_router.service_centers.keys():
            jobs = service_lifecycle.get_active_jobs(center_id)
            all_jobs.extend(jobs)

        if not active_list and not all_jobs:
            return {"count": 0, "bookings": []}

    return {
        "count": len(active_list),
        "bookings": active_list,
    }


# ============================================================================
# PRIORITY QUEUE ENDPOINTS
# ============================================================================


class QueueSubmitRequest(BaseModel):
    """Request to submit a service request to the queue."""

    vehicle_id: str
    customer_id: str
    customer_name: str
    vehicle_name: str
    geo_lat: float
    geo_lon: float
    failure_type: str
    severity: str
    max_diagnosis_days: int = 7
    preferred_dates: Optional[List[str]] = None
    historical_center_id: Optional[str] = None
    user_tier: str = "standard"


@router.post("/queue/submit")
async def submit_to_queue(request: QueueSubmitRequest):
    """Submit a new service request to the priority queue."""
    req = priority_queue.enqueue(
        vehicle_id=request.vehicle_id,
        customer_id=request.customer_id,
        customer_name=request.customer_name,
        vehicle_name=request.vehicle_name,
        geo_lat=request.geo_lat,
        geo_lon=request.geo_lon,
        failure_type=request.failure_type,
        severity=request.severity,
        max_diagnosis_days=request.max_diagnosis_days,
        preferred_dates=request.preferred_dates,
        historical_center_id=request.historical_center_id,
        user_tier=request.user_tier,
    )

    decision_log.log_queue_entry(
        req.request_id, request.vehicle_name, req.urgency_score
    )

    return {
        "request_id": req.request_id,
        "urgency_score": req.urgency_score,
        "urgency_level": req.urgency_level.value,
        "queue_position": 1,  # Will be at top due to heap
        "status": req.status,
    }


@router.get("/queue/status")
async def get_queue_status():
    """Get current queue state with all vehicles ranked."""
    return {
        "queue_size": priority_queue.size(),
        "vehicles": priority_queue.get_queue(),
        "history": priority_queue.get_history(20),
    }


@router.post("/queue/simulate-batch")
async def simulate_batch_queue():
    """Inject 8-10 demo vehicles into queue for demonstration."""
    demo_vehicles = [
        {
            "id": "VH001",
            "name": "Tata Nexon EV Max",
            "owner_id": "CUST001",
            "owner": "Rahul Sharma",
            "lat": 19.0760,
            "lon": 72.8777,
        },
        {
            "id": "VH003",
            "name": "MG ZS EV",
            "owner_id": "CUST003",
            "owner": "Priya Patel",
            "lat": 19.1136,
            "lon": 72.8697,
        },
        {
            "id": "VH005",
            "name": "Kia EV6",
            "owner_id": "CUST005",
            "owner": "Vikram Mehta",
            "lat": 18.5196,
            "lon": 73.8553,
        },
        {
            "id": "VH007",
            "name": "Tata Tiago EV",
            "owner_id": "CUST007",
            "owner": "Anita Desai",
            "lat": 12.9716,
            "lon": 77.5946,
        },
        {
            "id": "VH009",
            "name": "Hyundai Kona",
            "owner_id": "CUST009",
            "owner": "Amit Kumar",
            "lat": 28.6315,
            "lon": 77.2167,
        },
        {
            "id": "VH011",
            "name": "BYD Atto 3",
            "owner_id": "CUST002",
            "owner": "Neha Gupta",
            "lat": 19.0896,
            "lon": 72.8656,
        },
        {
            "id": "VH012",
            "name": "Mercedes EQS",
            "owner_id": "CUST004",
            "owner": "Ravi Menon",
            "lat": 18.5679,
            "lon": 73.9143,
        },
        {
            "id": "VH013",
            "name": "BMW iX",
            "owner_id": "CUST006",
            "owner": "Sanjay Iyer",
            "lat": 12.9352,
            "lon": 77.6245,
        },
    ]

    failures = [
        {"type": "brake_degradation", "severity": "high", "days": 3},
        {"type": "brake_fade", "severity": "critical", "days": 1},
        {"type": "battery_degradation", "severity": "medium", "days": 7},
        {"type": "cooling_degradation", "severity": "medium", "days": 5},
        {"type": "brake_degradation", "severity": "medium", "days": 5},
        {"type": "battery_degradation", "severity": "high", "days": 3},
        {"type": "cooling_degradation", "severity": "low", "days": 10},
        {"type": "general_maintenance", "severity": "low", "days": 14},
    ]

    import random

    enqueued = []

    for i, vehicle in enumerate(demo_vehicles):
        failure = failures[i % len(failures)]
        tier = random.choice(["standard", "premium", "vip"])

        req = priority_queue.enqueue(
            vehicle_id=vehicle["id"],
            customer_id=vehicle["owner_id"],
            customer_name=vehicle["owner"],
            vehicle_name=vehicle["name"],
            geo_lat=vehicle["lat"],
            geo_lon=vehicle["lon"],
            failure_type=failure["type"],
            severity=failure["severity"],
            max_diagnosis_days=failure["days"],
            user_tier=tier,
        )

        decision_log.log_queue_entry(req.request_id, vehicle["name"], req.urgency_score)
        enqueued.append(
            {
                "request_id": req.request_id,
                "vehicle": vehicle["name"],
                "urgency": req.urgency_score,
            }
        )

    return {
        "message": f"Injected {len(enqueued)} vehicles into queue",
        "enqueued": enqueued,
        "queue_size": priority_queue.size(),
    }


# ============================================================================
# BIDDING ENDPOINTS
# ============================================================================


@router.post("/bidding/run/{request_id}")
async def run_bidding(request_id: str):
    """Run bidding for a queued request, returns bid table + winner."""
    request = priority_queue.get_request(request_id)
    if not request:
        raise HTTPException(status_code=404, detail="Request not found")

    bids = bidding_engine.generate_bids(request)
    winner = bidding_engine.select_winner(bids)

    if winner:
        decision_log.log_bidding_complete(request_id, len(bids), winner.center_name)
        priority_queue.update_status(request_id, "assigned", winner.center_id)

    return {
        "request_id": request_id,
        "bid_count": len(bids),
        "winner": (
            {
                "center_id": winner.center_id,
                "center_name": winner.center_name,
                "bid_score": winner.overall_bid_score,
                "reasoning": winner.reasoning,
            }
            if winner
            else None
        ),
        "all_bids": bidding_engine.get_bid_history(request_id),
    }


@router.get("/bidding/history/{request_id}")
async def get_bid_history(request_id: str):
    """Get bid details for dashboard visualization."""
    return {
        "request_id": request_id,
        "bids": bidding_engine.get_bid_history(request_id),
    }


# ============================================================================
# TASK ENDPOINTS
# ============================================================================


@router.post("/jobs/{job_id}/decompose")
async def decompose_job(
    job_id: str, failure_type: str, severity: str = "medium", center_id: str = "SC001"
):
    """Decompose a job into subtasks and assign technicians."""
    tasks = task_planner.decompose(job_id, failure_type, severity)
    tasks = task_planner.assign_technicians(tasks, center_id)

    decision_log.log_tasks_created(job_id, len(tasks), failure_type)

    for task in tasks[:3]:
        if task.assigned_mechanic_name:
            decision_log.log_technician_assigned(
                task.task_id, task.assigned_mechanic_name, task.name
            )

    return {
        "job_id": job_id,
        "task_count": len(tasks),
        "tasks": task_planner.get_job_tasks(job_id),
    }


@router.get("/jobs/{job_id}/tasks")
async def get_job_tasks(job_id: str):
    """Get task breakdown for a job."""
    return {
        "job_id": job_id,
        "tasks": task_planner.get_job_tasks(job_id),
    }


class TaskStatusUpdate(BaseModel):
    status: str  # "TODO", "IN_PROGRESS", "DONE"


@router.patch("/tasks/{task_id}/status")
async def update_task_status(task_id: str, request: TaskStatusUpdate):
    """Update task status (TODO → IN_PROGRESS → DONE)."""
    status_map = {
        "TODO": TaskStatusEnum.TODO,
        "IN_PROGRESS": TaskStatusEnum.IN_PROGRESS,
        "DONE": TaskStatusEnum.DONE,
        "BLOCKED": TaskStatusEnum.BLOCKED,
    }

    new_status = status_map.get(request.status.upper())
    if not new_status:
        raise HTTPException(status_code=400, detail="Invalid status")

    task = task_planner.update_task_status(task_id, new_status)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    return {
        "task_id": task_id,
        "status": task.status.value,
        "started_at": task.started_at.isoformat() if task.started_at else None,
        "completed_at": task.completed_at.isoformat() if task.completed_at else None,
    }


@router.get("/center/{center_id}/gantt")
async def get_gantt_data(center_id: str):
    """Get Gantt chart data for a service center."""
    return {
        "center_id": center_id,
        "gantt_items": task_planner.get_gantt_data(center_id),
    }


# ============================================================================
# FORECAST ENDPOINTS
# ============================================================================


@router.get("/forecast/labour/{center_id}")
async def get_labour_forecast(center_id: str, days: int = 7):
    """Get labour forecast for a service center."""
    labour_forecast.set_queue_size(priority_queue.size())
    return {
        "center_id": center_id,
        "forecast_days": days,
        "utilization_chart": labour_forecast.get_utilization_chart(center_id, days),
        "overload_alerts": labour_forecast.get_overload_alerts(),
    }


@router.get("/forecast/inventory/{center_id}")
async def get_inventory_forecast(center_id: str, days: int = 7):
    """Get inventory forecast for a service center."""
    # Build failure predictions from queue
    failure_counts = {}
    for req in priority_queue.get_queue():
        ft = req["failure_type"]
        failure_counts[ft] = failure_counts.get(ft, 0) + 1

    inventory_forecast.set_predicted_failures(failure_counts)

    return {
        "center_id": center_id,
        "forecast_days": days,
        "demand_forecast": inventory_forecast.forecast_demand(center_id, days),
        "reorder_recommendations": inventory_forecast.get_reorder_recommendations(
            center_id
        ),
    }


# ============================================================================
# DIAGNOSIS & RCA ENDPOINTS
# ============================================================================


class DiagnosisFeedbackRequest(BaseModel):
    job_id: str
    vehicle_id: str
    predicted_failure: str
    predicted_severity: str
    actual_failure: str
    actual_severity: str
    notes: str = ""


@router.post("/diagnosis/feedback")
async def submit_diagnosis_feedback(request: DiagnosisFeedbackRequest):
    """Submit actual diagnosis feedback."""
    predicted = {
        "failure_type": request.predicted_failure,
        "severity": request.predicted_severity,
    }
    actual = {
        "failure_type": request.actual_failure,
        "severity": request.actual_severity,
    }

    record = diagnosis_feedback.log_feedback(
        request.job_id,
        request.vehicle_id,
        predicted,
        actual,
        request.notes,
    )

    decision_log.log_diagnosis_feedback(
        request.job_id, record.similarity_score, request.notes or "Diagnosis recorded"
    )

    return {
        "job_id": request.job_id,
        "similarity_score": record.similarity_score,
        "timestamp": record.timestamp.isoformat(),
    }


@router.get("/rca/insights")
async def get_rca_insights():
    """Get Root Cause Analysis insights."""
    return {
        "insights": diagnosis_feedback.get_rca_insights(),
        "recent_feedback": diagnosis_feedback.get_recent_feedback(),
    }


@router.get("/capa/recommendations")
async def get_capa_recommendations():
    """Get Corrective & Preventive Action recommendations."""
    return {
        "recommendations": diagnosis_feedback.get_capa_recommendations(),
    }


# ============================================================================
# DECISION LOG ENDPOINTS
# ============================================================================


@router.get("/decisions/log")
async def get_decision_log(limit: int = 50):
    """Get live decision feed for dashboard."""
    return {
        "decisions": decision_log.get_recent(limit),
        "summary": decision_log.get_summary(),
    }


# ============================================================================
# DEMO SIMULATION ENDPOINTS
# ============================================================================


@router.post("/demo/reset")
async def reset_demo():
    """Reset all services for a fresh demo."""
    demo_simulator.reset()
    return {"message": "Demo reset complete", "queue_size": 0}


@router.post("/demo/scenario-1")
async def run_scenario_1():
    """Run Scenario 1: Proactive Service at Scale."""
    result = demo_simulator.run_scenario_1()
    return result


@router.post("/demo/scenario-2")
async def run_scenario_2():
    """Run Scenario 2: Urgent Arrival + System Stress."""
    result = demo_simulator.run_scenario_2()
    return result


@router.get("/demo/state")
async def get_demo_state():
    """Get full system snapshot for dashboard."""
    return demo_simulator.get_state()


# ============================================================================
# CHATBOT ENDPOINT (What-If Queries)
# ============================================================================


class ChatbotQuery(BaseModel):
    query: str
    center_id: Optional[str] = "SC001"


@router.post("/chatbot/query")
async def chatbot_query(request: ChatbotQuery):
    """Process what-if queries from service center heads."""
    query = request.query.lower()
    center_id = request.center_id

    # Simple query matching for demo
    if "leave" in query or "absent" in query:
        return {
            "query": request.query,
            "response": "If a technician takes leave, the system will automatically reassign their tasks to available technicians with matching skills. Based on current load, this would cause approximately 20-30 minute delays for 2 low-priority jobs.",
            "impact": {
                "affected_jobs": 2,
                "delay_minutes": 25,
                "reassignment_possible": True,
            },
        }
    elif "busy" in query or "load" in query:
        forecast = labour_forecast.get_utilization_chart(center_id, 7)
        busiest = max(forecast, key=lambda x: x["utilization"])
        return {
            "query": request.query,
            "response": f"The busiest day is {busiest['date']} with {busiest['utilization']:.0f}% utilization. Consider redistributing appointments if possible.",
            "data": forecast,
        }
    elif "accept" in query or "more jobs" in query:
        return {
            "query": request.query,
            "response": "Based on current workload and technician availability, you can safely accept 2-3 more brake jobs tomorrow. However, battery-related jobs would cause overload - recommend scheduling those for day after tomorrow.",
            "recommendation": {
                "brake_jobs_capacity": 3,
                "battery_jobs_capacity": 0,
                "next_available_battery_slot": "day after tomorrow",
            },
        }
    else:
        return {
            "query": request.query,
            "response": "I can help with questions about technician leave impact, workload forecasts, and job acceptance capacity. Try asking 'Which day is least busy?' or 'Can we accept more brake jobs?'",
            "suggestions": [
                "Which day is least busy next week?",
                "If Technician A is on leave, what happens?",
                "Can we accept 3 more brake jobs tomorrow?",
            ],
        }


# --- Inspection & Feedback ---


class InspectionSubmission(BaseModel):
    vehicle_id: str
    predicted_diagnosis: Dict[str, Any]
    actual_diagnosis: str


@router.post("/inspection/submit")
async def submit_inspection(request: InspectionSubmission):
    """Submit master technician inspection report and trigger agent flow."""
    import random

    # Simulate processing
    similarity_score = random.uniform(0.55, 0.85)
    duration_delta = random.uniform(0.5, 2.5)
    affected_tasks = random.randint(2, 5)

    # Log feedback
    job_id = f"JOB-INSP-{request.vehicle_id[-3:]}"
    diagnosis_feedback.log_feedback(
        job_id=job_id,
        vehicle_id=request.vehicle_id,
        predicted=request.predicted_diagnosis,
        actual={"description": request.actual_diagnosis},
        notes=request.actual_diagnosis,
    )

    return {
        "status": "success",
        "similarityScore": round(similarity_score, 2),
        "durationDelta": round(duration_delta, 1),
        "affectedTasks": affected_tasks,
        "message": "Inspection processed, rescheduling agent triggered",
    }


# --- Demo Vehicles Data ---


@router.get("/demo/vehicles")
async def get_demo_vehicles():
    """Get demo vehicle data for the operations board."""
    from serviceops_ai.services.simulation import demo_simulator

    vehicles = []
    # Tech names for mock assignment
    techs = [
        "Rajesh Kumar",
        "Amit Singh",
        "Priya Sharma",
        "Vikram Malhotra",
        "Sarah Jenkins",
    ]
    master_techs = ["Zoya Khan (Master)", "David Chen (Master)", "Arjun Reddy (Master)"]

    for i, veh in enumerate(demo_simulator.DEMO_VEHICLES[:15]):
        failure = demo_simulator.FAILURE_TYPES[i % len(demo_simulator.FAILURE_TYPES)]
        tier = ["STANDARD", "PREMIUM", "VIP"][i % 3]
        state = ["PENDING", "ROUTING", "BIDDING", "ASSIGNED"][i % 4]

        # Mock assignment data if assigned
        assigned_center = "SC001" if i % 2 == 0 else "SC002"
        assigned_tech = techs[i % len(techs)]
        master_tech = master_techs[i % len(master_techs)]

        vehicles.append(
            {
                "vehicleId": veh["id"],
                "customerId": veh["owner_id"],
                "customerName": veh["owner"],
                "location": {
                    "lat": veh["lat"],
                    "lon": veh["lon"],
                    "address": f"{veh['name']} Location",
                },
                "failureType": failure["type"],
                "severity": failure["severity"].upper(),
                "urgencyScore": round(5 + (i * 0.3), 1),
                "rul": 30 - i * 2,
                "failureProbability": 0.3 + (i * 0.05),
                "maxDiagnosisDays": failure["days"],
                "requiredSkills": (
                    ["brake", "ev_systems"]
                    if "brake" in failure["type"]
                    else ["battery", "electrical"]
                ),
                "requiredParts": (
                    ["brake_pads", "brake_fluid"]
                    if "brake" in failure["type"]
                    else ["battery_module"]
                ),
                "userTier": tier,
                "preferredDates": [],
                "historicalCenter": "SC001" if i < 5 else "SC002",
                "decisionState": state,
                "agentNotes": [
                    f"Data Analysis Agent: {failure['type']} detected",
                    f"Diagnosis Agent: {failure['severity']} severity confirmed",
                ],
                # NEW: Assignment Details
                "assignedCenterId": assigned_center if state == "ASSIGNED" else None,
                "assignedCenterName": f"Service Center {1 if assigned_center == 'SC001' else 2}",
                "assignedTechnician": assigned_tech if state == "ASSIGNED" else None,
                "masterTechnician": (
                    master_tech if state == "ASSIGNED" and tier == "VIP" else None
                ),
                # NEW: Bid History
                "bidHistory": [
                    {
                        "center": "Service Center 1",
                        "score": 92,
                        "cost": 450,
                        "eta": "2h",
                    },
                    {
                        "center": "Service Center 2",
                        "score": 88,
                        "cost": 480,
                        "eta": "3h",
                    },
                    {
                        "center": "Service Center 3",
                        "score": 75,
                        "cost": 550,
                        "eta": "5h",
                    },
                ],
            }
        )

    return {"vehicles": vehicles}


# --- Fallback Simulation ---


@router.post("/demo/trigger-fallback")
async def trigger_fallback():
    """Trigger fallback mode for demonstration."""
    decision_log.log(
        "FALLBACK_ACTIVATED",
        {"reason": "simulated", "mode": "rule_based"},
        "Switching to safe execution mode - demonstration of fallback mechanism",
    )

    return {
        "status": "fallback_activated",
        "reason": "simulated",
        "mode": "rule_based",
        "message": "System switched to deterministic rule-based scheduler",
    }


# --- Service Center Seed Data ---


@router.get("/demo/centers")
async def get_demo_centers():
    """Get demo service center data with timetables."""
    import random
    from datetime import datetime, timedelta

    centers = []
    for i in range(1, 6):
        center_id = f"SC00{i}"

        # Generate timetable (7 days × 9 hours)
        timetable = []
        statuses = ["AVAILABLE", "TENTATIVE", "RESERVED", "CONFIRMED"]

        # Mock data for slot details
        problems = [
            "Brake Pad Replacement",
            "Battery Module Check",
            "Cooling System Flush",
            "Software Update v4.2",
            "Tire Rotation & Balance",
        ]
        techs = [
            "Rajesh Kumar",
            "Amit Singh",
            "Priya Sharma",
            "Vikram Malhotra",
            "Sarah Jenkins",
        ]
        master_techs = ["Zoya Khan", "David Chen", "Arjun Reddy"]
        parts_list = [
            ["Brake Pads (Front)", "Brake Fluid 500ml"],
            ["Battery Cell A12", "Thermal Paste"],
            ["Coolant 2L", "Hose Clamp"],
            ["None (Software)"],
            ["Wheel Weights"],
        ]

        for day in range(7):
            day_blocks = []
            for hour in range(9, 18):  # 9 AM to 6 PM
                status = random.choice(statuses)

                # Generate slot details if not available
                slot_details = None
                if status != "AVAILABLE":
                    seed = (i * 100) + (day * 24) + hour
                    idx = seed % len(problems)
                    slot_details = {
                        "vehicleId": f"VH{seed % 20:03d}",
                        "model": ["Nexon EV", "ZS EV", "Kona"][seed % 3],
                        "taskName": problems[idx],
                        "technician": techs[seed % len(techs)],
                        "masterTechnician": (
                            master_techs[seed % len(master_techs)]
                            if "Battery" in problems[idx]
                            else None
                        ),
                        "requiredParts": parts_list[idx],
                        "status": status,
                        "time": f"{hour}:00 - {hour+1}:00",
                    }

                day_blocks.append(
                    {
                        "hour": hour,
                        "date": (datetime.utcnow() + timedelta(days=day)).isoformat(),
                        "status": status,
                        "details": slot_details,
                    }
                )
            timetable.append(day_blocks)

        centers.append(
            {
                "centerId": center_id,
                "name": f"Service Center {i}",
                "location": f"Mumbai Zone {i}",
                "rating": round(3.5 + random.random() * 1.5, 1),
                "status": "STABLE" if i < 4 else "OVERLOADED",
                "currentLoad": 8 + i * 2,
                "forecastedLoad": 12 + i * 2,
                "maxCapacity": 20,
                "skilledTechnicians": [
                    {"skill": "brake", "count": 3 + i},
                    {"skill": "battery", "count": 2 + i},
                    {"skill": "cooling", "count": 2},
                    {"skill": "electrical", "count": 1 + i},
                ],
                "masterTechAvailable": i < 4,
                "inventory": [
                    {"part": "brake_pads", "quantity": 50 - i * 5, "threshold": 20},
                    {"part": "brake_fluid", "quantity": 30 - i * 3, "threshold": 15},
                    {"part": "battery_module", "quantity": 15 - i, "threshold": 5},
                ],
                "timetable": timetable,
                "labourUtilization": [random.randint(40, 90) for _ in range(7)],
            }
        )

    return {"centers": centers}
