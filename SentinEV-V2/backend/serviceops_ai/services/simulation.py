"""
Demo Simulation Controller for ServiceOps AI
Orchestrates the two judge-ready demo scenarios.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import time
import random
import json
from pathlib import Path

from serviceops_ai.services.priority_queue import priority_queue, ServiceRequest
from serviceops_ai.services.bidding import bidding_engine
from serviceops_ai.services.task_planner import task_planner, TaskStatus
from serviceops_ai.services.lifecycle import service_lifecycle, ServiceState
from serviceops_ai.services.workforce import workforce_manager
from serviceops_ai.services.inventory import inventory_manager
from serviceops_ai.services.forecasting import labour_forecast, inventory_forecast
from serviceops_ai.services.diagnosis import diagnosis_feedback
from serviceops_ai.services.decision_log import decision_log, DecisionType


class DemoSimulator:
    """
    Orchestrates complete demo scenarios for judges.
    Each scenario shows the full agentic flow with visual progression.
    """

    # Demo vehicle data
    DEMO_VEHICLES = [
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
            "name": "Hyundai Kona Electric",
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

    # Failure scenarios
    FAILURE_TYPES = [
        {"type": "brake_degradation", "severity": "high", "days": 3},
        {"type": "brake_fade", "severity": "critical", "days": 1},
        {"type": "battery_degradation", "severity": "medium", "days": 7},
        {"type": "cooling_degradation", "severity": "medium", "days": 5},
        {"type": "brake_degradation", "severity": "medium", "days": 5},
        {"type": "battery_degradation", "severity": "high", "days": 3},
        {"type": "cooling_degradation", "severity": "low", "days": 10},
        {"type": "general_maintenance", "severity": "low", "days": 14},
    ]

    def __init__(self):
        self._simulation_state = {
            "running": False,
            "scenario": None,
            "current_step": 0,
            "total_steps": 0,
            "step_description": "",
            "completed_steps": [],
        }
        self._processed_requests: List[Dict] = []

    def reset(self):
        """Reset all services for a fresh demo."""
        priority_queue.clear()
        task_planner.reset()
        diagnosis_feedback.reset()
        decision_log.clear()
        self._processed_requests.clear()
        self._simulation_state = {
            "running": False,
            "scenario": None,
            "current_step": 0,
            "total_steps": 0,
            "step_description": "",
            "completed_steps": [],
        }

        decision_log.log(
            DecisionType.SYSTEM_INFO,
            {"action": "reset"},
            "System reset for new demo scenario",
        )

    def get_state(self) -> Dict[str, Any]:
        """Get complete system state snapshot for dashboard."""
        return {
            "simulation": self._simulation_state,
            "queue": priority_queue.get_queue(),
            "queue_size": priority_queue.size(),
            "processed_requests": self._processed_requests,
            "decisions": decision_log.get_recent(30),
            "decision_summary": decision_log.get_summary(),
        }

    def run_scenario_1(self) -> Dict[str, Any]:
        """
        Scenario 1: Proactive Service at Scale
        - Multiple vehicles enter queue
        - Priority sorting
        - Bidding & center selection
        - Task decomposition
        - Forecasting
        """
        self.reset()
        self._simulation_state["running"] = True
        self._simulation_state["scenario"] = "SCENARIO_1"
        self._simulation_state["total_steps"] = 6

        results = {
            "scenario": "Proactive Service at Scale",
            "steps": [],
        }

        # Step 1: Queue multiple vehicles
        self._update_step(1, "Receiving service requests from fleet")
        enqueueed_requests = []

        for i, vehicle in enumerate(self.DEMO_VEHICLES[:6]):
            failure = self.FAILURE_TYPES[i % len(self.FAILURE_TYPES)]
            tier = random.choice(["standard", "premium", "vip"])

            request = priority_queue.enqueue(
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

            decision_log.log_queue_entry(
                request.request_id, vehicle["name"], request.urgency_score
            )

            enqueueed_requests.append(
                {
                    "request_id": request.request_id,
                    "vehicle": vehicle["name"],
                    "urgency": request.urgency_score,
                }
            )

        results["steps"].append(
            {
                "step": 1,
                "name": "Queue Entry",
                "description": f"{len(enqueueed_requests)} vehicles entered priority queue",
                "data": enqueueed_requests,
            }
        )

        # Step 2: Process top 3 requests through bidding
        self._update_step(2, "Running center bidding for high-priority requests")
        bidding_results = []

        for _ in range(3):
            request = priority_queue.dequeue()
            if not request:
                break

            bids = bidding_engine.generate_bids(request)
            winner = bidding_engine.select_winner(bids)

            if winner:
                decision_log.log_bidding_complete(
                    request.request_id, len(bids), winner.center_name
                )

                bidding_results.append(
                    {
                        "request_id": request.request_id,
                        "vehicle": request.vehicle_name,
                        "bids_received": len(bids),
                        "winner": winner.center_name,
                        "bid_score": winner.overall_bid_score,
                    }
                )

                # Update request status
                priority_queue.update_status(
                    request.request_id, "assigned", winner.center_id
                )
                request.assigned_center_id = winner.center_id
                self._processed_requests.append(
                    {
                        "request": request.__dict__,
                        "winning_bid": winner.__dict__,
                    }
                )

        results["steps"].append(
            {
                "step": 2,
                "name": "Center Bidding",
                "description": "Intelligent center selection via internal bidding",
                "data": bidding_results,
            }
        )

        # Step 3: Task decomposition for winning jobs
        self._update_step(3, "Decomposing jobs into subtasks")
        task_results = []

        for proc in self._processed_requests[:3]:
            req = proc["request"]
            job_id = f"JOB-{req['request_id'].split('-')[1]}"

            tasks = task_planner.decompose(job_id, req["failure_type"], req["severity"])

            # Assign technicians
            center_id = proc["winning_bid"]["center_id"]
            tasks = task_planner.assign_technicians(tasks, center_id)

            decision_log.log_tasks_created(job_id, len(tasks), req["failure_type"])

            for task in tasks[:2]:  # Log first 2 assignments
                if task.assigned_mechanic_name:
                    decision_log.log_technician_assigned(
                        task.task_id, task.assigned_mechanic_name, task.name
                    )

            task_results.append(
                {
                    "job_id": job_id,
                    "vehicle": req["vehicle_name"],
                    "task_count": len(tasks),
                    "tasks": [
                        {"name": t.name, "mechanic": t.assigned_mechanic_name}
                        for t in tasks
                    ],
                }
            )

        results["steps"].append(
            {
                "step": 3,
                "name": "Task Decomposition",
                "description": "Jobs broken into subtasks with technician assignments",
                "data": task_results,
            }
        )

        # Step 4: Labour forecasting
        self._update_step(4, "Forecasting labour demand")
        labour_forecast.set_queue_size(priority_queue.size())

        forecast_results = []
        for center_id in ["SC001", "SC002", "SC003"]:
            forecast = labour_forecast.get_utilization_chart(center_id)
            forecast_results.append(
                {
                    "center_id": center_id,
                    "forecast": forecast[:3],  # First 3 days
                }
            )

        overload_alerts = labour_forecast.get_overload_alerts()

        results["steps"].append(
            {
                "step": 4,
                "name": "Labour Forecast",
                "description": "7-day workload prediction generated",
                "data": {
                    "forecasts": forecast_results,
                    "overload_alerts": overload_alerts,
                },
            }
        )

        # Step 5: Inventory forecasting
        self._update_step(5, "Forecasting inventory demand")

        # Build failure predictions from queue
        failure_counts = {}
        for req in priority_queue.get_queue():
            ft = req["failure_type"]
            failure_counts[ft] = failure_counts.get(ft, 0) + 1

        inventory_forecast.set_predicted_failures(failure_counts)

        inv_results = []
        for center_id in ["SC001", "SC002"]:
            recommendations = inventory_forecast.get_reorder_recommendations(center_id)
            inv_results.append(
                {
                    "center_id": center_id,
                    "recommendations": recommendations[:3],
                }
            )

            for rec in recommendations:
                if rec.get("urgency") == "high":
                    decision_log.log_reorder(
                        center_id, rec.get("part", "unknown"), "high"
                    )

        results["steps"].append(
            {
                "step": 5,
                "name": "Inventory Forecast",
                "description": "Proactive reorder recommendations generated",
                "data": inv_results,
            }
        )

        # Step 6: Complete
        self._update_step(6, "Scenario 1 complete")

        results["steps"].append(
            {
                "step": 6,
                "name": "Complete",
                "description": f"Processed {len(self._processed_requests)} requests autonomously",
                "data": {
                    "total_queued": 6,
                    "total_processed": len(self._processed_requests),
                    "remaining_in_queue": priority_queue.size(),
                },
            }
        )

        self._simulation_state["running"] = False
        results["final_state"] = self.get_state()

        return results

    def run_scenario_2(self) -> Dict[str, Any]:
        """
        Scenario 2: Urgent Arrival + System Stress
        - Queue already has vehicles
        - Critical vehicle arrives
        - Preemption
        - Diagnosis feedback
        - RCA/CAPA
        """
        # Use existing queue or add some vehicles
        if priority_queue.size() < 3:
            self.run_scenario_1()  # Build baseline first

        self._simulation_state["running"] = True
        self._simulation_state["scenario"] = "SCENARIO_2"
        self._simulation_state["total_steps"] = 5

        results = {
            "scenario": "Urgent Arrival + System Stress",
            "steps": [],
        }

        # Step 1: Urgent vehicle arrives
        self._update_step(1, "CRITICAL: Urgent brake fade detected")

        urgent_vehicle = self.DEMO_VEHICLES[7]  # BMW iX
        urgent_request = priority_queue.enqueue(
            vehicle_id=urgent_vehicle["id"],
            customer_id=urgent_vehicle["owner_id"],
            customer_name=urgent_vehicle["owner"],
            vehicle_name=urgent_vehicle["name"],
            geo_lat=urgent_vehicle["lat"],
            geo_lon=urgent_vehicle["lon"],
            failure_type="brake_fade",
            severity="critical",
            max_diagnosis_days=1,
            user_tier="vip",
        )

        # Preempt
        priority_queue.preempt(urgent_request)

        decision_log.log_preemption(
            urgent_request.request_id,
            urgent_vehicle["name"],
            "Critical brake fade requires immediate attention",
        )

        results["steps"].append(
            {
                "step": 1,
                "name": "Critical Arrival",
                "description": "VIP vehicle with critical brake fade preempts queue",
                "data": {
                    "vehicle": urgent_vehicle["name"],
                    "owner": urgent_vehicle["owner"],
                    "urgency_score": urgent_request.urgency_score,
                    "queue_position": 1,
                },
            }
        )

        # Step 2: Fast-track bidding
        self._update_step(2, "Fast-tracked bidding for urgent case")

        request = priority_queue.dequeue()
        bids = bidding_engine.generate_bids(request)
        winner = bidding_engine.select_winner(bids)

        if winner:
            decision_log.log_center_selected(
                request.request_id,
                winner.center_name,
                f"Selected for critical brake service: excellent skill match, {winner.load_percentage}% current load",
            )

        results["steps"].append(
            {
                "step": 2,
                "name": "Fast-Track Bidding",
                "description": f"Urgent case assigned to {winner.center_name if winner else 'N/A'}",
                "data": {
                    "bids_received": len(bids),
                    "winner": winner.center_name if winner else None,
                    "bid_details": bidding_engine.get_bid_history(request.request_id)[
                        :3
                    ],
                },
            }
        )

        # Step 3: Vehicle arrives, diagnosis
        self._update_step(3, "Vehicle arrived - master technician diagnosis")

        job_id = f"JOB-URGENT-{request.request_id.split('-')[1]}"

        # Simulate diagnosis feedback (slightly different from prediction)
        predicted = {"failure_type": "brake_fade", "severity": "critical"}
        actual = {
            "failure_type": "brake_degradation",
            "severity": "critical",
        }  # Slightly different

        record = diagnosis_feedback.log_feedback(
            job_id,
            request.vehicle_id,
            predicted,
            actual,
            notes="Brake pads more worn than predicted, rotors also need attention",
        )

        decision_log.log_diagnosis_feedback(
            job_id,
            record.similarity_score,
            "Actual diagnosis differs slightly - additional rotor work needed",
        )

        results["steps"].append(
            {
                "step": 3,
                "name": "Diagnosis Feedback",
                "description": f"Similarity score: {record.similarity_score * 100:.0f}%",
                "data": {
                    "predicted": predicted,
                    "actual": actual,
                    "similarity_score": record.similarity_score,
                    "notes": record.diagnosis_notes,
                },
            }
        )

        # Step 4: Task re-planning
        self._update_step(4, "Re-planning tasks based on actual diagnosis")

        new_tasks = task_planner.replan(job_id, {"actual_failure_type": "brake_fade"})
        task_planner.assign_technicians(
            new_tasks, winner.center_id if winner else "SC001"
        )

        decision_log.log(
            DecisionType.SCHEDULE_UPDATED,
            {"job_id": job_id, "task_count": len(new_tasks)},
            "Tasks regenerated based on actual diagnosis - added rotor inspection",
            entity_id=job_id,
            entity_type="job",
            impact="ETA extended by 30 minutes",
        )

        results["steps"].append(
            {
                "step": 4,
                "name": "Task Re-Planning",
                "description": f"Regenerated {len(new_tasks)} tasks for actual diagnosis",
                "data": {
                    "job_id": job_id,
                    "task_count": len(new_tasks),
                    "tasks": task_planner.get_job_tasks(job_id),
                },
            }
        )

        # Step 5: RCA/CAPA insights
        self._update_step(5, "Generating RCA/CAPA insights")

        rca_insights = diagnosis_feedback.get_rca_insights()
        capa_recommendations = diagnosis_feedback.get_capa_recommendations()

        results["steps"].append(
            {
                "step": 5,
                "name": "RCA & CAPA",
                "description": "Manufacturing insights generated from service data",
                "data": {
                    "rca_insights": rca_insights,
                    "capa_recommendations": capa_recommendations,
                },
            }
        )

        self._simulation_state["running"] = False
        results["final_state"] = self.get_state()

        return results

    def _update_step(self, step: int, description: str):
        """Update simulation state for dashboard tracking."""
        self._simulation_state["current_step"] = step
        self._simulation_state["step_description"] = description
        self._simulation_state["completed_steps"].append(
            {
                "step": step,
                "description": description,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )


# Singleton
demo_simulator = DemoSimulator()
