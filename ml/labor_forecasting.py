"""
SentinEV - Labor Forecasting and Appointment Optimization
Predicts labor hours, staffing needs, and optimizes appointment scheduling.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import random
import math

from db.database import get_database


# Labor hours per component type
LABOR_HOURS = {
    "brakes": 2.0,
    "battery": 3.5,
    "motor": 4.0,
    "thermal": 2.5,
    "general": 1.5,
    "multiple": 5.0,
    "diagnostics": 1.0,
}

# Technician specialties and availability
TECHNICIANS = [
    {
        "id": "TECH-001",
        "name": "Alex Chen",
        "specialties": ["brakes", "general"],
        "capacity_hours": 8,
    },
    {
        "id": "TECH-002",
        "name": "Maria Garcia",
        "specialties": ["battery", "thermal"],
        "capacity_hours": 8,
    },
    {
        "id": "TECH-003",
        "name": "James Wilson",
        "specialties": ["motor", "battery"],
        "capacity_hours": 8,
    },
    {
        "id": "TECH-004",
        "name": "Sarah Kim",
        "specialties": ["brakes", "motor", "general"],
        "capacity_hours": 8,
    },
    {
        "id": "TECH-005",
        "name": "David Brown",
        "specialties": ["battery", "thermal", "diagnostics"],
        "capacity_hours": 6,
    },
]


class LaborForecaster:
    """
    Forecasts labor hours and staffing needs based on appointments.
    """

    def __init__(self):
        self.db = get_database()
        self.technicians = TECHNICIANS.copy()

    def predict_labor_hours(
        self, days: int = 7, center_id: str = None
    ) -> Dict[str, Any]:
        """
        Predict labor hours needed for the next N days.

        Args:
            days: Number of days to forecast
            center_id: Optional specific center

        Returns:
            Dict with daily labor predictions
        """
        predictions = []
        today = datetime.now().date()

        for i in range(days):
            date = today + timedelta(days=i)
            date_str = date.strftime("%Y-%m-%d")

            # Get appointments for this date
            appointments = self.db.get_appointments(status="scheduled")
            day_appointments = [
                a for a in appointments if a.get("scheduled_date") == date_str
            ]

            # Calculate scheduled labor hours
            scheduled_hours = sum(
                LABOR_HOURS.get(a.get("component", "general"), 1.5)
                for a in day_appointments
            )

            # Add predicted demand (from historical + forecast)
            demand = self.db.get_demand_history(center_id, days=1)
            baseline_demand = 5  # Default daily demand
            if demand:
                baseline_demand = sum(d.get("total", 0) for d in demand) / max(
                    len(demand), 1
                )

            # Predict additional walk-ins (20% of baseline)
            predicted_walkins = baseline_demand * 0.2 * LABOR_HOURS.get("general", 1.5)

            total_hours = scheduled_hours + predicted_walkins

            # Calculate staffing needs
            total_capacity = sum(t["capacity_hours"] for t in self.technicians)
            utilization = (
                (total_hours / total_capacity * 100) if total_capacity > 0 else 0
            )

            predictions.append(
                {
                    "date": date_str,
                    "day": date.strftime("%A"),
                    "scheduled_appointments": len(day_appointments),
                    "scheduled_hours": round(scheduled_hours, 1),
                    "predicted_walkins_hours": round(predicted_walkins, 1),
                    "total_labor_hours": round(total_hours, 1),
                    "capacity_hours": total_capacity,
                    "utilization_pct": round(utilization, 1),
                    "status": self._get_status(utilization),
                }
            )

        return {
            "forecast_days": days,
            "predictions": predictions,
            "summary": self._generate_summary(predictions),
            "staffing_recommendation": self.get_staffing_recommendation(predictions),
        }

    def _get_status(self, utilization: float) -> str:
        """Get utilization status."""
        if utilization < 50:
            return "low"
        elif utilization < 80:
            return "optimal"
        elif utilization < 100:
            return "busy"
        else:
            return "over_capacity"

    def _generate_summary(self, predictions: List[Dict]) -> Dict:
        """Generate forecast summary."""
        if not predictions:
            return {}

        total_hours = sum(p["total_labor_hours"] for p in predictions)
        avg_utilization = sum(p["utilization_pct"] for p in predictions) / len(
            predictions
        )
        busy_days = [p["date"] for p in predictions if p["utilization_pct"] > 80]
        low_days = [p["date"] for p in predictions if p["utilization_pct"] < 40]

        return {
            "total_labor_hours": round(total_hours, 1),
            "average_utilization": round(avg_utilization, 1),
            "busy_days": busy_days,
            "low_demand_days": low_days,
            "peak_day": (
                max(predictions, key=lambda x: x["total_labor_hours"])["date"]
                if predictions
                else None
            ),
        }

    def get_staffing_recommendation(
        self, predictions: List[Dict] = None
    ) -> Dict[str, Any]:
        """
        Recommend staffing adjustments based on forecast.
        """
        if not predictions:
            predictions = self.predict_labor_hours(7)["predictions"]

        recommendations = []

        for pred in predictions:
            util = pred["utilization_pct"]
            date = pred["date"]
            day = pred["day"]

            if util > 100:
                recommendations.append(
                    {
                        "date": date,
                        "day": day,
                        "action": "staff_up",
                        "message": f"⚠️ {day}: Over capacity ({util:.0f}%). Consider overtime or temp staff.",
                        "urgency": "high",
                    }
                )
            elif util > 85:
                recommendations.append(
                    {
                        "date": date,
                        "day": day,
                        "action": "monitor",
                        "message": f"📊 {day}: High demand ({util:.0f}%). Monitor for walk-ins.",
                        "urgency": "medium",
                    }
                )
            elif util < 30:
                recommendations.append(
                    {
                        "date": date,
                        "day": day,
                        "action": "staff_down",
                        "message": f"📉 {day}: Low demand ({util:.0f}%). Consider reduced hours.",
                        "urgency": "low",
                    }
                )

        return {
            "recommendations": recommendations,
            "summary": f"Found {len([r for r in recommendations if r['urgency'] == 'high'])} high-priority staffing alerts.",
        }

    def check_date_availability(
        self, date: str, component: str, center_id: str = None
    ) -> Dict[str, Any]:
        """
        Check availability for a specific date with detailed explanation.
        Used by chatbot for intelligent responses.

        Args:
            date: Date to check (YYYY-MM-DD)
            component: Component type
            center_id: Optional center filter

        Returns:
            Dict with availability info and explanations
        """
        # Get available slots for that date
        slots = self.db.get_available_slots(
            center_id=center_id, date=date, component=component
        )

        # Get all appointments for that date to calculate capacity
        all_appointments = self.db.get_appointments(status="scheduled")
        date_appointments = [
            a for a in all_appointments if a.get("scheduled_date") == date
        ]

        # Calculate utilization
        scheduled_hours = sum(
            LABOR_HOURS.get(a.get("component", "general"), 1.5)
            for a in date_appointments
        )
        total_capacity = sum(t["capacity_hours"] for t in self.technicians)
        utilization = (
            (scheduled_hours / total_capacity * 100) if total_capacity > 0 else 0
        )

        # Check specialist availability
        required_labor = LABOR_HOURS.get(component, 1.5)
        specialists = [t for t in self.technicians if component in t["specialties"]]

        # Also include general technicians who can handle any work
        general_techs = [t for t in self.technicians if "general" in t["specialties"]]
        all_available_techs = specialists if specialists else general_techs

        # Generate reasons/explanations
        reasons = []
        alternatives = []

        if utilization > 90:
            reasons.append(
                f"This day is {utilization:.0f}% booked - wait times may be longer"
            )

        if len(slots) == 0:
            reasons.append("No available slots remaining")
        elif len(slots) <= 2:
            reasons.append(f"Only {len(slots)} slot(s) remaining - high demand")

        # Note specialists but don't block availability
        if not specialists and general_techs:
            reasons.append(
                f"No dedicated {component} specialist, but general technicians available"
            )
        elif len(specialists) == 1:
            reasons.append(
                f"Only one {component} specialist on duty - limited capacity"
            )

        # Suggest alternatives if busy
        if utilization > 70 or len(slots) < 3:
            # Find better days
            for offset in range(1, 4):
                alt_date = (
                    datetime.strptime(date, "%Y-%m-%d") + timedelta(days=offset)
                ).strftime("%Y-%m-%d")
                alt_slots = self.db.get_available_slots(
                    center_id=center_id, date=alt_date, component=component
                )
                if len(alt_slots) > len(slots):
                    alternatives.append(
                        {
                            "date": alt_date,
                            "slots_available": len(alt_slots),
                            "reason": "More availability, shorter wait times",
                        }
                    )

        # Cost implication for delaying
        cost_warning = None
        if component in ["brakes", "motor"] and len(alternatives) > 0:
            cost_warning = "⚠️ Delaying brake/motor service may increase repair costs if issues worsen."

        # Available as long as slots exist and ANY technician can work
        is_available = len(slots) > 0 and len(all_available_techs) > 0

        return {
            "date": date,
            "available": is_available,
            "slots": slots,
            "slots_count": len(slots),
            "utilization_pct": round(utilization, 1),
            "specialists_available": (
                len(specialists) if specialists else len(general_techs)
            ),
            "specialist_names": [
                s["name"] for s in (specialists if specialists else general_techs)
            ],
            "reasons": reasons,
            "alternatives": alternatives,
            "cost_warning": cost_warning,
            "labor_hours_required": required_labor,
        }

    def suggest_optimal_date(
        self,
        component: str,
        urgency: str = "medium",
        preferred_date: str = None,
        days_range: int = 7,
    ) -> Dict[str, Any]:
        """
        Suggest optimal appointment date using scoring algorithm.

        Scoring factors:
        - Capacity utilization (lower = better)
        - Specialist availability
        - Distance from preferred date
        - Urgency (urgent = sooner is better)
        """
        today = datetime.now().date()
        candidates = []

        for i in range(days_range):
            date = (today + timedelta(days=i)).strftime("%Y-%m-%d")
            availability = self.check_date_availability(date, component)

            if not availability["available"]:
                continue

            # Calculate score (higher = better)
            score = 100

            # Utilization penalty (want lower utilization)
            score -= availability["utilization_pct"] * 0.5

            # Specialist bonus
            score += availability["specialists_available"] * 10

            # Urgency factor
            if urgency == "critical":
                score -= i * 20  # Heavy penalty for later dates
            elif urgency == "high":
                score -= i * 10
            else:
                score -= i * 2  # Light penalty

            # Preferred date bonus
            if preferred_date and date == preferred_date:
                score += 30

            candidates.append({"date": date, "score": round(score, 1), **availability})

        # Sort by score
        candidates.sort(key=lambda x: x["score"], reverse=True)

        return {
            "optimal_date": candidates[0] if candidates else None,
            "alternatives": candidates[1:4] if len(candidates) > 1 else [],
            "all_candidates": candidates,
        }


class AppointmentOptimizer:
    """
    Optimizes appointment scheduling using constraint satisfaction.
    Python-based "Tetris" scheduling algorithm.
    """

    def __init__(self):
        self.db = get_database()
        self.technicians = TECHNICIANS.copy()

    def optimize_schedule(self, date: str, center_id: str = None) -> Dict[str, Any]:
        """
        Optimize appointment schedule for a given date.
        Assigns technicians optimally based on:
        - Specialty match
        - Workload balance
        - Customer time preference

        Returns:
            Optimized schedule with technician assignments
        """
        # Get pending appointments for the date
        appointments = self.db.get_appointments(status="scheduled")
        date_appointments = [a for a in appointments if a.get("scheduled_date") == date]

        if not date_appointments:
            return {
                "date": date,
                "appointments": [],
                "message": "No appointments to optimize",
            }

        # Initialize technician workloads
        tech_workload = {t["id"]: 0 for t in self.technicians}
        tech_assignments = {t["id"]: [] for t in self.technicians}

        # Sort appointments by urgency (critical first) and time
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        date_appointments.sort(
            key=lambda x: (
                priority_order.get(x.get("urgency", "medium"), 2),
                x.get("scheduled_time", "12:00"),
            )
        )

        optimized = []
        unassigned = []

        for appt in date_appointments:
            component = appt.get("component", "general")
            labor_hours = LABOR_HOURS.get(component, 1.5)

            # Find best technician
            best_tech = self._find_best_technician(
                component, labor_hours, tech_workload
            )

            if best_tech:
                tech_workload[best_tech["id"]] += labor_hours
                tech_assignments[best_tech["id"]].append(appt["id"])

                optimized.append(
                    {
                        **appt,
                        "assigned_technician": best_tech["name"],
                        "technician_id": best_tech["id"],
                        "estimated_duration": f"{labor_hours} hours",
                    }
                )
            else:
                unassigned.append(
                    {
                        **appt,
                        "reason": f"No available technician with {component} expertise",
                    }
                )

        # Calculate optimization stats
        utilization = {}
        for tech in self.technicians:
            load = tech_workload[tech["id"]]
            cap = tech["capacity_hours"]
            utilization[tech["name"]] = {
                "hours_assigned": round(load, 1),
                "capacity": cap,
                "utilization_pct": round(load / cap * 100, 1) if cap > 0 else 0,
            }

        return {
            "date": date,
            "total_appointments": len(date_appointments),
            "optimized_appointments": optimized,
            "unassigned_appointments": unassigned,
            "technician_utilization": utilization,
            "workload_balance_score": self._calculate_balance_score(tech_workload),
            "message": f"Optimized {len(optimized)}/{len(date_appointments)} appointments",
        }

    def _find_best_technician(
        self, component: str, labor_hours: float, current_workload: Dict[str, float]
    ) -> Optional[Dict]:
        """Find best technician for an appointment."""
        candidates = []

        for tech in self.technicians:
            # Check capacity
            remaining_capacity = tech["capacity_hours"] - current_workload[tech["id"]]
            if remaining_capacity < labor_hours:
                continue

            # Calculate score
            score = 0

            # Specialty bonus
            if component in tech["specialties"]:
                score += 50

            # Load balance (prefer less loaded technicians)
            utilization = current_workload[tech["id"]] / tech["capacity_hours"]
            score += (1 - utilization) * 30

            candidates.append((tech, score))

        if not candidates:
            return None

        # Return highest scoring technician
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    def _calculate_balance_score(self, workload: Dict[str, float]) -> float:
        """Calculate how balanced the workload is (0-100, higher = more balanced)."""
        if not workload:
            return 100

        loads = list(workload.values())
        if max(loads) == 0:
            return 100

        # Calculate variance
        avg = sum(loads) / len(loads)
        variance = sum((l - avg) ** 2 for l in loads) / len(loads)
        std_dev = math.sqrt(variance)

        # Lower std_dev = more balanced = higher score
        max_std = avg  # Worst case: all load on one technician
        balance = max(0, 100 - (std_dev / max(max_std, 1) * 100))

        return round(balance, 1)

    def get_capacity_heatmap(self, days: int = 7) -> List[Dict]:
        """
        Generate capacity heatmap for visualization.
        Shows hourly capacity usage for next N days.
        """
        heatmap = []
        today = datetime.now().date()

        hours = range(8, 18)  # 8 AM to 6 PM

        for i in range(days):
            date = today + timedelta(days=i)
            date_str = date.strftime("%Y-%m-%d")

            day_data = {"date": date_str, "day": date.strftime("%A"), "hours": []}

            appointments = self.db.get_appointments(status="scheduled")
            date_appointments = [
                a for a in appointments if a.get("scheduled_date") == date_str
            ]

            for hour in hours:
                hour_str = f"{hour:02d}:00"
                # Count appointments starting at this hour
                hour_appts = [
                    a
                    for a in date_appointments
                    if a.get("scheduled_time", "").startswith(f"{hour:02d}:")
                ]

                load = len(hour_appts)
                max_capacity = len(self.technicians)

                day_data["hours"].append(
                    {
                        "hour": hour_str,
                        "appointments": load,
                        "capacity": max_capacity,
                        "intensity": min(load / max(max_capacity, 1), 1.0),
                    }
                )

            heatmap.append(day_data)

        return heatmap


# Singleton instances
_labor_forecaster: Optional[LaborForecaster] = None
_appointment_optimizer: Optional[AppointmentOptimizer] = None


def get_labor_forecaster() -> LaborForecaster:
    """Get or create LaborForecaster singleton."""
    global _labor_forecaster
    if _labor_forecaster is None:
        _labor_forecaster = LaborForecaster()
    return _labor_forecaster


def get_appointment_optimizer() -> AppointmentOptimizer:
    """Get or create AppointmentOptimizer singleton."""
    global _appointment_optimizer
    if _appointment_optimizer is None:
        _appointment_optimizer = AppointmentOptimizer()
    return _appointment_optimizer
