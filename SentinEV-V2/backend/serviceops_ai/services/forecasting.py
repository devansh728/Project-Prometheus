"""
Labour & Inventory Forecasting for ServiceOps AI
Predicts workload and parts demand for proactive planning.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import random

from serviceops_ai.services.workforce import workforce_manager
from serviceops_ai.services.inventory import inventory_manager


@dataclass
class DailyForecast:
    """Forecast for a single day."""

    date: str
    predicted_jobs: int
    predicted_hours: float
    utilization_ratio: float
    overload_risk: str  # "low", "medium", "high"


class LabourForecast:
    """
    Forecasts labour demand and utilization for service centers.
    """

    # Base job distribution by day of week (Mon=0, Sun=6)
    BASE_DISTRIBUTION = [0.18, 0.17, 0.16, 0.15, 0.18, 0.12, 0.04]

    def __init__(self):
        self._queue_size = 0  # Updated from priority queue
        self._historical_avg_jobs = 6  # Average jobs per day per center

    def set_queue_size(self, size: int):
        """Update with current queue size."""
        self._queue_size = size

    def forecast_workload(
        self,
        center_id: str,
        days: int = 7,
    ) -> List[DailyForecast]:
        """
        Forecast daily workload for next N days.
        """
        forecasts = []
        workload = workforce_manager.get_workload_summary(center_id)

        total_capacity = workload.get("total_capacity", 12)
        current_utilization = workload.get("utilization", 0.5)

        today = datetime.utcnow().date()

        for i in range(days):
            forecast_date = today + timedelta(days=i)
            day_of_week = forecast_date.weekday()

            # Base prediction from historical + queue
            base_jobs = (
                self._historical_avg_jobs * self.BASE_DISTRIBUTION[day_of_week] * 7
            )
            queue_factor = 1 + (self._queue_size * 0.05)  # Queue adds jobs

            predicted_jobs = int(
                base_jobs * queue_factor * (0.9 + random.random() * 0.2)
            )
            predicted_hours = predicted_jobs * 1.5  # Avg 1.5 hours per job

            if total_capacity > 0:
                utilization = min(
                    1.0, predicted_hours / (total_capacity * 2)
                )  # 2 hours per capacity slot
            else:
                utilization = 1.0  # Max utilization if no capacity

            # Determine overload risk
            if utilization > 0.85:
                risk = "high"
            elif utilization > 0.65:
                risk = "medium"
            else:
                risk = "low"

            forecasts.append(
                DailyForecast(
                    date=forecast_date.isoformat(),
                    predicted_jobs=predicted_jobs,
                    predicted_hours=round(predicted_hours, 1),
                    utilization_ratio=round(utilization, 2),
                    overload_risk=risk,
                )
            )

        return forecasts

    def get_utilization_chart(self, center_id: str, days: int = 7) -> List[Dict]:
        """Get data formatted for charting."""
        forecasts = self.forecast_workload(center_id, days)
        return [
            {
                "date": f.date,
                "utilization": f.utilization_ratio * 100,
                "risk": f.overload_risk,
            }
            for f in forecasts
        ]

    def get_overload_alerts(self) -> List[Dict]:
        """Get centers at risk of overload."""
        alerts = []

        # Check all known centers
        for center_id in ["SC001", "SC002", "SC003", "SC004"]:
            forecasts = self.forecast_workload(center_id, 3)
            for f in forecasts:
                if f.overload_risk == "high":
                    alerts.append(
                        {
                            "center_id": center_id,
                            "date": f.date,
                            "utilization": f.utilization_ratio,
                            "risk": "high",
                            "message": f"High workload expected on {f.date}",
                        }
                    )

        return alerts


class InventoryForecast:
    """
    Forecasts parts demand for proactive ordering.
    """

    # Parts consumed per failure type
    PARTS_PER_FAILURE = {
        "brake_degradation": {"brake_pads": 1, "brake_fluid": 0.5},
        "brake_fade": {"brake_pads": 1, "brake_fluid": 1},
        "battery_degradation": {"battery_module": 0.3, "bms_controller": 0.1},
        "cooling_degradation": {"coolant": 2, "thermostat": 0.5},
    }

    def __init__(self):
        self._predicted_failures: Dict[str, int] = {}  # failure_type -> count

    def set_predicted_failures(self, failures: Dict[str, int]):
        """Update with failure predictions from queue."""
        self._predicted_failures = failures

    def forecast_demand(
        self,
        center_id: str,
        days: int = 7,
    ) -> Dict[str, Any]:
        """
        Forecast parts demand for next N days.
        """
        demand = {}

        for failure_type, count in self._predicted_failures.items():
            parts = self.PARTS_PER_FAILURE.get(failure_type, {})
            for part, qty_per_job in parts.items():
                if part not in demand:
                    demand[part] = 0
                demand[part] += int(count * qty_per_job * (days / 7))

        # Add baseline demand
        demand["brake_pads"] = demand.get("brake_pads", 0) + 5
        demand["coolant"] = demand.get("coolant", 0) + 3

        # Get current inventory
        alerts = inventory_manager.get_reorder_alerts(center_id)

        return {
            "center_id": center_id,
            "forecast_days": days,
            "predicted_demand": demand,
            "reorder_alerts": alerts,
        }

    def get_reorder_recommendations(self, center_id: str) -> List[Dict]:
        """Get proactive reorder recommendations."""
        forecast = self.forecast_demand(center_id, 7)
        recommendations = []

        for part, predicted in forecast["predicted_demand"].items():
            # Check if we need to reorder
            if predicted > 3:  # Threshold
                recommendations.append(
                    {
                        "part": part,
                        "predicted_demand": predicted,
                        "action": "REORDER",
                        "urgency": "medium" if predicted < 10 else "high",
                        "message": f"Projected demand of {predicted} units in next 7 days",
                    }
                )

        # Add existing alerts
        for alert in forecast["reorder_alerts"]:
            recommendations.append(
                {
                    "part": alert["part_number"],
                    "current_qty": alert["current_quantity"],
                    "action": (
                        "URGENT_REORDER"
                        if alert["severity"] == "critical"
                        else "REORDER"
                    ),
                    "urgency": alert["severity"],
                    "message": f"Current stock at {alert['current_quantity']}, reorder point {alert['reorder_point']}",
                }
            )

        return recommendations


# Singletons
labour_forecast = LabourForecast()
inventory_forecast = InventoryForecast()
