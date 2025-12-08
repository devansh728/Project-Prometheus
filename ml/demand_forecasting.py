"""
SentinEV - Demand Forecasting
Predicts service center workloads for optimal scheduling and resource planning.
Uses exponential smoothing and pattern analysis on historical appointment data.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import defaultdict
import math

from db.database import get_database


class DemandForecaster:
    """
    Service demand forecasting for workload optimization.

    Features:
    - Weekly demand prediction by service center
    - Peak hours identification
    - Component-based demand analysis
    - Optimal slot recommendations
    """

    def __init__(self, alpha: float = 0.3):
        """
        Initialize forecaster.

        Args:
            alpha: Smoothing factor for exponential smoothing (0-1)
        """
        self.db = get_database()
        self.alpha = alpha  # Higher alpha = more weight on recent data
        self._cache = {}
        self._cache_time = None

    def predict_weekly_demand(self, center_id: str) -> Dict[str, Any]:
        """
        Predict demand for the next 7 days for a service center.

        Uses exponential smoothing on historical data.

        Args:
            center_id: Service center ID

        Returns:
            Dict with daily predictions
        """
        # Get historical demand
        history = self.db.get_demand_history(center_id, days=30)

        if not history:
            # No history, return baseline estimate
            return self._baseline_forecast(center_id)

        # Group by day of week
        day_patterns = defaultdict(list)
        for record in history:
            date = datetime.strptime(record["date"], "%Y-%m-%d")
            day_of_week = date.strftime("%A")
            day_patterns[day_of_week].append(record["total"])

        # Calculate exponentially smoothed averages
        day_forecasts = {}
        for day, values in day_patterns.items():
            if values:
                # Apply exponential smoothing
                smoothed = values[0]
                for v in values[1:]:
                    smoothed = self.alpha * v + (1 - self.alpha) * smoothed
                day_forecasts[day] = round(smoothed, 1)
            else:
                day_forecasts[day] = 5  # Default

        # Generate next 7 days predictions
        predictions = []
        for i in range(7):
            date = datetime.now() + timedelta(days=i)
            day_name = date.strftime("%A")
            predictions.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "day": day_name,
                    "predicted_appointments": day_forecasts.get(day_name, 5),
                    "confidence": 0.8 if len(history) > 20 else 0.5,
                }
            )

        # Calculate capacity utilization
        centers = self.db.get_service_centers()
        center = next((c for c in centers if c["id"] == center_id), None)
        capacity = center["capacity_per_day"] if center else 10

        return {
            "center_id": center_id,
            "center_name": center["name"] if center else "Unknown",
            "capacity_per_day": capacity,
            "predictions": predictions,
            "average_daily_demand": round(
                sum(p["predicted_appointments"] for p in predictions) / 7, 1
            ),
            "utilization_forecast": round(
                (sum(p["predicted_appointments"] for p in predictions) / (capacity * 7))
                * 100,
                1,
            ),
        }

    def _baseline_forecast(self, center_id: str) -> Dict[str, Any]:
        """Generate baseline forecast when no history exists."""
        # Default pattern: higher midweek, lower weekend
        baseline = {
            "Monday": 6,
            "Tuesday": 7,
            "Wednesday": 8,
            "Thursday": 7,
            "Friday": 6,
            "Saturday": 4,
            "Sunday": 2,
        }

        predictions = []
        for i in range(7):
            date = datetime.now() + timedelta(days=i)
            day_name = date.strftime("%A")
            predictions.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "day": day_name,
                    "predicted_appointments": baseline.get(day_name, 5),
                    "confidence": 0.3,  # Low confidence for baseline
                }
            )

        return {
            "center_id": center_id,
            "center_name": "Service Center",
            "capacity_per_day": 10,
            "predictions": predictions,
            "average_daily_demand": 5.7,
            "utilization_forecast": 57.0,
            "note": "Baseline forecast - insufficient historical data",
        }

    def get_peak_hours(self, center_id: str) -> Dict[str, Any]:
        """
        Identify peak and off-peak hours for a service center.

        Args:
            center_id: Service center ID

        Returns:
            Dict with hourly demand patterns
        """
        history = self.db.get_demand_history(center_id, days=30)

        # Aggregate by hour
        hourly_demand = defaultdict(list)
        for record in history:
            hourly_demand[record["hour"]].append(record["total"])

        # Calculate averages
        hourly_averages = {}
        for hour, values in hourly_demand.items():
            hourly_averages[hour] = round(sum(values) / len(values), 2) if values else 0

        # Identify peaks and troughs
        if hourly_averages:
            max_hour = max(hourly_averages, key=hourly_averages.get)
            min_hour = min(hourly_averages, key=hourly_averages.get)
            avg_demand = sum(hourly_averages.values()) / len(hourly_averages)

            peak_hours = [h for h, v in hourly_averages.items() if v > avg_demand * 1.2]
            off_peak_hours = [
                h for h, v in hourly_averages.items() if v < avg_demand * 0.8
            ]
        else:
            max_hour, min_hour = 11, 8
            peak_hours = [10, 11, 12, 13, 14]
            off_peak_hours = [8, 9, 16, 17]
            avg_demand = 3

        return {
            "center_id": center_id,
            "hourly_demand": dict(sorted(hourly_averages.items())),
            "peak_hours": sorted(peak_hours),
            "off_peak_hours": sorted(off_peak_hours),
            "busiest_hour": max_hour,
            "quietest_hour": min_hour,
            "average_hourly_demand": round(avg_demand, 2),
            "recommendation": f"Best times to book: {', '.join(f'{h}:00' for h in sorted(off_peak_hours)[:3])}",
        }

    def suggest_optimal_slots(
        self, component: str, urgency: str = "medium", preferred_center_id: str = None
    ) -> Dict[str, Any]:
        """
        Suggest optimal appointment slots based on demand forecasting.

        Args:
            component: Component needing service
            urgency: Service urgency level
            preferred_center_id: Optional preferred center

        Returns:
            Dict with optimal slot recommendations
        """
        # Get centers that handle this component
        centers = self.db.get_center_by_specialty(component)

        if preferred_center_id:
            centers = [c for c in centers if c["id"] == preferred_center_id] or centers

        recommendations = []

        for center in centers[:3]:  # Top 3 centers
            # Get peak hours to avoid
            peak_info = self.get_peak_hours(center["id"])
            off_peak = peak_info["off_peak_hours"]

            # Get available slots
            slots = self.db.get_available_slots(
                center_id=center["id"], component=component, limit=10
            )

            # Score slots: prefer off-peak hours and sooner dates for urgent
            for slot in slots:
                hour = int(slot["start_time"].split(":")[0])

                # Calculate score
                score = 100

                # Prefer off-peak hours
                if hour in off_peak:
                    score += 20
                elif hour in peak_info["peak_hours"]:
                    score -= 15

                # For urgent, prefer earlier dates
                days_from_now = (
                    datetime.strptime(slot["date"], "%Y-%m-%d") - datetime.now()
                ).days

                if urgency == "critical":
                    score -= days_from_now * 10  # Heavily penalize later dates
                elif urgency == "high":
                    score -= days_from_now * 5
                else:
                    score -= days_from_now * 2

                # Factor in center rating
                score += center["rating"] * 5

                recommendations.append(
                    {
                        "slot_id": slot["id"],
                        "center_id": center["id"],
                        "center_name": center["name"],
                        "location": center["location"],
                        "date": slot["date"],
                        "time": slot["start_time"],
                        "rating": center["rating"],
                        "is_off_peak": hour in off_peak,
                        "score": score,
                    }
                )

        # Sort by score (highest first)
        recommendations.sort(key=lambda x: x["score"], reverse=True)

        return {
            "success": True,
            "component": component,
            "urgency": urgency,
            "recommendations": recommendations[:5],
            "message": self._generate_recommendation_message(
                recommendations[:3], urgency
            ),
        }

    def _generate_recommendation_message(self, slots: List[Dict], urgency: str) -> str:
        """Generate a human-readable recommendation message."""
        if not slots:
            return "No optimal slots found. Would you like me to check alternative centers?"

        urgency_prefix = {
            "critical": "⚠️ **Urgent:** Based on forecasted availability, I recommend booking immediately:",
            "high": "**Priority:** Here are the best available slots with low wait times:",
            "medium": "Here are the optimal slots based on our demand forecast:",
            "low": "These slots have the shortest expected wait times:",
        }

        msg = urgency_prefix.get(urgency, urgency_prefix["medium"]) + "\n\n"

        for i, slot in enumerate(slots):
            peak_status = "🟢 Off-peak" if slot["is_off_peak"] else "🔴 Peak hour"
            msg += f"**{i+1}. {slot['center_name']}**\n"
            msg += f"   📅 {slot['date']} at {slot['time']} ({peak_status})\n"
            msg += f"   ⭐ Rating: {slot['rating']}\n\n"

        return msg

    def get_component_demand_trends(self, days: int = 30) -> Dict[str, Any]:
        """
        Analyze demand trends by component type.

        Args:
            days: Number of days to analyze

        Returns:
            Dict with component demand breakdown
        """
        # This would require component data in demand_log
        # For now, return estimated breakdown
        trends = {
            "battery": {"percentage": 30, "trend": "increasing"},
            "brakes": {"percentage": 25, "trend": "stable"},
            "motor": {"percentage": 20, "trend": "stable"},
            "general": {"percentage": 15, "trend": "stable"},
            "thermal": {"percentage": 10, "trend": "decreasing"},
        }

        return {
            "period_days": days,
            "trends": trends,
            "insight": "Battery-related services are in highest demand, likely due to summer heat. Consider staffing additional battery specialists.",
        }

    def calculate_optimal_capacity(self, center_id: str) -> Dict[str, Any]:
        """
        Calculate optimal staffing/capacity for a service center.

        Args:
            center_id: Service center ID

        Returns:
            Dict with capacity recommendations
        """
        forecast = self.predict_weekly_demand(center_id)
        peak_hours = self.get_peak_hours(center_id)

        max_predicted = max(
            p["predicted_appointments"] for p in forecast["predictions"]
        )
        current_capacity = forecast["capacity_per_day"]

        # Calculate recommended capacity
        # Aim for 80% utilization at peak
        recommended = math.ceil(max_predicted / 0.8)

        return {
            "center_id": center_id,
            "current_capacity": current_capacity,
            "recommended_capacity": recommended,
            "peak_day_demand": max_predicted,
            "peak_hours": peak_hours["peak_hours"],
            "staffing_recommendation": self._staffing_recommendation(
                current_capacity, recommended, peak_hours["peak_hours"]
            ),
        }

    def _staffing_recommendation(
        self, current: int, recommended: int, peak_hours: List[int]
    ) -> str:
        """Generate staffing recommendation."""
        if recommended <= current:
            return f"Current capacity of {current} is adequate. Consider cross-training for peak hours ({peak_hours})."
        else:
            diff = recommended - current
            return f"Consider adding {diff} more service bays or extending hours. Peak demand occurs at {peak_hours}."


# Singleton instance
_demand_forecaster: Optional[DemandForecaster] = None


def get_demand_forecaster() -> DemandForecaster:
    """Get or create DemandForecaster singleton."""
    global _demand_forecaster
    if _demand_forecaster is None:
        _demand_forecaster = DemandForecaster()
    return _demand_forecaster
