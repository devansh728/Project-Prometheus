"""
Synthetic Telematics Generator
Simulates real-time telemetry data from vehicles with configurable degradation patterns.
"""

import asyncio
import json
import random
from datetime import datetime
from typing import Dict, Any, Optional, AsyncGenerator
from pathlib import Path


class TelemetryGenerator:
    """Generates synthetic telemetry data with realistic degradation patterns."""

    def __init__(self, fleet_data_path: str = "data/fleet_seed.json"):
        self.fleet_data_path = Path(fleet_data_path)
        self.vehicles: Dict[str, Dict] = {}
        self.running = False
        self._load_fleet()

    def _load_fleet(self):
        """Load fleet configuration from JSON."""
        if self.fleet_data_path.exists():
            with open(self.fleet_data_path, "r") as f:
                data = json.load(f)
                for vehicle in data.get("vehicles", []):
                    self.vehicles[vehicle["id"]] = {
                        **vehicle,
                        "current_state": self._init_state(vehicle),
                    }

    def _init_state(self, vehicle: Dict) -> Dict[str, float]:
        """Initialize current sensor state from baseline config."""
        baseline = vehicle.get("baseline_config", {})
        return {key: config.get("mean", 0) for key, config in baseline.items()}

    def _apply_degradation(
        self, vehicle_id: str, state: Dict[str, float]
    ) -> Dict[str, float]:
        """Apply degradation effects based on vehicle configuration."""
        vehicle = self.vehicles.get(vehicle_id, {})
        deg_config = vehicle.get("degradation_config", {})
        rate = deg_config.get("rate", 0)
        component = deg_config.get("component")

        if rate > 0 and component:
            # Degradation effects by component
            effects = {
                "brake": {
                    "brake_pressure": -rate * 10,
                    "vibration_amplitude": rate * 0.5,
                },
                "battery": {"battery_voltage": -rate * 2, "engine_temp": rate * 5},
                "cooling": {"coolant_level": -rate * 0.5, "engine_temp": rate * 8},
            }

            for sensor, delta in effects.get(component, {}).items():
                if sensor in state:
                    state[sensor] = max(0, state[sensor] + delta)

        return state

    def _add_noise(self, vehicle_id: str, state: Dict[str, float]) -> Dict[str, float]:
        """Add realistic sensor noise based on baseline std deviations."""
        vehicle = self.vehicles.get(vehicle_id, {})
        baseline = vehicle.get("baseline_config", {})

        noisy_state = {}
        for sensor, value in state.items():
            std = baseline.get(sensor, {}).get("std", 0.1)
            noise = random.gauss(0, std)
            noisy_state[sensor] = round(value + noise, 3)

        return noisy_state

    def generate_telemetry(self, vehicle_id: str) -> Optional[Dict[str, Any]]:
        """Generate a single telemetry reading for a vehicle."""
        if vehicle_id not in self.vehicles:
            return None

        vehicle = self.vehicles[vehicle_id]

        # Update state with degradation
        current = vehicle["current_state"]
        current = self._apply_degradation(vehicle_id, current)
        vehicle["current_state"] = current

        # Add noise for this reading
        noisy = self._add_noise(vehicle_id, current)

        # Calculate simple anomaly score (deviations from baseline)
        baseline = vehicle.get("baseline_config", {})
        anomaly_score = 0.0
        for sensor, value in noisy.items():
            mean = baseline.get(sensor, {}).get("mean", value)
            std = baseline.get(sensor, {}).get("std", 1)
            if std > 0:
                z_score = abs(value - mean) / std
                anomaly_score += min(z_score / 3, 1)  # Normalize to 0-1 range

        anomaly_score = min(anomaly_score / len(noisy), 1.0)  # Average

        return {
            "vehicle_id": vehicle_id,
            "timestamp": datetime.utcnow().isoformat(),
            "sensors": noisy,
            "anomaly_score": round(anomaly_score, 4),
            "gps": {
                "lat": 19.0760 + random.uniform(-0.1, 0.1),
                "lon": 72.8777 + random.uniform(-0.1, 0.1),
            },
        }

    async def stream_telemetry(
        self, vehicle_id: str, interval_seconds: float = 1.0
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream telemetry data for a vehicle."""
        self.running = True
        while self.running:
            telemetry = self.generate_telemetry(vehicle_id)
            if telemetry:
                yield telemetry
            await asyncio.sleep(interval_seconds)

    async def stream_all_vehicles(
        self, interval_seconds: float = 2.0
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream telemetry from all vehicles in round-robin fashion."""
        self.running = True
        vehicle_ids = list(self.vehicles.keys())
        idx = 0

        while self.running and vehicle_ids:
            vehicle_id = vehicle_ids[idx]
            telemetry = self.generate_telemetry(vehicle_id)
            if telemetry:
                yield telemetry

            idx = (idx + 1) % len(vehicle_ids)
            await asyncio.sleep(interval_seconds / len(vehicle_ids))

    def stop(self):
        """Stop streaming."""
        self.running = False


# Singleton for use across the app
telemetry_generator = TelemetryGenerator()
