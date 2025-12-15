"""
SentinEV - Enhanced Physics-Based Telemetry Generator
Generates realistic per-vehicle datasets with driver behavior patterns
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Literal
from dataclasses import dataclass, field
import random


@dataclass
class VehicleConfig:
    """Configuration for a specific vehicle instance."""

    vehicle_id: str
    driver_profile: Literal["aggressive", "eco", "normal"] = "normal"
    vehicle_age_years: float = 1.0
    total_mileage_km: float = 20000.0
    battery_soh_pct: float = 98.0

    # Vehicle specs (can be customized)
    battery_capacity_kwh: float = 75.0
    motor_max_power_kw: float = 200.0
    vehicle_mass_kg: float = 2100.0
    frontal_area_m2: float = 2.35
    drag_coefficient: float = 0.24
    rolling_resistance: float = 0.008


class EnhancedTelemetryGenerator:
    """
    Generates realistic EV telemetry data using physics-based equations.

    Physics equations implemented:
    1. Power Draw: P = P_drag + P_rolling + P_accel + P_aux
       - P_drag = 0.5 × ρ × A × Cd × v³
       - P_rolling = Crr × m × g × v
       - P_accel = m × a × v

    2. Battery Temperature: T = T_ambient + (P × R × time) / thermal_mass
       - With thermal inertia modeling

    3. Wear Index: W = ∫ (|jerk| × speed × driver_factor) dt

    4. Regen Efficiency: R = base_efficiency × soc_factor × decel_smoothness
    """

    DRIVER_PROFILES = {
        "aggressive": {
            "speed_multiplier": 1.25,
            "acceleration_multiplier": 1.5,
            "braking_intensity": 1.4,
            "jerk_multiplier": 2.0,
            "wear_multiplier": 2.5,
            "regen_efficiency_range": (0.45, 0.65),
            "speed_variance": 15.0,
        },
        "normal": {
            "speed_multiplier": 1.0,
            "acceleration_multiplier": 1.0,
            "braking_intensity": 1.0,
            "jerk_multiplier": 1.0,
            "wear_multiplier": 1.0,
            "regen_efficiency_range": (0.70, 0.85),
            "speed_variance": 8.0,
        },
        "eco": {
            "speed_multiplier": 0.85,
            "acceleration_multiplier": 0.7,
            "braking_intensity": 0.6,
            "jerk_multiplier": 0.5,
            "wear_multiplier": 0.5,
            "regen_efficiency_range": (0.85, 0.95),
            "speed_variance": 5.0,
        },
    }

    WEATHER_CONDITIONS = {
        "hot": {
            "ambient_temp_c": 38.0,
            "hvac_draw_kw": 3.5,
            "battery_temp_offset": 8.0,
        },
        "moderate": {
            "ambient_temp_c": 22.0,
            "hvac_draw_kw": 1.0,
            "battery_temp_offset": 0.0,
        },
        "cold": {
            "ambient_temp_c": -5.0,
            "hvac_draw_kw": 4.0,
            "battery_temp_offset": -15.0,
        },
    }

    # Physical constants
    AIR_DENSITY = 1.225  # kg/m³
    GRAVITY = 9.81  # m/s²
    BATTERY_THERMAL_MASS = 150000  # J/°C
    BATTERY_INTERNAL_RESISTANCE = 0.05  # Ohms (simplified)

    def __init__(self, config: VehicleConfig, seed: Optional[int] = None):
        """Initialize generator with vehicle configuration."""
        self.config = config
        self.profile = self.DRIVER_PROFILES[config.driver_profile]

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # State variables
        self.current_speed_kmh = 0.0
        self.current_acceleration = 0.0
        self.previous_acceleration = 0.0
        self.battery_temp_c = 25.0
        self.battery_soc = 0.8
        self.cumulative_wear_index = 0.0
        self.motor_temp_c = 35.0
        self.inverter_temp_c = 40.0
        self.brake_temp_c = 25.0
        self.coolant_temp_c = 30.0
        self.odometer_km = config.total_mileage_km

        # Fault injection state
        self.active_faults: Dict[str, Dict] = {}

    def _get_driving_activity(self, hour: int, day_of_week: int) -> float:
        """
        Get driving activity level based on time.
        Returns activity level 0-1.
        """
        # Weekend vs weekday patterns
        is_weekend = day_of_week >= 5

        if is_weekend:
            # Weekend: late morning peaks, evening activity
            if 10 <= hour <= 12:
                return 0.7
            elif 14 <= hour <= 18:
                return 0.8
            elif 20 <= hour <= 22:
                return 0.5
            elif 0 <= hour <= 7:
                return 0.0
            else:
                return 0.3
        else:
            # Weekday: commute peaks
            if 7 <= hour <= 9:  # Morning commute
                return 0.9
            elif 12 <= hour <= 13:  # Lunch
                return 0.4
            elif 17 <= hour <= 19:  # Evening commute
                return 0.95
            elif 0 <= hour <= 6:
                return 0.0
            else:
                return 0.3

    def _generate_speed_profile(self, activity_level: float) -> float:
        """Generate realistic speed based on activity and driver profile."""
        if activity_level < 0.1:
            return 0.0

        # Base speeds for different scenarios
        base_speeds = {
            "highway": 110.0,
            "urban": 45.0,
            "suburban": 70.0,
        }

        # Probability of each scenario based on activity
        if activity_level > 0.7:
            # High activity = likely highway/commute
            scenario_probs = [0.5, 0.2, 0.3]
        else:
            # Lower activity = more urban driving
            scenario_probs = [0.1, 0.5, 0.4]

        scenario = np.random.choice(["highway", "urban", "suburban"], p=scenario_probs)
        base_speed = base_speeds[scenario]

        # Apply driver profile
        target_speed = base_speed * self.profile["speed_multiplier"]

        # Add realistic variance
        variance = self.profile["speed_variance"]
        speed = target_speed + np.random.normal(0, variance)

        return max(0, min(speed, 180))  # Cap at reasonable limits

    def _calculate_power_draw(self, speed_kmh: float, acceleration_ms2: float) -> float:
        """
        Calculate total power draw using physics equations.

        Power components:
        1. Air drag: P = 0.5 × ρ × A × Cd × v³
        2. Rolling resistance: P = Crr × m × g × v
        3. Acceleration: P = m × a × v
        4. Auxiliary loads (HVAC, etc.)
        """
        speed_ms = speed_kmh / 3.6

        # Air drag power (kW)
        p_drag = (
            0.5
            * self.AIR_DENSITY
            * self.config.frontal_area_m2
            * self.config.drag_coefficient
            * (speed_ms**3)
        ) / 1000

        # Rolling resistance power (kW)
        p_rolling = (
            self.config.rolling_resistance
            * self.config.vehicle_mass_kg
            * self.GRAVITY
            * speed_ms
        ) / 1000

        # Acceleration power (kW)
        p_accel = (self.config.vehicle_mass_kg * acceleration_ms2 * speed_ms) / 1000

        # Auxiliary power (HVAC, lights, etc.) - varies with conditions
        p_aux = 1.5  # Base auxiliary load

        # Apply driver profile multiplier for aggressive driving
        profile_multiplier = 1.0 + (self.profile["acceleration_multiplier"] - 1.0) * 0.2

        total_power = max(
            0, (p_drag + p_rolling + p_accel) * profile_multiplier + p_aux
        )

        # Apply any active faults
        if "high_power_draw" in self.active_faults:
            total_power *= self.active_faults["high_power_draw"].get("multiplier", 1.5)

        return min(total_power, self.config.motor_max_power_kw)

    def _calculate_battery_temperature(self, power_kw: float, dt_hours: float) -> float:
        """
        Calculate battery temperature with thermal inertia.

        T_new = T_current + (Q_in - Q_out) / thermal_mass
        where:
        - Q_in = power × internal_resistance × time (heat generation)
        - Q_out = cooling_rate × (T - T_ambient)
        """
        # Get ambient temperature
        ambient = 25.0  # Default moderate

        # Heat generation from I²R losses (simplified)
        current_estimate = power_kw * 1000 / 400  # Estimate current at 400V
        heat_generation = (current_estimate**2) * self.BATTERY_INTERNAL_RESISTANCE

        # Cooling rate (W/°C) - based on cooling system effectiveness
        cooling_rate = 50.0

        # Temperature change
        dt_seconds = dt_hours * 3600
        q_in = heat_generation * dt_seconds
        q_out = cooling_rate * (self.battery_temp_c - ambient) * dt_seconds

        delta_t = (q_in - q_out) / self.BATTERY_THERMAL_MASS

        new_temp = self.battery_temp_c + delta_t

        # Apply fault effects
        if "overheat" in self.active_faults:
            new_temp += self.active_faults["overheat"].get("temp_increase", 15.0)
        if "coolant_low" in self.active_faults:
            new_temp += 8.0  # Reduced cooling efficiency

        return np.clip(new_temp, -20, 80)

    def _calculate_wear_index(
        self, jerk_ms3: float, speed_kmh: float, dt_hours: float
    ) -> float:
        """
        Calculate wear accumulation.

        Wear = ∫ |jerk| × speed × driver_multiplier dt
        """
        wear_increment = (
            abs(jerk_ms3) * speed_kmh * self.profile["wear_multiplier"] * dt_hours
        )
        return self.cumulative_wear_index + wear_increment

    def _calculate_regen_efficiency(self, deceleration_ms2: float) -> float:
        """
        Calculate regenerative braking efficiency.

        Efficiency = base × soc_factor × smoothness_factor
        """
        if deceleration_ms2 >= 0:
            return 0.0

        # Base efficiency from driver profile
        min_eff, max_eff = self.profile["regen_efficiency_range"]

        # SOC factor - less efficient at high SOC
        soc_factor = 1.0 - max(0, (self.battery_soc - 0.8)) * 2

        # Smoothness factor - harsh braking reduces regen
        smoothness = 1.0 - min(1.0, abs(deceleration_ms2) / 5.0) * 0.3

        efficiency = ((min_eff + max_eff) / 2) * soc_factor * smoothness

        return np.clip(efficiency, 0, 1)

    def _update_motor_temperature(self, power_kw: float, dt_hours: float) -> float:
        """Update motor temperature based on load."""
        # Motor heats up with power usage, cools toward ambient
        heat_rate = power_kw * 0.03  # °C per kW-hour
        cool_rate = 0.5  # Natural cooling rate

        target_temp = 35 + power_kw * 0.4
        delta = (target_temp - self.motor_temp_c) * (1 - np.exp(-dt_hours * 2))

        new_temp = self.motor_temp_c + delta

        # Apply faults
        if "motor_resolver" in self.active_faults:
            new_temp += 5.0  # Friction from misalignment

        return np.clip(new_temp, 20, 150)

    def _update_brake_temperature(
        self,
        speed_kmh: float,
        deceleration_ms2: float,
        regen_efficiency: float,
        dt_hours: float,
    ) -> float:
        """Update brake temperature based on mechanical braking."""
        # Mechanical braking = total braking - regen
        if deceleration_ms2 < 0:
            total_braking_power = (
                abs(deceleration_ms2 * self.config.vehicle_mass_kg * speed_kmh / 3.6)
                / 1000
            )  # kW
            mechanical_braking = total_braking_power * (1 - regen_efficiency)
            heat_rate = mechanical_braking * 0.8  # Heat generation rate
        else:
            heat_rate = 0

        # Cooling
        cool_rate = (self.brake_temp_c - 25) * 0.1 * dt_hours * 3600 / 60

        new_temp = self.brake_temp_c + heat_rate - cool_rate

        # Apply faults
        if "brake_drag" in self.active_faults:
            new_temp += 50.0 * self.active_faults["brake_drag"].get("severity", 1.0)

        return np.clip(new_temp, 20, 600)

    def inject_fault(
        self,
        fault_type: str,
        severity: float = 1.0,
        duration_hours: Optional[float] = None,
    ) -> None:
        """
        Inject a fault into the simulation.

        Args:
            fault_type: Type of fault (overheat, cell_imbalance, inverter,
                       motor_resolver, brake_drag, coolant_low)
            severity: Fault severity multiplier (0.5-2.0)
            duration_hours: How long the fault lasts (None = permanent)
        """
        fault_configs = {
            "overheat": {"temp_increase": 15.0 * severity},
            "cell_imbalance": {"voltage_diff": 0.3 * severity},
            "inverter": {
                "power_reduction": 0.3 * severity,
                "temp_increase": 10 * severity,
            },
            "motor_resolver": {"drift_deg": 3.0 * severity, "efficiency_loss": 0.1},
            "brake_drag": {"severity": severity},
            "coolant_low": {"level_pct": 15.0 / severity},
            "high_power_draw": {"multiplier": 1.0 + 0.5 * severity},
        }

        if fault_type in fault_configs:
            self.active_faults[fault_type] = {
                **fault_configs[fault_type],
                "injected_at": datetime.now(),
                "duration_hours": duration_hours,
            }

    def clear_fault(self, fault_type: str) -> None:
        """Remove an active fault."""
        if fault_type in self.active_faults:
            del self.active_faults[fault_type]

    def clear_all_faults(self) -> None:
        """Remove all active faults."""
        self.active_faults.clear()

    def step(self, dt_hours: float = 1 / 3600) -> Dict:
        """
        Generate next time step of telemetry data.

        Args:
            dt_hours: Time step in hours (default: 1 second)

        Returns:
            Dictionary with all telemetry values
        """
        current_time = datetime.now()
        hour = current_time.hour
        day_of_week = current_time.weekday()

        # Get activity level
        activity = self._get_driving_activity(hour, day_of_week)

        # Generate speed
        target_speed = self._generate_speed_profile(activity)

        # Smooth speed transition
        speed_change_rate = 20.0 * dt_hours * 3600  # Max change per step
        speed_diff = target_speed - self.current_speed_kmh
        speed_change = np.clip(speed_diff, -speed_change_rate, speed_change_rate)
        self.current_speed_kmh += speed_change

        # Calculate acceleration and jerk
        speed_ms = self.current_speed_kmh / 3.6
        prev_speed_ms = (self.current_speed_kmh - speed_change) / 3.6
        self.previous_acceleration = self.current_acceleration
        self.current_acceleration = (
            (speed_ms - prev_speed_ms) / (dt_hours * 3600) if dt_hours > 0 else 0
        )
        jerk = (
            (self.current_acceleration - self.previous_acceleration) / (dt_hours * 3600)
            if dt_hours > 0
            else 0
        )

        # Apply driver profile jerk multiplier
        jerk *= self.profile["jerk_multiplier"]

        # Calculate power draw
        power_draw = self._calculate_power_draw(
            self.current_speed_kmh, self.current_acceleration
        )

        # Calculate regen efficiency
        regen_efficiency = self._calculate_regen_efficiency(self.current_acceleration)

        # Net power (negative when regenerating)
        if self.current_acceleration < 0 and self.current_speed_kmh > 5:
            regen_power = power_draw * regen_efficiency * 0.5  # Energy recovered
            net_power = power_draw - regen_power
        else:
            net_power = power_draw

        # Update SOC
        self.battery_soc -= (net_power * dt_hours) / self.config.battery_capacity_kwh
        self.battery_soc = np.clip(self.battery_soc, 0.1, 1.0)

        # Update temperatures
        self.battery_temp_c = self._calculate_battery_temperature(power_draw, dt_hours)
        self.motor_temp_c = self._update_motor_temperature(power_draw, dt_hours)
        self.inverter_temp_c = self.motor_temp_c * 0.9 + 5  # Correlated with motor
        self.brake_temp_c = self._update_brake_temperature(
            self.current_speed_kmh,
            self.current_acceleration,
            regen_efficiency,
            dt_hours,
        )
        self.coolant_temp_c = (self.battery_temp_c + self.motor_temp_c) / 2 - 5

        # Update wear
        self.cumulative_wear_index = self._calculate_wear_index(
            jerk, self.current_speed_kmh, dt_hours
        )

        # Update odometer
        self.odometer_km += self.current_speed_kmh * dt_hours

        # Calculate cell voltage (simplified - normally read from BMS)
        base_cell_voltage = 3.7 + (self.battery_soc - 0.5) * 0.8
        cell_voltage_diff = 0.05  # Normal variance
        if "cell_imbalance" in self.active_faults:
            cell_voltage_diff = self.active_faults["cell_imbalance"]["voltage_diff"]

        tire_circumference = 2 * np.pi * 0.3
        wheel_rpm = (self.current_speed_kmh * 1000 / 60) / tire_circumference
        motor_rpm = wheel_rpm * 9.0

        charging_state = "discharging"
        if self.current_speed_kmh == 0 and self.current_acceleration == 0:
            if 0 <= datetime.now().hour <= 6: # Simplified logic
                charging_state = "charging"
            else:
                charging_state = "idle"
        active_dtcs = []
        dtc_map = {
            "cell_imbalance": "P0A80", # Replace Hybrid Battery Pack
            "overheat": "P0A78",       # Drive Motor Inverter Performance
            "coolant_low": "P0A93",    # Inverter Cooling System Performance
            "motor_resolver": "P0A90", # Drive Motor Performance
        }
        for fault in self.active_faults:
            if fault in dtc_map:
                active_dtcs.append(dtc_map[fault])

        # Build telemetry packet
        telemetry = {
            "timestamp": int(datetime.now().timestamp()),
            "vehicle_id": self.config.vehicle_id,
            "driver_profile": self.config.driver_profile,
            # Motion data
            "speed_kmh": round(self.current_speed_kmh, 2),
            "acceleration_ms2": round(self.current_acceleration, 3),
            "jerk_ms3": round(jerk, 3),
            # Power data
            "power_draw_kw": round(power_draw, 2),
            "net_power_kw": round(net_power, 2),
            "regen_power_kw": (
                round(power_draw - net_power, 2) if net_power < power_draw else 0
            ),
            "regen_efficiency": round(regen_efficiency, 3),
            # Battery data
            "battery_soc_pct": round(self.battery_soc * 100, 1),
            "battery_temp_c": round(self.battery_temp_c, 1),
            "cell_voltage_avg_v": round(base_cell_voltage, 3),
            "cell_voltage_diff_v": round(cell_voltage_diff, 3),
            # Thermal data
            "motor_temp_c": round(self.motor_temp_c, 1),
            "inverter_temp_c": round(self.inverter_temp_c, 1),
            "brake_temp_c": round(self.brake_temp_c, 1),
            "coolant_temp_c": round(self.coolant_temp_c, 1),
            "ambient_temp_c": 25.0,  # Could be parameterized

            "motor_rpm": round(motor_rpm, 0),
            "charging_state": charging_state,
            "dtc_codes": json.dumps(active_dtcs),
            # Wear data
            "wear_index": round(self.cumulative_wear_index, 4),
            "odometer_km": round(self.odometer_km, 1),
            # Fault indicators
            "active_faults": list(self.active_faults.keys()),
            "fault_count": len(self.active_faults),
            "label_is_anomaly": 1 if len(self.active_faults) > 0 else 0,
            "label_failure_type": list(self.active_faults.keys())[0] if self.active_faults else "none"
        }

        return telemetry

    def generate_history(
        self, days: int = 60, interval_hours: float = 1.0
    ) -> pd.DataFrame:
        """
        Generate historical telemetry data.

        Args:
            days: Number of days of history
            interval_hours: Time between readings (default: hourly)

        Returns:
            DataFrame with telemetry history
        """
        # Reset state for clean history generation
        start_time = datetime.now() - timedelta(days=days)
        records = []

        total_steps = int(days * 24 / interval_hours)

        for i in range(total_steps):
            current_time = start_time + timedelta(hours=i * interval_hours)
            hour = current_time.hour
            day_of_week = current_time.weekday()

            # Get activity level
            activity = self._get_driving_activity(hour, day_of_week)

            # Generate telemetry
            target_speed = self._generate_speed_profile(activity)

            # Smooth speed transition
            speed_change_rate = 20.0 * interval_hours  # Max change per step
            speed_diff = target_speed - self.current_speed_kmh
            speed_change = np.clip(speed_diff, -speed_change_rate, speed_change_rate)
            self.current_speed_kmh += speed_change

            # Calculate metrics
            speed_ms = self.current_speed_kmh / 3.6
            prev_speed_ms = (self.current_speed_kmh - speed_change) / 3.6
            self.previous_acceleration = self.current_acceleration
            self.current_acceleration = (
                (speed_ms - prev_speed_ms) / (interval_hours * 3600)
                if interval_hours > 0
                else 0
            )
            jerk = (
                (self.current_acceleration - self.previous_acceleration)
                / (interval_hours * 3600)
                if interval_hours > 0
                else 0
            )
            jerk *= self.profile["jerk_multiplier"]

            power_draw = self._calculate_power_draw(
                self.current_speed_kmh, self.current_acceleration
            )
            regen_efficiency = self._calculate_regen_efficiency(
                self.current_acceleration
            )

            if self.current_acceleration < 0 and self.current_speed_kmh > 5:
                regen_power = power_draw * regen_efficiency * 0.5
                net_power = power_draw - regen_power
            else:
                net_power = power_draw

            # Update state
            self.battery_soc -= (
                net_power * interval_hours
            ) / self.config.battery_capacity_kwh
            self.battery_soc = np.clip(self.battery_soc, 0.1, 1.0)

            # Recharge overnight
            if 0 <= hour <= 6 and self.battery_soc < 0.8:
                self.battery_soc = min(0.9, self.battery_soc + 0.1)

            self.battery_temp_c = self._calculate_battery_temperature(
                power_draw, interval_hours
            )
            self.motor_temp_c = self._update_motor_temperature(
                power_draw, interval_hours
            )
            self.inverter_temp_c = self.motor_temp_c * 0.9 + 5
            self.brake_temp_c = self._update_brake_temperature(
                self.current_speed_kmh,
                self.current_acceleration,
                regen_efficiency,
                interval_hours,
            )
            self.coolant_temp_c = (self.battery_temp_c + self.motor_temp_c) / 2 - 5
            self.cumulative_wear_index = self._calculate_wear_index(
                jerk, self.current_speed_kmh, interval_hours
            )
            self.odometer_km += self.current_speed_kmh * interval_hours

            base_cell_voltage = 3.7 + (self.battery_soc - 0.5) * 0.8

            records.append(
                {
                    "timestamp": current_time.isoformat(),
                    "vehicle_id": self.config.vehicle_id,
                    "driver_profile": self.config.driver_profile,
                    "speed_kmh": round(self.current_speed_kmh, 2),
                    "acceleration_ms2": round(self.current_acceleration, 3),
                    "jerk_ms3": round(jerk, 3),
                    "power_draw_kw": round(power_draw, 2),
                    "net_power_kw": round(net_power, 2),
                    "regen_efficiency": round(regen_efficiency, 3),
                    "battery_soc_pct": round(self.battery_soc * 100, 1),
                    "battery_temp_c": round(self.battery_temp_c, 1),
                    "cell_voltage_avg_v": round(base_cell_voltage, 3),
                    "motor_temp_c": round(self.motor_temp_c, 1),
                    "inverter_temp_c": round(self.inverter_temp_c, 1),
                    "brake_temp_c": round(self.brake_temp_c, 1),
                    "coolant_temp_c": round(self.coolant_temp_c, 1),
                    "ambient_temp_c": 25.0 + np.random.normal(0, 3),
                    "wear_index": round(self.cumulative_wear_index, 4),
                    "odometer_km": round(self.odometer_km, 1),
                }
            )

        return pd.DataFrame(records)


def generate_fleet_datasets(
    output_dir: str = "data/telemetry", num_vehicles: int = 10, days: int = 60
) -> List[str]:
    """
    Generate telemetry datasets for a fleet of vehicles.

    Args:
        output_dir: Directory to save CSV files
        num_vehicles: Number of vehicles to generate
        days: Days of history per vehicle

    Returns:
        List of generated file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Define vehicle configurations with variety
    vehicle_configs = [
        VehicleConfig("VIN_001", "aggressive", 0.5, 15000, 99.0),
        VehicleConfig("VIN_002", "eco", 1.0, 25000, 97.0),
        VehicleConfig("VIN_003", "normal", 1.5, 35000, 95.0),
        VehicleConfig("VIN_004", "aggressive", 2.0, 50000, 93.0),
        VehicleConfig("VIN_005", "eco", 0.3, 8000, 99.5),
        VehicleConfig("VIN_006", "normal", 2.5, 60000, 91.0),
        VehicleConfig("VIN_007", "aggressive", 1.0, 30000, 96.0),
        VehicleConfig("VIN_008", "normal", 1.8, 45000, 94.0),
        VehicleConfig("VIN_009", "eco", 3.0, 75000, 88.0),
        VehicleConfig("VIN_010", "normal", 0.8, 20000, 98.0),
    ]

    generated_files = []

    for i, config in enumerate(vehicle_configs[:num_vehicles]):
        print(
            f"Generating data for {config.vehicle_id} ({config.driver_profile} driver)..."
        )

        generator = EnhancedTelemetryGenerator(config, seed=42 + i)
        df = generator.generate_history(days=days, interval_hours=1.0)

        filename = f"telemetry_{config.vehicle_id}.csv"
        filepath = output_path / filename
        df.to_csv(filepath, index=False)

        generated_files.append(str(filepath))
        print(f"  ✓ Saved {len(df)} records to {filename}")

    return generated_files


if __name__ == "__main__":
    # Generate fleet datasets
    print("🚗 SentinEV Fleet Telemetry Generator")
    print("=" * 50)

    print("Generating High-Res (1s) data for Anomaly Models...")
    generate_fleet_datasets(
        output_dir="data/telemetry/high_res", 
        num_vehicles=10, 
        days=3,
        interval_hours=1/3600
    )

    print("\n" + "=" * 50)
    print(f"✅ Generated {len(files)} vehicle datasets")
