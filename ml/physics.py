"""
SentinelEY - Physics-Based Vehicle Simulator
Generates realistic, physics-driven telematics data for EVs
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Literal
import json
from pathlib import Path


@dataclass
class VehicleSpecs:
    """Vehicle physical specifications."""

    battery_capacity_kwh: float = 75.0
    battery_max_temp: float = 60.0
    battery_internal_resistance_mohm: float = 50.0
    battery_thermal_mass: float = 150000.0  # J/°C
    motor_max_power_kw: float = 200.0
    motor_max_rpm: int = 12000
    vehicle_mass_kg: float = 2000.0
    frontal_area_m2: float = 2.3
    drag_coefficient: float = 0.24
    rolling_resistance: float = 0.01
    regen_max_power_kw: float = 75.0


@dataclass
class DriverProfile:
    """Driver behavior profile parameters."""

    name: str
    acceleration_multiplier: float = 1.0
    top_speed_tendency: float = 1.0
    braking_intensity: float = 1.0
    power_draw_multiplier: float = 1.0
    wear_accumulation_multiplier: float = 1.0
    regen_efficiency_range: Tuple[float, float] = (0.75, 0.85)
    jerk_threshold: float = 5.0  # m/s³


@dataclass
class WeatherCondition:
    """Environmental condition parameters."""

    name: str
    ambient_temp_c: float = 25.0
    battery_temp_offset_c: float = 0.0
    hvac_power_draw_kw: float = 1.0
    air_density_adjustment: float = 1.0


# Predefined profiles
DRIVER_PROFILES = {
    "aggressive": DriverProfile(
        name="aggressive",
        acceleration_multiplier=1.3,
        top_speed_tendency=1.2,
        braking_intensity=1.4,
        power_draw_multiplier=1.15,
        wear_accumulation_multiplier=3.0,
        regen_efficiency_range=(0.5, 0.7),
        jerk_threshold=8.0,
    ),
    "eco": DriverProfile(
        name="eco",
        acceleration_multiplier=0.7,
        top_speed_tendency=0.85,
        braking_intensity=0.6,
        power_draw_multiplier=0.9,
        wear_accumulation_multiplier=0.5,
        regen_efficiency_range=(0.9, 1.0),
        jerk_threshold=3.0,
    ),
    "normal": DriverProfile(
        name="normal",
        acceleration_multiplier=1.0,
        top_speed_tendency=1.0,
        braking_intensity=1.0,
        power_draw_multiplier=1.0,
        wear_accumulation_multiplier=1.0,
        regen_efficiency_range=(0.75, 0.85),
        jerk_threshold=5.0,
    ),
}

WEATHER_CONDITIONS = {
    "hot": WeatherCondition(
        name="hot",
        ambient_temp_c=40.0,
        battery_temp_offset_c=5.0,
        hvac_power_draw_kw=3.0,
        air_density_adjustment=0.95,
    ),
    "cold": WeatherCondition(
        name="cold",
        ambient_temp_c=5.0,
        battery_temp_offset_c=-10.0,
        hvac_power_draw_kw=5.0,
        air_density_adjustment=1.1,
    ),
    "moderate": WeatherCondition(
        name="moderate",
        ambient_temp_c=25.0,
        battery_temp_offset_c=0.0,
        hvac_power_draw_kw=1.0,
        air_density_adjustment=1.0,
    ),
}


class VehicleSimulator:
    """
    Physics-based EV telemetry generator.

    Generates realistic telematics data incorporating:
    - Physical power draw equations
    - Battery thermal dynamics
    - Wear accumulation physics
    - Driver behavior patterns
    - Environmental effects
    """

    # Physical constants
    AIR_DENSITY_BASE = 1.225  # kg/m³ at sea level, 15°C
    GRAVITY = 9.81  # m/s²

    def __init__(
        self,
        vehicle_id: str,
        driver_profile: Literal["aggressive", "eco", "normal"] = "normal",
        weather_condition: Literal["hot", "cold", "moderate"] = "moderate",
        vehicle_specs: Optional[VehicleSpecs] = None,
    ):
        """
        Initialize the vehicle simulator.

        Args:
            vehicle_id: Unique vehicle identifier
            driver_profile: Driver behavior type
            weather_condition: Environmental conditions
            vehicle_specs: Optional custom vehicle specifications
        """
        self.vehicle_id = vehicle_id
        self.driver = DRIVER_PROFILES.get(driver_profile, DRIVER_PROFILES["normal"])
        self.weather = WEATHER_CONDITIONS.get(
            weather_condition, WEATHER_CONDITIONS["moderate"]
        )
        self.specs = vehicle_specs or VehicleSpecs()

        # State variables
        self._current_time = datetime.now()
        self._speed = 0.0  # km/h
        self._prev_speed = 0.0  # for jerk calculation
        self._battery_temp = (
            self.weather.ambient_temp_c + self.weather.battery_temp_offset_c
        )
        self._battery_soc = 100.0  # State of charge %
        self._wear_index = 0.0
        self._cumulative_energy_kwh = 0.0
        self._trip_distance_km = 0.0

        # Time-based patterns for realistic simulation
        self._driving_pattern = self._create_driving_pattern()

    def _create_driving_pattern(self) -> Dict[int, float]:
        """
        Create a realistic driving pattern based on hour of day.
        Returns dict mapping hour -> activity level (0-1).
        """
        # Typical driving patterns: peaks in morning and evening commute
        pattern = {}
        for hour in range(24):
            if 7 <= hour <= 9:  # Morning commute
                pattern[hour] = 0.9
            elif 17 <= hour <= 19:  # Evening commute
                pattern[hour] = 0.85
            elif 10 <= hour <= 16:  # Midday activity
                pattern[hour] = 0.5
            elif 20 <= hour <= 22:  # Evening errands
                pattern[hour] = 0.4
            else:  # Night/early morning
                pattern[hour] = 0.1
        return pattern

    def _calculate_air_drag_power(self, speed_kmh: float) -> float:
        """
        Calculate power required to overcome air drag.

        Physics: P = 0.5 × ρ × A × Cd × v³

        Args:
            speed_kmh: Vehicle speed in km/h

        Returns:
            Power in kW
        """
        speed_ms = speed_kmh / 3.6  # Convert to m/s
        air_density = self.AIR_DENSITY_BASE * self.weather.air_density_adjustment

        # Power for air drag: P = 0.5 × ρ × A × Cd × v³
        power_watts = (
            0.5
            * air_density
            * self.specs.frontal_area_m2
            * self.specs.drag_coefficient
            * (speed_ms**3)
        )

        return power_watts / 1000  # Convert to kW

    def _calculate_rolling_resistance_power(self, speed_kmh: float) -> float:
        """
        Calculate power required to overcome rolling resistance.

        Physics: P = Crr × m × g × v

        Args:
            speed_kmh: Vehicle speed in km/h

        Returns:
            Power in kW
        """
        speed_ms = speed_kmh / 3.6

        # Power for rolling resistance: P = Crr × m × g × v
        power_watts = (
            self.specs.rolling_resistance
            * self.specs.vehicle_mass_kg
            * self.GRAVITY
            * speed_ms
        )

        return power_watts / 1000

    def _calculate_total_power_draw(
        self, speed_kmh: float, acceleration_ms2: float = 0.0
    ) -> float:
        """
        Calculate total power draw including propulsion and auxiliary.

        Args:
            speed_kmh: Current speed in km/h
            acceleration_ms2: Current acceleration in m/s²

        Returns:
            Total power draw in kW
        """
        # Base power components
        drag_power = self._calculate_air_drag_power(speed_kmh)
        rolling_power = self._calculate_rolling_resistance_power(speed_kmh)

        # Acceleration power: P = m × a × v
        speed_ms = speed_kmh / 3.6
        accel_power = (self.specs.vehicle_mass_kg * acceleration_ms2 * speed_ms) / 1000

        # Total propulsion power
        propulsion_power = drag_power + rolling_power + max(0, accel_power)

        # Apply driver behavior multiplier (aggressive = more power)
        propulsion_power *= self.driver.power_draw_multiplier

        # Add HVAC power
        total_power = propulsion_power + self.weather.hvac_power_draw_kw

        # Motor efficiency loss (typically 8-10%)
        motor_efficiency = 0.92
        total_power /= motor_efficiency

        return max(0, total_power)

    def _calculate_battery_temperature(
        self, power_kw: float, dt_seconds: float
    ) -> float:
        """
        Calculate battery temperature with thermal inertia.

        Physics: ΔT = (P × R × dt) / thermal_mass
        Temperature tends toward: T_ambient + (P × R_factor)

        Args:
            power_kw: Current power draw in kW
            dt_seconds: Time step in seconds

        Returns:
            New battery temperature in °C
        """
        # Heat generated: Q = I²R, approximated as proportional to power
        # Using internal resistance as heat generation factor
        heat_generation_factor = self.specs.battery_internal_resistance_mohm / 1000
        heat_generated_watts = (power_kw * 1000) * heat_generation_factor * 0.1

        # Target temperature based on power level
        target_temp = self.weather.ambient_temp_c + (power_kw * 0.15)

        # Thermal inertia: temperature changes slowly
        thermal_time_constant = 300  # seconds to reach ~63% of target
        alpha = dt_seconds / thermal_time_constant

        # Heat generation effect
        temp_rise_from_heat = (
            heat_generated_watts * dt_seconds
        ) / self.specs.battery_thermal_mass

        # Natural cooling toward ambient
        temp_diff = self._battery_temp - self.weather.ambient_temp_c
        cooling_rate = 0.01  # per second
        natural_cooling = temp_diff * cooling_rate * dt_seconds

        # New temperature
        new_temp = self._battery_temp + temp_rise_from_heat - natural_cooling

        # Ensure bounds
        new_temp = max(
            self.weather.ambient_temp_c - 5,
            min(self.specs.battery_max_temp + 10, new_temp),
        )

        return new_temp

    def _calculate_wear_index(
        self, jerk_ms3: float, speed_kmh: float, dt_seconds: float
    ) -> float:
        """
        Calculate wear accumulation.

        Physics: W = Jerk × Speed × Time × driver_multiplier

        Args:
            jerk_ms3: Rate of acceleration change in m/s³
            speed_kmh: Current speed in km/h
            dt_seconds: Time step in seconds

        Returns:
            New cumulative wear index
        """
        speed_ms = speed_kmh / 3.6

        # Wear accumulation formula
        wear_increment = abs(jerk_ms3) * speed_ms * dt_seconds
        wear_increment *= self.driver.wear_accumulation_multiplier

        # Normalize to reasonable scale
        wear_increment *= 0.0001

        return self._wear_index + wear_increment

    def _calculate_regen_efficiency(self, deceleration_ms2: float) -> float:
        """
        Calculate regenerative braking efficiency.

        Physics: R = decel_smoothness × SoC_factor × driver_factor

        Args:
            deceleration_ms2: Deceleration rate (negative acceleration)

        Returns:
            Regen efficiency as fraction (0-1)
        """
        # Base efficiency from driver profile
        min_eff, max_eff = self.driver.regen_efficiency_range

        # Deceleration smoothness factor (smoother = more efficient)
        decel_abs = abs(deceleration_ms2)
        if decel_abs < 1.0:
            smoothness_factor = 1.0
        elif decel_abs < 3.0:
            smoothness_factor = 0.9
        else:
            smoothness_factor = 0.7

        # SoC factor (can't regen well if battery is full)
        soc_factor = min(1.0, (100 - self._battery_soc) / 20 + 0.5)

        # Calculate final efficiency
        base_efficiency = np.random.uniform(min_eff, max_eff)
        final_efficiency = base_efficiency * smoothness_factor * soc_factor

        return min(1.0, max(0.0, final_efficiency))

    def _generate_speed_profile(self, hour: int) -> float:
        """
        Generate realistic speed based on time of day and driver profile.

        Args:
            hour: Hour of day (0-23)

        Returns:
            Speed in km/h
        """
        activity_level = self._driving_pattern[hour]

        if activity_level < 0.15:
            # Parked
            return 0.0

        # Base speed range
        base_speed = np.random.choice(
            [
                0,  # Stopped
                30,  # City slow
                50,  # City normal
                80,  # Highway
                100,  # Fast highway
            ],
            p=[0.1, 0.2, 0.35, 0.25, 0.1],
        )

        # Apply driver tendency
        speed = base_speed * self.driver.top_speed_tendency

        # Add some variation
        speed += np.random.normal(0, 5)

        # Apply activity level (less travel at night)
        if np.random.random() > activity_level:
            speed = 0

        return max(0, min(140, speed))

    def step(self, dt_seconds: float = 1.0) -> Dict:
        """
        Generate the next time step of telemetry data.

        Args:
            dt_seconds: Time step duration in seconds

        Returns:
            Dictionary with all telemetry values
        """
        # Update time
        self._current_time += timedelta(seconds=dt_seconds)
        hour = self._current_time.hour

        # Generate new speed
        target_speed = self._generate_speed_profile(hour)

        # Smooth speed transition (acceleration)
        max_speed_change = (
            20 * dt_seconds * self.driver.acceleration_multiplier
        )  # km/h per second
        speed_diff = target_speed - self._speed
        speed_change = np.clip(speed_diff, -max_speed_change * 1.5, max_speed_change)

        self._prev_speed = self._speed
        self._speed = max(0, self._speed + speed_change)

        # Calculate acceleration and jerk
        acceleration_kmh_s = speed_change / dt_seconds
        acceleration_ms2 = acceleration_kmh_s / 3.6

        # Simple jerk approximation
        prev_accel = (
            (self._speed - self._prev_speed) / 3.6 / dt_seconds if dt_seconds > 0 else 0
        )
        jerk_ms3 = (
            abs(acceleration_ms2 - prev_accel) / dt_seconds if dt_seconds > 0 else 0
        )

        # Calculate power draw
        power_draw = self._calculate_total_power_draw(self._speed, acceleration_ms2)

        # Calculate regen power if decelerating
        regen_power = 0.0
        regen_efficiency = 0.0
        if acceleration_ms2 < -0.5:  # Significant deceleration
            regen_efficiency = self._calculate_regen_efficiency(acceleration_ms2)
            # Kinetic energy recovery potential
            speed_ms = self._speed / 3.6
            potential_regen_kw = (
                0.5 * self.specs.vehicle_mass_kg * (speed_ms**2) / 1000 * 0.1
            )
            regen_power = min(
                self.specs.regen_max_power_kw, potential_regen_kw * regen_efficiency
            )

        # Net power consumption
        net_power = power_draw - regen_power

        # Update battery temperature
        self._battery_temp = self._calculate_battery_temperature(net_power, dt_seconds)

        # Update wear index
        self._wear_index = self._calculate_wear_index(jerk_ms3, self._speed, dt_seconds)

        # Update SoC (simple linear model)
        energy_used_kwh = net_power * dt_seconds / 3600
        self._cumulative_energy_kwh += max(0, energy_used_kwh)
        soc_change = (energy_used_kwh / self.specs.battery_capacity_kwh) * 100
        self._battery_soc = max(0, min(100, self._battery_soc - soc_change))

        # Update trip distance
        distance_km = self._speed * dt_seconds / 3600
        self._trip_distance_km += distance_km

        # Calculate efficiency (Wh/km)
        efficiency_wh_km = (
            (energy_used_kwh * 1000 / distance_km) if distance_km > 0.001 else 0
        )

        return {
            "timestamp": self._current_time.isoformat(),
            "vehicle_id": self.vehicle_id,
            "speed_kmh": round(self._speed, 1),
            "acceleration_ms2": round(acceleration_ms2, 2),
            "jerk_ms3": round(jerk_ms3, 3),
            "power_draw_kw": round(power_draw, 2),
            "regen_power_kw": round(regen_power, 2),
            "regen_efficiency": round(regen_efficiency, 2),
            "net_power_kw": round(net_power, 2),
            "battery_temp_c": round(self._battery_temp, 1),
            "battery_soc_percent": round(self._battery_soc, 1),
            "wear_index": round(self._wear_index, 4),
            "efficiency_wh_km": round(efficiency_wh_km, 1),
            "ambient_temp_c": self.weather.ambient_temp_c,
            "driver_profile": self.driver.name,
            "weather_condition": self.weather.name,
            "trip_distance_km": round(self._trip_distance_km, 2),
            "cumulative_energy_kwh": round(self._cumulative_energy_kwh, 2),
        }

    def generate_history(
        self, days: int = 60, interval_seconds: int = 3600
    ) -> pd.DataFrame:
        """
        Generate historical telemetry data.

        Args:
            days: Number of days of history to generate
            interval_seconds: Time interval between readings (default: hourly)

        Returns:
            DataFrame with telemetry history
        """
        # Reset to starting point in the past
        self._current_time = datetime.now() - timedelta(days=days)
        self._speed = 0.0
        self._prev_speed = 0.0
        self._battery_temp = (
            self.weather.ambient_temp_c + self.weather.battery_temp_offset_c
        )
        self._battery_soc = 100.0
        self._wear_index = 0.0
        self._cumulative_energy_kwh = 0.0
        self._trip_distance_km = 0.0

        records = []
        total_steps = int(days * 24 * 3600 / interval_seconds)

        for _ in range(total_steps):
            record = self.step(dt_seconds=interval_seconds)
            records.append(record)

            # Simulate charging (reset SoC periodically)
            if self._battery_soc < 20 and np.random.random() < 0.3:
                self._battery_soc = np.random.uniform(80, 100)

        return pd.DataFrame(records)

    def reset(self):
        """Reset simulator state to defaults."""
        self._current_time = datetime.now()
        self._speed = 0.0
        self._prev_speed = 0.0
        self._battery_temp = (
            self.weather.ambient_temp_c + self.weather.battery_temp_offset_c
        )
        self._battery_soc = 100.0
        self._wear_index = 0.0
        self._cumulative_energy_kwh = 0.0
        self._trip_distance_km = 0.0

    def inject_fault(self, fault_type: str, severity: float = 1.0) -> Dict:
        """
        Inject a simulated fault condition.

        Args:
            fault_type: Type of fault (overheat, cell_imbalance, inverter, etc.)
            severity: Severity multiplier (0.5 = mild, 1.0 = normal, 2.0 = severe)

        Returns:
            Modified telemetry reading with fault
        """
        # Get base reading
        reading = self.step()

        if fault_type == "overheat":
            # Battery overheating
            reading["battery_temp_c"] = min(70, 55 + (15 * severity))
            reading["power_draw_kw"] *= 1.2

        elif fault_type == "cell_imbalance":
            # Simulated cell imbalance (affects SoC readings)
            reading["battery_soc_percent"] = max(
                0, reading["battery_soc_percent"] - (10 * severity)
            )
            reading["cell_voltage_imbalance_mv"] = 80 * severity

        elif fault_type == "inverter":
            # Inverter issues (power fluctuations)
            reading["power_draw_kw"] *= 1 + 0.3 * severity * np.random.uniform(-1, 1)
            reading["inverter_temp_c"] = 85 + (15 * severity)

        elif fault_type == "motor_resolver":
            # Resolver drift (efficiency loss)
            reading["efficiency_wh_km"] *= 1 + 0.15 * severity
            reading["resolver_offset_deg"] = 5 * severity

        elif fault_type == "brake_drag":
            # Stuck brake caliper
            reading["power_draw_kw"] *= 1 + 0.2 * severity
            reading["efficiency_wh_km"] *= 1 + 0.2 * severity
            reading["brake_temp_diff_c"] = 40 * severity

        elif fault_type == "coolant_low":
            # Low coolant (thermal issues)
            reading["battery_temp_c"] += 10 * severity
            reading["coolant_level_percent"] = 70 - (20 * severity)

        reading["fault_injected"] = fault_type
        reading["fault_severity"] = severity

        return reading


def load_vehicle_specs_from_manual(
    manual_path: str, model: str = "Model X"
) -> VehicleSpecs:
    """
    Load vehicle specifications from the vehicle manual JSON.

    Args:
        manual_path: Path to vehicle_manual.json
        model: Vehicle model name

    Returns:
        VehicleSpecs instance
    """
    with open(manual_path, "r") as f:
        manual = json.load(f)

    # Get model-specific info
    model_info = manual.get("vehicle_models", {}).get(model, {})
    components = manual.get("components", {})
    physics = manual.get("physics_parameters", {})

    battery = components.get("battery", {})
    motor = components.get("motor", {})

    return VehicleSpecs(
        battery_capacity_kwh=model_info.get("battery_capacity_kwh", 75.0),
        battery_max_temp=battery.get("max_temp_c", 60.0),
        battery_internal_resistance_mohm=battery.get("internal_resistance_mohm", 50.0),
        battery_thermal_mass=battery.get("thermal_mass_j_per_c", 150000.0),
        motor_max_power_kw=model_info.get("motor_power_kw", 200.0),
        motor_max_rpm=motor.get("max_rpm", 12000),
        vehicle_mass_kg=physics.get("vehicle_mass_kg", 2000.0),
        frontal_area_m2=physics.get("frontal_area_m2", 2.3),
        drag_coefficient=physics.get("drag_coefficient", 0.24),
        rolling_resistance=physics.get("rolling_resistance", 0.01),
        regen_max_power_kw=components.get("brakes", {}).get("regen_max_power_kw", 75.0),
    )


if __name__ == "__main__":
    # Demo: Generate sample data
    print("🚗 SentinelEY Physics-Based Vehicle Simulator")
    print("=" * 50)

    # Create simulator
    simulator = VehicleSimulator(
        vehicle_id="DEMO-001", driver_profile="aggressive", weather_condition="hot"
    )

    print(f"\n📊 Vehicle: {simulator.vehicle_id}")
    print(f"👤 Driver: {simulator.driver.name}")
    print(f"🌡️ Weather: {simulator.weather.name}")

    # Generate a few sample readings
    print("\n📈 Sample Real-time Readings:")
    print("-" * 40)
    for i in range(5):
        reading = simulator.step(dt_seconds=60)  # 1-minute intervals
        print(
            f"  Speed: {reading['speed_kmh']:5.1f} km/h | "
            f"Power: {reading['power_draw_kw']:5.1f} kW | "
            f"Batt Temp: {reading['battery_temp_c']:4.1f}°C | "
            f"SoC: {reading['battery_soc_percent']:5.1f}%"
        )

    # Generate history
    print("\n📅 Generating 7 days of history...")
    history_df = simulator.generate_history(days=7)
    print(f"   Generated {len(history_df)} records")
    print(f"   Columns: {list(history_df.columns)}")

    # Show summary stats
    print("\n📊 Summary Statistics:")
    print(f"   Avg Speed: {history_df['speed_kmh'].mean():.1f} km/h")
    print(f"   Avg Power Draw: {history_df['power_draw_kw'].mean():.1f} kW")
    print(f"   Max Battery Temp: {history_df['battery_temp_c'].max():.1f}°C")
    print(f"   Total Distance: {history_df['trip_distance_km'].iloc[-1]:.1f} km")
