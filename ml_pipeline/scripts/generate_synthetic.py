"""
EV Telemetry Synthetic Data Generator
=====================================
Physics-based generator for realistic EV telemetry data with controlled failure injection.

Usage:
    python generate_synthetic.py --config ../config.json --output ../synthetic_data
"""

import json
import os
import random
import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import argparse


@dataclass
class VehicleProfile:
    """Configuration for a specific EV vehicle."""

    vehicle_id: str
    vehicle_type: str
    battery_kwh: float
    motor_kw: float
    ambient_temp: float
    max_speed_kph: float = 180.0
    max_motor_rpm: float = 15000.0
    battery_voltage_nominal: float = 400.0
    efficiency_wh_km: float = 150.0


@dataclass
class FailureEvent:
    """Represents a scheduled failure event."""

    failure_type: str
    start_ts: int
    failure_ts: int
    severity: int
    dtc_codes: List[str]
    lead_time_hours: float


@dataclass
class DrivingSegment:
    """Represents a driving or idle segment."""

    segment_type: str  # 'driving', 'idle', 'charging'
    start_ts: int
    duration_sec: int
    avg_speed: float = 0.0
    driving_style: str = "normal"  # 'normal', 'aggressive', 'eco'


class EVPhysicsModel:
    """Physics model for realistic EV telemetry generation."""

    def __init__(self, profile: VehicleProfile, ambient_temp: float = 25.0):
        self.profile = profile
        self.ambient_temp = ambient_temp

        # State variables
        self.battery_soc = random.uniform(70, 95)  # Start with decent charge
        self.motor_temp = ambient_temp + random.uniform(0, 5)
        self.inverter_temp = ambient_temp + random.uniform(0, 3)
        self.battery_temp = ambient_temp + random.uniform(-2, 5)
        self.battery_cell_delta = random.uniform(0.01, 0.05)
        self.odometer = random.uniform(5000, 50000)

        # Tire pressures (normal range 32-36 psi)
        self.tire_pressures = [random.uniform(33, 35) for _ in range(4)]

    def compute_battery_voltage(self, soc: float, current: float) -> float:
        """Compute battery voltage based on SOC and current draw."""
        # Voltage sag under load
        nominal = self.profile.battery_voltage_nominal
        soc_factor = 0.85 + 0.15 * (soc / 100)  # Voltage drops with SOC
        load_sag = abs(current) * 0.001  # Voltage sag under current
        return nominal * soc_factor - load_sag

    def compute_power_consumption(
        self, speed: float, accel: float, hvac_on: bool = False
    ) -> float:
        """Compute power consumption in kW based on driving conditions."""
        if speed < 1:
            # Idle consumption
            base_power = 0.5
        else:
            # Rolling resistance + aerodynamic drag
            rolling = 0.01 * speed * self.profile.motor_kw / 200
            aero = 0.00005 * speed**2 * self.profile.motor_kw / 200
            accel_power = max(0, accel * 0.3 * self.profile.motor_kw / 100)
            base_power = rolling + aero + accel_power

        # HVAC adds 2-5 kW
        hvac_power = random.uniform(2, 5) if hvac_on else 0

        return base_power + hvac_power

    def compute_regen_power(self, speed: float, brake_pct: float) -> float:
        """Compute regenerative braking power recovery."""
        if speed < 5 or brake_pct < 5:
            return 0
        # Regen efficiency typically 60-80%
        max_regen = min(self.profile.motor_kw * 0.3, speed * 0.5)
        return max_regen * (brake_pct / 100) * random.uniform(0.6, 0.8)

    def update_thermal_state(self, power_kw: float, speed: float, dt: float = 1.0):
        """Update thermal states based on power draw and cooling."""
        # Motor heating
        motor_heat = power_kw * 0.05  # 5% loss as heat
        motor_cooling = (self.motor_temp - self.ambient_temp) * 0.02 * (1 + speed / 50)
        self.motor_temp += (motor_heat - motor_cooling) * dt / 10
        self.motor_temp = np.clip(self.motor_temp, self.ambient_temp - 5, 150)

        # Inverter heating
        inverter_heat = power_kw * 0.02
        inverter_cooling = (self.inverter_temp - self.ambient_temp) * 0.03
        self.inverter_temp += (inverter_heat - inverter_cooling) * dt / 10
        self.inverter_temp = np.clip(self.inverter_temp, self.ambient_temp - 5, 120)

        # Battery thermal management (active cooling)
        battery_heat = power_kw * 0.01
        battery_cooling = (self.battery_temp - 25) * 0.05  # Target 25C
        self.battery_temp += (battery_heat - battery_cooling) * dt / 20
        self.battery_temp = np.clip(self.battery_temp, -30, 60)


class FailureInjector:
    """Injects realistic failure patterns into telemetry data."""

    FAILURE_PATTERNS = {
        "battery_degradation": {
            "affected_signals": [
                "battery_soc_pct",
                "battery_voltage_v",
                "battery_cell_delta_v",
            ],
            "pattern": "gradual_degradation",
            "severity": 3,
        },
        "motor_bearing_wear": {
            "affected_signals": ["accel_x", "accel_y", "accel_z", "motor_temp_c"],
            "pattern": "vibration_increase",
            "severity": 3,
        },
        "inverter_overheating": {
            "affected_signals": ["inverter_temp_c", "battery_current_a"],
            "pattern": "thermal_ramp",
            "severity": 4,
        },
        "thermal_runaway_risk": {
            "affected_signals": ["battery_temp_c", "battery_cell_delta_v"],
            "pattern": "rapid_thermal",
            "severity": 4,
        },
        "charging_fault": {
            "affected_signals": ["battery_current_a", "battery_voltage_v"],
            "pattern": "efficiency_drop",
            "severity": 2,
        },
    }

    def __init__(self, config: dict):
        self.config = config
        self.active_failures: Dict[str, FailureEvent] = {}

    def schedule_failures(
        self, vehicle_id: str, start_ts: int, duration_days: int
    ) -> List[FailureEvent]:
        """Schedule random failures for a vehicle."""
        failures = []
        end_ts = start_ts + duration_days * 86400

        for failure_type, failure_config in self.config.get(
            "failure_injection", {}
        ).items():
            prob = failure_config.get("probability", 0.0003)

            # Poisson process for failure occurrence
            expected_failures = prob * duration_days * 86400
            num_failures = np.random.poisson(max(0.5, expected_failures))

            for _ in range(min(num_failures, 2)):  # Max 2 per type
                # Random failure time
                failure_ts = random.randint(start_ts + 86400, end_ts - 3600)
                lead_time_hours = failure_config.get("lead_time_hours", 48)
                lead_time_sec = int(lead_time_hours * 3600)
                onset_ts = failure_ts - lead_time_sec

                if onset_ts > start_ts:
                    failures.append(
                        FailureEvent(
                            failure_type=failure_type,
                            start_ts=onset_ts,
                            failure_ts=failure_ts,
                            severity=failure_config.get("severity", 2),
                            dtc_codes=failure_config.get("dtc_codes", []),
                            lead_time_hours=lead_time_hours,
                        )
                    )

        return sorted(failures, key=lambda x: x.start_ts)

    def apply_failure_effect(
        self, frame: dict, failure: FailureEvent, current_ts: int
    ) -> dict:
        """Apply failure effects to a telemetry frame."""
        if current_ts < failure.start_ts:
            return frame

        progress = (current_ts - failure.start_ts) / (
            failure.failure_ts - failure.start_ts
        )
        progress = min(1.0, progress)

        pattern = self.FAILURE_PATTERNS.get(failure.failure_type, {})

        if failure.failure_type == "battery_degradation":
            # Gradual capacity fade and voltage sag
            frame["battery_soc_pct"] *= 1 - 0.15 * progress
            frame["battery_voltage_v"] *= 1 - 0.08 * progress
            frame["battery_cell_delta_v"] += 0.2 * progress

        elif failure.failure_type == "motor_bearing_wear":
            # Increased vibration
            vibration_mult = 1 + 2 * progress
            frame["accel_x"] *= vibration_mult
            frame["accel_y"] *= vibration_mult
            frame["accel_z"] *= vibration_mult
            # Temperature spikes
            if random.random() < 0.3 * progress:
                frame["motor_temp_c"] += random.uniform(10, 30) * progress

        elif failure.failure_type == "inverter_overheating":
            # Temperature ramp
            frame["inverter_temp_c"] += 40 * progress
            # Power derating
            if progress > 0.7:
                frame["battery_current_a"] *= 1 - 0.3 * (progress - 0.7) / 0.3

        elif failure.failure_type == "thermal_runaway_risk":
            # Rapid temperature increase
            frame["battery_temp_c"] += 25 * progress * (1 + random.uniform(0, 0.5))
            # Cell voltage divergence
            frame["battery_cell_delta_v"] += 0.3 * progress

        elif failure.failure_type == "charging_fault":
            if frame.get("charging_state") in ["charging_ac", "charging_dc"]:
                # Reduced charging efficiency
                frame["battery_current_a"] *= 1 - 0.4 * progress
                # Current fluctuations
                frame["battery_current_a"] += random.uniform(-20, 20) * progress

        # Add DTC codes near failure
        if progress > 0.8 and random.random() < 0.3:
            frame["dtc_codes"] = failure.dtc_codes

        return frame


class DrivingPatternGenerator:
    """Generates realistic driving patterns and schedules."""

    def __init__(self, seed: int = None):
        if seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_daily_schedule(self, day_start_ts: int) -> List[DrivingSegment]:
        """Generate a day's worth of driving segments."""
        segments = []
        current_ts = day_start_ts
        day_end_ts = day_start_ts + 86400

        # Morning commute (7-9 AM)
        morning_start = day_start_ts + random.randint(7 * 3600, 8 * 3600)
        if random.random() < 0.8:  # 80% chance of commute
            segments.append(
                DrivingSegment(
                    segment_type="driving",
                    start_ts=morning_start,
                    duration_sec=random.randint(1200, 3600),  # 20-60 min
                    avg_speed=random.uniform(30, 60),
                    driving_style=random.choice(
                        ["normal", "normal", "eco", "aggressive"]
                    ),
                )
            )

        # Midday errands (11 AM - 2 PM)
        if random.random() < 0.4:
            midday_start = day_start_ts + random.randint(11 * 3600, 13 * 3600)
            segments.append(
                DrivingSegment(
                    segment_type="driving",
                    start_ts=midday_start,
                    duration_sec=random.randint(600, 1800),
                    avg_speed=random.uniform(20, 40),
                    driving_style="normal",
                )
            )

        # Evening commute (5-7 PM)
        evening_start = day_start_ts + random.randint(17 * 3600, 18 * 3600)
        if random.random() < 0.8:
            segments.append(
                DrivingSegment(
                    segment_type="driving",
                    start_ts=evening_start,
                    duration_sec=random.randint(1200, 3600),
                    avg_speed=random.uniform(25, 55),
                    driving_style=random.choice(["normal", "eco"]),
                )
            )

        # Overnight charging (10 PM - 6 AM)
        if random.random() < 0.7:
            charge_start = day_start_ts + random.randint(22 * 3600, 23 * 3600)
            segments.append(
                DrivingSegment(
                    segment_type="charging",
                    start_ts=charge_start,
                    duration_sec=random.randint(4 * 3600, 8 * 3600),
                    avg_speed=0,
                )
            )

        return sorted(segments, key=lambda x: x.start_ts)


class SyntheticDataGenerator:
    """Main generator class that orchestrates data generation."""

    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            self.config = json.load(f)

        self.vehicle_profiles = self._load_vehicle_profiles()
        self.failure_injector = FailureInjector(self.config)
        self.pattern_generator = DrivingPatternGenerator()

    def _load_vehicle_profiles(self) -> List[VehicleProfile]:
        """Load vehicle profiles from config."""
        profiles = []
        for vp in self.config.get("vehicle_profiles", []):
            profiles.append(
                VehicleProfile(
                    vehicle_id=vp["vehicle_id"],
                    vehicle_type=vp["type"],
                    battery_kwh=vp["battery_kwh"],
                    motor_kw=vp["motor_kw"],
                    ambient_temp=vp["ambient_temp"],
                )
            )
        return profiles

    def generate_frame(
        self, physics: EVPhysicsModel, ts: int, segment: Optional[DrivingSegment] = None
    ) -> dict:
        """Generate a single telemetry frame."""
        profile = physics.profile

        # Determine driving state
        if segment is None or segment.segment_type == "idle":
            speed = 0
            throttle = 0
            brake = 0
            regen = 0
            motor_rpm = 0
            charging_state = "idle"
        elif segment.segment_type == "charging":
            speed = 0
            throttle = 0
            brake = 0
            regen = 0
            motor_rpm = 0
            charging_state = random.choice(["charging_ac", "charging_dc"])
        else:  # driving
            # Simulate speed with some variation
            base_speed = segment.avg_speed
            speed_noise = random.gauss(0, base_speed * 0.15)
            speed = max(0, min(profile.max_speed_kph, base_speed + speed_noise))

            throttle = int(
                np.clip(
                    speed / profile.max_speed_kph * 80 + random.gauss(0, 10), 0, 100
                )
            )

            # Braking events
            if random.random() < 0.05:
                brake = random.randint(20, 80)
                regen = int(brake * random.uniform(0.5, 0.9))
            else:
                brake = 0
                regen = 0

            motor_rpm = speed / profile.max_speed_kph * profile.max_motor_rpm
            charging_state = "discharging"

        # Compute power flow
        if charging_state in ["charging_ac", "charging_dc"]:
            power_kw = -random.uniform(7, 50)  # Charging (negative = into battery)
            current = (
                power_kw
                * 1000
                / physics.compute_battery_voltage(physics.battery_soc, 0)
            )
        else:
            power_kw = physics.compute_power_consumption(speed, throttle / 100)
            regen_power = physics.compute_regen_power(speed, brake)
            net_power = power_kw - regen_power
            current = (
                net_power
                * 1000
                / physics.compute_battery_voltage(physics.battery_soc, 0)
            )

        # Update SOC
        soc_delta = (
            -current
            / (
                profile.battery_kwh
                * 1000
                / physics.compute_battery_voltage(physics.battery_soc, 0)
            )
            / 3600
        )
        physics.battery_soc = np.clip(physics.battery_soc + soc_delta, 5, 100)

        # Update thermal state
        physics.update_thermal_state(abs(power_kw), speed)

        # Update odometer
        physics.odometer += speed / 3600  # km traveled in 1 second

        # Generate accelerometer data (vibration)
        accel_base = 0.1 if speed > 5 else 0.02
        accel_x = random.gauss(0, accel_base * (1 + speed / 100))
        accel_y = random.gauss(0, accel_base * 0.5)
        accel_z = random.gauss(1.0, accel_base * 0.3)  # Gravity baseline

        # GPS (simplified - random walk)
        gps_lat = 28.6139 + random.gauss(0, 0.01)  # Delhi area
        gps_lon = 77.2090 + random.gauss(0, 0.01)

        # HVAC power
        hvac_power = (
            random.uniform(0.5, 3) if abs(physics.ambient_temp - 22) > 5 else 0.2
        )

        frame = {
            "vehicle_id": profile.vehicle_id,
            "timestamp": ts,
            "speed_kph": round(speed, 2),
            "motor_rpm": round(motor_rpm, 1),
            "motor_temp_c": round(physics.motor_temp, 2),
            "inverter_temp_c": round(physics.inverter_temp, 2),
            "battery_soc_pct": round(physics.battery_soc, 2),
            "battery_voltage_v": round(
                physics.compute_battery_voltage(physics.battery_soc, current), 2
            ),
            "battery_current_a": round(current, 2),
            "battery_temp_c": round(physics.battery_temp, 2),
            "battery_cell_delta_v": round(physics.battery_cell_delta, 4),
            "hvac_power_kw": round(hvac_power, 2),
            "throttle_pct": throttle,
            "brake_pct": brake,
            "regen_pct": regen,
            "accel_x": round(accel_x, 4),
            "accel_y": round(accel_y, 4),
            "accel_z": round(accel_z, 4),
            "gps_lat": round(gps_lat, 6),
            "gps_lon": round(gps_lon, 6),
            "odometer_km": round(physics.odometer, 2),
            "charging_state": charging_state,
            "ambient_temp_c": round(physics.ambient_temp, 1),
            "tire_pressure_fl_psi": round(physics.tire_pressures[0], 1),
            "tire_pressure_fr_psi": round(physics.tire_pressures[1], 1),
            "tire_pressure_rl_psi": round(physics.tire_pressures[2], 1),
            "tire_pressure_rr_psi": round(physics.tire_pressures[3], 1),
            "dtc_codes": [],
        }

        return frame

    def generate_vehicle_data(
        self, profile: VehicleProfile, output_dir: Path
    ) -> Tuple[int, List[FailureEvent]]:
        """Generate synthetic data for a single vehicle."""
        duration_days = self.config["data_generation"]["duration_days_per_vehicle"]
        sampling_sec = self.config["data_generation"]["sampling_sec"]

        # Initialize physics model
        physics = EVPhysicsModel(profile, profile.ambient_temp)

        # Time range
        base_ts = int(datetime(2024, 1, 1).timestamp())
        start_ts = base_ts + random.randint(
            0, 86400 * 30
        )  # Random start within a month
        end_ts = start_ts + duration_days * 86400

        # Schedule failures
        failures = self.failure_injector.schedule_failures(
            profile.vehicle_id, start_ts, duration_days
        )

        # Generate driving schedule
        all_segments = []
        for day in range(duration_days):
            day_start = start_ts + day * 86400
            segments = self.pattern_generator.generate_daily_schedule(day_start)
            all_segments.extend(segments)

        # Generate frames
        frames = []
        raw_frames_file = output_dir / "raw_frames" / f"{profile.vehicle_id}.jsonl"

        with open(raw_frames_file, "w") as f:
            for ts in range(start_ts, end_ts, sampling_sec):
                # Find current segment
                current_segment = None
                for seg in all_segments:
                    if seg.start_ts <= ts < seg.start_ts + seg.duration_sec:
                        current_segment = seg
                        break

                # Generate frame
                frame = self.generate_frame(physics, ts, current_segment)

                # Apply any active failure effects
                for failure in failures:
                    if failure.start_ts <= ts <= failure.failure_ts:
                        frame = self.failure_injector.apply_failure_effect(
                            frame, failure, ts
                        )

                # Add failure labels
                frame["_labels"] = {
                    "has_active_failure": any(
                        f.start_ts <= ts <= f.failure_ts for f in failures
                    ),
                    "failure_within_7d": any(
                        ts <= f.failure_ts <= ts + 7 * 86400 for f in failures
                    ),
                    "failure_within_14d": any(
                        ts <= f.failure_ts <= ts + 14 * 86400 for f in failures
                    ),
                    "nearest_failure_type": next(
                        (
                            f.failure_type
                            for f in failures
                            if f.start_ts <= ts <= f.failure_ts
                        ),
                        None,
                    ),
                    "nearest_failure_severity": next(
                        (
                            f.severity
                            for f in failures
                            if f.start_ts <= ts <= f.failure_ts
                        ),
                        0,
                    ),
                }

                frames.append(frame)
                f.write(json.dumps(frame) + "\n")

        print(f"  Generated {len(frames)} frames for {profile.vehicle_id}")
        print(f"  Scheduled {len(failures)} failure events")

        return len(frames), failures

    def generate_aggregates_fast(self, output_dir: Path):
        """
        OPTIMIZED: Generate aggregated window features using vectorized Pandas operations.
        Uses pd.read_json and rolling() for C-optimized speed (1000x faster than loops).
        """
        print("\nGenerating aggregated windows (FAST MODE)...")

        windows_config = self.config["feature_engineering"]["windows"]

        numeric_cols = [
            "speed_kph",
            "motor_rpm",
            "motor_temp_c",
            "inverter_temp_c",
            "battery_soc_pct",
            "battery_voltage_v",
            "battery_current_a",
            "battery_temp_c",
            "battery_cell_delta_v",
            "hvac_power_kw",
            "throttle_pct",
            "brake_pct",
            "regen_pct",
            "accel_x",
            "accel_y",
            "accel_z",
        ]

        all_windows_dfs = []

        for raw_file in (output_dir / "raw_frames").glob("*.jsonl"):
            vehicle_id = raw_file.stem
            print(f"  Processing {vehicle_id}...")

            # FAST: Use pd.read_json with lines=True (C-optimized)
            df = pd.read_json(raw_file, lines=True)
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
            df = df.set_index("timestamp").sort_index()

            # Extract labels before aggregation
            if "_labels" in df.columns:
                labels_df = pd.json_normalize(df["_labels"])
                labels_df.index = df.index
            else:
                labels_df = pd.DataFrame(index=df.index)
                labels_df["has_active_failure"] = False
                labels_df["failure_within_7d"] = False
                labels_df["failure_within_14d"] = False
                labels_df["nearest_failure_severity"] = 0
                labels_df["nearest_failure_type"] = None

            # Process each window type
            for window_name, window_config in windows_config.items():
                window_sec = window_config["length_sec"]
                stride_sec = window_config["stride_sec"]

                print(
                    f"    Window: {window_name} ({window_sec}s, stride {stride_sec}s)"
                )

                # Use rolling with proper window size
                window_str = f"{window_sec}s"

                # VECTORIZED: Compute all aggregations at once using rolling
                agg_dict = {}
                for col in numeric_cols:
                    if col in df.columns:
                        rolling = df[col].rolling(
                            window_str, min_periods=int(window_sec * 0.9)
                        )
                        agg_dict[f"{col}_mean"] = rolling.mean()
                        agg_dict[f"{col}_std"] = rolling.std()
                        agg_dict[f"{col}_min"] = rolling.min()
                        agg_dict[f"{col}_max"] = rolling.max()

                # Create aggregated DataFrame
                agg_df = pd.DataFrame(agg_dict)

                # Resample to stride interval (skip rows efficiently)
                stride_str = f"{stride_sec}s"
                agg_df = agg_df.resample(stride_str).first().dropna()

                if len(agg_df) == 0:
                    continue

                # Add metadata
                agg_df["vehicle_id"] = vehicle_id
                agg_df["window_type"] = window_name
                agg_df["window_id"] = [
                    f"{vehicle_id}_{window_name}_{i}" for i in range(len(agg_df))
                ]
                agg_df["start_ts"] = (
                    agg_df.index - pd.Timedelta(seconds=window_sec)
                ).astype(int) // 10**9
                agg_df["end_ts"] = agg_df.index.astype(int) // 10**9

                # Derived features
                agg_df["power_kw_mean"] = (
                    agg_df.get("battery_voltage_v_mean", 0)
                    * agg_df.get("battery_current_a_mean", 0)
                    / 1000
                )
                agg_df["accel_magnitude_mean"] = np.sqrt(
                    agg_df.get("accel_x_mean", 0) ** 2
                    + agg_df.get("accel_y_mean", 0) ** 2
                    + agg_df.get("accel_z_mean", 0) ** 2
                )

                # Add labels (sample at stride intervals)
                labels_resampled = (
                    labels_df.resample(stride_str).last().reindex(agg_df.index)
                )
                agg_df["anomaly"] = (
                    labels_resampled.get("has_active_failure", False)
                    .fillna(False)
                    .astype(int)
                )
                agg_df["failure_7d"] = (
                    labels_resampled.get("failure_within_7d", False)
                    .fillna(False)
                    .astype(int)
                )
                agg_df["failure_14d"] = (
                    labels_resampled.get("failure_within_14d", False)
                    .fillna(False)
                    .astype(int)
                )
                agg_df["severity"] = (
                    labels_resampled.get("nearest_failure_severity", 0)
                    .fillna(0)
                    .astype(int)
                )
                agg_df["failure_type"] = labels_resampled.get(
                    "nearest_failure_type", "none"
                ).fillna("none")

                # DTC count (simplified - just check if any DTCs in window)
                if "dtc_codes" in df.columns:
                    dtc_series = df["dtc_codes"].apply(
                        lambda x: 1 if x and len(x) > 0 else 0
                    )
                    dtc_rolling = dtc_series.rolling(window_str, min_periods=1).sum()
                    dtc_resampled = (
                        dtc_rolling.resample(stride_str).first().reindex(agg_df.index)
                    )
                    agg_df["dtc_count"] = dtc_resampled.fillna(0).astype(int)
                else:
                    agg_df["dtc_count"] = 0

                # Reset index for parquet storage
                agg_df = agg_df.reset_index(drop=True)
                all_windows_dfs.append(agg_df)

        # Combine all windows
        if all_windows_dfs:
            final_df = pd.concat(all_windows_dfs, ignore_index=True)

            # Reorder columns for clarity
            meta_cols = ["window_id", "vehicle_id", "window_type", "start_ts", "end_ts"]
            label_cols = [
                "anomaly",
                "failure_7d",
                "failure_14d",
                "severity",
                "failure_type",
                "dtc_count",
            ]
            feature_cols = [
                c for c in final_df.columns if c not in meta_cols + label_cols
            ]
            final_df = final_df[meta_cols + feature_cols + label_cols]

            # Save to parquet
            agg_file = output_dir / "aggregates" / "all_windows.parquet"
            final_df.to_parquet(agg_file, index=False)

            print(f"\n  Saved {len(final_df):,} windows to {agg_file}")
            print(f"  Anomaly windows: {final_df['anomaly'].sum():,}")
            print(f"  Failure (7d) windows: {final_df['failure_7d'].sum():,}")
            print(f"  Failure (14d) windows: {final_df['failure_14d'].sum():,}")

            # Per-window type summary
            print("\n  Windows per type:")
            for wt in final_df["window_type"].unique():
                count = len(final_df[final_df["window_type"] == wt])
                print(f"    {wt}: {count:,}")

    def generate_aggregates(self, output_dir: Path):
        """Generate aggregated window features - calls fast version."""
        self.generate_aggregates_fast(output_dir)

    def run(self, output_dir: str):
        """Run the complete data generation pipeline."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "raw_frames").mkdir(exist_ok=True)
        (output_path / "aggregates").mkdir(exist_ok=True)

        print("=" * 60)
        print("EV Telemetry Synthetic Data Generator")
        print("=" * 60)
        print(f"Vehicles: {len(self.vehicle_profiles)}")
        print(
            f"Duration per vehicle: {self.config['data_generation']['duration_days_per_vehicle']} days"
        )
        print(f"Sampling rate: {self.config['data_generation']['sampling_sec']}s")
        print("=" * 60)

        all_failures = []
        total_frames = 0

        for profile in self.vehicle_profiles:
            print(
                f"\nGenerating data for {profile.vehicle_id} ({profile.vehicle_type})..."
            )
            frames, failures = self.generate_vehicle_data(profile, output_path)
            total_frames += frames
            all_failures.extend(failures)

        # Generate aggregates
        self.generate_aggregates(output_path)

        # Save failure manifest
        failure_manifest = {
            "failures": [
                {
                    "failure_type": f.failure_type,
                    "start_ts": f.start_ts,
                    "failure_ts": f.failure_ts,
                    "severity": f.severity,
                    "lead_time_hours": f.lead_time_hours,
                }
                for f in all_failures
            ]
        }
        with open(output_path / "failure_manifest.json", "w") as f:
            json.dump(failure_manifest, f, indent=2)

        print("\n" + "=" * 60)
        print("Generation Complete!")
        print(f"Total frames: {total_frames:,}")
        print(f"Total failures injected: {len(all_failures)}")
        print(f"Output directory: {output_path}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic EV telemetry data")
    parser.add_argument(
        "--config", type=str, default="config.json", help="Path to config.json"
    )
    parser.add_argument(
        "--output", type=str, default="synthetic_data", help="Output directory"
    )
    args = parser.parse_args()

    generator = SyntheticDataGenerator(args.config)
    generator.run(args.output)


if __name__ == "__main__":
    main()
